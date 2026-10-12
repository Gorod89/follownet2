"""P4: synthetic suite, cut-in rollout, episode reconstruction and confirmation, metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from follownet2.data import episodes as episodes_module
from follownet2.data.batch import WARMUP, EventBatch
from follownet2.data.episodes import confirm_triggers, leader_jumps, reconstruct_episode, reconstruct_episodes
from follownet2.data.loaders.synthetic import SyntheticConfig, make_events
from follownet2.data.reconstruction import ReconstructionConfig
from follownet2.data.schema import DT, Event
from follownet2.models.mechanistic import IDM
from follownet2.models.persistence import PersistenceBaseline, ZeroAcceleration
from follownet2.protocols.p4 import evaluate_p4, quantile, rollout_cut_in, synthetic_suite, TTC_MAX
from follownet2.protocols.scenarios import positions

from conftest import rms

IDS = [
    *[f"pulse_a{a}_v{v}" for a in (-4, -6, -8) for v in (15, 25, 35)],
    "stop_v30",
    "cutin_tau0.5_v15", "cutin_tau0.5_v25", "cutin_tau1.0_v15", "cutin_tau1.0_v25",
]
CFG = ReconstructionConfig(sigma_x=0.3, sigma_v=None, cutoff_hz=0.5)
SUMMARY_KEYS = {
    "collision_rate_synthetic", "collision_rate_empirical", "min_ttc_q05", "min_ttc_q25", "min_ttc_q50",
    "min_ttc_q05_empirical", "min_ttc_q25_empirical", "min_ttc_q50_empirical", "exceedance_4", "exceedance_6",
    "exceedance_4_empirical", "exceedance_6_empirical", "amplification_mean", "n_scenarios", "n_episodes",
    "n_invalid", "min_ttc_q50_recorded", "exceedance_4_recorded",
}


def fabricated(kind: str, n: int = 150, speed: float = 20.0, gap: float = 35.0, at: int = 60) -> Event:
    """Episode on the benchmark integrator: calm following, a leader braking at -6 m/s^2 for
    2.5 s (the follower copies it 1.5 s later), or a cut-in 8 m (0.4 s) ahead of the follower."""
    acc_lead, acc = np.zeros(n), np.zeros(n)
    if kind == "brake":
        acc_lead[at : at + 25] = -6.0
        acc[at + 15 : at + 40] = -6.0
    integrate = lambda a: np.clip(speed + np.concatenate([[0.0], np.cumsum(a[:-1])]) * DT, 0.0, None)  # noqa: E731
    v_lead, v = integrate(acc_lead), integrate(acc)
    x_lead, x = positions(v_lead, gap), positions(v, 0.0)
    if kind == "cut_in":
        x_lead[at:] = x[at] + 8.0 + (x_lead[at:] - x_lead[at])
    a = np.append(np.diff(v) / DT, 0.0)
    a[-1] = a[-2]
    return Event(
        event_id=f"toy/a/{kind}|L|nm{at}", dataset="toy_nearmiss", site="a", follower_id=f"toy/a/{kind}",
        leader_id="toy/a/L", t=np.arange(n) * DT, s=x_lead - x, dv=v - v_lead, v=v, a=a, v_lead=v_lead,
        x_lead=x_lead, x_follower=x,
        meta={"data_version": "raw", "position_source": "track",
              "episode": {"triggers": {"calm": [], "brake": ["brake"], "cut_in": ["cut_in"]}[kind]}},
    )


@pytest.fixture(scope="module")
def suite():
    return synthetic_suite()


@pytest.fixture(scope="module")
def episodes():
    return [fabricated("brake"), fabricated("cut_in")]


@pytest.fixture(scope="module")
def idm_p4(suite, episodes):
    return evaluate_p4(IDM(), suite, episodes, device="cpu")


def test_suite_has_the_fourteen_scenarios(suite):
    assert [ev.event_id for ev in suite] == IDS
    for ev in suite:
        k_c, info = ev.meta["manoeuvre_start"], ev.meta["scenario"]
        assert k_c == 600 and ev.v_lead[k_c] == info["v0"] and ev.v_lead[: k_c + 1].std() == 0.0
        acc = np.diff(ev.v_lead[k_c:]) / DT
        if info["kind"] == "pulse":
            assert acc.min() == pytest.approx(info["a"]) and acc.max() == pytest.approx(1.0)
            assert ev.v_lead.min() == pytest.approx(info["v0"] - 10.0) and ev.v_lead[-1] == pytest.approx(info["v0"])
            assert np.all(ev.v_lead[-200:] == ev.v_lead[-1]) and info["tau"] is None
        elif info["kind"] == "stop":
            assert acc.min() == pytest.approx(-4.0) and ev.v_lead[-201:].max() < 1e-9
        else:
            assert info["kind"] == "cut_in" and info["tau"] in (0.5, 1.0) and np.all(ev.v_lead == info["v0"])


def test_idm_does_not_collide_in_the_mildest_pulse(idm_p4):
    _, tables = idm_p4
    row = tables["scenarios"].set_index("event_id").loc["pulse_a-4_v15"]
    assert not row["collided"] and not row["collided_hold"]
    assert row["min_ttc"] > 1.0 and row["min_acc"] < 0.0 and row["amplification"] > 0.0


def test_constant_speed_follower_collides_in_the_full_stop(suite):
    stop = [ev for ev in suite if ev.event_id == "stop_v30"]
    summary, tables = evaluate_p4(ZeroAcceleration(), stop, [])
    row = tables["scenarios"].iloc[0]
    assert row["collided"] and not row["collided_hold"] and 0.0 < row["first_collision"] < 10.0
    assert row["min_ttc"] == 0.0 and row["min_gap"] <= 0.0
    assert summary["collision_rate_synthetic"] == 1.0 and summary["n_episodes"] == 0


def test_cut_in_gap_is_tau_times_speed_and_the_batch_is_deterministic(suite):
    cut = [ev for ev in suite if ev.meta["scenario"]["kind"] == "cut_in"]
    tau = torch.tensor([ev.meta["scenario"]["tau"] for ev in cut])
    batch = EventBatch.from_events(cut, dtype=torch.float32)
    first = rollout_cut_in(IDM(), batch, tau, start=600)
    again = rollout_cut_in(IDM(), batch, tau, start=600)
    for name in ("s", "v", "a", "collided", "first_collision"):
        assert torch.equal(getattr(first, name), getattr(again, name))
    gap, target = first.s[:, 601], tau * first.v[:, 600]
    np.testing.assert_allclose(gap.numpy(), target.numpy(), rtol=0.01)
    assert torch.all(first.s[:, 600] - gap > 5.0)  # the follower had settled far behind the old leader
    alone = rollout_cut_in(IDM(), EventBatch.from_events(cut[3:], dtype=torch.float32), float(tau[3]), start=600)
    assert torch.equal(alone.s[0], first.s[3]) and torch.equal(alone.a[0], first.a[3])


def test_reconstruct_episode_keeps_a_leader_jump_and_reduces_the_noise():
    clean = make_events(SyntheticConfig(n_drivers=1, events_per_driver=1, theta_cv=0.0, duration=(30.0, 30.0), seed=4))[0]
    n, at, shift = len(clean), len(clean) // 2, 15.0
    rng = np.random.default_rng(0)
    x_lead = clean.x_lead + 0.3 * rng.standard_normal(n)
    x = clean.x_follower + 0.3 * rng.standard_normal(n)
    # speeds and accelerations of the noisy tracks; every leader has its own continuous speed
    v_lead, v = np.clip(np.gradient(x_lead, DT), 0.0, None), np.clip(np.gradient(x, DT), 0.0, None)
    a = np.gradient(v, DT)
    x_lead[:at] += shift  # the first leader drives 15 m further ahead: a cut-in at `at`
    raw = Event(
        event_id="toy/a/1|L|nm0", dataset="toy_nearmiss", site="a", follower_id="toy/a/1", leader_id="toy/a/L",
        t=clean.t, s=x_lead - x, dv=v - v_lead, v=v, a=a, v_lead=v_lead, x_lead=x_lead, x_follower=x,
        meta={"data_version": "raw", "position_source": "track", "episode": {"triggers": ["cut_in"]}},
    )
    raw.validate()
    assert np.flatnonzero(leader_jumps(raw)).tolist() == [at]
    rec = reconstruct_episode(raw, CFG)
    rec.validate()
    assert rec.data_version == "reconstructed" and rec.meta["a_source"] == "reconstruction"
    info = rec.meta["reconstruction"]
    assert info["jump_samples"] == [at] and info["raw_pieces"] == [] and len(info["status"]) == 3
    true_drop = clean.s[at] - clean.s[at - 1] - shift
    assert abs((rec.s[at] - rec.s[at - 1]) - true_drop) < 1.0
    assert np.flatnonzero(leader_jumps(rec)).tolist() == [at]
    assert rms(rec.a) < rms(raw.a) and rms(rec.a - clean.a) < 0.1 * rms(raw.a - clean.a)
    assert rec.s.min() >= CFG.s_min - 1e-9


def test_triggers_are_confirmed_on_fabricated_episodes():
    assert episodes_module.WARMUP == WARMUP  # mirrored there to keep torch out of the worker processes
    calm, brake, cut = fabricated("calm"), fabricated("brake"), fabricated("cut_in")
    early = fabricated("cut_in", at=20)  # the cut-in happens inside the observed warm-up
    assert confirm_triggers(calm) == []
    assert "brake" in confirm_triggers(brake) and "cut_in" not in confirm_triggers(brake)
    assert "cut_in" in confirm_triggers(cut) and "brake" not in confirm_triggers(cut)
    assert confirm_triggers(early) == ["cut_in"] and confirm_triggers(early, after=WARMUP) == []
    done, report = reconstruct_episodes([calm, brake, cut, early], CFG, workers=1)
    assert [ev.event_id for ev in done] == [brake.event_id, cut.event_id]
    counts = ("n_in", "n_failed", "n_reconstructed", "n_confirmed", "n_only_warmup", "n_unconfirmed")
    assert [report[k] for k in counts] == [4, 0, 4, 2, 1, 1]
    assert report["confirmed_triggers"]["brake"] == 1 and report["confirmed_triggers"]["cut_in"] == 1
    assert report["confirmation_rate"]["brake"] == 1.0 and report["confirmation_rate"]["cut_in"] == 1.0
    assert report["confirmation_rate_after_warmup"]["cut_in"] == 0.5
    for ev in done:
        ev.validate()
        assert ev.meta["episode"]["confirmed_triggers"] == confirm_triggers(ev)
        assert ev.meta["episode"]["triggers_after_warmup"] == confirm_triggers(ev, after=WARMUP) != []


@pytest.mark.parametrize("model", ["persistence", "idm"])
def test_evaluate_p4_on_the_suite_and_two_episodes(model, suite, episodes, idm_p4):
    summary, tables = idm_p4 if model == "idm" else evaluate_p4(PersistenceBaseline(), suite, episodes)
    assert SUMMARY_KEYS <= set(summary) and summary["n_scenarios"] == 14 and summary["n_episodes"] == 2
    json.dumps(summary, allow_nan=False)  # strict JSON: an infinite quantile is null
    scen, epi = tables["scenarios"], tables["episodes"]
    # the recorded cut-in never closes in: its min TTC is the cap TTC_MAX, the median stays finite
    assert epi["min_ttc_recorded"].iloc[1] == TTC_MAX and epi["min_ttc_recorded"].iloc[0] < TTC_MAX
    assert summary["min_ttc_q50_recorded"] is not None and summary["min_ttc_q50_recorded"] <= TTC_MAX
    assert scen["event_id"].tolist() == IDS and epi["event_id"].tolist() == [ev.event_id for ev in episodes]
    assert np.isfinite(scen.loc[scen["kind"] != "cut_in", "amplification"]).all()
    assert scen.loc[scen["kind"] == "cut_in", "amplification"].isna().all()
    assert epi["trigger_brake"].tolist() == [True, False] and epi["trigger_cut_in"].tolist() == [False, True]
    brake = epi.iloc[0]
    assert brake["min_acc_recorded"] == pytest.approx(-6.0) and brake["exceedance_4_recorded"] > 0.0
    assert np.isfinite(brake["min_ttc_recorded"]) and summary["exceedance_4_recorded"] > 0.0
    if model == "persistence":
        # the placeholder warm-up has no acceleration: persistence keeps the speed, like the
        # constant-speed follower, and hits the leader in every pulse and in the stop
        assert summary["collision_rate_synthetic"] == pytest.approx(10 / 14)
        assert (scen.loc[scen["collided"], "min_ttc"] == 0.0).all() and summary["min_ttc_q05"] == 0.0
    else:
        assert summary["collision_rate_synthetic"] == 0.0 and summary["n_invalid"] == 0
        assert summary["amplification_mean"] > 0.0


def test_quantile_keeps_infinite_ttc():
    assert quantile([1.0, 2.0, 3.0], 0.5) == 2.0 and quantile([1.0, np.inf], 0.5) == np.inf
    assert quantile([1.0, 2.0, np.inf], 0.25) == 1.5 and quantile([], 0.5) is None
