"""Persistence identity, rollout determinism and the mechanistic models."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from trafficloop.data.batch import WARMUP, EventBatch
from trafficloop.data.loaders.synthetic import IDM_NOMINAL
from trafficloop.models.base import ThreeChannelAdapter
from trafficloop.models.mechanistic import IDM, Gipps, Newell, gipps_speed, idm_acc
from trafficloop.models.persistence import PersistenceBaseline, ZeroAcceleration
from trafficloop.protocols.p0 import RecordedAcceleration, evaluate_p0, pooled_rmse
from trafficloop.sim.rollout import rollout, rollout_errors

from conftest import rms


# ------------------------------------------------------------------ persistence (E5)
@pytest.mark.parametrize("fixture", ["clean_events", "noisy_events"])
def test_persistence_identity(fixture, request):
    """RMSE_a of persistence at 0.1 s equals the RMS of the observed jerk times 0.1 s."""
    events = request.getfixturevalue(fixture)
    events = events[0] if isinstance(events, tuple) else events
    # unclipped: the accelerations of the noisy records leave the range of the benchmark
    scores = pooled_rmse(evaluate_p0(PersistenceBaseline(), events, clip=None))
    dt = 0.1
    jerk = np.concatenate([np.diff(ev.a)[WARMUP - 1 : len(ev) - 2] / dt for ev in events])
    assert scores["rmse_a"] == pytest.approx(rms(jerk * dt), rel=1e-12)


def test_p0_clips_the_prediction_like_a_rollout(clean_events):
    """A formula that asks for -300 m/s^2 at a small gap is scored as the limit of -8."""
    wild = ThreeChannelAdapter(lambda h: torch.full(h.shape[:-2], -300.0, dtype=h.dtype), name="wild")
    limit = ThreeChannelAdapter(lambda h: torch.full(h.shape[:-2], -8.0, dtype=h.dtype), name="limit")
    clipped = pooled_rmse(evaluate_p0(wild, clean_events))
    assert clipped == pytest.approx(pooled_rmse(evaluate_p0(limit, clean_events)))
    assert pooled_rmse(evaluate_p0(wild, clean_events, clip=None))["rmse_a"] > 290.0
    inside = pooled_rmse(evaluate_p0(PersistenceBaseline(), clean_events))
    assert inside == pooled_rmse(evaluate_p0(PersistenceBaseline(), clean_events, clip=None))


def test_p0_counts_every_prediction_instant(clean_events):
    table = evaluate_p0(PersistenceBaseline(), clean_events)
    assert list(table["event_id"]) == [ev.event_id for ev in clean_events]
    assert list(table["n"]) == [len(ev) - 1 - WARMUP for ev in clean_events]


def test_recorded_acceleration_is_exact_on_consistent_data(clean_events, noisy_events):
    clean = pooled_rmse(evaluate_p0(RecordedAcceleration(), clean_events))
    assert clean["rmse_a"] == 0.0 and clean["rmse_v"] < 1e-12 and clean["rmse_s"] < 1e-12
    noisy = pooled_rmse(evaluate_p0(RecordedAcceleration(), noisy_events[0]))
    assert noisy["rmse_v"] > 0.05 and noisy["rmse_s"] > 0.05  # floor set by the measurement noise


def test_p0_scores_are_proportional_on_consistent_data(clean_events):
    for model in (PersistenceBaseline(), ZeroAcceleration(), IDM((30.0, 1.2, 2.5, 1.2, 2.0))):
        scores = pooled_rmse(evaluate_p0(model, clean_events))
        assert scores["rmse_v"] == pytest.approx(0.1 * scores["rmse_a"], rel=1e-6)
        assert scores["rmse_s"] == pytest.approx(0.01 * scores["rmse_a"], rel=1e-6)


# --------------------------------------------------------------------------- rollout
def test_true_model_reproduces_the_events(clean_events):
    batch = EventBatch.from_events(clean_events)
    result = rollout(IDM(IDM_NOMINAL), batch)
    err = rollout_errors(result, batch)
    assert float(err.sse_s.max()) < 1e-18 and float(err.sse_v.max()) < 1e-18
    assert not bool(result.collided.any()) and result.n_invalid == 0
    assert err.n.tolist() == [len(ev) - 1 - WARMUP for ev in clean_events]
    one_step = pooled_rmse(evaluate_p0(IDM(IDM_NOMINAL), clean_events))
    assert one_step["rmse_a"] < 1e-12


def test_rollout_is_deterministic_and_batch_independent(clean_events):
    model = Gipps()
    batch = EventBatch.from_events(clean_events)
    first, second = rollout(model, batch), rollout(model, batch)
    assert torch.equal(first.s, second.s) and torch.equal(first.v, second.v)
    alone = rollout(model, EventBatch.from_events(clean_events[2:3]))
    n = len(clean_events[2])
    torch.testing.assert_close(alone.s[0, :n], first.s[2, :n], rtol=0, atol=1e-12)


def test_rollout_keeps_the_observed_warm_up(clean_events):
    batch = EventBatch.from_events(clean_events)
    result = rollout(ZeroAcceleration(), batch)
    assert torch.equal(result.s[:, : WARMUP + 1], batch.s[:, : WARMUP + 1])
    assert not bool(result.sim_mask[:, : WARMUP + 1].any())
    assert torch.allclose(result.v[:, WARMUP:], batch.v[:, WARMUP : WARMUP + 1].expand_as(result.v[:, WARMUP:]))


def test_acceleration_clip_and_speed_floor(clean_events):
    batch = EventBatch.from_events(clean_events[:4])
    hard = ThreeChannelAdapter(lambda h: torch.full(h.shape[:-2], -50.0, dtype=h.dtype), name="brake")
    result = rollout(hard, batch)
    assert float(result.a.min()) >= -8.0 - 1e-12 and float(result.v.min()) >= 0.0
    assert float(result.v[:, -1].max()) == 0.0
    wild = ThreeChannelAdapter(lambda h: torch.full(h.shape[:-2], 50.0, dtype=h.dtype), name="floor_it")
    result = rollout(wild, batch)
    assert float(result.a.max()) <= 4.0 + 1e-12
    assert bool(result.collided.all())
    first = result.first_collision
    assert bool((first > WARMUP).all())
    for e in range(4):
        k = int(first[e])
        assert float(result.s[e, k]) <= 0.0 < float(result.s[e, k - 1])


def test_adaptive_models_see_the_observed_warm_up_once(clean_events):
    """The hook gets the 30 observed rows that end at the initial condition, per event."""

    class Remembering(PersistenceBaseline):
        name = "remembering"
        adaptive = True

        def begin(self, hist):
            self.seen = hist.clone()
            self.offset = hist[:, -1, 2] * 0.0 + 0.5  # one value per event

        def forward(self, hist):
            offset = self.offset.reshape(-1, *[1] * (hist.dim() - 3))
            return hist[..., -1, 3] * 0.0 + offset

    batch = EventBatch.from_events(clean_events[:3])
    model = Remembering()
    result = rollout(model, batch)
    assert model.seen.shape == (3, WARMUP, 4)
    assert torch.equal(model.seen, batch.state()[:, 1 : WARMUP + 1])
    assert torch.allclose(result.a[:, WARMUP], torch.full((3,), 0.5, dtype=result.a.dtype))
    table = evaluate_p0(model, clean_events[:3])
    expected = [float(np.sum((0.5 - ev.a[WARMUP:-1]) ** 2)) for ev in clean_events[:3]]
    np.testing.assert_allclose(table["sse_a"], expected, rtol=1e-9)


def test_non_finite_outputs_are_counted_and_replaced(clean_events):
    batch = EventBatch.from_events(clean_events[:2])
    broken = ThreeChannelAdapter(lambda h: torch.full(h.shape[:-2], float("nan"), dtype=h.dtype), name="nan")
    result = rollout(broken, batch)
    assert result.n_invalid > 0 and bool(torch.isfinite(result.s).all())


def test_candidate_dimension_matches_single_rollouts(clean_events):
    batch = EventBatch.from_events(clean_events[:5])
    thetas = torch.tensor([IDM_NOMINAL, (25.0, 1.0, 3.0, 1.5, 2.0), (40.0, 2.0, 1.0, 0.8, 1.0)], dtype=torch.float64)
    together = rollout(IDM(), batch, params=thetas.unsqueeze(1))
    assert together.s.shape[:2] == (3, 5)
    for p in range(3):
        single = rollout(IDM(thetas[p]), batch)
        torch.testing.assert_close(together.s[p], single.s, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- mechanistic models
def test_sign_convention_of_dv():
    """Closing in (dv > 0) must make IDM and Gipps brake harder than opening (dv < 0)."""
    s, v = torch.tensor(30.0, dtype=torch.float64), torch.tensor(20.0, dtype=torch.float64)
    theta = [torch.tensor(p, dtype=torch.float64) for p in IDM_NOMINAL]
    closing = idm_acc(s, torch.tensor(3.0, dtype=torch.float64), v, *theta)
    opening = idm_acc(s, torch.tensor(-3.0, dtype=torch.float64), v, *theta)
    assert closing < opening
    model = Gipps()
    hist = torch.zeros(2, model.history, 4, dtype=torch.float64)
    hist[..., 0], hist[..., 2] = 30.0, 20.0
    hist[0, :, 1], hist[1, :, 1] = 3.0, -3.0
    acc = model(hist)
    assert acc[0] < acc[1]
    # Newell reads the gap and the follower speed only: the leader enters through the gap history
    model = Newell()
    hist = torch.rand(5, model.history, 4, dtype=torch.float64) * 10.0 + 10.0
    other = hist.clone()
    other[..., 1] = -other[..., 1]
    assert torch.equal(model(hist), model(other))


def test_idm_equilibrium():
    v0, T, s0, a, b = IDM_NOMINAL
    v = torch.linspace(1.0, 30.0, 30, dtype=torch.float64)
    s_e = (s0 + v * T) / torch.sqrt(1.0 - (v / v0) ** 4)
    acc = IDM().acc_state(s_e, torch.zeros_like(v), v)
    assert float(acc.abs().max()) < 1e-12


def test_gipps_steady_state_and_safety():
    a, b, v_des, tau, s0, b_hat = (torch.tensor(p, dtype=torch.float64) for p in Gipps.default)
    far = gipps_speed(torch.tensor(500.0, dtype=torch.float64), v_des, v_des, a, b, v_des, tau, s0, b_hat)
    assert float(far) == pytest.approx(float(v_des))  # free flow: desired speed is a fixed point
    standing = gipps_speed(s0, torch.tensor(0.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64), a, b, v_des, tau, s0, b_hat)
    assert float(standing) == pytest.approx(0.0, abs=1e-12)  # stopped at the jam gap behind a stopped leader


def newell_event(tau=1.2, d=4.0, n=400, brake=True):
    """Exact Newell trajectory behind a leader that brakes from 15 to 9 m/s and recovers.

    The speed of the leader changes for the first time at sample 101.
    """
    from trafficloop.data.schema import Event

    m = int(round(tau / 0.1))
    acc = np.zeros(n + m)
    if brake:
        acc[m + 100 : m + 130] = -2.0
        acc[m + 220 : m + 250] = 1.5
    v_all = 15.0 + np.concatenate([[0.0], np.cumsum(acc[:-1]) * 0.1])
    x_all = 100.0 + np.cumsum(v_all) * 0.1
    x_lead, v_lead = x_all[m:], v_all[m:]
    x, v = x_all[:n] - d, v_all[:n]  # the leader, shifted by tau and d
    a = np.append(np.diff(v) / 0.1, 0.0)
    event = Event("e", "toy", "a", "toy/a/1", "toy/a/0", np.arange(n) * 0.1, x_lead - x, v - v_lead, v, a,
                  v_lead, x_lead, x, {"data_version": "raw"})
    event.validate()
    return event, m


def test_newell_follows_a_shifted_leader():
    """The follower repeats the leader tau later and d behind, braking included."""
    tau, d = 1.2, 4.0
    event, m = newell_event(tau, d)
    batch = EventBatch.from_events([event])
    result = rollout(Newell((tau, d, 33.0)), batch)
    err = rollout_errors(result, batch)
    assert float(err.sse_s.max()) < 1e-16 and float(err.sse_v.max()) < 1e-16
    lead_acc = np.diff(event.v_lead) / 0.1
    follower_acc = result.a[0].numpy()
    np.testing.assert_allclose(follower_acc[WARMUP + m : -1], lead_acc[WARMUP : -m], atol=1e-9)
    assert pooled_rmse(evaluate_p0(Newell((tau, d, 33.0)), [event]))["rmse_a"] < 1e-9

    shifted = copy.deepcopy(event)
    shifted.x_follower = event.x_lead - 60.0  # too far back: closes in at the free speed, then settles
    shifted.s = shifted.x_lead - shifted.x_follower
    result = rollout(Newell((tau, d, 20.0)), EventBatch.from_events([shifted]))
    assert float(result.s[0, -1]) == pytest.approx(d + tau * float(event.v_lead[-1]), abs=1e-6)
    assert not bool(result.collided.any())


@pytest.mark.parametrize(
    "model, delay, gap_time",
    [
        (IDM(), 1, 1.2),
        (Gipps((1.7, 3.0, 33.3, 0.5, 2.0, 3.5)), 5, 1.2),
        (Gipps((1.7, 3.0, 33.3, 1.5, 2.0, 3.5)), 15, 1.2),
        (Newell((1.2, 4.0, 33.3)), 12, 1.2),
        (Newell((2.0, 4.0, 33.3)), 20, 2.0),
    ],
)
def test_response_is_delayed_by_the_reaction_time(model, delay, gap_time):
    """The first acceleration that differs is the one applied `delay` samples after the leader changed.

    The events start close to the steady state of the model, so that no output is saturated.
    """
    braking, _ = newell_event(tau=gap_time, brake=True)
    cruising, _ = newell_event(tau=gap_time, brake=False)
    assert np.flatnonzero(braking.v_lead != cruising.v_lead)[0] == 101
    a = rollout(model, EventBatch.from_events([braking])).a[0].numpy()
    b = rollout(model, EventBatch.from_events([cruising])).a[0].numpy()
    assert np.flatnonzero(np.abs(a - b) > 1e-12)[0] == 100 + delay


@pytest.mark.parametrize("cls", [IDM, Gipps, Newell])
def test_defaults_lie_inside_the_calibration_bounds(cls):
    bounds = cls.bounds_tensor()
    theta = torch.as_tensor(cls.default, dtype=torch.float64)
    assert bool(((theta >= bounds[:, 0]) & (theta <= bounds[:, 1])).all())
    assert cls().param_dict() == pytest.approx(dict(zip(cls.param_names, cls.default)))
