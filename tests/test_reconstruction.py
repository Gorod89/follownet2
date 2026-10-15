"""Physics-informed reconstruction: constraints, consistency, noise reduction."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from trafficloop.data.batch import EventBatch
from trafficloop.data.extraction import positions_from_gap
from trafficloop.data.loaders.synthetic import IDM_NOMINAL
from trafficloop.data.reconstruction import (
    ReconstructionConfig,
    consistency_residual,
    reconstruct_event,
    reconstruct_events,
    reconstruct_trajectory,
    summarise,
)
from trafficloop.models.mechanistic import IDM
from trafficloop.protocols.p0 import RecordedAcceleration, evaluate_p0, pooled_rmse
from trafficloop.sim.rollout import rollout, rollout_errors

from conftest import rms

CFG = ReconstructionConfig(sigma_x=0.3, sigma_v=0.2, cutoff_hz=0.5)


@pytest.fixture(scope="module")
def reconstructed(noisy_events):
    events, _ = noisy_events
    done, failed = reconstruct_events(events, CFG)
    assert failed == []
    return done


def test_constraints_hold(reconstructed):
    for ev in reconstructed:
        ev.validate()
        assert ev.data_version == "reconstructed"
        assert ev.s.min() >= CFG.s_min - 1e-6
        assert ev.v.min() >= 0.0 and ev.v_lead.min() >= 0.0
        assert ev.a.min() >= CFG.a_min - 1e-9 and ev.a.max() <= CFG.a_max + 1e-9
        jerk = np.diff(ev.a[:-1]) / CFG.dt
        assert np.max(np.abs(jerk)) <= CFG.jerk_max + 1e-3
        assert ev.meta["reconstruction"]["method"] and ev.meta["a_source"] == "reconstruction"


def test_result_is_a_trajectory_of_the_benchmark_integrator(reconstructed):
    for ev in reconstructed:
        residual = consistency_residual(ev)
        assert residual["rms_speed_residual"] < 1e-12 and residual["rms_position_residual"] < 1e-9
        lead_step = ev.x_lead[1:] - ev.x_lead[:-1] - ev.v_lead[1:] * CFG.dt
        assert np.max(np.abs(lead_step)) < 1e-9
    oracle = pooled_rmse(evaluate_p0(RecordedAcceleration(), reconstructed))
    assert oracle["rmse_v"] < 1e-12 and oracle["rmse_s"] < 1e-9


def test_noise_is_reduced(noisy_events, reconstructed):
    events, truths = noisy_events
    for raw, truth, rec in zip(events, truths, reconstructed):
        assert rms(rec.s - truth["s"]) < 0.6 * rms(raw.s - truth["s"])
        assert rms(rec.v - truth["v"]) < 0.6 * rms(raw.v - truth["v"])
        assert rms(rec.a - truth["a"]) < 0.25 * rms(raw.a - truth["a"])


def test_clean_data_are_left_almost_unchanged(clean_events):
    for ev in clean_events[:4]:
        rec = reconstruct_event(ev, ReconstructionConfig(sigma_x=0.3, sigma_v=0.2, cutoff_hz=1.0))
        assert rms(rec.s - ev.s) < 0.05 and rms(rec.v - ev.v) < 0.05 and rms(rec.a - ev.a) < 0.15
    batch = EventBatch.from_events([reconstruct_event(ev, CFG) for ev in clean_events[:4]])
    err = rollout_errors(rollout(IDM(IDM_NOMINAL), batch), batch)
    assert float((err.sse_s.sum() / err.n.sum()).sqrt()) < 0.3


def test_gap_stays_positive_when_the_record_overlaps(clean_events):
    ev = copy.deepcopy(clean_events[0])
    ev.x_follower = ev.x_follower + (ev.s.min() - 0.05)  # smallest recorded gap: 5 cm
    ev.s = ev.x_lead - ev.x_follower
    rec = reconstruct_event(ev, CFG)
    assert rec.s.min() >= CFG.s_min - 1e-6
    assert rec.meta["reconstruction"]["gap_bound_active"] is True
    assert rec.meta["reconstruction"]["min_gap_raw"] == pytest.approx(0.05, abs=1e-9)


def test_bounds_bind_on_an_impossible_record():
    n = 200
    v = np.full(n, 20.0)
    v[100:] = 5.0  # the record drops by 15 m/s within one step
    x = np.cumsum(v) * 0.1
    cfg = ReconstructionConfig(sigma_x=0.1, sigma_v=0.1, cutoff_hz=2.0, a_min=-4.0, jerk_max=5.0)
    out = reconstruct_trajectory(x, v, cfg)
    assert out["ok"]
    assert out["a"].min() >= -4.0 - 1e-9 and np.max(np.abs(np.diff(out["a"][:-1]))) <= 0.5 + 1e-3
    assert out["a"].min() < -3.9  # the bound is active
    assert out["v"][0] == pytest.approx(20.0, abs=0.5) and out["v"][-1] == pytest.approx(5.0, abs=0.5)


def test_gap_based_sources_fit_the_measured_gap(noisy_events):
    events, truths = noisy_events
    raw, truth = copy.deepcopy(events[0]), truths[0]
    raw.s = truth["s"] + 0.2 * np.random.default_rng(0).standard_normal(len(raw))
    raw.s = np.clip(raw.s, 0.5, None)
    raw.x_lead, raw.x_follower = positions_from_gap(raw.v_lead, raw.s)
    raw.meta["position_source"] = "gap"
    raw.validate()
    rec = reconstruct_event(raw, ReconstructionConfig(sigma_x=0.2, sigma_v=0.2))
    assert rms(rec.s - truth["s"]) < 0.6 * rms(raw.s - truth["s"])


def test_parallel_run_equals_serial_run(noisy_events):
    events, _ = noisy_events
    serial, _ = reconstruct_events(events, CFG, workers=1)
    parallel, failed = reconstruct_events(events, CFG, workers=2, chunksize=2)
    assert failed == [] and [e.event_id for e in parallel] == [e.event_id for e in serial]
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a.s, b.s)
    stats = summarise(serial)
    assert stats["n_events"] == len(events) and stats["q99_max_abs_jerk"] <= CFG.jerk_max + 1e-3
    assert stats["q99_max_abs_jerk"] < stats["q99_max_abs_jerk_raw"]


def test_position_outliers_do_not_bend_the_trajectory(clean_events):
    """Tracking errors of several metres over a second: the quadratic loss follows them, Huber does not."""
    ev = copy.deepcopy(clean_events[0])
    rng = np.random.default_rng(3)
    ev.x_follower = ev.x_follower + 0.1 * rng.standard_normal(len(ev))
    ev.x_follower[200:210] -= 4.0
    ev.s = ev.x_lead - ev.x_follower
    truth = clean_events[0]
    plain = ReconstructionConfig(sigma_x=0.3, sigma_v=None, cutoff_hz=0.5, robust_iterations=0)
    robust = ReconstructionConfig(sigma_x=0.3, sigma_v=None, cutoff_hz=0.5)
    err_plain = rms(reconstruct_event(ev, plain).x_follower[150:260] - truth.x_follower[150:260])
    rec = reconstruct_event(ev, robust)
    err_robust = rms(rec.x_follower[150:260] - truth.x_follower[150:260])
    assert err_robust < 0.4 * err_plain
    assert rec.meta["reconstruction"]["outlier_share"][1] > 0.01
    assert np.max(np.abs(rec.a[150:260])) < np.max(np.abs(reconstruct_event(ev, plain).a[150:260]))


def test_cutoff_defines_the_jerk_weight():
    w_x, w_v, w_j = ReconstructionConfig(sigma_x=0.5, sigma_v=None, cutoff_hz=0.5).weights()
    wc = 2 * np.pi * 0.5
    assert w_v == 0.0 and w_x == pytest.approx(4.0) and w_j * wc**6 == pytest.approx(w_x)
    with pytest.raises(ValueError):
        ReconstructionConfig(sigma_x=None, sigma_v=None)
