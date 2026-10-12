from __future__ import annotations

import numpy as np
import pytest
import torch

from follownet2.data.batch import WARMUP, EventBatch
from follownet2.data.schema import DT
from follownet2.models.mechanistic import IDM
from follownet2.protocols.scenarios import (
    HOLD,
    constant_speed_event,
    initial_gap,
    positions,
    scenario_event,
    speed_from_segments,
    with_hold,
)
from follownet2.sim.rollout import rollout


def test_speed_from_segments_integrates_piecewise_constant_acceleration():
    v = speed_from_segments(25.0, [(-3.0, 5.0), (0.0, 2.0), (1.5, 10.0)])
    assert len(v) == 1 + 50 + 20 + 100
    assert v[0] == 25.0 and v[50] == pytest.approx(10.0) and v[70] == pytest.approx(10.0)
    assert v[-1] == pytest.approx(25.0)
    assert speed_from_segments(3.0, [(-4.0, 2.0)]).min() == 0.0  # floored at zero


def test_hold_phase_and_positions():
    v, start = with_hold(np.array([12.0, 11.0, 10.0]), hold=60.0)
    assert start == 600 and len(v) == 603
    assert np.all(v[:600] == 12.0) and v[600] == 12.0
    x = positions(v)
    assert x[0] == 0.0 and x[1] == pytest.approx(v[1] * DT) and x[-1] == pytest.approx(np.sum(v[1:]) * DT)
    with pytest.raises(ValueError):
        with_hold(np.array([1.0]), hold=1.0)


def test_scenario_event_matches_the_batch_conventions():
    v = speed_from_segments(20.0, [(-2.0, 5.0), (0.0, 5.0)])
    ev = scenario_event(v, initial_gap(20.0), "test", hold=HOLD, meta={"scenario": {"kind": "pulse"}})
    assert ev.meta["manoeuvre_start"] == 600 and ev.meta["data_version"] == "raw"
    assert ev.meta["scenario"] == {"kind": "pulse"}
    assert np.all(ev.s == initial_gap(20.0)) and np.all(ev.dv == 0.0)
    assert np.allclose(ev.s, ev.x_lead - ev.x_follower)
    batch = EventBatch.from_events([ev], dtype=torch.float64)
    state = batch.state()
    assert torch.all(state[0, :WARMUP, 0] == initial_gap(20.0))
    assert torch.all(state[0, :WARMUP, 3] == 0.0)  # a_prev of the hold
    assert np.allclose(batch.dx_lead[0, :-1].numpy(), ev.v_lead[1:] * DT)


def test_the_idm_settles_during_the_hold_and_survives_the_pulse():
    v = speed_from_segments(25.0, [(-3.0, 5.0), (0.0, 10.0), (1.5, 10.0), (0.0, 20.0)])
    ev = scenario_event(v, initial_gap(25.0), "pulse")
    result = rollout(IDM(), EventBatch.from_events([ev], dtype=torch.float64))
    start = ev.meta["manoeuvre_start"]
    assert not bool(result.collided[0])
    # settled at the end of the hold (within 0.5 m of the analytic equilibrium, 47.8 m, which
    # differs from the 39.5 m of the reference rule), then the pulse closes the gap
    assert abs(float(result.a[0, start - 50 : start].abs().max())) < 2e-3
    assert abs(float(result.s[0, start]) - 47.819) < 0.5
    assert float(result.s[0, start:].min()) < float(result.s[0, start]) - 10.0


def test_constant_speed_event():
    ev = constant_speed_event(15.0, 30.0, "eq")
    assert np.all(ev.v_lead == 15.0) and len(ev) == 600 + 301
