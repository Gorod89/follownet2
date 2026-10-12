from __future__ import annotations

import numpy as np
import pytest
import torch

from follownet2.data.batch import WARMUP, EventBatch
from follownet2.models.mechanistic import IDM
from follownet2.models.persistence import PersistenceBaseline
from follownet2.protocols.p1 import evaluate_horizon, evaluate_p1, pooled_horizon, window_starts
from follownet2.sim.rollout import rollout


def test_window_starts_are_full_windows_with_stride_equal_to_the_horizon():
    assert window_starts(331, 100).tolist() == [30, 130, 230]  # 230 + 100 = 330 = n - 1
    assert window_starts(330, 100).tolist() == [30, 130]
    assert window_starts(131, 100).tolist() == [30]
    assert window_starts(130, 100).tolist() == []
    assert window_starts(150, 300).tolist() == []


def test_the_true_model_has_no_error_at_any_horizon(clean_events):
    model = IDM(clean_events[0].meta["theta_true"])
    summary, tables = evaluate_p1(model, clean_events, horizons=(10.0, 30.0), dtype=torch.float64)
    for key in ("h10", "h30", "whole"):
        assert summary[key]["rmse_s"] < 1e-6 and summary[key]["rmse_v"] < 1e-6
        assert summary[key]["collision_rate"] == 0.0
    assert summary["h10"]["n_windows"] > summary["h30"]["n_windows"] > 0
    assert summary["h10"]["n_samples"] == 100 * summary["h10"]["n_windows"]
    assert set(tables["h10"]["event_id"]) <= {ev.event_id for ev in clean_events}


def test_the_first_window_is_the_prefix_of_the_whole_event_rollout(clean_events):
    model = PersistenceBaseline()
    ev = max(clean_events, key=len)
    table = evaluate_horizon(model, [ev], 10.0, dtype=torch.float64)
    first = table[table["start"] == WARMUP].iloc[0]
    batch = EventBatch.from_events([ev], dtype=torch.float64)
    whole = rollout(model, batch)
    sim = slice(WARMUP + 1, WARMUP + 101)
    sse_s = float(((whole.s[0, sim] - batch.s[0, sim]) ** 2).sum())
    sse_v = float(((whole.v[0, sim] - batch.v[0, sim]) ** 2).sum())
    assert first["n"] == 100
    assert first["sse_s"] == pytest.approx(sse_s, rel=1e-9) and first["sse_v"] == pytest.approx(sse_v, rel=1e-9)


def test_later_windows_start_from_the_observed_state(clean_events):
    """A window at k0 is rolled from the recorded state at k0, not from the previous window."""
    model = PersistenceBaseline()
    ev = max(clean_events, key=len)
    starts = window_starts(len(ev), 100)
    assert len(starts) >= 2
    table = evaluate_horizon(model, [ev], 10.0, dtype=torch.float64)
    k0 = int(starts[1])
    batch = EventBatch.from_events([ev], dtype=torch.float64, start=[k0 - WARMUP], length=WARMUP + 101)
    assert float(batch.s[0, WARMUP]) == pytest.approx(ev.s[k0])
    result = rollout(model, batch)
    sim = slice(WARMUP + 1, WARMUP + 101)
    sse_s = float(((result.s[0, sim] - batch.s[0, sim]) ** 2).sum())
    row = table[table["start"] == k0].iloc[0]
    assert row["sse_s"] == pytest.approx(sse_s, rel=1e-9)


def test_determinism_and_pooling(clean_events):
    model = PersistenceBaseline()
    a = evaluate_horizon(model, clean_events, 10.0)
    b = evaluate_horizon(model, clean_events, 10.0)
    assert np.array_equal(a["sse_s"].to_numpy(), b["sse_s"].to_numpy())
    pooled = pooled_horizon(a)
    assert pooled["rmse_s"] == pytest.approx(np.sqrt(a["sse_s"].sum() / a["n"].sum()))
    assert pooled["n_events"] == a["event_id"].nunique()
    assert 0.0 <= pooled["event_collision_rate"] <= 1.0
    empty = evaluate_horizon(model, [clean_events[0]], 1000.0)
    assert len(empty) == 0 and pooled_horizon(empty)["n_windows"] == 0
