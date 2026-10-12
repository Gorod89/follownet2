"""P1: closed loop over horizons of 10 s and 30 s (windows) and over the whole event.

A window of horizon ``h`` samples starts at ``k0 = warmup + j h`` (``j = 0, 1, ...``) and needs
``k0 + h <= len(event) - 1``: full windows only, stride equal to the horizon, so that every
simulated sample belongs to exactly one window of a horizon. The follower is rolled from the
observed state at ``k0`` with the observed history before it (``docs/protocols.md``, section 1).
The whole-event rollout is :func:`follownet2.protocols.closed_loop.evaluate_whole_event`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from follownet2.data.batch import WARMUP, EventBatch
from follownet2.data.schema import DT, Event
from follownet2.models.base import ACC_CLIP, CFModel
from follownet2.protocols.closed_loop import evaluate_whole_event, pooled_closed_loop
from follownet2.sim.rollout import rollout, rollout_errors

#: horizons of the windows [s]
HORIZONS: tuple[float, ...] = (10.0, 30.0)
COLUMNS = ("event_id", "follower_id", "start", "n", "sse_s", "sse_v", "collided", "first_collision")


def window_starts(n_samples: int, horizon: int, warmup: int = WARMUP) -> np.ndarray:
    """Prediction instants ``k0`` of the full windows of ``horizon`` samples in an event."""
    last = n_samples - 1 - horizon
    if last < warmup:
        return np.zeros(0, dtype=np.int64)
    return np.arange(warmup, last + 1, horizon, dtype=np.int64)


@torch.no_grad()
def evaluate_horizon(
    model: CFModel,
    events: Sequence[Event],
    horizon_s: float,
    warmup: int = WARMUP,
    clip: tuple[float, float] = ACC_CLIP,
    batch_windows: int = 4096,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> pd.DataFrame:
    """One row per window: ids, start instant, simulated samples, squared-error sums, collision.

    ``first_collision`` counts samples after ``start`` (``-1`` without collision).
    """
    events = list(events)
    h = int(round(horizon_s / DT))
    items = [(i, int(k0)) for i, ev in enumerate(events) for k0 in window_starts(len(ev), h, warmup)]
    if not items:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in COLUMNS})
    model = model.to(device).eval()
    tables = []
    for lo in range(0, len(items), batch_windows):
        chunk = items[lo : lo + batch_windows]
        batch = EventBatch.from_events(
            [events[i] for i, _ in chunk],
            device=device,
            dtype=dtype,
            start=[k0 - warmup for _, k0 in chunk],
            length=warmup + h + 1,
        )
        result = rollout(model, batch, warmup, clip)
        err = rollout_errors(result, batch)
        first = result.first_collision.cpu().numpy()
        tables.append(
            pd.DataFrame(
                {
                    "event_id": [events[i].event_id for i, _ in chunk],
                    "follower_id": [events[i].follower_id for i, _ in chunk],
                    "start": [k0 for _, k0 in chunk],
                    "n": err.n.cpu().numpy(),
                    "sse_s": err.sse_s.double().cpu().numpy(),
                    "sse_v": err.sse_v.double().cpu().numpy(),
                    "collided": err.collided.cpu().numpy(),
                    "first_collision": np.where(first >= 0, first - warmup, -1),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def pooled_horizon(table: pd.DataFrame) -> dict[str, float]:
    if len(table) == 0:
        return {"rmse_s": float("nan"), "rmse_v": float("nan"), "collision_rate": float("nan"),
                "event_collision_rate": float("nan"), "n_windows": 0, "n_events": 0, "n_samples": 0}
    n = float(table["n"].sum())
    per_event = table.groupby("event_id")["collided"].any()
    return {
        "rmse_s": float(np.sqrt(table["sse_s"].sum() / n)),
        "rmse_v": float(np.sqrt(table["sse_v"].sum() / n)),
        "collision_rate": float(table["collided"].mean()),
        "event_collision_rate": float(per_event.mean()),
        "n_windows": int(len(table)),
        "n_events": int(len(per_event)),
        "n_samples": int(n),
    }


def horizon_key(horizon_s: float) -> str:
    return f"h{int(round(horizon_s))}"


@torch.no_grad()
def evaluate_p1(
    model: CFModel,
    events: Sequence[Event],
    horizons: Sequence[float] = HORIZONS,
    warmup: int = WARMUP,
    clip: tuple[float, float] = ACC_CLIP,
    batch_windows: int = 4096,
    batch_events: int = 1024,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Summary per horizon (``h10``, ``h30``, ``whole``) and the per-window tables."""
    events = list(events)
    summary: dict = {}
    tables: dict[str, pd.DataFrame] = {}
    for horizon in horizons:
        key = horizon_key(horizon)
        tables[key] = evaluate_horizon(model, events, horizon, warmup, clip, batch_windows, device, dtype)
        summary[key] = {"horizon_s": float(horizon), **pooled_horizon(tables[key])}
    tables["whole"] = evaluate_whole_event(model, events, warmup, clip, batch_events, device, dtype)
    summary["whole"] = pooled_closed_loop(tables["whole"])
    return summary, tables
