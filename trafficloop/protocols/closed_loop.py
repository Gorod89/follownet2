"""Closed-loop scores of any model over whole events (the core of protocol P1).

The follower is rolled behind the recorded leader from the end of the warm-up to the end of
the event; one row per event with the sums that the driver-level bootstrap resamples.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from trafficloop.data.batch import WARMUP, EventBatch, length_buckets
from trafficloop.data.schema import Event
from trafficloop.models.base import ACC_CLIP, CFModel
from trafficloop.sim.rollout import rollout, rollout_errors


@torch.no_grad()
def evaluate_whole_event(
    model: CFModel,
    events: Sequence[Event],
    warmup: int = WARMUP,
    clip: tuple[float, float] = ACC_CLIP,
    batch_events: int = 1024,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> pd.DataFrame:
    events = list(events)
    model = model.to(device).eval()
    tables = []
    for index in length_buckets([len(ev) for ev in events], batch_events):
        chunk = [events[i] for i in index]
        batch = EventBatch.from_events(chunk, device=device, dtype=dtype)
        result = rollout(model, batch, warmup, clip)
        err = rollout_errors(result, batch)
        table = pd.DataFrame(
            {
                "event_id": [ev.event_id for ev in chunk],
                "follower_id": [ev.follower_id for ev in chunk],
                "n": err.n.cpu().numpy(),
                "sse_s": err.sse_s.double().cpu().numpy(),
                "sse_v": err.sse_v.double().cpu().numpy(),
                "collided": err.collided.cpu().numpy(),
                "first_collision": result.first_collision.cpu().numpy(),
            }
        )
        table.index = index
        tables.append(table)
    return pd.concat(tables).sort_index().reset_index(drop=True)


def pooled_closed_loop(table: pd.DataFrame) -> dict[str, float]:
    n = float(table["n"].sum())
    return {
        "rmse_s": float(np.sqrt(table["sse_s"].sum() / n)),
        "rmse_v": float(np.sqrt(table["sse_v"].sum() / n)),
        "collision_rate": float(table["collided"].mean()),
        "n_events": int(len(table)),
        "n_samples": int(n),
    }
