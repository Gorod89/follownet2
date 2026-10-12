"""P0: one-step prediction.

At every instant ``k >= warmup`` the model sees the observed history up to ``t[k]`` and
predicts the acceleration applied over the next 0.1 s. Scores:

* ``RMSE_a``: prediction against the recorded ``a[k]``;
* ``RMSE_v``: ``v[k] + a_hat * dt`` against the recorded ``v[k+1]``;
* ``RMSE_s``: ``s[k] + (x_lead[k+1] - x_lead[k]) - v_hat * dt`` against the recorded ``s[k+1]``.

The prediction is clipped to the acceleration range of the benchmark, ``[-8, 4]``, like in
every rollout: a prediction outside the range is a prediction of its limit (without the clip
the closed formulas are scored by a few states with a small gap, where they ask for several
hundred m/s^2). In the reconstructed data version the three scores are proportional
(``RMSE_v = dt * RMSE_a``, ``RMSE_s = dt^2 * RMSE_a``) because the data obey the integration
scheme exactly; in the raw version they differ by the inconsistency of the recorded series,
which :class:`RecordedAcceleration` measures.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from follownet2.data.batch import WARMUP, EventBatch, length_buckets
from follownet2.data.schema import Event
from follownet2.models.base import ACC_CLIP, CFModel


class RecordedAcceleration(CFModel):
    """Oracle returning the recorded acceleration: the noise floor of RMSE_v and RMSE_s.

    Not a model: it is handled inside :func:`one_step`, which feeds it the target.
    """

    name = "recorded_acceleration"
    family = "baseline"


@torch.no_grad()
def one_step(
    model: CFModel,
    batch: EventBatch,
    warmup: int = WARMUP,
    clip: tuple[float, float] | None = ACC_CLIP,
    max_windows: int = 1 << 18,
) -> dict[str, torch.Tensor]:
    """Per-event sums of squared one-step errors, each of shape ``[E]``.

    ``clip=None`` scores the raw output of the model.
    """
    history, dt = model.history, batch.dt
    if warmup + 1 < history:
        raise ValueError(f"warm-up of {warmup} samples is shorter than the history {history} of {model.name}")
    k = torch.arange(warmup, batch.n_steps - 1, device=batch.device)
    if isinstance(model, RecordedAcceleration):
        a_hat = batch.a[:, k]
    elif model.adaptive:
        # adapted once per event on the observed warm-up, then asked for all its instants
        state = batch.state()
        model.begin(state[:, warmup - WARMUP + 1 : warmup + 1])
        windows = state.unfold(1, history, 1).permute(0, 1, 3, 2)[:, k - history + 1]
        chunk = max(1, max_windows // max(windows.shape[0], 1))
        parts = [model(windows[:, i : i + chunk]) for i in range(0, windows.shape[1], chunk)]
        a_hat = torch.cat(parts, dim=1).to(batch.s.dtype)
    else:
        windows = batch.state().unfold(1, history, 1).permute(0, 1, 3, 2)  # [E, T - H + 1, H, 4]
        windows = windows[:, k - history + 1]
        flat = windows.reshape(-1, history, windows.shape[-1])
        out = [model(flat[i : i + max_windows]) for i in range(0, flat.shape[0], max_windows)]
        a_hat = torch.cat(out).reshape(windows.shape[:2]).to(batch.s.dtype)
    invalid = ~torch.isfinite(a_hat)
    if clip is not None and not isinstance(model, RecordedAcceleration):
        a_hat = torch.where(invalid, torch.full_like(a_hat, clip[0]), a_hat).clamp(clip[0], clip[1])
    v_hat = (batch.v[:, k] + a_hat * dt).clamp_min(0.0)
    s_hat = batch.s[:, k] + batch.dx_lead[:, k] - v_hat * dt
    mask = batch.mask[:, k + 1]
    zero = torch.zeros((), dtype=batch.s.dtype, device=batch.device)
    sq = lambda err: torch.where(mask, err * err, zero).sum(-1)  # noqa: E731
    return {
        "n": mask.sum(-1),
        "sse_a": sq(a_hat - batch.a[:, k]),
        "sse_v": sq(v_hat - batch.v[:, k + 1]),
        "sse_s": sq(s_hat - batch.s[:, k + 1]),
        "n_invalid": (invalid & mask).sum(-1),
    }


def evaluate_p0(
    model: CFModel,
    events: Sequence[Event],
    warmup: int = WARMUP,
    batch_events: int = 1024,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    clip: tuple[float, float] | None = ACC_CLIP,
) -> pd.DataFrame:
    """One row per event: ids, sample count and sums of squared errors."""
    events = list(events)
    if not isinstance(model, RecordedAcceleration):
        model = model.to(device).eval()
    tables = []
    for index in length_buckets([len(ev) for ev in events], batch_events):
        chunk = [events[i] for i in index]
        sums = one_step(model, EventBatch.from_events(chunk, device=device, dtype=dtype), warmup, clip)
        table = pd.DataFrame({name: value.double().cpu().numpy() for name, value in sums.items()})
        table.insert(0, "follower_id", [ev.follower_id for ev in chunk])
        table.insert(0, "event_id", [ev.event_id for ev in chunk])
        table.index = index
        tables.append(table)
    return pd.concat(tables).sort_index().reset_index(drop=True)


def pooled_rmse(table: pd.DataFrame, prefix: str = "sse_") -> dict[str, float]:
    """Sample-weighted RMSE over all events for every ``sse_*`` column."""
    n = float(table["n"].sum())
    return {
        "rmse_" + col[len(prefix) :]: float(np.sqrt(table[col].sum() / n))
        for col in table.columns
        if col.startswith(prefix)
    }
