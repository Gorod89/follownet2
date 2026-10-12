"""Closed-loop rollout of a follower behind a recorded leader.

Integration (binding, identical in cf-stability): semi-implicit Euler with a 0.1 s step,

    a[k]   = clip(model(history up to k), -8, 4)
    v[k+1] = max(v[k] + a[k] * dt, 0)
    x[k+1] = x[k] + v[k+1] * dt
    s[k+1] = x_lead[k+1] - x[k+1] = s[k] + (x_lead[k+1] - x_lead[k]) - v[k+1] * dt

The first ``warmup`` samples of every event are observed; the state at index ``warmup`` is
the initial condition and the first simulated sample has index ``warmup + 1``. All models
are rolled from the same index whatever their own history length, so that they are scored
on identical samples. A collision is ``s <= 0`` at any simulated sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from follownet2.data.batch import A_PREV, WARMUP, EventBatch, S, V
from follownet2.models.base import ACC_CLIP, CFModel, MechanisticModel


@torch.jit.script
def advance(
    a: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    dx_lead: torch.Tensor,
    v_lead_next: torch.Tensor,
    low: float,
    high: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the integrator: the next state row (s, dv, v, a_prev) and the number of
    non-finite accelerations, which are replaced by the lower clip.

    Scripted so that the GPU runs it as few kernels; nothing is read back from the device.
    """
    bad = ~torch.isfinite(a)
    a = torch.where(bad, torch.full_like(a, low), a).clamp(low, high)
    v_new = (v + a * dt).clamp_min(0.0)
    s_new = s + dx_lead - v_new * dt
    row = torch.stack([s_new, v_new - v_lead_next, v_new, (v_new - v) / dt], dim=-1)
    return row, bad.sum()


@dataclass
class RolloutResult:
    s: torch.Tensor  # [*lead, E, T] gap; observed up to index k0, simulated afterwards
    v: torch.Tensor
    a: torch.Tensor  # acceleration realised over the step k -> k+1 (zero outside the rollout)
    sim_mask: torch.Tensor  # [E, T] True for simulated samples inside the event
    collided: torch.Tensor  # [*lead, E]
    first_collision: torch.Tensor  # [*lead, E] sample index, -1 without collision
    n_invalid: int  # non-finite model outputs (replaced by the lower clip)
    k0: int


@torch.no_grad()
def rollout(
    model: CFModel,
    batch: EventBatch,
    warmup: int = WARMUP,
    clip: tuple[float, float] = ACC_CLIP,
    params: torch.Tensor | None = None,
) -> RolloutResult:
    """Roll every event of the batch from index ``warmup`` to its end.

    ``params`` (mechanistic models only) has shape ``[*lead, E or 1, D]``; the result then
    carries the leading dimensions ``lead`` (one rollout per candidate parameter vector).
    """
    n_steps, dt = batch.n_steps, batch.dt
    history = model.history
    if warmup + 1 < history:
        raise ValueError(f"warm-up of {warmup} samples is shorter than the history {history} of {model.name}")
    if n_steps < warmup + 2:
        raise ValueError(f"events of {n_steps} samples are too short for a warm-up of {warmup}")
    state = batch.state()
    if params is not None:
        if not isinstance(model, MechanisticModel):
            raise TypeError("params can be given for mechanistic models only")
        params = params.to(device=state.device, dtype=state.dtype)
        lead = params.shape[:-2]
        state = state.expand(*lead, *state.shape)
    state = state.clone()
    if params is None and warmup >= WARMUP - 1:
        model.begin(state[..., warmup - WARMUP + 1 : warmup + 1, :])
    acc = torch.zeros_like(state[..., 0])
    n_invalid = torch.zeros((), dtype=torch.long, device=state.device)
    for k in range(warmup, n_steps - 1):
        hist = state[..., k - history + 1 : k + 1, :]
        a = model(hist) if params is None else model.acc(hist, params)
        row, n_bad = advance(
            a, state[..., k, S], state[..., k, V], batch.dx_lead[:, k], batch.v_lead[:, k + 1], clip[0], clip[1], dt
        )
        state[..., k + 1, :] = row
        acc[..., k] = row[..., A_PREV]
        n_invalid = n_invalid + n_bad
    steps = torch.arange(n_steps, device=state.device)
    sim_mask = batch.mask & (steps > warmup)
    hit = (state[..., S] <= 0.0) & sim_mask
    collided = hit.any(dim=-1)
    first = torch.where(hit, steps, torch.full_like(steps, n_steps)).amin(dim=-1)
    first = torch.where(collided, first, torch.full_like(first, -1))
    return RolloutResult(state[..., S], state[..., V], acc, sim_mask, collided, first, int(n_invalid), warmup)


@dataclass
class RolloutErrors:
    """Sums over the simulated samples of every event, shape ``[*lead, E]``."""

    sse_s: torch.Tensor
    sse_v: torch.Tensor
    ss_s: torch.Tensor  # [E] sum of squares of the observed gap
    ss_v: torch.Tensor
    n: torch.Tensor  # [E] simulated samples
    n_overlap: torch.Tensor  # samples with s <= 0
    collided: torch.Tensor


def rollout_errors(result: RolloutResult, batch: EventBatch) -> RolloutErrors:
    mask = result.sim_mask
    zero = torch.zeros((), dtype=batch.s.dtype, device=batch.s.device)
    err_s = torch.where(mask, result.s - batch.s, zero)
    err_v = torch.where(mask, result.v - batch.v, zero)
    return RolloutErrors(
        sse_s=(err_s * err_s).sum(-1),
        sse_v=(err_v * err_v).sum(-1),
        ss_s=torch.where(mask, batch.s * batch.s, zero).sum(-1),
        ss_v=torch.where(mask, batch.v * batch.v, zero).sum(-1),
        n=mask.sum(-1),
        n_overlap=((result.s <= 0.0) & mask).sum(-1),
        collided=result.collided,
    )
