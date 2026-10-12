"""Closed-loop rollout with gradients, and the batched environment of the policy learners.

Both advance the follower exactly like :func:`follownet2.sim.rollout.rollout` (semi-implicit
Euler, 0.1 s, acceleration clipped to [-8, 4], speed floored at 0), but start from any
prediction instant of a :class:`WindowSet` and keep only a sliding history instead of the
whole event, so that gradients can flow and memory stays small.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from follownet2.data.batch import WARMUP, S, V
from follownet2.models.base import ACC_CLIP, CFModel
from follownet2.train.windows import WindowSet


def advance_row(
    a: torch.Tensor, row: torch.Tensor, dx_lead: torch.Tensor, v_lead_next: torch.Tensor, clip, dt: float
) -> torch.Tensor:
    """Next state row (s, dv, v, a_prev) after applying the acceleration ``a`` to ``row``."""
    a = a.clamp(clip[0], clip[1])
    v_new = (row[..., V] + a * dt).clamp_min(0.0)
    s_new = row[..., S] + dx_lead - v_new * dt
    return torch.stack([s_new, v_new - v_lead_next, v_new, (v_new - row[..., V]) / dt], dim=-1)


def rollout_windows(
    model: CFModel,
    windows: WindowSet,
    index: torch.Tensor,
    horizon: int,
    clip: tuple[float, float] = ACC_CLIP,
    rows: int = WARMUP,
) -> dict[str, torch.Tensor]:
    """Roll the model for ``horizon`` steps from the instants ``index``; gradients are kept.

    Returns the simulated ``s``, ``v`` (state at the end of every step) and ``a`` (acceleration
    realised over it), each ``[B, horizon]``, next to the recorded ones of
    :meth:`WindowSet.future` under ``s_obs``, ``v_obs``, ``a_obs``.
    """
    future = windows.future(index, horizon)
    hist = windows.history(index, rows)
    dt = windows.batch.dt
    if rows == WARMUP:
        model.begin(hist)
    out = []
    for h in range(horizon):
        a = model(hist)
        row = advance_row(a, hist[:, -1], future["dx_lead"][:, h], future["v_lead"][:, h], clip, dt)
        out.append(row)
        hist = torch.cat([hist[:, 1:], row.unsqueeze(1)], dim=1)
    sim = torch.stack(out, dim=1)  # [B, horizon, 4]
    return {
        "s": sim[..., 0], "v": sim[..., 2], "a": sim[..., 3],
        "s_obs": future["s"], "v_obs": future["v"], "a_obs": future["a"],
    }


def closed_loop_loss(sim: dict[str, torch.Tensor], s_scale: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
    """Squared errors of gap and speed over the horizon, each in units of its standard deviation."""
    return torch.mean(((sim["s"] - sim["s_obs"]) / s_scale) ** 2) + torch.mean(((sim["v"] - sim["v_obs"]) / v_scale) ** 2)


@dataclass
class StepInfo:
    """Quantities of one environment step, each ``[B]``."""

    s: torch.Tensor  # gap at the end of the step
    v: torch.Tensor
    dv: torch.Tensor
    v_lead: torch.Tensor
    a: torch.Tensor  # acceleration realised over the step
    jerk: torch.Tensor  # change of the realised acceleration per second
    collided: torch.Tensor  # s <= 0 at the end of the step
    s_obs: torch.Tensor  # recorded follower, for imitation rewards
    v_obs: torch.Tensor
    a_obs: torch.Tensor
    # the episode was running during the step; False: the row is the frozen state after a collision
    alive: torch.Tensor | None = None


class FollowEnv:
    """Batched car-following environment behind recorded leaders, without gradients.

    An episode starts at a prediction instant of the windows with the observed history and
    lasts ``horizon`` steps or until a collision. The observation is the history
    ``[B, rows, 4]`` in physical units, the action the acceleration in m/s^2.
    """

    def __init__(self, windows: WindowSet, horizon: int, clip: tuple[float, float] = ACC_CLIP, rows: int = WARMUP) -> None:
        self.windows, self.horizon, self.clip, self.rows = windows, horizon, clip, rows
        self.dt = windows.batch.dt
        self.t = 0
        self.hist: torch.Tensor | None = None
        self.future: dict[str, torch.Tensor] = {}
        self.alive: torch.Tensor | None = None

    @torch.no_grad()
    def reset(self, index: torch.Tensor) -> torch.Tensor:
        self.future = self.windows.future(index, self.horizon)
        self.hist = self.windows.history(index, self.rows).clone()
        self.alive = torch.ones(len(index), dtype=torch.bool, device=self.hist.device)
        self.t = 0
        return self.hist

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, StepInfo, torch.Tensor]:
        """Apply the actions; returns the next observation, the step quantities and ``done``.

        Episodes that ended keep their last state; their entries of ``done`` stay true.
        """
        if self.hist is None or self.t >= self.horizon:
            raise RuntimeError("reset the environment before stepping")
        h, last = self.t, self.hist[:, -1]
        row = advance_row(action, last, self.future["dx_lead"][:, h], self.future["v_lead"][:, h], self.clip, self.dt)
        row = torch.where(self.alive.unsqueeze(-1), row, last)
        collided = (row[:, 0] <= 0.0) & self.alive
        info = StepInfo(
            s=row[:, 0], v=row[:, 2], dv=row[:, 1], v_lead=self.future["v_lead"][:, h], a=row[:, 3],
            jerk=(row[:, 3] - last[:, 3]) / self.dt, collided=collided, alive=self.alive.clone(),
            s_obs=self.future["s"][:, h], v_obs=self.future["v"][:, h], a_obs=self.future["a"][:, h],
        )
        self.hist = torch.cat([self.hist[:, 1:], row.unsqueeze(1)], dim=1)
        self.t += 1
        self.alive = self.alive & ~collided
        done = ~self.alive | (self.t >= self.horizon)
        return self.hist, info, done
