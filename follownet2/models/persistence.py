"""Persistence baseline: the next acceleration equals the last known one."""

from __future__ import annotations

import torch

from follownet2.data.batch import A_PREV
from follownet2.models.base import CFModel


class PersistenceBaseline(CFModel):
    """``a_hat(t + 0.1) = a(t)``.

    Its one-step error is the acceleration increment, so the RMSE of the acceleration is
    exactly the root mean square of ``jerk * dt`` (identity tested in E5). In a closed-loop
    rollout it is a constant-acceleration follower.
    """

    name = "persistence"
    family = "baseline"
    history = 1

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return hist[..., -1, A_PREV]


class ZeroAcceleration(CFModel):
    """Constant-speed follower, the trivial reference below persistence."""

    name = "constant_speed"
    family = "baseline"
    history = 1

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(hist[..., -1, A_PREV])
