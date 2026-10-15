"""Mechanistic car-following models: IDM, Gipps, Newell.

All formulas use the net gap ``s`` and ``dv = v - v_lead`` (positive when closing in).
Gipps and Newell are delay models; with a 0.1 s step they are evaluated at every step on
the state delayed by the reaction time, read from the state history (linear interpolation
between rows for reaction times that are not a multiple of the step).

The closed formulas are TorchScript functions: on the GPU their elementwise operations are
fused into one kernel, which matters in rollouts, where a formula is evaluated once per step
on small tensors and the launch of the kernels is the whole cost.
"""

from __future__ import annotations

import torch

from trafficloop.data.batch import DV, S, V, WARMUP
from trafficloop.data.schema import DT
from trafficloop.models.base import MechanisticModel


@torch.jit.script
def idm_acc(
    s: torch.Tensor,
    dv: torch.Tensor,
    v: torch.Tensor,
    v0: torch.Tensor,
    T: torch.Tensor,
    s0: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    delta: float = 4.0,
    s_floor: float = 0.1,
) -> torch.Tensor:
    """Intelligent Driver Model (Treiber, Hennecke, Helbing 2000)."""
    s = s.clamp_min(s_floor)
    v = v.clamp_min(0.0)
    s_star = s0 + (v * T + v * dv / (2.0 * torch.sqrt(a * b))).clamp_min(0.0)
    return a * (1.0 - (v / v0) ** delta - (s_star / s) ** 2)


class IDM(MechanisticModel):
    name = "idm"
    history = 1
    memoryless = True
    param_names = ("v0", "T", "s0", "a", "b")
    bounds = ((10.0, 45.0), (0.3, 3.0), (0.5, 10.0), (0.2, 4.0), (0.5, 6.0))
    default = (33.3, 1.5, 2.0, 1.0, 1.5)

    def acc(self, hist: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        now = hist[..., -1, :]
        v0, T, s0, a, b = params.unbind(-1)
        return idm_acc(now[..., S], now[..., DV], now[..., V], v0, T, s0, a, b)

    def acc_state(self, s, dv, v, params: torch.Tensor | None = None) -> torch.Tensor:
        """Acceleration as a function of the state, for the analytic stability criterion."""
        params = self.params.to(s.dtype) if params is None else params
        v0, T, s0, a, b = params.unbind(-1)
        return idm_acc(s, dv, v, v0, T, s0, a, b)


def _delay_rows(n: int, tau: torch.Tensor, dt: float, shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
    """The two history rows around ``t_now + dt - tau`` and the weight of the later one.

    The history has ``n`` rows (last row = now) and the batch shape ``shape``; ``tau``
    broadcasts against it and is limited to ``[dt, n * dt]``. Returns the row indices
    ``[*shape, 2]`` and the weight, which has the shape of ``tau``.
    """
    pos = (n - tau / dt).clamp(0.0, n - 1.0)
    lo = pos.floor().clamp(max=max(n - 2, 0))
    rows = torch.stack([lo, (lo + 1).clamp(max=n - 1)], dim=-1).long()
    return rows.expand(*shape, 2), pos - lo


def _delayed(values: torch.Tensor, tau: torch.Tensor, dt: float) -> torch.Tensor:
    """Value of a history series ``[..., H]`` at ``t_now + dt - tau`` (linear interpolation)."""
    rows, w = _delay_rows(values.shape[-1], tau, dt, values.shape[:-1])
    left, right = values.gather(-1, rows).unbind(-1)
    return left + w * (right - left)


@torch.jit.script
def gipps_speed(
    s: torch.Tensor,
    v: torch.Tensor,
    v_lead: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    v_des: torch.Tensor,
    tau: torch.Tensor,
    s0: torch.Tensor,
    b_hat: torch.Tensor,
) -> torch.Tensor:
    """Speed at ``t + tau`` of Gipps (1981), with positive braking rates ``b`` and ``b_hat``."""
    v = v.clamp_min(0.0)
    ratio = v / v_des
    v_acc = v + 2.5 * a * tau * (1.0 - ratio) * torch.sqrt(0.025 + ratio)
    radicand = b * b * tau * tau + b * (2.0 * (s - s0) - v * tau + v_lead.clamp_min(0.0) ** 2 / b_hat)
    v_safe = -b * tau + torch.sqrt(radicand.clamp_min(0.0))
    return torch.minimum(v_acc, v_safe).clamp_min(0.0)


class Gipps(MechanisticModel):
    """Gipps (1981) as a delayed map, ``v(t + tau) = G(state(t))``.

    The speed reached at the end of the step is ``G`` of the state ``tau`` earlier, so the
    output acceleration is ``(G - v_now) / dt``. ``s0`` is the net gap kept at standstill
    (the effective size of the leader minus its physical length).
    """

    name = "gipps"
    history = WARMUP
    param_names = ("a", "b", "V", "tau", "s0", "b_hat")
    bounds = ((0.5, 4.0), (0.5, 8.0), (10.0, 45.0), (0.3, 3.0), (0.5, 10.0), (0.5, 8.0))
    default = (1.7, 3.0, 33.3, 0.7, 2.0, 3.5)

    def acc(self, hist: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        a, b, v_des, tau, s0, b_hat = (p.unsqueeze(-1) for p in params.unbind(-1))
        # the formula is evaluated on the two rows around the delayed instant only
        rows, w = _delay_rows(hist.shape[-2], tau.squeeze(-1), DT, hist.shape[:-2])
        s, dv, v = (hist[..., c].gather(-1, rows) for c in (S, DV, V))
        left, right = gipps_speed(s, v, v - dv, a, b, v_des, tau, s0, b_hat).unbind(-1)
        return (left + w * (right - left) - hist[..., -1, V]) / DT


class Newell(MechanisticModel):
    """Newell (2002): the follower repeats the leader shifted by ``tau`` and ``d``.

    ``x(t + tau) = min(x(t) + V * tau, x_lead(t) - d)`` with ``d`` the net gap at standstill
    (positions are bumper positions). At every step the target position is
    ``min(x_now + V * dt, x_lead(t_now + dt - tau) - d)``; the rollout clips the implied
    acceleration, which makes it the bounded-acceleration variant of the model.
    """

    name = "newell"
    history = WARMUP
    param_names = ("tau", "d", "V")
    bounds = ((0.3, 3.0), (0.5, 15.0), (10.0, 45.0))
    default = (1.2, 4.0, 33.3)

    def acc(self, hist: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        tau, d, v_des = params.unbind(-1)
        s, v = hist[..., S], hist[..., V]
        # follower displacement from every row to now (semi-implicit Euler: x[i] - x[i-1] = v[i] * dt)
        travelled = torch.cumsum(v, dim=-1)
        lead_rel = s - (travelled[..., -1:] - travelled) * DT  # x_lead(row) - x_now
        step = torch.minimum(v_des * DT, _delayed(lead_rel, tau, DT) - d)
        v_next = (step / DT).clamp_min(0.0)
        return (v_next - v[..., -1]) / DT


MECHANISTIC = {cls.name: cls for cls in (IDM, Gipps, Newell)}
