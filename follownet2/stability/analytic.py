"""Analytic string stability of memoryless car-following models (model contract, section 3).

A memoryless model is a law ``a = f(s, dv, v)`` with ``dv = v - v_lead``. Linearised about
the equilibrium ``(s_e(v), 0, v)``, a platoon of identical followers is string stable when
(Wilson and Ward 2011, written for ``dv = v - v_lead``)

    m = f_v^2 + 2 f_v f_dv - 2 f_s >= 0,

and driving is rational when ``f_s > 0``, ``f_dv < 0``, ``f_v < 0``.

Every function takes a model of the benchmark with ``memoryless = True`` and evaluates it
through its interface, on one-row histories ``[..., 1, 4]`` with the channels
``(s, dv, v, a_prev = 0)``, so nothing but ``forward`` is needed. Inputs are built in float64
on the device of the model; the model computes in its own dtype.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import torch
from torch import Tensor

from follownet2.models.base import CFModel

#: equilibrium speeds of the penalty and of the unstable fraction [m/s]: 5, 6, ..., 30
SPEEDS: tuple[float, ...] = tuple(float(v) for v in range(5, 31))

Speeds = Sequence[float] | Tensor


def _device(model: CFModel) -> torch.device:
    for tensor in itertools.chain(model.parameters(), model.buffers()):
        return tensor.device
    return torch.device("cpu")


def _as_float64(x, device: torch.device) -> Tensor:
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def acceleration(model: CFModel, s, dv, v) -> Tensor:
    """``f(s, dv, v)`` of a memoryless model; the three arguments broadcast against each other."""
    if not getattr(model, "memoryless", False):
        raise ValueError(f"{model.name} is not memoryless: the analytic criterion does not apply")
    device = _device(model)
    s, dv, v = torch.broadcast_tensors(*(_as_float64(x, device) for x in (s, dv, v)))
    return model(torch.stack([s, dv, v, torch.zeros_like(s)], dim=-1).unsqueeze(-2))


@torch.no_grad()
def equilibrium_gap(model: CFModel, v: Speeds, low: float = 0.5, high: float = 300.0, iterations: int = 60) -> Tensor:
    """Equilibrium gap ``s_e(v)``: root of ``f(s, 0, v) = 0`` in ``[low, high]`` by bisection.

    Vectorised over the speeds; ``nan`` where ``f(low, 0, v)`` and ``f(high, 0, v)`` do not have
    opposite signs. Sixty halvings of ``[0.5, 300]`` reach the float64 resolution of the gap.
    """
    v = _as_float64(v, _device(model))
    lo, hi, zero = torch.full_like(v, low), torch.full_like(v, high), torch.zeros_like(v)
    sign_lo = torch.sign(acceleration(model, lo, zero, v))
    found = sign_lo * torch.sign(acceleration(model, hi, zero, v)) < 0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        same = torch.sign(acceleration(model, mid, zero, v)) == sign_lo
        lo, hi = torch.where(same, mid, lo), torch.where(same, hi, mid)
    return torch.where(found, 0.5 * (lo + hi), torch.full_like(v, float("nan")))


def partials(model: CFModel, s, dv, v, create_graph: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    """``(f_s, f_dv, f_v)`` at the states ``(s, dv, v)`` by automatic differentiation.

    With ``create_graph`` the derivatives stay differentiable with respect to the parameters
    of the model (for a loss); otherwise they carry no graph. An input the output does not
    depend on has a zero derivative.
    """
    device = _device(model)
    state = torch.broadcast_tensors(*(_as_float64(x, device) for x in (s, dv, v)))
    leaves = [x.detach().clone().requires_grad_(True) for x in state]
    with torch.enable_grad():
        acc = acceleration(model, *leaves)
        if acc.requires_grad:
            grads = torch.autograd.grad(acc.sum(), leaves, create_graph=create_graph, allow_unused=True)
        else:  # a model that does not look at its input
            grads = (None, None, None)
    f_s, f_dv, f_v = (torch.zeros_like(x) if g is None else g for x, g in zip(leaves, grads))
    return f_s, f_dv, f_v


def string_margin(f_s: Tensor, f_dv: Tensor, f_v: Tensor) -> Tensor:
    """``m = f_v^2 + 2 f_v f_dv - 2 f_s``; string stable where ``m >= 0``."""
    return f_v**2 + 2.0 * f_v * f_dv - 2.0 * f_s


@torch.no_grad()
def equilibrium_margin(model: CFModel, speeds: Speeds = SPEEDS) -> Tensor:
    """Margin ``m`` at the equilibrium of every speed; ``nan`` where there is no equilibrium."""
    v = _as_float64(speeds, _device(model)).reshape(-1)
    s_e = equilibrium_gap(model, v)
    found = torch.isfinite(s_e)
    m = torch.full_like(v, float("nan"))
    if bool(found.any()):
        m[found] = string_margin(*partials(model, s_e[found], 0.0, v[found])).to(m.dtype)
    return m


def stability_penalty(
    model: CFModel, speeds: Speeds = SPEEDS, margin: float = 0.0, neighbourhood: float = 0.0
) -> Tensor:
    """``mean(relu(margin - m)) + mean(relu(-f_s) + relu(f_dv) + relu(f_v))`` over the equilibria.

    Speeds without an equilibrium are left out (zero when none has one). The penalty is
    differentiable with respect to the parameters of the model at fixed equilibrium gaps:
    the bisection is not differentiated.

    With ``neighbourhood = r > 0`` the margin term is also evaluated at the four states
    ``(s_e (1 +- r), dv = -+ 5 r m/s, v_e)`` around every equilibrium (mean over the four), a
    proxy for string stability under finite perturbations: the linearisation at the exact
    equilibrium says nothing about a model whose Jacobian changes within a metre of gap or a
    metre per second of relative speed.
    """
    v = _as_float64(speeds, _device(model)).reshape(-1)
    s_e = equilibrium_gap(model, v)
    keep = torch.isfinite(s_e)
    if not bool(keep.any()):
        return torch.zeros((), dtype=torch.float64, device=v.device)
    f_s, f_dv, f_v = partials(model, s_e[keep], 0.0, v[keep], create_graph=True)
    relu = torch.relu
    total = relu(margin - string_margin(f_s, f_dv, f_v)).mean() + (relu(-f_s) + relu(f_dv) + relu(f_v)).mean()
    if neighbourhood > 0.0:
        for gap_factor in (1.0 - neighbourhood, 1.0 + neighbourhood):
            for dv in (-5.0 * neighbourhood, 5.0 * neighbourhood):
                g_s, g_dv, g_v = partials(model, s_e[keep] * gap_factor, dv, v[keep], create_graph=True)
                total = total + relu(margin - string_margin(g_s, g_dv, g_v)).mean() / 4.0
    return total


def unstable_fraction(model: CFModel, speeds: Speeds = SPEEDS) -> float:
    """Share of the equilibria at ``speeds`` with ``m < 0``; ``nan`` when no speed has one."""
    m = equilibrium_margin(model, speeds)
    found = torch.isfinite(m)
    return float((m[found] < 0).double().mean()) if bool(found.any()) else float("nan")


def idm_partials(v: Speeds, theta: Sequence[float] | Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Closed-form ``(f_s, f_dv, f_v)`` of the IDM (``delta = 4``) at its equilibrium ``(s_e(v), 0, v)``.

    ``theta = (v0, T, s0, a, b)``. With ``z = 1 - (v / v0)^4`` the equilibrium gap is
    ``s_e = (s0 + v T) / sqrt(z)``, so that ``s*(v, 0) / s_e = sqrt(z)``, and

        f_s  =  2 a (s*)^2 / s_e^3                              =  2 a z / s_e
        f_dv = -2 a s* / s_e^2 * v / (2 sqrt(a b))              = -v sqrt(a / b) sqrt(z) / s_e
        f_v  = -4 a v^3 / v0^4 - 2 a s* / s_e^2 * T             = -4 a v^3 / v0^4 - 2 a T sqrt(z) / s_e

    (``ds*/dv = T`` at ``dv = 0``). ``nan`` for ``v >= v0``, where there is no equilibrium.
    """
    v = torch.as_tensor(v, dtype=torch.float64)
    v0, T, s0, a, b = torch.as_tensor(theta, dtype=torch.float64, device=v.device).unbind(-1)
    z = 1.0 - (v / v0) ** 4
    root = torch.sqrt(torch.where(z > 0, z, torch.full_like(z, float("nan"))))
    s_e = (s0 + v * T) / root
    f_s = 2.0 * a * root**2 / s_e
    f_dv = -v * torch.sqrt(a / b) * root / s_e
    f_v = -4.0 * a * v**3 / v0**4 - 2.0 * a * T * root / s_e
    return f_s, f_dv, f_v
