"""Analytic string stability of memoryless models: equilibria, partial derivatives, margin, penalty."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from trafficloop.models.base import CFModel
from trafficloop.models.mechanistic import IDM, Gipps, idm_acc
from trafficloop.models.network import mlp
from trafficloop.stability.analytic import (
    SPEEDS,
    acceleration,
    equilibrium_gap,
    equilibrium_margin,
    idm_partials,
    partials,
    stability_penalty,
    string_margin,
    unstable_fraction,
)

F64 = torch.float64
V = torch.tensor(SPEEDS, dtype=F64)
STABLE = (33.3, 1.5, 2.0, 2.0, 1.5)  # a = 2 m/s^2: string stable at every speed
UNSTABLE = (33.3, 1.0, 2.0, 0.5, 1.5)  # small time gap and small a: unstable below 25.6 m/s


def closed_margin(v, theta) -> torch.Tensor:
    return string_margin(*idm_partials(torch.as_tensor(v, dtype=F64), theta))


def penalty_value(model, speeds, margin: float = 0.0) -> float:
    return float(stability_penalty(model, speeds, margin).detach())


def closed_sign_changes(theta) -> list[float]:
    """Speeds in (0, v0) where the closed-form margin of the IDM changes sign."""
    grid = np.linspace(0.01, theta[0] - 0.01, 4001)
    m = closed_margin(grid, theta).numpy()
    f = lambda v: float(closed_margin([v], theta)[0])  # noqa: E731
    return [brentq(f, grid[i], grid[i + 1], xtol=1e-12) for i in np.flatnonzero(np.diff(np.sign(m)) != 0)]


class Braking(CFModel):
    """Brakes at 1 m/s^2 whatever the state: memoryless, without equilibrium, not reading its input."""

    name = "braking"
    memoryless = True

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return torch.full(hist.shape[:-2], -1.0, dtype=hist.dtype)


def test_the_helper_evaluates_any_memoryless_model_through_its_interface():
    s, dv, v = torch.tensor([10.0, 30.0], dtype=F64), torch.tensor([[1.0], [-2.0]], dtype=F64), 15.0
    out = acceleration(IDM(UNSTABLE), s, dv, v)
    theta = torch.tensor(UNSTABLE, dtype=F64)
    assert out.shape == (2, 2) and out.dtype == F64
    assert torch.equal(out, idm_acc(s, dv, torch.full_like(s, v), *theta.unbind()))
    with pytest.raises(ValueError, match="memoryless"):
        acceleration(Gipps(), 10.0, 0.0, 15.0)
    braking = Braking()
    assert all(torch.equal(g, torch.zeros(3, dtype=F64)) for g in partials(braking, [10.0, 20.0, 30.0], 0.0, 15.0))
    assert bool(torch.isnan(equilibrium_gap(braking, V)).all()) and math.isnan(unstable_fraction(braking, V))
    assert penalty_value(braking, V) == 0.0


@pytest.mark.parametrize("theta", [IDM.default, STABLE, UNSTABLE, (40.0, 2.5, 5.0, 0.8, 3.0)])
def test_equilibrium_gap_equals_the_idm_formula(theta):
    v0, T, s0, _, _ = theta
    formula = (s0 + V * T) / torch.sqrt(1.0 - (V / v0) ** 4)
    gap = equilibrium_gap(IDM(theta), V)
    torch.testing.assert_close(gap, formula, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("theta", [IDM.default, STABLE, UNSTABLE])
def test_autograd_partials_equal_the_closed_form_of_the_idm(theta):
    model = IDM(theta)
    auto = partials(model, equilibrium_gap(model, V), 0.0, V)
    for got, expected in zip(auto, idm_partials(V, theta)):
        assert not got.requires_grad
        assert float(((got - expected) / expected).abs().max()) < 1e-6
    closed = closed_margin(V, theta)
    assert float(((string_margin(*auto) - closed) / closed).abs().max()) < 1e-6
    torch.testing.assert_close(equilibrium_margin(model, V), string_margin(*auto))


def test_the_margin_changes_sign_where_the_closed_form_does():
    low, high = closed_sign_changes(IDM.default)  # the IDM defaults are string unstable in between
    assert low == pytest.approx(0.9083, abs=1e-4) and high == pytest.approx(18.6138, abs=1e-4)
    model = IDM()
    inside = (V > low) & (V < high)
    assert int(inside.sum()) == 14  # 5 .. 18 m/s
    assert unstable_fraction(model, V) == pytest.approx(float(inside.double().mean()), abs=1e-15)
    m = equilibrium_margin(model, [low - 0.01, low + 0.01, high - 0.01, high + 0.01])
    assert m[0] > 0 > m[1] and m[2] < 0 < m[3]
    assert unstable_fraction(model, torch.linspace(low + 0.01, high - 0.01, 12)) == 1.0
    assert unstable_fraction(model, torch.linspace(high + 0.01, 30.0, 12)) == 0.0
    (edge,) = closed_sign_changes(UNSTABLE)
    assert unstable_fraction(IDM(UNSTABLE), V) == pytest.approx(float((V < edge).double().mean()), abs=1e-15)


def test_speeds_without_equilibrium_give_nan_and_are_left_out():
    theta = (20.0, 1.0, 2.0, 0.5, 1.5)  # no equilibrium from v0 = 20 m/s on
    model = IDM(theta)
    below, above = V[V < 20.0], V[V >= 20.0]
    gap = equilibrium_gap(model, V)
    assert bool(torch.isnan(gap[V >= 20.0]).all()) and bool(torch.isfinite(gap[V < 20.0]).all())
    assert bool(torch.isnan(equilibrium_margin(model, V)[V >= 20.0]).all())
    fraction = unstable_fraction(model, V)
    assert fraction == unstable_fraction(model, below) and 0.0 < fraction < 1.0
    expected = float(torch.relu(-closed_margin(below, theta)).mean())
    assert penalty_value(model, V) == pytest.approx(expected, rel=1e-6)
    assert math.isnan(unstable_fraction(model, above)) and penalty_value(model, above) == 0.0


def test_the_penalty_is_zero_when_stable_and_positive_otherwise():
    assert float(closed_margin(V, STABLE).min()) > 0.1
    assert penalty_value(IDM(STABLE), V) == 0.0
    assert penalty_value(IDM(STABLE), V, margin=0.2) > 0.0  # a margin moves the threshold
    penalty = penalty_value(IDM(UNSTABLE), V)
    # the IDM drives rationally (f_s > 0, f_dv < 0, f_v < 0): only the margin term is left
    assert penalty > 0.0 and penalty == pytest.approx(float(torch.relu(-closed_margin(V, UNSTABLE)).mean()), rel=1e-6)


class ResidualIDM(CFModel):
    """Unstable IDM plus a small network: memoryless, with an equilibrium at every speed."""

    name = "residual_idm"
    memoryless = True

    def __init__(self) -> None:
        super().__init__()
        self.net = mlp([3, 16, 1]).double()
        self.register_buffer("theta", torch.tensor(UNSTABLE, dtype=F64))

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        s, dv, v = hist[..., -1, :3].unbind(-1)
        x = torch.stack([s / 50.0, dv, v / 30.0], dim=-1)
        return idm_acc(s, dv, v, *self.theta.unbind()) + 0.1 * self.net(x).squeeze(-1)


def test_the_penalty_has_gradients_with_respect_to_the_parameters_of_a_network():
    torch.manual_seed(0)
    model = ResidualIDM()
    assert not equilibrium_gap(model, V).requires_grad  # the bisection is not differentiated
    penalty = stability_penalty(model, V)
    assert penalty.requires_grad and float(penalty.detach()) > 0.0
    penalty.backward()
    for name, p in model.net.named_parameters():
        if name == "2.bias":  # shifts the equilibrium gap only, which is held fixed: no gradient
            assert p.grad is None
        else:
            assert bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0.0
    with torch.no_grad():
        for p in model.net.parameters():
            if p.grad is not None:
                p -= 0.05 * p.grad
    assert penalty_value(model, V) < float(penalty.detach())  # the gradient points downhill
