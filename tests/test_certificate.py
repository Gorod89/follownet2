"""A priori certificate of the certified residual IDM (port of cf-stability, decisions D71, D72, D86): exact box
minimum, agreement with cf-stability, monotone budget, enforcement that survives training, cores without budget,
the margin term of a stable core."""

from __future__ import annotations

import json
import math

import pytest
import torch

from trafficloop.calibration.calibrate import CalibrationConfig, calibrate_global
from trafficloop.data.loaders.synthetic import IDM_NOMINAL, SyntheticConfig, make_events
from trafficloop.models import registry
from trafficloop.models.mechanistic import IDM, idm_acc
from trafficloop.stability import certificate
from trafficloop.stability.analytic import SPEEDS, equilibrium_gap, idm_partials, string_margin

# global IDM fits of cf-stability on FollowNet HighD, fold 0 (its tests): free, and with a margin of 0.2 / 0.5
CORE_FREE = (29.57, 0.676, 2.015, 0.351, 0.5)
CORE_M02 = (35.2, 0.98, 3.19, 1.57, 0.5)
CORE_M05 = (35.1, 0.94, 4.71, 2.74, 0.5)
V = torch.tensor([5.0, 12.0, 20.0, 29.0], dtype=torch.float64)
UNIT = 1.0 / torch.tensor([20.0, 2.0, 10.0], dtype=torch.float64)  # bounds per unit of r_max * lipschitz, norms 1
N_SCAN = 200


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def model(core, r_max: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    m = registry.create("residual_idm_certified", r_max=r_max)
    m.set_core(core)
    return m


def random_cases(n: int, seed: int, stationary_inside: bool) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)

    def u(lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(n, generator=gen, dtype=torch.float64)

    f_s, f_dv, bounds = u(-0.1, 0.5), u(-1.0, 0.2), torch.stack((u(0.0, 0.2), u(0.0, 0.3), u(0.2, 0.5)), dim=-1)
    f_v = -f_dv + u(-0.1, 0.1) if stationary_inside else u(-1.0, 0.2)
    return f_s, f_dv, f_v, bounds


def brute_force(f_s: float, f_dv: float, f_v: float, b: torch.Tensor, n: int = 801) -> float:
    d_s = torch.tensor([-b[0], 0.0, b[0]], dtype=torch.float64)
    d_dv = torch.linspace(-b[1], b[1], n, dtype=torch.float64)
    d_v = torch.linspace(-b[2], b[2], n, dtype=torch.float64)
    p, q, s = f_v + d_v[:, None, None], f_dv + d_dv[None, :, None], f_s + d_s[None, None, :]
    return (p**2 + 2.0 * p * q - 2.0 * s).min().item()


def regression_training(m, steps: int) -> None:
    """Train the residual on other data: random accelerations on random states (Adam, training mode)."""
    gen = torch.Generator().manual_seed(7)
    x = torch.tensor([2.0, -4.0, 0.0]) + torch.tensor([80.0, 8.0, 35.0]) * torch.rand(512, 3, generator=gen)
    hist = torch.cat([x, torch.zeros(512, 1)], dim=-1).unsqueeze(1)
    target = torch.sin(x[:, 0] / 7.0) + 0.5 * torch.cos(x[:, 2])
    opt = torch.optim.Adam(m.g.parameters(), lr=1e-2)
    m.train()
    for _ in range(steps):
        loss = torch.mean((m.residual(hist) - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()


# ------------------------------------------------------------------- the box minimum
@pytest.mark.parametrize("stationary_inside", [False, True])
def test_guaranteed_margin_is_the_minimum_over_the_box(stationary_inside):
    f_s, f_dv, f_v, bounds = random_cases(15, int(stationary_inside), stationary_inside)
    exact = certificate.guaranteed_margin(f_s, f_dv, f_v, bounds)
    for k in range(len(f_s)):
        brute = brute_force(f_s[k].item(), f_dv[k].item(), f_v[k].item(), bounds[k])
        assert exact[k].item() <= brute + 1e-12 and brute - exact[k].item() < 1e-6
    zero = torch.zeros(3, dtype=torch.float64)
    plain = string_margin(f_s, f_dv, f_v)  # zero bounds: the criterion itself
    torch.testing.assert_close(certificate.guaranteed_margin(f_s, f_dv, f_v, zero), plain, rtol=0, atol=1e-15)


def test_a_vanishing_residual_leaves_the_margin_of_the_core():
    """With ``r_max -> 0`` the feasible spacings shrink to the equilibrium of the core and the bounds to zero."""
    v = torch.tensor(SPEEDS, dtype=torch.float64)
    theta = torch.tensor(CORE_M02, dtype=torch.float64)
    out = certificate.a_priori(theta, v, torch.full_like(v, 1e-9), torch.zeros(len(v), 3, dtype=torch.float64), N_SCAN)
    s_e = equilibrium_gap(IDM(CORE_M02), v)
    assert out["feasible"].all() and torch.allclose(out["s_worst"], s_e, rtol=1e-6)
    torch.testing.assert_close(out["guaranteed_margin"], certificate.core_margin(theta, v), rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(certificate.core_margin(theta, v), string_margin(*idm_partials(v, theta)))
    f = certificate.core_partials(theta, s_e, v)  # automatic differentiation of the benchmark IDM
    torch.testing.assert_close(torch.stack(f), torch.stack(idm_partials(v, theta)), rtol=1e-9, atol=1e-12)


def test_the_mathematics_is_that_of_cf_stability(monkeypatch):
    """Values of ``certify_core`` and of the budget of cf-stability on its spacing range [1, 200] m (computed with
    ``cf_stability.stability.certificate``, n_scan 200)."""
    monkeypatch.setattr(certificate, "S_MIN", 1.0)
    monkeypatch.setattr(certificate, "S_MAX", 200.0)
    theta = torch.tensor(CORE_M02, dtype=torch.float64)
    bounds = torch.tensor([0.005, 0.05, 0.01], dtype=torch.float64).expand(len(V), 3)
    out = certificate.a_priori(theta, V, torch.full_like(V, 0.3), bounds, N_SCAN)
    reference = {
        "guaranteed_margin": [0.012505543526337037, 0.06560101672807422, 0.08744561153800984, 0.09211345374929651],
        "s_low": [7.413986269077892, 13.77674849068144, 21.860338321623978, 36.98709409410067],
        "s_high": [8.997167371728025, 16.762761912274797, 27.148317794501633, 53.56750951563087],
        "s_worst": [8.997167371728025, 16.762761912274797, 27.148317794501633, 53.56750951563087],
    }
    for key, values in reference.items():
        torch.testing.assert_close(out[key], torch.tensor(values, dtype=torch.float64), rtol=1e-12, atol=0)
    assert out["holds"].all()
    core = torch.tensor(CORE_M05, dtype=torch.float64)

    def holds_at(beta):
        o = certificate.a_priori(core, V, torch.ones_like(V), beta[:, None] * UNIT, N_SCAN)
        return o["holds"] | ~o["feasible"]

    budget = certificate.largest(holds_at, len(V))
    expected = [0.12440311797681913, 0.3230053677113596, 0.46350666533912965, 0.23830394048049328]
    torch.testing.assert_close(budget, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0)
    ends = certificate.feasible_ends(CORE_FREE, V, 1.0)  # a = 0.351 < r_max: no upper end
    torch.testing.assert_close(ends["s_low"], torch.tensor([2.7501948009766948, 5.180148752999585, 8.142855445361201,
                                                            12.643107129785834], dtype=torch.float64), rtol=1e-12, atol=0)
    assert torch.isinf(ends["s_high"]).all() and not ends["bounded"].any() and not ends["within"].any()


class _Shifted(torch.nn.Module):
    """A memoryless law ``f_idm + r`` with a constant ``r``."""

    memoryless = True
    name = "shifted"

    def __init__(self, core: IDM, r: float) -> None:
        super().__init__()
        self.core, self.r = core, r

    def forward(self, hist):
        return self.core(hist) + self.r


def test_the_feasible_spacings_are_where_the_residual_can_cancel_the_core():
    theta = torch.tensor(CORE_M05, dtype=torch.float64)
    v = torch.tensor(SPEEDS, dtype=torch.float64)
    ends = certificate.feasible_ends(theta, v, 0.5)
    assert ends["bounded"].all() and ends["within"].all()
    acc = lambda s: idm_acc(s, torch.zeros_like(s), v, *theta.unbind())  # noqa: E731
    torch.testing.assert_close(acc(ends["s_low"]), torch.full_like(v, -0.5))
    torch.testing.assert_close(acc(ends["s_high"]), torch.full_like(v, 0.5))
    for r in (-0.5, 0.0, 0.5):  # a constant residual moves the equilibrium inside the interval
        s_e = equilibrium_gap(_Shifted(IDM(CORE_M05), r), v)
        assert ((s_e >= ends["s_low"] - 1e-6) & (s_e <= ends["s_high"] + 1e-6)).all()


# --------------------------------------------------------------------------- budget
def test_budget_with_r_max_kept_is_the_largest_that_holds():
    m = model(CORE_M02, r_max=0.5)
    m.lipschitz = 0.3
    budget = certificate.max_residual_budget(m, V, n_scan=N_SCAN, keep_r_max=True)
    assert (budget > 0).all()
    unit = certificate.residual_bounds(m) / (0.5 * 0.3)  # bounds per unit of r_max * lipschitz
    theta = m.theta.detach()
    for scale, expected in ((0.999, True), (1.001, False)):  # the feasible spacings stay those of r_max = 0.5
        out = certificate.a_priori(theta, V, torch.full_like(V, 0.5), (scale * budget)[:, None] * unit, N_SCAN)
        assert ((out["holds"] | ~out["feasible"]) == expected).all()
    m.lipschitz = 0.7  # the ratio r_max : lipschitz does not enter
    assert torch.equal(certificate.max_residual_budget(m, V, n_scan=N_SCAN, keep_r_max=True), budget)
    smaller = model(CORE_M02, r_max=0.3)  # a smaller amplitude admits fewer spacings: a larger budget
    larger = certificate.max_residual_budget(smaller, V, n_scan=N_SCAN, keep_r_max=True)
    assert (larger >= budget).all() and (larger > budget).any()


def test_enforced_budget_certifies_and_survives_training():
    m = model(CORE_M05)  # r_max = 1, lipschitz = 1
    assert not certificate.certificate_a_priori(m, n_scan=N_SCAN)["holds"].all()
    budget = certificate.enforce_budget(m, 0.9, n_scan=N_SCAN)
    assert budget["r_max"] == m.r_max == 1.0 and budget["safety"] == 0.9 and budget["admissible"] > 0
    assert m.lipschitz == budget["lipschitz"] == pytest.approx(0.9 * budget["admissible"], rel=1e-12)
    before = certificate.certificate_a_priori(m, n_scan=N_SCAN)
    assert before["holds"].all() and before["n_holds"] == len(SPEEDS)
    regression_training(m, 100)  # the certificate sees the weights through the layer norms only
    after = certificate.certificate_a_priori(m, n_scan=N_SCAN)
    assert after["holds"].all()
    assert math.prod(after["layer_norms"]) == pytest.approx(1.0, abs=0.06)  # spectral normalisation
    assert float(m.residual(torch.zeros(1, 1, 4) + 10.0).detach().abs()) > 1e-3  # the residual did change
    m.lipschitz = 1.5 * budget["admissible"] / m.r_max  # beyond the admissible budget it fails somewhere
    above = certificate.certificate_a_priori(m, n_scan=N_SCAN)
    assert not above["holds"].all()


def test_cores_without_a_budget_raise_and_the_margin_of_the_core_buys_budget():
    admissible = {}
    for name, core in (("m05", CORE_M05), ("m02", CORE_M02)):
        admissible[name] = certificate.enforce_budget(model(core, r_max=0.5), 0.9, n_scan=N_SCAN)["admissible"]
    assert admissible["m05"] > admissible["m02"] > 0
    for core in (CORE_M02, CORE_FREE):  # r_max = 1 admits spacings where these cores are string unstable
        m = model(core)
        with pytest.raises(ValueError, match="no admissible residual budget"):
            certificate.enforce_budget(m, 0.9, n_scan=N_SCAN)
        assert m.lipschitz == 1.0
    with pytest.raises(ValueError, match="safety"):
        certificate.enforce_budget(model(CORE_M05), 1.5)


def test_certificate_json_is_plain():
    m = model(CORE_M05, r_max=0.5)
    certificate.enforce_budget(m, 0.9, n_scan=N_SCAN)
    cert = certificate.to_json(certificate.certificate_a_priori(m, n_scan=N_SCAN))
    text = json.dumps(cert, allow_nan=False)
    assert '"holds"' in text and len(cert["v"]) == len(SPEEDS) and cert["n_holds"] == len(SPEEDS)


# ----------------------------------------------------------- core with a stability margin
def test_stability_shortfall_is_the_term_of_d72():
    cores = torch.tensor([CORE_M05, CORE_FREE, IDM_NOMINAL], dtype=torch.float32)
    term = certificate.stability_shortfall(cores, 0.2)
    assert term.dtype == torch.float32 and term.shape == (3,)
    v = torch.tensor(SPEEDS, dtype=torch.float64)
    for k, core in enumerate(cores.double()):
        m = torch.nan_to_num(certificate.core_margin(core, v), nan=-10.0)
        assert float(term[k]) == pytest.approx(10.0 * float(torch.clamp(0.2 - m, min=0.0).sum()), rel=1e-6)
    assert float(term[0]) == 0.0  # margin above 0.2 everywhere
    assert float(term[1]) > 10.0 * 10.0  # v0 < 30 m/s: one speed without equilibrium counts -10


def smallest_margin(fit: dict) -> float:
    """Smallest closed-form margin of a calibrated IDM over 5..30 m/s, -10 at a speed without equilibrium."""
    margin = certificate.core_margin([fit["params"][name] for name in IDM.param_names])
    return float(torch.nan_to_num(margin, nan=certificate.NO_EQUILIBRIUM).min())


def test_calibration_with_the_margin_term_gives_a_stable_core():
    """The IDM default of the synthetic drivers is string unstable from 0.9 to 18.6 m/s; with the term of D72 the
    pooled fit trades accuracy for a margin of 0.2 on 5..30 m/s."""
    events = make_events(SyntheticConfig(n_drivers=6, events_per_driver=1, theta_cv=0.0, duration=(30.0, 40.0), seed=4))
    cfg = CalibrationConfig(maxiter=60, device="cpu", dtype="float64", seed=0)
    free = calibrate_global(IDM, events, cfg)
    again = calibrate_global(IDM, events, cfg, penalty=None)
    assert again == free and "fit_objective" not in free  # no term: the calibration of the entries is unchanged
    stable = calibrate_global(IDM, events, cfg, penalty=lambda p: certificate.stability_shortfall(p, 0.2))
    assert smallest_margin(free) < 0.0 and smallest_margin(stable) >= 0.2 - 1e-3
    assert stable["fit_objective"] >= free["objective"] and stable["objective"] >= stable["fit_objective"]
