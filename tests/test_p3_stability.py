"""P3 string stability: equilibria, step responses, gains and verdicts (E5 sanity of ``docs/protocols.md``).

Reference for the IDM: the loop linearised at the equilibrium, with ``e, y, u`` the deviations of
the gap, the follower speed and the leader speed:

    a[k] = f_s e[k] + f_dv (y[k] - u[k]) + f_v y[k],   y[k+1] = y[k] + dt a[k],
    e[k+1] = e[k] + dt (u[k+1] - y[k+1])                (the integrator of the benchmark)

Its speed transfer function is ``H(z) = (dt^2 f_s z - dt f_dv (z - 1)) / ((z - 1)^2 - dt (f_dv + f_v)
(z - 1) + dt^2 f_s z)``. As ``dt -> 0`` this tends to the continuous closed form of the contract,
``|H(i w)|^2 = (f_s^2 + f_dv^2 w^2) / ((f_s - w^2)^2 + (f_dv + f_v)^2 w^2)``. The two differ by
more than 1 % from 0.043 Hz on (lowest over the three parameter sets) and by 2-7.6 % at 0.5 Hz.
The numerical gain is therefore compared with the discrete-time equivalent up to 0.5 Hz,
truncated after ``response`` like the estimator of P3. It is compared with the continuous form
up to 0.04 Hz.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import brentq
from scipy.signal import lfilter

from trafficloop.data.schema import DT
from trafficloop.models.base import CFModel
from trafficloop.models.mechanistic import IDM
from trafficloop.models.persistence import PersistenceBaseline, ZeroAcceleration
from trafficloop.protocols.p3 import (
    DELTA,
    DELTA_LARGE,
    FREQUENCIES,
    PRACTICAL_TOL,
    TABLE_FREQUENCIES,
    TOL,
    FrequencyResponse,
    evaluate_p3,
    frequency_response,
    step_events,
    summarise,
)
from trafficloop.protocols.scenarios import initial_gap
from trafficloop.stability.analytic import SPEEDS, idm_partials, string_margin

F64 = torch.float64
SUMMARY_KEYS = {
    "unstable_fraction",
    "unstable_fraction_1e3",
    "unstable_fraction_large",
    "unstable_fraction_analytic",
    "max_gain",
    "max_gain_large",
    "n_collided_large",
    "peak_frequency",
    "n_equilibria",
    "n_converged",
    "n_tracks",
    "n_speeds",
    "agreement",
    "tracks_share",
    "n_invalid",
    "tol",
}
GAIN_COLUMNS = [f"gain_{f:g}" for f in TABLE_FREQUENCIES]
VERDICTS = ["unstable_numerical", "unstable_1e3", "unstable_large", "unstable_analytic"]
FLAGS = ["tracks", "tracks_large", "collided_large", *VERDICTS]  # nullable booleans
MASKED_FLAGS = FLAGS[1:]  # <NA> where the hold did not converge (``tracks`` is defined everywhere)
# the columns of the contract, with the analytic equilibrium gap, the practical band and the
# finite-amplitude pass added
COLUMNS = ["v_e", "s_e", "s_e_analytic", "converged", "tracks", "max_gain", "peak_frequency", *GAIN_COLUMNS]
COLUMNS += ["margin", "f_s", "f_dv", "f_v", "unstable_numerical", "unstable_1e3", "max_gain_large", "unstable_large"]
COLUMNS += ["tracks_large", "collided_large", "unstable_analytic"]

# The E5 parameter sets (v0, T, s0, a, b) with the boundaries the test states (computed from the
# closed forms below, checked by the test). All verdicts are "unstable below" their boundary.
# * analytic: the margin m = f_v^2 + 2 f_v f_dv - 2 f_s changes sign (idm_partials, string_margin);
# * numerical(tol): the largest discrete-time gain over FREQUENCIES crosses 1 + tol.
# The default tol = 1e-5 lies above the noise floor at the step of 0.05 m/s: the largest stable
# gain is 1 + 6e-11 in float64 and 1 + 2.4e-6 for a float32 IDM. It is below the smallest excess
# on the grid, so every speed agrees. Near the boundary the gain exceeds 1 by only about
# m^2 / (8 f_s^2): for the defaults the excess is 8.5e-4 at 17 m/s and 1.1e-4 at 18 m/s, and
# for the second set 1.4e-4 at 15 m/s. In the practical band (1e-3) those speeds are stable,
# so the band agrees with the analytic verdict at 24/26 and 17/18 of the equilibria.
assert TOL == 1e-5 and PRACTICAL_TOL == 1e-3 and DELTA == 0.05 and DELTA_LARGE == 1.0


@dataclass(frozen=True)
class ParameterSet:
    theta: tuple[float, ...]
    analytic: float | None  # boundary of the analytic verdict [m/s]
    numerical: float | None  # boundary of the numerical verdict at the default tol = 1e-5
    practical: float | None  # boundary of the practical band, max_gain > 1 + 1e-3
    agreement_1e3: float  # share of the equilibria where the band and the analytic verdict agree
    not_settled: tuple[float, ...]  # speeds whose G(0) still differs from 1 by 0.5 % or more after 300 s
    # speeds whose follower ends more than 5 % of the step away from the leader, small and large step
    not_tracking: tuple[float, ...]
    not_tracking_large: tuple[float, ...]
    # finite-amplitude pass (1 m/s, measured): unstable_fraction_large and max_gain_large
    fraction_large: float
    max_gain_large: float


SETS = {
    # gains of at most 1.007; the margin also changes sign at 0.908 m/s, below the grid
    "default": ParameterSet(
        (33.3, 1.5, 2.0, 1.0, 1.5), 18.6138, 18.3204, 16.8277, 24 / 26, (), (), (), 13 / 26, 1.00803
    ),
    # gain up to 1.09; nothing converges from v0 = 22.4 m/s on (18 converged speeds). At 22 m/s
    # the slow mode (136 s) leaves the follower 12 % short of the step after 300 s: 17 equilibria.
    "unstable": ParameterSet(
        (22.4, 0.95, 3.49, 1.24, 2.48), 15.2218, 15.1082, 14.7217, 16 / 17, (22.0,), (22.0,), (22.0,), 10 / 17, 1.08843
    ),
    # string stable below v0 = 23.9 m/s (19 equilibria). From 23 m/s, the 1 m/s step up leads to
    # 24 m/s, above v0, which the follower cannot reach.
    "stable": ParameterSet((23.9, 1.43, 2.05, 1.05, 0.53), None, None, None, 1.0, (23.0,), (), (23.0,), 0.0, 1.0),
}
#: band of the comparison with the continuous closed form [Hz]
F_CONTINUOUS = 0.04


# ---------------------------------------------------------------------------- closed forms
def partials_of(theta, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(x.numpy() for x in idm_partials(torch.as_tensor(np.atleast_1d(v), dtype=F64), theta))


def continuous_gain(f, f_s, f_dv, f_v) -> np.ndarray:
    """``|H(i w)|`` of the contract."""
    w2 = (2.0 * np.pi * np.asarray(f)) ** 2
    return np.sqrt((f_s**2 + f_dv**2 * w2) / ((f_s - w2) ** 2 + (f_dv + f_v) ** 2 * w2))


def discrete_filter(f_s, f_dv, f_v, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """``H(z)`` of the module docstring as coefficients ``(b, a)`` in powers of ``z^-1``."""
    b = np.array([0.0, dt * dt * f_s - dt * f_dv, dt * f_dv])
    a = np.array([1.0, -2.0 - dt * (f_dv + f_v) + dt * dt * f_s, 1.0 + dt * (f_dv + f_v)])
    return b, a


def discrete_gain(f, f_s, f_dv, f_v) -> np.ndarray:
    """``|H(exp(2 pi i f dt))|``, the discrete-time equivalent of the closed form."""
    b, a = discrete_filter(f_s, f_dv, f_v)
    z1 = np.exp(-2j * np.pi * np.asarray(f) * DT)
    return np.abs(np.polyval(b[::-1], z1) / np.polyval(a[::-1], z1))


def truncated_gain(f, f_s, f_dv, f_v, n: int) -> np.ndarray:
    """The estimator of P3 applied to the discrete-time equivalent: DFT of its first ``n`` impulse samples."""
    impulse = np.zeros(n)
    impulse[0] = 1.0
    h = lfilter(*discrete_filter(f_s, f_dv, f_v), impulse)
    return np.abs(np.exp(-2j * np.pi * DT * np.outer(f, np.arange(n))) @ h)


def margin(theta, v) -> np.ndarray:
    return string_margin(*(torch.as_tensor(x) for x in partials_of(theta, v))).numpy()


def max_discrete_gain(theta, v) -> np.ndarray:
    return np.array([discrete_gain(FREQUENCIES, *p).max() for p in zip(*partials_of(theta, v))])


def sign_changes(fun, low: float, high: float, n: int = 501) -> list[float]:
    grid = np.linspace(low, high, n)
    values = fun(grid)
    scalar = lambda x: float(fun(np.array([x]))[0])  # noqa: E731
    return [brentq(scalar, grid[i], grid[i + 1], xtol=1e-10) for i in np.flatnonzero(np.diff(np.sign(values)) != 0)]


# ---------------------------------------------------------------------------- the three IDM runs
@dataclass
class Run:
    model: IDM
    response: FrequencyResponse
    summary: dict
    table: pd.DataFrame
    seconds: float


@pytest.fixture(scope="module")
def idm_runs() -> dict[str, Run]:
    """P3 (= summarise o frequency_response) of every parameter set on the full speed grid, defaults."""
    runs = {}
    for name, spec in SETS.items():
        model = IDM(spec.theta)
        start = time.perf_counter()
        response = frequency_response(model, SPEEDS, device="cpu")
        summary, tables = summarise(model, response)
        runs[name] = Run(model, response, summary, tables["speeds"], time.perf_counter() - start)
        print(f"P3 of the IDM {spec.theta} on the full grid of {len(SPEEDS)} speeds: {runs[name].seconds:.2f} s")
    return runs


def test_the_step_scenarios_hold_then_step_after_the_manoeuvre_start():
    events = step_events([10.0, 20.0], [25.0, 40.0])  # the small step, hold 120 s, response 300 s
    assert [ev.event_id for ev in events] == ["p3_v10_up0.05", "p3_v20_up0.05", "p3_v10_down0.05", "p3_v20_down0.05"]
    for ev, v_e, sign, s0 in zip(events, (10.0, 20.0, 10.0, 20.0), (1, 1, -1, -1), (25.0, 40.0, 25.0, 40.0)):
        k_s = ev.meta["manoeuvre_start"]
        assert k_s == 1200 and len(ev) == k_s + 3000 + 1 and ev.meta["v_e"] == v_e
        assert np.all(ev.v_lead[: k_s + 1] == v_e) and np.all(ev.v_lead[k_s + 1 :] == v_e + sign * DELTA)
        assert np.all(ev.s == s0) and np.all(ev.v == ev.v_lead)
    with pytest.raises(ValueError, match="hold"):  # less than 10 s of simulated hold
        frequency_response(IDM(), [10.0], hold=12.0)
    with pytest.raises(ValueError, match="non-negative"):  # the step of 1 m/s down from 0.5 m/s
        frequency_response(IDM(), [0.5])


def test_the_full_grid_of_the_idm_defaults(idm_runs):
    run = idm_runs["default"]
    summary, table = run.summary, run.table
    assert set(summary) == SUMMARY_KEYS and list(table.columns) == COLUMNS
    json.dumps(summary, allow_nan=False)
    assert summary["n_speeds"] == summary["n_equilibria"] == len(SPEEDS) == 26
    assert summary["n_converged"] == summary["n_tracks"] == 26
    assert summary["n_invalid"] == 0 and summary["tracks_share"] == 1.0
    # started at the analytic equilibrium gap, the follower stays there during the hold
    np.testing.assert_allclose(table.s_e, table.s_e_analytic, rtol=1e-10)
    # the gains exceed 1 by at most 0.007, around 0.02 Hz (contract)
    assert 1.006 < summary["max_gain"] < 1.007
    assert summary["peak_frequency"] == pytest.approx(0.02, abs=0.002)
    assert summary["tol"] == TOL
    assert summary["unstable_fraction"] == pytest.approx(14 / 26)  # 5 .. 18 m/s, like the analytic verdict
    assert summary["unstable_fraction_analytic"] == pytest.approx(14 / 26)
    assert summary["unstable_fraction_1e3"] == pytest.approx(12 / 26)  # 5 .. 16 m/s in the practical band
    # the follower reacts one step after the leader; G(0) = 1 for a follower that tracks the step
    assert np.all(run.response.impulse[:, 0] == 0.0)
    np.testing.assert_allclose(run.response.gain[:, 0], 1.0, atol=1e-4)


@pytest.mark.parametrize("name", list(SETS))
def test_e5_the_gains_match_the_closed_form(idm_runs, name):
    spec, run = SETS[name], idm_runs[name]
    r = run.response
    conv = r.converged
    f_s, f_dv, f_v = partials_of(spec.theta, r.speeds)
    # the partials of the protocol (autograd at the numerical s_e) are those of the closed form
    for column, expected in (("f_s", f_s), ("f_dv", f_dv), ("f_v", f_v)):
        np.testing.assert_allclose(run.table[column][conv], expected[conv], rtol=1e-6)
    np.testing.assert_allclose(run.table.margin[conv], margin(spec.theta, r.speeds)[conv], rtol=1e-6, atol=1e-9)

    band, low = FREQUENCIES <= 0.5, FREQUENCIES <= F_CONTINUOUS
    n = r.impulse.shape[1]
    settled = conv & (np.abs(r.gain[:, 0] - 1.0) < 0.005)
    assert tuple(r.speeds[conv & ~settled]) == spec.not_settled
    assert tuple(r.speeds[conv & ~r.tracks]) == spec.not_tracking
    # The discrete-time equivalent up to 0.5 Hz, at every speed with an equilibrium. Measured: at
    # most 4e-5 for the defaults, and 3.5e-3 for the second set at 22 m/s, 0.4 m/s below v0. There
    # the equilibrium gap is 75, 92 and 131 m at 21.8, 22 and 22.2 m/s, and the residual of the
    # finite step, which scales with delta^2, was 5 % at 0.2 m/s.
    for i in np.flatnonzero(conv):
        reference = truncated_gain(FREQUENCIES[band], f_s[i], f_dv[i], f_v[i], n)
        assert np.abs(r.gain[i, band] / reference - 1.0).max() < 0.01, r.speeds[i]
    # the continuous closed form up to 0.04 Hz, where the response settles (measured: <= 9.1e-3)
    for i in np.flatnonzero(settled):
        reference = continuous_gain(FREQUENCIES[low], f_s[i], f_dv[i], f_v[i])
        assert np.abs(r.gain[i, low] / reference - 1.0).max() < 0.01, r.speeds[i]


def stated(boundary: float | None) -> list[float]:
    return [] if boundary is None else [boundary]


def unstable_below(v: np.ndarray, boundary: float | None) -> np.ndarray:
    return np.zeros(len(v), dtype=bool) if boundary is None else v < boundary


def equilibria(table: pd.DataFrame) -> pd.DataFrame:
    """The rows the summary is taken over: converged, and the follower tracks the small step."""
    return table[table.converged & table.tracks.to_numpy(dtype=bool)]


@pytest.mark.parametrize("name", list(SETS))
def test_e5_the_verdicts_follow_the_stated_boundaries(idm_runs, name):
    spec, run = SETS[name], idm_runs[name]
    high = min(spec.theta[0], max(SPEEDS)) - 0.01
    assert sign_changes(lambda v: margin(spec.theta, v), 5.0, high) == pytest.approx(stated(spec.analytic), abs=1e-3)
    crossing = sign_changes(lambda v: max_discrete_gain(spec.theta, v) - 1.0 - TOL, 5.0, high)
    assert crossing == pytest.approx(stated(spec.numerical), abs=1e-3)

    conv = run.table[run.table.converged]
    v = conv.v_e.to_numpy()
    analytic = conv.unstable_analytic.to_numpy(dtype=bool)
    assert np.array_equal(analytic, unstable_below(v, spec.analytic))
    assert np.array_equal(conv.unstable_numerical.to_numpy(dtype=bool), unstable_below(v, spec.numerical))
    # E5 at the default tolerance: the verdicts agree at every equilibrium (more than 95 %)
    eq = equilibria(run.table)
    expected = eq.unstable_analytic.to_numpy(dtype=bool).mean()
    assert run.summary["tol"] == TOL
    assert run.summary["agreement"] == 1.0 > 0.95
    assert run.summary["unstable_fraction"] == pytest.approx(expected)
    assert run.summary["unstable_fraction_analytic"] == pytest.approx(expected)


@pytest.mark.parametrize("name", list(SETS))
def test_the_practical_band_differs_next_to_the_boundary(idm_runs, name):
    spec, run = SETS[name], idm_runs[name]
    high = min(spec.theta[0], max(SPEEDS)) - 0.01
    crossing = sign_changes(lambda v: max_discrete_gain(spec.theta, v) - 1.0 - PRACTICAL_TOL, 5.0, high)
    assert crossing == pytest.approx(stated(spec.practical), abs=1e-3)

    conv = run.table[run.table.converged]
    v = conv.v_e.to_numpy()
    assert np.array_equal(conv.unstable_1e3.to_numpy(dtype=bool), unstable_below(v, spec.practical))
    eq = equilibria(run.table)
    band = eq.unstable_1e3.to_numpy(dtype=bool)
    assert run.summary["unstable_fraction_1e3"] == pytest.approx(band.mean())
    # the speeds between the practical and the analytic boundary disagree (17, 18 and 15 m/s)
    analytic = eq.unstable_analytic.to_numpy(dtype=bool)
    assert np.mean(band == analytic) == pytest.approx(spec.agreement_1e3)
    # summarised with tol = 1e-3, the verdict is the band and the agreement is the band's
    loose, tables = summarise(run.model, run.response, tol=PRACTICAL_TOL)
    assert loose["tol"] == PRACTICAL_TOL and loose["agreement"] == pytest.approx(spec.agreement_1e3)
    assert loose["unstable_fraction"] == loose["unstable_fraction_1e3"] == run.summary["unstable_fraction_1e3"]
    assert tables["speeds"].unstable_numerical.equals(run.table.unstable_1e3)


@pytest.mark.parametrize("name", list(SETS))
def test_the_finite_amplitude_pass(idm_runs, name):
    spec, run = SETS[name], idm_runs[name]
    r, table = run.response, run.table
    assert r.delta_large == DELTA_LARGE and r.gain_large.shape == r.gain.shape
    conv = table.converged.to_numpy()
    large = table.max_gain_large.to_numpy()
    np.testing.assert_array_equal(large[conv], r.gain_large[conv].max(-1))
    assert np.isnan(large[~conv]).all() and table.unstable_large[~conv].isna().all()
    assert np.array_equal(table.unstable_large[conv].to_numpy(dtype=bool), large[conv] > 1.0 + PRACTICAL_TOL)
    assert run.summary["unstable_fraction_large"] == pytest.approx(spec.fraction_large)
    assert run.summary["max_gain_large"] == pytest.approx(spec.max_gain_large, abs=1e-5)
    # the IDM never collides after a step of 1 m/s; it tracks it except where v_e + 1 m/s is out of reach
    assert run.summary["n_collided_large"] == 0 and not table.collided_large[conv].any()
    assert tuple(table.v_e[conv & ~table.tracks_large.fillna(False).to_numpy()]) == spec.not_tracking_large
    assert np.array_equal(table.tracks_large[conv].to_numpy(dtype=bool), r.tracks_large[conv])
    # over the equilibria; the small step stays the reference of the practical band
    eq = equilibria(table)
    assert run.summary["unstable_fraction_large"] == pytest.approx(eq.unstable_large.to_numpy(dtype=bool).mean())
    assert run.summary["max_gain_large"] == pytest.approx(eq.max_gain_large.max())
    assert run.summary["unstable_fraction_1e3"] == pytest.approx(eq.unstable_1e3.to_numpy(dtype=bool).mean())


def test_the_finite_amplitude_pass_can_be_switched_off():
    summary, tables = evaluate_p3(IDM(), (10.0,), device="cpu", delta_large=None)
    assert summary["max_gain_large"] is None and summary["unstable_fraction_large"] is None
    assert summary["n_collided_large"] is None and tables["speeds"].max_gain_large.isna().all()
    assert all(tables["speeds"][c].isna().all() for c in ("unstable_large", "tracks_large", "collided_large"))
    assert summary["n_equilibria"] == 1 and summary["unstable_fraction"] == 1.0


def test_the_stable_set_has_no_unstable_speed(idm_runs):
    summary, table = idm_runs["stable"].summary, idm_runs["stable"].table
    assert summary["unstable_fraction"] == 0.0 and summary["unstable_fraction_analytic"] == 0.0
    assert summary["unstable_fraction_1e3"] == summary["unstable_fraction_large"] == 0.0
    assert summary["agreement"] == 1.0 and summary["n_equilibria"] == 19
    assert (table.margin[table.converged] > 0.0).all()
    assert summary["max_gain"] <= 1.0 + 1e-9  # the largest gain is G(0) of a follower that tracks


@pytest.mark.parametrize("name", ["unstable", "stable"])
def test_speeds_from_v0_on_have_no_equilibrium(idm_runs, name, tmp_path):
    spec, run = SETS[name], idm_runs[name]
    table = run.table
    below = (table.v_e < spec.theta[0]).to_numpy()
    assert np.array_equal(table.converged.to_numpy(), below)
    assert run.summary["n_converged"] == int(below.sum()) < len(SPEEDS)  # 18 and 19 of 26
    # an equilibrium also tracks the small step: 17 and 19
    assert run.summary["n_equilibria"] == run.summary["n_tracks"] == int(below.sum()) - len(spec.not_tracking)
    above = table[~below]
    numeric = ["s_e", "s_e_analytic", "max_gain", "peak_frequency", *GAIN_COLUMNS, "margin", "f_s", "f_dv", "f_v"]
    numeric += ["max_gain_large"]
    assert above[numeric].isna().all().all()
    assert all(above[c].isna().all() for c in MASKED_FLAGS) and not above.tracks.any()
    # the fractions are over the equilibria, the share of tracking over the converged speeds
    eq = equilibria(table)
    assert run.summary["unstable_fraction"] == pytest.approx(eq.unstable_numerical.to_numpy(dtype=bool).mean())
    assert run.summary["tracks_share"] == pytest.approx(table.tracks[below].to_numpy(dtype=bool).mean())
    # what the caller writes: JSON summary, parquet table
    json.dumps(run.summary, allow_nan=False)
    table.to_parquet(tmp_path / "p3.speeds.parquet", index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "p3.speeds.parquet"), table)


@pytest.mark.parametrize("model_class", [PersistenceBaseline, ZeroAcceleration])
def test_the_baselines_never_track_a_step_and_are_reported(model_class):
    speeds = (5.0, 17.0, 30.0)
    summary, tables = evaluate_p3(model_class(), speeds, device="cpu")
    table = tables["speeds"]
    assert set(summary) == SUMMARY_KEYS and list(table.columns) == COLUMNS
    json.dumps(summary, allow_nan=False)
    # a constant-speed follower keeps its starting gap and its speed through the hold and after the step
    assert table.converged.all() and summary["n_converged"] == summary["n_speeds"] == 3
    np.testing.assert_allclose(table.s_e, [initial_gap(v) for v in speeds], rtol=1e-9)
    assert not table.tracks.any() and not table.tracks_large.any() and summary["tracks_share"] == 0.0
    # it never follows the leader: no equilibrium, so no fraction and no gain (not "stable")
    assert summary["n_tracks"] == summary["n_equilibria"] == 0 and summary["n_invalid"] == 0
    for key in ("unstable_fraction", "unstable_fraction_1e3", "unstable_fraction_large", "max_gain", "max_gain_large"):
        assert summary[key] is None, key
    assert summary["peak_frequency"] is None and summary["tol"] == TOL
    # the table keeps every converged speed: G = 0 in both passes (G(0) = 0)
    assert (table[["max_gain", *GAIN_COLUMNS, "max_gain_large"]] == 0.0).all().all()
    # 1 m/s slower, the leader is caught up within the starting gap divided by 1 m/s (< 300 s)
    assert table.collided_large.all() and summary["n_collided_large"] == 3
    # not memoryless: no analytic part
    assert summary["unstable_fraction_analytic"] is None and summary["agreement"] is None
    assert table[["s_e_analytic", "margin", "f_s", "f_dv", "f_v"]].isna().all().all()
    assert table.unstable_analytic.isna().all()


class Float32IDM(IDM):
    """The IDM computed in float32 inside the float64 rollout, like a network."""

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return super().forward(hist.float()).to(hist.dtype)


def test_a_model_computing_in_float32_gives_the_same_verdicts(idm_runs):
    speeds = (5.0, 17.0, 18.0, 19.0, 25.0)
    model = Float32IDM()
    response = frequency_response(model, speeds)
    reference = idm_runs["default"]
    rows = np.searchsorted(reference.response.speeds, speeds)
    assert response.converged.all() and response.n_invalid == 0
    # At the step of 0.05 m/s, the float32 noise of the gain (measured: <= 2.2e-6) stays far below
    # the excess of the unstable speeds (>= 1.1e-4) and below half the tolerance at the stable ones.
    np.testing.assert_allclose(response.gain.max(-1), reference.response.gain[rows].max(-1), rtol=0.0, atol=1e-5)
    assert response.gain[np.array(speeds) > 18.6].max() < 1.0 + TOL / 2
    for tol in (TOL, PRACTICAL_TOL):
        table = summarise(model, response, tol)[1]["speeds"]
        expected = summarise(reference.model, reference.response, tol)[1]["speeds"].iloc[rows].reset_index(drop=True)
        for column in FLAGS:
            assert table[column].equals(expected[column]), column


class Braking(CFModel):
    """Memoryless, brakes at 1 m/s^2 whatever the state: no equilibrium at any speed."""

    name = "braking"
    memoryless = True

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return torch.full(hist.shape[:-2], -1.0, dtype=hist.dtype, device=hist.device)


def test_a_model_without_any_equilibrium_is_reported_with_nulls():
    summary, tables = evaluate_p3(Braking(), (5.0, 30.0), device="cpu")
    json.dumps(summary, allow_nan=False)
    counts = {"n_equilibria", "n_converged", "n_tracks", "n_invalid"}
    assert all(summary[key] == 0 for key in counts) and summary["n_speeds"] == 2 and summary["tol"] == TOL
    for key in SUMMARY_KEYS - counts - {"n_speeds", "tol"}:
        assert summary[key] is None, key
    table = tables["speeds"]
    assert not table.converged.any() and not table.tracks.any()
    assert table.drop(columns=["v_e", "converged", "tracks"]).isna().all().all()


def test_two_calls_are_identical_and_a_speed_does_not_depend_on_the_batch(idm_runs):
    first = evaluate_p3(IDM(), speeds=(20.0, 10.0), device="cpu")
    second = evaluate_p3(IDM(), speeds=(20.0, 10.0), device="cpu")
    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[1]["speeds"], second[1]["speeds"], check_exact=True)
    full = idm_runs["default"].table.set_index("v_e")  # the same model on the full grid
    pd.testing.assert_frame_equal(first[1]["speeds"].set_index("v_e"), full.loc[[20.0, 10.0]], check_exact=True)
