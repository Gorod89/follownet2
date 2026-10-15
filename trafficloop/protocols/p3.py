"""P3: string stability (``docs/protocols.md``, section 3).

The numerical frequency-response test applies to every model, and the analytic criterion of
:mod:`trafficloop.stability.analytic` to memoryless ones. Both are evaluated at the equilibrium
speeds ``v_e = 5, 6, ..., 30 m/s``.

For every speed and step size, two scenarios share a hold of ``hold = 120 s`` at ``v_e``, warm-up
included. At the step index ``k_s = meta["manoeuvre_start"]``, the leader speed changes to
``v_e + delta`` in the first scenario and to ``v_e - delta`` in the second (``v_lead[k_s + 1] -
v_lead[k_s] = +-delta``), then stays there for ``response = 300 s``. All scenarios run in one
batch through :func:`trafficloop.sim.rollout.rollout`, in float64 by default. The model
computes in its own dtype: the networks cast their inputs to float32 and their output back.

Steps. The small step ``delta = 0.05 m/s`` measures the small-signal transfer function, which
both the string-stability verdict and the analytic criterion are about. What a finite step adds
is nonlinearity, and it grows with ``delta^2``. For the IDM the deviation from the linearised
loop is 4e-5 at 0.05 m/s against 6.5e-4 at 0.2 m/s; 0.4 m/s below ``v0`` it is 0.35 % against
5 %. The NGSIM ``stability_mlp`` has positive analytic margins and gains within 1 + 1.2e-6 at
0.05 m/s, yet a step of 0.2 m/s grows by 0.3-0.6 %: that excess is finite-amplitude
nonlinearity, not string instability. The second step ``delta_large = 1.0 m/s`` (same hold,
both signs, same batch) measures this practical, finite-perturbation amplification on its
own: ``max_gain_large`` and ``unstable_large = max_gain_large > 1 + 1e-3``. ``collided_large``
flags a gap ``s <= 0`` after this step. A speed that collides keeps its ``max_gain_large``,
because a runaway is a finding, and the summary reports ``n_collided_large`` next to the
largest gain. In both passes the follower tracks the step (``tracks``, ``tracks_large``) when
its final speed is within 5 % of the step of the new leader speed, for both signs.

* Equilibrium: the follower starts at the analytic equilibrium gap of a memoryless model when
  that gap is finite, otherwise at ``initial_gap(v_e)``. A speed is converged when, over the
  last 10 s of the hold of every scenario of the speed, ``max |a| < 2e-3 m/s^2``,
  ``|v - v_e| < 0.01 m/s`` and ``0 < s < 300 m``. ``s_e`` is the mean gap over those samples.
  Up to the step, the scenarios are the constant-speed scenario of the contract (the leader
  input is identical), so the hold doubles as the equilibrium search. A speed is an
  *equilibrium* when it is converged and the follower also tracks the small step. Persistence
  and constant speed settle at every speed but never follow the leader (``G = 0``); counted
  as equilibria, they would rank as the most string-stable models. The fractions, the
  agreement and the largest gains of the summary are taken over the equilibria.
  ``n_converged``, ``n_tracks`` and ``n_equilibria`` count the three sets, and the table keeps
  every speed.
* Impulse response: ``h[k] = (v_f[k_s + k + 1] - v_f[k_s + k]) / (+-delta)``, averaged over
  the two signs. The even-order terms of the nonlinearity, and any residual drift left from
  the hold, are common to both scenarios and cancel.
* Gain: ``G(f) = |sum_k h[k] exp(-2 pi i f k dt)|`` on ``f = 0`` and 200 log-spaced
  frequencies from 0.002 to 2 Hz. For a linear follower, ``h`` is the impulse response of the
  speed transfer function of the closed loop under the integrator of the benchmark,
  truncated after ``response``. ``G`` is then the gain of a speed perturbation from one
  vehicle to the next in a homogeneous platoon, and string instability is ``G > 1``.
  ``G(0) = (v_f(end) - v_f(k_s)) / delta``: a follower that does not track the step (the
  persistence and constant-speed baselines) has ``G(0) = 0``.

The discrete-time loop is not the continuous one. For a memoryless follower linearised at
its equilibrium, the speed transfer function under the integrator is

    H(z) = (dt^2 f_s z - dt f_dv (z - 1)) / ((z - 1)^2 - dt (f_dv + f_v) (z - 1) + dt^2 f_s z),

which tends to the continuous ``(f_s - f_dv s) / (s^2 - (f_dv + f_v) s + f_s)`` as ``dt -> 0``.
To first order in ``dt``, its low-frequency margin is ``m - dt f_s f_v``. A rational follower
(``f_s > 0``, ``f_v < 0``) is therefore slightly more stable in discrete time, and its boundary
lies slightly below the analytic one.

Tolerances. The verdict ``unstable_numerical`` follows the mathematical criterion ``G > 1``
with ``tol = 1e-5``, so a speed is unstable when ``max_gain > 1 + 1e-5`` at the small step.
This lies above the noise of the float64 rollout at 0.05 m/s. A string-stable IDM reaches a
largest gain of 1 + 6e-11, and at most 1 + 2.4e-6 when the IDM computes in float32. The
float32 ``stability_mlp`` stays within 1.3e-6 of its float64 copy. The tolerance also lies
below the excess of the unstable IDM speeds next to a boundary (at least 1.1e-4). Near the
boundary the excess gain is only about ``m^2 / (8 f_s^2)``, so a loose tolerance moves the
boundary by up to about 2 m/s (1.8 m/s for the IDM defaults at 1e-3). The practical band
``max_gain > 1 + 1e-3``, at least 0.1 % growth per vehicle, is reported next to the verdict
(``unstable_1e3``, ``unstable_fraction_1e3``) instead of replacing it. The closed-form check
(E5) compares ``G`` with the discrete-time ``H(z)`` above, truncated after ``response`` like the
estimator, up to 0.5 Hz. The continuous closed form departs from ``H(z)`` by more than 1 %
above about 0.04 Hz, and by a few percent at 0.5 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from trafficloop.data.batch import WARMUP, EventBatch
from trafficloop.data.schema import DT, Event
from trafficloop.models.base import CFModel
from trafficloop.protocols.scenarios import initial_gap, scenario_event
from trafficloop.sim.rollout import rollout
from trafficloop.stability.analytic import SPEEDS, Speeds, equilibrium_gap, partials, string_margin

#: leader speed step of the small-signal test [m/s], in the linear regime of the follower
DELTA = 0.05
#: leader speed step of the finite-amplitude test [m/s], judged with PRACTICAL_TOL
DELTA_LARGE = 1.0
#: hold before the step [s], warm-up included
HOLD = 120.0
#: duration of the response after the step [s]
RESPONSE = 300.0
#: a speed is numerically unstable when its largest gain exceeds 1 + TOL (the criterion G > 1)
TOL = 1e-5
#: practical band reported next to it: at least 0.1 % growth of a speed perturbation per vehicle
PRACTICAL_TOL = 1e-3
#: end of the hold over which convergence is decided [s], and its thresholds
SETTLE = 10.0
A_SETTLED = 2e-3  # m/s^2, max |a|
V_SETTLED = 0.01  # m/s, max |v - v_e|
S_MAX = 300.0  # m, the gap stays inside (0, S_MAX)
#: the follower tracks a step when its final speed is within this share of the step of the leader's
TRACK_SHARE = 0.05
#: frequencies of the gain [Hz]: 0 and a log grid from 0.002 to 2 Hz
FREQUENCIES: np.ndarray = np.concatenate([[0.0], np.geomspace(0.002, 2.0, 200)])
#: frequencies reported as columns ``gain_<f>`` of the table [Hz]
TABLE_FREQUENCIES: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
COLUMNS = (
    "v_e",
    "s_e",
    "s_e_analytic",
    "converged",
    "tracks",
    "max_gain",
    "peak_frequency",
    *(f"gain_{f:g}" for f in TABLE_FREQUENCIES),
    "margin",
    "f_s",
    "f_dv",
    "f_v",
    "unstable_numerical",
    "unstable_1e3",
    "max_gain_large",
    "unstable_large",
    "tracks_large",
    "collided_large",
    "unstable_analytic",
)


def step_events(
    speeds: Sequence[float],
    s_init: Sequence[float],
    delta: float = DELTA,
    hold: float = HOLD,
    response: float = RESPONSE,
    dt: float = DT,
) -> list[Event]:
    """The step scenarios: first every speed stepping up by ``delta``, then every speed stepping down.

    The leader holds ``v_e`` for ``hold`` seconds, up to and including the step index
    ``k_s = meta["manoeuvre_start"]``. From ``k_s + 1`` on, it holds ``v_e +- delta`` for
    ``round(response / dt)`` samples. The follower starts at the gap ``s_init`` of its speed.
    """
    n_response = int(round(response / dt))
    events = []
    for sign, tag in ((1.0, "up"), (-1.0, "down")):
        for v_e, s0 in zip(speeds, s_init):
            v_lead = np.full(n_response + 1, float(v_e) + sign * delta)
            v_lead[0] = float(v_e)
            meta = {"protocol": "p3", "v_e": float(v_e), "delta": sign * delta}
            event_id = f"p3_v{float(v_e):g}_{tag}{delta:g}"
            events.append(scenario_event(v_lead, float(s0), event_id, hold, meta=meta, dt=dt))
    return events


def dft_gain(h: np.ndarray, frequencies: Sequence[float] | np.ndarray, dt: float = DT) -> np.ndarray:
    """``|sum_k h[..., k] exp(-2 pi i f k dt)|`` for every frequency: shape ``[..., n_frequencies]``.

    One matrix-vector product per row, so a row's gain does not depend on the other rows.
    """
    h = np.asarray(h, dtype=np.float64)
    phase = 2.0 * np.pi * dt * np.outer(np.asarray(frequencies, dtype=np.float64), np.arange(h.shape[-1]))
    cos, sin = np.cos(phase), np.sin(phase)
    rows = h.reshape(-1, h.shape[-1])
    gain = np.stack([np.hypot(cos @ row, sin @ row) for row in rows]) if len(rows) else np.zeros((0, len(phase)))
    return gain.reshape(*h.shape[:-1], len(phase))


@dataclass
class FrequencyResponse:
    """Numerical part of P3, one entry per speed (``[n]``) or per speed and frequency."""

    speeds: np.ndarray  # [n] v_e
    s_init: np.ndarray  # [n] starting gap
    s_e_analytic: np.ndarray  # [n] analytic equilibrium gap (memoryless models), nan otherwise
    converged: np.ndarray  # [n] bool
    s_e: np.ndarray  # [n] mean gap over the end of the hold, nan where not converged
    impulse: np.ndarray  # [n, K] averaged impulse response h
    frequencies: np.ndarray  # [F]
    gain: np.ndarray  # [n, F] G(f) on ``frequencies``
    table_gain: np.ndarray  # [n, len(TABLE_FREQUENCIES)]
    tracks: np.ndarray  # [n] bool, both signs end within TRACK_SHARE * delta of the new leader speed
    final_error: np.ndarray  # [n, 2] v_f(end) - v_lead(end) for the steps up and down
    n_invalid: int
    delta: float
    step_index: int
    # the finite-amplitude pass (None when it is switched off)
    delta_large: float | None = None
    impulse_large: np.ndarray | None = None  # [n, K]
    gain_large: np.ndarray | None = None  # [n, F]
    tracks_large: np.ndarray | None = None  # [n] bool, within TRACK_SHARE * delta_large
    final_error_large: np.ndarray | None = None  # [n, 2]
    collided_large: np.ndarray | None = None  # [n] bool, s <= 0 after the step, either sign


@torch.no_grad()
def frequency_response(
    model: CFModel,
    speeds: Speeds = SPEEDS,
    delta: float = DELTA,
    hold: float = HOLD,
    response: float = RESPONSE,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    delta_large: float | None = DELTA_LARGE,
) -> FrequencyResponse:
    """Equilibria, step responses and gains of ``model`` at ``speeds``, from one batched rollout.

    The steps ``delta`` and ``delta_large`` (both signs each, identical hold) run in the same
    batch. ``delta_large=None`` skips the finite-amplitude pass.
    """
    speeds = np.asarray(torch.as_tensor(speeds, dtype=torch.float64).reshape(-1).cpu(), dtype=np.float64)
    n = len(speeds)
    if n == 0:
        raise ValueError("no speeds to evaluate")
    deltas = [float(delta)] if delta_large is None else [float(delta), float(delta_large)]
    for step in deltas:
        if step <= 0 or bool((speeds - step < 0).any()):
            raise ValueError(f"a step must be positive and leave the leader speed non-negative, got {step}")
    n_settle = int(round(SETTLE / DT))
    if int(round(hold / DT)) - n_settle <= WARMUP:
        raise ValueError(f"a hold of {hold} s leaves less than {SETTLE} s of simulated hold after the warm-up")
    model = model.to(device).eval()
    s_analytic = np.full(n, np.nan)
    if getattr(model, "memoryless", False):
        s_analytic = equilibrium_gap(model, speeds).double().cpu().numpy()
    s_init = np.where(np.isfinite(s_analytic), s_analytic, [initial_gap(v) for v in speeds])

    # rows: for every step, all speeds stepping up, then all speeds stepping down
    events = [event for step in deltas for event in step_events(speeds, s_init, step, hold, response)]
    k_s = int(events[0].meta["manoeuvre_start"])
    batch = EventBatch.from_events(events, device=device, dtype=dtype)
    result = rollout(model, batch)
    s, v, a = (x.double().cpu().numpy() for x in (result.s, result.v, result.a))

    hold_part = slice(k_s - n_settle, k_s)
    target = np.tile(speeds, 2 * len(deltas))[:, None]
    s_hold = s[:, hold_part]
    settled = (
        (np.abs(a[:, hold_part]) < A_SETTLED).all(-1)
        & (np.abs(v[:, hold_part] - target) < V_SETTLED).all(-1)
        & ((s_hold > 0.0) & (s_hold < S_MAX)).all(-1)
    )
    converged = settled.reshape(-1, n).all(0)  # the holds are identical: every scenario of a speed
    s_e = np.where(converged, s_hold.mean(-1).reshape(-1, n).mean(0), np.nan)

    # [steps, 2, n, K]: v_f[k_s + k + 1] - v_f[k_s + k] of the steps up and down
    increments = np.diff(v[:, k_s:], axis=-1).reshape(len(deltas), 2, n, -1)
    impulses = [0.5 * (up / step + down / -step) for (up, down), step in zip(increments, deltas)]
    final = (v[:, -1] - batch.v_lead[:, -1].double().cpu().numpy()).reshape(len(deltas), 2, n)
    tracks = [(np.abs(error) < TRACK_SHARE * step).all(0) for error, step in zip(final, deltas)]
    collided = (s[:, k_s + 1 :] <= 0.0).any(-1).reshape(len(deltas), 2, n).any(1)  # [steps, n]
    large = len(deltas) > 1
    return FrequencyResponse(
        speeds=speeds,
        s_init=s_init,
        s_e_analytic=s_analytic,
        converged=converged,
        s_e=s_e,
        impulse=impulses[0],
        frequencies=FREQUENCIES.copy(),
        gain=dft_gain(impulses[0], FREQUENCIES),
        table_gain=dft_gain(impulses[0], TABLE_FREQUENCIES),
        tracks=tracks[0],
        final_error=final[0].T,
        n_invalid=int(result.n_invalid),
        delta=deltas[0],
        step_index=k_s,
        delta_large=deltas[1] if large else None,
        impulse_large=impulses[1] if large else None,
        gain_large=dft_gain(impulses[1], FREQUENCIES) if large else None,
        tracks_large=tracks[1] if large else None,
        final_error_large=final[1].T if large else None,
        collided_large=collided[1] if large else None,
    )


def _share(flags: np.ndarray) -> float | None:
    return float(np.mean(flags)) if len(flags) else None


def summarise(
    model: CFModel, response: FrequencyResponse, tol: float = TOL
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Summary and table ``speeds`` of P3 from the numerical part, plus the analytic part.

    The table has every speed. Its numerical columns are ``nan`` (verdicts ``<NA>``) where the
    hold did not converge, except ``tracks``, which is defined for every speed. Its analytic
    columns are ``nan`` everywhere for a model that is not memoryless. The equilibria of the
    summary are the converged speeds whose follower also tracks the small step. Every fraction,
    the agreement, ``max_gain`` and ``max_gain_large`` are taken over the equilibria, and are
    ``None`` when there is none. ``tracks_share`` is taken over the converged speeds, and
    ``n_collided_large`` counts the converged speeds that collide after the 1 m/s step.
    """
    r = response
    n = len(r.speeds)
    conv = r.converged
    equilibrium = conv & r.tracks
    gain = np.where(conv[:, None], r.gain, np.nan)
    max_gain = gain.max(-1)
    peak = np.where(conv, r.frequencies[np.argmax(r.gain, axis=-1)], np.nan)
    unstable_numerical = conv & (max_gain > 1.0 + tol)
    unstable_1e3 = conv & (max_gain > 1.0 + PRACTICAL_TOL)
    large = conv & (r.gain_large is not None)  # speeds with a finite-amplitude result
    max_gain_large = np.where(large, np.nan if r.gain_large is None else r.gain_large.max(-1), np.nan)
    unstable_large = large & (max_gain_large > 1.0 + PRACTICAL_TOL)
    # a speed that collides keeps its max_gain_large: a runaway is a finding, counted next to it
    tracks_large = large & (False if r.tracks_large is None else r.tracks_large)
    collided_large = large & (False if r.collided_large is None else r.collided_large)

    memoryless = bool(getattr(model, "memoryless", False))
    f_s, f_dv, f_v, margin = (np.full(n, np.nan) for _ in range(4))
    if memoryless and conv.any():
        derivatives = partials(model, r.s_e[conv], 0.0, r.speeds[conv])
        f_s[conv], f_dv[conv], f_v[conv] = (d.double().cpu().numpy() for d in derivatives)
        margin[conv] = string_margin(*(torch.as_tensor(d) for d in (f_s[conv], f_dv[conv], f_v[conv]))).numpy()
    analytic = np.isfinite(margin) & conv
    unstable_analytic = analytic & (margin < 0.0)

    table = pd.DataFrame(
        {
            "v_e": r.speeds,
            "s_e": r.s_e,
            "s_e_analytic": r.s_e_analytic,
            "converged": conv,
            "tracks": pd.arrays.BooleanArray(r.tracks.copy(), np.zeros(n, dtype=bool)),
            "max_gain": max_gain,
            "peak_frequency": peak,
            **{f"gain_{f:g}": np.where(conv, r.table_gain[:, j], np.nan) for j, f in enumerate(TABLE_FREQUENCIES)},
            "margin": margin,
            "f_s": f_s,
            "f_dv": f_dv,
            "f_v": f_v,
            "unstable_numerical": pd.arrays.BooleanArray(unstable_numerical, ~conv),
            "unstable_1e3": pd.arrays.BooleanArray(unstable_1e3, ~conv),
            "max_gain_large": max_gain_large,
            "unstable_large": pd.arrays.BooleanArray(unstable_large, ~large),
            "tracks_large": pd.arrays.BooleanArray(tracks_large, ~large),
            "collided_large": pd.arrays.BooleanArray(collided_large, ~large),
            "unstable_analytic": pd.arrays.BooleanArray(unstable_analytic, ~analytic),
        },
        columns=list(COLUMNS),
    )
    eq, eq_large, eq_analytic = equilibrium, equilibrium & large, equilibrium & analytic
    best = int(np.nanargmax(np.where(eq, max_gain, np.nan))) if eq.any() else None
    summary = {
        "unstable_fraction": _share(unstable_numerical[eq]),
        "unstable_fraction_1e3": _share(unstable_1e3[eq]),
        "unstable_fraction_large": _share(unstable_large[eq_large]),
        "max_gain_large": float(np.nanmax(max_gain_large[eq_large])) if eq_large.any() else None,
        "n_collided_large": int(collided_large.sum()) if large.any() else None,
        "unstable_fraction_analytic": _share(unstable_analytic[eq_analytic]) if memoryless else None,
        "max_gain": None if best is None else float(max_gain[best]),
        "peak_frequency": None if best is None else float(peak[best]),
        "n_equilibria": int(eq.sum()),
        "n_converged": int(conv.sum()),
        "n_tracks": int(r.tracks.sum()),
        "n_speeds": int(n),
        "agreement": _share(unstable_numerical[eq_analytic] == unstable_analytic[eq_analytic]) if memoryless else None,
        "tracks_share": _share(r.tracks[conv]),
        "n_invalid": int(r.n_invalid),
        "tol": float(tol),
    }
    return summary, {"speeds": table}


def evaluate_p3(
    model: CFModel,
    speeds: Speeds = SPEEDS,
    delta: float = DELTA,
    hold: float = HOLD,
    response: float = RESPONSE,
    tol: float = TOL,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    delta_large: float | None = DELTA_LARGE,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """String stability of ``model``: summary and table ``speeds`` (``docs/protocols.md``, section 3)."""
    response_ = frequency_response(model, speeds, delta, hold, response, device, dtype, delta_large)
    return summarise(model, response_, tol)
