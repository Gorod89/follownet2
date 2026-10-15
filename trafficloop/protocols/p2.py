"""P2: platoon metrics (docs/protocols.md, section 2).

Every profile is driven once by a platoon of ``max(n_vehicles)`` followers
(:func:`~trafficloop.sim.platoon.simulate_platoon`); the smaller platoons are its prefixes. The
platoon starts at the model's own equilibrium gap at ``v0 = v[0]`` (:func:`settle_gap`): a single
follower behind a leader at ``v0`` for 120 s (``scenarios.constant_speed_event``, ``rollout``),
converged by the rule of P3 when over the last 10 s ``max |a| < 2e-3 m/s^2``,
``|v - v0| < 0.01 m/s`` and ``0 < s < 300 m``; the starting gap is then the mean gap of those
10 s, otherwise ``initial_gap(v0)`` (``s_init_converged`` False). With any other starting gap
follower ``i`` would have to fall back by ``i (s_e - s_init)``, which no hold absorbs for 100
vehicles. Metrics use the manoeuvre window (from the manoeuvre start to the end) unless stated
otherwise:

1. ``std_v[n]`` of every vehicle ``n = 0..N``. Per profile, the concave-growth error is the RMS
   over the recorded followers of ``std_v_sim[n] - std_v_rec[n]``; the growth exponent ``beta``
   is the least-squares slope of ``log std_v[n]`` on ``log n`` over the followers with
   ``std_v > 0``.
2. Hysteresis: the signed shoelace area of the closed polygon ``(s_n(t), v_n(t))`` [m^2/s],
   positive when the loop runs counterclockwise with ``s`` on the horizontal axis; per profile
   the RMS over the recorded followers of ``area_sim[n] - area_rec[n]``.
3. Wave speed at the deepest leader deceleration (``t_min = argmin v_0`` over the window,
   ``v_before = max v_0`` over the 30 s of the window before it, threshold their mean): the first
   downward crossing of the threshold by every vehicle after the leader's crossing, times and
   positions interpolated linearly between samples; the slope of ``x = c t + d`` over the
   crossing vehicles (the leader included, at least five). The positions are the physical
   front-bumper positions ``x_n - n L`` (``L`` the profile's vehicle length): the simulator's frame
   has no vehicle lengths, and a wave speed relative to the road needs the physical positions.
4. Fundamental diagram: ``k = 1000 / (s + L)`` [veh/km] and ``q = 3.6 k v`` [veh/h] for every
   window sample of the followers and every sample of ``source_events`` (``L`` =
   ``meta["leader_length"]`` or 4.5 m); bins of 5 veh/km on [0, 150); the bins with at least 200
   samples on both sides are compared (RMS of the mean-flow difference, absolute and relative
   to the source).
5. Collisions: followers with ``s <= 0`` at any simulated sample (the hold after the warm-up
   included).
6. Settled: followers with ``|a| < 0.05 m/s^2`` over the last 5 s of the hold.

Summary: ``concave_growth_error``, ``hysteresis_error`` and ``growth_exponent_<n>`` are means over
the recorded profiles (every profile but the synthetic ones: the OpenACC windows of P2, the human
platoon windows of P2h); ``wave_speed`` is the mean over the recorded profiles of the fit over the
leader and the first 20 followers (comparable between models), ``wave_speed_<n>`` the same over
the crossing vehicles of the platoon of ``n`` (``n`` other than 20), ``wave_speed_pulse`` and
``wave_speed_pulse_<n>`` the same over the synthetic profiles; profiles without a fit are left out
and ``n_wave_profiles`` counts the recorded profiles in ``wave_speed``. Profiles may differ in
length and in the number of recorded followers: every metric is computed per profile over its own
window. ``hysteresis_area_<n>`` is the mean ``|area|`` of the first ``n`` followers over all
profiles; ``fd_*`` pool the largest
platoon of all profiles; ``collisions_<n>`` are summed over all profiles; ``settled_share`` and
``n_invalid`` cover the largest platoon of all profiles; ``s_init_converged_share`` is the share of
profiles whose starting gap converged. ``<n>`` runs over ``n_vehicles`` (20 and 100 by default).
``n_profiles_in_range`` counts the profiles inside ``speed_range`` (:func:`in_speed_range`; all
without a range), and the keys ``concave_growth_error``, ``hysteresis_error``,
``growth_exponent_<n>``, ``wave_speed``, ``collisions_<n>`` and ``settled_share`` with the suffix
``_in_range`` repeat their computation over those profiles (null when none is in range).
Tables: ``profiles`` (one row per profile and platoon size, with ``in_range``) and ``vehicles`` (one
row per profile and vehicle of the largest platoon, the leader as index 0).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from trafficloop.data.batch import EventBatch
from trafficloop.data.regimes import RegimeConfig
from trafficloop.data.schema import DT, Event
from trafficloop.models.base import ACC_CLIP, CFModel
from trafficloop.protocols.profiles import SYNTHETIC, LeaderProfile
from trafficloop.protocols.scenarios import HOLD, constant_speed_event, initial_gap
from trafficloop.sim.platoon import PlatoonResult, simulate_platoon
from trafficloop.sim.rollout import rollout

N_VEHICLES = (20, 100)
#: starting-gap search, the equilibrium rule of P3 (protocols/p3.py)
EQ_HOLD = 120.0  # s, single follower behind a leader at v0, warm-up included
EQ_WINDOW = 10.0  # s, end of the hold over which convergence is decided
EQ_ACC = 2e-3  # m/s^2, max |a|
EQ_SPEED = 0.01  # m/s, max |v - v0|
EQ_GAP_MAX = 300.0  # m, the gap stays inside (0, EQ_GAP_MAX)
#: followers (behind the leader) of the wave-speed fit that enters ``wave_speed``
WAVE_FOLLOWERS = 20
#: m/s, a profile is in the speed range (low, high) when its leader stays in [low - 1, high + 1]
SPEED_MARGIN = 1.0
FD_BIN_WIDTH = 5.0  # veh/km
FD_MAX_DENSITY = 150.0  # veh/km
FD_MIN_SAMPLES = 200
#: leader length of source events without ``meta["leader_length"]`` (the density proxy's default)
SOURCE_LENGTH = RegimeConfig().default_length
SETTLE_WINDOW = 5.0  # s, end of the hold
SETTLE_ACC = 0.05  # m/s^2
WAVE_LOOKBACK = 30.0  # s
WAVE_MIN_VEHICLES = 5

_NAN = float("nan")


def _rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else _NAN


def _mean(values: Iterable[float]) -> float:
    """Mean of the finite values, nan without any."""
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else _NAN


# ----------------------------------------------------------------------------- metric helpers
def shoelace_area(s, v) -> np.ndarray:
    """Signed area of the closed polygon ``(s(t), v(t))`` along the last axis [m^2/s].

    Positive when the loop runs counterclockwise with ``s`` to the right and ``v`` up. The
    coordinates are centred first (the area does not depend on the origin), which avoids the
    cancellation of large products.
    """
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    s = s - s.mean(axis=-1, keepdims=True)
    v = v - v.mean(axis=-1, keepdims=True)
    return 0.5 * np.sum(s * np.roll(v, -1, axis=-1) - np.roll(s, -1, axis=-1) * v, axis=-1)


def growth_exponent(std_v) -> float:
    """Least-squares slope of ``log std_v[n]`` on ``log n`` over the followers ``n >= 1`` with
    ``std_v[n] > 0``; ``std_v[0]`` is the leader. nan with fewer than two such followers."""
    std = np.asarray(std_v, dtype=np.float64)[1:]
    n = np.arange(1, len(std) + 1, dtype=np.float64)
    keep = np.isfinite(std) & (std > 0)
    if keep.sum() < 2:
        return _NAN
    x, y = np.log(n[keep]), np.log(std[keep])
    x = x - x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def wave_crossings(v, x, start: int, dt: float = DT, lookback: float = WAVE_LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    """Threshold crossings of the deepest leader deceleration (module docstring, item 3).

    ``v`` and ``x`` are the speeds and positions ``[V, T]`` of the vehicles ``0..V-1`` (0 = leader);
    the window runs from the sample ``start`` to the end. Returns the time [s after ``start``] and
    the position of the first downward crossing of every vehicle after the leader's, nan for the
    vehicles that do not cross (all nan when the leader does not decelerate in the window).
    """
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n_veh, n_steps = v.shape
    t_cross, x_cross = np.full(n_veh, _NAN), np.full(n_veh, _NAN)
    lead = v[0]
    k_min = start + int(np.argmin(lead[start:]))
    k_lo = max(start, k_min - int(round(lookback / dt)))
    k_before = k_lo + int(np.argmax(lead[k_lo : k_min + 1]))
    threshold = 0.5 * (lead[k_before] + lead[k_min])
    down = (v[:, :-1] >= threshold) & (v[:, 1:] < threshold)  # a crossing inside the step j -> j + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(down, (v[:, :-1] - threshold) / (v[:, :-1] - v[:, 1:]), 0.0)
    first_lead = np.flatnonzero(down[0, k_before:k_min])
    if len(first_lead) == 0:
        return t_cross, x_cross
    j0 = k_before + int(first_lead[0])
    instant = np.arange(n_steps - 1) + frac  # crossing instants [samples] where ``down``
    after = down & (instant > j0 + frac[0, j0])
    after[0] = False
    has = after.any(axis=1)
    j = np.argmax(after, axis=1)
    has[0], j[0] = True, j0
    rows = np.flatnonzero(has)
    j, w = j[rows], frac[rows, j[rows]]
    t_cross[rows] = (j + w - start) * dt
    x_cross[rows] = x[rows, j] + w * (x[rows, j + 1] - x[rows, j])
    return t_cross, x_cross


def fit_wave_speed(t_cross, x_cross, min_vehicles: int = WAVE_MIN_VEHICLES) -> tuple[float, int]:
    """Slope ``c`` [m/s] of the least-squares line ``x = c t + d`` through the crossings and the
    number of crossing vehicles; ``c`` is nan with fewer than ``min_vehicles`` crossings."""
    t = np.asarray(t_cross, dtype=np.float64)
    x = np.asarray(x_cross, dtype=np.float64)
    keep = np.isfinite(t) & np.isfinite(x)
    n = int(keep.sum())
    if n < min_vehicles:
        return _NAN, n
    t, x = t[keep] - t[keep].mean(), x[keep]
    denom = float(np.dot(t, t))
    return (float(np.dot(t, x - x.mean()) / denom) if denom > 0 else _NAN), n


def fd_samples(s, v, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Density ``k = 1000 / (s + L)`` [veh/km] and flow ``q = k v 3.6`` [veh/h] of every sample."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 1000.0 / (np.asarray(s, dtype=np.float64) + float(length))
        return k, k * np.asarray(v, dtype=np.float64) * 3.6


def fd_histogram(k, q, width: float = FD_BIN_WIDTH, k_max: float = FD_MAX_DENSITY) -> tuple[np.ndarray, np.ndarray]:
    """Flow sums and sample counts in the density bins ``[0, width), ..., [k_max - width, k_max)``;
    samples outside the range or not finite are left out."""
    n_bins = int(round(k_max / width))
    k, q = np.ravel(k), np.ravel(q)
    keep = np.isfinite(k) & np.isfinite(q) & (k >= 0.0) & (k < k_max)
    index = np.minimum((k[keep] / width).astype(np.int64), n_bins - 1)
    return np.bincount(index, weights=q[keep], minlength=n_bins), np.bincount(index, minlength=n_bins)


def fd_compare(sim, source, min_samples: int = FD_MIN_SAMPLES) -> tuple[float, float, int]:
    """``(fd_error, fd_error_rel, fd_bins_compared)`` of two :func:`fd_histogram` results."""
    (sim_q, sim_n), (src_q, src_n) = sim, source
    both = (np.asarray(sim_n) >= min_samples) & (np.asarray(src_n) >= min_samples)
    n_bins = int(both.sum())
    if n_bins == 0:
        return _NAN, _NAN, 0
    q_sim, q_src = sim_q[both] / sim_n[both], src_q[both] / src_n[both]
    diff = q_sim - q_src
    positive = q_src > 0
    return _rms(diff), (_rms(diff[positive] / q_src[positive]) if positive.any() else _NAN), n_bins


def source_histogram(events: Iterable[Event]) -> tuple[np.ndarray, np.ndarray]:
    """:func:`fd_histogram` of every sample of the events (``L`` = ``meta["leader_length"]`` or 4.5 m)."""
    sums, counts = fd_histogram(np.empty(0), np.empty(0))
    for ev in events:
        length = ev.meta.get("leader_length")
        if length is None or not np.isfinite(length) or length <= 0:
            length = SOURCE_LENGTH
        q, n = fd_histogram(*fd_samples(ev.s, ev.v, float(length)))
        sums, counts = sums + q, counts + n
    return sums, counts


def settled_vehicles(result: PlatoonResult, window: float = SETTLE_WINDOW, threshold: float = SETTLE_ACC) -> np.ndarray:
    """``[N + 1]``: ``|a| < threshold`` over the last ``window`` seconds of the hold (the leader
    drives at constant speed there)."""
    end = result.manoeuvre_start
    begin = max(end - int(round(window / result.dt)), result.warmup)
    return np.all(np.abs(result.a[:, begin:end]) < threshold, axis=1)


# ----------------------------------------------------------------------------- protocol
@torch.no_grad()
def settle_gap(
    model: CFModel,
    v0: float,
    hold: float = EQ_HOLD,
    clip: tuple[float, float] = ACC_CLIP,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[float, bool]:
    """Starting gap of the platoon: the model's equilibrium gap at ``v0`` and whether it converged.

    One follower behind a leader at ``v0`` (``constant_speed_event``, starting gap
    ``initial_gap(v0)``) is rolled for ``hold`` seconds, alone in its batch so that the result does
    not depend on the other profiles. Converged when over the last ``EQ_WINDOW`` seconds
    ``max |a| < EQ_ACC``, ``max |v - v0| < EQ_SPEED`` and ``0 < s < EQ_GAP_MAX``; the gap is then
    the mean gap of those samples, otherwise ``initial_gap(v0)``.
    """
    v0 = float(v0)
    event = constant_speed_event(v0, 0.0, "p2_settle", hold=hold)
    result = rollout(model, EventBatch.from_events([event], device=device, dtype=dtype), clip=clip)
    end = int(event.meta["manoeuvre_start"])
    part = slice(end - int(round(EQ_WINDOW / DT)), end)
    s, v, a = (x[0, part].double().cpu().numpy() for x in (result.s, result.v, result.a))
    converged = bool(
        (np.abs(a) < EQ_ACC).all() and (np.abs(v - v0) < EQ_SPEED).all() and ((s > 0.0) & (s < EQ_GAP_MAX)).all()
    )
    return (float(s.mean()), True) if converged else (initial_gap(v0), False)


def platoon_metrics(
    profile: LeaderProfile,
    result: PlatoonResult,
    s_init: float,
    s_init_converged: bool,
    source: tuple[np.ndarray, np.ndarray],
    sizes: Sequence[int],
) -> tuple[list[dict], pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]], float]:
    """Rows of the ``profiles`` table (one per platoon size), the ``vehicles`` table of the profile,
    the FD histograms per size and the wave speed over the leader and the first
    :data:`WAVE_FOLLOWERS` followers."""
    start, n_max = result.manoeuvre_start, result.n_followers
    v, s = result.v[:, start:], result.s[:, start:]
    std_v = v.std(axis=1)
    area = np.concatenate([[_NAN], shoelace_area(s, v[1:])])
    length = profile.vehicle_length
    x_front = result.x - length * np.arange(n_max + 1, dtype=np.float64)[:, None]
    t_cross, x_cross = wave_crossings(result.v, x_front, start, result.dt)
    settled = settled_vehicles(result)
    std_rec, area_rec = np.full(n_max + 1, _NAN), np.full(n_max + 1, _NAN)
    for follower in profile.followers:
        if 1 <= follower.index <= n_max:
            std_rec[follower.index] = np.std(follower.v)
            area_rec[follower.index] = shoelace_area(follower.s, follower.v)
    recorded = np.isfinite(std_rec)
    concave = _rms(std_v[recorded] - std_rec[recorded]) if recorded.any() else _NAN
    hysteresis = _rms(area[recorded] - area_rec[recorded]) if recorded.any() else _NAN
    k, q = fd_samples(s, v[1:], length)
    rows, histograms = [], {}
    for n in sizes:
        histograms[n] = fd_histogram(k[:n], q[:n])
        fd_error, fd_rel, fd_bins = fd_compare(histograms[n], source)
        wave, n_crossed = fit_wave_speed(t_cross[: n + 1], x_cross[: n + 1])
        rows.append(
            {
                "profile": profile.name,
                "source": profile.source,
                "n_vehicles": int(n),
                "n_recorded": int(recorded.sum()),
                "vehicle_length": float(length),
                "s_init": float(s_init),
                "s_init_converged": bool(s_init_converged),
                "concave_growth_error": concave,
                "hysteresis_error": hysteresis,
                "growth_exponent": growth_exponent(std_v[: n + 1]),
                "hysteresis_area": float(np.mean(np.abs(area[1 : n + 1]))),
                "wave_speed": wave,
                "n_crossed": n_crossed,
                "fd_error": fd_error,
                "fd_error_rel": fd_rel,
                "fd_bins_compared": fd_bins,
                "collisions": int(result.collided[:n].sum()),
                "settled_share": float(settled[1 : n + 1].mean()),
                "n_invalid": int(result.invalid[:n].sum()),
            }
        )
    vehicles = pd.DataFrame(
        {
            "profile": profile.name,
            "index": np.arange(n_max + 1),
            "std_v": std_v,
            "area": area,
            "t_cross": t_cross,
            "x_cross": x_cross,
            "collided": np.concatenate([[False], result.collided]),
            "settled": settled,
            "std_v_rec": std_rec,
            "area_rec": area_rec,
        }
    )
    wave = fit_wave_speed(t_cross[: WAVE_FOLLOWERS + 1], x_cross[: WAVE_FOLLOWERS + 1])[0]
    return rows, vehicles, histograms, wave


@torch.no_grad()
def in_speed_range(profile: LeaderProfile, speed_range: tuple[float, float] | None) -> bool:
    """Whether the leader speed of ``profile`` stays within ``[low - 1, high + 1]`` m/s; the synthetic
    profiles always are (the runner chooses their speed), and every profile is without a range."""
    if speed_range is None or profile.source == SYNTHETIC:
        return True
    low, high = (float(x) for x in speed_range)
    return bool(profile.v.min() >= low - SPEED_MARGIN and profile.v.max() <= high + SPEED_MARGIN)


def evaluate_p2(
    model: CFModel,
    profiles: Sequence[LeaderProfile],
    source_events: Iterable[Event],
    n_vehicles: Sequence[int] = N_VEHICLES,
    hold: float = HOLD,
    clip: tuple[float, float] = ACC_CLIP,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    speed_range: tuple[float, float] | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Module docstring; ``speed_range`` ``(low, high)`` [m/s] (the speeds of the training data)
    adds the ``*_in_range`` keys over the profiles that stay inside it (``in_speed_range``)."""
    profiles = list(profiles)
    if not profiles:
        raise ValueError("P2 needs at least one leader profile")
    sizes = sorted({int(n) for n in n_vehicles})
    if sizes[0] < 1:
        raise ValueError(f"platoon sizes must be positive, got {list(n_vehicles)}")
    n_max = sizes[-1]
    model = model.to(device).eval()
    source = source_histogram(source_events)
    pooled_q, pooled_n = np.zeros_like(source[0]), np.zeros_like(source[1])
    records, rows, vehicle_tables = [], [], []
    settled, followers = 0.0, 0
    for profile in profiles:
        s_init, converged = settle_gap(model, profile.v[0], clip=clip, device=device, dtype=dtype)
        result = simulate_platoon(model, profile.v, n_max, s_init, hold, clip, device, dtype)
        profile_rows, vehicles, histograms, wave = platoon_metrics(profile, result, s_init, converged, source, sizes)
        inside = in_speed_range(profile, speed_range)
        for row in profile_rows:
            row["in_range"] = inside
        rows += profile_rows
        vehicle_tables.append(vehicles)
        n_settled = float(vehicles["settled"].to_numpy()[1:].sum())
        records.append(
            {
                "profile": profile,
                "rows": {row["n_vehicles"]: row for row in profile_rows},
                "wave": wave,
                "converged": converged,
                "in_range": inside,
                "settled": n_settled,
            }
        )
        pooled_q, pooled_n = pooled_q + histograms[n_max][0], pooled_n + histograms[n_max][1]
        settled += n_settled
        followers += n_max

    recorded = [r for r in records if r["profile"].is_recorded]  # OpenACC (P2) or human platoon windows (P2h)
    pulse = [r for r in records if r["profile"].source == SYNTHETIC]
    other_sizes = [n for n in sizes if n != WAVE_FOLLOWERS]
    fd_error, fd_rel, fd_bins = fd_compare((pooled_q, pooled_n), source)
    summary = {
        "concave_growth_error": _mean(r["rows"][n_max]["concave_growth_error"] for r in recorded),
        "hysteresis_error": _mean(r["rows"][n_max]["hysteresis_error"] for r in recorded),
        **{f"growth_exponent_{n}": _mean(r["rows"][n]["growth_exponent"] for r in recorded) for n in sizes},
        **{f"hysteresis_area_{n}": _mean(r["rows"][n]["hysteresis_area"] for r in records) for n in sizes},
        "wave_speed": _mean(r["wave"] for r in recorded),
        **{f"wave_speed_{n}": _mean(r["rows"][n]["wave_speed"] for r in recorded) for n in other_sizes},
        "wave_speed_pulse": _mean(r["wave"] for r in pulse),
        **{f"wave_speed_pulse_{n}": _mean(r["rows"][n]["wave_speed"] for r in pulse) for n in other_sizes},
        "n_wave_profiles": int(sum(bool(np.isfinite(r["wave"])) for r in recorded)),
        "fd_error": fd_error,
        "fd_error_rel": fd_rel,
        "fd_bins_compared": fd_bins,
        **{f"collisions_{n}": int(sum(r["rows"][n]["collisions"] for r in records)) for n in sizes},
        "n_profiles": len(profiles),
        "settled_share": settled / followers,
        "s_init_converged_share": float(np.mean([r["converged"] for r in records])),
        "n_invalid": int(sum(r["rows"][n_max]["n_invalid"] for r in records)),
    }
    inside = [r for r in records if r["in_range"]]
    recorded_in = [r for r in inside if r["profile"].is_recorded]
    in_range = {
        "concave_growth_error_in_range": _mean(r["rows"][n_max]["concave_growth_error"] for r in recorded_in),
        "hysteresis_error_in_range": _mean(r["rows"][n_max]["hysteresis_error"] for r in recorded_in),
        **{f"growth_exponent_{n}_in_range": _mean(r["rows"][n]["growth_exponent"] for r in recorded_in) for n in sizes},
        "wave_speed_in_range": _mean(r["wave"] for r in recorded_in),
        **{f"collisions_{n}_in_range": int(sum(r["rows"][n]["collisions"] for r in inside)) for n in sizes},
        "settled_share_in_range": sum(r["settled"] for r in inside) / (n_max * len(inside)) if inside else None,
    }
    summary["n_profiles_in_range"] = len(inside)
    summary.update(in_range if inside else dict.fromkeys(in_range))  # null without a profile in range
    tables = {"profiles": pd.DataFrame(rows), "vehicles": pd.concat(vehicle_tables, ignore_index=True)}
    return summary, tables
