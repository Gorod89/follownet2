"""FollowNet-compatible extraction of car-following events.

The FollowNet repository ships no extraction code, so the criteria of the paper are
re-implemented here (and kept identical to the data contract of cf-stability):

A sample is *valid* when the leader exists, all values are finite, ``min_spacing <= s < max_spacing``
and both speeds are non-negative. A segment boundary is placed wherever a sample is
invalid, the leader id changes, the lane of the follower or of the leader changes, the
segment key of the source changes (driving mode), or the time grid has a gap. Events are the maximal runs of valid samples with

1. a single, unchanged leader and no lane change of either vehicle;
2. at least ``min_duration / dt`` samples: 150 for 15 s. FollowNet counts a duration as the
   number of samples times the step (its 15 s windows hold 375 samples at 25 Hz and 150 at
   10 Hz), so ``Event.duration``, the time from the first to the last sample, is 14.9 s for
   the shortest event;
3. spacing of at least ``min_spacing`` (0.5 m: smaller net gaps are overlapping or touching
   bounding boxes, not following) and below ``max_spacing`` (120 m) at every sample;
4. a moving follower (``v > moving_speed_eps``) in at least ``min_moving_fraction`` of the samples.

Speeds between ``-negative_speed_tol`` and zero are measurement noise of a standing vehicle
and are set to zero; a speed below that ends the segment.

Loaders of track-based sources build a standard *tracks* frame and call
:func:`tracks_to_pairs`; sources that are already segmented into pairs build
:class:`PairSeries` objects directly. Everything passes through :func:`extract_events`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Mapping

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

from trafficloop.data.schema import DT, Event

TRACK_REQUIRED = ("group", "vehicle_id", "t", "x_front", "x_rear", "v")
TRACK_OPTIONAL = ("site", "lane", "a", "leader", "length")


@dataclass
class ExtractionConfig:
    min_duration: float = 15.0
    max_duration: float | None = None
    min_spacing: float = 0.5
    max_spacing: float = 120.0
    moving_speed_eps: float = 0.1
    min_moving_fraction: float = 0.8
    negative_speed_tol: float = 0.5
    smooth_speeds: bool = False
    savgol_window: int = 11
    savgol_order: int = 2
    # "auto": dataset acceleration when the source has one, otherwise Savitzky-Golay;
    # "savgol": always differentiate the follower speed
    accel_source: str = "auto"
    dt: float = DT

    def __post_init__(self) -> None:
        if self.max_duration is not None and self.max_duration < 2 * self.min_duration:
            raise ValueError("max_duration must be at least 2 * min_duration")
        if self.accel_source not in ("auto", "savgol"):
            raise ValueError(f"unknown accel_source {self.accel_source!r}")

    @property
    def min_samples(self) -> int:
        return int(round(self.min_duration / self.dt))


@dataclass
class PairSeries:
    """A follower on a uniform 10 Hz grid together with its instantaneous leader.

    Positions follow the bumper convention of the schema: ``x_follower`` is the FRONT
    bumper of the follower and ``x_lead`` the REAR bumper of the leader. Unobserved
    samples are NaN; ``leader_id`` is ``None`` where there is no leader.
    """

    dataset: str
    site: str
    follower_id: str
    t: np.ndarray
    x_follower: np.ndarray
    v: np.ndarray
    leader_id: np.ndarray
    x_lead: np.ndarray
    v_lead: np.ndarray
    a: np.ndarray | None = None
    lane_follower: np.ndarray | None = None
    lane_leader: np.ndarray | None = None
    # per-sample label; a change of the label is a segment boundary (driving mode, part of a run)
    segment_key: np.ndarray | None = None
    # per-sample arrays aggregated into the event meta (median of numbers, first value otherwise)
    extras: dict[str, np.ndarray] = field(default_factory=dict)
    # copied into the meta of every event of this follower
    meta: dict = field(default_factory=dict)


def positions_from_gap(v_lead: np.ndarray, s: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """Positions for sources without a longitudinal coordinate (OpenACC, FollowNet).

    ``x_lead`` is the trapezoidal integral of ``v_lead`` starting at 0 and
    ``x_follower = x_lead - s``.
    """
    v_lead = np.asarray(v_lead, dtype=np.float64)
    x_lead = cumulative_trapezoid(np.nan_to_num(v_lead), dx=dt, initial=0.0)
    return x_lead, x_lead - np.asarray(s, dtype=np.float64)


def to_grid(
    t: np.ndarray,
    columns: Mapping[str, np.ndarray],
    dt: float = DT,
    nearest: Iterable[str] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Put samples taken at the times ``t`` [s] on the grid ``t = g * dt``.

    Used by the gap-based loaders. Time stamps on the grid (within ``1e-6 * dt``) are placed
    as they are and grid points without a sample are NaN. Otherwise every column is
    interpolated linearly, the columns in ``nearest`` take the nearest sample, and grid points
    inside a hole of the source longer than 1.5 times the median source step are NaN, as in
    ``resample_tracks``. Repeated time stamps keep their first sample.
    """
    t = np.asarray(t, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t = t[order]
    first = np.ones(len(t), dtype=bool)
    first[1:] = np.diff(t) > 1e-9
    t = t[first]
    values = {name: np.asarray(col, dtype=np.float64)[order][first] for name, col in columns.items()}
    ratio = t / dt
    on_grid = np.round(ratio)
    if len(t) and np.all(np.abs(ratio - on_grid) < 1e-6):
        g = on_grid.astype(np.int64)
        grid = np.arange(g[0], g[-1] + 1)
        out = {}
        for name, col in values.items():
            out[name] = np.full(len(grid), np.nan)
            out[name][g - g[0]] = col
        return grid * dt, out
    if len(t) < 2:
        return np.empty(0), {name: np.empty(0) for name in values}
    grid = np.arange(math.ceil(t[0] / dt - 1e-6), math.floor(t[-1] / dt + 1e-6) + 1)
    tg = grid * dt
    right = np.clip(np.searchsorted(t, tg, side="left"), 1, len(t) - 1)
    left = right - 1
    hole = t[right] - t[left] > 1.5 * float(np.median(np.diff(t))) + 1e-9
    closest = np.where(tg - t[left] <= t[right] - tg, left, right)
    nearest = set(nearest)
    out = {}
    for name, col in values.items():
        out[name] = col[closest] if name in nearest else np.interp(tg, t, col)
        out[name][hole] = np.nan
    return tg, out


def savgol_derivative(y: np.ndarray, dt: float, window: int = 11, order: int = 2, deriv: int = 1) -> np.ndarray:
    n = len(y)
    if n < 3:  # one or two samples between two leader changes: no filter, no gradient
        y = np.asarray(y, dtype=np.float64)
        slope = (y[-1] - y[0]) / dt if n == 2 else 0.0
        return y.copy() if deriv == 0 else np.full(n, slope if deriv == 1 else 0.0)
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= order:
        out = np.asarray(y, dtype=np.float64)
        for _ in range(deriv):
            out = np.gradient(out, dt)
        return out
    return savgol_filter(y, window, order, deriv=deriv, delta=dt, mode="interp")


def _aggregate_extra(values: np.ndarray):
    values = np.asarray(values)
    if values.dtype.kind in "fiu":
        finite = values[np.isfinite(values.astype(np.float64))]
        return float(np.median(finite)) if len(finite) else None
    if values.dtype.kind == "b":
        return bool(values[0])
    first = values[0]
    return first.item() if isinstance(first, np.generic) else first


def _changed(arr: np.ndarray) -> np.ndarray:
    """True at k when arr[k] differs from arr[k-1] (k = 0 is False)."""
    out = np.zeros(len(arr), dtype=bool)
    out[1:] = arr[1:] != arr[:-1]
    return out


def valid_mask(pair: PairSeries, cfg: ExtractionConfig) -> np.ndarray:
    has_leader = np.array([lid is not None and lid == lid for lid in pair.leader_id], dtype=bool)
    s = pair.x_lead - pair.x_follower
    with np.errstate(invalid="ignore"):
        ok = (
            has_leader
            & np.isfinite(pair.x_follower)
            & np.isfinite(pair.v)
            & np.isfinite(pair.x_lead)
            & np.isfinite(pair.v_lead)
            & (s >= cfg.min_spacing)
            & (s < cfg.max_spacing)
            & (pair.v >= -cfg.negative_speed_tol)
            & (pair.v_lead >= -cfg.negative_speed_tol)
        )
        if pair.a is not None and cfg.accel_source == "auto":
            ok &= np.isfinite(pair.a)
    return ok


def segment_runs(pair: PairSeries, cfg: ExtractionConfig) -> list[tuple[int, int]]:
    """Maximal runs ``[start, stop)`` of valid samples without any segment boundary."""
    n = len(pair.t)
    if n == 0:
        return []
    ok = valid_mask(pair, cfg)
    brk = np.zeros(n, dtype=bool)
    brk[0] = True
    brk[1:] |= np.abs(np.diff(pair.t) - cfg.dt) > 1e-6
    brk |= _changed(np.asarray(pair.leader_id, dtype=object))
    if pair.segment_key is not None:
        brk |= _changed(np.asarray(pair.segment_key, dtype=object))
    for lane in (pair.lane_follower, pair.lane_leader):
        if lane is not None:
            lane = np.asarray(lane, dtype=np.float64)
            both = np.zeros(n, dtype=bool)
            both[1:] = np.isfinite(lane[1:]) & np.isfinite(lane[:-1])
            brk |= _changed(lane) & both
    runs = []
    start = None
    for k in range(n):
        if not ok[k]:
            if start is not None:
                runs.append((start, k))
                start = None
            continue
        if start is None:
            start = k
        elif brk[k]:
            runs.append((start, k))
            start = k
    if start is not None:
        runs.append((start, n))
    return runs


def _chunks(start: int, stop: int, cfg: ExtractionConfig) -> list[tuple[int, int]]:
    n = stop - start
    if cfg.max_duration is None:
        return [(start, stop)]
    n_max = int(round(cfg.max_duration / cfg.dt))
    if n <= n_max:
        return [(start, stop)]
    parts = np.array_split(np.arange(start, stop), math.ceil(n / n_max))
    return [(int(p[0]), int(p[-1]) + 1) for p in parts]


def extract_events(pair: PairSeries, cfg: ExtractionConfig | None = None, counters: dict | None = None) -> list[Event]:
    """Cut one follower series into raw car-following events.

    ``counters`` carries the running segment index per (follower, leader) pair across
    calls, which keeps event ids unique when a follower is processed in several pieces.
    """
    cfg = cfg or ExtractionConfig()
    counters = {} if counters is None else counters
    events: list[Event] = []
    for run_start, run_stop in segment_runs(pair, cfg):
        if run_stop - run_start < cfg.min_samples:
            continue
        for start, stop in _chunks(run_start, run_stop, cfg):
            sl = slice(start, stop)
            n = stop - start
            v = pair.v[sl].astype(np.float64)
            if n < cfg.min_samples or np.mean(v > cfg.moving_speed_eps) < cfg.min_moving_fraction:
                continue
            v_lead = pair.v_lead[sl].astype(np.float64)
            n_negative = int(np.sum(v < 0.0) + np.sum(v_lead < 0.0))
            v, v_lead = np.clip(v, 0.0, None), np.clip(v_lead, 0.0, None)
            if pair.a is not None and cfg.accel_source == "auto":
                a, a_source = pair.a[sl].astype(np.float64), "dataset"
            else:
                a = savgol_derivative(v, cfg.dt, cfg.savgol_window, cfg.savgol_order)
                a_source = "savgol"
            if cfg.smooth_speeds:
                v = np.clip(savgol_derivative(v, cfg.dt, cfg.savgol_window, cfg.savgol_order, deriv=0), 0.0, None)
                v_lead = np.clip(
                    savgol_derivative(v_lead, cfg.dt, cfg.savgol_window, cfg.savgol_order, deriv=0), 0.0, None
                )
            leader_id = str(pair.leader_id[start])
            key = (pair.follower_id, leader_id)
            k = counters.get(key, 0)
            counters[key] = k + 1
            meta = {
                **pair.meta,
                **{name: _aggregate_extra(values[sl]) for name, values in pair.extras.items()},
                "t0": float(pair.t[start]),
                "a_source": a_source,
                "speeds": "savgol" if cfg.smooth_speeds else "measured",
                "data_version": "raw",
            }
            if n_negative:
                meta["negative_speeds_set_to_zero"] = n_negative
            if pair.lane_follower is not None:
                lane = np.asarray(pair.lane_follower, dtype=np.float64)[start]
                meta["lane"] = int(lane) if np.isfinite(lane) else None
            x_lead = pair.x_lead[sl].astype(np.float64)
            x_follower = pair.x_follower[sl].astype(np.float64)
            events.append(
                Event(
                    event_id=f"{pair.follower_id}|{leader_id.rsplit('/', 1)[-1]}|{k}",
                    dataset=pair.dataset,
                    site=pair.site,
                    follower_id=pair.follower_id,
                    leader_id=leader_id,
                    t=np.arange(n, dtype=np.float64) * cfg.dt,
                    s=x_lead - x_follower,
                    dv=v - v_lead,
                    v=v,
                    a=a,
                    v_lead=v_lead,
                    x_lead=x_lead,
                    x_follower=x_follower,
                    meta=meta,
                )
            )
    return events


def extract_all(pairs: Iterable[PairSeries], cfg: ExtractionConfig | None = None) -> list[Event]:
    cfg = cfg or ExtractionConfig()
    counters: dict = {}
    events: list[Event] = []
    for pair in pairs:
        events.extend(extract_events(pair, cfg, counters))
    return events


# --------------------------------------------------------------------------- tracks
def _check_tracks(tracks: pd.DataFrame) -> None:
    missing = [c for c in TRACK_REQUIRED if c not in tracks.columns]
    if missing:
        raise ValueError(f"tracks frame lacks columns {missing}")


def resample_tracks(tracks: pd.DataFrame, dt: float = DT, max_gap: float | None = None) -> pd.DataFrame:
    """Put every track on the common grid ``t = g * dt`` (integer column ``g``).

    Sources recorded at 10 Hz on that grid are passed through. Other sampling rates
    (highD and exiD 25 Hz, CitySim 30 Hz) are interpolated linearly; ``lane``,
    ``leader`` and other non-float columns take the value of the nearest source sample.
    Grid points inside a hole of the source track longer than ``max_gap`` seconds are
    dropped (default: 1.5 times the median source step of that track).
    """
    _check_tracks(tracks)
    tracks = tracks.sort_values(["group", "vehicle_id", "t"], kind="stable").reset_index(drop=True)
    ratio = tracks["t"].to_numpy(dtype=np.float64) / dt
    same = (tracks["group"].to_numpy()[1:] == tracks["group"].to_numpy()[:-1]) & (
        tracks["vehicle_id"].to_numpy()[1:] == tracks["vehicle_id"].to_numpy()[:-1]
    )
    steps = np.diff(np.round(ratio))[same]
    # a source on the grid but slower than 10 Hz (every second grid point) is interpolated too
    native = len(steps) == 0 or float(np.median(steps)) == 1.0
    if native and np.all(np.abs(ratio - np.round(ratio)) < 1e-6):
        out = tracks.copy()
        out["g"] = np.round(ratio).astype(np.int64)
        return out.drop_duplicates(["group", "vehicle_id", "g"], keep="first").reset_index(drop=True)

    float_cols = [c for c in tracks.columns if c not in ("group", "vehicle_id", "t") and tracks[c].dtype.kind == "f"]
    float_cols = [c for c in float_cols if c not in ("lane", "leader", "length")]
    other_cols = [c for c in tracks.columns if c not in ("group", "vehicle_id", "t", *float_cols)]
    pieces = []
    for (group, vehicle), track in tracks.groupby(["group", "vehicle_id"], sort=False):
        t = track["t"].to_numpy(dtype=np.float64)
        if len(t) < 2:
            continue
        g = np.arange(math.ceil(t[0] / dt - 1e-9), math.floor(t[-1] / dt + 1e-9) + 1, dtype=np.int64)
        tg = g * dt
        right = np.clip(np.searchsorted(t, tg, side="left"), 1, len(t) - 1)
        left = right - 1
        step = t[right] - t[left]
        limit = max_gap if max_gap is not None else 1.5 * float(np.median(np.diff(t)))
        keep = step <= limit + 1e-9
        nearest = np.where(tg - t[left] <= t[right] - tg, left, right)
        piece = {"group": group, "vehicle_id": vehicle, "g": g[keep], "t": tg[keep]}
        for col in float_cols:
            piece[col] = np.interp(tg, t, track[col].to_numpy(dtype=np.float64))[keep]
        for col in other_cols:
            piece[col] = track[col].to_numpy()[nearest][keep]
        pieces.append(pd.DataFrame(piece))
    if not pieces:
        return tracks.iloc[0:0].assign(g=np.array([], dtype=np.int64))
    return pd.concat(pieces, ignore_index=True)


def derive_leaders(frame: pd.DataFrame) -> pd.DataFrame:
    """Add column ``leader``: the next vehicle ahead in the same (group, g, lane).

    ``x_front`` must increase in the driving direction. Used for sources without a
    leader field (TGSIM: lane_kf and xloc_kf within run_index).
    """
    if "lane" not in frame.columns:
        raise ValueError("derive_leaders needs a lane column")
    frame = frame.sort_values(["group", "g", "lane", "x_front"], kind="stable").reset_index(drop=True)
    ahead = frame.groupby(["group", "g", "lane"], sort=False)["vehicle_id"].shift(-1)
    frame["leader"] = ahead
    return frame


def tracks_to_pairs(
    tracks: pd.DataFrame,
    dataset: str,
    dt: float = DT,
    extras: Iterable[str] = (),
    leader_extras: Iterable[str] = (),
    meta: dict | None = None,
) -> Iterator[PairSeries]:
    """Turn a standard tracks frame into one :class:`PairSeries` per follower.

    Required columns: ``group`` (recording or run inside which vehicle ids and time are
    consistent), ``vehicle_id``, ``t`` [s], ``x_front`` and ``x_rear`` [m] (bumper
    positions along the driving direction, increasing downstream), ``v`` [m/s, >= 0].
    Optional: ``site`` (default: the group), ``lane``, ``a``, ``length``, ``leader`` (raw id
    of the leader; missing, 0 or negative when there is none; derived from ``lane`` and
    ``x_front`` when the column is absent). ``extras`` are per-vehicle columns copied to
    the event meta for the follower; ``leader_extras`` likewise for the leader, prefixed
    with ``leader_``.
    """
    frame = resample_tracks(tracks, dt)
    if "leader" not in frame.columns:
        frame = derive_leaders(frame)
    leader = pd.to_numeric(frame["leader"], errors="coerce") if frame["leader"].dtype.kind in "fiu" else frame["leader"]
    if leader.dtype.kind in "fiu":
        frame["leader"] = leader.where(leader > 0)
    has_lane = "lane" in frame.columns
    has_acc = "a" in frame.columns
    extras = list(extras)
    leader_extras = list(dict.fromkeys(["length", *leader_extras])) if "length" in frame.columns else list(leader_extras)

    lead_cols = ["group", "vehicle_id", "g", "x_rear", "v", *(["lane"] if has_lane else []), *leader_extras]
    lead = frame[lead_cols].rename(
        columns={
            "vehicle_id": "leader",
            "x_rear": "x_lead",
            "v": "v_lead",
            "lane": "lane_leader",
            **{c: f"leader_{c}" for c in leader_extras},
        }
    )
    known = frame["leader"].notna()
    merged = frame[known].merge(lead, on=["group", "leader", "g"], how="left")
    frame = pd.concat([merged, frame[~known]], ignore_index=True)
    frame = frame.sort_values(["group", "vehicle_id", "g"], kind="stable")

    for (group, vehicle), track in frame.groupby(["group", "vehicle_id"], sort=False):
        g = track["g"].to_numpy()
        full = np.arange(g[0], g[-1] + 1)
        pos = g - g[0]
        n = len(full)

        def spread(col: str, dtype=np.float64, fill=np.nan) -> np.ndarray:
            out = np.full(n, fill, dtype=dtype)
            out[pos] = track[col].to_numpy()
            return out

        site = str(track["site"].iloc[0]) if "site" in track.columns else str(group)
        prefix = f"{dataset}/{site}/"
        tag = "" if str(group) == site else f"{group}."
        leader_raw = np.full(n, None, dtype=object)
        raw = track["leader"].to_numpy()
        seen = track["x_lead"].notna().to_numpy() if "x_lead" in track.columns else np.zeros(len(track), bool)
        leader_raw[pos[seen]] = [f"{prefix}{tag}{_fmt_id(r)}" for r in raw[seen]]
        pair_meta = {**(meta or {}), "group": str(group)}
        if "length" in track.columns:
            pair_meta["follower_length"] = float(track["length"].iloc[0])
        for col in extras:
            pair_meta[col] = _aggregate_extra(track[col].to_numpy())
        yield PairSeries(
            dataset=dataset,
            site=site,
            follower_id=f"{prefix}{tag}{_fmt_id(vehicle)}",
            t=full * dt,
            x_follower=spread("x_front"),
            v=spread("v"),
            leader_id=leader_raw,
            x_lead=spread("x_lead") if "x_lead" in track.columns else np.full(n, np.nan),
            v_lead=spread("v_lead") if "v_lead" in track.columns else np.full(n, np.nan),
            a=spread("a") if has_acc else None,
            lane_follower=spread("lane") if has_lane else None,
            lane_leader=spread("lane_leader") if has_lane and "lane_leader" in track.columns else None,
            extras={
                f"leader_{c}": spread(f"leader_{c}", dtype=track[f"leader_{c}"].to_numpy().dtype, fill=_fill_of(track[f"leader_{c}"]))
                for c in leader_extras
                if f"leader_{c}" in track.columns
            },
            meta=pair_meta,
        )


def _fmt_id(value) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _fill_of(series: pd.Series):
    kind = series.to_numpy().dtype.kind
    if kind == "f":
        return np.nan
    if kind == "b":
        return False
    if kind in "iu":
        return 0
    return None
