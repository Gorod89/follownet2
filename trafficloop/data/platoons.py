"""Human-driven platoon windows from track data: the leader profiles of protocol P2h.

The recorded followers of the OpenACC profiles of P2 (:mod:`trafficloop.protocols.profiles`) are ACC
vehicles in 17 of the 19 windows. This module cuts platoon windows with human followers out of the
track-based sources NGSIM and TGSIM, from the tracks frames of their loaders
(:func:`trafficloop.data.loaders.ngsim.load_tracks`, :func:`trafficloop.data.loaders.tgsim.load_tracks`:
bumper positions along the driving direction, ``lane``, ``leader``), put on the 10 Hz grid and
given leaders exactly as for the events (:func:`~trafficloop.data.extraction.resample_tracks`,
:func:`~trafficloop.data.extraction.derive_leaders`), and turns them into
:class:`~trafficloop.protocols.profiles.LeaderProfile` objects (docs/protocols.md, section 2, P2h).

Rules:

1. *Chain* (:func:`pair_runs`, :func:`find_chains`): a leader (vehicle 0) and at least
   ``min_followers`` followers in one lane; the leader of follower ``k`` is vehicle ``k - 1`` at
   every sample, present at the same sample in the same lane at a raw net gap in
   ``[MIN_SPACING, MAX_SPACING)`` (the valid samples of the events, 0.5 to 120 m), and no follower
   changes lanes. The followers are the samples that have a leader in the loader's frame (NGSIM:
   autos in the mainline lanes) and are not flagged ``automated`` (TGSIM); any vehicle can lead
   the chain.
2. *Window* (:func:`best_window`): inside the common interval of the chain, the longest stretch
   of at most ``window`` and at least ``min_window`` seconds (``round(T / dt) + 1`` samples for
   ``T`` seconds) whose leader speed is at least ``min_start_speed`` at the first sample and spans
   at least ``min_speed_range`` (a perturbation to follow); among stretches of equal length the
   one with the largest variance of the leader speed, then the earliest. The candidates are cut
   on the speed of the loader.
3. *Selection* per site (:func:`select_windows`): candidates ranked by the number of followers,
   then the duration (then group, start and leader, for determinism); a candidate that shares a
   vehicle with a window already chosen is skipped; at most ``max_per_site`` windows.
4. *Reconstruction* (:func:`reconstruct_chain`): every chosen window goes through the
   reconstruction of the events (:func:`~trafficloop.data.reconstruction.reconstruct_trajectory`,
   docs/reconstruction.md, the settings of the dataset) on the front-bumper positions: the leader
   first, then every follower with the bound ``x_front <= x_front_prev - L_prev - s_min`` against
   its reconstructed predecessor and the rigid shift of
   :func:`~trafficloop.data.reconstruction.reconstruct_event` where the solver misses the bound by
   its tolerance. Rule 2 is checked again on the reconstructed leader speed; a window that fails
   it, or whose solver fails, is replaced by the next candidate.

A profile holds the reconstructed leader speed ``v``; per follower ``k`` its speed and its net gap
``s_k = x_front[k-1] - L[k-1] - x_front[k]`` (rear bumper of the predecessor minus front bumper, the
gap of the events), ``driver = "Human"``; ``t0`` is the time of the first sample in the recording
(NGSIM: since the first time stamp of the location, TGSIM: since the start of the run);
``vehicle_length`` is the mean length of the chain, leader included (lengths are constant per
vehicle in both sources; the median over the window is taken). ``source`` is
``<dataset>/<site>/<group>/lane<lane>/leader<id>/frame<frame>``, the frame of the first sample being
NGSIM's ``frame_id`` of the original file or the 0.1 s step of the TGSIM run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np
import pandas as pd

from trafficloop.data.extraction import ExtractionConfig, _fmt_id, derive_leaders, resample_tracks
from trafficloop.data.reconstruction import ReconstructionConfig, reconstruct_trajectory
from trafficloop.data.schema import DT
from trafficloop.protocols.profiles import MIN_FOLLOWERS, WINDOW, LeaderProfile, RecordedFollower

CACHE = "data/scenarios/human_profiles.json"  # written by scripts/build_human_profiles.py
MIN_WINDOW = 60.0  # s, shortest window
MIN_START_SPEED = 3.0  # m/s, leader speed at the first sample (the hold of the platoon runs at it)
MIN_SPEED_RANGE = 3.0  # m/s, max - min of the leader speed over the window
MAX_PER_SITE = 20  # windows kept per site
#: m, the net gap of every follower sample lies in [MIN_SPACING, MAX_SPACING), as in the events
MIN_SPACING = ExtractionConfig.min_spacing
MAX_SPACING = ExtractionConfig.max_spacing
HUMAN = "Human"
#: frame column flagging automated vehicles (TGSIM); flagged samples are no followers
AUTOMATED = "automated"

T = TypeVar("T")


def samples(seconds: float, dt: float = DT) -> int:
    """Samples of a window of ``seconds`` (both ends included, as the OpenACC windows)."""
    return int(round(seconds / dt)) + 1


@dataclass(frozen=True)
class Chain:
    """Vehicles of one lane in platoon order (leader first) with their common interval
    ``[start, stop)`` of grid indices."""

    vehicles: tuple
    lane: float
    start: int
    stop: int

    @property
    def n_followers(self) -> int:
        return len(self.vehicles) - 1


@dataclass(frozen=True)
class Candidate:
    """A chain of group ``group`` (its rank among the groups of the site) with its window
    ``[start, start + n)`` of grid indices (rule 2)."""

    group: int
    chain: Chain
    start: int
    n: int

    @property
    def n_followers(self) -> int:
        return self.chain.n_followers

    @property
    def duration(self) -> float:
        return (self.n - 1) * DT

    def rank(self) -> tuple:
        """Sort key of rule 3: more followers first, then longer windows."""
        return (-self.n_followers, -self.n, self.group, self.start, str(self.chain.vehicles[0]))


# ----------------------------------------------------------------------------- chains
def grid_frame(tracks: pd.DataFrame, dt: float = DT) -> pd.DataFrame:
    """A loader's tracks frame on the 10 Hz grid (column ``g``) with leaders, as
    :func:`~trafficloop.data.extraction.tracks_to_pairs` prepares it."""
    frame = resample_tracks(tracks, dt)
    return derive_leaders(frame) if "leader" not in frame.columns else frame


def _leader_codes(ids: np.ndarray, leader: pd.Series) -> np.ndarray:
    """Index of every sample's leader in the sorted ``ids``; -1 without one (missing, unknown, or
    not positive for numeric ids, as in ``tracks_to_pairs``)."""
    out = np.full(len(leader), -1, dtype=np.int64)
    if ids.dtype.kind in "iuf":
        values = pd.to_numeric(leader, errors="coerce").to_numpy(dtype=np.float64)
        known = np.isfinite(values) & (values > 0)
        index = pd.Index(ids.astype(np.float64))
    else:
        values = leader.to_numpy()
        known = leader.notna().to_numpy()
        index = pd.Index(ids)
    out[known] = index.get_indexer(values[known])
    return out


def pair_runs(
    frame: pd.DataFrame,
    min_samples: int = 1,
    automated: str | None = AUTOMATED,
    min_spacing: float = MIN_SPACING,
    max_spacing: float = MAX_SPACING,
) -> pd.DataFrame:
    """Maximal runs of samples in which a vehicle follows one leader in its own lane.

    ``frame`` is one group on the grid (:func:`grid_frame`: ``vehicle_id``, ``g``, ``x_front``,
    ``x_rear``, ``v``, ``lane``, ``leader``). A sample belongs to a run when its leader is present at
    the same grid index in the same lane, the net gap ``x_rear(leader) - x_front`` lies in
    ``[min_spacing, max_spacing)`` (the valid samples of the events), position and speed are finite
    and the vehicle is not flagged in the column ``automated`` (when the frame has it); a run ends
    where the leader or the lane changes, a grid index is missing or a sample does not belong.
    Returns ``follower, leader, lane, start, stop`` (grid indices ``[start, stop)``) of the runs of
    at least ``min_samples`` samples, sorted by follower and start.
    """
    columns = ["follower", "leader", "lane", "start", "stop"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    ids, codes = np.unique(frame["vehicle_id"].to_numpy(), return_inverse=True)
    g = frame["g"].to_numpy(dtype=np.int64)
    order = np.lexsort((g, codes))
    codes, g = codes[order].astype(np.int64), g[order]
    lane = frame["lane"].to_numpy(dtype=np.float64)[order]
    x_front = frame["x_front"].to_numpy(dtype=np.float64)[order]
    x_rear = frame["x_rear"].to_numpy(dtype=np.float64)[order]
    lead = _leader_codes(ids, frame["leader"])[order]
    member = np.isfinite(x_front) & np.isfinite(frame["v"].to_numpy(dtype=np.float64)[order])
    if automated is not None and automated in frame.columns:
        member &= ~frame[automated].fillna(False).to_numpy(dtype=bool)[order]
    # the leader's sample at the same grid index: keys are sorted as (vehicle, g)
    g0 = int(g.min())
    span = int(g.max()) - g0 + 1
    key = codes * span + (g - g0)
    want = lead * span + (g - g0)
    pos = np.minimum(np.searchsorted(key, want), len(key) - 1)
    found = (lead >= 0) & (key[pos] == want)
    lane_lead = np.where(found, lane[pos], np.nan)
    gap = np.where(found, x_rear[pos], np.nan) - x_front
    with np.errstate(invalid="ignore"):
        member &= found & np.isfinite(lane) & (lane_lead == lane) & (gap >= min_spacing) & (gap < max_spacing)
    new = np.ones(len(g), dtype=bool)
    new[1:] = (
        (codes[1:] != codes[:-1])
        | (g[1:] != g[:-1] + 1)
        | (lead[1:] != lead[:-1])
        | (lane[1:] != lane[:-1])
        | (member[1:] != member[:-1])
    )
    first = np.flatnonzero(new)
    end = np.append(first[1:], len(g))
    keep = member[first] & (end - first >= min_samples)
    first, end = first[keep], end[keep]
    return pd.DataFrame(
        {
            "follower": ids[codes[first]],
            "leader": ids[lead[first]],
            "lane": lane[first],
            "start": g[first],
            "stop": g[end - 1] + 1,
        },
        columns=columns,
    )


def find_chains(runs: pd.DataFrame, min_followers: int = MIN_FOLLOWERS, min_samples: int = 2) -> list[Chain]:
    """Every chain of at least ``min_followers`` followers built from ``runs`` (:func:`pair_runs`)
    whose common interval holds at least ``min_samples`` samples, for every vehicle as the chain
    leader: chain ``(c_0, ..., c_m)`` needs a run of ``c_k`` behind ``c_(k-1)`` for every ``k``,
    all in one lane, and its interval is the intersection of theirs. The prefixes of a chain with
    at least ``min_followers`` followers are chains too."""
    by_leader: dict = {}
    for run in runs.itertuples(index=False):
        by_leader.setdefault(run.leader, []).append(run)
    chains: list[Chain] = []

    def extend(path: list, lane: float, start: int, stop: int) -> None:
        if len(path) - 1 >= min_followers:
            chains.append(Chain(tuple(path), lane, start, stop))
        for run in by_leader.get(path[-1], ()):
            lo, hi = max(start, int(run.start)), min(stop, int(run.stop))
            if run.lane == lane and hi - lo >= min_samples and run.follower not in path:
                extend([*path, run.follower], lane, lo, hi)

    for run in runs.itertuples(index=False):  # every run is the first link of the chains of its leader
        if int(run.stop) - int(run.start) >= min_samples:
            extend([run.leader, run.follower], float(run.lane), int(run.start), int(run.stop))
    return chains


def best_window(
    v: np.ndarray,
    min_samples: int,
    max_samples: int,
    min_start_speed: float = MIN_START_SPEED,
    min_range: float = MIN_SPEED_RANGE,
) -> tuple[int, int] | None:
    """Start and number of samples of the window of rule 2 inside the leader speed ``v`` of a
    chain's common interval; ``None`` without one."""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n < min_samples or max_samples < min_samples:
        return None
    starts = np.arange(n - min_samples + 1)
    length = np.minimum(n - starts, max_samples)
    high = np.maximum.accumulate(v[::-1])[::-1][starts]  # windows that run to the end of v
    low = np.minimum.accumulate(v[::-1])[::-1][starts]
    full = n - starts > max_samples
    if full.any():
        rolling = pd.Series(v).rolling(max_samples)
        top = rolling.max().to_numpy()[max_samples - 1 :]  # top[k] = max(v[k : k + max_samples])
        bottom = rolling.min().to_numpy()[max_samples - 1 :]
        high[full], low[full] = top[starts[full]], bottom[starts[full]]
    ok = (v[starts] >= min_start_speed) & (high - low >= min_range)
    if not ok.any():
        return None
    c1 = np.concatenate([[0.0], np.cumsum(v)])
    c2 = np.concatenate([[0.0], np.cumsum(v * v)])
    end = starts + length
    mean = (c1[end] - c1[starts]) / length
    var = (c2[end] - c2[starts]) / length - mean * mean
    index = np.flatnonzero(ok)
    best = index[np.lexsort((starts[index], -var[index], -length[index]))[0]]
    return int(starts[best]), int(length[best])


class Tracks:
    """The samples of one group on the grid, sorted by vehicle and grid index, as arrays:
    ``x_front``, ``v``, ``length`` and ``frame`` (NGSIM ``frame_id``, otherwise the grid index)."""

    def __init__(self, frame: pd.DataFrame, group, site) -> None:
        self.group, self.site = str(group), str(site)
        ids, codes = np.unique(frame["vehicle_id"].to_numpy(), return_inverse=True)
        g = frame["g"].to_numpy(dtype=np.int64)
        order = np.lexsort((g, codes))
        self.ids, self.g = ids, g[order]
        codes = codes[order]
        self.first = np.searchsorted(codes, np.arange(len(ids)), side="left")
        self.stop = np.searchsorted(codes, np.arange(len(ids)), side="right")
        self.code = {vehicle: k for k, vehicle in enumerate(ids)}
        length = frame["length"] if "length" in frame.columns else frame["x_front"] - frame["x_rear"]
        self.columns = {
            "x_front": frame["x_front"].to_numpy(dtype=np.float64)[order],
            "v": frame["v"].to_numpy(dtype=np.float64)[order],
            "length": length.to_numpy(dtype=np.float64)[order],
            "frame": (frame["frame_id"].to_numpy(dtype=np.int64) if "frame_id" in frame.columns else g)[order],
        }
        # per vehicle: the class of its first sample, automated when any sample is flagged
        self.vehicle_class = (
            frame["vehicle_class"].to_numpy()[order][self.first] if "vehicle_class" in frame.columns else np.full(len(ids), None)
        )
        flags = frame[AUTOMATED].fillna(False).to_numpy(dtype=bool)[order] if AUTOMATED in frame.columns else None
        self.automated = (
            np.logical_or.reduceat(flags, self.first) if flags is not None and len(ids) else np.zeros(len(ids), dtype=bool)
        )

    def series(self, vehicle, start: int, n: int, column: str) -> np.ndarray:
        """``column`` of ``vehicle`` over the grid indices ``start .. start + n - 1`` (all present)."""
        k = self.code[vehicle]
        lo, hi = int(self.first[k]), int(self.stop[k])
        i = lo + int(np.searchsorted(self.g[lo:hi], start))
        if i + n > hi or self.g[i] != start or self.g[i + n - 1] != start + n - 1:
            raise ValueError(f"vehicle {vehicle} of group {self.group} is not present at every index {start}..{start + n - 1}")
        return self.columns[column][i : i + n]


def group_candidates(
    tracks: Tracks,
    runs: pd.DataFrame,
    group: int,
    min_followers: int = MIN_FOLLOWERS,
    window: float = WINDOW,
    min_window: float = MIN_WINDOW,
    min_start_speed: float = MIN_START_SPEED,
    min_speed_range: float = MIN_SPEED_RANGE,
) -> tuple[list[Candidate], int]:
    """The candidates of rules 1 and 2 of one group (rank ``group`` within its site) and the number
    of chains found."""
    n_min, n_max = samples(min_window), samples(window)
    chains = find_chains(runs, min_followers, n_min)
    out = []
    for chain in chains:
        lead = tracks.series(chain.vehicles[0], chain.start, chain.stop - chain.start, "v")
        found = best_window(lead, n_min, n_max, min_start_speed, min_speed_range)
        if found is not None:
            out.append(Candidate(group, chain, chain.start + found[0], found[1]))
    return out, len(chains)


def select_windows(
    candidates: Iterable[Candidate], build: Callable[[Candidate], T | None], max_windows: int = MAX_PER_SITE
) -> tuple[list[T], int]:
    """Rule 3: the candidates in the order of :meth:`Candidate.rank`, each skipped when it shares a
    vehicle (of the same group) with a window already chosen; ``build`` turns a candidate into the
    window or returns ``None`` to reject it (rule 4). Returns the windows and the number of
    rejected candidates."""
    used: set = set()
    chosen, rejected = [], 0
    for candidate in sorted(candidates, key=Candidate.rank):
        if len(chosen) >= max_windows:
            break
        keys = {(candidate.group, vehicle) for vehicle in candidate.chain.vehicles}
        if keys & used:
            continue
        item = build(candidate)
        if item is None:
            rejected += 1
            continue
        chosen.append(item)
        used |= keys
    return chosen, rejected


# ----------------------------------------------------------------------------- reconstruction
def reconstruct_chain(
    x_front: Sequence[np.ndarray], v: Sequence[np.ndarray], lengths: Sequence[float], cfg: ReconstructionConfig
) -> dict | None:
    """Rule 4 on the measured front-bumper positions and speeds of a chain (leader first) with
    constant vehicle lengths. Returns ``x`` (front bumpers), ``v``, ``a`` ``[V, n]``, the gaps ``s``
    ``[V - 1, n]`` and the diagnostics per vehicle; ``None`` when a solver fails."""
    xs, vs, acc, gaps, info = [], [], [], [], []
    for k, (x_obs, v_obs) in enumerate(zip(x_front, v)):
        x_obs = np.asarray(x_obs, dtype=np.float64)
        upper = None if k == 0 else xs[-1] - lengths[k - 1] - cfg.s_min
        out = reconstruct_trajectory(x_obs, np.asarray(v_obs, dtype=np.float64), cfg, x_upper=upper)
        if not out["ok"]:
            return None
        x = out["x"]
        if k:
            short = cfg.s_min - float(np.min(xs[-1] - lengths[k - 1] - x))
            if short > 0.0:  # the solver meets the bound to its tolerance only
                x = x - short
            gaps.append(xs[-1] - lengths[k - 1] - x)
        jerk = np.diff(out["a"][:-1]) / cfg.dt
        info.append(
            {
                "status": out["status"],
                "outlier_share": out["outliers"],
                "rms_position_change": float(np.sqrt(np.mean((x - x_obs) ** 2))),
                "max_abs_acc": float(np.max(np.abs(out["a"]))),
                "max_abs_jerk": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
            }
        )
        xs.append(x)
        vs.append(out["v"])
        acc.append(out["a"])
    return {"x": np.stack(xs), "v": np.stack(vs), "a": np.stack(acc), "s": np.stack(gaps), "vehicles": info}


def _fmt_lane(lane: float) -> str:
    return str(int(lane)) if float(lane).is_integer() else str(lane)


def window_profile(
    tracks: Tracks,
    candidate: Candidate,
    dataset: str,
    cfg: ReconstructionConfig,
    min_start_speed: float = MIN_START_SPEED,
    min_speed_range: float = MIN_SPEED_RANGE,
) -> tuple[LeaderProfile, dict] | None:
    """Rule 4 for one candidate of ``tracks``: the profile and its record (``meta["windows"]`` of
    the cache), ``None`` when a solver fails or the reconstructed leader breaks rule 2."""
    chain, start, n = candidate.chain, candidate.start, candidate.n
    x_obs = [tracks.series(vehicle, start, n, "x_front") for vehicle in chain.vehicles]
    v_obs = [tracks.series(vehicle, start, n, "v") for vehicle in chain.vehicles]
    lengths = [float(np.median(tracks.series(vehicle, start, n, "length"))) for vehicle in chain.vehicles]
    rec = reconstruct_chain(x_obs, v_obs, lengths, cfg)
    if rec is None:
        return None
    lead = rec["v"][0]
    if lead[0] < min_start_speed or lead.max() - lead.min() < min_speed_range:
        return None
    frame = int(tracks.series(chain.vehicles[0], start, 1, "frame")[0])
    lane, leader = _fmt_lane(chain.lane), _fmt_id(chain.vehicles[0])
    tag = tracks.group if tracks.group.startswith(tracks.site) else f"{tracks.site}-{tracks.group}"
    name = f"{dataset}_{tag}_lane{lane}_v{leader}_f{frame}"
    source = f"{dataset}/{tracks.site}/{tracks.group}/lane{lane}/leader{leader}/frame{frame}"
    followers = [RecordedFollower(k, rec["v"][k], rec["s"][k - 1], HUMAN) for k in range(1, len(chain.vehicles))]
    profile = LeaderProfile(name, source, lead, round(start * DT, 6), followers, float(np.mean(lengths)))
    raw_gaps = np.stack([x_obs[k - 1] - lengths[k - 1] - x_obs[k] for k in range(1, len(chain.vehicles))])
    codes = [tracks.code[vehicle] for vehicle in chain.vehicles]
    record = {
        "name": name,
        "source": source,
        "dataset": dataset,
        "site": tracks.site,
        "group": tracks.group,
        "lane": float(chain.lane),
        "frame": frame,
        "t0": profile.t0,
        "duration": round(candidate.duration, 6),
        "n_followers": len(followers),
        "vehicles": [_fmt_id(vehicle) for vehicle in chain.vehicles],
        "classes": [None if tracks.vehicle_class[c] is None else str(tracks.vehicle_class[c]) for c in codes],
        "leader_automated": bool(tracks.automated[codes[0]]),
        "lengths": [round(length, 4) for length in lengths],
        "vehicle_length": profile.vehicle_length,
        "leader_speed": {
            "start": float(lead[0]),
            "min": float(lead.min()),
            "max": float(lead.max()),
            "range": float(lead.max() - lead.min()),
            "range_raw": float(v_obs[0].max() - v_obs[0].min()),
        },
        "follower_speed": {"min": float(rec["v"][1:].min()), "max": float(rec["v"][1:].max())},
        "gap": {"min": float(rec["s"].min()), "median": float(np.median(rec["s"])), "max": float(rec["s"].max()),
                "min_raw": float(raw_gaps.min()), "max_raw": float(raw_gaps.max())},
        "reconstruction": {
            "status": sorted({item["status"] for item in rec["vehicles"]}),
            "max_outlier_share": max(item["outlier_share"] for item in rec["vehicles"]),
            "max_rms_position_change": max(item["rms_position_change"] for item in rec["vehicles"]),
            "max_abs_acc": max(item["max_abs_acc"] for item in rec["vehicles"]),
            "max_abs_jerk": max(item["max_abs_jerk"] for item in rec["vehicles"]),
        },
    }
    return profile, record


# ----------------------------------------------------------------------------- sources
def site_windows(
    groups: Sequence[Tracks],
    runs: Sequence[pd.DataFrame],
    dataset: str,
    cfg: ReconstructionConfig,
    min_followers: int = MIN_FOLLOWERS,
    max_windows: int = MAX_PER_SITE,
    window: float = WINDOW,
    min_window: float = MIN_WINDOW,
    min_start_speed: float = MIN_START_SPEED,
    min_speed_range: float = MIN_SPEED_RANGE,
) -> tuple[list[LeaderProfile], list[dict], dict]:
    """Rules 1-4 over the groups of one site (``runs[i]`` the pair runs of ``groups[i]``): the
    profiles and records of the chosen windows in the order of their group and start, and the
    counts (chains, candidates, rejected after the reconstruction, chosen)."""
    candidates, n_chains = [], 0
    for index, (tracks, group_runs) in enumerate(zip(groups, runs)):
        found, n = group_candidates(tracks, group_runs, index, min_followers, window, min_window, min_start_speed,
                                    min_speed_range)
        candidates += found
        n_chains += n

    def build(candidate: Candidate):
        made = window_profile(groups[candidate.group], candidate, dataset, cfg, min_start_speed, min_speed_range)
        return None if made is None else (candidate, *made)

    chosen, rejected = select_windows(candidates, build, max_windows)
    chosen.sort(key=lambda item: (item[0].group, item[0].start, item[0].chain.lane))
    counts = {"chains": n_chains, "candidates": len(candidates), "rejected": rejected, "windows": len(chosen)}
    return [profile for _, profile, _ in chosen], [record for _, _, record in chosen], counts


def track_profiles(
    frames: Iterable[tuple[pd.DataFrame, dict]],
    dataset: str,
    cfg: ReconstructionConfig,
    min_followers: int = MIN_FOLLOWERS,
    max_per_site: int = MAX_PER_SITE,
    window: float = WINDOW,
    min_window: float = MIN_WINDOW,
    min_start_speed: float = MIN_START_SPEED,
    min_speed_range: float = MIN_SPEED_RANGE,
) -> tuple[list[LeaderProfile], list[dict], dict[str, dict]]:
    """Platoon windows of a track source: ``frames`` are the ``(tracks frame, meta)`` items of a
    loader's ``load_tracks`` (one group each, the groups of a site consecutive). Returns the
    profiles and records site by site and the counts per site (``groups`` and the counts of
    :func:`site_windows`)."""
    profiles: list[LeaderProfile] = []
    records: list[dict] = []
    counts: dict[str, dict] = {}
    site, groups, runs = None, [], []

    def flush() -> None:
        if site is None:
            return
        found, rows, count = site_windows(groups, runs, dataset, cfg, min_followers, max_per_site, window, min_window,
                                          min_start_speed, min_speed_range)
        profiles.extend(found)
        records.extend(rows)
        counts[site] = {"groups": len(groups), **count}

    for frame, _ in frames:
        if frame.empty:
            continue
        frame = grid_frame(frame)
        frame_site = str(frame["site"].iloc[0]) if "site" in frame.columns else str(frame["group"].iloc[0])
        if frame_site != site:
            flush()
            site, groups, runs = frame_site, [], []
        groups.append(Tracks(frame, frame["group"].iloc[0], frame_site))
        runs.append(pair_runs(frame, samples(min_window)))
    flush()
    return profiles, records, counts
