"""Leader profiles of protocol P2: OpenACC platoon windows and the synthetic braking pulse.

A :class:`LeaderProfile` is the manoeuvre speed series of a platoon leader (the simulator adds the
hold phase) with the recorded followers of the same window, the empirical reference of the
concave-growth and hysteresis metrics (docs/protocols.md, section 2).

OpenACC (layout in :mod:`trafficloop.data.loaders.openacc`, 10 Hz): the ten AstaZero platoon runs
(``AstaZero/ASta_*.csv``, five vehicles, 13-28 m/s) and the ZalaZone dynamic-platform runs
(``ZalaZone/dynamic_part*.csv``, up to ten ACC followers, perturbations at 4-11 m/s; no ``Driver``
columns, absent vehicles have no columns). From every run the window of ``window`` seconds, among
the starts every 10 s from the first sample, with the largest variance of the leader speed
``Speed1``, among the windows where the leader and at least four followers are valid throughout
(speed above 0.5 m/s, finite ``IVS``); the recorded followers of the window are the vehicles behind
the leader up to the first one that is not valid throughout (the four AstaZero followers are
always all valid). A window of ``window`` seconds holds ``round(window / dt) + 1`` samples. Of the
ZalaZone runs the ten windows with the largest leader-speed variance are kept. Follower ``j``
(vehicle ``j + 1`` of the file) has the speed ``Speed<j+1>``, the gap ``IVS<j>`` and the driver
tag of ``Driver<j+1>`` (the file flag ``ACC`` when the column is absent). The vehicle length is the
mean over the platoon of the lengths in the campaign's ``*_veh_specifications.csv`` when that file
has a length column (no published campaign has one), otherwise ``VEHICLE_LENGTH``.

Profiles are cached as JSON (arrays as lists, floats written with their shortest exact
representation) by :func:`save_profiles` and read back by :func:`load_profiles`.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from trafficloop.data.extraction import to_grid
from trafficloop.data.loaders.openacc import CAMPAIGNS, DEFAULT_RAW_DIR, MODE_CODES, read_run_info
from trafficloop.data.schema import DT
from trafficloop.protocols.scenarios import VEHICLE_LENGTH
from trafficloop.utils import file_sha256, read_json, resolve_path

#: ``LeaderProfile.source`` of the recorded profiles starts with this tag, the pulse has SYNTHETIC
OPENACC = "openacc"
SYNTHETIC = "synthetic"
WINDOW = 300.0  # s
WINDOW_STEP = 10.0  # s between candidate window starts
MIN_SPEED = 0.5  # m/s, every vehicle of the window above it
MIN_FOLLOWERS = 4  # recorded followers a window needs
CACHE = "data/scenarios/openacc_profiles.json"
_COLUMN = re.compile(r"(Speed|IVS|Driver)(\d+)")


@dataclass(frozen=True)
class ProfileCampaign:
    """Recorded runs that give leader profiles: one window per run; ``keep`` limits the campaign to
    the windows with the largest leader-speed variance (``None`` keeps all)."""

    folder: str
    pattern: str
    keep: int | None = None


PROFILE_CAMPAIGNS = (ProfileCampaign("AstaZero", "ASta_*.csv"), ProfileCampaign("ZalaZone", "dynamic_part*.csv", keep=10))


@dataclass
class RecordedFollower:
    index: int  # 1 = first follower
    v: np.ndarray  # speed over the window [m/s]
    s: np.ndarray  # gap to the vehicle in front [m]
    driver: str  # "ACC" or "Human"

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.v = np.ascontiguousarray(self.v, dtype=np.float64)
        self.s = np.ascontiguousarray(self.s, dtype=np.float64)


@dataclass
class LeaderProfile:
    name: str
    # "openacc/<campaign>/<file>" for the OpenACC windows, "<dataset>/<site>/..." for the human platoon
    # windows of the track sources (trafficloop.data.platoons), "synthetic" for the pulse
    source: str
    v: np.ndarray  # leader speed of the manoeuvre [m/s], 10 Hz, without the hold
    t0: float  # window start in the time of the source file [s] (0 for synthetic profiles)
    followers: list[RecordedFollower] = field(default_factory=list)
    vehicle_length: float = VEHICLE_LENGTH

    def __post_init__(self) -> None:
        self.v = np.ascontiguousarray(self.v, dtype=np.float64)
        self.t0 = float(self.t0)
        self.vehicle_length = float(self.vehicle_length)

    @property
    def is_openacc(self) -> bool:
        return self.source.split("/", 1)[0] == OPENACC

    @property
    def is_recorded(self) -> bool:
        """A recorded window (OpenACC or a track source), not a synthetic profile."""
        return self.source.split("/", 1)[0] != SYNTHETIC


# ----------------------------------------------------------------------------- synthetic pulse
def _ramp(v_from: float, v_to: float, rate: float, dt: float = DT) -> np.ndarray:
    """Samples after ``v_from`` at the constant rate ``rate`` until ``v_to``, which is hit exactly."""
    delta = v_to - v_from
    if delta == 0.0:
        return np.empty(0)
    if rate == 0.0 or math.copysign(1.0, rate) != math.copysign(1.0, delta):
        raise ValueError(f"a rate of {rate} m/s^2 does not lead from {v_from} to {v_to} m/s")
    n = int(math.ceil(abs(delta) / (abs(rate) * dt) - 1e-9))
    out = v_from + rate * dt * np.arange(1, n + 1, dtype=np.float64)
    out[-1] = v_to
    return out


def braking_pulse(
    v0: float = 25.0,
    v_low: float = 10.0,
    decel: float = -3.0,
    accel: float = 1.5,
    hold_low: float = 10.0,
    tail: float = 240.0,
) -> LeaderProfile:
    """Synthetic pulse: from ``v0`` at ``decel`` to ``v_low``, ``hold_low`` seconds there, back to
    ``v0`` at ``accel``, ``tail`` seconds at ``v0`` (the simulator prepends the hold). The tail of
    240 s lets the wave (about 2 s per vehicle) run through a platoon of 100 within the window."""
    if not 0.0 <= v_low < v0:
        raise ValueError(f"the pulse needs 0 <= v_low < v0, got {v_low} and {v0}")
    v = np.concatenate(
        [
            [float(v0)],
            _ramp(v0, v_low, decel),
            np.full(int(round(hold_low / DT)), float(v_low)),
            _ramp(v_low, v0, accel),
            np.full(int(round(tail / DT)), float(v0)),
        ]
    )
    return LeaderProfile("braking_pulse", SYNTHETIC, v, 0.0, [], VEHICLE_LENGTH)


# ----------------------------------------------------------------------------- OpenACC windows
def _name_key(name: str) -> str:
    """``Audi (A8)`` and ``Audi(A8)`` -> ``audia8`` (spellings of the specification and run files)."""
    return re.sub(r"[^0-9a-z]", "", name.lower())


def campaign_vehicle_length(folder: str | Path, vehicles: Sequence[str]) -> float | None:
    """Mean length [m] of ``vehicles`` from the campaign's specification file, ``None`` without one."""
    for path in sorted(Path(folder).glob("*veh_specifications.csv")):
        table = pd.read_csv(path, encoding="utf-8-sig")
        column = next((c for c in table.columns if "length" in c.lower()), None)
        if column is None:
            continue
        unit = column.lower()
        scale = 1e-3 if "mm" in unit else 1e-2 if "cm" in unit else 1.0
        lengths = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64) * scale
        known = {
            _name_key(str(name)): length
            for name, length in zip(table.iloc[:, 0], lengths)
            if np.isfinite(length) and length > 0
        }
        chosen = [known[_name_key(v)] for v in vehicles if _name_key(v) in known]
        if chosen:
            return float(np.mean(chosen))
    return None


def _read_run(path: Path):
    """Run metadata, the time grid and the columns ``Speed<k>``, ``IVS<k>``, ``Driver<k>`` (codes)."""
    info = read_run_info(path)
    if info is None:
        return None
    frame = pd.read_csv(
        path,
        skiprows=info.header_line,
        encoding="utf-8-sig",
        usecols=lambda column: column == "Time" or _COLUMN.fullmatch(column) is not None,
    )
    columns = {}
    for name in frame.columns:
        if name.startswith("Driver"):
            text = frame[name].astype("string").str.strip().str.lower()
            columns[name] = text.map(MODE_CODES).to_numpy(dtype=np.float64, na_value=np.nan)
        elif name != "Time":
            columns[name] = frame[name].to_numpy(dtype=np.float64)
    drivers = [name for name in columns if name.startswith("Driver")]
    t, grid = to_grid(frame["Time"].to_numpy(dtype=np.float64), columns, nearest=drivers)
    return info, t, grid


def best_window(lead: np.ndarray, valid: np.ndarray, n_window: int, stride: int, min_vehicles: int) -> tuple[int, int] | None:
    """The window of ``n_window`` samples (start a multiple of ``stride``) with the largest variance
    of ``lead`` among those where at least the first ``min_vehicles`` rows of ``valid`` ``[V, T]``
    (vehicles in platoon order) are valid throughout. Returns its start (the earliest on ties) and
    the number of leading vehicles valid throughout it; ``None`` without such a window."""
    n_veh, n_steps = valid.shape
    starts = np.arange(0, n_steps - n_window + 1, stride)
    if len(starts) == 0:
        return None
    invalid = np.concatenate([np.zeros((n_veh, 1), dtype=np.int64), np.cumsum(~valid, axis=1)], axis=1)
    whole = invalid[:, starts + n_window] == invalid[:, starts]  # [V, starts]: valid throughout
    chain = np.cumprod(whole, axis=0).sum(axis=0)  # leading vehicles valid throughout
    keep = chain >= min_vehicles
    if not keep.any():
        return None
    starts, chain = starts[keep], chain[keep]
    variance = np.array([np.var(lead[i : i + n_window]) for i in starts])
    best = int(np.argmax(variance))
    return int(starts[best]), int(chain[best])


def _driver_tag(codes: np.ndarray | None, acc_flag: str) -> str:
    if codes is not None:
        known = codes[np.isfinite(codes)]
        if len(known):
            return "ACC" if np.mean(known == MODE_CODES["acc"]) >= 0.5 else "Human"
    return {"1": "ACC", "0": "Human"}.get(acc_flag, "unknown")


def run_profile(path: Path, campaign: str, window: float = WINDOW) -> LeaderProfile | None:
    """The profile of one run (module docstring); ``None`` with a warning when it has no window."""
    run = _read_run(path)
    if run is None:
        return None
    info, t, grid = run
    n_veh = 0  # vehicles 1..n_veh with a speed column and the gap to the vehicle in front
    while f"Speed{n_veh + 1}" in grid and (n_veh == 0 or f"IVS{n_veh}" in grid):
        n_veh += 1
    if n_veh < MIN_FOLLOWERS + 1:
        warnings.warn(f"{path.name}: {max(n_veh - 1, 0)} followers with Speed and IVS columns, fewer than {MIN_FOLLOWERS}")
        return None
    factor = CAMPAIGNS[campaign].speed_factor
    speeds = np.stack([grid[f"Speed{k}"] * factor for k in range(1, n_veh + 1)])
    gaps = np.stack([grid[f"IVS{k}"] for k in range(1, n_veh)])
    valid = speeds > MIN_SPEED
    valid[1:] &= np.isfinite(gaps)
    n_window = int(round(window / DT)) + 1
    found = best_window(speeds[0], valid, n_window, int(round(WINDOW_STEP / DT)), MIN_FOLLOWERS + 1)
    if found is None:
        warnings.warn(f"{path.name}: no window of {window} s with the leader and {MIN_FOLLOWERS} followers valid")
        return None
    i0, n_valid = found
    sl = slice(i0, i0 + n_window)
    followers = []
    for j in range(1, n_valid):
        codes = grid.get(f"Driver{j + 1}")
        tag = _driver_tag(None if codes is None else codes[sl], info.acc_flag)
        followers.append(RecordedFollower(j, speeds[j, sl], gaps[j - 1, sl], tag))
    length = campaign_vehicle_length(path.parent, info.vehicles)
    return LeaderProfile(
        name=path.stem,
        source=f"{OPENACC}/{campaign}/{path.name}",
        v=speeds[0, sl],
        t0=round(float(t[i0]), 6),
        followers=followers,
        vehicle_length=VEHICLE_LENGTH if length is None else length,
    )


def openacc_profiles(raw_dir: str | Path | None = None, window: float = WINDOW) -> list[LeaderProfile]:
    """The profiles of :data:`PROFILE_CAMPAIGNS` in campaign and file order (module docstring);
    runs without a window and missing campaign folders are skipped with a warning."""
    root = resolve_path(DEFAULT_RAW_DIR if raw_dir is None else raw_dir)
    profiles, found = [], False
    for campaign in PROFILE_CAMPAIGNS:
        paths = sorted((root / campaign.folder).glob(campaign.pattern))
        if not paths:
            warnings.warn(f"no {campaign.folder}/{campaign.pattern} under {root}")
            continue
        found = True
        runs = [p for p in (run_profile(path, campaign.folder, window) for path in paths) if p is not None]
        if campaign.keep is not None and len(runs) > campaign.keep:
            ranked = sorted(range(len(runs)), key=lambda i: -float(np.var(runs[i].v)))  # stable: earlier file first
            runs = [runs[i] for i in sorted(ranked[: campaign.keep])]
        profiles += runs
    if not found:
        raise FileNotFoundError(f"no OpenACC platoon runs ({[c.folder + '/' + c.pattern for c in PROFILE_CAMPAIGNS]}) under {root}")
    return profiles


# ----------------------------------------------------------------------------- cache
def profile_to_dict(profile: LeaderProfile) -> dict:
    return {
        "name": profile.name,
        "source": profile.source,
        "t0": profile.t0,
        "vehicle_length": profile.vehicle_length,
        "v": profile.v.tolist(),
        "followers": [
            {"index": f.index, "driver": f.driver, "v": f.v.tolist(), "s": f.s.tolist()} for f in profile.followers
        ],
    }


def profile_from_dict(item: dict) -> LeaderProfile:
    followers = [RecordedFollower(f["index"], f["v"], f["s"], f["driver"]) for f in item.get("followers", [])]
    return LeaderProfile(item["name"], item["source"], item["v"], item["t0"], followers, item["vehicle_length"])


def save_profiles(profiles: Sequence[LeaderProfile], path: str | Path = CACHE, meta: dict | None = None) -> Path:
    """Write the profiles as compact JSON (through a temporary file); relative paths are taken
    from the repository root."""
    path = resolve_path(path)
    payload = {"dt": DT, "meta": dict(meta or {}), "profiles": [profile_to_dict(p) for p in profiles]}
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    part.replace(path)
    return path


def load_profiles(path: str | Path = CACHE) -> list[LeaderProfile]:
    payload = read_json(resolve_path(path))
    if abs(float(payload.get("dt", DT)) - DT) > 1e-12:
        raise ValueError(f"{path}: profiles sampled at dt = {payload['dt']}, expected {DT}")
    return [profile_from_dict(item) for item in payload["profiles"]]


def build_openacc_cache(raw_dir: str | Path | None = None, path: str | Path = CACHE, window: float = WINDOW) -> Path:
    """Run :func:`openacc_profiles` and cache the result with the selection rule and file hashes."""
    root = resolve_path(DEFAULT_RAW_DIR if raw_dir is None else raw_dir)
    profiles = openacc_profiles(root, window)
    meta = {
        "rule": (
            "per run the window with the largest variance of Speed1 among starts every 10 s where the leader and at "
            "least min_followers followers have speed > min_speed and finite IVS throughout; recorded followers up to "
            "the first one not valid throughout; per campaign the `keep` windows with the largest variance"
        ),
        "campaigns": [{"folder": c.folder, "pattern": c.pattern, "keep": c.keep} for c in PROFILE_CAMPAIGNS],
        "window": float(window),
        "window_step": WINDOW_STEP,
        "min_speed": MIN_SPEED,
        "min_followers": MIN_FOLLOWERS,
        "sha256": {p.source: file_sha256(root / p.source.split("/", 1)[1]) for p in profiles},
    }
    return save_profiles(profiles, path, meta)
