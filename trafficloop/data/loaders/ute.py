"""Ubiquitous Traffic Eyes (UTE) trajectories (github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye), raw data version.

Source format (headers and first rows of the released ``frenet.csv`` files read through the
Google Drive links; the per-site ``data_format.pdf`` was not read):

* ``raw_dir/<site>/frenet.csv`` per site of :data:`SITES`, one recording per site (the
  released "sample data").
* Frenet coordinates ``longitudinalDistance(m)`` and ``lateralDistance(m)``, ``speed(km/h)``,
  ``acceleration(m/s^2)``, ``laneID``, ``vehicleID``, ``time(s)`` at the video frame rate
  (24, 30 or 60 fps), ``vehicleLength`` in m (in pixels in SQM-W-1), and ``leader_id`` (-1:
  none) in SQM-W-1 and SQM-W-2 only. Column order and presence of ``frame`` differ per site.

Decisions:

* One sub-parser per site (:data:`SITES`) checks the verified header and maps the columns.
  SQM-N-4 ships ``frenet-down.xlsx`` and ``frenet-up.xlsx`` whose columns were not verified:
  it raises ``NotImplementedError``.
* site = group = site name; ``t = time(s)`` (resampled to 10 Hz by ``tracks_to_pairs``),
  ``v = speed / 3.6``, ``a = acceleration``, ``lane = laneID``; leader = ``leader_id`` where the
  site has it, otherwise ``derive_leaders``. Ids are strings, so that id 0 can be a leader.
* ``longitudinalDistance`` is taken as the position of the vehicle centre [assumed]:
  ``x_front``/``x_rear`` = position plus/minus half the length. SQM-W-1 gives lengths in pixels
  without a scale, so its positions are centres and its gaps centre to centre
  (``meta["gap_definition"]``).
* The direction of ``longitudinalDistance`` relative to the traffic is not documented
  [assumed either way]: per lane, the coordinate is negated when the vehicles of the lane move
  towards decreasing values. Lanes of opposite directions must have different ``laneID``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from trafficloop.data.extraction import PairSeries, tracks_to_pairs

POSITION_SOURCE = "track"
DATASET = "ute"
KMH = 1.0 / 3.6  # m/s per km/h

SQM_W_1 = (
    "vehicleID", "frame", "time(s)", "laneID", "longitudinalDistance(m)", "speed(km/h)", "acceleration(m/s^2)",
    "lateralDistance(m)", "vehicleLength(pixel)", "vehicleWidth(pixel)", "leader_id", "follower_id",
    "leader_distance(m)", "follower_distance(m)",
)
SQM_W_2 = tuple(c.replace("(pixel)", "(m)") for c in SQM_W_1)
YTDJ_3 = (
    "vehicleID", "laneID", "time(s)", "lateralDistance(m)", "longitudinalDistance(m)", "speed(km/h)",
    "acceleration(m/s^2)", "vehicleLength(m)", "vehicleWidth(m)",
)
XAM_N_6 = (
    "frame", "time(s)", "vehicleID", "laneID", "longitudinalDistance(m)", "lateralDistance(m)", "speed(km/h)",
    "acceleration(m/s^2)", "vehicleLength(m)", "vehicleWidth(m)",
)
RML_7 = (
    "frame", "vehicleID", "laneID", "time(s)", "lateralDistance(m)", "longitudinalDistance(m)", "speed(km/h)",
    "acceleration(m/s^2)", "vehicleLength(m)", "vehicleWidth(m)",
)


def _ids(values: pd.Series) -> pd.Series:
    return values.astype(np.int64).astype(str)


def _tracks(raw: pd.DataFrame, length: str | None, leader: bool) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "vehicle_id": _ids(raw["vehicleID"]),
            "t": raw["time(s)"].to_numpy(dtype=np.float64),
            "position": raw["longitudinalDistance(m)"].to_numpy(dtype=np.float64),
            "v": raw["speed(km/h)"].to_numpy(dtype=np.float64) * KMH,
            "a": raw["acceleration(m/s^2)"].to_numpy(dtype=np.float64),
            "lane": raw["laneID"].to_numpy(),
        }
    )
    if length is not None:
        out["length"] = raw[length].to_numpy(dtype=np.float64)
    if leader:
        out["leader"] = _ids(raw["leader_id"]).where(raw["leader_id"].to_numpy() >= 0)
    return out


def _parse_sqm_pixel(raw: pd.DataFrame) -> pd.DataFrame:
    """SQM-W-1: leader given, vehicle lengths in pixels (not usable)."""
    return _tracks(raw, None, leader=True)


def _parse_sqm(raw: pd.DataFrame) -> pd.DataFrame:
    """SQM-W-2: leader given, vehicle lengths in m."""
    return _tracks(raw, "vehicleLength(m)", leader=True)


def _parse_frenet(raw: pd.DataFrame) -> pd.DataFrame:
    """YTDJ-3, XAM-N-5, XAM-N-6, RML-7, PKDD-8, XAM-S-9: no leader, lengths in m."""
    return _tracks(raw, "vehicleLength(m)", leader=False)


@dataclass(frozen=True)
class Site:
    fps: float
    header: tuple[str, ...] | None  # verified header of frenet.csv; None: not verified
    parser: Callable[[pd.DataFrame], pd.DataFrame] | None


SITES = {
    "SQM-W-1": Site(24, SQM_W_1, _parse_sqm_pixel),
    "SQM-W-2": Site(24, SQM_W_2, _parse_sqm),
    "YTDJ-3": Site(24, YTDJ_3, _parse_frenet),
    "SQM-N-4": Site(30, None, None),
    "XAM-N-5": Site(30, YTDJ_3, _parse_frenet),
    "XAM-N-6": Site(60, XAM_N_6, _parse_frenet),
    "RML-7": Site(30, RML_7, _parse_frenet),
    "PKDD-8": Site(30, RML_7, _parse_frenet),
    "XAM-S-9": Site(30, RML_7, _parse_frenet),
}
IMPLEMENTED = tuple(name for name, site in SITES.items() if site.parser is not None)


def _oriented(p: np.ndarray, vehicle: np.ndarray, t: np.ndarray, lane: np.ndarray) -> np.ndarray:
    """``p``, negated on the lanes whose vehicles move towards decreasing ``p``."""
    order = np.lexsort((t, vehicle))
    step = np.zeros(len(p))
    same = vehicle[order[1:]] == vehicle[order[:-1]]
    step[order[1:]] = np.where(same, np.diff(p[order]), 0.0)
    flow = pd.Series(step).groupby(lane).transform("sum").to_numpy()
    return np.where(flow < 0.0, -p, p)


def _site_frame(path: Path, name: str, site: Site) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        header = tuple(next(csv.reader(fh), []))
    if header != site.header:
        raise ValueError(f"UTE {name}: {path.name} has the header {header}, the verified one is {site.header}")
    tracks = site.parser(pd.read_csv(path))
    position = _oriented(
        tracks["position"].to_numpy(),
        pd.factorize(tracks["vehicle_id"])[0],
        tracks["t"].to_numpy(),
        tracks["lane"].to_numpy(),
    )
    half = tracks["length"].to_numpy() / 2.0 if "length" in tracks.columns else 0.0
    return tracks.drop(columns="position").assign(
        group=name, site=name, x_front=position + half, x_rear=position - half
    )


def load_pairs(
    raw_dir: str | Path,
    sites: Sequence[str] = IMPLEMENTED,
    max_groups: int | None = None,
    dataset: str = DATASET,
) -> Iterator[PairSeries]:
    """Follower series of the UTE sites, one site (recording) at a time; ``max_groups`` stops
    after the first n sites; ``dataset`` is the name in the ids."""
    raw_dir = Path(raw_dir)
    for name in list(sites)[:max_groups]:
        if name not in SITES:
            raise KeyError(f"unknown UTE site {name!r}; known: {sorted(SITES)}")
        site = SITES[name]
        if site.parser is None:
            raise NotImplementedError(f"UTE site {name}: the columns of its trajectory files are not verified")
        path = raw_dir / name / "frenet.csv"
        if not path.is_file():
            raise FileNotFoundError(f"UTE site {name}: {path} is missing; expected raw_dir/<site>/frenet.csv")
        frame = _site_frame(path, name, site)
        meta = {"native_rate_hz": site.fps, "source_file": f"{name}/frenet.csv", "ids": "source"}
        if "length" not in frame.columns:
            meta["gap_definition"] = "centre to centre (vehicle lengths in pixels without a scale)"
        yield from tracks_to_pairs(frame, dataset, meta=meta)
