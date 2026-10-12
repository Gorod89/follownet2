"""CitySim drone trajectories (UCF SST lab, by request form), raw data version, freeway sites.

Source format (CitySim wiki; no data file was seen):

* trajectory CSV files ``raw_dir/<site>/<recording>.csv`` at 30 fps; other CSV files of a site
  folder (recording metadata, signal timing) are recognised by their header and skipped.
* columns ``frameNum``, ``carId``, the centre, head, tail and bounding-box points in pixels
  (``headX``, ``headY``, ...) and the same points in feet with the suffix ``ft``
  (``headXft``, ``headYft``, ``tailXft``, ``tailYft``, ...) [spelling assumed from the wiki],
  ``*Lat``/``*Lon`` (US sites only), ``speed`` [mph], ``heading``, ``course`` and ``laneId``.
  There is no leader column and no acceleration.

Decisions:

* site = folder name, group = file name without extension, ``t = frameNum / frame_rate``
  (30 fps by default; resampled to 10 Hz by ``tracks_to_pairs``), ``v = speed`` in m/s,
  ``lane = laneId``; no ``a``.
* Longitudinal coordinate per lane and recording: the head and tail points (feet) are projected
  on the principal axis of all head points of the lane, oriented along the mean motion of the
  lane's vehicles: ``x_front`` and ``x_rear`` are the projections of head and tail, the length
  is their difference. Positions of different lanes are not comparable, so they are blanked in
  the frames next to a lane change (a lane change ends an event anyway).
* Leaders are derived (next vehicle ahead in the same lane and time step). Ids are strings.

Limitation: the projection on a straight axis is valid for straight freeway segments only; on
curved lanes it shortens gaps (the chord of the lane instead of its length). Lanes of opposite
directions must have different ``laneId``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from follownet2.data.extraction import PairSeries, tracks_to_pairs

POSITION_SOURCE = "track"
DATASET = "citysim"

FEET = 0.3048  # m per ft
MPH = 0.44704  # m/s per mph
COLUMNS = ("frameNum", "carId", "headXft", "headYft", "tailXft", "tailYft", "speed", "laneId")
LAYOUT = "raw_dir/<site>/<recording>.csv with the columns " + ", ".join(COLUMNS)


def _header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return next(csv.reader(fh), [])


def _recordings(raw_dir: Path, sites: Sequence[str] | None) -> list[Path]:
    """Trajectory files of the selected site folders (header with ``frameNum`` and ``carId``)."""
    folders = sorted(p for p in raw_dir.iterdir() if p.is_dir()) if raw_dir.is_dir() else []
    if sites is not None:
        folders = [p for p in folders if p.name in set(sites)]
    files = [f for folder in folders for f in sorted(folder.glob("*.csv"))]
    files = [f for f in files if {"framenum", "carid"} <= {c.lower() for c in _header(f)}]
    if not files:
        raise FileNotFoundError(f"no CitySim trajectory file in {raw_dir}; expected {LAYOUT}")
    return files


def _lane_coordinates(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Head and tail projected on the principal axis of the lane's head points [ft]."""
    head = rows[["headXft", "headYft"]].to_numpy(dtype=np.float64)
    tail = rows[["tailXft", "tailYft"]].to_numpy(dtype=np.float64)
    origin = head.mean(axis=0)
    centred = head - origin
    axis = np.linalg.eigh(centred.T @ centred)[1][:, -1]
    order = np.lexsort((rows["frameNum"].to_numpy(), rows["carId"].to_numpy()))
    same = rows["carId"].to_numpy()[order[1:]] == rows["carId"].to_numpy()[order[:-1]]
    motion = np.diff(head[order], axis=0)[same].sum(axis=0)
    if motion @ axis < 0.0:
        axis = -axis
    return centred @ axis, (tail - origin) @ axis


def _recording_frame(path: Path, site: str, frame_rate: float) -> pd.DataFrame:
    names = {c.lower(): c for c in _header(path)}
    missing = [c for c in COLUMNS if c.lower() not in names]
    if missing:
        raise ValueError(f"{path.name}: CitySim columns {missing} are missing; expected {LAYOUT}")
    rows = pd.read_csv(path, usecols=[names[c.lower()] for c in COLUMNS])
    rows = rows.rename(columns={names[c.lower()]: c for c in COLUMNS})
    rows = rows.sort_values(["carId", "frameNum"], kind="stable").reset_index(drop=True)
    head = np.full(len(rows), np.nan)
    tail = np.full(len(rows), np.nan)
    for _, index in rows.groupby("laneId").groups.items():
        head[index], tail[index] = _lane_coordinates(rows.loc[index])
    length = (head - tail) * FEET
    vehicle = rows["carId"].to_numpy()
    lane = rows["laneId"].to_numpy()
    changed = (lane[1:] != lane[:-1]) & (vehicle[1:] == vehicle[:-1])
    blank = np.zeros(len(rows), dtype=bool)
    blank[1:] |= changed
    blank[:-1] |= changed
    head[blank] = tail[blank] = np.nan
    return pd.DataFrame(
        {
            "group": path.stem,
            "site": site,
            "vehicle_id": rows["carId"].astype(np.int64).astype(str),
            "t": rows["frameNum"].to_numpy(dtype=np.float64) / frame_rate,
            "x_front": head * FEET,
            "x_rear": tail * FEET,
            "v": rows["speed"].to_numpy(dtype=np.float64) * MPH,
            "lane": lane,
            "length": length,
        }
    )


def load_pairs(
    raw_dir: str | Path,
    sites: Sequence[str] | None = None,
    frame_rate: float = 30.0,
    max_groups: int | None = None,
    dataset: str = DATASET,
) -> Iterator[PairSeries]:
    """Follower series of CitySim freeway recordings, one recording file at a time.

    ``sites`` selects site folders (all by default); ``max_groups`` stops after the first n
    recordings; ``dataset`` is the name in the ids.
    """
    raw_dir = Path(raw_dir)
    for path in _recordings(raw_dir, sites)[:max_groups]:
        site = path.parent.name
        frame = _recording_frame(path, site, frame_rate)
        meta = {"native_rate_hz": frame_rate, "source_file": f"{site}/{path.name}", "ids": "source"}
        yield from tracks_to_pairs(frame, dataset, meta=meta)
