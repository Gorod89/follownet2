"""highD drone trajectories (levelXdata, by application), raw data version.

Source format (highD format PDF, column names confirmed by the highD-dataset tools):

* per recording ``XX`` (01..60) three files in ``raw_dir``: ``XX_tracks.csv`` (one row per
  vehicle and frame), ``XX_tracksMeta.csv`` (one row per vehicle), ``XX_recordingMeta.csv``.
* Image coordinates in m: x grows to the right, y downwards, lanes run along x. ``x``, ``y``
  are the upper-left corner of the bounding box and ``width`` its extent along x, i.e. the
  vehicle length. ``xVelocity`` [m/s], ``xAcceleration`` [m/s^2] are signed along x.
* ``drivingDirection`` (tracks meta): 1 = upper lanes, moving towards decreasing x;
  2 = lower lanes, moving towards increasing x. ``precedingId`` is the same-lane leader
  (0: none), ``laneId`` the lane, ``frame`` the frame number, ``frameRate`` [Hz] and
  ``locationId`` come from the recording meta.

Decisions:

* ``drivingDirection == 2``: ``x_rear = x``, ``x_front = x + width``, ``v = xVelocity``,
  ``a = xAcceleration``. ``drivingDirection == 1``: the axis is flipped, ``x_front = -x``,
  ``x_rear = -(x + width)``, ``v = -xVelocity``, ``a = -xAcceleration``.
* ``t = (frame - 1) / frameRate`` with the frame rate of the recording meta (25 Hz in the
  documentation, not hard-coded); ``tracks_to_pairs`` resamples to the 10 Hz grid.
* site = ``locationId``, group = recording id (two digits), leader = ``precedingId``,
  lane = ``laneId``, length = ``width``; ``vehicle_class`` (``class`` of the tracks meta:
  Car, Truck) is kept for follower and leader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from follownet2.data.extraction import PairSeries, tracks_to_pairs

POSITION_SOURCE = "track"
DATASET = "highd"

TRACK_COLUMNS = ["frame", "id", "x", "width", "xVelocity", "xAcceleration", "precedingId", "laneId"]
VEHICLE_COLUMNS = ["id", "class", "drivingDirection"]
RECORDING_COLUMNS = ["id", "frameRate", "locationId"]
LAYOUT = "raw_dir/XX_tracks.csv, raw_dir/XX_tracksMeta.csv and raw_dir/XX_recordingMeta.csv per recording XX"


def _recordings(raw_dir: Path, recordings: Sequence[int] | None) -> list[str]:
    prefixes = sorted(p.name.removesuffix("_tracks.csv") for p in raw_dir.glob("*_tracks.csv"))
    if recordings is not None:
        wanted = {int(r) for r in recordings}
        prefixes = [p for p in prefixes if p.isdigit() and int(p) in wanted]
    if not prefixes:
        raise FileNotFoundError(f"no highD recording in {raw_dir}; expected {LAYOUT}")
    return prefixes


def _recording_frame(raw_dir: Path, prefix: str) -> tuple[pd.DataFrame, float]:
    """Tracks frame of one recording and its frame rate."""
    files = {kind: raw_dir / f"{prefix}_{kind}.csv" for kind in ("tracks", "tracksMeta", "recordingMeta")}
    missing = [p.name for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"highD recording {prefix}: {missing} missing in {raw_dir}; expected {LAYOUT}")
    recording = pd.read_csv(files["recordingMeta"], usecols=RECORDING_COLUMNS).iloc[0]
    rate = float(recording["frameRate"])
    vehicles = pd.read_csv(files["tracksMeta"], usecols=VEHICLE_COLUMNS)
    tracks = pd.read_csv(files["tracks"], usecols=TRACK_COLUMNS).merge(vehicles, on="id", how="left", validate="many_to_one")
    direction = tracks["drivingDirection"].to_numpy()
    if not np.isin(direction, (1, 2)).all():
        raise ValueError(f"highD recording {prefix}: drivingDirection other than 1 or 2 (or a vehicle missing in the tracks meta)")
    left = direction == 1
    sign = np.where(left, -1.0, 1.0)
    x = tracks["x"].to_numpy(dtype=np.float64)
    length = tracks["width"].to_numpy(dtype=np.float64)
    frame = pd.DataFrame(
        {
            "group": f"{int(recording['id']):02d}",
            "site": str(int(recording["locationId"])),
            "vehicle_id": tracks["id"],
            "t": (tracks["frame"] - 1) / rate,
            "x_front": np.where(left, -x, x + length),
            "x_rear": np.where(left, -(x + length), x),
            "v": sign * tracks["xVelocity"].to_numpy(dtype=np.float64),
            "a": sign * tracks["xAcceleration"].to_numpy(dtype=np.float64),
            "lane": tracks["laneId"],
            "leader": tracks["precedingId"],
            "length": length,
            "vehicle_class": tracks["class"],
        }
    )
    return frame, rate


def load_pairs(
    raw_dir: str | Path,
    recordings: Sequence[int] | None = None,
    max_groups: int | None = None,
    dataset: str = DATASET,
) -> Iterator[PairSeries]:
    """Follower series of highD, one recording at a time.

    ``recordings`` selects recording ids (all by default); ``max_groups`` stops after the
    first n recordings; ``dataset`` is the name in the ids.
    """
    raw_dir = Path(raw_dir)
    for prefix in _recordings(raw_dir, recordings)[:max_groups]:
        frame, rate = _recording_frame(raw_dir, prefix)
        meta = {"native_rate_hz": rate, "source_file": f"{prefix}_tracks.csv", "ids": "source"}
        yield from tracks_to_pairs(frame, dataset, extras=["vehicle_class"], leader_extras=["vehicle_class"], meta=meta)
