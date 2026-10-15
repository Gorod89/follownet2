"""NGSIM US-101 and I-80 vehicle trajectories (Socrata table ``8ect-6jqj``), raw data version.

Source format (Socrata metadata, checked against the parquet files of cf-stability):

* ``raw_dir/ngsim_<location>_raw.parquet`` with Socrata field names: all 25 fields (the files
  of cf-stability) or the 13 fields selected by the download tool (``conf/sources.yaml``, table
  names ``ngsim_<location>_raw``, no ``location`` column); or Socrata CSV exports
  ``raw_dir/*.csv`` with the same fields. Column names are matched case-insensitively, so the
  display names of the UI export (``Vehicle_ID``, ``Global_Time``, ...) work as well; CSV rows
  are attributed to a location by the ``location`` column. A parquet file of a location takes
  precedence over the CSV files.
* One row per vehicle and 100 ms frame, imperial units: ``local_y`` [ft] is the FRONT centre of
  the vehicle along the road, ``v_length`` [ft], ``v_vel`` [ft/s], ``v_acc`` [ft/s^2],
  ``global_time`` [ms since 1970], ``frame_id`` (frame of the original 15-minute file),
  ``lane_id``, ``preceding`` (id of the same-lane leader, 0: none), ``v_class`` (1 motorcycle,
  2 auto, 3 truck), ``space_headway`` [ft, front to front].

Decisions:

* The rows of a location come from three original 15-minute files, which overlap in time by
  about 1.5 min. ``frame_id`` restarts in every file, so ``global_time - 100 * frame_id`` is
  constant inside a file: the file of a row is the rank of that offset among the distinct
  offsets of its location, and the group is ``"<location>-p<k>"``. More than six distinct
  offsets raise ``ValueError`` (``frame_id`` is then not what the rule assumes). Vehicle ids are
  unique inside a file and ``preceding`` refers to the same file (on the real data every leader
  is found). The rows of one file are selected with a filter on the offset, so memory stays
  bounded by one file.
* Exact duplicate rows (about 704,000 in us-101) are dropped first. Rows of a file that still
  share ``(vehicle_id, global_time)`` contradict each other and are dropped as well (none on the
  real data).
* Conversions: ``FEET``; ``x_front = local_y``, ``x_rear = local_y - v_length``,
  ``v = v_vel``, ``a = v_acc`` (the dataset acceleration is kept in the raw data version),
  ``length = v_length``, ``lane = lane_id``, ``leader = preceding``;
  ``t = (global_time - t0) / 1000`` with ``t0`` the first time stamp of the location. Every
  time stamp must lie on the 100 ms grid.
* Followers are samples in the selected lanes (``lanes``: mainline by default) of vehicles of
  ``follower_classes`` (autos by default). Other samples keep their positions, so any vehicle
  can be a leader, but they have no leader themselves.
* ``vehicle_class`` (follower and leader) is the class name of ``v_class``.
* :func:`load_tracks` yields the tracks frame of every file (with ``frame_id``), which
  :func:`load_pairs` turns into follower series and :mod:`trafficloop.data.platoons` into platoon
  windows.

Limitation, left as in the source: in the overlap of two files NGSIM matched leaders within
each file only, so a vehicle of the other file can be between a follower and its ``preceding``
(0.22 % of the follower samples of us-101, 0.31 % of i-80), and a few vehicles are recorded in
both files (17 in us-101, 20 in i-80).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as ds

from trafficloop.data.extraction import PairSeries, tracks_to_pairs

POSITION_SOURCE = "track"
DATASET = "ngsim"

FEET = 0.3048  # m per ft
NATIVE_RATE_HZ = 10
STEP_MS = 100
MAX_FILES = 6  # distinct offsets global_time - 100 * frame_id per location (3 on the real data)
CLASS_NAMES = {1: "motorcycle", 2: "auto", 3: "truck"}
DEFAULT_LANES = MappingProxyType({"us-101": (1, 2, 3, 4, 5), "i-80": (1, 2, 3, 4, 5, 6)})

# Socrata field names read by the loader and their types in a CSV export
FIELDS = {
    "vehicle_id": pa.int64(),
    "frame_id": pa.int64(),
    "global_time": pa.int64(),
    "local_y": pa.float64(),
    "v_length": pa.float64(),
    "v_class": pa.int64(),
    "v_vel": pa.float64(),
    "v_acc": pa.float64(),
    "lane_id": pa.int64(),
    "preceding": pa.int64(),
}
LAYOUT = (
    "raw_dir/ngsim_<location>_raw.parquet or Socrata CSV exports raw_dir/*.csv with the fields "
    + ", ".join([*FIELDS, "location"])
)

log = logging.getLogger(__name__)


@dataclass
class _Source:
    """One file with a map from Socrata field names to its column names."""

    path: Path
    dataset: ds.Dataset
    names: dict[str, str]

    @classmethod
    def open(cls, path: Path) -> "_Source":
        if path.suffix == ".parquet":
            dataset = ds.dataset(path, format="parquet")
            header = dataset.schema.names
        else:
            with open(path, newline="", encoding="utf-8-sig") as fh:
                header = next(csv.reader(fh), [])
        names = {name.lower(): name for name in header}
        required = [*FIELDS, *([] if path.suffix == ".parquet" else ["location"])]
        missing = [f for f in required if f not in names]
        if missing:
            raise ValueError(f"{path.name}: NGSIM fields {missing} are missing; expected {LAYOUT}")
        if path.suffix != ".parquet":
            # a scan converts only the projected columns, so the other fields cannot fail it
            types = {names[f]: typ for f, typ in {**FIELDS, "location": pa.string()}.items()}
            convert = pacsv.ConvertOptions(column_types=types)
            dataset = ds.dataset(path, format=ds.CsvFileFormat(convert_options=convert))
        return cls(path, dataset, names)

    def scan(self, fields: Sequence[str], location: str, offset: int | None = None) -> pd.DataFrame:
        """Rows of ``location``; with ``offset`` only those of the file with that offset."""
        expr = ds.field(self.names["location"]) == location if "location" in self.names else None
        if offset is not None:
            of_file = ds.field(self.names["global_time"]) - ds.field(self.names["frame_id"]) * STEP_MS == offset
            expr = of_file if expr is None else expr & of_file
        table = self.dataset.to_table(columns=[self.names[f] for f in fields], filter=expr)
        # without the pandas metadata nullable Int64 columns (download tool) come back as int64
        return table.rename_columns(list(fields)).to_pandas(ignore_metadata=True)


def _open_location(raw_dir: Path, location: str) -> tuple[list[_Source], int, np.ndarray]:
    """The files that hold rows of ``location``, its first time stamp and the sorted offsets
    ``global_time - 100 * frame_id`` of its original 15-minute files."""
    parquet = raw_dir / f"ngsim_{location}_raw.parquet"
    paths = [parquet] if parquet.is_file() else sorted(raw_dir.glob("*.csv"))
    sources, offsets, starts = [], [], []
    for path in paths:
        source = _Source.open(path)
        rows = source.scan(["global_time", "frame_id"], location)
        if len(rows):
            sources.append(source)
            t = rows["global_time"].to_numpy()
            offsets.append(np.unique(t - STEP_MS * rows["frame_id"].to_numpy()))
            starts.append(int(t.min()))
    if not sources:
        raise FileNotFoundError(f"no NGSIM rows of location {location!r} in {raw_dir}; expected {LAYOUT}")
    t0 = min(starts)
    offsets = np.unique(np.concatenate(offsets))
    if len(offsets) > MAX_FILES:
        raise ValueError(
            f"NGSIM {location}: {len(offsets)} distinct global_time - 100 * frame_id, more than {MAX_FILES} "
            "files: frame_id does not restart per 15-minute file"
        )
    # global_time = offset + 100 * frame_id: the stamps lie on the grid if the offsets do
    if np.any((offsets - t0) % STEP_MS):
        raise ValueError(f"NGSIM {location}: time stamps off the {STEP_MS} ms grid")
    return sources, t0, offsets


def _file_frame(
    sources: list[_Source],
    location: str,
    k: int,
    offset: int,
    t0: int,
    lanes: Sequence[int] | None,
    follower_classes: Sequence[int],
) -> pd.DataFrame:
    """Tracks frame of the original file ``k`` of a location."""
    rows = pd.concat([s.scan(list(FIELDS), location, offset) for s in sources], ignore_index=True)
    group = f"{location}-p{k}"
    n = len(rows)
    rows = rows.drop_duplicates()
    if len(rows) < n:
        log.info("ngsim %s: dropped %d exact duplicate rows", group, n - len(rows))
    clash = rows.duplicated(["vehicle_id", "global_time"], keep=False)
    if clash.any():
        log.warning("ngsim %s: dropped %d contradicting rows of repeated (vehicle_id, global_time)", group, int(clash.sum()))
        rows = rows[~clash]
    rows = rows.reset_index(drop=True)
    is_follower = rows["v_class"].isin(list(follower_classes))
    if lanes is not None:
        is_follower &= rows["lane_id"].isin(list(lanes))
    vehicle_class = rows["v_class"].map(CLASS_NAMES).fillna(rows["v_class"].astype(str))
    return pd.DataFrame(
        {
            "group": group,
            "site": location,
            "vehicle_id": rows["vehicle_id"],
            "t": (rows["global_time"] - t0) / 1000.0,
            "x_front": rows["local_y"] * FEET,
            "x_rear": (rows["local_y"] - rows["v_length"]) * FEET,
            "v": rows["v_vel"] * FEET,
            "a": rows["v_acc"] * FEET,
            "lane": rows["lane_id"],
            "leader": rows["preceding"].where(is_follower),
            "length": rows["v_length"] * FEET,
            "vehicle_class": vehicle_class,
            "frame_id": rows["frame_id"],
        }
    )


def load_tracks(
    raw_dir: str | Path,
    locations: Sequence[str] = ("us-101", "i-80"),
    lanes: Mapping[str, Sequence[int] | None] | None = DEFAULT_LANES,
    follower_classes: Sequence[int] = (2,),
    max_groups: int | None = None,
) -> Iterator[tuple[pd.DataFrame, dict]]:
    """Tracks frames of the NGSIM freeway locations, one original 15-minute file at a time, with
    the meta of their follower series.

    The frame holds the columns of :func:`~trafficloop.data.extraction.tracks_to_pairs` plus
    ``vehicle_class`` and ``frame_id`` (the frame of the original file); ``leader`` is set on the
    samples of followers only (``lanes``, ``follower_classes``; see :func:`load_pairs`).
    """
    raw_dir = Path(raw_dir)
    done = 0
    for location in locations:
        sources, t0, offsets = _open_location(raw_dir, location)
        meta = {
            "native_rate_hz": NATIVE_RATE_HZ,
            "source_file": ";".join(s.path.relative_to(raw_dir).as_posix() for s in sources),
            "ids": "source",
        }
        follower_lanes = None if lanes is None else lanes.get(location)
        for k, offset in enumerate(offsets):
            if max_groups is not None and done >= max_groups:
                return
            frame = _file_frame(sources, location, k, int(offset), t0, follower_lanes, follower_classes)
            done += 1
            yield frame, meta


def load_pairs(
    raw_dir: str | Path,
    locations: Sequence[str] = ("us-101", "i-80"),
    lanes: Mapping[str, Sequence[int] | None] | None = DEFAULT_LANES,
    follower_classes: Sequence[int] = (2,),
    max_groups: int | None = None,
    dataset: str = DATASET,
) -> Iterator[PairSeries]:
    """Follower series of the NGSIM freeway locations, one original 15-minute file at a time.

    ``lanes`` maps a location to its follower lanes; ``None`` (or a location without entry)
    keeps all lanes. ``follower_classes`` are ``v_class`` codes. ``max_groups`` stops after the
    first n files (counted over all locations). ``dataset`` is the name in the ids.
    """
    for frame, meta in load_tracks(raw_dir, locations, lanes, follower_classes, max_groups):
        yield from tracks_to_pairs(frame, dataset, extras=["vehicle_class"], leader_extras=["vehicle_class"], meta=meta)
