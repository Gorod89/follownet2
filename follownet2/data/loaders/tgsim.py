"""TGSIM trajectories (ITS DataHub, Socrata), raw data version.

Source format (Socrata view metadata and descriptions of the six tables):

* one file per data set in ``raw_dir``: ``<key>.parquet`` as written by the download tool
  (``conf/sources.yaml``, ``<key>`` a key of :data:`DATASETS`), otherwise the Socrata CSV
  export named in :data:`DATASETS`; the option ``files`` names the file explicitly. Columns
  ``id, time, xloc_kf, yloc_kf, lane_kf, speed_kf, acceleration_kf, length_smoothed,
  width_smoothed, type_most_common``, plus ``av`` (I-90/I-94) or ``acc`` (I-294) and
  ``run_index`` (all but I-395). Names are matched case-insensitively (the I-90/I-94 Moving
  table calls the id column ``ID``).
* ``time`` [s] from the start of the run, 0.1 s steps; ``xloc_kf``, ``yloc_kf`` [m] are
  coordinates in the reference image of the run (origin top left, 1 px = 0.3 m), not along the
  road; ``lane_kf`` the Kalman-filtered lane id; ``speed_kf`` [m/s] (a magnitude);
  ``acceleration_kf`` [m/s^2]; ``length_smoothed`` [m] the bounding-box length.
* no leader column; vehicle ids are reused across runs (and include 0 in I-294).

Decisions:

* group = ``run_index`` (the whole file for I-395, which has no runs), site = data set name.
* Longitudinal coordinate: one image axis per data set (:data:`DATASETS`, option ``axis``):
  ``"x"``, ``"y"`` or ``"pca"``, the principal axis of all positions of the run, for roads that
  run diagonally through the image. I-395 runs along y [verified by the coordinate ranges];
  I-294 mainly along x and I-90/I-94 diagonally [assumed from the coordinate ranges].
* Driving direction: per run and lane the coordinate is negated when the vehicles of the lane
  move towards decreasing values (sum of their displacements). I-294 has both directions, on
  lanes of opposite sign.
* Positions are taken as vehicle centres [assumed: the documentation does not say]:
  ``x_front = p + length/2``, ``x_rear = p - length/2`` with ``length = length_smoothed``.
* ``v = speed_kf``, ``a = acceleration_kf``, ``lane = lane_kf``; leaders come from
  ``derive_leaders`` (next vehicle ahead in the same run, lane and time step). Ids are strings,
  so that id 0 can be a leader.
* ``automated``: ``av`` or ``acc`` equal to yes in any case (I-294 L1 mixes No/Yes/YES), or
  ``type_most_common == 4`` in I-395; kept for the follower (``automated``) and the leader
  (``leader_automated``). ``vehicle_class`` is ``type_most_common`` (named for I-395 codes).
* Rows of an (id, time) key that occurs more than once in a run are dropped altogether and
  counted in the log (26,649 such pairs in I-395; one row of each looks like a sentinel).
* Foggy Bottom (urban network of 49 regions with x/y speeds) is not supported.

Limitations: a straight axis measures gaps along a chord, so on curved segments the gap is
underestimated by the cosine of the angle between road and axis; the centre-line files named
in the TGSIM descriptions would fix this but their location is unknown.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.dataset as ds

from follownet2.data.extraction import PairSeries, tracks_to_pairs

POSITION_SOURCE = "track"
DATASET = "tgsim"
NATIVE_RATE_HZ = 10


@dataclass(frozen=True)
class DataSet:
    file: str
    axis: str | None  # "x", "y" or "pca"; None: not supported
    flag: str | None = None  # automation column with yes/no values
    runs: bool = True  # has run_index
    automated_type: int | None = None  # type_most_common code of automated vehicles
    type_names: Mapping[int, str] | None = None  # names of numeric type_most_common codes


DATASETS = {
    "i90_94_stationary": DataSet("I90_94_stationary_final.csv", "pca", flag="av"),
    "i90_94_moving": DataSet("I90_94_moving_final.csv", "pca", flag="av"),
    "i294_l1": DataSet("I294_L1_final.csv", "x", flag="acc"),
    "i294_l2": DataSet("I294_L2_final.csv", "x", flag="acc"),
    "i395": DataSet(
        "I395-final.csv", "y", runs=False, automated_type=4,
        type_names={1: "passenger car", 2: "truck", 3: "bus", 4: "automated vehicle"},
    ),
    "foggy_bottom": DataSet("TGSIM-Foggy Bottom-Data.csv", None, runs=False),
}
BASE_FIELDS = (
    "id", "time", "xloc_kf", "yloc_kf", "lane_kf", "speed_kf", "acceleration_kf", "length_smoothed", "type_most_common",
)
TEXT_FIELDS = ("type_most_common", "av", "acc")
AXES = ("x", "y", "pca")

log = logging.getLogger(__name__)


def _layout() -> str:
    return "one file per data set in raw_dir, <key>.parquet or the CSV export: " + ", ".join(
        f"{k}.parquet or {d.file}" for k, d in DATASETS.items()
    )


def _source(raw_dir: Path, name: str, spec: DataSet, files: Mapping[str, str] | None) -> Path:
    candidates = [files[name]] if files and name in files else [f"{name}.parquet", spec.file]
    for candidate in candidates:
        if (raw_dir / candidate).is_file():
            return raw_dir / candidate
    raise FileNotFoundError(f"TGSIM {name}: none of {candidates} in {raw_dir}; expected {_layout()}")


def _open(path: Path, fields: Sequence[str]) -> tuple[ds.Dataset, dict[str, str]]:
    if path.suffix == ".parquet":
        source = ds.dataset(path, format="parquet")
        names = {name.lower(): name for name in source.schema.names}
    else:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            names = {name.lower(): name for name in next(csv.reader(fh), [])}
    missing = [f for f in fields if f not in names]
    if missing:
        raise ValueError(f"{path.name}: TGSIM columns {missing} are missing")
    if path.suffix != ".parquet":
        types = {names[f]: pa.string() if f in TEXT_FIELDS else pa.float64() for f in fields}
        source = ds.dataset(path, format=ds.CsvFileFormat(convert_options=pacsv.ConvertOptions(column_types=types)))
    return source, names


def _principal_axis(xy: np.ndarray) -> np.ndarray:
    ok = np.isfinite(xy).all(axis=1)
    centred = xy[ok] - xy[ok].mean(axis=0)
    return np.linalg.eigh(centred.T @ centred)[1][:, -1]


def _along_axis(xy: np.ndarray, axis: str) -> np.ndarray:
    if axis == "x":
        return xy[:, 0]
    if axis == "y":
        return xy[:, 1]
    return xy @ _principal_axis(xy)


def _oriented(p: np.ndarray, vehicle: np.ndarray, t: np.ndarray, lane: np.ndarray) -> np.ndarray:
    """``p``, negated on the lanes whose vehicles move towards decreasing ``p``."""
    order = np.lexsort((t, vehicle))
    step = np.zeros(len(p))
    same = vehicle[order[1:]] == vehicle[order[:-1]]
    step[order[1:]] = np.where(same, np.diff(p[order]), 0.0)
    flow = pd.Series(step).groupby(lane).transform("sum").to_numpy()
    return np.where(flow < 0.0, -p, p)


def _run_frame(rows: pd.DataFrame, name: str, spec: DataSet, axis: str, group) -> pd.DataFrame:
    key = pd.DataFrame({"id": rows["id"].to_numpy(), "step": np.round(rows["time"].to_numpy() * 10.0)})
    repeated = key.duplicated(keep=False).to_numpy()
    if repeated.any():
        n_keys = len(key[repeated].drop_duplicates())
        log.info("tgsim %s group %s: dropped %d rows of %d repeated (id, time) keys", name, group, int(repeated.sum()), n_keys)
        rows = rows[~repeated]
    p = _along_axis(rows[["xloc_kf", "yloc_kf"]].to_numpy(dtype=np.float64), axis)
    rows, p = rows[np.isfinite(p)], p[np.isfinite(p)]
    vehicle = rows["id"].to_numpy(dtype=np.float64)
    t = rows["time"].to_numpy(dtype=np.float64)
    lane = rows["lane_kf"].to_numpy(dtype=np.float64)
    p = _oriented(p, vehicle, t, lane)
    length = rows["length_smoothed"].to_numpy(dtype=np.float64)
    kind = rows["type_most_common"]
    if spec.type_names is not None:
        code = pd.to_numeric(kind, errors="coerce")
        kind = code.map(spec.type_names).fillna(kind)
        automated = code.eq(spec.automated_type)
    else:
        automated = rows[spec.flag].str.strip().str.lower().eq("yes")
    return pd.DataFrame(
        {
            "group": group,
            "site": name,
            "vehicle_id": pd.Series(vehicle.astype(np.int64)).astype(str).to_numpy(),
            "t": t,
            "x_front": p + length / 2.0,
            "x_rear": p - length / 2.0,
            "v": rows["speed_kf"].to_numpy(dtype=np.float64),
            "a": rows["acceleration_kf"].to_numpy(dtype=np.float64),
            "lane": lane,
            "length": length,
            "automated": automated.to_numpy(dtype=bool),
            "vehicle_class": kind.to_numpy(),
        }
    )


def load_pairs(
    raw_dir: str | Path,
    datasets: Sequence[str] = ("i90_94_stationary", "i90_94_moving", "i294_l1", "i294_l2", "i395"),
    files: Mapping[str, str] | None = None,
    axis: Mapping[str, str] | None = None,
    max_groups: int | None = None,
    dataset: str = DATASET,
) -> Iterator[PairSeries]:
    """Follower series of the TGSIM freeway data sets, one run at a time.

    ``files`` and ``axis`` override the file name and the longitudinal axis of a data set;
    ``max_groups`` stops after the first n runs (counted over all data sets); ``dataset`` is
    the name in the ids.
    """
    raw_dir = Path(raw_dir)
    done = 0
    for name in datasets:
        if name not in DATASETS:
            raise KeyError(f"unknown TGSIM data set {name!r}; known: {sorted(DATASETS)}")
        spec = DATASETS[name]
        if spec.axis is None:
            raise NotImplementedError(f"TGSIM {name}: urban network without a longitudinal axis, not supported")
        path = _source(raw_dir, name, spec, files)
        run_axis = (axis or {}).get(name, spec.axis)
        if run_axis not in AXES:
            raise ValueError(f"TGSIM {name}: axis {run_axis!r} is not one of {AXES}")
        fields = [*BASE_FIELDS, *([spec.flag] if spec.flag else []), *(["run_index"] if spec.runs else [])]
        source, names = _open(path, fields)
        if spec.runs:
            runs = np.unique(source.to_table(columns=[names["run_index"]]).column(0).to_numpy())
            parts = [(int(r), ds.field(names["run_index"]) == r) for r in runs]
        else:
            parts = [(name, None)]
        meta = {"native_rate_hz": NATIVE_RATE_HZ, "source_file": path.name, "ids": "source", "axis": run_axis}
        for group, expr in parts:
            if max_groups is not None and done >= max_groups:
                return
            table = source.to_table(columns=[names[f] for f in fields], filter=expr).rename_columns(fields)
            # without the pandas metadata nullable Int64 columns (download tool) come back as int64
            rows = table.to_pandas(ignore_metadata=True)
            frame = _run_frame(rows, name, spec, run_axis, group)
            done += 1
            yield from tracks_to_pairs(
                frame, dataset, extras=["automated", "vehicle_class"], leader_extras=["automated", "vehicle_class"], meta=meta
            )
