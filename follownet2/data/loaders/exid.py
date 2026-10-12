"""exiD drone trajectories of highway exits and entries (levelXdata, by application), raw data version.

Source format (exiD "Highway Dataset Format" PDF of 3/8/2024; names confirmed by the
drone-dataset-tools):

* per recording ``XX`` in ``raw_dir``: ``XX_tracks.csv`` (one row per track and frame),
  ``XX_tracksMeta.csv`` (``trackId``, ``class``, ...) and ``XX_recordingMeta.csv``
  (``recordingId``, ``locationId``, ``frameRate`` [Hz]); the file list of the PDF says
  ``XX_recordingsMeta.csv``, which is accepted as well.
* base columns: ``trackId``, ``frame``, ``length`` [m], ``lonVelocity`` [m/s],
  ``lonAcceleration`` [m/s^2] (longitudinal: along the heading of the vehicle).
* map-based enrichment, present only "if part of the purchased dataset package": ``laneletId``
  and ``lonLaneletPos`` [m] (semicolon-separated lists, one entry per lanelet that contains the
  centroid; ``lonLaneletPos`` is the position of the centroid along the centre line from the
  start of the lanelet), ``traveledDistance`` [m] (accumulated along the track), ``laneChange``
  (set in the first frame in which the centroid is in the new lane), ``leadId`` (-1: none) and
  ``leadDHW`` [m] (-1: none). The format text defines ``leadDHW`` as the distance from the front
  bumper of the vehicle to the rear bumper of its lead vehicle: it is the net gap already, so
  the conversion to the net gap with the vehicle lengths is the identity.

Decisions:

* site = ``locationId``, group = recording id (two digits), ``t = frame / frameRate``
  (resampled to 10 Hz), ``v = lonVelocity``, ``a = lonAcceleration``, length = ``length``,
  ``vehicle_class`` = ``class`` of the tracks meta (follower and leader). Ids are strings, so
  that track 0 can be a leader.
* Mode ``"leadDHW"`` (columns ``leadId``, ``leadDHW``, ``traveledDistance``): leader =
  ``leadId`` and the gap is ``leadDHW``. Every vehicle has the continuous coordinate
  ``p = traveledDistance`` (own origin), ``x_rear = p - length/2``; a follower's front bumper is
  expressed in the coordinate of its current leader, ``x_front = x_rear(leader) - leadDHW``
  (i.e. ``x_lead = x_front + leadDHW``), so ``s = leadDHW`` exactly and events do not end at
  lanelet borders. ``lane`` counts the lane changes of the vehicle (``laneChange``), so a lane
  change of follower or leader still ends an event; its value carries no meaning across
  vehicles. The resampling to 10 Hz is done before this construction because ``leadDHW``
  belongs to a pair: it is blanked in the frames next to a change of ``leadId``.
* Mode ``"lonLaneletPos"`` (otherwise; ``laneletId``, ``lonLaneletPos``): the first entry of
  each list is used. ``lonLaneletPos`` is continuous inside one lanelet only, so
  ``lane = laneletId`` and a change of lanelet is a lane change: events end at lanelet borders
  (positions next to a change are blanked before the resampling). ``x_front``/``x_rear`` are the
  centroid position plus/minus half the length. The leader is ``leadId`` while it is in the
  same lanelet as the follower (positions in different lanelets are not comparable), and
  derived from lanelet and position (``derive_leaders``) when ``leadId`` is absent.

Whether the academic exiD release contains the enrichment columns is not verified; without
them the loader raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from follownet2.data.extraction import PairSeries, resample_tracks, tracks_to_pairs

POSITION_SOURCE = "track"
DATASET = "exid"

BASE_COLUMNS = ["trackId", "frame", "length", "lonVelocity", "lonAcceleration"]
DHW_COLUMNS = ["leadId", "leadDHW", "traveledDistance"]
LANELET_COLUMNS = ["laneletId", "lonLaneletPos"]
OPTIONAL_COLUMNS = ["laneChange", "leadId"]
LAYOUT = "raw_dir/XX_tracks.csv, raw_dir/XX_tracksMeta.csv and raw_dir/XX_recordingMeta.csv per recording XX"


def _files(raw_dir: Path, prefix: str) -> dict[str, Path]:
    files = {kind: raw_dir / f"{prefix}_{kind}.csv" for kind in ("tracks", "tracksMeta", "recordingMeta")}
    if not files["recordingMeta"].is_file():
        files["recordingMeta"] = raw_dir / f"{prefix}_recordingsMeta.csv"
    missing = [p.name for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"exiD recording {prefix}: {missing} missing in {raw_dir}; expected {LAYOUT}")
    return files


def _first_entry(values: pd.Series) -> np.ndarray:
    """First entry of semicolon-separated lists, as floats (NaN where empty)."""
    first = values.astype(str).str.split(";").str[0]
    return pd.to_numeric(first, errors="coerce").to_numpy(dtype=np.float64, copy=True)


def _next_to_change(values: pd.Series, vehicle: np.ndarray) -> np.ndarray:
    """True where a value differs from the previous or the next frame of the same vehicle
    (the frame must be sorted by vehicle and time)."""
    key = values.astype(str).to_numpy()
    changed = (key[1:] != key[:-1]) & (vehicle[1:] == vehicle[:-1])
    out = np.zeros(len(key), dtype=bool)
    out[1:] |= changed
    out[:-1] |= changed
    return out


def _leader_ids(lead: pd.Series) -> pd.Series:
    return lead.astype(np.int64).astype(str).where(lead.to_numpy() >= 0)


def _dhw_mode(tracks: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    p = tracks["traveledDistance"].to_numpy(dtype=np.float64)
    length = frame["length"].to_numpy()
    frame["x_front"], frame["x_rear"] = p + length / 2.0, p - length / 2.0
    gap = tracks["leadDHW"].to_numpy(dtype=np.float64, copy=True)
    has_gap = gap >= 0.0
    frame["leader"] = _leader_ids(tracks["leadId"]).where(has_gap)
    gap[~has_gap | _next_to_change(frame["leader"], tracks["trackId"].to_numpy())] = np.nan
    frame["gap"] = gap
    if "laneChange" in tracks.columns:
        changes = pd.to_numeric(tracks["laneChange"], errors="coerce").fillna(0).astype(bool)
        frame["lane"] = changes.groupby(tracks["trackId"]).cumsum().to_numpy()

    grid = resample_tracks(frame)
    lead = grid[["group", "vehicle_id", "g", "x_rear"]].rename(columns={"vehicle_id": "leader", "x_rear": "x_rear_leader"})
    x_rear_leader = grid[["group", "leader", "g"]].merge(lead, on=["group", "leader", "g"], how="left")["x_rear_leader"]
    x_rear_leader = x_rear_leader.to_numpy(dtype=np.float64)
    gap = grid["gap"].to_numpy(dtype=np.float64)
    ok = np.isfinite(x_rear_leader) & np.isfinite(gap)
    grid["x_front"] = np.where(ok, x_rear_leader - gap, grid["x_front"].to_numpy())
    grid["leader"] = grid["leader"].where(ok)
    return grid.drop(columns=["gap", "g"])


def _lanelet_mode(tracks: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    lanelet = _first_entry(tracks["laneletId"])
    position = _first_entry(tracks["lonLaneletPos"])
    position[_next_to_change(pd.Series(lanelet), tracks["trackId"].to_numpy())] = np.nan
    length = frame["length"].to_numpy()
    frame["x_front"], frame["x_rear"] = position + length / 2.0, position - length / 2.0
    frame["lane"] = lanelet
    if "leadId" in tracks.columns:
        leader = _leader_ids(tracks["leadId"])
        where = pd.DataFrame({"leader": frame["vehicle_id"], "frame": tracks["frame"], "leader_lanelet": lanelet})
        leader_lanelet = pd.DataFrame({"leader": leader, "frame": tracks["frame"]}).merge(
            where, on=["leader", "frame"], how="left"
        )["leader_lanelet"]
        frame["leader"] = leader.where(leader_lanelet.to_numpy() == lanelet)
    return frame


def _recording_frame(files: dict[str, Path]) -> tuple[pd.DataFrame, float, str]:
    """Tracks frame of one recording (on the 10 Hz grid in mode leadDHW), frame rate and mode."""
    header = pd.read_csv(files["tracks"], nrows=0).columns
    if all(c in header for c in DHW_COLUMNS):
        mode = "leadDHW"
    elif all(c in header for c in LANELET_COLUMNS):
        mode = "lonLaneletPos"
    else:
        raise ValueError(
            f"{files['tracks'].name}: the map-based enrichment columns are missing; exiD needs {DHW_COLUMNS} or {LANELET_COLUMNS}"
        )
    columns = [*BASE_COLUMNS, *(DHW_COLUMNS if mode == "leadDHW" else LANELET_COLUMNS), *OPTIONAL_COLUMNS]
    tracks = pd.read_csv(
        files["tracks"], usecols=lambda c: c in columns, dtype={c: str for c in LANELET_COLUMNS}
    )
    recording = pd.read_csv(files["recordingMeta"], usecols=["recordingId", "locationId", "frameRate"]).iloc[0]
    rate = float(recording["frameRate"])
    vehicles = pd.read_csv(files["tracksMeta"], usecols=["trackId", "class"])
    tracks = tracks.merge(vehicles, on="trackId", how="left", validate="many_to_one")
    tracks = tracks.sort_values(["trackId", "frame"], kind="stable").reset_index(drop=True)
    frame = pd.DataFrame(
        {
            "group": f"{int(recording['recordingId']):02d}",
            "site": str(int(recording["locationId"])),
            "vehicle_id": tracks["trackId"].astype(str),
            "t": tracks["frame"] / rate,
            "v": tracks["lonVelocity"].to_numpy(dtype=np.float64),
            "a": tracks["lonAcceleration"].to_numpy(dtype=np.float64),
            "length": tracks["length"].to_numpy(dtype=np.float64),
            "vehicle_class": tracks["class"],
        }
    )
    frame = _dhw_mode(tracks, frame) if mode == "leadDHW" else _lanelet_mode(tracks, frame)
    return frame, rate, mode


def load_pairs(
    raw_dir: str | Path,
    recordings: Sequence[int] | None = None,
    max_groups: int | None = None,
    dataset: str = DATASET,
) -> Iterator[PairSeries]:
    """Follower series of exiD, one recording at a time.

    ``recordings`` selects recording ids (all by default); ``max_groups`` stops after the
    first n recordings; ``dataset`` is the name in the ids.
    """
    raw_dir = Path(raw_dir)
    prefixes = sorted(p.name.removesuffix("_tracks.csv") for p in raw_dir.glob("*_tracks.csv"))
    if recordings is not None:
        wanted = {int(r) for r in recordings}
        prefixes = [p for p in prefixes if p.isdigit() and int(p) in wanted]
    if not prefixes:
        raise FileNotFoundError(f"no exiD recording in {raw_dir}; expected {LAYOUT}")
    for prefix in prefixes[:max_groups]:
        frame, rate, mode = _recording_frame(_files(raw_dir, prefix))
        meta = {"native_rate_hz": rate, "source_file": f"{prefix}_tracks.csv", "ids": "source", "gap_source": mode}
        yield from tracks_to_pairs(frame, dataset, extras=["vehicle_class"], leader_extras=["vehicle_class"], meta=meta)
