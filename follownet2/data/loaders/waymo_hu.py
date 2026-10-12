"""Car-following pairs of the Waymo Open Dataset extracted by Hu et al. (2022).

Source: Mendeley Data 10.17632/wfn2c3437n, version 2 (CC BY 4.0); docs/data_layout.md (part A) has
the verified layout. The authors' processing code (``waymo_data_process.py``) documents it:

* ``all_seg_paired_cf_trj_final_with_large_vehicle.csv``: one row per vehicle of a pair and
  time stamp, columns ``segment_id, local_veh_id, length, local_time, follower_id,
  leader_id, filter_pos, filter_speed, filter_accer`` [-, -, m, s, -, -, m, m/s, m/s^2]. Every
  pair ``(segment_id, follower_id, leader_id)`` is one contiguous block of rows, the leader
  first. ``filter_pos`` is the distance travelled by the vehicle CENTRE; the leader's is
  offset by the initial centre distance, so the net gap is ``filter_pos_leader -
  filter_pos_follower - (length_leader + length_follower) / 2`` (``net_distance`` of the
  code). Time stamps are 20 s segments at about 10 Hz with a jitter of 0.01 s.
* Local ids are unique within a 20 s segment only; id 0 is the automated vehicle, the
  Waymo car, whose length is fixed at 5.18 m.
* ``AV_HV_and_HV_AV_CF_pair_record_with_large_vehicle.csv`` has one row per segment: the
  human that follows the automated vehicle (``follower``) and the human it follows
  (``leader``), 0 for none; ``HV_HV_pair_record_with_large_vehicle.csv`` lists the pairs of
  two humans. The ``*_code`` columns are exclusion reason codes and are not used.

Decisions:

* The pair records decide which vehicle of a pair is automated; pairs missing from them are
  skipped and counted in :func:`last_report`. Option ``followers``: ``"human"`` (default),
  ``"automated"`` or ``"all"``.
* Human drivers: ``waymo_hu/<site>/<segment>.<local id>``. The automated vehicle is one and
  the same driver in every segment: ``waymo_hu/<site>/waymo_driver`` as follower and as
  leader, with ``meta["automated"]`` and ``meta["leader_automated"]``.
* The site is ``"all"``: the pair files carry no location (the per-segment ``location`` is
  only in the 2.1 GB object files, which are not read).
* The centre-to-centre distance is converted to the net gap with the published lengths;
  ``filter_accer`` is the source acceleration. The negative speeds of this version (7.3 % of
  the samples, 0.25 % below -0.1 m/s; fixed in version 3) are left: extraction cuts
  invalid samples.
* The file is streamed in chunks; a segment is a group (``max_groups`` counts segments).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from follownet2.data.extraction import PairSeries, positions_from_gap, to_grid
from follownet2.utils import resolve_path

POSITION_SOURCE = "gap"
DATASET = "waymo_hu"
DEFAULT_RAW_DIR = "../cf-stability/data/raw/waymo"
SOURCE_URL = "https://doi.org/10.17632/wfn2c3437n.2"
TRAJECTORY_FILE = "all_seg_paired_cf_trj_final_with_large_vehicle.csv"
AV_RECORD_FILE = "AV_HV_and_HV_AV_CF_pair_record_with_large_vehicle.csv"
HV_RECORD_FILE = "HV_HV_pair_record_with_large_vehicle.csv"
COLUMNS = ("segment_id", "local_veh_id", "length", "local_time", "follower_id", "leader_id",
           "filter_pos", "filter_speed", "filter_accer")
KEY = ["segment_id", "follower_id", "leader_id"]
AV_LOCAL_ID = 0
AUTOMATED_DRIVER = "waymo_driver"
SITE = "all"
FOLLOWERS = ("human", "automated", "all")
GAP_DEFINITION = "bumper to bumper: centre distance of filter_pos minus half of both published lengths"
CHUNK_ROWS = 200_000

_REPORT: dict = {}


def pair_roles(raw_dir: Path) -> dict[tuple[int, int, int], tuple[bool, bool]]:
    """``(segment, follower, leader) -> (follower automated, leader automated)`` from the records."""
    roles = {}
    records = pd.read_csv(raw_dir / AV_RECORD_FILE)
    for segment, follower, leader in records[["segment_id", "follower", "leader"]].itertuples(index=False):
        if leader != 0:  # the automated vehicle follows this human
            roles[(int(segment), AV_LOCAL_ID, int(leader))] = (True, False)
        if follower != 0:  # this human follows the automated vehicle
            roles[(int(segment), int(follower), AV_LOCAL_ID)] = (False, True)
    records = pd.read_csv(raw_dir / HV_RECORD_FILE)
    for segment, follower, leader in records[["segment_id", "follower", "leader"]].itertuples(index=False):
        roles[(int(segment), int(follower), int(leader))] = (False, False)
    return roles


def pair_blocks(path: Path, chunk_rows: int = CHUNK_ROWS) -> Iterator[pd.DataFrame]:
    """The rows of one pair at a time, reading ``chunk_rows`` rows at a time."""
    carry = None
    for chunk in pd.read_csv(path, usecols=list(COLUMNS), chunksize=chunk_rows):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        key = chunk[KEY].to_numpy(dtype=np.float64)
        starts = np.flatnonzero(np.r_[True, np.any(key[1:] != key[:-1], axis=1)])
        for start, stop in zip(starts[:-1], starts[1:]):
            yield chunk.iloc[start:stop]
        carry = chunk.iloc[starts[-1]:]
    if carry is not None and len(carry):
        yield carry


def last_report() -> dict:
    """Counts of the last :func:`load_pairs` run (filled while it is consumed)."""
    return dict(_REPORT)


def load_pairs(
    raw_dir: str | Path | None = None,
    *,
    followers: str = "human",
    max_groups: int | None = None,
) -> Iterator[PairSeries]:
    """One :class:`PairSeries` per car-following pair on the 10 Hz grid.

    ``followers`` selects pairs by the follower (``"human"``, ``"automated"``, ``"all"``);
    ``max_groups`` limits the number of 20 s segments.
    """
    if followers not in FOLLOWERS:
        raise ValueError(f"followers must be one of {FOLLOWERS}, not {followers!r}")
    root = resolve_path(DEFAULT_RAW_DIR if raw_dir is None else raw_dir)
    missing = [name for name in (TRAJECTORY_FILE, AV_RECORD_FILE, HV_RECORD_FILE) if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Waymo pairs of Hu et al.: {missing} not found in {root}; expected <raw_dir>/{TRAJECTORY_FILE}, "
            f"<raw_dir>/{AV_RECORD_FILE} and <raw_dir>/{HV_RECORD_FILE} of {SOURCE_URL}"
        )
    roles = pair_roles(root)
    _REPORT.clear()
    _REPORT.update(pairs_read=0, pairs=0, pairs_not_in_records=0, pairs_other_followers=0, pairs_without_common_time=0,
                   segments=0)
    segments: set[int] = set()
    seen: set[tuple[int, int, int]] = set()
    for rows in pair_blocks(root / TRAJECTORY_FILE, CHUNK_ROWS):
        key = tuple(int(rows[column].iloc[0]) for column in KEY)
        if key in seen:
            raise ValueError(f"{TRAJECTORY_FILE}: the rows of pair {key} are not one contiguous block")
        seen.add(key)
        _REPORT["pairs_read"] += 1
        if key not in roles:
            _REPORT["pairs_not_in_records"] += 1
            continue
        automated, leader_automated = roles[key]
        if followers != "all" and automated != (followers == "automated"):
            _REPORT["pairs_other_followers"] += 1
            continue
        if key[0] not in segments:
            if max_groups is not None and len(segments) >= max_groups:
                continue
            segments.add(key[0])
            _REPORT["segments"] = len(segments)
        pair = _pair_series(rows, key, automated, leader_automated)
        if pair is None:
            _REPORT["pairs_without_common_time"] += 1
            continue
        _REPORT["pairs"] += 1
        yield pair


def _driver_id(segment: int, local_id: int, automated: bool) -> str:
    return f"{DATASET}/{SITE}/{AUTOMATED_DRIVER if automated else f'{segment}.{local_id}'}"


def _pair_series(
    rows: pd.DataFrame, key: tuple[int, int, int], automated: bool, leader_automated: bool
) -> PairSeries | None:
    segment, follower, leader = key
    follower_rows = rows[rows["local_veh_id"] == follower]
    leader_rows = rows[rows["local_veh_id"] == leader]
    both = follower_rows.merge(leader_rows, on="local_time", suffixes=("_f", "_l"))
    if both.empty:
        return None
    length, leader_length = float(follower_rows["length"].iloc[0]), float(leader_rows["length"].iloc[0])
    centre = both["filter_pos_l"].to_numpy() - both["filter_pos_f"].to_numpy()
    t, cols = to_grid(
        both["local_time"].to_numpy(),
        {
            "s": centre - 0.5 * (length + leader_length),
            "v": both["filter_speed_f"].to_numpy(),
            "a": both["filter_accer_f"].to_numpy(),
            "v_lead": both["filter_speed_l"].to_numpy(),
        },
    )
    x_lead, x_follower = positions_from_gap(cols["v_lead"], cols["s"])
    return PairSeries(
        dataset=DATASET,
        site=SITE,
        follower_id=_driver_id(segment, follower, automated),
        t=t,
        x_follower=x_follower,
        v=cols["v"],
        leader_id=np.full(len(t), _driver_id(segment, leader, leader_automated), dtype=object),
        x_lead=x_lead,
        v_lead=cols["v_lead"],
        a=cols["a"],
        extras={"leader_length": np.full(len(t), leader_length)},
        meta={
            "source_file": TRAJECTORY_FILE,
            "group": str(segment),  # the 20 s segment: its vehicles drive at the same time
            "native_rate_hz": 10.0,
            "ids": "source",
            "gap_definition": GAP_DEFINITION,
            "segment": segment,
            "automated": automated,
            "leader_automated": leader_automated,
            "follower_length": length,
        },
    )
