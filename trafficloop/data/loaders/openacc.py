"""OpenACC car-following campaigns of the JRC: vehicle platoons with commercial ACC systems.

Source: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/
(CC BY 4.0), one folder per campaign with one CSV per run; docs/data_layout.md (part A) has the
verified layout.

File format: metadata lines ``Date,DD,MM,YYYY``, ``Vehicle_order,<name>,...``,
``Number_of_vehicles,n``, ``ACC,<0|1|2|blank>`` and ``Distance_setting,<setting>``, padded
with empty fields; then a header and one row per 0.1 s with ``Time`` [s] and, for platoon
position ``k`` (1 = leader), ``Speed<k>`` [m/s], positions, ``Driver<k>`` (``ACC`` or
``Human``) and ``IVS<k>``, the spacing [m] between vehicles ``k`` and ``k + 1`` "after bumper
to bumper correction" (campaign notes). Missing values are empty fields; a vehicle without
data has no columns at all (ZalaZone).

Decisions:

* The number of metadata lines is not assumed: the header is the first line whose first
  field is ``Time``. CSV files without such a line (vehicle specifications) are skipped.
* One file is one group; the campaign folder is the site (:data:`CAMPAIGNS`). Pair ``k`` is
  vehicle ``k + 1`` behind vehicle ``k`` with the gap ``IVS<k>``; it needs the columns
  ``Speed<k>``, ``Speed<k+1>`` and ``IVS<k>``. The file ``Time`` is kept: it lies on the 0.1 s
  grid in every published file, other files are interpolated to the grid.
* ``follower_id = openacc/<site>/<vehicle name>`` (lower case, spaces to underscores): the car
  with its controller is the driver of the driver-level split, the same in every file of a
  campaign. A file without vehicle names gets ``veh<k>`` and ``meta["ids"] = "pseudo"``.
* The driving mode of the follower is the segment key, so a change of mode ends an event,
  and ``extras["acc_on"]`` holds it per sample (``meta["acc_on"]`` of an event): from
  ``Driver<k+1>`` when the file has it, otherwise from the file flag ``ACC`` (0: manual
  throughout, 1: ACC throughout, 2 or blank: unknown, ``None``); ``meta["acc_on_source"]``
  says which.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from trafficloop.data.extraction import PairSeries, positions_from_gap, to_grid
from trafficloop.utils import resolve_path

POSITION_SOURCE = "gap"
DATASET = "openacc"
DEFAULT_RAW_DIR = "../cf-stability/data/raw/openacc"
SOURCE_URL = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/"
GAP_DEFINITION = "bumper to bumper (IVS after bumper-to-bumper correction, JRC campaign notes)"
MAX_METADATA_LINES = 20
MODE_CODES = {"acc": 1.0, "human": 0.0}
# file flag ACC: 0 manual and 1 ACC during the whole run; 2 (mixed) and blank say nothing per vehicle
FLAG_CODES = {"0": 0.0, "1": 1.0}
_VEHICLE_COLUMN = re.compile(r"(Speed|IVS|Driver)(\d+)")


@dataclass(frozen=True)
class Campaign:
    site: str
    car_following: bool = True
    speed_factor: float = 1.0  # to m/s
    # side file naming, per run with Distance_setting "SL", the vehicles with the long setting
    long_setting_file: str | None = None


# Speed is m/s in every campaign: the notes say so and |d(E, N)/dt| / Speed lies within
# 0.986-1.027 in all 139 car-following files. Absent vehicles, empty fields, missing Driver
# columns and the ZalaZone column order (Speed, E, N, Lon, Lat, Alt; Lat/Lon in degrees) need
# no entry: columns are selected by name and missing values are NaN.
CAMPAIGNS: dict[str, Campaign] = {
    "AstaZero": Campaign("astazero"),  # Driver<k> for all; the leader is on ACC even in manual runs
    "Casale": Campaign("casale"),  # only Driver2; vehicle 1 has no Alt and U
    "Cherasco": Campaign("cherasco"),  # two files lack Alt, U, VE, VN, VU; negative IVS in six files
    "JRC low speed": Campaign("jrc_low_speed"),  # Driver absent for some followers, in four files for all
    "Vicolungo": Campaign("vicolungo"),  # negative IVS in two files
    # no Driver column (the file flag decides); vehicles missing from a run have no columns
    "ZalaZone": Campaign("zalazone", long_setting_file="Long setting vehs in mixed timegap experiment handling.csv"),
    # one car with 20 drivers: no leader, no IVS
    "One_vehicle_multiple_drivers_on_road_campaign": Campaign("one_vehicle", car_following=False),
}

_REPORT: dict = {}


@dataclass
class RunInfo:
    """Metadata lines of one run."""

    header_line: int
    vehicles: list[str]
    n_vehicles: int | None
    acc_flag: str
    distance_setting: str
    date: str | None


def read_run_info(path: Path) -> RunInfo | None:
    """Metadata of a data file; ``None`` when no line starts with the field ``Time``."""
    fields: dict[str, list[str]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for number, line in enumerate(fh):
            cells = [cell.strip() for cell in line.rstrip("\r\n").split(",")]
            if cells[0] == "Time":
                break
            if number >= MAX_METADATA_LINES:
                return None
            if cells[0]:
                fields[cells[0]] = [cell for cell in cells[1:] if cell]
        else:
            return None
    day_month_year = fields.get("Date", [])
    date = None
    if len(day_month_year) == 3 and all(part.isdigit() for part in day_month_year):
        day, month, year = (int(part) for part in day_month_year)
        date = f"{year:04d}-{month:02d}-{day:02d}"
    count = fields.get("Number_of_vehicles", [])
    return RunInfo(
        header_line=number,
        vehicles=fields.get("Vehicle_order", []),
        n_vehicles=int(count[0]) if count and count[0].isdigit() else None,
        acc_flag=(fields.get("ACC") or [""])[0],
        distance_setting=" ".join(fields.get("Distance_setting", [])),
        date=date,
    )


def normalise_name(name: str) -> str:
    return "_".join(name.strip().lower().split())


def read_long_settings(path: Path) -> dict[str, set[str]]:
    """Run (file stem) -> normalised names of the vehicles with the long time-gap setting."""
    table = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False).iloc[:, :2]
    return {run.strip(): {normalise_name(name) for name in names.split("/")} for run, names in table.itertuples(index=False)}


def last_report() -> dict:
    """Counts of the last :func:`load_pairs` run (filled while it is consumed)."""
    return {key: dict(value) if isinstance(value, dict) else value for key, value in _REPORT.items()}


def load_pairs(
    raw_dir: str | Path | None = None,
    *,
    campaigns: Sequence[str] | None = None,
    max_groups: int | None = None,
) -> Iterator[PairSeries]:
    """One :class:`PairSeries` per follower of every run.

    ``campaigns`` selects campaign folders (default: every car-following campaign of
    :data:`CAMPAIGNS` present in ``raw_dir``); ``max_groups`` limits the number of files.
    """
    root = resolve_path(DEFAULT_RAW_DIR if raw_dir is None else raw_dir)
    names = [name for name in CAMPAIGNS if CAMPAIGNS[name].car_following] if campaigns is None else list(campaigns)
    unknown = [name for name in names if name not in CAMPAIGNS]
    if unknown:
        raise ValueError(f"unknown OpenACC campaigns {unknown}; known: {sorted(CAMPAIGNS)}")
    names = [name for name in names if CAMPAIGNS[name].car_following]
    missing = [name for name in names if not (root / name).is_dir()]
    if (campaigns is not None and missing) or len(missing) == len(names):
        raise FileNotFoundError(
            f"OpenACC: campaign folders {missing} not found in {root}; expected <raw_dir>/<campaign>/*.csv "
            f"as published at {SOURCE_URL}"
        )
    _REPORT.clear()
    _REPORT.update(files=0, pairs=0, pairs_without_columns=0, files_per_site={}, pairs_per_site={})
    for name in names:
        if name in missing:
            continue
        campaign = CAMPAIGNS[name]
        folder = root / name
        long_setting = {}
        if campaign.long_setting_file and (folder / campaign.long_setting_file).is_file():
            long_setting = read_long_settings(folder / campaign.long_setting_file)
        for path in sorted(folder.glob("*.csv")):
            info = read_run_info(path)
            if info is None:
                continue
            if max_groups is not None and _REPORT["files"] >= max_groups:
                return
            _REPORT["files"] += 1
            _REPORT["files_per_site"][campaign.site] = _REPORT["files_per_site"].get(campaign.site, 0) + 1
            yield from _run_pairs(path, root, campaign, info, long_setting.get(path.stem, set()))


def _run_pairs(path: Path, root: Path, campaign: Campaign, info: RunInfo, long_setting: set[str]) -> Iterator[PairSeries]:
    frame = pd.read_csv(
        path,
        skiprows=info.header_line,
        encoding="utf-8-sig",
        usecols=lambda column: column == "Time" or _VEHICLE_COLUMN.fullmatch(column) is not None,
    )
    positions = [int(c[5:]) for c in frame.columns if c.startswith("Speed")]
    n = max(len(info.vehicles), info.n_vehicles or 0, *positions)
    if info.vehicles and len(info.vehicles) < max(positions, default=0):
        raise ValueError(f"{path}: Vehicle_order names {len(info.vehicles)} vehicles, Speed columns go to {max(positions)}")
    names = [normalise_name(v) for v in info.vehicles] or [f"veh{k}" for k in range(1, n + 1)]
    t = frame["Time"].to_numpy(dtype=np.float64)
    native_rate = round(1.0 / float(np.median(np.diff(t))), 6) if len(t) > 1 else None
    prefix = f"{DATASET}/{campaign.site}/"
    for k in range(1, n):
        if any(c not in frame.columns for c in (f"Speed{k}", f"Speed{k + 1}", f"IVS{k}")):
            _REPORT["pairs_without_columns"] += 1
            continue
        mode, mode_source = _follower_mode(frame, k + 1, info.acc_flag)
        tg, cols = to_grid(
            t,
            {
                "s": frame[f"IVS{k}"].to_numpy(dtype=np.float64),
                "v": frame[f"Speed{k + 1}"].to_numpy(dtype=np.float64) * campaign.speed_factor,
                "v_lead": frame[f"Speed{k}"].to_numpy(dtype=np.float64) * campaign.speed_factor,
                "mode": mode,
            },
            nearest=("mode",),
        )
        known = np.isfinite(cols["mode"])
        is_acc = cols["mode"] == MODE_CODES["acc"]
        acc_on = np.where(known, is_acc, None)
        segment_key = None
        if mode_source == "driver column":
            segment_key = np.where(known, np.where(is_acc, "acc", "human"), "unknown").astype(object)
        setting = info.distance_setting
        if setting == "SL":  # mixed run: short setting except the vehicles named in the side file
            setting = "L" if names[k] in long_setting else "S"
        x_lead, x_follower = positions_from_gap(cols["v_lead"], cols["s"])
        _REPORT["pairs"] += 1
        _REPORT["pairs_per_site"][campaign.site] = _REPORT["pairs_per_site"].get(campaign.site, 0) + 1
        yield PairSeries(
            dataset=DATASET,
            site=campaign.site,
            follower_id=prefix + names[k],
            t=tg,
            x_follower=x_follower,
            v=cols["v"],
            leader_id=np.full(len(tg), prefix + names[k - 1], dtype=object),
            x_lead=x_lead,
            v_lead=cols["v_lead"],
            segment_key=segment_key,
            extras={"acc_on": acc_on},
            meta={
                "source_file": path.relative_to(root).as_posix(),
                "group": path.relative_to(root).with_suffix("").as_posix(),  # the run of the platoon
                "native_rate_hz": native_rate,
                "ids": "source" if info.vehicles else "pseudo",
                "gap_definition": GAP_DEFINITION,
                "date": info.date,
                "platoon_position": k + 1,
                "acc_setting": info.acc_flag,
                "distance_setting": setting,
                "acc_on_source": mode_source,
            },
        )


def _follower_mode(frame: pd.DataFrame, position: int, acc_flag: str) -> tuple[np.ndarray, str]:
    """Per-sample mode code of a vehicle (1 ACC, 0 human, NaN unknown) and where it comes from."""
    column = f"Driver{position}"
    if column in frame.columns:
        text = frame[column].astype("string").str.strip().str.lower()
        return text.map(MODE_CODES).to_numpy(dtype=np.float64, na_value=np.nan), "driver column"
    if acc_flag in FLAG_CODES:
        return np.full(len(frame), FLAG_CODES[acc_flag]), "file flag"
    return np.full(len(frame), np.nan), "unknown"
