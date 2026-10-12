"""NGSIM loader: fabricated Socrata tables (cf-stability parquet, download-tool parquet, CSV
export) and the first file of us-101.

``frame_id`` counts the frames of an original 15-minute file and restarts in every file, as in
the real tables; files may overlap in time.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml

from follownet2.data.extraction import ExtractionConfig, extract_all
from follownet2.data.loaders.ngsim import FEET, FIELDS, load_pairs
from follownet2.utils import REPO_ROOT, resolve_path

T0 = 1_118_846_979_700  # ms, first time stamp of the real us-101 table
PARQUET_COLUMNS = [  # cf-stability: all 25 fields
    "vehicle_id", "frame_id", "total_frames", "global_time", "v_class", "lane_id", "preceding", "following",
    "local_x", "local_y", "global_x", "global_y", "v_length", "v_width", "v_vel", "v_acc", "space_headway",
    "time_headway", "o_zone", "d_zone", "int_id", "section_id", "direction", "movement", "location",
]
DOWNLOAD_COLUMNS = [  # conf/sources.yaml, tables ngsim_<location>_raw
    "vehicle_id", "frame_id", "global_time", "local_y", "v_vel", "v_acc", "lane_id", "preceding", "following",
    "space_headway", "time_headway", "v_length", "v_class",
]
DOWNLOAD_INTEGERS = ["vehicle_id", "frame_id", "global_time", "lane_id", "preceding", "following", "v_class"]
EXPORT_HEADER = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time", "Local_X", "Local_Y", "Global_X", "Global_Y",
    "v_length", "v_Width", "v_Class", "v_Vel", "v_Acc", "Lane_ID", "O_Zone", "D_Zone", "Int_ID", "Section_ID",
    "Direction", "Movement", "Preceding", "Following", "Space_Headway", "Time_Headway", "Location",
]
V0, ACC = 40.0, 0.5  # ft/s, ft/s^2: every platoon accelerates uniformly, so its gaps stay constant


def platoon(ids, lengths, gaps, start=0.0, duration=16.0, lane=2, classes=None, location="us-101", file_start=0.0):
    """Rows of vehicles ``ids`` in one lane; ids[0] leads and ids[i] follows ids[i-1] at the net
    gap ``gaps[i-1]`` [ft]. Times are in s after T0; ``file_start`` is the start of the original
    file, from which ``frame_id`` counts."""
    tau = np.arange(round(duration * 10) + 1) * 0.1
    t_ms = T0 + round(start * 1000) + np.arange(len(tau), dtype=np.int64) * 100
    y = 1500.0 + V0 * tau + 0.5 * ACC * tau**2
    classes = classes or [2] * len(ids)
    frames = []
    for i, vid in enumerate(ids):
        headway = 0.0 if i == 0 else lengths[i - 1] + gaps[i - 1]
        y = y - headway
        frames.append(pd.DataFrame({
            "vehicle_id": vid, "frame_id": (t_ms - T0 - round(file_start * 1000)) // 100 + 1, "total_frames": 0,
            "global_time": t_ms, "v_class": classes[i], "lane_id": lane, "preceding": 0 if i == 0 else ids[i - 1],
            "following": ids[i + 1] if i + 1 < len(ids) else 0, "local_x": 12.0 * lane - 6.0, "local_y": y,
            "global_x": 6451000.0 + y, "global_y": 1872000.0 + y, "v_length": float(lengths[i]), "v_width": 6.0,
            "v_vel": V0 + ACC * tau, "v_acc": ACC, "space_headway": headway,
            "time_headway": np.round(headway / (V0 + ACC * tau), 2), "o_zone": "NA", "d_zone": "NA",
            "int_id": "NA", "section_id": "NA", "direction": "NA", "movement": "NA", "location": location,
        }))
    return pd.concat(frames, ignore_index=True).round(6)


def write(raw_dir: Path, rows: pd.DataFrame, fmt: str = "parquet") -> Path:
    raw_dir.mkdir(exist_ok=True)
    rows = rows.copy()
    file = rows["global_time"] - 100 * rows["frame_id"]
    rows["total_frames"] = rows.groupby([rows["location"], file, rows["vehicle_id"]])["vehicle_id"].transform("size")
    for location, part in rows.groupby("location"):
        path = raw_dir / f"ngsim_{location}_raw.parquet"
        if fmt == "parquet":  # without the pandas metadata, to keep the file small
            table = pa.Table.from_pandas(part[PARQUET_COLUMNS], preserve_index=False).replace_schema_metadata()
            pq.write_table(table, path, compression="zstd", store_schema=False)
        elif fmt == "download":  # as follownet2/data/download.py writes a Socrata table
            table = pa.Table.from_pandas(part[DOWNLOAD_COLUMNS].astype({c: "Int64" for c in DOWNLOAD_INTEGERS}), preserve_index=False)
            pq.write_table(table, path, compression="zstd")
    if fmt == "csv":
        rows[[h.lower() for h in EXPORT_HEADER]].to_csv(raw_dir / "ngsim_export.csv", header=EXPORT_HEADER, index=False)
    return raw_dir


def run(raw_dir: Path, **options):
    options.setdefault("locations", ("us-101",))
    events = extract_all(load_pairs(raw_dir, **options), ExtractionConfig())
    for ev in events:
        ev.validate()
    return events


def by_follower(events):
    return {ev.follower_id.split("/")[2]: ev for ev in events}


LANE_2 = dict(ids=[1, 2, 3], lengths=[14.0, 16.0, 15.0], gaps=[45.0, 60.0])


@pytest.fixture(scope="module")
def mixed(tmp_path_factory) -> Path:
    """One file: autos in lane 2, an auto behind a truck in lane 3, a truck behind an auto in
    lane 4 and two autos on the on-ramp (lane 7)."""
    rows = pd.concat([
        platoon(**LANE_2, lane=2),
        platoon([4, 5], [45.0, 15.0], [70.0], lane=3, classes=[3, 2]),
        platoon([6, 7], [15.0, 50.0], [80.0], lane=4, classes=[2, 3]),
        platoon([8, 9], [15.0, 15.0], [90.0], lane=7),
    ])
    return write(tmp_path_factory.mktemp("ngsim") / "mixed", rows)


def lane_2_events(raw_dir: Path):
    return [ev for ev in run(raw_dir) if ev.meta["lane"] == 2]


def test_platoon(mixed):
    events = lane_2_events(mixed)
    assert sorted(ev.event_id for ev in events) == [
        "ngsim/us-101/us-101-p0.2|us-101-p0.1|0",
        "ngsim/us-101/us-101-p0.3|us-101-p0.2|0",
    ]
    for ev in events:
        k = int(ev.follower_id[-1]) - 1
        assert len(ev) == 161 and ev.site == "us-101" and ev.dataset == "ngsim" and ev.meta["t0"] == 0.0
        np.testing.assert_allclose(ev.s, LANE_2["gaps"][k - 1] * FEET, atol=1e-6)
        np.testing.assert_allclose(ev.v, (V0 + ACC * ev.t) * FEET, atol=1e-9)
        np.testing.assert_allclose(ev.a, ACC * FEET, atol=1e-12)
        assert ev.meta["follower_length"] == pytest.approx(LANE_2["lengths"][k] * FEET)
        assert ev.meta["leader_length"] == pytest.approx(LANE_2["lengths"][k - 1] * FEET)
        assert ev.meta["vehicle_class"] == "auto" and ev.meta["leader_vehicle_class"] == "auto"
        assert ev.meta["a_source"] == "dataset" and ev.meta["native_rate_hz"] == 10 and ev.meta["ids"] == "source"
        assert ev.meta["source_file"] == "ngsim_us-101_raw.parquet" and ev.meta["group"] == "us-101-p0"


def test_leader_of_another_class(mixed):
    events = by_follower(run(mixed))
    assert "us-101-p0.7" not in events  # a truck is no follower by default
    meta = events["us-101-p0.5"].meta
    assert meta["vehicle_class"] == "auto" and meta["leader_vehicle_class"] == "truck"
    assert meta["leader_length"] == pytest.approx(45.0 * FEET)
    trucks = by_follower(run(mixed, follower_classes=(2, 3)))
    assert trucks["us-101-p0.7"].meta["vehicle_class"] == "truck"


def test_only_mainline_lanes_are_followers(mixed):
    assert "us-101-p0.9" not in by_follower(run(mixed))  # us-101 lane 7: on-ramp
    np.testing.assert_allclose(by_follower(run(mixed, lanes=None))["us-101-p0.9"].s, 90.0 * FEET, atol=1e-6)


def test_exact_duplicate_rows_are_dropped(mixed, tmp_path):
    rows = platoon(**LANE_2)
    doubled = pd.concat([rows, rows.iloc[100:300]], ignore_index=True).sample(frac=1.0, random_state=0)
    dirty = run(write(tmp_path, doubled))
    clean = lane_2_events(mixed)
    assert [ev.event_id for ev in dirty] == [ev.event_id for ev in clean]
    for a, b in zip(clean, dirty):
        np.testing.assert_array_equal(a.s, b.s)
        np.testing.assert_array_equal(a.v, b.v)


def test_contradicting_rows_are_dropped(tmp_path, caplog):
    rows = platoon(**LANE_2)
    t = (rows["global_time"] - T0) / 1000.0
    other = rows[(rows["vehicle_id"] == 2) & (t > 15.05)].assign(local_y=lambda r: r["local_y"] + 3.0)
    events = run(write(tmp_path, pd.concat([rows, other])))
    assert "dropped 20 contradicting rows" in caplog.text
    # vehicle 2 has no sample after 15.0 s: both events end there
    assert [(ev.follower_id[-1], len(ev)) for ev in events] == [("2", 151), ("3", 151)]


def test_parquet_of_the_download_tool(mixed, tmp_path):
    events = run(write(tmp_path, platoon(**LANE_2), "download"))
    clean = lane_2_events(mixed)
    assert [ev.event_id for ev in events] == [ev.event_id for ev in clean]
    for a, b in zip(clean, events):
        np.testing.assert_array_equal(a.s, b.s)
        assert a.meta == b.meta


def test_csv_export_with_display_names_and_two_locations(tmp_path):
    rows = pd.concat([
        platoon([1, 2], [15.0, 15.0], [40.0], start=30.0, file_start=30.0),
        platoon([7], [15.0], [], location="i-80", duration=0.4),  # i-80 rows before the us-101 ones
    ])
    events = run(write(tmp_path, rows, "csv"), locations=("us-101", "i-80"))
    assert [ev.follower_id for ev in events] == ["ngsim/us-101/us-101-p0.2"]
    assert events[0].meta["t0"] == 0.0  # t0 is the first time stamp of the location
    assert events[0].meta["source_file"] == "ngsim_export.csv"
    np.testing.assert_allclose(events[0].s, 40.0 * FEET, atol=1e-6)


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> Path:
    """Three original files of us-101; file 1 starts at 880 s, before file 0 ends (925 s).

    File 0: 2 follows 1 at 30 ft (0-16 s); 5 follows 4 at 55 ft, 4 enters at 893 s and 5 at
    903 s. File 1: the ids 4 and 5 again (900-916 s, lane 3, 70 ft) and 2 behind 1 at 80 ft
    (950-966 s). File 2 (from 2000 s): 2 behind 1 at 90 ft.
    """
    late = platoon([4, 5], [15.0, 16.0], [55.0], start=893.0, duration=32.0)
    late = late[~((late["vehicle_id"] == 5) & (late["global_time"] < T0 + 903_000))]
    rows = pd.concat([
        platoon([1, 2], [14.0, 15.0], [30.0]),
        late,
        platoon([4, 5], [15.0, 15.0], [70.0], start=900.0, lane=3, file_start=880.0),
        platoon([1, 2], [15.0, 17.0], [80.0], start=950.0, file_start=880.0),
        platoon([1, 2], [15.0, 15.0], [90.0], start=2000.0, file_start=2000.0),
    ])
    return write(tmp_path_factory.mktemp("ngsim") / "files", rows)


def test_groups_are_the_original_files(files):
    events = by_follower(run(files))
    gaps = {name: float(np.median(ev.s)) / FEET for name, ev in events.items()}
    assert gaps == pytest.approx({"us-101-p0.2": 30, "us-101-p0.5": 55, "us-101-p1.2": 80, "us-101-p1.5": 70, "us-101-p2.2": 90})
    assert {name: ev.meta["group"] for name, ev in events.items()} == {name: name.split(".")[0] for name in events}


def test_same_vehicle_id_in_two_files(files):
    events = by_follower(run(files))
    first, second = events["us-101-p0.5"], events["us-101-p1.5"]  # both drive at 903-916 s
    assert first.leader_id == "ngsim/us-101/us-101-p0.4" and second.leader_id == "ngsim/us-101/us-101-p1.4"
    assert first.meta["lane"] == 2 and second.meta["lane"] == 3
    np.testing.assert_allclose(second.s, 70.0 * FEET, atol=1e-6)


def test_leader_that_entered_before_the_900_s_mark(files):
    follower = by_follower(run(files))["us-101-p0.5"]
    assert follower.meta["t0"] == pytest.approx(903.0) and len(follower) == 221
    np.testing.assert_allclose(follower.s, 55.0 * FEET, atol=1e-6)


def test_max_groups_counts_files(files):
    assert sorted(by_follower(run(files, max_groups=1))) == ["us-101-p0.2", "us-101-p0.5"]
    assert sorted(by_follower(run(files, max_groups=2))) == ["us-101-p0.2", "us-101-p0.5", "us-101-p1.2", "us-101-p1.5"]


def test_more_than_six_files_raise(tmp_path):
    rows = pd.concat([platoon([1], [15.0], [], start=100.0 * k, duration=0.2, file_start=100.0 * k) for k in range(7)])
    with pytest.raises(ValueError, match="more than 6 files"):
        run(write(tmp_path, rows))


def test_time_stamps_off_the_grid_raise(tmp_path):
    rows = platoon([1], [15.0], [], duration=0.3)
    rows.loc[2, "global_time"] += 50
    with pytest.raises(ValueError, match="off the 100 ms grid"):
        run(write(tmp_path, rows))


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="ngsim_<location>_raw.parquet"):
        run(tmp_path)
    write(tmp_path, platoon([1], [15.0], [], location="i-80", duration=0.2), "csv")
    with pytest.raises(FileNotFoundError, match="us-101"):
        run(tmp_path)
    pd.DataFrame({"Vehicle_ID": [1], "Location": ["us-101"]}).to_csv(tmp_path / "other.csv", index=False)
    with pytest.raises(ValueError, match="other.csv"):
        run(tmp_path)


def config() -> dict:
    return yaml.safe_load((REPO_ROOT / "conf" / "dataset" / "ngsim.yaml").read_text(encoding="utf-8"))


def test_config_matches_the_loader_and_the_download_tool():
    cfg = config()
    assert cfg["name"] == cfg["loader"] == "ngsim" and cfg["position_source"] == "track"
    inspect.signature(load_pairs).bind(cfg["raw_dir"], **cfg["options"], dataset=cfg["name"])
    sources = yaml.safe_load((REPO_ROOT / "conf" / "sources.yaml").read_text(encoding="utf-8"))
    tables = {t["name"]: t for t in sources["ngsim"]["tables"]}
    for location in cfg["options"]["locations"]:
        assert set(FIELDS) <= set(tables[f"ngsim_{location}_raw"]["select"]), location


# ----------------------------------------------------------------------------- real data
@pytest.mark.data
def test_real_us101_first_file():
    """File 0 of us-101, lanes 1-2: every leader is found and the net gap agrees with NGSIM's
    front-to-front headway."""
    raw_dir = resolve_path(config()["raw_dir"])
    path = raw_dir / "ngsim_us-101_raw.parquet"
    if not path.is_file():
        pytest.skip(f"{path} is absent")
    events = run(raw_dir, locations=("us-101",), lanes={"us-101": [1, 2]}, max_groups=1)
    assert len({ev.follower_id for ev in events}) > 500 and len(events) > 500
    assert all(ev.meta["group"] == "us-101-p0" and ev.meta["lane"] in (1, 2) for ev in events)

    table = ds.dataset(path)
    index = table.to_table(columns=["global_time", "frame_id"]).to_pandas(ignore_metadata=True)
    t0 = int(index["global_time"].min())
    first = int((index["global_time"] - 100 * index["frame_id"]).min())
    of_file = ds.field("global_time") - ds.field("frame_id") * 100 == first
    rows = table.to_table(
        columns=["vehicle_id", "global_time", "lane_id", "preceding", "space_headway"], filter=of_file
    ).to_pandas(ignore_metadata=True).drop_duplicates(["vehicle_id", "global_time"])
    present = pd.MultiIndex.from_frame(rows[["vehicle_id", "global_time"]])
    followers = rows[rows["preceding"] > 0]
    assert pd.MultiIndex.from_arrays([followers["preceding"], followers["global_time"]]).isin(present).all()

    headway = rows.set_index(["vehicle_id", "global_time"])["space_headway"]
    residuals = []
    for ev in events:
        raw_id = int(ev.follower_id.rsplit(".", 1)[1])
        times = t0 + np.round((ev.meta["t0"] + ev.t) * 1000).astype(np.int64)
        h = headway.reindex(pd.MultiIndex.from_arrays([np.full(len(ev), raw_id), times])).to_numpy()
        residuals.append(ev.s - (h * FEET - ev.meta["leader_length"]))
    assert np.mean(np.abs(np.concatenate(residuals)) < 0.5) > 0.99
