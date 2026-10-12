"""TGSIM loader: fabricated Socrata CSV exports and download-tool parquet files."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from follownet2.data.extraction import ExtractionConfig, extract_all
from follownet2.data.loaders.tgsim import DATASETS, load_pairs
from follownet2.utils import REPO_ROOT

BASE = ["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "speed_kf", "acceleration_kf", "length_smoothed", "width_smoothed", "type_most_common"]
V0, ACC = 15.0, 0.3  # m/s, m/s^2 of every vehicle: gaps stay constant
T_START = 5.0  # s, first time stamp of a run


def platoon(ids, lengths, gaps, origin, heading, lane, types, flags=None, run=None, duration=16.0):
    """Centre positions of a platoon along the unit vector ``heading`` [image axes]; ids[0] leads."""
    tau = np.arange(round(duration * 10) + 1) * 0.1
    along = V0 * tau + 0.5 * ACC * tau**2
    heading = np.asarray(heading, dtype=np.float64) / np.linalg.norm(heading)
    offset, frames = 0.0, []
    for i, vid in enumerate(ids):
        if i > 0:
            offset += lengths[i - 1] / 2 + gaps[i - 1] + lengths[i] / 2
        centre = np.asarray(origin) + np.outer(along - offset, heading)
        frame = pd.DataFrame({
            "id": vid, "time": T_START + tau, "xloc_kf": centre[:, 0], "yloc_kf": centre[:, 1], "lane_kf": lane,
            "speed_kf": V0 + ACC * tau, "acceleration_kf": ACC, "length_smoothed": lengths[i], "width_smoothed": 2.0,
            "type_most_common": types[i],
        })
        if flags is not None:
            frame[flags[0]] = flags[1][i]
        if run is not None:
            frame["run_index"] = run
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).round(9)


def run(raw_dir: Path, **options):
    events = extract_all(load_pairs(raw_dir, **options), ExtractionConfig())
    for ev in events:
        ev.validate()
    return sorted(events, key=lambda ev: (ev.follower_id, ev.meta["t0"]))


TILT = 0.96  # cosine between the road and the image y axis (heading 0.28, -0.96)


def i395():
    # eastbound traffic runs towards decreasing y in the reference image; vehicle 12 is automated
    return platoon([11, 12, 13], [4.6, 5.0, 12.5], [15.0, 22.0], origin=(90.0, 600.0), heading=(0.28, -TILT),
                   lane=-4, types=[1, 4, 3])


def along_y(gap, follower_length, leader_length):
    """Net gap measured along the image y axis instead of the tilted road."""
    return TILT * (leader_length / 2 + gap + follower_length / 2) - (leader_length + follower_length) / 2


@pytest.fixture(scope="module")
def i395_events(tmp_path_factory):
    """I-395 export with 5 corrupt duplicate (id, time) pairs of vehicle 13 at 20.2-20.6 s."""
    rows = i395()
    corrupt = rows[(rows["id"] == 13) & rows["time"].between(20.2, 20.6)].assign(
        speed_kf=0.0, acceleration_kf=-20.0, length_smoothed=15.0, type_most_common=2
    )
    raw_dir = tmp_path_factory.mktemp("tgsim_i395")
    pd.concat([rows, corrupt]).sort_values(["id", "time"])[BASE].to_csv(raw_dir / "I395-final.csv", index=False)
    logger = logging.getLogger("follownet2.data.loaders.tgsim")
    records: list[logging.LogRecord] = []
    handler = logging.Handler(logging.INFO)
    handler.emit = records.append
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.INFO)
    try:
        events = run(raw_dir, datasets=["i395"])
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    return events, [r.getMessage() for r in records]


def test_i395_along_y_against_the_image_axis(i395_events):
    events, _ = i395_events
    assert [(ev.follower_id, ev.leader_id) for ev in events] == [
        ("tgsim/i395/12", "tgsim/i395/11"), ("tgsim/i395/13", "tgsim/i395/12"),
    ]
    for ev, gap in zip(events, (15.0, 22.0)):
        np.testing.assert_allclose(ev.s, along_y(gap, ev.meta["follower_length"], ev.meta["leader_length"]), atol=1e-6)
        assert np.all(np.diff(ev.x_follower) > 0)
        np.testing.assert_allclose(ev.v, V0 + ACC * (ev.meta["t0"] - T_START + ev.t), atol=1e-9)
        np.testing.assert_allclose(ev.a, ACC, atol=1e-12)
        assert ev.meta["native_rate_hz"] == 10 and ev.meta["source_file"] == "I395-final.csv"
        assert ev.meta["lane"] == -4 and ev.meta["axis"] == "y"


def test_automation_flag_of_follower_and_leader(i395_events):
    events, _ = i395_events
    automated, behind = events[0].meta, events[1].meta
    assert automated["automated"] is True and automated["leader_automated"] is False
    assert behind["automated"] is False and behind["leader_automated"] is True
    assert automated["vehicle_class"] == "automated vehicle" and automated["leader_vehicle_class"] == "passenger car"
    assert behind["vehicle_class"] == "bus"


def test_corrupt_duplicates_are_dropped_and_counted(i395_events):
    events, messages = i395_events
    assert messages == ["tgsim i395 group i395: dropped 10 rows of 5 repeated (id, time) keys"]
    # both rows of each pair are gone: the track of vehicle 13 has a hole from 20.2 to 20.6 s
    [behind] = [ev for ev in events if ev.follower_id.endswith("13")]
    assert (behind.meta["t0"], len(behind)) == (pytest.approx(5.0), 152)
    assert behind.meta["follower_length"] == 12.5 and behind.meta["vehicle_class"] == "bus"
    assert np.all(behind.v > 0) and np.all(behind.a > 0)


def test_i294_runs_directions_and_flag_case(tmp_path):
    rows = pd.concat([
        platoon([0, 5], [4.8, 5.1], [18.0], (1000.0, 300.0), (1.0, 0.0), 3, ["small-vehicle"] * 2, ("acc", ["No", "YES"]), run=3),
        platoon([5, 0], [16.0, 4.8], [26.0], (3000.0, 250.0), (-1.0, 0.0), -3, ["large-vehicle", "small-vehicle"], ("acc", ["Yes", "No"]), run=7),
    ])
    rows[[*BASE, "acc", "run_index"]].to_csv(tmp_path / "I294_L1_final.csv", index=False)
    events = {ev.follower_id: ev for ev in run(tmp_path, datasets=["i294_l1"])}
    assert {f: ev.leader_id for f, ev in events.items()} == {
        "tgsim/i294_l1/3.5": "tgsim/i294_l1/3.0",  # id 0 leads
        "tgsim/i294_l1/7.0": "tgsim/i294_l1/7.5",  # ids reused in another run, southbound (decreasing x)
    }
    for follower, gap in [("3.5", 18.0), ("7.0", 26.0)]:
        ev = events[f"tgsim/i294_l1/{follower}"]
        np.testing.assert_allclose(ev.s, gap, atol=1e-6)
        assert np.all(np.diff(ev.x_follower) > 0) and np.all(np.diff(ev.x_lead) > 0)
    assert events["tgsim/i294_l1/3.5"].meta["automated"] is True
    southbound = events["tgsim/i294_l1/7.0"].meta
    assert southbound["leader_automated"] is True and southbound["leader_vehicle_class"] == "large-vehicle"
    assert southbound["group"] == "7" and southbound["lane"] == -3
    assert [ev.follower_id for ev in run(tmp_path, datasets=["i294_l1"], max_groups=1)] == ["tgsim/i294_l1/3.5"]


def test_principal_axis_of_a_diagonal_road(tmp_path):
    rows = platoon([1, 2], [4.5, 4.7], [12.0], (1400.0, 200.0), (-np.cos(0.6), np.sin(0.6)), 2,
                   ["small-vehicle"] * 2, ("av", ["no", "yes"]), run=1)
    rows = rows.rename(columns={"id": "ID"})  # display name of the I-90/I-94 Moving table
    rows[["ID", *BASE[1:], "av", "run_index"]].to_csv(tmp_path / "I90_94_moving_final.csv", index=False)
    [ev] = run(tmp_path, datasets=["i90_94_moving"])
    np.testing.assert_allclose(ev.s, 12.0, atol=1e-6)
    assert ev.follower_id == "tgsim/i90_94_moving/1.2" and ev.meta["automated"] is True and ev.meta["axis"] == "pca"
    # projected on the image x axis the same gap is a chord of the tilted road
    [chord] = run(tmp_path, datasets=["i90_94_moving"], axis={"i90_94_moving": "x"})
    np.testing.assert_allclose(chord.s, np.cos(0.6) * (4.5 / 2 + 12.0 + 4.7 / 2) - (4.5 + 4.7) / 2, atol=1e-6)


def test_parquet_of_the_download_tool_comes_first(tmp_path):
    rows = i395()
    frame = rows[BASE].astype({"id": "Int64", "lane_kf": "Int64", "type_most_common": "Int64"})  # conf/sources.yaml
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp_path / "i395.parquet", compression="zstd")
    rows[BASE].iloc[:3].to_csv(tmp_path / "I395-final.csv", index=False)  # three rows: no event
    events = run(tmp_path, datasets=["i395"])
    assert [ev.follower_id for ev in events] == ["tgsim/i395/12", "tgsim/i395/13"]
    assert events[0].meta["source_file"] == "i395.parquet" and events[0].meta["automated"] is True
    np.testing.assert_allclose(events[1].s, along_y(22.0, 12.5, 5.0), atol=1e-6)
    assert run(tmp_path, datasets=["i395"], files={"i395": "I395-final.csv"}) == []


def test_unsupported_and_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="I395-final.csv"):
        run(tmp_path, datasets=["i395"])
    with pytest.raises(NotImplementedError, match="foggy_bottom"):
        run(tmp_path, datasets=["foggy_bottom"])


def test_config_matches_the_loader_and_the_download_tool():
    cfg = yaml.safe_load((REPO_ROOT / "conf" / "dataset" / "tgsim.yaml").read_text(encoding="utf-8"))
    assert cfg["name"] == cfg["loader"] == "tgsim" and cfg["position_source"] == "track"
    inspect.signature(load_pairs).bind(cfg["raw_dir"], **cfg["options"], dataset=cfg["name"])
    sources = yaml.safe_load((REPO_ROOT / "conf" / "sources.yaml").read_text(encoding="utf-8"))
    tables = {t["name"]: t for t in sources["tgsim"]["tables"]}
    assert set(tables) == set(DATASETS)  # <key>.parquet
    for name in cfg["options"]["datasets"]:
        spec = DATASETS[name]
        needed = {"id", "time", "xloc_kf", "yloc_kf", "lane_kf", "speed_kf", "acceleration_kf", "length_smoothed",
                  "type_most_common", *([spec.flag] if spec.flag else []), *(["run_index"] if spec.runs else [])}
        assert needed <= set(tables[name]["select"]), name
