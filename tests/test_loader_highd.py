"""highD loader: fabricated recordings in the documented three-file format.

Recordings are written at 4 Hz to keep the files small (the frame rate is read from the
recording meta; a rate whose frames lie on the 10 Hz grid would not be interpolated by
``resample_tracks``); the 25 Hz of the real data is used where the resampling is under test.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trafficloop.data.extraction import ExtractionConfig, extract_all
from trafficloop.data.loaders.highd import load_pairs
from trafficloop.utils import REPO_ROOT

TRACKS = [
    "frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration",
    "frontSightDistance", "backSightDistance", "dhw", "thw", "ttc", "precedingXVelocity", "precedingId",
    "followingId", "leftPrecedingId", "leftAlongsideId", "leftFollowingId", "rightPrecedingId",
    "rightAlongsideId", "rightFollowingId", "laneId",
]
TRACKS_META = [
    "id", "width", "height", "initialFrame", "finalFrame", "numFrames", "class", "drivingDirection",
    "traveledDistance", "minXVelocity", "maxXVelocity", "meanXVelocity", "minDHW", "minTHW", "minTTC",
    "numLaneChanges",
]
RECORDING_META = [
    "id", "frameRate", "locationId", "speedLimit", "month", "weekDay", "startTime", "duration",
    "totalDrivenDistance", "totalDrivenTime", "numVehicles", "numCars", "numTrucks", "upperLaneMarkings",
    "lowerLaneMarkings",
]
V0, ACC = 25.0, 0.5  # m/s, m/s^2 of every vehicle: the gaps stay constant
# (id, class, length, driving direction, lane, leader, net gap to the leader [m])
BOTH = [
    (1, "Car", 4.5, 2, 5, 0, None),
    (2, "Truck", 12.0, 2, 5, 1, 20.0),
    (3, "Truck", 16.0, 1, 2, 0, None),
    (4, "Car", 4.8, 1, 2, 3, 25.0),
]


def write_recording(raw_dir: Path, vehicles=BOTH, rec=1, rate=4, location=2, duration=16.0, acc=ACC) -> Path:
    frames = np.arange(round(duration * rate) + 1) + 1
    t = (frames - 1) / rate
    travelled = V0 * t + 0.5 * acc * t**2
    lengths = {v[0]: v[2] for v in vehicles}
    rows, meta, behind = [], [], {}
    for vid, cls, length, direction, lane, leader, gap in vehicles:
        # distance of the front bumper behind the head of the platoon
        behind[vid] = 0.0 if leader == 0 else behind[leader] + lengths[leader] + gap
        right = direction == 2  # lower lanes, moving right: the front is the right edge of the box
        x = 300.0 - behind[vid] + travelled - length if right else 100.0 + behind[vid] - travelled
        speed = (V0 + acc * t) * (1 if right else -1)
        rows.append(pd.DataFrame({
            "frame": frames, "id": vid, "x": x, "y": 10.0 + 3.5 * lane, "width": length, "height": 2.0,
            "xVelocity": speed, "yVelocity": 0, "xAcceleration": acc if right else -acc, "yAcceleration": 0,
            "frontSightDistance": 100, "backSightDistance": 100, "dhw": 0.0 if leader == 0 else gap + length,
            "thw": 0, "ttc": 0, "precedingXVelocity": 0 if leader == 0 else speed, "precedingId": leader,
            "followingId": 0, "leftPrecedingId": 0, "leftAlongsideId": 0, "leftFollowingId": 0,
            "rightPrecedingId": 0, "rightAlongsideId": 0, "rightFollowingId": 0, "laneId": lane,
        }))
        meta.append({
            "id": vid, "width": length, "height": 2.0, "initialFrame": 1, "finalFrame": frames[-1],
            "numFrames": len(frames), "class": cls, "drivingDirection": direction, "traveledDistance": travelled[-1],
            "minXVelocity": V0, "maxXVelocity": V0 + acc * duration, "meanXVelocity": V0 + acc * duration / 2,
            "minDHW": -1 if leader == 0 else gap + length, "minTHW": -1, "minTTC": -1, "numLaneChanges": 0,
        })
    prefix = raw_dir / f"{rec:02d}"
    pd.concat(rows)[TRACKS].round(6).to_csv(f"{prefix}_tracks.csv", index=False)
    pd.DataFrame(meta)[TRACKS_META].to_csv(f"{prefix}_tracksMeta.csv", index=False)
    pd.DataFrame([{
        "id": rec, "frameRate": rate, "locationId": location, "speedLimit": -1.0, "month": 9, "weekDay": "Tue",
        "startTime": "08:38", "duration": duration, "totalDrivenDistance": 0.0, "totalDrivenTime": 0.0,
        "numVehicles": len(vehicles), "numCars": 2, "numTrucks": 2, "upperLaneMarkings": "8.51;12.59;16.43",
        "lowerLaneMarkings": "21.00;24.96;28.80",
    }])[RECORDING_META].to_csv(f"{prefix}_recordingMeta.csv", index=False)
    return raw_dir


def run(raw_dir: Path, **options):
    events = extract_all(load_pairs(raw_dir, **options), ExtractionConfig())
    for ev in events:
        ev.validate()
    return {ev.follower_id: ev for ev in events}


def test_both_driving_directions(tmp_path):
    events = run(write_recording(tmp_path))
    assert {f: ev.leader_id for f, ev in events.items()} == {"highd/2/01.2": "highd/2/01.1", "highd/2/01.4": "highd/2/01.3"}
    for follower, gap, length, leader_length in [(2, 20.0, 12.0, 4.5), (4, 25.0, 4.8, 16.0)]:
        ev = events[f"highd/2/01.{follower}"]
        assert len(ev) == 161 and ev.meta["t0"] == 0.0 and ev.meta["group"] == "01"
        np.testing.assert_allclose(ev.s, gap, atol=1e-6)
        np.testing.assert_allclose(ev.v, V0 + ACC * ev.t, atol=1e-9)
        np.testing.assert_allclose(ev.a, ACC, atol=1e-12)
        assert np.all(np.diff(ev.x_follower) > 0) and np.all(np.diff(ev.x_lead) > 0)
        assert ev.meta["follower_length"] == length and ev.meta["leader_length"] == leader_length
    upper = events["highd/2/01.4"]  # drivingDirection 1: the axis is flipped
    np.testing.assert_allclose(upper.x_follower[0], -(100.0 + 16.0 + 25.0), atol=1e-9)
    assert upper.meta["lane"] == 2 and upper.meta["vehicle_class"] == "Car" and upper.meta["leader_vehicle_class"] == "Truck"
    assert events["highd/2/01.2"].meta["vehicle_class"] == "Truck"
    assert upper.meta["native_rate_hz"] == 4.0 and upper.meta["source_file"] == "01_tracks.csv"


def test_resampling_from_25_hz(tmp_path):
    events = run(write_recording(tmp_path, BOTH[2:], rate=25, duration=15.2, acc=0.0))
    ev = events["highd/2/01.4"]
    assert len(ev) == 153 and ev.meta["native_rate_hz"] == 25.0
    np.testing.assert_allclose(np.diff(ev.t), 0.1)
    np.testing.assert_allclose(ev.s, 25.0, atol=1e-6)
    np.testing.assert_allclose(ev.v, V0, atol=1e-9)
    np.testing.assert_allclose(ev.x_follower, -141.0 + V0 * ev.t, atol=1e-9)


def test_recordings_and_max_groups(tmp_path):
    write_recording(tmp_path, BOTH[:2], rec=1, location=2, rate=1.6)
    write_recording(tmp_path, BOTH[:2], rec=12, location=4, rate=1.6)
    assert sorted(run(tmp_path)) == ["highd/2/01.2", "highd/4/12.2"]
    assert sorted(run(tmp_path, recordings=[12])) == ["highd/4/12.2"]
    assert sorted(run(tmp_path, max_groups=1)) == ["highd/2/01.2"]


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="XX_tracks.csv"):
        run(tmp_path)
    write_recording(tmp_path, BOTH[:1], duration=1.0)
    (tmp_path / "01_recordingMeta.csv").unlink()
    with pytest.raises(FileNotFoundError, match="01_recordingMeta.csv"):
        run(tmp_path)


def test_config_matches_the_loader():
    cfg = yaml.safe_load((REPO_ROOT / "conf" / "dataset" / "highd.yaml").read_text(encoding="utf-8"))
    assert cfg["name"] == cfg["loader"] == "highd" and cfg["position_source"] == "track"
    inspect.signature(load_pairs).bind(cfg["raw_dir"], **cfg["options"], dataset=cfg["name"])
