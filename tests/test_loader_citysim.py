"""CitySim loader: a fabricated freeway recording in the layout of the CitySim wiki.

The recording is written at 1.6 fps (and read with ``frame_rate=1.6``) to keep the file small
(a rate whose frames lie on the 10 Hz grid would not be interpolated by ``resample_tracks``);
the site is outside the US, so there are no Lat/Lon columns.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from follownet2.data.extraction import ExtractionConfig, extract_all
from follownet2.data.loaders.citysim import FEET, MPH, load_pairs
from follownet2.utils import REPO_ROOT

POINTS = [
    "carCenterX", "carCenterY", "headX", "headY", "tailX", "tailY", "boundingBox1X", "boundingBox1Y",
    "boundingBox2X", "boundingBox2Y", "boundingBox3X", "boundingBox3Y", "boundingBox4X", "boundingBox4Y",
]
COLUMNS = ["frameNum", "carId", *POINTS, *(c + "ft" for c in POINTS), "speed", "heading", "course", "laneId"]
RATE, SPEED = 1.6, 45.0  # fps, mph (66 ft/s, 41.25 ft per frame)
FT_PER_PX = 0.5
UNIT = np.array([-0.6, -0.8])  # direction of travel in the image: towards decreasing x and y
NORMAL = np.array([0.8, -0.6])


def vehicle(car, head, length, duration, lane, switch=None, width=6.0) -> pd.DataFrame:
    """Rows of one car whose head is ``head`` ft down the road at t = 0. Lane k lies 12 ft beside
    lane k-1; ``switch = (time, lane)`` moves the car to another lane."""
    frames = np.arange(round(duration * RATE) + 1)
    t = frames / RATE
    lanes = np.full(len(t), float(lane)) if switch is None else np.where(t < switch[0], float(lane), float(switch[1]))
    front = np.array([2000.0, 1500.0]) + np.outer(head + SPEED * 5280 / 3600 * t, UNIT) + np.outer(12.0 * lanes, NORMAL)
    rear = front - length * UNIT
    side = width / 2 * NORMAL
    feet = np.hstack([(front + rear) / 2, front, rear, front + side, front - side, rear - side, rear + side])
    rows = pd.DataFrame(np.hstack([feet / FT_PER_PX, feet]), columns=[*POINTS, *(c + "ft" for c in POINTS)])
    return rows.assign(frameNum=frames, carId=car, speed=SPEED, heading=233.13, course=233.13, laneId=lanes.astype(int))


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    """Lane 1: the truck 2 follows car 1 at 60 ft for 16 s. Car 5 follows car 4 in lane 2 at
    70 ft, moves to lane 3 at 17 s and follows car 8 there at 90 ft."""
    rows = pd.concat([
        vehicle(1, 200.0, 15.0, 16.25, lane=1),
        vehicle(2, 200.0 - 15.0 - 60.0, 40.0, 16.25, lane=1),
        vehicle(4, 70.0 + 15.0, 15.0, 35.0, lane=2),
        vehicle(5, 0.0, 16.0, 35.0, lane=2, switch=(17.0, 3)),
        vehicle(8, 90.0 + 15.0, 15.0, 35.0, lane=3),
    ])
    root = tmp_path_factory.mktemp("citysim")
    (root / "FreewayC").mkdir()
    rows[COLUMNS].round(6).to_csv(root / "FreewayC" / "FreewayC-01.csv", index=False)
    meta = {"fileName": "FreewayC-01", "frameSizeX": 3840, "frameSizeY": 2160, "recordingFrameRate": RATE,
            "totalFrames": 57, "recordingTime": "16:00", "duration": 35}
    pd.DataFrame([meta]).to_csv(root / "FreewayC" / "metadata.csv", index=False)
    return root


def run(raw_dir: Path, **options):
    options.setdefault("frame_rate", RATE)
    events = extract_all(load_pairs(raw_dir, **options), ExtractionConfig())
    for ev in events:
        ev.validate()
    return sorted(events, key=lambda ev: (ev.follower_id, ev.meta["t0"]))


def test_lane_axis_orientation_and_units(site):
    events = run(site)
    assert [(ev.follower_id, ev.leader_id) for ev in events] == [
        ("citysim/FreewayC/FreewayC-01.2", "citysim/FreewayC/FreewayC-01.1"),
        ("citysim/FreewayC/FreewayC-01.5", "citysim/FreewayC/FreewayC-01.4"),
        ("citysim/FreewayC/FreewayC-01.5", "citysim/FreewayC/FreewayC-01.8"),
    ]
    truck = events[0]
    assert len(truck) == 163 and truck.meta["native_rate_hz"] == RATE and truck.meta["lane"] == 1
    np.testing.assert_allclose(truck.s, 60.0 * FEET, atol=1e-6)
    np.testing.assert_allclose(truck.v, SPEED * MPH, atol=1e-9)
    np.testing.assert_allclose(np.diff(truck.x_follower), SPEED * MPH * 0.1, atol=1e-9)  # oriented with the motion
    assert truck.meta["follower_length"] == pytest.approx(40.0 * FEET)
    assert truck.meta["leader_length"] == pytest.approx(15.0 * FEET)
    assert truck.meta["source_file"] == "FreewayC/FreewayC-01.csv"


def test_lane_change_ends_the_event_and_positions_are_per_lane(site):
    behind_4, behind_8 = run(site)[1:]
    assert behind_4.meta["t0"] == 0.0 and behind_4.meta["t0"] + behind_4.duration < 17.0
    assert behind_8.meta["t0"] > 17.0 and behind_8.meta["lane"] == 3
    np.testing.assert_allclose(behind_4.s, 70.0 * FEET, atol=1e-6)
    np.testing.assert_allclose(behind_8.s, 90.0 * FEET, atol=1e-6)


def test_sites_and_missing_input(site, tmp_path):
    with pytest.raises(FileNotFoundError, match="<site>/<recording>.csv"):
        run(site, sites=["FreewayB"])
    with pytest.raises(FileNotFoundError):
        run(tmp_path)
    (tmp_path / "FreewayD").mkdir()
    pd.DataFrame({"frameNum": [0], "carId": [1], "speed": [50.0]}).to_csv(tmp_path / "FreewayD" / "a.csv", index=False)
    with pytest.raises(ValueError, match="headXft"):
        run(tmp_path)


def test_config_matches_the_loader():
    cfg = yaml.safe_load((REPO_ROOT / "conf" / "dataset" / "citysim.yaml").read_text(encoding="utf-8"))
    assert cfg["name"] == cfg["loader"] == "citysim" and cfg["position_source"] == "track"
    inspect.signature(load_pairs).bind(cfg["raw_dir"], **cfg["options"], dataset=cfg["name"])
