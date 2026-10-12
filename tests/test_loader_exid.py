"""exiD loader: fabricated recordings with and without the map-based enrichment columns.

Recordings are written at 1.6 Hz to keep the files small (the frame rate is read from the
recording meta; a rate whose frames lie on the 10 Hz grid would not be interpolated by
``resample_tracks``).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from follownet2.data.extraction import ExtractionConfig, extract_all
from follownet2.data.loaders.exid import load_pairs
from follownet2.utils import REPO_ROOT

TRACKS = [
    "recordingId", "trackId", "frame", "trackLifetime", "xCenter", "yCenter", "heading", "width", "length",
    "xVelocity", "yVelocity", "xAcceleration", "yAcceleration", "lonVelocity", "latVelocity", "lonAcceleration",
    "latAcceleration",
]
ENRICHMENT = [
    "traveledDistance", "latLaneCenterOffset", "laneWidth", "laneletId", "laneChange", "lonLaneletPos",
    "laneletLength", "leadDHW", "leadDV", "leadTHW", "leadTTC", "leadId", "rearId", "leftLeadId", "leftRearId",
    "leftAlongsideId", "rightLeadId", "rightRearId", "rightAlongsideId",
]
TRACKS_META = ["recordingId", "trackId", "initialFrame", "finalFrame", "numFrames", "width", "length", "class"]
RECORDING_META = [
    "recordingId", "locationId", "frameRate", "speedLimit", "weekday", "startTime", "duration", "numTracks",
    "numVehicles", "numVrus", "latLocation", "lonLocation", "xUtmOrigin", "yUtmOrigin", "orthoPxToMeter",
]
V0 = 10.0  # m/s
# (trackId, class, length, leader, net gap [m]); the centroid of track 0 starts at 20 m
VEHICLES = [(0, "car", 4.5, -1, None), (1, "car", 4.0, 0, 20.0), (2, "truck", 12.0, 1, 25.0)]


def write(raw_dir: Path, rec=4, location=1, rate=1.6, duration=16.25, border=100.0, vehicles=VEHICLES,
          drop=(), edit=None, recording_meta="recordingMeta") -> Path:
    """Lanelet 100 ends and lanelet 101 starts at ``border`` [m]; lanelet 205 (a diverge)
    overlaps lanelet 100 over its last 30 m. ``edit(tracks)`` modifies the tracks before writing."""
    frames = np.arange(round(duration * rate) + 1)
    t = frames / rate
    centre, rows, meta = {}, [], []
    for vid, cls, length, leader, gap in vehicles:
        start = 20.0 if leader < 0 else centre[leader][0] - (VEHICLES[leader][2] / 2 + gap + length / 2)
        s = centre[vid] = start + V0 * t
        ahead = s >= border
        overlap = (s >= border - 30.0) & ~ahead
        position = np.where(ahead, s - border, s).round(6).astype(str)
        rows.append(pd.DataFrame({
            "recordingId": rec, "trackId": vid, "frame": frames, "trackLifetime": frames, "xCenter": 500.0 + s,
            "yCenter": -20.0, "heading": 0.0, "width": 1.9, "length": length, "xVelocity": V0, "yVelocity": 0,
            "xAcceleration": 0, "yAcceleration": 0, "lonVelocity": V0, "latVelocity": 0, "lonAcceleration": 0,
            "latAcceleration": 0, "traveledDistance": s - s[0], "latLaneCenterOffset": 0, "laneWidth": 3.5,
            "laneletId": np.where(ahead, "101", np.where(overlap, "100;205", "100")), "laneChange": 0,
            "lonLaneletPos": np.where(overlap, [f"{p};{p - border + 30.0:g}" for p in position.astype(float)], position),
            "laneletLength": np.where(ahead, "300", np.where(overlap, f"{border:g};60", f"{border:g}")),
            "leadDHW": -1.0 if leader < 0 else (centre[leader] - VEHICLES[leader][2] / 2) - (s + length / 2),
            "leadDV": 0 if leader >= 0 else -1000, "leadTHW": -1, "leadTTC": -1, "leadId": leader, "rearId": -1,
            "leftLeadId": -1, "leftRearId": -1, "leftAlongsideId": "", "rightLeadId": -1, "rightRearId": -1,
            "rightAlongsideId": "",
        }))
        meta.append({"recordingId": rec, "trackId": vid, "initialFrame": 0, "finalFrame": frames[-1],
                     "numFrames": len(frames), "width": 1.9, "length": length, "class": cls})
    tracks = pd.concat(rows, ignore_index=True)
    if edit is not None:
        edit(tracks)
    tracks[[c for c in TRACKS + ENRICHMENT if c not in drop]].round(6).to_csv(raw_dir / f"{rec:02d}_tracks.csv", index=False)
    pd.DataFrame(meta)[TRACKS_META].to_csv(raw_dir / f"{rec:02d}_tracksMeta.csv", index=False)
    pd.DataFrame([{
        "recordingId": rec, "locationId": location, "frameRate": rate, "speedLimit": 36.11, "weekday": "monday",
        "startTime": 9, "duration": duration, "numTracks": len(vehicles), "numVehicles": len(vehicles), "numVrus": 0,
        "latLocation": 50.8, "lonLocation": 6.1, "xUtmOrigin": 300000.0, "yUtmOrigin": 5600000.0, "orthoPxToMeter": 0.01,
    }])[RECORDING_META].to_csv(raw_dir / f"{rec:02d}_{recording_meta}.csv", index=False)
    return raw_dir


def run(raw_dir: Path, **options):
    events = extract_all(load_pairs(raw_dir, **options), ExtractionConfig())
    for ev in events:
        ev.validate()
    return sorted(events, key=lambda ev: (ev.follower_id, ev.meta["t0"]))


def span(ev) -> tuple[float, float]:
    return ev.meta["t0"], ev.meta["t0"] + ev.duration


def change_lane_and_leader(tracks: pd.DataFrame) -> None:
    """Track 1 changes lane at 16.875 s; track 2 follows track 0 from 34.375 s on."""
    tracks.loc[(tracks["trackId"] == 1) & (tracks["frame"] == 27), "laneChange"] = 1
    switch = (tracks["trackId"] == 2) & (tracks["frame"] >= 55)
    tracks.loc[switch, "leadId"] = 0
    tracks.loc[switch, "leadDHW"] += 4.0 + 20.0  # plus length and gap of track 1


@pytest.fixture(scope="module")
def enriched(tmp_path_factory) -> list:
    raw_dir = tmp_path_factory.mktemp("exid_enriched")
    return run(write(raw_dir, duration=52.5, edit=change_lane_and_leader))


def test_lead_dhw_is_the_gap_across_lanelet_borders(enriched):
    first = enriched[0]  # track 1 behind track 0, both cross the lanelet border at 100 m
    assert (first.follower_id, first.leader_id) == ("exid/1/04.1", "exid/1/04.0")
    assert first.meta["t0"] == 0.0 and first.duration > 16.0
    np.testing.assert_allclose(first.s, 20.0, atol=1e-6)
    np.testing.assert_allclose(np.diff(first.x_follower), V0 * 0.1, atol=1e-9)
    np.testing.assert_allclose(first.v, V0, atol=1e-12)
    for ev in enriched:
        assert ev.meta["gap_source"] == "leadDHW" and ev.meta["native_rate_hz"] == 1.6
    truck = enriched[-1]
    assert truck.meta["vehicle_class"] == "truck" and truck.meta["follower_length"] == 12.0


def test_lane_changes_end_events(enriched):
    # track 1 changes lane at 16.875 s: its own events and those of track 2 behind it end there
    # (the lane of a 10 Hz sample is the one of the nearest frame: 16.25 s for 16.5 s)
    ends = [(ev.follower_id[-1], ev.leader_id[-1], *span(ev)) for ev in enriched]
    assert [e[:2] for e in ends] == [("1", "0"), ("1", "0"), ("2", "1"), ("2", "1"), ("2", "0")]
    assert ends[0][3] == pytest.approx(16.5) and ends[1][2] == pytest.approx(16.6)
    assert ends[2][3] == pytest.approx(16.5) and ends[3][2] == pytest.approx(16.6)


def test_lead_dhw_is_not_mixed_across_a_leader_change(enriched):
    # leadDHW of the frames at 33.75 and 34.375 s around the change of leadId is not used
    behind_1, behind_0 = enriched[3], enriched[4]
    assert span(behind_1)[1] == pytest.approx(33.1) and span(behind_0)[0] == pytest.approx(35.0)
    np.testing.assert_allclose(behind_1.s, 25.0, atol=1e-6)
    np.testing.assert_allclose(behind_0.s, 49.0, atol=1e-6)
    assert behind_0.meta["leader_length"] == 4.5


@pytest.mark.parametrize("drop", [("leadDHW", "traveledDistance"), ("leadDHW", "traveledDistance", "leadId")])
def test_lanelet_positions_end_events_at_lanelet_borders(tmp_path, drop):
    events = run(write(tmp_path, duration=40.0, border=200.0, vehicles=VEHICLES[:2], drop=drop))
    assert [(ev.follower_id, ev.leader_id, ev.meta["lane"]) for ev in events] == [
        ("exid/1/04.1", "exid/1/04.0", 100), ("exid/1/04.1", "exid/1/04.0", 101),
    ]
    # track 0 enters lanelet 101 at 18.0 s, track 1 at 20.425 s; positions of the frames next
    # to a change of lanelet (17.5 and 18.125 s, 20.0 and 20.625 s) are not used
    assert span(events[0]) == (0.0, pytest.approx(16.8))
    assert span(events[1]) == (pytest.approx(21.3), pytest.approx(40.0))
    for ev in events:
        np.testing.assert_allclose(ev.s, 20.0, atol=1e-6)
        assert ev.meta["gap_source"] == "lonLaneletPos"


def test_missing_enrichment_and_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="XX_tracks.csv"):
        run(tmp_path)
    write(tmp_path, duration=1.0, drop=ENRICHMENT)
    with pytest.raises(ValueError, match="enrichment"):
        run(tmp_path)
    (tmp_path / "04_tracksMeta.csv").unlink()
    with pytest.raises(FileNotFoundError, match="04_tracksMeta.csv"):
        run(tmp_path)


def test_recordings_meta_file_name_of_the_pdf(tmp_path):
    write(tmp_path, rec=7, location=3, vehicles=VEHICLES[:2], recording_meta="recordingsMeta")
    write(tmp_path, rec=9, location=3, vehicles=VEHICLES[:2])
    assert [ev.follower_id for ev in run(tmp_path)] == ["exid/3/07.1", "exid/3/09.1"]
    assert [ev.follower_id for ev in run(tmp_path, recordings=[9])] == ["exid/3/09.1"]
    assert [ev.follower_id for ev in run(tmp_path, max_groups=1)] == ["exid/3/07.1"]


def test_config_matches_the_loader():
    cfg = yaml.safe_load((REPO_ROOT / "conf" / "dataset" / "exid.yaml").read_text(encoding="utf-8"))
    assert cfg["name"] == cfg["loader"] == "exid" and cfg["position_source"] == "track"
    inspect.signature(load_pairs).bind(cfg["raw_dir"], **cfg["options"], dataset=cfg["name"])
