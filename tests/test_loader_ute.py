"""UTE loader: fabricated frenet.csv files in the verified per-site layouts.

Files are written at 4 fps to keep them small (the loader takes the time from ``time(s)``; a
rate whose frames lie on the 10 Hz grid would not be interpolated by ``resample_tracks``); the
documented 24 fps is used where the resampling is under test.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from follownet2.data.extraction import ExtractionConfig, extract_all
from follownet2.data.loaders.ute import SITES, load_pairs
from follownet2.utils import REPO_ROOT

V0, ACC = 12.0, 0.32  # m/s, m/s^2 of every vehicle: gaps stay constant
PIXEL = 0.05  # m per pixel of the fabricated SQM-W-1 video


def platoon(ids, lengths, gaps, lane, sign=1.0, fps=4.0, start=500.0, duration=16.0, acc=ACC):
    """Rows of one lane under the UTE column names; ids[0] leads. ``sign`` < 0: traffic runs
    towards decreasing longitudinalDistance."""
    frame = np.arange(round(duration * fps) + 1)
    t = frame / fps
    travelled = V0 * t + 0.5 * acc * t**2
    offset, rows = 0.0, []
    for i, vid in enumerate(ids):
        if i > 0:
            offset += lengths[i - 1] / 2 + gaps[i - 1] + lengths[i] / 2
        rows.append(pd.DataFrame({
            "vehicleID": vid, "frame": frame, "time(s)": t, "laneID": lane,
            "longitudinalDistance(m)": start + sign * (travelled - offset), "speed(km/h)": (V0 + acc * t) * 3.6,
            "acceleration(m/s^2)": acc, "lateralDistance(m)": 3.5 * lane, "vehicleLength(m)": lengths[i],
            "vehicleWidth(m)": 1.8, "vehicleLength(pixel)": lengths[i] / PIXEL, "vehicleWidth(pixel)": 1.8 / PIXEL,
            "leader_id": ids[i - 1] if i > 0 else -1, "follower_id": ids[i + 1] if i + 1 < len(ids) else -1,
            "leader_distance(m)": gaps[i - 1] if i > 0 else -1, "follower_distance(m)": gaps[i] if i + 1 < len(ids) else -1,
        }))
    return pd.concat(rows, ignore_index=True).round(6)


def write(raw_dir: Path, site: str, rows: pd.DataFrame, header=None) -> Path:
    (raw_dir / site).mkdir(exist_ok=True)
    rows[list(header or SITES[site].header)].to_csv(raw_dir / site / "frenet.csv", index=False)
    return raw_dir


def run(raw_dir: Path, **options):
    events = extract_all(load_pairs(raw_dir, **options), ExtractionConfig())
    for ev in events:
        ev.validate()
    return {ev.follower_id: ev for ev in events}


def test_sqm_w_2_with_leader_ids(tmp_path):
    write(tmp_path, "SQM-W-2", platoon([0, 7, 3], [4.6, 4.2, 11.0], [18.0, 26.0], lane=2))
    events = run(tmp_path, sites=["SQM-W-2"])
    assert {f: ev.leader_id for f, ev in events.items()} == {"ute/SQM-W-2/7": "ute/SQM-W-2/0", "ute/SQM-W-2/3": "ute/SQM-W-2/7"}
    for follower, gap, length in [("7", 18.0, 4.2), ("3", 26.0, 11.0)]:
        ev = events[f"ute/SQM-W-2/{follower}"]
        assert len(ev) == 161
        np.testing.assert_allclose(ev.s, gap, atol=1e-6)
        np.testing.assert_allclose(ev.v, V0 + ACC * ev.t, atol=1e-9)
        np.testing.assert_allclose(ev.a, ACC, atol=1e-12)
        assert ev.meta["follower_length"] == length and ev.meta["native_rate_hz"] == 24
        assert ev.meta["source_file"] == "SQM-W-2/frenet.csv" and "gap_definition" not in ev.meta


def test_sqm_w_1_lengths_in_pixels_give_centre_gaps(tmp_path):
    write(tmp_path, "SQM-W-1", platoon([1, 2], [4.6, 4.2], [18.0], lane=4))
    ev = run(tmp_path, sites=["SQM-W-1"])["ute/SQM-W-1/2"]
    np.testing.assert_allclose(ev.s, 18.0 + (4.6 + 4.2) / 2, atol=1e-6)
    assert "centre to centre" in ev.meta["gap_definition"] and "follower_length" not in ev.meta


@pytest.mark.parametrize("site", ["YTDJ-3", "XAM-N-6", "RML-7"])
def test_derived_leaders_in_both_directions(tmp_path, site):
    rows = pd.concat([
        platoon([5, 6], [4.4, 4.8], [15.0], lane=1, sign=-1.0),
        platoon([8, 9], [4.5, 4.5], [20.0], lane=3, start=450.0),
    ])
    events = run(write(tmp_path, site, rows), sites=[site])
    assert {f: ev.leader_id for f, ev in events.items()} == {f"ute/{site}/6": f"ute/{site}/5", f"ute/{site}/9": f"ute/{site}/8"}
    for follower, gap in [("6", 15.0), ("9", 20.0)]:
        ev = events[f"ute/{site}/{follower}"]
        assert len(ev) == 161 and ev.meta["native_rate_hz"] == SITES[site].fps
        np.testing.assert_allclose(ev.s, gap, atol=1e-6)
        assert np.all(np.diff(ev.x_follower) > 0)


def test_resampling_from_24_fps(tmp_path):
    # constant speed: 0.5 m per frame, so the positions are exact in the file despite 1/24 s steps
    write(tmp_path, "YTDJ-3", platoon([1, 2], [4.5, 4.9], [21.0], lane=2, fps=24.0, duration=15.2, acc=0.0))
    ev = run(tmp_path, sites=["YTDJ-3"])["ute/YTDJ-3/2"]
    assert len(ev) == 153 and ev.meta["native_rate_hz"] == 24
    np.testing.assert_allclose(np.diff(ev.t), 0.1)
    np.testing.assert_allclose(ev.s, 21.0, atol=1e-6)
    np.testing.assert_allclose(ev.v, V0, atol=1e-9)


def test_unverified_site_raises(tmp_path):
    with pytest.raises(NotImplementedError, match="SQM-N-4"):
        run(tmp_path, sites=["SQM-N-4"])


def test_header_and_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="frenet.csv"):
        run(tmp_path, sites=["PKDD-8"])
    write(tmp_path, "PKDD-8", platoon([1], [4.5], [], lane=1, duration=1.0), header=SITES["YTDJ-3"].header)
    with pytest.raises(ValueError, match="PKDD-8"):
        run(tmp_path, sites=["PKDD-8"])


def test_max_groups(tmp_path):
    write(tmp_path, "XAM-N-5", platoon([1, 2], [4.5, 4.5], [20.0], lane=1))
    write(tmp_path, "XAM-S-9", platoon([1, 2], [4.5, 4.5], [25.0], lane=1))
    assert sorted(run(tmp_path, sites=["XAM-N-5", "XAM-S-9"])) == ["ute/XAM-N-5/2", "ute/XAM-S-9/2"]
    assert sorted(run(tmp_path, sites=["XAM-N-5", "XAM-S-9"], max_groups=1)) == ["ute/XAM-N-5/2"]


def test_config_matches_the_loader():
    cfg = yaml.safe_load((REPO_ROOT / "conf" / "dataset" / "ute.yaml").read_text(encoding="utf-8"))
    assert cfg["name"] == cfg["loader"] == "ute" and cfg["position_source"] == "track"
    inspect.signature(load_pairs).bind(cfg["raw_dir"], **cfg["options"], dataset=cfg["name"])
    assert all(SITES[site].parser is not None for site in cfg["options"]["sites"])
