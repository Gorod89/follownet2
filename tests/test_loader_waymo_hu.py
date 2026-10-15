"""Waymo pairs of Hu et al. (2022): fabricated trajectory and pair-record files in the published format."""

from __future__ import annotations

from itertools import islice

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid

from trafficloop.data.extraction import ExtractionConfig, extract_all
from trafficloop.data.loaders import waymo_hu
from trafficloop.data.loaders.waymo_hu import last_report, load_pairs, pair_blocks
from trafficloop.utils import resolve_path

REAL_DIR = resolve_path(waymo_hu.DEFAULT_RAW_DIR)
T = np.round(np.arange(200) * 0.1, 2)  # one 20 s segment at 10 Hz
AV_LENGTH = 5.18


def extract(pairs, cfg: ExtractionConfig | None = None) -> list:
    """``extract_all`` plus ``Event.validate()`` on every event."""
    events = extract_all(pairs, cfg or ExtractionConfig())
    for ev in events:
        ev.validate()
    return events


def pair_block(segment, follower, leader, lengths, v_f, v_l, d0, t=T) -> pd.DataFrame:
    """Leader rows, then follower rows; positions of the vehicle centres as in the source."""
    rows = []
    for veh, length, v, offset in ((leader, lengths[1], v_l, d0), (follower, lengths[0], v_f, 0.0)):
        rows.append(pd.DataFrame({
            "segment_id": float(segment), "local_veh_id": float(veh), "length": length, "local_time": t,
            "follower_id": follower, "leader_id": leader, "filter_pos": offset + cumulative_trapezoid(v, t, initial=0.0),
            "filter_speed": v, "filter_accer": np.gradient(v, t),
        }))
    return pd.concat(rows, ignore_index=True)


def write_source(root, blocks, av_record=("1,7,5,0,0", "2,0,0,1,1"), hv_record=("2,3,2",)) -> None:
    pd.concat(blocks, ignore_index=True).to_csv(root / waymo_hu.TRAJECTORY_FILE, index=False)
    (root / waymo_hu.AV_RECORD_FILE).write_text(
        "segment_id,follower,leader,follower_code,leader_code\n" + "\n".join(av_record) + "\n", encoding="utf-8")
    (root / waymo_hu.HV_RECORD_FILE).write_text(
        "segment_id,follower,leader\n" + "\n".join(hv_record) + "\n", encoding="utf-8")


def standard_blocks() -> list[pd.DataFrame]:
    """Segment 1: the AV follows human 5, human 7 follows the AV; segment 2: human 3 follows human 2."""
    wave = 12.0 + np.sin(0.3 * T)
    return [
        pair_block(1, 0, 5, (AV_LENGTH, 4.3), np.full(200, 11.8), wave, 35.0),
        pair_block(1, 7, 0, (4.6, AV_LENGTH), np.full(200, 12.5), wave, 40.0),
        pair_block(2, 3, 2, (4.5, 4.0), np.full(200, 8.0), np.full(200, 7.5), 25.0),
    ]


@pytest.mark.parametrize(
    "followers, expected",
    [
        ("human", [("1.7", "waymo_driver", False, True), ("2.3", "2.2", False, False)]),
        ("automated", [("waymo_driver", "1.5", True, False)]),
        ("all", [("waymo_driver", "1.5", True, False), ("1.7", "waymo_driver", False, True), ("2.3", "2.2", False, False)]),
    ],
)
def test_follower_selection_and_ids(tmp_path, followers, expected):
    write_source(tmp_path, standard_blocks())
    pairs = list(load_pairs(tmp_path, followers=followers))
    found = [
        (p.follower_id.removeprefix("waymo_hu/all/"), p.leader_id[0].removeprefix("waymo_hu/all/"),
         p.meta["automated"], p.meta["leader_automated"])
        for p in pairs
    ]
    assert found == expected
    assert all(p.site == "all" and p.meta["ids"] == "source" for p in pairs)


def test_net_gap_from_centre_positions_and_the_sign(tmp_path):
    write_source(tmp_path, standard_blocks())
    pairs = list(load_pairs(tmp_path, followers="all"))
    wave = 12.0 + np.sin(0.3 * T)
    centre = 40.0 + cumulative_trapezoid(wave - 12.5, T, initial=0.0)
    human_behind_av = pairs[1]
    gap = human_behind_av.x_lead - human_behind_av.x_follower
    np.testing.assert_allclose(gap, centre - 0.5 * (4.6 + AV_LENGTH), atol=1e-9)
    np.testing.assert_allclose(human_behind_av.t, T, atol=1e-12)
    assert human_behind_av.meta["follower_length"] == 4.6 and human_behind_av.meta["segment"] == 1
    np.testing.assert_allclose(human_behind_av.a, 0.0, atol=1e-9)  # the dataset acceleration of the follower
    events = extract(pairs)
    assert [len(ev) for ev in events] == [200, 200, 200]
    assert all(ev.meta["a_source"] == "dataset" for ev in events)
    assert events[1].meta["leader_length"] == AV_LENGTH and np.mean(events[1].dv) > 0  # 12.5 m/s behind 12 m/s
    np.testing.assert_allclose(events[2].dv, 0.5)


def test_negative_speeds_are_left_for_the_extraction(tmp_path):
    v = np.full(200, 8.0)
    v[170:174] = -0.05
    write_source(tmp_path, [pair_block(2, 3, 2, (4.5, 4.0), v, np.full(200, 7.5), 25.0)])
    pair = next(iter(load_pairs(tmp_path)))
    assert np.all(pair.v[170:174] == -0.05)
    events = extract([pair])  # noise of a standing vehicle: set to zero by the extraction
    assert [len(ev) for ev in events] == [200] and events[0].v.min() == 0.0
    assert events[0].meta["negative_speeds_set_to_zero"] == 4
    pair.v[170:174] = -0.8
    assert [len(ev) for ev in extract([pair])] == [170]


def test_time_jitter_is_interpolated_and_vehicles_are_aligned(tmp_path):
    t = T.copy()
    t[50] = 5.01  # frames arrive 0.11 s and 0.09 s apart
    t[120] = 11.99
    v_l = 7.0 + 0.05 * t
    block = pair_block(2, 3, 2, (4.5, 4.0), np.full(200, 8.0), v_l, 25.0, t=t)
    leader_rows = block.index[block["local_veh_id"] == 2.0]
    block = block.drop(leader_rows[-1])  # the leader lacks the last frame
    block = block[np.abs(block["local_time"] - 15.0) > 1e-9]  # both lack the frame at 15 s: a 0.2 s hole
    write_source(tmp_path, [block])
    pair = next(iter(load_pairs(tmp_path)))
    np.testing.assert_allclose(pair.t, np.arange(199) * 0.1, atol=1e-9)
    assert np.isnan(pair.v[150]) and np.all(np.isfinite(np.delete(pair.v, 150)))
    np.testing.assert_allclose(np.delete(pair.v_lead, 150), np.delete(7.0 + 0.05 * pair.t, 150), atol=1e-9)
    assert [len(ev) for ev in extract([pair])] == [150]  # 150 and 48 samples around the hole


def test_pairs_missing_from_the_records_are_skipped(tmp_path):
    blocks = [*standard_blocks(), pair_block(2, 9, 3, (4.1, 4.5), np.full(200, 8.0), np.full(200, 8.0), 20.0)]
    write_source(tmp_path, blocks)
    assert len(list(load_pairs(tmp_path, followers="all"))) == 3
    report = last_report()
    assert report["pairs_read"] == 4 and report["pairs_not_in_records"] == 1 and report["segments"] == 2


def test_pair_blocks_survive_chunk_boundaries(tmp_path):
    write_source(tmp_path, standard_blocks())
    path = tmp_path / waymo_hu.TRAJECTORY_FILE
    small = [(tuple(b[waymo_hu.KEY].iloc[0]), len(b)) for b in pair_blocks(path, chunk_rows=7)]
    assert small == [((1.0, 0, 5), 400), ((1.0, 7, 0), 400), ((2.0, 3, 2), 400)]
    assert small == [(tuple(b[waymo_hu.KEY].iloc[0]), len(b)) for b in pair_blocks(path)]


def test_split_pair_block_is_an_error(tmp_path):
    first, second, third = standard_blocks()
    write_source(tmp_path, [first, second, third, first])
    with pytest.raises(ValueError, match="not one contiguous block"):
        list(load_pairs(tmp_path, followers="all"))


def test_max_groups_counts_segments_and_missing_input(tmp_path):
    write_source(tmp_path, standard_blocks())
    assert [p.meta["segment"] for p in load_pairs(tmp_path, followers="all", max_groups=1)] == [1, 1]
    with pytest.raises(ValueError, match="followers must be"):
        list(load_pairs(tmp_path, followers="av"))
    (tmp_path / waymo_hu.HV_RECORD_FILE).unlink()
    with pytest.raises(FileNotFoundError, match=waymo_hu.HV_RECORD_FILE):
        list(load_pairs(tmp_path))


@pytest.mark.data
@pytest.mark.skipif(not (REAL_DIR / waymo_hu.TRAJECTORY_FILE).is_file(), reason="Waymo pair files not on disk")
def test_real_waymo_pairs(monkeypatch):
    monkeypatch.setattr(waymo_hu, "CHUNK_ROWS", 4000)  # stop after the first pairs of the file
    pairs = list(islice(load_pairs(REAL_DIR, followers="all"), 12))
    assert last_report()["pairs_not_in_records"] == 0
    assert pairs[0].follower_id == "waymo_hu/all/waymo_driver" and pairs[0].leader_id[0] == "waymo_hu/all/3.5"
    assert all(p.meta["automated"] or p.meta["leader_automated"] for p in pairs)  # the file starts with AV pairs
    assert extract(pairs)
