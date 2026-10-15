"""Extraction invariants (FollowNet-compatible criteria)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trafficloop.data.extraction import (
    ExtractionConfig,
    PairSeries,
    extract_all,
    extract_events,
    positions_from_gap,
    resample_tracks,
    tracks_to_pairs,
)
from trafficloop.data.schema import DT


def make_pair(n=900, seed=0, **changes) -> PairSeries:
    rng = np.random.default_rng(seed)
    t = 100.0 + np.arange(n) * DT
    v_lead = 15.0 + 2.0 * np.sin(0.05 * t) + 0.05 * rng.standard_normal(n)
    v = 15.0 + 2.0 * np.sin(0.05 * (t - 1.5)) + 0.05 * rng.standard_normal(n)
    x_lead = 50.0 + np.cumsum(v_lead) * DT
    x = 20.0 + np.cumsum(v) * DT
    pair = PairSeries(
        dataset="toy",
        site="a",
        follower_id="toy/a/7",
        t=t,
        x_follower=x,
        v=v,
        leader_id=np.array(["toy/a/3"] * n, dtype=object),
        x_lead=x_lead,
        v_lead=v_lead,
        lane_follower=np.full(n, 2.0),
        lane_leader=np.full(n, 2.0),
        extras={"leader_length": np.full(n, 4.2)},
        meta={"position_source": "track"},
    )
    for name, value in changes.items():
        setattr(pair, name, value)
    return pair


def check_invariants(events, cfg: ExtractionConfig):
    assert len({ev.event_id for ev in events}) == len(events)
    for ev in events:
        ev.validate()
        assert len(ev) >= cfg.min_samples
        assert ev.t[0] == 0.0 and np.allclose(np.diff(ev.t), DT)
        assert np.all(ev.s >= cfg.min_spacing) and np.all(ev.s < cfg.max_spacing)
        assert np.mean(ev.v > cfg.moving_speed_eps) >= cfg.min_moving_fraction
        assert ev.meta["data_version"] == "raw"
        assert ev.event_id.startswith(ev.follower_id + "|")
        if cfg.max_duration is not None:
            assert ev.duration <= cfg.max_duration + 1e-9


def test_uninterrupted_pair_gives_one_event():
    cfg = ExtractionConfig()
    events = extract_events(make_pair(), cfg)
    assert len(events) == 1 and len(events[0]) == 900
    check_invariants(events, cfg)
    assert events[0].meta["t0"] == pytest.approx(100.0)
    assert events[0].meta["leader_length"] == pytest.approx(4.2)
    assert events[0].meta["a_source"] == "savgol"


def test_leader_change_splits_and_short_pieces_are_dropped():
    pair = make_pair()
    pair.leader_id[400:520] = "toy/a/4"  # 12 s behind another leader: too short
    events = extract_events(pair, ExtractionConfig())
    assert [len(ev) for ev in events] == [400, 380]
    assert all(ev.leader_id == "toy/a/3" for ev in events)
    assert [ev.event_id for ev in events] == ["toy/a/7|3|0", "toy/a/7|3|1"]


@pytest.mark.parametrize("lane", ["lane_follower", "lane_leader"])
def test_lane_change_is_a_boundary(lane):
    pair = make_pair()
    getattr(pair, lane)[300:] = 3.0
    assert [len(ev) for ev in extract_events(pair, ExtractionConfig())] == [300, 600]


def test_invalid_samples_cut_the_series():
    pair = make_pair()
    pair.x_lead[200] = np.nan  # missing leader sample
    pair.x_lead[500:503] += 500.0  # gap beyond 120 m
    pair.leader_id[700] = None
    events = extract_events(pair, ExtractionConfig())
    assert [len(ev) for ev in events] == [200, 299, 197, 199]
    check_invariants(events, ExtractionConfig())


def test_touching_vehicles_are_not_following():
    pair = make_pair()
    pair.x_lead[400:405] = pair.x_follower[400:405] + 0.3  # net gap of 30 cm
    assert [len(ev) for ev in extract_events(pair, ExtractionConfig())] == [400, 495]
    assert [len(ev) for ev in extract_events(pair, ExtractionConfig(min_spacing=0.1))] == [900]


def test_time_gap_is_a_boundary():
    pair = make_pair()
    pair.t[450:] += 0.3
    assert [len(ev) for ev in extract_events(pair, ExtractionConfig())] == [450, 450]


def test_segment_key_is_a_boundary():
    pair = make_pair()
    pair.segment_key = np.array(["acc"] * 500 + ["human"] * 400, dtype=object)
    assert [len(ev) for ev in extract_events(pair, ExtractionConfig())] == [500, 400]


def test_fifteen_seconds_are_150_samples():
    """FollowNet counts samples times step: its 15 s windows have 150 samples at 10 Hz."""
    cfg = ExtractionConfig()
    assert cfg.min_samples == 150
    assert len(extract_events(make_pair(n=149), cfg)) == 0
    events = extract_events(make_pair(n=150), cfg)
    assert len(events) == 1 and events[0].duration == pytest.approx(14.9)


def test_small_negative_speeds_are_noise_and_large_ones_end_the_segment():
    pair = make_pair()
    pair.v[100:110] = -0.2
    pair.v_lead[300] = -0.4
    events = extract_events(pair, ExtractionConfig())
    assert [len(ev) for ev in events] == [900]
    assert events[0].v.min() == 0.0 and events[0].v_lead.min() == 0.0
    assert events[0].meta["negative_speeds_set_to_zero"] == 11
    events[0].validate()
    pair.v[500] = -0.6
    assert [len(ev) for ev in extract_events(pair, ExtractionConfig())] == [500, 399]
    strict = ExtractionConfig(negative_speed_tol=0.0)
    # the first piece, 100 samples, is too short
    assert [len(ev) for ev in extract_events(pair, strict)] == [190, 199, 399]


def test_standing_follower_is_dropped():
    pair = make_pair(n=400)
    pair.v[:150] = 0.0  # standing for 37 % of the samples
    assert extract_events(pair, ExtractionConfig()) == []


def test_long_runs_are_chunked():
    cfg = ExtractionConfig(max_duration=30.0)
    events = extract_events(make_pair(n=900), cfg)
    assert [len(ev) for ev in events] == [300, 300, 300]
    check_invariants(events, cfg)
    with pytest.raises(ValueError):
        ExtractionConfig(max_duration=20.0)


def test_dataset_acceleration_is_kept_when_present():
    pair = make_pair()
    pair.a = np.gradient(pair.v, DT)
    event = extract_events(pair, ExtractionConfig())[0]
    assert event.meta["a_source"] == "dataset"
    np.testing.assert_allclose(event.a, pair.a)
    assert extract_events(pair, ExtractionConfig(accel_source="savgol"))[0].meta["a_source"] == "savgol"


def test_positions_from_gap():
    v_lead = np.full(50, 10.0)
    s = np.linspace(20.0, 25.0, 50)
    x_lead, x = positions_from_gap(v_lead, s)
    np.testing.assert_allclose(x_lead, 10.0 * DT * np.arange(50))
    np.testing.assert_allclose(x_lead - x, s)


def toy_tracks(rate_hz=10.0, n_vehicles=4, duration=40.0, with_leader=True) -> pd.DataFrame:
    """A platoon on one lane; vehicle 1 is in front. Vehicle 3 changes lane at t = 20 s."""
    t = np.arange(0.0, duration + 1e-9, 1.0 / rate_hz)
    rows = []
    for i in range(1, n_vehicles + 1):
        v = 12.0 + np.sin(0.2 * (t - 1.2 * i))
        x_front = 200.0 - 30.0 * i + 12.0 * t - 5.0 * np.cos(0.2 * (t - 1.2 * i))
        lane = np.where((i == 3) & (t >= 20.0), 2, 1)
        leader = np.full(len(t), i - 1)
        if i == 3:
            leader = np.where(t >= 20.0, 0, leader)
        if i == 4:
            leader = np.where(t >= 20.0, 2, leader)
        frame = pd.DataFrame(
            {"group": "rec1", "site": "s1", "vehicle_id": i, "t": t, "x_front": x_front, "x_rear": x_front - 4.5,
             "v": v, "lane": lane, "length": 4.5, "automated": i == 2}
        )
        if with_leader:
            frame["leader"] = leader
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("with_leader", [True, False])
def test_tracks_to_pairs(with_leader):
    cfg = ExtractionConfig()
    pairs = list(tracks_to_pairs(toy_tracks(with_leader=with_leader), "toy", extras=["automated"], leader_extras=["automated"]))
    assert [p.follower_id for p in pairs] == ["toy/s1/rec1.1", "toy/s1/rec1.2", "toy/s1/rec1.3", "toy/s1/rec1.4"]
    events = extract_all(pairs, cfg)
    check_invariants(events, cfg)
    found = sorted((ev.follower_id.rsplit(".", 1)[1], ev.leader_id.rsplit(".", 1)[1], len(ev)) for ev in events)
    # 2 follows 1 for 40 s; 3 follows 2 for 20 s, then leaves; 4 follows 3, then 2
    assert found == [("2", "1", 401), ("3", "2", 200), ("4", "2", 201), ("4", "3", 200)]
    by_pair = {(f, l): ev for ev in events for f, l in [(ev.follower_id[-1], ev.leader_id[-1])]}
    assert by_pair[("3", "2")].meta["leader_automated"] is True
    assert by_pair[("2", "1")].meta["automated"] is True and by_pair[("2", "1")].meta["leader_automated"] is False
    assert by_pair[("4", "3")].meta["leader_length"] == pytest.approx(4.5)
    gap = by_pair[("2", "1")].s
    assert np.all(np.abs(gap - (30.0 - 4.5)) < 12.0)


def test_resampling_from_25_hz_matches_native_10_hz():
    native = extract_all(tracks_to_pairs(toy_tracks(10.0), "toy"), ExtractionConfig())
    resampled = extract_all(tracks_to_pairs(toy_tracks(25.0), "toy"), ExtractionConfig())
    assert [ev.event_id for ev in resampled] == [ev.event_id for ev in native]
    for a, b in zip(native, resampled):
        assert abs(len(a) - len(b)) <= 1
        n = min(len(a), len(b))
        assert np.max(np.abs(a.s[:n] - b.s[:n])) < 0.02
        assert np.max(np.abs(a.v[:n] - b.v[:n])) < 0.02


def test_a_source_slower_than_the_grid_is_interpolated():
    """5 Hz samples lie on the 10 Hz grid; they must be interpolated, not passed with holes."""
    native = extract_all(tracks_to_pairs(toy_tracks(10.0), "toy"), ExtractionConfig())
    slow = extract_all(tracks_to_pairs(toy_tracks(5.0), "toy"), ExtractionConfig())
    assert [ev.event_id for ev in slow] == [ev.event_id for ev in native]
    for a, b in zip(native, slow):
        n = min(len(a), len(b))
        assert abs(len(a) - len(b)) <= 2 and np.max(np.abs(a.s[:n] - b.s[:n])) < 0.05


def test_resampling_does_not_bridge_holes():
    tracks = toy_tracks(25.0, n_vehicles=1)
    tracks = tracks[(tracks["t"] < 10.0) | (tracks["t"] > 12.0)]
    grid = resample_tracks(tracks)
    t = grid["g"].to_numpy() * DT
    assert not np.any((t > 10.0 + 1e-9) & (t < 12.0 - 1e-9))
    assert np.any(t < 10.0) and np.any(t > 12.0)
