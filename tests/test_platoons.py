"""Human platoon windows of protocol P2h (trafficloop/data/platoons.py, docs/protocols.md, P2h):
chains on fabricated tracks, window rules, selection without shared vehicles, the profile schema,
the NGSIM file layout and P2 on human windows of unequal length."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from test_p2_platoon import short_pulse

from trafficloop.data.loaders import ngsim
from trafficloop.data.platoons import (
    CACHE,
    HUMAN,
    Candidate,
    Chain,
    best_window,
    find_chains,
    grid_frame,
    pair_runs,
    samples,
    select_windows,
    track_profiles,
)
from trafficloop.data.reconstruction import ReconstructionConfig
from trafficloop.models.mechanistic import IDM
from trafficloop.protocols import p2
from trafficloop.protocols.evaluate import RUNNERS, leader_profiles
from trafficloop.protocols.profiles import (
    LeaderProfile,
    RecordedFollower,
    load_profiles,
    profile_from_dict,
    profile_to_dict,
    save_profiles,
)
from trafficloop.sim.platoon import simulate_platoon
from trafficloop.utils import REPO_ROOT, to_plain

DT = 0.1
GAP, LENGTH = 10.0, 4.5  # m
N_MIN, N_MAX = samples(60.0), samples(300.0)
TGSIM_LIKE = ReconstructionConfig(sigma_x=0.3, cutoff_hz=0.5)


def leader_speed(t: np.ndarray) -> np.ndarray:
    return 10.0 + 3.0 * np.sin(2 * np.pi * t / 40.0)


def lane_tracks(n_vehicles: int = 6, duration: float = 150.0, change: tuple[int, int] | None = None,
                automated: tuple[int, ...] = ()) -> pd.DataFrame:
    """Tracks frame of a platoon in lane 2 (vehicle 1 leads, every vehicle repeats its speed at the
    net gap GAP, so the platoon is rigid); ``change = (vehicle, k)`` moves that vehicle to lane 3
    from sample ``k`` on, ``automated`` flags vehicles (TGSIM)."""
    t = np.arange(round(duration / DT) + 1) * DT
    x = 500.0 + 10.0 * t - 3.0 * 40.0 / (2 * np.pi) * (np.cos(2 * np.pi * t / 40.0) - 1.0)
    frames = []
    for k in range(n_vehicles):
        lane = np.full(len(t), 2.0)
        if change is not None and change[0] == k + 1:
            lane[change[1]:] = 3.0
        front = x - k * (GAP + LENGTH)
        frames.append(pd.DataFrame({
            "group": "run1", "site": "site", "vehicle_id": k + 1, "t": t, "x_front": front, "x_rear": front - LENGTH,
            "v": leader_speed(t), "lane": lane, "length": LENGTH, "automated": k + 1 in automated,
        }))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- chains
def test_lane_change_cuts_the_chain():
    runs = pair_runs(grid_frame(lane_tracks(change=(4, 700))))
    rows = lambda follower: runs[runs["follower"] == follower][["leader", "start", "stop"]].values.tolist()  # noqa: E731
    assert rows(4) == [[3, 0, 700]]  # alone in lane 3 afterwards: no leader
    assert rows(5) == [[4, 0, 700], [3, 700, 1501]]  # the lane change ends the run behind 4
    assert rows(1) == [] and set(runs["lane"]) == {2.0}
    chains = find_chains(runs, min_followers=4, min_samples=N_MIN)
    assert {(c.vehicles, c.start, c.stop) for c in chains} == {
        ((1, 2, 3, 4, 5, 6), 0, 700), ((1, 2, 3, 4, 5), 0, 700), ((2, 3, 4, 5, 6), 0, 700), ((1, 2, 3, 5, 6), 700, 1501)
    }
    # with the lane change at 50 s the long chain lasts 50 s only: the chain behind it is left
    chains = find_chains(pair_runs(grid_frame(lane_tracks(change=(4, 500)))), 4, N_MIN)
    assert [(c.vehicles, c.start, c.stop) for c in chains] == [((1, 2, 3, 5, 6), 500, 1501)]
    assert find_chains(runs, min_followers=6, min_samples=N_MIN) == []


def test_automated_vehicles_only_lead_and_gaps_must_be_valid():
    runs = pair_runs(grid_frame(lane_tracks(automated=(3,))))
    assert 3 not in set(runs["follower"]) and 3 in set(runs["leader"])  # an automated vehicle may lead
    assert [c.vehicles for c in find_chains(runs, 3, N_MIN)] == [(3, 4, 5, 6)]
    frame = lane_tracks()
    jump = (frame["vehicle_id"] == 5) & np.round(frame["t"] / DT).between(700, 705)
    frame.loc[jump, ["x_front", "x_rear"]] += GAP + 0.5  # 0.6 s of a negative gap behind vehicle 4
    runs = pair_runs(grid_frame(frame))
    assert runs[runs["follower"] == 5][["start", "stop"]].values.tolist() == [[0, 700], [706, 1501]]
    assert pair_runs(grid_frame(frame), max_spacing=GAP - 0.01).empty  # every other gap is GAP


# ----------------------------------------------------------------------------- window rules
def test_window_rules():
    t = np.arange(5001) * DT
    v = leader_speed(t)
    assert best_window(v[:600], N_MIN, N_MAX) is None  # shorter than 60 s
    assert best_window(v[:1000], N_MIN, N_MAX) == (0, 1000)  # the whole interval below 300 s
    slow = np.r_[np.full(50, 2.9), v[:900]]
    assert best_window(slow, N_MIN, N_MAX) == (50, 900)  # starts at the first sample at 3 m/s
    assert best_window(10.0 + 1.45 * np.sin(2 * np.pi * t[:2000] / 40.0), N_MIN, N_MAX) is None  # range 2.9 m/s
    assert best_window(np.r_[np.full(1000, 2.0), v[:600]], N_MIN, N_MAX) is None  # too slow, then too short
    # longer than 300 s: 3001 samples, the start with the largest variance of the leader speed
    wave = 15.0 + np.where(t < 400.0, 1.0, 4.0) * np.sin(2 * np.pi * t / 30.0)
    start, n = best_window(wave, N_MIN, N_MAX)
    variances = np.array([np.var(wave[k : k + N_MAX]) for k in range(len(wave) - N_MAX + 1)])
    assert n == N_MAX and start > 999 and variances[start] == pytest.approx(variances.max(), rel=1e-9)
    # the range is taken over the window only: a dip outside it does not count
    flat = np.full(5001, 15.0)
    flat[4000:4300] -= 5.0 * np.sin(np.pi * np.arange(300) / 300)  # at least 3 m/s deep on 4061..4239
    start, n = best_window(flat, N_MIN, N_MIN)
    assert n == N_MIN and 4061 - N_MIN < start <= 4239 and np.ptp(flat[start : start + n]) >= 3.0
    assert best_window(flat[:4000], N_MIN, N_MAX) is None


def candidate(group: int, vehicles, start: int, n: int) -> Candidate:
    return Candidate(group, Chain(tuple(vehicles), 2.0, start, start + n), start, n)


def test_selection_ranks_and_never_reuses_a_vehicle():
    candidates = [
        candidate(0, [1, 2, 3, 4, 5], 0, 700),
        candidate(0, [1, 2, 3, 4, 5, 6], 0, 650),  # most followers: first, although shorter
        candidate(0, [7, 8, 9, 10, 11], 0, 900),  # the longest with four followers
        candidate(0, [6, 12, 13, 14, 15], 800, 700),  # vehicle 6 is taken (another time does not matter)
        candidate(1, [6, 12, 13, 14, 15], 0, 700),  # the same ids in another group are other vehicles
        candidate(0, [20, 21, 22, 23, 24], 0, 800),  # rejected by the reconstruction step
    ]
    chosen, rejected = select_windows(candidates, lambda c: None if c.chain.vehicles[0] == 20 else c, max_windows=10)
    assert [(c.group, c.chain.vehicles) for c in chosen] == [
        (0, (1, 2, 3, 4, 5, 6)), (0, (7, 8, 9, 10, 11)), (1, (6, 12, 13, 14, 15))
    ]
    assert rejected == 1
    used = [(c.group, v) for c in chosen for v in c.chain.vehicles]
    assert len(used) == len(set(used))
    chosen, _ = select_windows(candidates, lambda c: c, max_windows=2)
    assert [c.chain.vehicles for c in chosen] == [(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11)]


# ----------------------------------------------------------------------------- profiles
@pytest.fixture(scope="module")
def fabricated_profiles():
    frame = lane_tracks(n_vehicles=6, change=(4, 700))
    return track_profiles([(frame, {})], "tgsim", TGSIM_LIKE)


def test_profiles_of_fabricated_tracks(fabricated_profiles, tmp_path):
    profiles, records, counts = fabricated_profiles
    assert counts == {"site": {"groups": 1, "chains": 4, "candidates": 4, "rejected": 0, "windows": 1}}
    (prof,) = profiles  # the chain after the lane change shares vehicles with the first one
    assert prof.name == "tgsim_site-run1_lane2_v1_f0" and prof.source == "tgsim/site/run1/lane2/leader1/frame0"
    assert prof.is_recorded and not prof.is_openacc and prof.t0 == 0.0 and len(prof.v) == 700
    assert [f.index for f in prof.followers] == [1, 2, 3, 4, 5] and {f.driver for f in prof.followers} == {HUMAN}
    assert prof.vehicle_length == pytest.approx(LENGTH)
    t = np.arange(700) * DT
    np.testing.assert_allclose(prof.v, leader_speed(t), atol=0.05)
    for f in prof.followers:
        assert len(f.v) == len(f.s) == 700
        np.testing.assert_allclose(f.s, GAP, atol=0.05)
        np.testing.assert_allclose(f.v, prof.v, atol=0.02)
    (record,) = records
    assert record["vehicles"] == ["1", "2", "3", "4", "5", "6"] and record["duration"] == pytest.approx(69.9)
    assert record["gap"]["min_raw"] == pytest.approx(GAP) and record["reconstruction"]["status"] == ["solved"]
    json.dumps(record)
    # schema round trip: the LeaderProfile layout of the OpenACC cache, nothing else
    item = profile_to_dict(prof)
    assert set(item) == {"name", "source", "t0", "vehicle_length", "v", "followers"}
    assert set(item["followers"][0]) == {"index", "driver", "v", "s"}
    back = load_profiles(save_profiles(profiles, tmp_path / "human.json", {"windows": records}))
    for a, b in ((back[0], prof), (profile_from_dict(item), prof)):
        assert (a.name, a.source, a.t0, a.vehicle_length) == (b.name, b.source, b.t0, b.vehicle_length)
        np.testing.assert_array_equal(a.v, b.v)
        for fa, fb in zip(a.followers, b.followers):
            assert (fa.index, fa.driver) == (fb.index, fb.driver)
            np.testing.assert_array_equal(fa.v, fb.v)
            np.testing.assert_array_equal(fa.s, fb.s)
    assert track_profiles([(lane_tracks(n_vehicles=6, change=(4, 700)), {})], "tgsim", TGSIM_LIKE)[1] == records


FT = ngsim.FEET
T0 = 1_118_846_979_700  # ms


def ngsim_rows(classes, lengths_ft, duration: float = 80.0, gap_ft: float = 30.0) -> pd.DataFrame:
    """One NGSIM file of us-101: vehicles 11, 12, ... in lane 2 in platoon order (download-tool
    columns); every vehicle repeats the leader speed 33 + 10 sin(2 pi t / 40) ft/s."""
    tau = np.arange(round(duration * 10) + 1) * 0.1
    y = 1500.0 + 33.0 * tau - 10.0 * 40.0 / (2 * np.pi) * (np.cos(2 * np.pi * tau / 40.0) - 1.0)
    speed = 33.0 + 10.0 * np.sin(2 * np.pi * tau / 40.0)
    frames, offset = [], 0.0
    for i, (cls, length) in enumerate(zip(classes, lengths_ft)):
        if i:
            offset += lengths_ft[i - 1] + gap_ft
        frames.append(pd.DataFrame({
            "vehicle_id": 11 + i, "frame_id": np.arange(len(tau)) + 1, "global_time": T0 + 100 * np.arange(len(tau)),
            "local_y": y - offset, "v_length": length, "v_class": cls, "v_vel": speed, "v_acc": 0.0, "lane_id": 2,
            "preceding": 0 if i == 0 else 10 + i,
        }))
    return pd.concat(frames, ignore_index=True)


def test_ngsim_file_to_windows(tmp_path):
    """The loader's tracks: a truck follows no one (follower classes) but may lead a chain."""
    ngsim_rows([2, 2, 3, 2, 2, 2, 2], [15, 15, 40, 15, 15, 15, 15]).to_parquet(tmp_path / "ngsim_us-101_raw.parquet")
    frames = list(ngsim.load_tracks(tmp_path, locations=["us-101"]))
    assert len(frames) == 1 and frames[0][0]["frame_id"].min() == 1
    profiles, records, counts = track_profiles(frames, "ngsim", ReconstructionConfig(sigma_x=0.5, cutoff_hz=0.35))
    (prof,) = profiles
    assert prof.source == "ngsim/us-101/us-101-p0/lane2/leader13/frame1" and prof.name == "ngsim_us-101-p0_lane2_v13_f1"
    assert len(prof.followers) == 4 and len(prof.v) == 801 and prof.t0 == 0.0
    assert prof.vehicle_length == pytest.approx((40 + 4 * 15) / 5 * FT)
    assert records[0]["classes"] == ["truck", "auto", "auto", "auto", "auto"]
    np.testing.assert_allclose(prof.followers[0].s, 30 * FT, atol=0.05)  # behind the truck: its rear bumper
    assert counts["us-101"] == {"groups": 1, "chains": 1, "candidates": 1, "rejected": 0, "windows": 1}
    assert track_profiles(frames, "ngsim", ReconstructionConfig(), min_followers=5)[0] == []  # four followers only


# ----------------------------------------------------------------------------- P2 on human windows
def human_profile(name: str, duration: float, scales: dict[int, float], s_init: float) -> tuple[LeaderProfile, object]:
    """A human window whose recorded followers are the IDM platoon itself, the speed deviations of
    follower ``n`` scaled by ``1 + scales[n]``; returns the profile and the simulated platoon."""
    t = np.arange(round(duration / DT) + 1) * DT
    v_lead = 12.0 + 3.0 * np.sin(2 * np.pi * t / 20.0)
    sim = simulate_platoon(IDM(), v_lead, 10, s_init)
    start = sim.manoeuvre_start
    followers = []
    for n, c in scales.items():
        v = sim.v[n, start:]
        followers.append(RecordedFollower(n, v.mean() + (1 + c) * (v - v.mean()), sim.s[n - 1, start:], HUMAN))
    return LeaderProfile(name, f"ngsim/us-101/us-101-p0/lane2/leader{name}/frame1", v_lead, 0.0, followers, 4.6), sim


def test_evaluate_p2_on_human_windows_of_unequal_length(clean_events):
    s_init, converged = p2.settle_gap(IDM(), 12.0)  # both profiles start at 12 m/s
    assert converged
    exact, _ = human_profile("a", 25.0, {n: 0.0 for n in range(1, 5)}, s_init)  # 4 followers, 25 s
    scaled, sim = human_profile("b", 40.0, {1: 0.0, 2: 0.5, 3: 0.0, 4: 0.0, 5: 0.0, 6: -0.25}, s_init)  # 6, 40 s
    summary, tables = p2.evaluate_p2(IDM(), [exact, scaled, short_pulse()], clean_events, n_vehicles=(5, 10),
                                     speed_range=(0.0, 20.0))
    table = tables["profiles"].set_index(["profile", "n_vehicles"])
    assert table.loc[("a", 10), "n_recorded"] == 4 and table.loc[("b", 10), "n_recorded"] == 6
    assert table.loc[("a", 10), "concave_growth_error"] == pytest.approx(0.0, abs=1e-9)
    assert table.loc[("a", 10), "hysteresis_error"] == pytest.approx(0.0, abs=1e-6)
    std = sim.v[:, sim.manoeuvre_start:].std(axis=1)
    expected = math.sqrt((0.5 * std[2]) ** 2 / 6 + (0.25 * std[6]) ** 2 / 6)
    assert table.loc[("b", 10), "concave_growth_error"] == pytest.approx(expected, rel=1e-9)
    # recorded windows of any source enter the means; the pulse does not
    assert summary["concave_growth_error"] == pytest.approx(expected / 2, rel=1e-9)
    assert summary["n_profiles"] == 3 and summary["n_profiles_in_range"] == 3
    assert summary["concave_growth_error_in_range"] == pytest.approx(summary["concave_growth_error"])
    for key in ("hysteresis_error", "growth_exponent_5", "growth_exponent_10", "hysteresis_area_10"):
        assert np.isfinite(summary[key]), key
    assert summary["collisions_10"] == 0 and summary["n_invalid"] == 0
    vehicles = tables["vehicles"]
    assert len(vehicles) == 3 * 11 and vehicles.groupby("profile")["std_v_rec"].count().to_dict() == {
        "a": 4, "b": 6, "braking_pulse": 0
    }


# ----------------------------------------------------------------------------- protocol p2h
def test_p2h_is_p2_on_the_human_cache(tmp_path):
    conf = OmegaConf.load(REPO_ROOT / "conf" / "evaluate.yaml")
    base, human = to_plain(conf.p2), to_plain(conf.p2h)
    assert base.pop("profiles") == "data/scenarios/openacc_profiles.json" and human.pop("profiles") == CACHE
    assert human == base and RUNNERS["p2h"] is RUNNERS["p2"]
    pulse = {"v0": 15.0, "v_low": 6.0, "hold_low": 2.0, "tail": 3.0}
    missing = {"profiles": str(tmp_path / "human_profiles.json"), "window": 300.0}
    with pytest.raises(FileNotFoundError, match="build_human_profiles"):
        leader_profiles(None, missing, pulse)  # never built implicitly
    window = LeaderProfile("w", "tgsim/site/run1/lane2/leader1/frame0", np.full(5, 12.0), 0.0,
                           [RecordedFollower(1, np.full(5, 12.0), np.full(5, 20.0), HUMAN)], 4.5)
    save_profiles([window], missing["profiles"])
    loaded = leader_profiles(None, missing, pulse)
    assert [p.name for p in loaded] == ["w", "braking_pulse"] and loaded[0].is_recorded and not loaded[1].is_recorded
