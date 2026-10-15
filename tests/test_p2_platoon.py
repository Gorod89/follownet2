"""P2: platoon simulator, leader profiles and platoon metrics (docs/protocols.md, section 2)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from trafficloop.data.batch import WARMUP, EventBatch
from trafficloop.models.base import ThreeChannelAdapter
from trafficloop.models.mechanistic import IDM
from trafficloop.protocols import p2
from trafficloop.protocols import profiles as profiles_module
from trafficloop.protocols.profiles import (
    CACHE,
    LeaderProfile,
    RecordedFollower,
    braking_pulse,
    load_profiles,
    openacc_profiles,
    save_profiles,
)
from trafficloop.protocols.scenarios import HOLD, initial_gap, scenario_event, with_hold
from trafficloop.sim.platoon import simulate_platoon
from trafficloop.sim.rollout import rollout
from trafficloop.stability.analytic import equilibrium_gap
from trafficloop.utils import resolve_path

DT = 0.1
N_HOLD = int(round(HOLD / DT))
SUMMARY_KEYS = {
    "concave_growth_error", "hysteresis_error", "growth_exponent_20", "growth_exponent_100", "hysteresis_area_20",
    "hysteresis_area_100", "wave_speed", "wave_speed_100", "wave_speed_pulse", "wave_speed_pulse_100",
    "n_wave_profiles", "fd_error", "fd_error_rel", "fd_bins_compared", "collisions_20", "collisions_100",
    "n_profiles", "settled_share", "s_init_converged_share", "n_invalid",
    "n_profiles_in_range", "concave_growth_error_in_range", "hysteresis_error_in_range", "growth_exponent_20_in_range",
    "growth_exponent_100_in_range", "wave_speed_in_range", "collisions_20_in_range", "collisions_100_in_range",
    "settled_share_in_range",
}


def short_pulse() -> LeaderProfile:
    """20 s manoeuvre: 25 -> 10 m/s at -3 m/s^2, 2 s low, back at +1.5 m/s^2, 3 s at 25 m/s."""
    return braking_pulse(hold_low=2.0, tail=3.0)


def run(n: int, profile: LeaderProfile | None = None, model=None, dtype=torch.float32, s_init: float | None = None):
    profile = profile or short_pulse()
    s_init = initial_gap(profile.v[0]) if s_init is None else s_init
    return simulate_platoon(model or IDM(), profile.v, n, s_init, dtype=dtype)


@pytest.fixture(scope="module")
def platoon10():
    return run(10)


# ----------------------------------------------------------------------------- simulator
def test_braking_pulse_series():
    v = braking_pulse().v
    assert len(v) == 2651 and v[0] == v[-1] == 25.0 and v.min() == 10.0  # 5 + 10 + 10 + 240 s
    acc = np.round(np.diff(v) / DT, 9)
    assert set(np.unique(acc)) == {-3.0, 0.0, 1.5}
    assert np.sum(acc == -3.0) == 50 and np.sum(acc == 1.5) == 100 and np.sum(v == 10.0) == 101
    assert len(short_pulse().v) == 201


def test_idm_platoon_on_the_pulse(platoon10):
    r, v_lead = platoon10, short_pulse().v
    n_steps = N_HOLD + len(v_lead)
    assert r.x.shape == r.v.shape == r.a.shape == (11, n_steps) and r.s.shape == (10, n_steps)
    assert r.manoeuvre_start == N_HOLD and r.n_invalid == 0 and not r.collided.any()
    assert np.all(r.first_collision == -1) and r.s[:, WARMUP + 1 :].min() > 5.0
    np.testing.assert_array_equal(r.v[0], with_hold(v_lead)[0])
    s_init = np.float32(initial_gap(25.0))
    assert np.all(r.s[:, : WARMUP + 1] == s_init) and np.all(r.v[1:, : WARMUP + 1] == 25.0)
    # gaps are the position differences; the speeds integrate the realised accelerations
    np.testing.assert_allclose(r.s, r.x[:-1] - r.x[1:], atol=1e-3)
    k = np.arange(WARMUP, n_steps - 1)
    np.testing.assert_allclose(r.v[1:, k + 1], r.v[1:, k] + r.a[1:, k] * DT, atol=1e-4)
    assert np.all(r.a[1:, :WARMUP] == 0.0) and r.a.min() >= -8.0 - 1e-4 and r.a.max() <= 4.0 + 1e-4
    assert r.v[1:, N_HOLD:].min() < 15.0  # the followers take part in the pulse


def test_prefix_property():
    big, small = run(100), run(20)
    head = big.prefix(20)
    for name in ("x", "v", "a", "s", "collided", "first_collision", "invalid"):
        np.testing.assert_array_equal(getattr(head, name), getattr(small, name), err_msg=name)
    assert head.n_invalid == small.n_invalid and head.manoeuvre_start == small.manoeuvre_start


def test_determinism(platoon10):
    again = run(10)
    for name in ("x", "v", "a", "s", "collided", "first_collision", "invalid"):
        np.testing.assert_array_equal(getattr(again, name), getattr(platoon10, name), err_msg=name)


def test_single_follower_is_the_rollout_of_the_scenario():
    """The step rule is that of sim/rollout.py: one follower equals the rolled scenario event."""
    pulse = short_pulse()
    s_init = initial_gap(pulse.v[0])
    r = run(1, pulse, dtype=torch.float64)
    batch = EventBatch.from_events([scenario_event(pulse.v, s_init, "pulse")], dtype=torch.float64)
    ref = rollout(IDM(), batch)
    np.testing.assert_allclose(r.s[0], ref.s[0].numpy(), atol=1e-8)
    np.testing.assert_allclose(r.v[1], ref.v[0].numpy(), atol=1e-9)
    np.testing.assert_allclose(r.a[1], ref.a[0].numpy(), atol=1e-7)


def test_non_finite_and_clipped_outputs():
    constant = lambda value: ThreeChannelAdapter(lambda h: torch.full(h.shape[:-2], value, dtype=h.dtype), name="const")  # noqa: E731
    steps = N_HOLD + len(short_pulse().v) - 1 - WARMUP
    r = run(10, model=constant(float("nan")))
    assert r.n_invalid == 10 * steps and np.all(r.invalid == steps)
    assert r.v[1, WARMUP + 1] == pytest.approx(25.0 - 0.8, abs=1e-5)  # replaced by the lower clip
    assert r.v[1:, -1].max() == 0.0 and not r.collided.any()
    # +1e9 is clipped to +4; equal speeds keep the gaps behind follower 1, which drives through
    r = run(10, model=constant(1e9))
    assert r.n_invalid == 0 and r.a[1:, WARMUP:-1].max() == pytest.approx(4.0, abs=1e-2)  # float32 at 300 m/s
    assert r.collided.tolist() == [True] + [False] * 9
    assert r.first_collision[0] > WARMUP and r.s[0, -1] < -100.0  # a collided vehicle keeps driving


# ----------------------------------------------------------------------------- metric helpers
def test_shoelace_area_of_a_rectangle():
    edge = np.linspace(0.0, 1.0, 11)[:-1]
    s = np.concatenate([10 + 2 * edge, np.full(10, 12.0), 12 - 2 * edge, np.full(10, 10.0)])
    v = np.concatenate([np.full(10, 20.0), 20 + 3 * edge, np.full(10, 23.0), 23 - 3 * edge])
    assert p2.shoelace_area(s, v) == pytest.approx(6.0, rel=1e-12)  # counterclockwise
    assert p2.shoelace_area(s[::-1], v[::-1]) == pytest.approx(-6.0, rel=1e-12)
    np.testing.assert_allclose(p2.shoelace_area(np.stack([s, s + 1e4]), np.stack([v, 2 * v])), [6.0, 12.0], rtol=1e-9)


def crossing_pattern(n_veh=8, shift=15, spacing=25.0, n_steps=400):
    """Newell's pattern: vehicle n repeats the leader ``shift`` samples later and ``spacing`` m
    behind, so the threshold crossings lie on a line of slope -spacing / (shift dt)."""
    pad = shift * (n_veh - 1)
    k = np.arange(-pad, n_steps)
    v = np.clip(20.0 - 9.6 * (k - 100) / 51.0, 10.4, 20.0)  # 20 -> 10.4 m/s over k = 100..151
    x = np.concatenate([[0.0], np.cumsum(v[1:]) * DT])
    rows = [slice(pad - n * shift, pad - n * shift + n_steps) for n in range(n_veh)]
    return np.stack([v[r] for r in rows]), np.stack([x[r] - n * spacing for n, r in enumerate(rows)])


def test_wave_speed_of_a_crossing_pattern():
    v, x = crossing_pattern()
    t_cross, x_cross = p2.wave_crossings(v, x, start=50)
    # the threshold (20 + 10.4) / 2 is the middle of the ramp: the leader crosses it at k = 125.5
    lead = (125.5 - 50) * DT
    np.testing.assert_allclose(t_cross, lead + 1.5 * np.arange(8), atol=1e-9)
    speed, n_crossed = p2.fit_wave_speed(t_cross, x_cross)
    assert n_crossed == 8 and speed == pytest.approx(-25.0 / 1.5, rel=1e-9)
    assert math.isnan(p2.fit_wave_speed(t_cross[:4], x_cross[:4])[0])  # fewer than five vehicles
    flat = p2.wave_crossings(np.full((3, 100), 20.0), np.zeros((3, 100)), start=40)
    assert np.isnan(flat[0]).all()  # no deceleration, no wave


def test_fd_binning():
    k, q = p2.fd_samples(np.array([15.5, -4.5]), np.array([20.0, 1.0]), 4.5)
    assert k[0] == pytest.approx(50.0) and q[0] == pytest.approx(3600.0) and not np.isfinite(k[1])
    sim = p2.fd_histogram(
        np.r_[np.full(300, 12.0), np.full(300, 22.0), np.full(100, 32.0), [-1.0, 150.0, np.inf]],
        np.r_[np.full(300, 1000.0), np.full(300, 1500.0), np.full(100, 1700.0), [0.0, 0.0, 0.0]],
    )
    assert len(sim[1]) == 30 and sim[1][[2, 4, 6]].tolist() == [300, 300, 100] and sim[1].sum() == 700
    src = p2.fd_histogram(
        np.r_[np.full(250, 13.0), np.full(400, 24.9), np.full(500, 30.0)],
        np.r_[np.full(250, 900.0), np.tile([1400.0, 1600.0], 200), np.full(500, 2000.0)],
    )
    error, relative, n_bins = p2.fd_compare(sim, src)
    assert n_bins == 2  # bin 6 has 100 simulated samples only
    assert error == pytest.approx(math.sqrt(100.0**2 / 2)) and relative == pytest.approx(math.sqrt((100 / 900) ** 2 / 2))
    assert p2.fd_compare(sim, p2.source_histogram([]))[2] == 0
    events = [scenario_event(np.full(301, 20.0), 15.5, "a"), scenario_event(np.full(201, 20.0), 15.5, "b")]
    events[1].meta["leader_length"] = 10.0  # k = 1000 / 25.5 = 39.2 veh/km
    sums, counts = p2.source_histogram(events)
    assert counts[10] == len(events[0]) and counts[7] == len(events[1]) and counts.sum() == 901 + 801  # hold included
    assert sums[10] / counts[10] == pytest.approx(3600.0)


def test_growth_exponent_of_a_power_law():
    n = np.arange(1, 51, dtype=np.float64)
    std = np.r_[7.0, 0.3 * n**0.6]  # index 0 is the leader
    assert p2.growth_exponent(std) == pytest.approx(0.6, rel=1e-12)
    std[[5, 17]] = 0.0  # followers without dispersion are left out
    assert p2.growth_exponent(std) == pytest.approx(0.6, rel=1e-12)
    assert math.isnan(p2.growth_exponent([1.0, 0.5, 0.0]))


# ----------------------------------------------------------------------------- protocol
def test_settle_gap():
    s_e, converged = p2.settle_gap(IDM(), 25.0)
    assert converged and s_e == pytest.approx(float(equilibrium_gap(IDM(), [25.0])[0]), abs=0.02)  # 47.8 m
    r = run(10, s_init=s_e)  # a platoon started there is settled at the end of the hold
    assert p2.settled_vehicles(r).all() and np.abs(r.s[:, : r.manoeuvre_start] - s_e).max() < 0.01
    pushy = ThreeChannelAdapter(lambda h: torch.ones(h.shape[:-2], dtype=h.dtype), name="pushy")
    assert p2.settle_gap(pushy, 25.0) == (initial_gap(25.0), False)


def recorded_like(sim, start: int, scales: dict[int, float]) -> list[RecordedFollower]:
    """Recorded followers whose speed deviations are the simulated ones scaled by ``1 + c``."""
    out = []
    for n, c in scales.items():
        v = sim.v[n, start:]
        out.append(RecordedFollower(n, v.mean() + (1 + c) * (v - v.mean()), sim.s[n - 1, start:] + 5.0, "ACC"))
    return out


def test_errors_against_recorded_followers_and_summary():
    t = np.arange(201) * DT
    v_lead = 20.0 + 3.0 * np.sin(2 * np.pi * t / 10.0)
    s_init, converged = p2.settle_gap(IDM(), v_lead[0])  # evaluate_p2 starts the platoon there
    assert converged
    sim = run(10, LeaderProfile("ref", "openacc/x.csv", v_lead, 0.0), s_init=s_init)
    start = sim.manoeuvre_start
    std = sim.v[:, start:].std(axis=1)
    area = p2.shoelace_area(sim.s[:, start:], sim.v[1:, start:])  # area[n - 1] of follower n
    scales_a, scales_b = {1: 0.0, 2: 0.5, 3: -0.25}, {1: 0.2}
    profiles = [
        LeaderProfile("fab_a", "openacc/test/a.csv", v_lead, 12.0, recorded_like(sim, start, scales_a)),
        LeaderProfile("fab_b", "openacc/test/b.csv", v_lead, 34.0, recorded_like(sim, start, scales_b)),
        short_pulse(),
    ]
    summary, tables = p2.evaluate_p2(IDM(), profiles, [])

    def rms(values):
        return math.sqrt(np.mean(np.square(values)))

    concave = [rms([c * std[n] for n, c in s.items()]) for s in (scales_a, scales_b)]
    hysteresis = [rms([c * area[n - 1] for n, c in s.items()]) for s in (scales_a, scales_b)]
    assert summary["concave_growth_error"] == pytest.approx(np.mean(concave), rel=1e-9)
    assert summary["hysteresis_error"] == pytest.approx(np.mean(hysteresis), rel=1e-9)

    assert set(summary) == SUMMARY_KEYS and all(type(value) in (int, float) for value in summary.values())
    json.dumps(summary)
    assert summary["n_profiles"] == 3 and summary["n_invalid"] == 0
    assert summary["collisions_20"] == summary["collisions_100"] == 0
    assert math.isnan(summary["fd_error"]) and summary["fd_bins_compared"] == 0  # no source events
    assert summary["s_init_converged_share"] == 1.0 and summary["settled_share"] == 1.0  # started at equilibrium
    assert summary["n_profiles_in_range"] == 3  # no speed range: every profile is in range
    for key in ("concave_growth_error", "hysteresis_error", "growth_exponent_20", "growth_exponent_100", "wave_speed",
                "collisions_20", "collisions_100", "settled_share"):
        assert summary[f"{key}_in_range"] == pytest.approx(summary[key], nan_ok=True), key
    assert summary["hysteresis_area_20"] > summary["hysteresis_area_100"] > 0.0  # the 20 s manoeuvre misses the tail
    profiles_table, vehicles = tables["profiles"], tables["vehicles"]
    assert len(profiles_table) == 6 and profiles_table["n_vehicles"].tolist() == [20, 100] * 3
    assert profiles_table["s_init_converged"].all() and profiles_table["s_init"].iloc[0] == s_init
    wave_20 = profiles_table[profiles_table["n_vehicles"] == 20].set_index("profile")["wave_speed"]
    assert summary["wave_speed_pulse"] == pytest.approx(wave_20["braking_pulse"])  # leader and 20 followers
    assert summary["n_wave_profiles"] == int(np.isfinite(wave_20[["fab_a", "fab_b"]]).sum())
    pulse_rows = profiles_table[profiles_table["profile"] == "braking_pulse"]
    assert pulse_rows["concave_growth_error"].isna().all() and (pulse_rows["n_recorded"] == 0).all()
    assert len(vehicles) == 3 * 101
    assert {"profile", "index", "std_v", "area", "t_cross", "x_cross", "collided", "settled"} <= set(vehicles.columns)
    first = vehicles[vehicles["profile"] == "fab_a"]
    np.testing.assert_allclose(first["std_v"].to_numpy()[:11], std, rtol=1e-12)  # the prefix property again
    np.testing.assert_allclose(first["std_v_rec"].to_numpy()[1:4], [(1 + c) * std[n] for n, c in scales_a.items()], rtol=1e-9)


def test_speed_range_flags_and_summary():
    t = np.arange(201) * DT
    wave = np.sin(2 * np.pi * t / 10.0)
    fast = LeaderProfile("fast", "openacc/test/fast.csv", 21.5 + 0.5 * wave, 0.0)  # up to 22 > 20 + 1 m/s
    slow = LeaderProfile("slow", "openacc/test/slow.csv", 8.0 + 2.0 * wave, 0.0)
    for profile in (fast, slow):
        profile.followers = [RecordedFollower(1, profile.v * 0.9 + 1.0, 20.0 + 3.0 * wave, "ACC")]
    flags = [p2.in_speed_range(p, (0.0, 20.0)) for p in (fast, slow, short_pulse())]
    assert flags == [False, True, True]  # the synthetic pulse at 25 m/s is always in range
    edge = LeaderProfile("edge", "openacc/test/edge.csv", np.array([4.0, 21.0]), 0.0)
    assert p2.in_speed_range(edge, (5.0, 20.0)) and not p2.in_speed_range(edge, (5.1, 20.0))
    assert not p2.in_speed_range(edge, (5.0, 19.9)) and p2.in_speed_range(edge, None)

    summary, tables = p2.evaluate_p2(IDM(), [fast, slow, short_pulse()], [], n_vehicles=(5, 10), speed_range=(0.0, 20.0))
    table, vehicles = tables["profiles"], tables["vehicles"]
    assert table["in_range"].tolist() == [False, False, True, True, True, True]
    assert summary["n_profiles_in_range"] == 2 and summary["n_profiles"] == 3
    row = table[(table["profile"] == "slow") & (table["n_vehicles"] == 10)].iloc[0]
    assert summary["concave_growth_error_in_range"] == pytest.approx(row["concave_growth_error"])
    assert summary["hysteresis_error_in_range"] == pytest.approx(row["hysteresis_error"])
    assert summary["growth_exponent_10_in_range"] == pytest.approx(row["growth_exponent"])
    assert summary["wave_speed_in_range"] == pytest.approx(row["wave_speed"], nan_ok=True)  # 10 < 20 followers
    assert summary["concave_growth_error"] != pytest.approx(summary["concave_growth_error_in_range"])  # fast counts
    inside = table[table["in_range"] & (table["n_vehicles"] == 5)]
    assert summary["collisions_5_in_range"] == inside["collisions"].sum()
    followers = vehicles[vehicles["profile"].isin(["slow", "braking_pulse"]) & (vehicles["index"] > 0)]
    assert summary["settled_share_in_range"] == pytest.approx(followers["settled"].mean())

    none, _ = p2.evaluate_p2(IDM(), [fast], [], n_vehicles=(5,), speed_range=(0.0, 20.0))
    assert none["n_profiles_in_range"] == 0
    assert all(none[key] is None for key in none if key.endswith("_in_range") and key != "n_profiles_in_range")
    assert '"settled_share_in_range": null' in json.dumps(none)


# ----------------------------------------------------------------------------- OpenACC windows
def write_run(path, columns: dict[str, np.ndarray], vehicles: list[str], flag: str = "1") -> None:
    pad = "," * 20
    lines = [
        "Date,4,7,2019" + pad,
        "Vehicle_order," + ",".join(vehicles) + pad,
        f"Number_of_vehicles,{len(vehicles)}" + pad,
        f"ACC,{flag}" + pad,
        "Distance_setting,min" + pad,
        ",".join(columns),
    ]
    for row in zip(*columns.values()):
        lines.append(",".join(cell if isinstance(cell, str) else ("" if np.isnan(cell) else f"{cell:.4f}") for cell in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_openacc_window_selection(tmp_path):
    folder = tmp_path / "AstaZero"
    folder.mkdir()
    n = 811  # 81 s at 10 Hz
    i = np.arange(n)
    lead = np.full(n, 15.0)
    lead[:200] += 6.0 * np.sin(2 * np.pi * i[:200] / 50)  # largest variance, but vehicle 3 nearly stops
    lead[200:400] += 2.0 * np.sin(2 * np.pi * i[200:400] / 50)  # the expected window: start 200
    lead[500:700] += 4.0 * np.sin(2 * np.pi * i[500:700] / 50)  # second largest, but IVS3 is missing at 600
    columns = {"Time": 1.0 + 0.1 * i}
    for k in range(1, 6):
        columns[f"Speed{k}"] = lead + 0.1 * (k - 1)
    columns["Speed3"][150] = 0.3
    for k in range(1, 6):
        columns[f"Driver{k}"] = np.full(n, "Human" if k == 5 else "ACC", dtype=object)
    for k in range(1, 5):
        columns[f"IVS{k}"] = 20.0 + k + 0.01 * i
    columns["IVS3"][600] = np.nan
    vehicles = ["Audi(A8)", "Tesla(Model3)", "BMW(X5)", "Audi(A6)", "Mercedes(AClass)"]
    write_run(folder / "ASta_040719_platoon1.csv", columns, vehicles)
    short = {name: values[:150] for name, values in columns.items()}
    write_run(folder / "ASta_040719_platoon2.csv", short, vehicles)  # shorter than the window
    (folder / "AstaZero_veh_specifications.csv").write_text(
        "Vehicles,Length (mm),Top speed (km/h)\nAudi (A8),5300,250\nTesla (Model 3),4700,210\n"
        "BMW (X5),4900,230\nMercedes (A Class),4400,250\nAudi (A6),4900,246\n",
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="platoon2"):
        profiles = openacc_profiles(tmp_path, window=20.0)
    assert len(profiles) == 1
    prof = profiles[0]
    assert prof.name == "ASta_040719_platoon1" and prof.source == "openacc/AstaZero/ASta_040719_platoon1.csv"
    assert prof.is_openacc and prof.t0 == pytest.approx(21.0) and len(prof.v) == 201
    np.testing.assert_allclose(prof.v, lead[200:401], atol=1e-4)
    assert [f.index for f in prof.followers] == [1, 2, 3, 4]
    assert [f.driver for f in prof.followers] == ["ACC", "ACC", "ACC", "Human"]
    np.testing.assert_allclose(prof.followers[1].v, columns["Speed3"][200:401], atol=1e-4)
    np.testing.assert_allclose(prof.followers[2].s, columns["IVS3"][200:401], atol=1e-4)
    assert prof.vehicle_length == pytest.approx(4.84)  # mean of the five lengths in mm
    path = save_profiles([prof, braking_pulse()], tmp_path / "profiles.json", meta={"note": "test"})
    back = load_profiles(path)
    assert [b.name for b in back] == [prof.name, "braking_pulse"] and back[1].source == "synthetic"
    assert back[0].t0 == prof.t0 and back[0].vehicle_length == prof.vehicle_length
    np.testing.assert_array_equal(back[0].v, prof.v)
    for a, b in zip(back[0].followers, prof.followers):
        assert (a.index, a.driver) == (b.index, b.driver)
        np.testing.assert_array_equal(a.v, b.v)
        np.testing.assert_array_equal(a.s, b.s)


def zalazone_columns(n: int, n_veh: int, amplitude: float, absent: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    """Columns of a ZalaZone run: Time, Speed/E/N/Lon/Lat/Alt per vehicle, then the IVS; no Driver."""
    i = np.arange(n)
    columns = {"Time": 0.2 + 0.1 * i}
    lead = 8.0 + np.where(i < 200, amplitude, amplitude / 3) * np.sin(2 * np.pi * i / 50)
    for k in range(1, n_veh + 1):
        if k not in absent:
            columns[f"Speed{k}"] = lead + 0.05 * (k - 1)
            for name in ("E", "N", "Lon", "Lat", "Alt"):
                columns[f"{name}{k}"] = np.full(n, 100.0 + k)
    for k in range(1, n_veh):
        if k not in absent and k + 1 not in absent:
            columns[f"IVS{k}"] = 10.0 + k + 0.01 * i
    return columns


def test_zalazone_runs(tmp_path, monkeypatch):
    folder = tmp_path / "ZalaZone"
    folder.mkdir()
    names = ["SMART_TARGET", "BMW_I3", "MERCEDES_GLE450", "JAGUAR_I_PACE", "TESLA_MODELX", "TESLA_MODEL3", "AUDI_A4"]
    first = zalazone_columns(401, 7, amplitude=3.0)
    first["Speed6"][50:60] = np.nan  # the window of the largest variance keeps followers 1-4 only
    write_run(folder / "dynamic_part1.csv", first, names)
    write_run(folder / "dynamic_part2.csv", zalazone_columns(401, 7, 3.0, absent=(3,)), names)  # one follower
    write_run(folder / "dynamic_part3.csv", zalazone_columns(401, 6, amplitude=0.6), names[:6])
    with pytest.warns(UserWarning, match="dynamic_part2"):
        profiles = openacc_profiles(tmp_path, window=20.0)
    assert [p.name for p in profiles] == ["dynamic_part1", "dynamic_part3"]
    one, three = profiles
    assert one.source == "openacc/ZalaZone/dynamic_part1.csv" and one.is_openacc and one.t0 == pytest.approx(0.2)
    assert [f.index for f in one.followers] == [1, 2, 3, 4] and {f.driver for f in one.followers} == {"ACC"}
    np.testing.assert_allclose(one.v, first["Speed1"][:201], atol=1e-4)
    np.testing.assert_allclose(one.followers[3].v, first["Speed5"][:201], atol=1e-4)
    np.testing.assert_allclose(one.followers[3].s, first["IVS4"][:201], atol=1e-4)
    assert len(three.followers) == 5 and one.vehicle_length == three.vehicle_length == 4.5
    keep_one = (profiles_module.ProfileCampaign("ZalaZone", "dynamic_part*.csv", keep=1),)
    monkeypatch.setattr(profiles_module, "PROFILE_CAMPAIGNS", keep_one)
    with pytest.warns(UserWarning, match="dynamic_part2"):
        assert [p.name for p in openacc_profiles(tmp_path, window=20.0)] == ["dynamic_part1"]  # largest variance


@pytest.mark.data
def test_cached_openacc_profiles():
    path = resolve_path(CACHE)
    if not path.exists():
        pytest.skip(f"{path} not built (profiles.build_openacc_cache)")
    profiles = load_profiles(path)
    astazero = [p for p in profiles if p.source.startswith("openacc/AstaZero/")]
    zalazone = [p for p in profiles if p.source.startswith("openacc/ZalaZone/")]
    assert len(astazero) == 10 and 1 <= len(zalazone) <= 10 and len(profiles) == len(astazero) + len(zalazone)
    assert all(len(p.followers) == 4 for p in astazero) and all(len(p.followers) >= 4 for p in zalazone)
    for prof in profiles:
        assert len(prof.v) == 3001 and prof.v.min() > 0.5
        assert all(len(f.v) == len(f.s) == 3001 and f.v.min() > 0.5 and np.isfinite(f.s).all() for f in prof.followers)
