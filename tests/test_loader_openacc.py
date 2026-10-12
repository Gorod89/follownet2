"""OpenACC loader: fabricated campaign CSVs with the published metadata lines and quirks."""

from __future__ import annotations

import numpy as np
import pytest

from follownet2.data.extraction import ExtractionConfig, extract_all
from follownet2.data.loaders import openacc
from follownet2.data.loaders.openacc import last_report, load_pairs
from follownet2.utils import resolve_path

REAL_DIR = resolve_path(openacc.DEFAULT_RAW_DIR)
T0 = 31564.7  # campaigns keep a large time offset with float artefacts


def extract(pairs, cfg: ExtractionConfig | None = None) -> list:
    """``extract_all`` plus ``Event.validate()`` on every event."""
    events = extract_all(pairs, cfg or ExtractionConfig())
    for ev in events:
        ev.validate()
    return events


def platoon(n: int = 401, rate_hz: float = 10.0) -> dict[str, np.ndarray]:
    """Three vehicles: 2 closes in on 1, 3 falls back behind 2."""
    tau = np.arange(n) / rate_hz
    return {
        "Time": T0 + tau,
        "Speed1": 15.0 + 0.5 * np.sin(0.2 * tau),
        "Speed2": np.full(n, 15.5),
        "Speed3": np.full(n, 15.2),
        "IVS1": 40.0 - 0.5 * tau + 2.5 * (1.0 - np.cos(0.2 * tau)),  # 40 m + integral of (Speed1 - Speed2)
        "IVS2": 35.0 + 0.3 * tau,
    }


def write_run(path, meta_lines: list[str], columns: dict[str, np.ndarray | list]) -> None:
    """A run file: metadata lines, header, rows; NaN and None are written as empty fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    n = len(columns[names[0]])

    def cell(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return repr(float(value)) if not isinstance(value, str) else value

    rows = [",".join(cell(columns[name][i]) for name in names) for i in range(n)]
    path.write_text("\n".join([*meta_lines, ",".join(names), *rows]) + "\n", encoding="utf-8")


ASTAZERO_META = [
    "Date,4,7,2019,,,,,,,,",
    "Vehicle_order,Audi(A8),Tesla(Model3),BMW(X5),,,,,,,,",
    "Number_of_vehicles,3,,,,,,,,,,",
    "ACC,2,,,,,,,,,,",
    "Distance_setting,min",
]


def test_platoon_gives_one_pair_per_follower_with_the_ivs_gap(tmp_path):
    columns = platoon()
    write_run(
        tmp_path / "AstaZero" / "ASta_040719_platoon3.csv",
        ASTAZERO_META,
        {**columns, "Driver1": ["ACC"] * 401, "Driver2": ["ACC"] * 401, "Driver3": ["Human"] * 401},
    )
    pairs = list(load_pairs(tmp_path))
    assert [(p.follower_id, p.leader_id[0]) for p in pairs] == [
        ("openacc/astazero/tesla(model3)", "openacc/astazero/audi(a8)"),
        ("openacc/astazero/bmw(x5)", "openacc/astazero/tesla(model3)"),
    ]
    first = pairs[0]
    assert first.site == "astazero" and first.dataset == "openacc"
    np.testing.assert_allclose(first.t, np.round(columns["Time"] / 0.1) * 0.1)  # file time kept, on the grid
    assert first.meta == {
        "source_file": "AstaZero/ASta_040719_platoon3.csv",
        "group": "AstaZero/ASta_040719_platoon3",
        "native_rate_hz": 10.0,
        "ids": "source",
        "gap_definition": openacc.GAP_DEFINITION,
        "date": "2019-07-04",
        "platoon_position": 2,
        "acc_setting": "2",
        "distance_setting": "min",
        "acc_on_source": "driver column",
    }
    events = extract(pairs)
    assert [len(ev) for ev in events] == [401, 401]
    closing, opening = events
    np.testing.assert_allclose(closing.s, columns["IVS1"], atol=1e-9)
    np.testing.assert_allclose(closing.v, columns["Speed2"])
    np.testing.assert_allclose(closing.v_lead, columns["Speed1"])
    assert closing.dv[0] == pytest.approx(0.5) and np.mean(closing.dv) > 0 and np.all(opening.dv < 0)
    assert closing.meta["acc_on"] is True and opening.meta["acc_on"] is False
    assert closing.meta["t0"] == pytest.approx(T0)
    assert last_report()["pairs_per_site"] == {"astazero": 2}


@pytest.mark.parametrize(
    "meta_lines",
    [
        ASTAZERO_META[:4],  # no Distance_setting line
        [*ASTAZERO_META, "Comment,recorded twice"],  # a sixth line
        ASTAZERO_META[1:],  # no Date line
    ],
)
def test_header_is_the_line_that_starts_with_time(tmp_path, meta_lines):
    write_run(tmp_path / "Vicolungo" / "run.csv", meta_lines, platoon())
    pairs = list(load_pairs(tmp_path))
    assert [p.follower_id for p in pairs] == ["openacc/vicolungo/tesla(model3)", "openacc/vicolungo/bmw(x5)"]
    assert all(np.all(np.isfinite(p.v)) and len(p.t) == 401 for p in pairs)


def test_file_without_metadata_gets_pseudo_ids(tmp_path):
    write_run(tmp_path / "Casale" / "part3.csv", [], platoon())
    pairs = list(load_pairs(tmp_path))
    assert [p.follower_id for p in pairs] == ["openacc/casale/veh2", "openacc/casale/veh3"]
    assert pairs[0].meta["ids"] == "pseudo" and pairs[0].meta["date"] is None


def test_change_of_driving_mode_ends_an_event(tmp_path):
    columns = platoon(501)
    columns["Driver2"] = ["ACC"] * 250 + ["Human"] * 251
    write_run(tmp_path / "Cherasco" / "JRC-Cher_300818_rec1_part3.csv", ASTAZERO_META, columns)
    pair = next(iter(load_pairs(tmp_path)))
    assert list(pair.segment_key[248:252]) == ["acc", "acc", "human", "human"]
    events = [ev for ev in extract([pair]) if ev.follower_id.endswith("tesla(model3)")]
    assert [(len(ev), ev.meta["acc_on"]) for ev in events] == [(250, True), (251, False)]


def test_zalazone_layout_absent_vehicle_and_file_flag(tmp_path):
    n = 300
    columns = platoon(n)
    zala = {"Time": np.arange(n) * 0.1 + 200.1}  # vehicle 1 absent: no Speed1 nor IVS1 columns
    for k, speed in ((2, columns["Speed1"]), (3, columns["Speed2"]), (4, columns["Speed3"])):
        zala.update({f"Speed{k}": speed, f"E{k}": np.zeros(n), f"N{k}": np.zeros(n), f"Lon{k}": np.full(n, 16.84),
                     f"Lat{k}": np.full(n, 46.89), f"Alt{k}": np.full(n, 230.0)})
    zala.update({"IVS2": columns["IVS1"], "IVS3": columns["IVS2"]})
    meta = ["Date,9,10,2019", "Vehicle_order,SKODA_TARGET,MAZDA_3,TOYOTA_RAV4,AUDI_E_TRON,", "Number_of_vehicles,4"]
    write_run(tmp_path / "ZalaZone" / "handling_part1.csv", [*meta, "ACC,1", "Distance_setting,S"], zala)
    write_run(tmp_path / "ZalaZone" / "handling_part22.csv", [*meta, "ACC,0", "Distance_setting,"], zala)
    pairs = list(load_pairs(tmp_path))
    assert [(p.follower_id, p.meta["source_file"]) for p in pairs] == [
        ("openacc/zalazone/toyota_rav4", "ZalaZone/handling_part1.csv"),
        ("openacc/zalazone/audi_e_tron", "ZalaZone/handling_part1.csv"),
        ("openacc/zalazone/toyota_rav4", "ZalaZone/handling_part22.csv"),
        ("openacc/zalazone/audi_e_tron", "ZalaZone/handling_part22.csv"),
    ]
    assert last_report()["pairs_without_columns"] == 2  # pair 1 of each file
    assert all(p.segment_key is None and p.meta["acc_on_source"] == "file flag" for p in pairs)
    events = extract(pairs)
    assert [ev.meta["acc_on"] for ev in events] == [True, True, False, False]
    np.testing.assert_allclose(events[0].s, columns["IVS1"], atol=1e-9)


def test_unknown_mode_without_driver_column_and_blank_flag(tmp_path):
    meta = ["Date,30,9,2020", "Vehicle_order,Ford,Hyundai", "Number_of_vehicles,2", "ACC,",
            "Distance_setting,second shortest"]
    columns = platoon()
    write_run(tmp_path / "JRC low speed" / "part18.csv", meta,
              {k: columns[k] for k in ("Time", "Speed1", "Speed2", "IVS1")} | {"Driver1": ["ACC"] * 401})
    pair = next(iter(load_pairs(tmp_path)))
    assert pair.site == "jrc_low_speed" and pair.follower_id == "openacc/jrc_low_speed/hyundai"
    assert pair.segment_key is None and pair.meta["acc_on_source"] == "unknown"
    assert pair.meta["distance_setting"] == "second shortest"
    event = extract([pair])[0]
    assert event.meta["acc_on"] is None


def test_empty_fields_cut_the_events(tmp_path):
    columns = platoon()
    columns["IVS1"][200:220] = np.nan  # 2 s without spacing
    columns["Speed2"] = columns["Speed2"].astype(object)
    columns["Speed2"][380:] = None
    write_run(tmp_path / "Casale" / "part4.csv", ASTAZERO_META, columns)
    pair = next(iter(load_pairs(tmp_path)))
    assert np.isnan(pair.x_follower[210]) and np.isnan(pair.v[390])
    assert [len(ev) for ev in extract([pair])] == [200, 160]


def test_the_follower_is_the_same_car_across_files(tmp_path):
    columns = platoon()
    write_run(tmp_path / "Vicolungo" / "VC-JRC_260219_part1.csv",
              ["Vehicle_order,Mitsubishi(SpaceStar),Ford S-Max,KIA(Niro)", "ACC,1"], columns)
    write_run(tmp_path / "Vicolungo" / "VC-JRC_270219_part1.csv",
              ["Vehicle_order,Mitsubishi(SpaceStar),KIA(Niro),Ford S-Max", "ACC,1"], columns)
    followers = [p.follower_id.rsplit("/", 1)[1] for p in load_pairs(tmp_path)]
    assert followers == ["ford_s-max", "kia(niro)", "kia(niro)", "ford_s-max"]
    events = extract(load_pairs(tmp_path))
    assert len({ev.event_id for ev in events}) == 4 and len({ev.follower_id for ev in events}) == 2


def test_off_grid_time_is_interpolated_to_the_grid(tmp_path):
    columns = platoon(801, rate_hz=20.0)
    columns["Time"] = columns["Time"] + 0.02
    write_run(tmp_path / "AstaZero" / "ASta_050719_platoon1.csv", ASTAZERO_META, columns)
    pair = next(iter(load_pairs(tmp_path)))
    np.testing.assert_allclose(pair.t[[0, -1]], [T0 + 0.1, T0 + 40.0])
    assert np.max(np.abs(np.diff(pair.t) - 0.1)) < 1e-6 and pair.meta["native_rate_hz"] == pytest.approx(20.0)
    tau = pair.t - T0
    np.testing.assert_allclose(pair.v_lead, 15.0 + 0.5 * np.sin(0.2 * (tau - 0.02)), atol=1e-4)


def test_mixed_time_gap_setting_from_the_side_file(tmp_path):
    meta = ["Date,9,10,2019", "Vehicle_order,SKODA_TARGET,BMW_I3,MAZDA_3", "Number_of_vehicles,3", "ACC,1",
            "Distance_setting,SL"]
    write_run(tmp_path / "ZalaZone" / "handling_part5.csv", meta, platoon())
    (tmp_path / "ZalaZone" / openacc.CAMPAIGNS["ZalaZone"].long_setting_file).write_text(
        "filename,Long timegap setting\nhandling_part5,BMW_I3\nhandling_part12,BMW_I3/MERCEDES_GLE450\n", encoding="utf-8"
    )
    (tmp_path / "ZalaZone" / "ZalaZONE_veh_specifications.csv").write_text(
        "Vehicles,Max power (kW)\nMazda (3),96\n", encoding="utf-8"
    )
    settings = {p.follower_id: p.meta["distance_setting"] for p in load_pairs(tmp_path)}
    assert settings == {"openacc/zalazone/bmw_i3": "L", "openacc/zalazone/mazda_3": "S"}
    assert last_report()["files"] == 1  # the side files are not runs


def test_single_vehicle_campaign_is_skipped_and_max_groups_counts_files(tmp_path):
    single = tmp_path / "One_vehicle_multiple_drivers_on_road_campaign" / "D1_part1.csv"
    write_run(single, [], {"Time": np.arange(5) * 0.1, "Speed": np.zeros(5), "Engine Speed": np.zeros(5),
                           "Gear": np.zeros(5)})
    for name in ("part1.csv", "part2.csv", "part3.csv"):
        write_run(tmp_path / "Casale" / name, ASTAZERO_META, platoon())
    assert len(list(load_pairs(tmp_path))) == 6
    files = [p.meta["source_file"] for p in load_pairs(tmp_path, max_groups=2)]
    assert files == ["Casale/part1.csv"] * 2 + ["Casale/part2.csv"] * 2
    with pytest.raises(FileNotFoundError, match="<raw_dir>/<campaign>"):
        list(load_pairs(tmp_path, campaigns=["Casale", "Cherasco"]))
    with pytest.raises(ValueError, match="unknown OpenACC campaigns"):
        list(load_pairs(tmp_path, campaigns=["Ispra"]))
    with pytest.raises(FileNotFoundError):
        list(load_pairs(tmp_path / "nowhere"))


@pytest.mark.data
@pytest.mark.skipif(not (REAL_DIR / "Casale" / "part10.csv").is_file(), reason="OpenACC files not on disk")
def test_real_casale_run():
    pairs = list(load_pairs(REAL_DIR, campaigns=["Casale"], max_groups=1))
    assert [p.follower_id for p in pairs] == ["openacc/casale/hyundai"]
    assert pairs[0].leader_id[0] == "openacc/casale/rexton" and pairs[0].meta["acc_on_source"] == "driver column"
    events = extract(pairs, ExtractionConfig(max_duration=60.0))
    assert events and all(ev.meta["acc_on"] is True for ev in events)
