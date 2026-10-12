"""Analyses of milestone 4 (follownet2/experiments/analysis.py) and scripts/m4_report.py on a
fabricated results tree: two datasets (``alpha`` with every protocol, ``beta`` for the transfer),
models whose errors scale with a quality factor ``q`` (lower is better), the paired arms of E2
with their split file and event index, a second seed, an event split and transfer files."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from follownet2.experiments import analysis as an
from follownet2.stats import bootstrap_groups, holm

REPO = Path(__file__).resolve().parents[1]
#: model -> (family, quality); twin is an exact copy of persistence
MODELS = {"persistence": ("baseline", 1.0), "twin": ("baseline", 1.0), "mlp": ("network", 2.0),
          "lstm": ("network", 3.0), "idm": ("mechanistic", 4.0)}
ALPHA = [f"d{i:02d}" for i in range(12)]  # fold 0 tests d00-d05, fold 1 d06-d11
BETA = [f"b{i:02d}" for i in range(8)]
AFFECTED = {0: {"d00", "d01", "d02"}, 1: {"d06", "d07"}}
PROFILES = ["P1", "P2", "P3", "P4", "pulse"]  # P4 out of range, the pulse without recorded followers
SPEEDS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_BOOT = 200


# ----------------------------------------------------------------------------- fabrication


def pooled(table: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(table[column].sum() / table["n"].sum()))


def base_events(drivers, seed: int, per_driver: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"event_id": f"{d}|L|{k}", "follower_id": d, "n": int(rng.integers(150, 400)), "g": float(rng.gamma(2.0, 1.0))}
            for d in drivers for k in range(per_driver)]
    return pd.DataFrame(rows)


def model_tables(base: pd.DataFrame, q) -> dict[str, pd.DataFrame]:
    """Per-event and per-window tables of a model of quality ``q`` (scalar or one value per event)."""
    q = np.broadcast_to(np.asarray(q, dtype=np.float64), (len(base),))
    n, g, ids = base["n"].to_numpy(np.float64), base["g"].to_numpy(), base[["event_id", "follower_id"]]
    p0 = ids.assign(n=n, sse_a=q * g * n, sse_v=0.01 * q * g * n, sse_s=1e-4 * q * g * n, n_invalid=0.0)
    whole = ids.assign(n=base["n"].to_numpy(), sse_s=4.0 * q * g * n, sse_v=0.25 * q * g * n, collided=False, first_collision=-1)
    windows = []
    for (event, follower), qe, ge, ne in zip(ids.itertuples(index=False), q, g, base["n"]):
        for j, start in enumerate(range(30, ne - 100, 100)):
            windows.append({"event_id": event, "follower_id": follower, "start": start, "n": 100,
                            "sse_s": 100 * qe * ge * (1 + 0.1 * j), "sse_v": 10 * qe * ge * (1 + 0.1 * j),
                            "collided": False, "first_collision": -1})
    h10 = pd.DataFrame(windows)
    h30 = h10[h10["start"] == 30].assign(n=300, sse_s=lambda t: 3 * t["sse_s"], sse_v=lambda t: 3 * t["sse_v"])
    return {"p0": p0, "whole": whole, "h10": h10, "h30": h30.reset_index(drop=True)}


def block(table: pd.DataFrame, windows: bool = True) -> dict:
    out = {"rmse_s": pooled(table, "sse_s"), "rmse_v": pooled(table, "sse_v"), "collision_rate": float(table["collided"].mean()),
           "n_events": int(table["event_id"].nunique()), "n_samples": int(table["n"].sum())}
    if windows:
        out["n_windows"] = len(table)
    return out


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def protocol(directory: Path, name: str, summary: dict, provenance: str, tables: dict | None = None) -> None:
    write_json(directory / f"{name}.json", {"protocol": name, "model": {"provenance": {"config_hash": provenance}}, "summary": summary})
    for key, table in (tables or {}).items():
        table.to_parquet(directory / f"{name}.{key}.parquet", index=False)


def write_training(directory: Path, label: str, family: str, tables: dict, config_hash: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    test = {"p0": {"rmse_a": pooled(tables["p0"], "sse_a"), "rmse_s": pooled(tables["p0"], "sse_s"), "rmse_v": pooled(tables["p0"], "sse_v")},
            "closed_loop_whole_event": block(tables["whole"], windows=False)}
    write_json(directory / "train.json", {"model": label, "family": family, "config_hash": config_hash, "test": test})
    tables["p0"].to_parquet(directory / "test_p0.events.parquet", index=False)
    tables["whole"].to_parquet(directory / "test_closed_loop.events.parquet", index=False)


def write_p1(directory: Path, tables: dict, provenance: str, name: str = "p1") -> None:
    summary = {"h10": block(tables["h10"]), "h30": block(tables["h30"]), "whole": block(tables["whole"], windows=False)}
    protocol(directory, name, summary, provenance, {"h10": tables["h10"], "h30": tables["h30"], "whole": tables["whole"]})


def write_p2_p4(directory: Path, label: str, q: float, provenance: dict) -> None:
    rows = []
    for i, profile in enumerate(PROFILES):
        c = float(i + 1)
        concave = np.nan if profile == "pulse" else c * (6.0 - q)  # reversed against the P0 ranking
        hysteresis = np.nan if profile == "pulse" else 10.0 * c * q
        for n_vehicles in (20, 100):
            collisions = [1, 0, 2, 5, 3][i] if label == "lstm" and n_vehicles == 100 else 0
            rows.append({"profile": profile, "source": "synthetic" if profile == "pulse" else "openacc", "n_vehicles": n_vehicles,
                         "concave_growth_error": concave, "hysteresis_error": hysteresis, "collisions": collisions,
                         "in_range": profile != "P4"})
    profiles = pd.DataFrame(rows)
    part = profiles[profiles["in_range"] & (profiles["n_vehicles"] == 100)]
    summary = {"concave_growth_error_in_range": float(part["concave_growth_error"].mean()),
               "hysteresis_error_in_range": float(part["hysteresis_error"].mean()),
               "collisions_100_in_range": int(part["collisions"].sum())}
    protocol(directory, "p2", summary, provenance["p2"], {"profiles": profiles})
    tracks = label not in ("persistence", "twin")
    unstable = {"idm": [0] * 6, "mlp": [1, 1, 0, 0, 0, 0], "lstm": [1] * 6}.get(label, [0] * 6)
    speeds = pd.DataFrame({"v_e": SPEEDS, "converged": True, "tracks": tracks, "unstable_numerical": [bool(u) for u in unstable]})
    protocol(directory, "p3", {"unstable_fraction": float(np.mean(unstable)) if tracks else None}, provenance["p3"], {"speeds": speeds})
    collided = {"lstm": [1, 0, 1, 1], "mlp": [0, 0, 1, 0]}.get(label, [0, 0, 0, 0])
    scenarios = pd.DataFrame({"event_id": ["s0", "s1", "s2", "s3"], "v0": [15.0, 20.0, 25.0, 35.0], "in_range": [True, True, True, False],
                              "collided": [bool(c) for c in collided]})
    followers = ["e0", "e0", "e1", "e1", "e2", "e2", "e3", "e4", "e5"]
    hits = {"lstm": [1, 1, 0, 1, 0, 0, 1, 0, 1], "mlp": [0, 1, 0, 0, 0, 0, 1, 0, 0], "idm": [0, 0, 0, 0, 0, 0, 0, 0, 1]}.get(label, [1, 0, 0, 0, 0, 0, 0, 0, 0])
    episodes = pd.DataFrame({"event_id": [f"{f}|nm{i}" for i, f in enumerate(followers)], "follower_id": followers,
                             "collided": [bool(h) for h in hits]})
    summary = {"collision_rate_synthetic_in_range": float(scenarios.loc[scenarios["in_range"], "collided"].mean()),
               "collision_rate_empirical": float(episodes["collided"].mean())}
    protocol(directory, "p4", summary, provenance["p4"], {"scenarios": scenarios, "episodes": episodes})


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    top = tmp_path_factory.mktemp("m4")
    root = top / "models"
    alpha, beta = base_events(ALPHA, 1), base_events(BETA, 2)
    for label, (family, q) in MODELS.items():
        d = an.run_dir(root, "alpha", label)
        tables = model_tables(alpha, q)
        write_training(d, label, family, tables, f"h-{label}")
        write_p1(d, tables, f"h-{label}")
        # lstm: P2 computed from an earlier checkpoint (stale)
        write_p2_p4(d, label, q, {"p2": "old" if label == "lstm" else f"h-{label}", "p3": f"h-{label}", "p4": f"h-{label}"})
    # beta and the transfer files (mlp and persistence)
    for label in ("mlp", "persistence"):
        family, q = MODELS[label]
        d = an.run_dir(root, "beta", label)
        own = model_tables(beta, q)
        write_training(d, label, family, own, f"hb-{label}")
        write_p1(d, own, f"hb-{label}")
        foreign_q = 1.5 * q if label == "mlp" else q  # persistence is not trained: identical on beta
        write_p1(an.run_dir(root, "alpha", label), model_tables(beta, foreign_q), f"h-{label}", name="p1_beta")
        on_alpha = model_tables(alpha, 1.2 * q if label == "mlp" else q)
        summary = {"h10": block(on_alpha["h10"]), "h30": block(on_alpha["h30"]), "whole": block(on_alpha["whole"], windows=False)}
        protocol(d, "p1_alpha", summary, f"hb-{label}")  # milestone-3 style: no per-window tables
    # a second training seed of mlp
    write_training(an.run_dir(root, "alpha", "mlp", seed=1), "mlp", "network", model_tables(alpha, 2.2), "h-mlp-1")
    # E2: paired arms of folds 0 and 1; the leak halves the error of idm on the affected drivers, mlp is unaffected
    splits, events = top / "splits", top / "events"
    explicit, index = {}, [*alpha[["event_id", "follower_id"]].itertuples(index=False)]
    for fold in (0, 1):
        drivers = ALPHA[6 * fold: 6 * fold + 6]
        probe = alpha[alpha["follower_id"].isin(drivers)].reset_index(drop=True)
        leak_ids = [f"{d}|L|9" for d in sorted(AFFECTED[fold])]
        train_ids = [f"t{fold}{i}|L|0" for i in range(4)] + leak_ids
        index += [(e, e.split("|")[0]) for e in train_ids]
        explicit[str(fold)] = {"train": train_ids, "val": [], "test": list(probe["event_id"])}
        hit = probe["follower_id"].isin(AFFECTED[fold]).to_numpy()
        for label in ("idm", "mlp", "persistence"):
            family, q = MODELS[label]
            clean = model_tables(probe, q)
            leaky = model_tables(probe, np.where(hit, 0.5 * q, q)) if label == "idm" else clean
            for arm, tables in ((an.CLEAN, clean), (an.LEAKY, leaky)):
                d = an.run_dir(root, "alpha", label, arm, fold)
                write_training(d, label, family, tables, f"h-{label}-{arm}-{fold}")
                write_p1(d, tables, f"h-{label}-{arm}-{fold}")
    write_json(splits / f"alpha_{an.LEAKY}.json", {"dataset": "alpha", "kind": an.LEAKY, "explicit": explicit})
    (events / "alpha" / an.VERSION).mkdir(parents=True)
    pd.DataFrame(index, columns=["event_id", "follower_id"]).to_parquet(events / "alpha" / an.VERSION / "events.parquet", index=False)
    # E2 unpaired: event split of idm and mlp
    for label in ("idm", "mlp"):
        family, q = MODELS[label]
        d = an.run_dir(root, "alpha", label, "event", 0)
        tables = model_tables(alpha, 0.8 * q)
        write_training(d, label, family, tables, f"h-{label}-event")
        write_p1(d, tables, f"h-{label}-event")
    return {"top": top, "root": root, "splits": splits, "events": events, "alpha": alpha, "beta": beta}


# ----------------------------------------------------------------------------- helpers and E1


def test_tables_are_checked_against_their_summary(world, tmp_path):
    d = an.run_dir(world["root"], "alpha", "mlp")
    table, problem = an.own_p1_table(d, "h10")
    assert problem is None and len(table) > 0
    # milestone 3: the transfer evaluation overwrote the own tables with the windows of another dataset
    copy = tmp_path / "run"
    copy.mkdir()
    for name in ("p1.json", "train.json", "test_closed_loop.events.parquet"):
        (copy / name).write_bytes((d / name).read_bytes())
    foreign = model_tables(world["beta"], 2.0)
    for horizon in ("h10", "whole"):
        foreign[horizon].to_parquet(copy / f"p1.{horizon}.parquet", index=False)
    table, problem = an.own_p1_table(copy, "h10")
    assert table is None and "n_events" in problem
    whole, problem = an.own_p1_table(copy, "whole")  # falls back to the table of the training run
    assert problem is None and whole["event_id"].nunique() == world["alpha"]["event_id"].nunique()


def test_common_draw_equals_bootstrap_groups(world):
    table = pd.read_parquet(an.run_dir(world["root"], "alpha", "lstm") / "test_p0.events.parquet")
    universe, counts = an.common_draw([table], 150, 7)
    direct = bootstrap_groups(table, "follower_id", None, 150, 7, columns=["sse_a"])["rmse_a"].to_numpy()
    np.testing.assert_allclose(an.group_replicates(table, "rmse_a", universe, counts), direct, rtol=1e-12)


def test_e1_ratio_of_a_copy_of_persistence_is_one(world):
    e1 = an.e1_ratios(world["root"], "alpha", None, n_boot=N_BOOT, seed=3).set_index("model")
    twin = e1.loc["twin"]
    assert twin["ratio"] == pytest.approx(1.0) and twin["low"] <= 1.0 <= twin["high"]
    assert twin["low"] == pytest.approx(1.0) and twin["high"] == pytest.approx(1.0)  # common random numbers
    mlp = e1.loc["mlp"]
    assert mlp["ratio"] == pytest.approx(np.sqrt(2.0)) and mlp["low"] <= mlp["ratio"] <= mlp["high"]
    assert mlp["n_seeds"] == 2 and mlp["seed_min"] == pytest.approx(np.sqrt(2.0)) and mlp["seed_max"] == pytest.approx(np.sqrt(2.2))
    assert mlp["n_folds"] == 2 and mlp["folds_min"] == pytest.approx(np.sqrt(2.0)) and mlp["folds_max"] == pytest.approx(np.sqrt(2.0))
    assert e1.loc["lstm", "n_folds"] == 0 and np.isnan(e1.loc["lstm", "seed_min"])  # one seed, no paired runs
    assert e1.loc["idm", "family"] == "mechanistic"


# ----------------------------------------------------------------------------- E2


def test_e2_inflation_sign_affected_subset_and_holm(world):
    affected = an.affected_followers("alpha", world["splits"], world["events"])
    assert affected == {k: frozenset(v) for k, v in AFFECTED.items()}
    e2 = an.e2_paired(world["root"], "alpha", ("idm", "mlp"), n_boot=N_BOOT, seed=0, affected=affected)
    assert set(e2["subset"]) == {"all", "affected"} and set(e2["metric"]) == set(an.E2_METRICS)
    idm = e2[(e2["model"] == "idm")].set_index(["subset", "metric"])
    for metric in an.E2_METRICS:
        assert idm.loc[("all", metric), "inflation"] > 0 and idm.loc[("all", metric), "mean_difference"] > 0
        assert idm.loc[("affected", metric), "inflation"] > idm.loc[("all", metric), "inflation"]
        assert idm.loc[("affected", metric), "low"] > 0
    # sse halved on every affected event: the error falls by 1 - sqrt(1/2) on the affected subset
    assert idm.loc[("affected", "p0_rmse_a"), "inflation"] == pytest.approx(1 - np.sqrt(0.5))
    assert (idm["n_folds"] == 2).all() and (idm["n_affected_drivers"] == 5).all()
    assert idm.loc[("affected", "p0_rmse_a"), "n_drivers"] == 5 and idm.loc[("all", "p0_rmse_a"), "n_drivers"] == 12
    mlp = e2[(e2["model"] == "mlp")]
    assert (mlp["inflation"] == 0).all() and (mlp["p_bootstrap"] == 1).all()
    for _, part in e2.groupby(["subset", "metric"]):
        np.testing.assert_allclose(part["p_holm"], holm(part["p_bootstrap"].to_numpy()))
        assert (part["p_holm"] >= part["p_bootstrap"] - 1e-15).all()
    means = an.e2_class_means(e2).set_index(["subset", "metric", "family"])
    assert means.loc[("all", "p0_rmse_a", "mechanistic"), "mean_inflation"] > 0
    assert means.loc[("all", "p0_rmse_a", "network"), "mean_inflation"] == 0
    only_all = an.e2_paired(world["root"], "alpha", ("idm",), n_boot=50, seed=0, affected=None)
    assert set(only_all["subset"]) == {"all"}


def test_e2_unpaired(world):
    table = an.e2_unpaired(world["root"], "alpha", ("idm", "mlp", "gipps")).set_index(["model", "metric"])
    row = table.loc[("idm", "p0_rmse_a")]
    assert row["driver"] == pytest.approx(np.sqrt(4.0) * row["event"] / np.sqrt(3.2))
    assert row["inflation"] == pytest.approx(1 - np.sqrt(0.8)) and row["difference"] > 0
    assert set(table.index.get_level_values("model")) == {"idm", "mlp"}  # gipps has no runs


# ----------------------------------------------------------------------------- E3


def test_e3_rankings_and_stale_files(world):
    rankings = an.e3_rankings(world["root"], "alpha").set_index("model")
    assert rankings.loc["lstm", "stale"] == "p2" and rankings.loc["mlp", "stale"] == ""
    assert np.isnan(rankings.loc["persistence", "p3_unstable_fraction"])  # no equilibrium
    assert rankings.loc["mlp", "p3_unstable_fraction"] == pytest.approx(1 / 3)
    assert rankings.loc["lstm", "p2_collisions_100"] == 6  # P1, P2, P3 and the pulse, not P4
    assert rankings["composite"].notna().all()


def test_e3_kendall_identical_and_reversed_rankings(world):
    kendall = an.e3_kendall(world["root"], "alpha", n_boot=N_BOOT, seed=1).set_index(["ranking_a", "ranking_b"])
    same = kendall.loc[("P0", "P1 whole")]  # both scale with q in every replicate
    assert same["tau"] == pytest.approx(1.0) and same["low"] == pytest.approx(1.0) and same["high"] == pytest.approx(1.0)
    assert kendall.loc[("P0", "P1 h10"), "tau"] == pytest.approx(1.0) and kendall.loc[("P0", "P1 h10"), "note"] == ""
    reversed_ = kendall.loc[("P0", "P2 concave")]  # concave error falls with q
    assert reversed_["tau"] == pytest.approx(-1.0) and reversed_["high"] == pytest.approx(-1.0)
    assert (kendall["n_models"] == len(MODELS)).all()
    speed = an.e3_spacing_vs_speed(world["root"], "alpha", n_boot=N_BOOT, seed=1)
    assert list(speed["horizon"]) == ["h10", "h30", "whole"] and np.allclose(speed["tau"], 1.0) and np.allclose(speed["low"], 1.0)


def test_e3_replicates_reproduce_the_points(world):
    rankings = an.e3_rankings(world["root"], "alpha")
    samples, reasons = an.e3_replicates(world["root"], "alpha", rankings, n_boot=N_BOOT, seed=2)
    assert not reasons
    # persistence and twin have no P3 equilibrium: NaN in every replicate (ranked last)
    assert np.isnan(samples["p3_unstable_fraction"]["persistence"]).all()
    point = rankings.set_index("model")
    for column in ("p0_rmse_a", "p1_h10_rmse_s", "p2_concave_growth_error", "p4_collision_rate_empirical"):
        for model in MODELS:
            x = samples[column][model]
            assert np.nanmin(x) <= point.loc[model, column] <= np.nanmax(x)


def test_e3_top3_and_changes():
    rankings = pd.DataFrame({
        "dataset": "x", "model": ["a", "b", "c", "d", "e"], "family": "network",
        "p0_rmse_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "p1_whole_rmse_s": [5.0, 1.0, 2.0, 3.0, 4.0],
        "composite": [0.5, 0.9, 0.1, 0.2, 0.2],
    })
    top = an.e3_top3(rankings).set_index("ranking")
    assert list(top.loc["P0", ["top1", "top2", "top3"]]) == ["a", "b", "c"]
    assert list(top.loc["P1 whole", ["top1", "top2", "top3"]]) == ["b", "c", "d"]
    assert list(top.loc["composite", ["top1", "top2", "top3"]]) == ["c", "d", "e"] and top.loc["composite", "tied"] == ""
    changes = an.e3_top3_changes(rankings)
    assert changes == [
        "x: P0 -> P1 whole: enter d; leave a",
        "x: P1 whole -> composite: enter e; leave b",
        "x: P0 -> composite: enter d, e; leave a, b",
    ]
    tied = rankings.assign(composite=[0.1, 0.2, 0.3, 0.3, 0.9])
    assert an.e3_top3(tied).set_index("ranking").loc["composite", "tied"] == "d"


# ----------------------------------------------------------------------------- E4


def test_e4_diagonal_is_one_and_intervals(world):
    e4 = an.e4_transfer(world["root"], ["mlp", "persistence"], n_boot=N_BOOT, seed=0)
    diagonal = e4[e4["trained_on"] == e4["tested_on"]]
    assert len(diagonal) == 8 and np.allclose(diagonal["ratio"], 1.0)
    assert np.allclose(diagonal["low"], 1.0) and np.allclose(diagonal["high"], 1.0)
    rows = e4.set_index(["model", "trained_on", "tested_on", "horizon"])
    ab = rows.loc[("mlp", "alpha", "beta", "whole")]
    assert ab["ratio"] == pytest.approx(np.sqrt(1.5)) and ab["low"] <= ab["ratio"] <= ab["high"] and ab["note"] == ""
    assert ab["low"] == pytest.approx(np.sqrt(1.5)) and ab["n_drivers"] == len(BETA)  # same base events: exact ratio
    ba = rows.loc[("mlp", "beta", "alpha", "whole")]  # milestone-3 style transfer file: point value only
    assert ba["ratio"] == pytest.approx(np.sqrt(1.2)) and ba["rmse_s_own"] == rows.loc[("mlp", "alpha", "alpha", "whole"), "rmse_s"]
    assert np.isnan(ba["low"]) and "p1_alpha.whole.parquet" in ba["note"]
    assert rows.loc[("persistence", "alpha", "beta", "h10"), "ratio"] == pytest.approx(1.0)
    summary = an.e4_summary(e4).set_index(["model", "horizon"])
    assert summary.loc[("mlp", "whole"), "mean_degradation"] == pytest.approx((np.sqrt(1.5) + np.sqrt(1.2)) / 2)
    assert summary.loc[("mlp", "whole"), "worst_target"] == "beta" and summary.loc[("mlp", "whole"), "n_pairs"] == 2
    targets = an.e4_targets(e4).set_index(["tested_on", "horizon"])
    assert targets.loc[("beta", "whole"), "mean_degradation"] == pytest.approx((np.sqrt(1.5) + 1.0) / 2)


# ----------------------------------------------------------------------------- report


def test_report_runs_end_to_end(world):
    spec = importlib.util.spec_from_file_location("m4_report", REPO / "scripts" / "m4_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = world["top"] / "m4"
    path = module.main([f"root={world['root']}", f"out={out}", "n_boot=40", f"splits={world['splits']}", f"events={world['events']}"])
    text = path.read_text(encoding="utf-8")
    for heading in ("## Campaigns", "## E1", "## E2", "## E3", "## E4", "### alpha"):
        assert heading in text
    assert "not run" in text and "Not available for ngsim, follownet_highd" in text
    for name in ("e1_ratios", "e2_paired", "e2_unpaired", "e3_rankings", "e3_kendall", "e4_transfer"):
        assert (out / f"{name}.csv").exists()
    e2 = pd.read_csv(out / "e2_paired.csv")
    assert set(e2["subset"]) == {"all", "affected"} and set(e2["model"]) == {"idm", "mlp"}
    kendall = pd.read_csv(out / "e3_kendall.csv")
    assert "P1 whole rmse_s" in set(kendall["ranking_a"])
    assert len(pd.read_csv(out / "e3_rankings.csv")) == len(MODELS)
