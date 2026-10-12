"""Leaderboard of milestone 5 (follownet2/leaderboard/, scripts/make_leaderboard.py) on a fabricated
results root: dataset ``alpha`` with every protocol for two admitted models (``persistence``,
``idm``) and one refused model (``mlp``), a second training seed, the clean arms of two paired
folds, an ``event``-split run and transfer files; dataset ``beta`` with P1 only and no 30 s window."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from follownet2.experiments.analysis import e4_transfer, run_dir
from follownet2.leaderboard import DRIVER_LEAK, entry_label, validate_entry
from follownet2.leaderboard.render import dataset_markdown, html_page, leaderboard_markdown
from follownet2.leaderboard.stability import rank_stability_figure
from follownet2.leaderboard.tables import (
    GROUPS,
    MAIN_COLUMNS,
    build_leaderboard,
    composite_replicates,
    transfer_means,
    transfer_table,
)
from follownet2.stats import bootstrap_groups, percentile_ci

REPO = Path(__file__).resolve().parents[1]
N_BOOT = 100
#: model -> (family, quality of the P0/P1 tables, quality of P2); lower is better
MODELS = {"persistence": ("baseline", 1.0, 1.0), "idm": ("mechanistic", 1.6, 0.5), "mlp": ("network", 1.2, 0.8)}
ALPHA = [f"a{i:02d}" for i in range(10)]
BETA = [f"b{i:02d}" for i in range(6)]
PROFILES = ["P1", "P2", "P3", "pulse"]  # P3 outside the speed range
COLLISIONS = {"persistence": [1, 0, 2, 0], "idm": [0, 0, 0, 0], "mlp": [3, 1, 0, 0]}
UNSTABLE = {"persistence": [0, 0, 0, 0], "idm": [0, 0, 1, 0], "mlp": [1, 1, 1, 0]}


# ----------------------------------------------------------------------------- fabrication


def pooled(table: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(table[column].sum() / table["n"].sum()))


def base_events(drivers, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"event_id": f"{d}|L|{k}", "follower_id": d, "n": int(rng.integers(250, 450)), "g": float(rng.gamma(2.0, 1.0))}
                         for d in drivers for k in range(2)])


def model_tables(base: pd.DataFrame, q: float, windows30: bool = True) -> dict[str, pd.DataFrame]:
    n, g, ids = base["n"].to_numpy(np.float64), base["g"].to_numpy(), base[["event_id", "follower_id"]]
    p0 = ids.assign(n=n, sse_a=q * g * n, sse_s=1e-4 * q * g * n, sse_v=1e-2 * q * g * n)
    whole = ids.assign(n=base["n"].to_numpy(), sse_s=4.0 * q * g * n, sse_v=0.25 * q * g * n, collided=False)
    rows = []
    for (event, follower), ge, ne in zip(ids.itertuples(index=False), g, base["n"]):
        for j, start in enumerate(range(30, ne - 100, 100)):
            rows.append({"event_id": event, "follower_id": follower, "start": start, "n": 100, "sse_s": 100 * q * ge * (1 + 0.1 * j),
                         "sse_v": 10 * q * ge * (1 + 0.1 * j), "collided": False})
    h10 = pd.DataFrame(rows)
    h30 = h10[h10["start"] == 30].assign(n=300, sse_s=lambda t: 3 * t["sse_s"], sse_v=lambda t: 3 * t["sse_v"])
    return {"p0": p0, "whole": whole, "h10": h10, "h30": (h30 if windows30 else h10.iloc[0:0]).reset_index(drop=True)}


def block(table: pd.DataFrame, windows: bool = True) -> dict:
    if not len(table):
        return {"rmse_s": float("nan"), "rmse_v": float("nan"), "collision_rate": float("nan"), "n_events": 0, "n_samples": 0, "n_windows": 0}
    out = {"rmse_s": pooled(table, "sse_s"), "rmse_v": pooled(table, "sse_v"), "collision_rate": float(table["collided"].mean()),
           "n_events": int(table["event_id"].nunique()), "n_samples": int(table["n"].sum())}
    if windows:
        out["n_windows"] = len(table)
    return out


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def edit_json(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def checklist(name: str, split: str, mechanistic: bool) -> dict:
    out = {"data_version": "reconstructed", "reconstruction_method": "qp-semi-implicit-v1", "split_type": split,
           "mechanistic_calibration_protocol": "closed loop, NRMSE(s)+NRMSE(v), differential evolution" if mechanistic else "not a mechanistic model",
           "seeds": {"global": 0, "train": 0, "split": 0}}
    kind = name.split("_")[0]
    if kind in ("p1", "p2", "p4"):
        out["closed_loop_horizon"] = "10 s, 30 s, whole event; warm-up 3 s"
    if kind == "p3":
        out["stability_test"] = "step-response gain on 0.002-2 Hz, delta 0.05 m/s, v_e 5..30 m/s"
    if kind == "p4":
        out["rare_event_suite"] = {"synthetic": "p4-v1", "episodes": ["tgsim_nearmiss/reconstructed"]}
    return out


def protocol(directory: Path, name: str, model_hash: str, split: str, mechanistic: bool, summary: dict, tables: dict | None = None) -> None:
    write_json(directory / f"{name}.json", {
        "protocol": name.split("_")[0], "config_hash": f"cfg-{name}", "git_revision": "rev-eval", "data_version": "reconstructed",
        "model": {"label": directory.parent.name, "provenance": {"config_hash": model_hash}},
        "checklist": checklist(name, split, mechanistic), "summary": summary})
    for key, table in (tables or {}).items():
        table.to_parquet(directory / f"{name}.{key}.parquet", index=False)


def p1_summary(t: dict) -> dict:
    return {"h10": block(t["h10"]), "h30": block(t["h30"]), "whole": block(t["whole"], windows=False)}


def write_run(root: Path, dataset: str, label: str, q: float, base: pd.DataFrame, split: str = "driver", fold: int = 0, seed: int = 0,
              protocols=("p1", "p2", "p3", "p4"), windows30: bool = True, q2: float | None = None) -> Path:
    family, _, default_q2 = MODELS[label]
    q2 = default_q2 if q2 is None else q2
    d = run_dir(root, dataset, label, split, fold, seed)
    t = model_tables(base, q, windows30)
    model_hash = f"h-{dataset}-{label}-{split}{fold}-s{seed}"
    write_json(d / "train.json", {
        "model": label, "family": family, "config_hash": model_hash, "git_revision": "rev-train", "data_version": "reconstructed",
        "split": {"kind": split, "fold": fold, "seed": 0},
        "test": {"p0": {"rmse_a": pooled(t["p0"], "sse_a"), "rmse_s": pooled(t["p0"], "sse_s"), "rmse_v": pooled(t["p0"], "sse_v")},
                 "closed_loop_whole_event": block(t["whole"], windows=False)}})
    t["p0"].to_parquet(d / "test_p0.events.parquet", index=False)
    t["whole"].to_parquet(d / "test_closed_loop.events.parquet", index=False)
    mech = family == "mechanistic"
    if "p1" in protocols:
        protocol(d, "p1", model_hash, split, mech, p1_summary(t), {h: t[h] for h in ("h10", "h30", "whole")})
    if "p2" in protocols:
        rows = []
        for i, profile in enumerate(PROFILES):
            for n_vehicles in (20, 100):
                rows.append({"profile": profile, "source": "synthetic" if profile == "pulse" else "openacc", "n_vehicles": n_vehicles,
                             "concave_growth_error": np.nan if profile == "pulse" else (i + 1) * q2,
                             "hysteresis_error": np.nan if profile == "pulse" else 10.0 * (i + 1) * q2,
                             "collisions": COLLISIONS[label][i] if n_vehicles == 100 else 0, "in_range": profile != "P3"})
        profiles = pd.DataFrame(rows)
        part = profiles[profiles["in_range"] & (profiles["n_vehicles"] == 100)]
        protocol(d, "p2", model_hash, split, mech, {"concave_growth_error_in_range": float(part["concave_growth_error"].mean()),
                                                    "hysteresis_error_in_range": float(part["hysteresis_error"].mean()),
                                                    "collisions_100_in_range": int(part["collisions"].sum())}, {"profiles": profiles})
    if "p3" in protocols:
        tracks = label != "persistence"
        unstable = UNSTABLE[label]
        speeds = pd.DataFrame({"v_e": [5.0, 10.0, 15.0, 20.0], "converged": True, "tracks": tracks,
                               "unstable_numerical": [bool(u) for u in unstable], "unstable_large": True})
        protocol(d, "p3", model_hash, split, mech, {"unstable_fraction": float(np.mean(unstable)) if tracks else None,
                                                    "unstable_fraction_large": 1.0 if tracks else None}, {"speeds": speeds})
    if "p4" in protocols:
        protocol(d, "p4", model_hash, split, mech, {"collision_rate_synthetic_in_range": 0.5 * q2, "collision_rate_empirical": 0.1 * q})
    return d


def write_transfer(root: Path, source: str, label: str, target: str, base: pd.DataFrame, q: float, windows30: bool) -> None:
    d = run_dir(root, source, label)
    t = model_tables(base, q, windows30)
    model_hash = json.loads((d / "train.json").read_text(encoding="utf-8"))["config_hash"]
    protocol(d, f"p1_{target}", model_hash, "driver", MODELS[label][0] == "mechanistic", p1_summary(t), {h: t[h] for h in ("h10", "h30", "whole")})


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("leaderboard") / "models"
    alpha, beta = base_events(ALPHA, 1), base_events(BETA, 2)
    for label, (_, q, _) in MODELS.items():
        write_run(root, "alpha", label, q, alpha)
    # refused: the P4 block lacks the closed-loop horizon
    edit_json(run_dir(root, "alpha", "mlp") / "p4.json", lambda data: data["checklist"].pop("closed_loop_horizon"))
    # second training seed; idm's seed 1 has worse P2 errors than persistence (the composite rank varies with the seed)
    for label, q, q2 in (("persistence", 1.0, None), ("idm", 1.7, 2.0)):
        write_run(root, "alpha", label, q, alpha, seed=1, q2=q2)
    for fold, drivers, q_idm in ((0, ALPHA[:5], 1.6), (1, ALPHA[5:], 1.5)):
        probe = alpha[alpha["follower_id"].isin(drivers)].reset_index(drop=True)
        write_run(root, "alpha", "persistence", 1.0, probe, split="paired-clean", fold=fold, protocols=("p1",))
        write_run(root, "alpha", "idm", q_idm, probe, split="paired-clean", fold=fold, protocols=("p1",))
    write_run(root, "alpha", "idm", 1.4, alpha, split="event", protocols=("p1",))
    for label, q in (("persistence", 1.0), ("idm", 1.3)):
        write_run(root, "beta", label, q, beta, protocols=("p1",), windows30=False)
    write_transfer(root, "alpha", "persistence", "beta", beta, 1.0, windows30=False)
    write_transfer(root, "alpha", "idm", "beta", beta, 1.3 * 1.25, windows30=False)
    write_transfer(root, "beta", "persistence", "alpha", alpha, 1.0, windows30=True)
    write_transfer(root, "beta", "idm", "alpha", alpha, 1.6 * 1.44, windows30=True)
    boards = build_leaderboard(root, n_boot=N_BOOT, seed=0, date="2026-01-01")
    return {"root": root, "alpha": alpha, "beta": beta, "boards": boards}


def copy_run(world, tmp_path, label: str = "idm") -> Path:
    """A copy of the alpha run of ``label`` under the same layout."""
    target = run_dir(tmp_path / "models", "alpha", label)
    shutil.copytree(run_dir(world["root"], "alpha", label), target)
    return target


# ----------------------------------------------------------------------------- checklist


def test_complete_checklist_is_admitted(world):
    root = world["root"]
    for directory in (run_dir(root, "alpha", "idm"), run_dir(root, "alpha", "persistence"), run_dir(root, "beta", "idm"),
                      run_dir(root, "alpha", "idm", "event"), run_dir(root, "alpha", "idm", seed=1)):
        assert validate_entry(directory) == [], directory


def test_empty_stability_test_is_refused(world, tmp_path):
    d = copy_run(world, tmp_path)
    edit_json(d / "p3.json", lambda data: data["checklist"].update(stability_test=""))
    assert validate_entry(d) == ["p3.json: stability_test empty"]


def test_other_data_version_is_refused(world, tmp_path):
    d = copy_run(world, tmp_path)
    edit_json(d / "p2.json", lambda data: data["checklist"].update(data_version="raw"))
    assert validate_entry(d) == ["p2.json: data_version raw differs from the run's reconstructed"]


def test_every_rule_names_file_and_field(world, tmp_path):
    d = copy_run(world, tmp_path)
    edit_json(d / "train.json", lambda data: data.pop("git_revision"))
    edit_json(d / "p1.json", lambda data: data["model"]["provenance"].update(config_hash="h-old"))
    edit_json(d / "p1_beta.json", lambda data: data["checklist"].update(split_type="event", closed_loop_horizon=None))
    edit_json(d / "p4.json", lambda data: data["checklist"]["seeds"].update(split=None))
    edit_json(d / "p2.json", lambda data: data.pop("checklist"))
    problems = validate_entry(d)
    assert problems == [
        "train.json: git_revision missing",
        "p1.json: model.provenance.config_hash h-old differs from train.json's h-alpha-idm-driver0-s0",
        "p1_beta.json: closed_loop_horizon empty",
        "p1_beta.json: split_type event differs from the run's driver",
        "p2.json: no checklist block",
        "p4.json: seeds.split empty",
    ]
    # a run without any protocol file
    empty = run_dir(tmp_path / "other", "alpha", "idm")
    empty.mkdir(parents=True)
    shutil.copy(d / "train.json", empty / "train.json")
    edit_json(empty / "train.json", lambda data: data.update(git_revision="rev"))
    assert validate_entry(empty) == ["no protocol file"]


def test_event_split_is_labelled(world):
    root = world["root"]
    assert entry_label(run_dir(root, "alpha", "idm", "event")) == DRIVER_LEAK
    assert entry_label(run_dir(root, "alpha", "idm")) == ""
    board = world["boards"]["alpha"]
    assert list(board.leak["label"]) == [DRIVER_LEAK]
    assert list(board.leak["run"]) == ["event-fold0/idm/seed0"]
    assert board.leak["p1_whole_rmse_s_driver"].iloc[0] == pytest.approx(board.main.set_index("model").at["idm", "p1_whole_rmse_s"])
    assert sorted(board.main["model"]) == ["idm", "persistence"]  # the event run is not ranked
    assert board.counts() == {"admitted": 2, "refused": 1, "leak_admitted": 1, "leak_refused": 0, "runs": 10, "runs_refused": 1}


# ----------------------------------------------------------------------------- tables


def test_main_table_columns_and_order(world):
    board = world["boards"]["alpha"]
    main = board.main.set_index("model", drop=False)
    assert list(board.main.columns) == MAIN_COLUMNS
    # sorted by the composite (idm wins every input), although persistence has the lower P0 and P1 errors
    assert list(board.main["model"]) == ["idm", "persistence"]
    assert list(board.main["composite"]) == [0.0, 1.0]
    assert list(board.main["composite_rank"]) == [1.0, 2.0]
    assert (board.main["composite_rank_low"] <= board.main["composite_rank"]).all()
    assert (board.main["composite_rank"] <= board.main["composite_rank_high"]).all()
    # P0 ratio and P1 intervals: driver bootstrap with common random numbers (the draw of bootstrap_groups)
    d_idm, d_ref = run_dir(world["root"], "alpha", "idm"), run_dir(world["root"], "alpha", "persistence")
    p0 = {m: bootstrap_groups(pd.read_parquet(d / "test_p0.events.parquet"), n_boot=N_BOOT, seed=0)["rmse_a"].to_numpy()
          for m, d in (("idm", d_idm), ("persistence", d_ref))}
    assert main.at["idm", "p0_ratio"] == pytest.approx(np.sqrt(1.6))  # RMSE ~ sqrt(quality)
    assert main.at["persistence", "p0_ratio"] == pytest.approx(1.0)
    assert (main.at["idm", "p0_ratio_low"], main.at["idm", "p0_ratio_high"]) == pytest.approx(percentile_ci(p0["idm"] / p0["persistence"]))
    whole = bootstrap_groups(pd.read_parquet(d_idm / "test_closed_loop.events.parquet"), n_boot=N_BOOT, seed=0)["rmse_s"]
    assert (main.at["idm", "p1_whole_rmse_s_low"], main.at["idm", "p1_whole_rmse_s_high"]) == pytest.approx(percentile_ci(whole))
    for column in ("p1_h10_rmse_s", "p1_h30_rmse_s", "p1_whole_rmse_s"):
        assert ((board.main[f"{column}_low"] <= board.main[column]) & (board.main[column] <= board.main[f"{column}_high"])).all()
    # in-range P2/P4 values, P3 at 0.05 and 1 m/s (persistence has no equilibrium)
    assert main.at["idm", "p2_concave_growth_error"] == pytest.approx(0.5 * (1 + 2) / 2)
    assert main.at["persistence", "p2_collisions_100"] == 1
    assert main.at["idm", "p3_unstable_fraction"] == pytest.approx(0.25)
    assert main.at["idm", "p3_unstable_fraction_large"] == pytest.approx(1.0)
    assert np.isnan(main.at["persistence", "p3_unstable_fraction"])
    # spreads: seeds 0-1 of the driver split, clean arms of folds 0-1
    assert (main.at["idm", "p0_ratio_seed_min"], main.at["idm", "p0_ratio_seed_max"]) == pytest.approx(np.sqrt([1.6, 1.7]))
    assert main.at["idm", "n_seeds"] == 2
    assert (main.at["idm", "p0_ratio_fold_min"], main.at["idm", "p0_ratio_fold_max"]) == pytest.approx(np.sqrt([1.5, 1.6]))
    assert main.at["idm", "n_folds"] == 2
    # rank distributions for the plot
    assert list(board.composite_ranks.index) == ["idm", "persistence"]
    assert list(board.p1_ranks.index) == ["persistence", "idm"]


def test_composite_rank_interval_covers_training_seeds(world):
    root, board = world["root"], world["boards"]["alpha"]
    samples, notes, seeds_of = composite_replicates(root, "alpha", board.runs, ["idm", "persistence"], n_boot=200, seed=0)
    assert seeds_of == {"idm": [0, 1], "persistence": [0, 1]} and notes == []
    # seed 0 of idm wins every input (composite 0; 0.125 or 0.375 with ties when a replicate misses P1 or draws only the
    # pulse); seed 1 loses both P2 errors (0.5 or 0.625; 0.375 for a replicate of pulses only)
    assert (samples["idm"] == 0.0).any() and (samples["idm"] >= 0.5).any()
    assert set(np.unique(samples["idm"])) <= {0.0, 0.125, 0.375, 0.5, 0.625}
    main = board.main.set_index("model")
    assert main.at["idm", "composite"] == 0.0 and main.at["idm", "composite_rank"] == 1.0  # the point stays seed 0
    assert main.at["idm", "composite_rank_low"] == 1.0 and main.at["idm", "composite_rank_high"] >= 1.5
    assert any(note.startswith("Composite rank interval") and "training seeds" in note for note in board.notes)
    # a refused seed run does not enter: only seed 0 of idm remains
    runs = board.runs.copy()
    runs.loc[(runs["model"] == "idm") & (runs["seed"] == 1), "admitted"] = False
    samples, _, seeds_of = composite_replicates(root, "alpha", runs, ["idm", "persistence"], n_boot=200, seed=0)
    assert seeds_of["idm"] == [0] and (samples["idm"] < 0.5).all()


def test_refused_entry_is_listed_and_never_ranked(world):
    board = world["boards"]["alpha"]
    assert "mlp" not in set(board.main["model"])
    assert board.refused.to_dict("records") == [
        {"run": "driver-fold0/mlp/seed0", "model": "mlp", "role": "ranked entry", "label": "", "problems": "p4.json: closed_loop_horizon missing"}]
    assert "mlp" not in set(board.composite_ranks.index) | set(board.p1_ranks.index)


def test_dataset_with_p1_only(world):
    board = world["boards"]["beta"]
    assert list(board.main.columns) == [c for g in ("id", "p0", "p1_h10", "p1_whole") for c in GROUPS[g]]
    assert list(board.main["model"]) == ["persistence", "idm"]  # by the whole-event P1 error
    assert board.composite_ranks is None and board.p1_ranks is not None
    assert board.leak.empty and board.refused.empty


def test_transfer_table(world):
    root = world["root"]
    board = world["boards"]["alpha"]
    transfer = board.transfer.set_index("model")
    assert list(board.transfer["model"]) == ["idm", "persistence"] and set(board.transfer["tested_on"]) == {"beta"}
    whole = lambda d, name: json.loads((d / f"{name}.json").read_text(encoding="utf-8"))["summary"]["whole"]["rmse_s"]  # noqa: E731
    expected = whole(run_dir(root, "alpha", "idm"), "p1_beta") / whole(run_dir(root, "beta", "idm"), "p1")
    assert transfer.at["idm", "ratio"] == pytest.approx(expected)
    assert transfer.at["idm", "low"] <= transfer.at["idm", "ratio"] <= transfer.at["idm", "high"]
    assert transfer.at["persistence", "ratio"] == pytest.approx(1.0)
    means = transfer_means(board.transfer).set_index("model")
    assert means.at["idm", "mean_degradation"] == pytest.approx(expected) and means.at["idm", "n_targets"] == 1
    assert set(world["boards"]["beta"].transfer["tested_on"]) == {"alpha"}
    # without an admitted run on the target (the denominator) the ratio is left out
    e4 = e4_transfer(root, None, 20, 0, horizons=("whole",))
    dropped = transfer_table(e4, "alpha", board.main, admitted={("alpha", "idm"), ("alpha", "persistence")})
    assert len(dropped) == 2 and dropped["ratio"].isna().all() and dropped["note"].str.startswith("no admitted run").all()
    assert transfer_means(dropped)["n_targets"].tolist() == [0, 0]


# ----------------------------------------------------------------------------- pages and plot


def test_rank_stability_figure(world):
    boards = world["boards"]
    alpha = boards["alpha"]
    families = dict(zip(alpha.main["model"], alpha.main["family"]))
    figure = rank_stability_figure("alpha", alpha.composite_ranks, alpha.p1_ranks, families, N_BOOT)
    assert len(figure.axes) == 2
    assert [t.get_text() for t in figure.axes[0].get_yticklabels()] == ["idm", "persistence"]
    assert f"{N_BOOT} bootstrap replicates" in figure.get_suptitle() and "alpha" in figure.get_suptitle()
    beta = boards["beta"]
    assert len(rank_stability_figure("beta", beta.composite_ranks, beta.p1_ranks, {}, N_BOOT).axes) == 1


def test_markdown_sections(world):
    text = dataset_markdown(world["boards"]["alpha"], image="rank_stability_alpha.png")
    for heading in ("## alpha", "### Main table", "### Driver leak (event split)", "### Transfer", "### Refused entries", "### Provenance"):
        assert heading in text
    assert "p4.json: closed_loop_horizon missing" in text and DRIVER_LEAK in text
    assert "1.265 [" in text  # the P0 ratio of idm, sqrt(1.6), with its interval
    page = leaderboard_markdown(world["boards"], {"alpha": "rank_stability_alpha.png"})
    assert page.index("## alpha") < page.index("## beta")


def test_pages_are_written(world, tmp_path):
    spec = importlib.util.spec_from_file_location("make_leaderboard", REPO / "scripts" / "make_leaderboard.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = tmp_path / "leaderboard"
    boards = script.main([f"root={world['root']}", f"out={out}", "n_boot=50", "date=2026-01-01"])
    assert set(boards) == {"alpha", "beta"}
    for name in ("LEADERBOARD.md", "alpha.md", "beta.md", "index.html", "rank_stability_alpha.png", "rank_stability_alpha.pdf",
                 "rank_stability_beta.png", "rank_stability_beta.pdf"):
        assert (out / name).stat().st_size > 0, name
    page = (out / "index.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "<style>" in page
    assert page.count("data:image/png;base64,") == 2
    assert 'id="ds-alpha"' in page and 'id="ds-beta"' in page
    assert "http://" not in page and "https://" not in page and "<script" not in page  # self-contained
    assert "p4.json: closed_loop_horizon missing" in page and "Provenance" in page
    assert "rank_stability_alpha.png" in (out / "alpha.md").read_text(encoding="utf-8")
    # restricted to one dataset
    only = script.main([f"root={world['root']}", f"out={tmp_path / 'only'}", "datasets=beta", "n_boot=20"])
    assert set(only) == {"beta"} and not (tmp_path / "only" / "alpha.md").exists()
    with pytest.raises(SystemExit):
        script.main([f"root={world['root']}", f"out={tmp_path / 'none'}", "datasets=gamma"])


def test_html_page_without_images(world):
    page = html_page(world["boards"])
    assert "<table>" in page and "data:image" not in page
