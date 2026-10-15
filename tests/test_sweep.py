"""Experiment E5 (trafficloop/experiments/sweep.py, scripts/sweep.py) without training: the sampling of
the configurations, the hydra overrides of a run, the validation events and criteria, and the stages
train (with a fabricated runner), validate, select, evaluate and report on fabricated run directories."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from trafficloop import pipeline
from trafficloop.data.splits import event_table, make_split
from trafficloop.experiments import sweep as sw
from trafficloop.models.registry import create
from trafficloop.protocols.p0 import evaluate_p0, pooled_rmse
from trafficloop.protocols.p1 import evaluate_p1
from trafficloop.train.run import label, model_dir, train_config
from trafficloop.utils import config_hash

REPO = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- configurations


def test_sampling_is_deterministic_and_independent_of_n():
    for model in sw.MODELS:
        first = sw.sample_configs(model, 8, seed=0)
        assert first == sw.sample_configs(model, 8, seed=0)
        assert sw.sample_configs(model, 12, seed=0)[:8] == first
        other = sw.sample_configs(model, 8, seed=1)
        assert other[0] == first[0] and other[1:] != first[1:]
        assert [c["index"] for c in first] == list(range(8))
        assert [c["label"] for c in first] == [f"sweep_{model}_{i:02d}" for i in range(8)]


def test_samples_lie_in_the_search_space():
    for model in sw.MODELS:
        draws = sw.sample_configs(model, 40, seed=3)[1:]
        for c in draws:
            if model == "mlp":
                assert len(c["hidden"]) in sw.MLP_LAYERS and len(set(c["hidden"])) == 1 and c["hidden"][0] in sw.WIDTHS
            else:
                assert c["hidden"] in sw.WIDTHS
            assert c["history"] in sw.HISTORIES and c["weight_decay"] in sw.WEIGHT_DECAYS
            assert c["batch_size"] in sw.BATCH_SIZES and isinstance(c["use_a_prev"], bool)
            assert sw.LR_RANGE[0] <= c["lr"] <= sw.LR_RANGE[1] and float(f"{c['lr']:.3g}") == c["lr"]
        for name, values in (("history", sw.HISTORIES), ("batch_size", sw.BATCH_SIZES), ("use_a_prev", sw.A_PREV)):
            assert {c[name] for c in draws} == set(values)
        lr = np.log10([c["lr"] for c in draws])
        assert lr.min() < -3.4 and lr.max() > -2.8  # spread over the log range -4 .. -2.52


def test_configuration_zero_is_the_reference_run():
    lstm = sw.sample_configs("lstm", 1, 0)[0]
    assert lstm == {"index": 0, "label": "sweep_lstm_00", "hidden": 64, "history": 10, "use_a_prev": False,
                    "lr": 0.001, "weight_decay": 0.0, "batch_size": 4096}
    assert sw.sample_configs("mlp", 1, 5)[0]["hidden"] == [64, 64]
    # the overrides of configuration 0 change nothing but the label and the results root
    sweep = sw.Sweep(dataset="ngsim", model="lstm", n=1)
    swept = pipeline.run_config(sw.compose_config("train", sw.run_overrides(sweep, lstm)))
    plain = pipeline.run_config(sw.compose_config("train", ["dataset=ngsim", "model=lstm"]))
    assert swept["model"].pop("label") == "sweep_lstm_00" and swept["paths"].pop("results") == "results/sweeps"
    plain["paths"].pop("results")
    assert swept == plain


def test_run_overrides_set_the_run_directory_label_and_hyperparameters():
    sweep = sw.Sweep(dataset="ngsim", model="mlp", n=4, seed=2, fold=1, overrides=("train.optim.epochs=1",))
    config = sw.sample_configs("mlp", 4, 2)[3]
    cfg = sw.compose_config("train", sw.run_overrides(sweep, config))
    assert label(cfg) == "sweep_mlp_03" and model_dir(cfg) == sweep.run_dir("sweep_mlp_03")
    assert model_dir(cfg).is_relative_to(REPO / "results" / "sweeps" / "models")
    assert not model_dir(cfg).is_relative_to(REPO / "results" / "models")  # never seen by the leaderboard
    assert pipeline._plain(cfg.model.kwargs) == {k: config[k] for k in ("history", "use_a_prev", "hidden")}
    optim, default = train_config(cfg), sw.compose_config("train", ["model=mlp"]).train.optim
    assert (optim.lr, optim.weight_decay, optim.batch_size) == (config["lr"], config["weight_decay"], config["batch_size"])
    assert optim.epochs == 1 and (optim.patience, optim.batches_per_epoch) == (default.patience, default.batches_per_epoch)
    assert (cfg.train.seed, cfg.train.fold, cfg.train.split, cfg.train.version) == (2, 1, "driver", "reconstructed")


def test_overrides_of_the_keys_of_the_sweep_are_refused():
    for item in ("train.optim.lr=0.1", "++model.label=x", "paths.results=elsewhere", "model.kwargs.hidden=8",
                 "train.optim={}", "model.optim.batch_size=8", "train.seed=3", "train.max_train_events"):
        with pytest.raises(sw.SweepError):
            sw.Sweep(overrides=(item,))
    with pytest.raises(sw.SweepError):
        sw.Sweep(model="seq2seq")
    sw.Sweep(overrides=("train.max_train_events=300", "train.optim.epochs=1", "train.device=cpu"))


def test_one_directory_holds_one_sweep(tmp_path):
    root = str(tmp_path / "sweeps")
    sw.write_configs(sw.Sweep(model="mlp", n=3, root=root))
    sw.write_configs(sw.Sweep(model="mlp", n=5, root=root))  # more configurations of the same sweep
    saved = read(tmp_path / "sweeps" / "ngsim" / "mlp" / "configs.json")
    assert saved["n"] == 5 and len(saved["configs"]) == 5 and saved["configs"][0]["hidden"] == [64, 64]
    assert saved["configs"] == sw.sample_configs("mlp", 5, 0)
    for other in ({"seed": 1}, {"fold": 2}, {"split": "event"}):
        with pytest.raises(sw.SweepError):
            sw.write_configs(sw.Sweep(model="mlp", n=3, root=root, **other))


def test_command_line():
    spec = importlib.util.spec_from_file_location("sweep_script", REPO / "scripts" / "sweep.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sweep, stages = script.arguments(["dataset=ngsim", "model=mlp", "n=2", "stage=train,validate", "root=results/sweeps_x",
                                      "protocols=[p1,p3]", "train.optim.epochs=1", "+train.extra=1"])
    assert stages == ("train", "validate")
    assert (sweep.model, sweep.n, sweep.seed, sweep.root, sweep.protocols) == ("mlp", 2, 0, "results/sweeps_x", ("p1", "p3"))
    assert sweep.overrides == ("train.optim.epochs=1", "+train.extra=1")
    sweep, stages = script.arguments(["model=gru"])
    assert stages == sw.STAGES and sweep.protocols == sw.PROTOCOLS and (sweep.dataset, sweep.n) == ("ngsim", 16)
    for bad in (["epochs=1"], ["model=knn"], ["train.optim.lr=0.1"], ["n=x"]):
        with pytest.raises(SystemExit):
            script.arguments(bad)


# ----------------------------------------------------------------------------- validation


def test_validation_events_come_from_the_validation_part(driver_events, monkeypatch):
    split = make_split(event_table(driver_events), "driver", 5, 0, 0.1875, "synthetic")
    monkeypatch.setattr(sw.pipeline, "load_events", lambda cfg, version=None: driver_events)
    monkeypatch.setattr(sw.Split, "load", staticmethod(lambda path: split))
    cfg = OmegaConf.create({"train": {"version": "reconstructed", "split": "driver", "fold": 1, "seed": 0},
                            "protocol": {"warmup": 30}, "paths": {"splits": "data/splits"}, "dataset": {"name": "synthetic"}})
    parts = split.partition(1)
    every = sw.validation_events(cfg, max_events=None)
    assert [ev.event_id for ev in every] == [e for e in parts["val"] if e in driver_events] and len(every) > 5
    capped = sw.validation_events(cfg, max_events=5)
    assert len(capped) == 5 and [ev.event_id for ev in capped] == [ev.event_id for ev in sw.validation_events(cfg, 5)]
    assert {ev.event_id for ev in capped} <= set(parts["val"]) and not {ev.event_id for ev in capped} & set(parts["test"])


def test_validation_criteria_are_the_10_s_windows_of_p1(driver_events):
    torch.manual_seed(0)
    model = create("mlp", history=10)
    events = list(driver_events)[:12]
    cfg = OmegaConf.create({"protocol": {"warmup": 30, "clip": [-8.0, 4.0]}, "p1": {"batch_windows": 64}})
    criteria = sw.validation_criteria(model, events, cfg, "cpu")
    summary, _ = evaluate_p1(model, events, [10.0], 30, (-8.0, 4.0), 4096)
    assert criteria["val_rmse_s_10s"] == pytest.approx(summary["h10"]["rmse_s"], rel=1e-6)
    assert criteria["val_collision_rate_10s"] == summary["h10"]["collision_rate"]
    assert criteria["val_n_windows_10s"] == summary["h10"]["n_windows"] > 12 and criteria["val_n_events"] == 12
    one_step = pooled_rmse(evaluate_p0(model, events, 30, dtype=torch.float32, clip=(-8.0, 4.0)))
    assert criteria["val_rmse_a_p0"] == pytest.approx(one_step["rmse_a"], rel=1e-6)


# ----------------------------------------------------------------------------- fabricated sweep

RMSE_A = {0: 0.30, 1: 0.20, 3: 0.10, 4: 0.05, 5: 0.01}  # 2 is never trained
RMSE_S = {0: 5.0, 1: 3.0, 3: 9.0, 5: 0.5}  # 4 has no criteria, 5 the criteria of an earlier checkpoint


@pytest.fixture()
def fabricated(tmp_path):
    """Six lstm configurations: 0, 1 and 3 validated (with a checkpoint file), 2 missing, 4 trained
    without criteria, 5 with the criteria of an earlier checkpoint (4 and 5 without model.pt)."""
    sweep = sw.Sweep(model="lstm", n=6, root=str(tmp_path / "sweeps"), protocols=("p1", "p3"))
    configs = sw.write_configs(sweep)
    key = config_hash(sw.validation_settings(sweep, sw.compose_config("evaluate", sw.run_overrides(sweep, configs[0]))))
    for index, a in RMSE_A.items():
        directory = sweep.run_dir(configs[index]["label"])
        write_json(directory / "train.json", {
            "config_hash": f"run{index}", "n_parameters": 1000 + index, "test": {"p0": {"rmse_a": 1.5 * a}},
            "training": {"best_val_rmse_a": a, "epochs_run": 10 + index, "seconds": 2.5 * index, "n_train_events": 9461,
                         "config": {"epochs": 40, "batches_per_epoch": 200, "patience": 6}},
        })
        if index in RMSE_S:
            owner = "earlier" if index == 5 else f"run{index}"
            write_json(directory / "sweep.json", {
                "config_hash": key, "model": {"config_hash": owner},
                "criteria": {"val_rmse_a": a, "val_rmse_s_10s": RMSE_S[index], "val_collision_rate_10s": 0.01 * index,
                             "val_rmse_a_p0": 1.2 * a, "val_rmse_v_10s": 1.0},
            })
        if index in (0, 1, 3):
            (directory / "model.pt").write_bytes(b"")
    return sweep, configs


def test_validate_writes_the_table_from_the_run_directories(fabricated):
    sweep, configs = fabricated
    frame = sw.stage_validate(sweep, configs)  # nothing to compute: no event is loaded
    assert list(frame.columns) == list(sw.COLUMNS)
    assert list(frame["status"]) == ["validated", "validated", "missing", "validated", "trained", "trained"]
    table = pd.read_csv(sweep.directory / "sweep.csv")
    assert list(table["label"]) == [c["label"] for c in configs]
    np.testing.assert_allclose(table["val_rmse_a"], [0.30, 0.20, np.nan, 0.10, 0.05, 0.01])
    np.testing.assert_allclose(table["val_rmse_s_10s"], [5.0, 3.0, np.nan, 9.0, np.nan, np.nan])
    np.testing.assert_allclose(table["val_rmse_a_p0"], [0.36, 0.24, np.nan, 0.12, np.nan, np.nan])
    assert table.loc[3, "epochs"] == 13 and table.loc[3, "seconds"] == 7.5 and table.loc[3, "n_parameters"] == 1003
    assert list(table["lr"]) == [c["lr"] for c in configs] and list(table["hidden"]) == [c["hidden"] for c in configs]
    failures = read(sweep.directory / "failures.json")["validate"]
    assert [(f["label"], f["status"]) for f in failures] == [("sweep_lstm_04", "no checkpoint (model.pt)"),
                                                             ("sweep_lstm_05", "no checkpoint (model.pt)")]


def test_select_takes_validated_runs_only(fabricated):
    sweep, configs = fabricated
    selection = sw.stage_select(sweep, configs)
    assert selection["one_step"]["label"] == "sweep_lstm_03" and selection["closed_loop"]["label"] == "sweep_lstm_01"
    assert not selection["coincide"] and selection["n_validated"] == 3 and selection["reference"]["index"] == 0
    assert selection["one_step"]["hyperparameters"] == {k: configs[3][k] for k in sw.HYPERPARAMETERS}
    assert selection["closed_loop"]["val_rmse_s_10s"] == 3.0 and selection["one_step"]["val_rmse_a_p0"] == pytest.approx(0.12)
    assert selection["kendall_tau_criteria"]["tau"] == pytest.approx(-1 / 3)  # one concordant pair of three
    saved = read(sweep.directory / "selection.json")
    assert saved["one_step"]["index"] == 3 and (saved["dataset"], saved["model"], saved["n"]) == ("ngsim", "lstm", 6)


def test_select_breaks_ties_by_the_lower_index(fabricated):
    sweep, configs = fabricated
    frame = sw.collect(sweep, configs)
    frame.loc[frame["index"] == 3, "val_rmse_s_10s"] = 3.0
    assert sw.select(frame, configs)["closed_loop"]["index"] == 1
    frame.loc[frame["index"] == 1, "val_rmse_a"] = 0.10
    chosen = sw.select(frame, configs)
    assert chosen["one_step"]["index"] == 1 and chosen["coincide"]
    frame["val_rmse_s_10s"] = np.nan
    with pytest.raises(sw.SweepError):
        sw.select(frame, configs)


def test_evaluate_scores_the_reference_and_both_selections(fabricated):
    sweep, configs = fabricated
    with pytest.raises(sw.SweepError):  # no selection yet
        sw.stage_evaluate(sweep, configs, runner=lambda script, args: None)
    sw.stage_select(sweep, configs)
    calls = []
    assert sw.stage_evaluate(sweep, configs, runner=lambda script, args: calls.append((script, list(args)))) == []
    assert [script for script, _ in calls] == ["evaluate.py"] * 3
    labels = [next(a for a in args if a.startswith("++model.label=")).split("=", 1)[1] for _, args in calls]
    assert labels == ["sweep_lstm_00", "sweep_lstm_03", "sweep_lstm_01"]
    by_label = {c["label"]: c for c in configs}
    for (_, args), name in zip(calls, labels):
        assert args == [*sw.run_overrides(sweep, by_label[name]), "protocols=[p1,p3]"]
    unknown = sw.Sweep(model="lstm", n=6, root=sweep.root, protocols=("p1", "p9"))
    with pytest.raises(sw.SweepError):
        sw.stage_evaluate(unknown, configs, runner=lambda script, args: None)


def test_report_lists_the_criteria_and_the_protocols(fabricated):
    sweep, configs = fabricated
    sw.stage_validate(sweep, configs)
    sw.stage_select(sweep, configs)
    reference = sweep.run_dir("sweep_lstm_00")
    owner = {"model": {"provenance": {"config_hash": "run0"}}}
    write_json(reference / "p1.json", {**owner, "summary": {"h10": {"rmse_s": 2.5}, "whole": {"rmse_s": 42.0}}})
    write_json(reference / "p2.json", {**owner, "summary": {"collisions_100": 9, "collisions_100_in_range": 7}})
    write_json(reference / "p3.json", {"model": {"provenance": {"config_hash": "earlier"}}, "summary": {"unstable_fraction": 0.75}})
    scores = sw.protocol_scores(reference)
    assert (scores["p1_whole_rmse_s"], scores["p2_collisions_100_in_range"], scores["p3_unstable_fraction"]) == (42.0, 7.0, 0.75)
    text = sw.stage_report(sweep, configs).read_text(encoding="utf-8")
    assert "| sweep_lstm_00 | reference | 0.45 | 2.5 | 42 | 7 | 0.75 | p3 |" in text
    assert "| sweep_lstm_03 | one-step selection | 0.15 | - | - | - | - | - |" in text
    assert "| sweep_lstm_01 | closed-loop selection |" in text
    assert "different configurations" in text and "-0.333" in text and "`epochs` 40, `batches_per_epoch` 200" in text
    assert "| 02 | " in text and "| missing |" in text
    assert "| validate | sweep_lstm_04 | no checkpoint (model.pt) |" in text


def test_train_runs_every_unfinished_configuration_once(tmp_path):
    sweep = sw.Sweep(model="gru", n=3, root=str(tmp_path / "sweeps"), overrides=("train.optim.epochs=1",))
    configs = sw.write_configs(sweep)
    calls = []

    def runner(script, args):
        calls.append(list(args))
        cfg = sw.compose_config("train", args)
        if label(cfg) == "sweep_gru_02":
            return "1"
        write_json(model_dir(cfg) / "train.json", {"config_hash": config_hash(pipeline.run_config(cfg))})
        return None

    failures = sw.stage_train(sweep, configs, runner)
    assert len(calls) == 3 and failures == [{"label": "sweep_gru_02", "status": "1"}]
    assert (sweep.run_dir("sweep_gru_01") / "train.json").exists()
    calls.clear()
    assert sw.stage_train(sweep, configs, runner) == failures and len(calls) == 1  # finished runs are skipped
    assert read(sweep.directory / "failures.json")["train"] == failures
    calls.clear()
    sw.stage_train(sw.Sweep(model="gru", n=3, root=sweep.root, overrides=("train.optim.epochs=2",)), configs, runner)
    assert len(calls) == 3  # another budget is another configuration hash
