"""Train one model on one dataset, data version, split and fold (entry of scripts/train.py).

Writes ``results/models/<dataset>/<version>/<split>-fold<k>/<model>/seed<s>/`` with the
checkpoint ``model.pt`` and ``train.json`` (training log, provenance, and the scores of the
model on the test part: one step and closed loop over the whole event).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from follownet2 import pipeline
from follownet2.calibration.calibrate import calibrate_global, calibrate_per_event
from follownet2.data.splits import Split
from follownet2.models.base import MechanisticModel
from follownet2.models.registry import MEDIAN_SUFFIX, create, model_class, save_model
from follownet2.protocols.closed_loop import evaluate_whole_event, pooled_closed_loop
from follownet2.protocols.p0 import evaluate_p0, pooled_rmse
from follownet2.train.trainer import TrainConfig
from follownet2.train.windows import WindowSet
from follownet2.utils import config_hash, git_revision, read_json, resolve_path, seed_everything, write_json


def label(cfg) -> str:
    """Name of the run: the model, or the label of a training variant of it (``lstm_loop``)."""
    return cfg.model.get("label") or cfg.model.name


def model_dir(cfg) -> Path:
    t = cfg.train
    return (
        resolve_path(cfg.paths.results) / "models" / cfg.dataset.name / t.version
        / f"{t.split}-fold{t.fold}" / label(cfg) / f"seed{t.seed}"
    )


def score_model(cfg, model, test: list) -> tuple[dict, dict]:
    """Scores on the test part: one step (P0) and closed loop over the whole event."""
    device = pipeline._device(cfg.train.device)
    clip = tuple(cfg.protocol.clip)
    one_step = evaluate_p0(model, test, cfg.protocol.warmup, device=device, dtype=torch.float32, clip=clip)
    closed = evaluate_whole_event(model, test, cfg.protocol.warmup, clip, device=device)
    scores = {"p0": pooled_rmse(one_step), "closed_loop_whole_event": pooled_closed_loop(closed)}
    return scores, {"test_p0": one_step, "test_closed_loop": closed}


def _sample(events: list, n: int | None, seed: int) -> list:
    if n is None or len(events) <= n:
        return events
    keep = np.sort(np.random.default_rng([seed, 11]).choice(len(events), size=n, replace=False))
    return [events[i] for i in keep]


def train_config(cfg) -> TrainConfig:
    values = {**pipeline._plain(cfg.train.optim), **pipeline._plain(cfg.model.get("optim"))}
    return TrainConfig(seed=cfg.train.seed, device=pipeline._device(cfg.train.device), **values)


def fit_mechanistic(cfg, train_events: list):
    """Pooled calibration, or the median of the per-event estimates for ``<model>_median``."""
    cal = pipeline.calibration_config(cfg)
    cls = model_class(cfg.model.name)
    started = time.time()
    if cfg.model.name.endswith(MEDIAN_SUFFIX):
        sample = _sample(train_events, cfg.calibration.per_event_sample, cfg.train.seed)
        table = calibrate_per_event(cls, sample, cal)
        ok = table[~table["collided"]]
        params = [float(ok[name].median()) for name in cls.param_names]
        log = {"kind": "median of per-event estimates", "n_events": len(sample), "n_used": int(len(ok))}
    else:
        sample = _sample(train_events, cfg.calibration.global_sample, cfg.train.seed)
        fit = calibrate_global(cls, sample, cal)
        params = [fit["params"][name] for name in cls.param_names]
        log = {"kind": "pooled", **{k: fit[k] for k in ("objective", "rmse_s", "rmse_v", "collision_rate", "n_events")}}
    log.update(params=dict(zip(cls.param_names, params)), seconds=round(time.time() - started, 1))
    return create(cfg.model.name, params=params), log


def train_model(cfg) -> dict:
    t = cfg.train
    key = config_hash(pipeline.run_config(cfg))
    done = model_dir(cfg) / "train.json"
    if done.exists() and read_json(done).get("config_hash") == key and not cfg.train.get("force", False):
        return read_json(done)  # a run that was interrupted continues after the finished models
    seed_everything(t.seed)
    torch.set_num_threads(int(t.threads))
    device = pipeline._device(t.device)
    warmup = cfg.protocol.warmup
    events = pipeline.load_events(cfg, t.version)
    split = Split.load(pipeline.split_path(cfg, t.split))
    parts = {name: pipeline.part_events(events, split, t.fold, name) for name in ("train", "val", "test")}

    if issubclass(model_class(cfg.model.name), MechanisticModel):
        model, log = fit_mechanistic(cfg, pipeline._usable(parts["train"], warmup))
    else:
        model = create(cfg.model.name, **pipeline._plain(cfg.model.get("kwargs")))
        optim = train_config(cfg)
        horizon = max(model.train_horizon, optim.closed_loop_horizon if optim.closed_loop_weight > 0 else 1)
        usable = lambda part, n: _sample(pipeline._usable(parts[part], warmup + horizon), n, t.seed)  # noqa: E731
        train = WindowSet.from_events(usable("train", t.max_train_events), device=device, horizon=horizon)
        val = WindowSet.from_events(usable("val", t.max_val_events), device=device, horizon=horizon)
        log = model.fit(train, val, optim)
        log.update(n_train_events=train.batch.n_events, n_train_windows=len(train), n_val_windows=len(val))
        del train, val

    scores, tables = score_model(cfg, model, pipeline._usable(parts["test"], warmup))
    provenance = {
        "dataset": cfg.dataset.name,
        "data_version": t.version,
        "split": {"kind": t.split, "fold": int(t.fold), "seed": int(split.seed)},
        "seed": int(t.seed),
        "config_hash": key,
        "git_revision": git_revision(),
    }
    out = model_dir(cfg)
    save_model(model, out / "model.pt", provenance)
    summary = {
        "model": label(cfg),
        "class": cfg.model.name,
        "family": model.family,
        **provenance,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "training": log,
        "test": scores,
    }
    write_json(out / "train.json", summary)
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.events.parquet", index=False)
    return summary


def rescore_model(cfg) -> dict:
    """Score the checkpoint of a finished run again (after a change of a protocol)."""
    from follownet2.models.registry import load_model

    out = model_dir(cfg)
    summary = read_json(out / "train.json")
    model, _ = load_model(out / "model.pt", pipeline._device(cfg.train.device))
    events = pipeline.load_events(cfg, cfg.train.version)
    split = Split.load(pipeline.split_path(cfg, cfg.train.split))
    test = pipeline._usable(pipeline.part_events(events, split, cfg.train.fold, "test"), cfg.protocol.warmup)
    summary["test"], tables = score_model(cfg, model, test)
    write_json(out / "train.json", summary)
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.events.parquet", index=False)
    return summary
