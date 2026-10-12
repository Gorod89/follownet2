"""Run the protocols P1-P4 (and the transfer part of P5) on one trained run.

Every protocol writes ``<p>.json`` (configuration, hash, provenance, summary) and
``<p>.<table>.parquet`` into the run directory
``results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/``; a result whose
configuration hash matches is not computed again unless ``force`` is set.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf

from follownet2 import pipeline
from follownet2.data.schema import EventSet, events_dir
from follownet2.data.splits import Split
from follownet2.models.registry import load_model
from follownet2.train.run import label, model_dir
from follownet2.utils import REPO_ROOT, config_hash, git_revision, read_json, resolve_path, write_json

PROTOCOLS = ("p1", "p2", "p3", "p4")
Runner = Callable[[object, object, dict, str], tuple[dict, dict]]

# full float32 on the GPU: TF32 arithmetic in cuDNN recurrent layers and matmuls adds noise
# of the order of 1e-3 to the acceleration of recurrent models, which the protocols must not see
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def test_events(cfg, dataset: str, version: str, split_kind: str, fold: int) -> list:
    """Usable test events of a dataset under a split kind and fold."""
    events = EventSet.from_parquet(events_dir(resolve_path(cfg.paths.events), dataset, version))
    split = Split.load(resolve_path(cfg.paths.splits) / f"{dataset}_{split_kind}.json")
    return pipeline._usable(pipeline.part_events(events, split, fold, "test"), cfg.protocol.warmup)


def _p1(cfg, model, pcfg: dict, device: str, dataset: str | None = None) -> tuple[dict, dict]:
    from follownet2.protocols.p1 import evaluate_p1

    t = cfg.train
    events = test_events(cfg, dataset or cfg.dataset.name, t.version, t.split, t.fold)
    return evaluate_p1(
        model, events, pcfg["horizons"], cfg.protocol.warmup, tuple(cfg.protocol.clip), pcfg["batch_windows"], device=device
    )


def speed_range(events, low: float = 0.01, high: float = 0.99) -> tuple[float, float]:
    """Speed quantiles of the follower speeds of the events [m/s]."""
    speeds = np.concatenate([ev.v for ev in events]) if len(events) else np.zeros(1)
    return float(np.quantile(speeds, low)), float(np.quantile(speeds, high))


def pulse_parameters(pcfg: dict, events) -> dict:
    """Parameters of the synthetic braking pulse; ``v0: auto`` takes the 95 % speed quantile
    of the training data (at most ``v0_max``) and ``v_low = v_low_ratio * v0``."""
    pulse = dict(pcfg["pulse"])
    v0_max, ratio = pulse.pop("v0_max", 25.0), pulse.pop("v_low_ratio", 0.4)
    if pulse.get("v0") in (None, "auto"):
        speeds = np.concatenate([ev.v for ev in events]) if len(events) else np.full(1, v0_max)
        pulse["v0"] = round(min(float(np.quantile(speeds, 0.95)), float(v0_max)), 2)
        pulse["v_low"] = round(float(ratio) * pulse["v0"], 2)
    return pulse


def leader_profiles(cfg, pcfg: dict, pulse: dict) -> list:
    """The OpenACC leader profiles (cached) and the synthetic braking pulse."""
    from follownet2.protocols.profiles import braking_pulse, load_profiles, openacc_profiles, save_profiles

    path = resolve_path(pcfg["profiles"])
    if not path.exists():
        dataset = OmegaConf.load(REPO_ROOT / "conf" / "dataset" / "openacc.yaml")
        profiles = openacc_profiles(resolve_path(dataset.raw_dir), pcfg["window"])
        path.parent.mkdir(parents=True, exist_ok=True)
        save_profiles(profiles, path)
    return [*load_profiles(path), braking_pulse(**pulse)]


def _p2(cfg, model, pcfg: dict, device: str) -> tuple[dict, dict]:
    from follownet2.protocols.p2 import evaluate_p2

    source = pipeline._sample(list(pipeline.load_events(cfg, cfg.train.version)), pcfg["source_sample"], cfg.seed)
    pulse = pulse_parameters(pcfg, source)
    low, high = speed_range(source)
    summary, tables = evaluate_p2(
        model, leader_profiles(cfg, pcfg, pulse), source, tuple(pcfg["n_vehicles"]), pcfg["hold"], tuple(cfg.protocol.clip),
        device=device, speed_range=(low, high),
    )
    summary["pulse"] = pulse
    summary["speed_range"] = [round(low, 2), round(high, 2)]
    return summary, tables


def _p3(cfg, model, pcfg: dict, device: str) -> tuple[dict, dict]:
    """P3 runs on the CPU in float64 whatever ``train.device`` says.

    On the GPU the recurrent layers use reduced-precision (TF32) arithmetic: the follower's
    acceleration then carries noise of the order of 1e-3 m/s^2, the settle criterion
    (``max |a| < 2e-3``) never holds, and an LSTM that settles at 15 speeds on the CPU has
    "no equilibrium" on the GPU. The whole grid takes seconds on the CPU.
    """
    from follownet2.protocols.p3 import evaluate_p3

    summary, tables = evaluate_p3(
        model, delta=pcfg["delta"], delta_large=pcfg.get("delta_large", 1.0), hold=pcfg["hold"],
        response=pcfg["response"], tol=pcfg["tol"], device=pcfg.get("device", "cpu"),
    )
    model.to(device)
    return summary, tables


def near_miss_episodes(cfg, names, max_episodes: int | None) -> list:
    episodes = []
    for name in names:
        directory = events_dir(resolve_path(cfg.paths.events), f"{name}_nearmiss", "reconstructed")
        if directory.exists():
            episodes.extend(EventSet.from_parquet(directory))
        else:
            warnings.warn(f"no reconstructed near-miss episodes at {directory}: run scripts/reconstruct_episodes.py dataset={name}")
    episodes = pipeline._usable(episodes, cfg.protocol.warmup)
    return pipeline._sample(episodes, max_episodes, cfg.seed)


def _p4(cfg, model, pcfg: dict, device: str) -> tuple[dict, dict]:
    from follownet2.protocols.p4 import evaluate_p4, synthetic_suite

    episodes = near_miss_episodes(cfg, pcfg["episodes"], pcfg["max_episodes"])
    summary, tables = evaluate_p4(
        model, synthetic_suite(pcfg["hold"]), episodes, cfg.protocol.warmup, tuple(cfg.protocol.clip), device=device
    )
    # the synthetic suite spans 15 to 35 m/s: the scenarios inside the speed range of the
    # training data are summarised separately (the same rule as the P2 profiles)
    source = pipeline._sample(list(pipeline.load_events(cfg, cfg.train.version)), 5000, cfg.seed)
    low, high = speed_range(source)
    scen = tables["scenarios"]
    in_range = (scen["v0"] >= low - 1.0) & (scen["v0"] <= high + 1.0)
    scen["in_range"] = in_range.to_numpy()
    part = scen[in_range]
    summary["speed_range"] = [round(low, 2), round(high, 2)]
    summary["n_scenarios_in_range"] = int(len(part))
    summary["collision_rate_synthetic_in_range"] = float(part["collided"].mean()) if len(part) else None
    summary["min_ttc_q50_in_range"] = float(part["min_ttc"].median()) if len(part) else None
    summary["exceedance_4_in_range"] = (
        float(part["n_exceed_4"].sum() / part["n_samples"].sum()) if len(part) and "n_exceed_4" in part else None
    )
    return summary, tables


RUNNERS: dict[str, Runner] = {"p1": _p1, "p2": _p2, "p3": _p3, "p4": _p4}


#: configuration keys that select what is run, not how: they do not enter the result hash
BOOKKEEPING = ("transfer_datasets",)


def _protocol_keys(pcfg: dict) -> dict:
    return {k: v for k, v in pcfg.items() if k not in BOOKKEEPING}


def checklist(cfg, name: str, pcfg: dict) -> dict:
    """The reporting checklist block of ``docs/checklist.md`` for one protocol result."""
    t = cfg.train
    manifest = read_json(events_dir(resolve_path(cfg.paths.events), cfg.dataset.name, t.version) / "manifest.json")
    train = read_json(model_dir(cfg) / "train.json")
    mechanistic = train.get("family") == "mechanistic"
    block = {
        "data_version": t.version,
        "reconstruction_method": manifest.get("reconstruction", {}).get("method") or ("none" if t.version == "raw" else None),
        "split_type": t.split,
        "mechanistic_calibration_protocol": (
            "closed loop, NRMSE(s)+NRMSE(v), differential evolution, "
            f"{train['training'].get('kind', 'pooled')} parameters from the training part"
            if mechanistic else "not a mechanistic model"
        ),
        "seeds": {"global": int(cfg.seed), "train": int(t.seed), "split": train.get("split", {}).get("seed")},
    }
    horizons = {
        "p1": "10 s, 30 s, whole event; warm-up 3 s",
        "p2": f"hold {pcfg.get('hold')} s then the leader profile; warm-up 3 s",
        "p4": f"hold {pcfg.get('hold')} s then the manoeuvre, episodes from the end of the warm-up; warm-up 3 s",
    }
    if name in horizons:
        block["closed_loop_horizon"] = horizons[name]
    if name == "p3":
        block["stability_test"] = (
            f"analytic criterion (memoryless models) and step-response gain on 0.002-2 Hz, "
            f"delta {pcfg.get('delta')} m/s, v_e 5..30 m/s"
        )
    if name == "p4":
        block["rare_event_suite"] = {"synthetic": "p4-v1", "episodes": [f"{n}_nearmiss/reconstructed" for n in pcfg.get("episodes", [])]}
    return block


def _run(cfg, model, provenance: dict, name: str, pcfg: dict, key_cfg: dict, target: Path, runner, device: str) -> dict:
    key = config_hash(key_cfg)
    if target.exists() and not cfg.get("force", False):
        old = read_json(target)
        same_model = old.get("model", {}).get("provenance", {}).get("config_hash") == provenance.get("config_hash")
        if old.get("config_hash") == key and same_model:  # a retrained checkpoint makes the result stale
            if "checklist" not in old:
                old["checklist"] = checklist(cfg, name, pcfg)
                write_json(target, old)
            return old["summary"]
    started = time.time()
    summary, tables = runner(cfg, model, pcfg, device)
    write_json(
        target,
        {
            "protocol": name,
            "config": key_cfg,
            "config_hash": key,
            "data_version": cfg.train.version,
            "model": {"label": label(cfg), "class": cfg.model.name, "family": model.family, "provenance": provenance},
            "git_revision": git_revision(),
            "seconds": round(time.time() - started, 1),
            "checklist": checklist(cfg, name, pcfg),
            "summary": summary,
        },
    )
    for table_name, table in tables.items():  # p1_<other>.<table>.parquet for the transfer scores
        table.to_parquet(target.with_name(f"{target.stem}.{table_name}.parquet"), index=False)
    return summary


def evaluate_run(cfg) -> dict:
    """Protocol results of the run selected by the configuration (``conf/evaluate.yaml``)."""
    t = cfg.train
    out = model_dir(cfg)
    if not (out / "model.pt").exists():
        raise FileNotFoundError(f"no trained run at {out}: run scripts/train.py first")
    torch.set_num_threads(int(t.threads))
    device = pipeline._device(t.device)
    model, provenance = load_model(out / "model.pt", device)
    common = {
        "warmup": int(cfg.protocol.warmup),
        "clip": [float(c) for c in cfg.protocol.clip],
        "version": t.version,
        "split": t.split,
        "fold": int(t.fold),
    }
    results: dict = {}
    for name in cfg.protocols:
        pcfg = pipeline._plain(cfg[name])
        key_cfg = {"protocol": name, **common, **_protocol_keys(pcfg)}
        results[name] = _run(cfg, model, provenance, name, pcfg, key_cfg, out / f"{name}.json", RUNNERS[name], device)
    pcfg = pipeline._plain(cfg.p1)
    for other in pcfg.get("transfer_datasets") or []:
        if other == cfg.dataset.name:
            continue
        key_cfg = {"protocol": "p1", "tested_on": other, **common, **_protocol_keys(pcfg)}
        runner = lambda cfg, model, pcfg, device, other=other: _p1(cfg, model, pcfg, device, other)  # noqa: E731
        results[f"p1_{other}"] = _run(cfg, model, provenance, "p1", pcfg, key_cfg, out / f"p1_{other}.json", runner, device)
    return {"model": label(cfg), "dataset": cfg.dataset.name, "directory": str(out), **results}
