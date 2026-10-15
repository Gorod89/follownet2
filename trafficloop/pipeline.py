"""Milestone 1 pipeline: events in two data versions, splits, baselines, calibration.

The functions take the hydra config (see conf/config.yaml) and are called by the thin
scripts in scripts/. Everything written to disk carries the config hash and the data
version tag.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from trafficloop import compat
from trafficloop.calibration.calibrate import (
    CalibrationConfig,
    calibrate_global,
    calibrate_per_event,
    parameter_spread,
    score_events,
)
from trafficloop.data.extraction import ExtractionConfig, extract_events
from trafficloop.data.rare_events import EpisodeConfig, mine_episodes
from trafficloop.data.reconstruction import ReconstructionConfig, consistency_residual, reconstruct_events, summarise
from trafficloop.data.regimes import RegimeConfig, assign_regimes
from trafficloop.data.registry import get_loader
from trafficloop.data.schema import DATA_VERSIONS, Event, EventSet, events_dir
from trafficloop.data.splits import (
    DRIVER_DISJOINT,
    Split,
    check_leakage,
    event_table,
    make_legacy_split,
    make_split,
)
from trafficloop.models.mechanistic import MECHANISTIC
from trafficloop.models.persistence import PersistenceBaseline, ZeroAcceleration
from trafficloop.protocols.p0 import RecordedAcceleration, evaluate_p0, pooled_rmse
from trafficloop.results import write_result
from trafficloop.utils import (
    config_hash,
    file_sha256,
    git_revision,
    read_json,
    require_free_space,
    resolve_path,
    to_plain,
    write_json,
)

NEAR_MISS_SUFFIX = "_nearmiss"


def _plain(cfg: Any) -> dict:
    return to_plain(cfg) or {}


def run_config(cfg) -> dict:
    """The configuration that identifies a result: everything but what only steers the run
    (the list of steps, the number of threads, the switch that repeats a finished run)."""
    plain = _plain(cfg)
    plain.pop("steps", None)
    plain.pop("rescore", None)
    if isinstance(plain.get("train"), dict):
        for key in ("force", "threads"):
            plain["train"].pop(key, None)
    return plain


def _resume(path: Path, key: str) -> dict | None:
    """Summary row of a calibration that was finished with this configuration, if there is one.

    The GPU is shared between several jobs and fails now and then with an error that leaves
    the process unusable; a run is then started again and continues after the last finished
    calibration.
    """
    if not path.exists():
        return None
    saved = read_json(path)
    return saved["row"] if saved.get("config_hash") == key else None


def _device(name: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name


def extraction_config(cfg) -> ExtractionConfig:
    return ExtractionConfig(**{**_plain(cfg.extraction), **_plain(cfg.dataset.get("extraction"))})


def reconstruction_config(cfg) -> ReconstructionConfig:
    return ReconstructionConfig(**{**_plain(cfg.reconstruction), **_plain(cfg.dataset.get("reconstruction"))})


def calibration_config(cfg) -> CalibrationConfig:
    c = cfg.calibration
    return CalibrationConfig(
        popsize=c.popsize,
        maxiter=c.maxiter,
        tol=c.tol,
        seed=cfg.seed,
        warmup=cfg.protocol.warmup,
        clip=tuple(cfg.protocol.clip),
        batch_events=c.batch_events,
        max_rollouts=c.max_rollouts,
        device=_device(c.device),
        dtype=c.dtype,
        bounds={k: tuple(v) for k, v in _plain(c.get("bounds")).items()},
    )


def source_files(raw_dir: Path | None, hash_below: int = 64 << 20) -> list[dict]:
    """Files of the raw directory with sizes; sha256 for files below ``hash_below`` bytes."""
    if raw_dir is None or not raw_dir.is_dir():
        return []
    out = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        size = path.stat().st_size
        entry = {"file": path.relative_to(raw_dir).as_posix(), "bytes": size}
        if size <= hash_below:
            entry["sha256"] = file_sha256(path)
        out.append(entry)
    return out


def _limit(events: list[Event], max_events: int | None, seed: int) -> list[Event]:
    """Keep all events of randomly drawn followers until ``max_events`` are reached."""
    if max_events is None or len(events) <= max_events:
        return events
    by_follower: dict[str, list[Event]] = {}
    for ev in events:
        by_follower.setdefault(ev.follower_id, []).append(ev)
    names = sorted(by_follower)
    keep: list[Event] = []
    for i in np.random.default_rng([seed, 77]).permutation(len(names)):
        keep.extend(by_follower[names[i]])
        if len(keep) >= max_events:
            break
    chosen = {ev.event_id for ev in keep}
    return [ev for ev in events if ev.event_id in chosen]


def load_raw_events(cfg) -> tuple[list[Event], list[Event]]:
    """Loader, extraction, regime labels: the raw events and the near-miss episodes.

    The source is read once; every follower series is cut into FollowNet-compatible events
    and searched for near-miss episodes.
    """
    ds = cfg.dataset
    loader = get_loader(ds.loader)
    raw_dir = None if ds.get("raw_dir") is None else resolve_path(ds.raw_dir)
    options = _plain(ds.get("options"))
    load = loader.load_events if hasattr(loader, "load_events") else loader.load_pairs
    if "dataset" in inspect.signature(load).parameters:
        options.setdefault("dataset", ds.name)
    episodes: list[Event] = []
    if hasattr(loader, "load_events"):
        events = list(loader.load_events(raw_dir, **options))
    else:
        extraction = extraction_config(cfg)
        mining = _plain(cfg.get("episodes"))
        enabled = mining.pop("enabled", False)
        if "triggers" in mining:
            mining["triggers"] = tuple(mining["triggers"])
        episode_cfg = EpisodeConfig(**mining)
        events, counters, episode_counters = [], {}, {}
        for pair in loader.load_pairs(raw_dir, **options):
            events.extend(extract_events(pair, extraction, counters))
            if enabled:
                episodes.extend(mine_episodes(pair, episode_cfg, episode_counters))
    source = ds.get("position_source") or getattr(loader, "POSITION_SOURCE", "track")
    for ev in [*events, *episodes]:
        if ev.dataset != ds.name or not ev.follower_id.startswith(ds.name + "/"):
            raise ValueError(
                f"loader {ds.loader} returned dataset {ev.dataset!r} and follower {ev.follower_id!r}, "
                f"the configuration is named {ds.name!r}"
            )
    for ev in episodes:
        # the ids keep the prefix of the dataset, so that the splits of the dataset apply
        ev.dataset = ds.name + NEAR_MISS_SUFFIX
    for ev in [*events, *episodes]:
        ev.meta.setdefault("position_source", source)
        ev.meta["data_version"] = "raw"
    events = _limit(events, ds.get("max_events"), cfg.seed)
    assign_regimes([*events, *episodes], RegimeConfig(**_plain(cfg.regimes)))
    return events, episodes


def write_events(cfg, events: EventSet, dataset: str, version: str, manifest: dict) -> Path:
    """Write an event directory after checking that the volume has room for it."""
    directory = events_dir(resolve_path(cfg.paths.events), dataset, version)
    n_samples = sum(len(ev) for ev in events)
    columns = 6 if cfg.storage.compact else 8
    require_free_space(directory, n_samples * columns * 8, reserve=int(cfg.storage.reserve_mb) << 20)
    return events.to_parquet(directory, manifest, compact=cfg.storage.compact)


def build_dataset(cfg) -> dict:
    """Write ``data/events/<dataset>/{raw,reconstructed}`` and return the summary."""
    ds = cfg.dataset
    started = time.time()
    events, episodes = load_raw_events(cfg)
    if not events:
        raise RuntimeError(f"dataset {ds.name}: no event passed the extraction criteria")
    raw = EventSet(events)
    raw.validate()
    raw_dir = None if ds.get("raw_dir") is None else resolve_path(ds.raw_dir)
    manifest = {
        "dataset": ds.name,
        "loader": ds.loader,
        "config": {
            "dataset": _plain(ds),
            "extraction": _plain(cfg.extraction),
            "reconstruction": _plain(cfg.reconstruction),
            "regimes": _plain(cfg.regimes),
            "episodes": _plain(cfg.get("episodes")),
            "seed": cfg.seed,
        },
        "git_revision": git_revision(),
        "siblings": compat.sibling_status(),
        "source_files": source_files(raw_dir),
        "regimes": _count(ev.meta["regime"] for ev in events),
        "sites": _count(ev.site for ev in events),
        "a_source": _count(ev.meta.get("a_source") for ev in events),
        "events_per_follower": _per_follower(events),
    }
    manifest["config_hash"] = config_hash(manifest["config"])
    write_events(cfg, raw, ds.name, "raw", {**manifest, "consistency": _consistency(events)})

    rec_cfg = reconstruction_config(cfg)
    done, failed = reconstruct_events(events, rec_cfg, workers=cfg.workers)
    reconstructed = EventSet(done)
    reconstructed.validate()
    write_events(
        cfg, reconstructed, ds.name, "reconstructed",
        {**manifest, "reconstruction": summarise(done), "failed_events": failed, "consistency": _consistency(done)},
    )
    near_miss = None
    if episodes:
        episode_set = EventSet(episodes)
        episode_set.validate()
        triggers = _count(t for ev in episodes for t in ev.meta["episode"]["triggers"])
        ttc = np.array([ev.meta["episode"]["min_ttc"] or np.inf for ev in episodes])
        near_miss = {
            **episode_set.summary(),
            "triggers": triggers,
            "with_cut_in": int(sum(ev.meta["episode"]["cut_in"] for ev in episodes)),
            "min_ttc_quantiles": {q: float(np.quantile(ttc[np.isfinite(ttc)], float(q))) for q in ("0.05", "0.5")}
            if np.isfinite(ttc).any() else None,
        }
        write_events(cfg, episode_set, ds.name + NEAR_MISS_SUFFIX, "raw", {**manifest, "near_miss": near_miss})
    return {
        "dataset": ds.name,
        "config_hash": manifest["config_hash"],
        "raw": raw.summary(),
        "reconstructed": reconstructed.summary(),
        "near_miss": near_miss,
        "n_failed_reconstruction": len(failed),
        "regimes": manifest["regimes"],
        "sites": manifest["sites"],
        "events_per_follower": manifest["events_per_follower"],
        "reconstruction": summarise(done),
        "seconds": round(time.time() - started, 1),
    }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def _per_follower(events: Sequence[Event]) -> dict:
    counts = np.array(list(_count(ev.follower_id for ev in events).values()))
    return {
        "mean": float(counts.mean()),
        "max": int(counts.max()),
        "share_followers_with_several_events": float(np.mean(counts > 1)),
        "share_events_of_followers_with_several_events": float(counts[counts > 1].sum() / counts.sum()),
    }


def _consistency(events: Sequence[Event]) -> dict:
    res = [consistency_residual(ev) for ev in events]
    return {k: float(np.sqrt(np.mean([r[k] ** 2 for r in res]))) for k in res[0]}


def load_events(cfg, version: str | None = None) -> EventSet:
    version = version or cfg.version
    return EventSet.from_parquet(events_dir(resolve_path(cfg.paths.events), cfg.dataset.name, version))


def split_path(cfg, kind: str) -> Path:
    return resolve_path(cfg.paths.splits) / f"{cfg.dataset.name}_{kind}.json"


def build_splits(cfg) -> dict:
    """Splits are defined on the raw version and shared by both versions."""
    table = event_table(load_events(cfg, "raw"))
    out = {}
    kinds = list(cfg.splits.kinds)
    if table["legacy_split"].notna().all():
        kinds.append("legacy")
    for kind in kinds:
        try:
            split = make_legacy_split(table, cfg.dataset.name) if kind == "legacy" else make_split(
                table, kind, cfg.splits.n_folds, cfg.seed, cfg.splits.val_fraction, cfg.dataset.name,
                max_unused=cfg.splits.max_unused, block_seconds=cfg.splits.block_seconds,
            )
        except ValueError as err:
            out[kind] = {"skipped": str(err)}
            continue
        report = check_leakage(split, table)
        split.save(split_path(cfg, kind))
        out[kind] = {
            "n_folds": split.n_folds,
            "driver_disjoint": split.driver_disjoint if kind in DRIVER_DISJOINT else False,
            "notes": split.notes,
            **report["folds"]["0"],
        }
    write_json(resolve_path(cfg.paths.splits) / f"{cfg.dataset.name}_report.json", out)
    return out


def part_events(events: EventSet, split: Split, fold: int, part: str) -> list[Event]:
    return [events[e] for e in split.partition(fold)[part] if e in events]


def _sample(events: list[Event], n: int | None, seed: int) -> list[Event]:
    if n is None or len(events) <= n:
        return events
    index = np.sort(np.random.default_rng([seed, 5]).choice(len(events), size=n, replace=False))
    return [events[i] for i in index]


def _usable(events: Sequence[Event], warmup: int) -> list[Event]:
    return [ev for ev in events if len(ev) >= warmup + 2]


def _checklist(cfg, version: str, split: str, events: EventSet, **more) -> dict:
    methods = {ev.meta.get("reconstruction", {}).get("method") for ev in events} - {None}
    return {
        "data_version": version,
        "reconstruction_method": sorted(methods)[0] if methods else ("none" if version == "raw" else None),
        "split_type": split,
        "seeds": {"global": cfg.seed},
        **more,
    }


def run_p0_baselines(cfg) -> list[dict]:
    """P0 for persistence, constant speed and the recorded acceleration, on the test parts."""
    fold = cfg.calibration.fold
    device = _device(cfg.calibration.device)
    rows = []
    for version in DATA_VERSIONS:
        events = load_events(cfg, version)
        for kind in [*cfg.splits.kinds, "legacy"]:
            if not split_path(cfg, kind).exists():
                continue
            split = Split.load(split_path(cfg, kind))
            test = _usable(part_events(events, split, min(fold, split.n_folds - 1), "test"), cfg.protocol.warmup)
            jerk = np.concatenate([np.diff(ev.a)[cfg.protocol.warmup - 1 : len(ev) - 2] for ev in test])
            recorded = np.concatenate([ev.a[cfg.protocol.warmup - 1 : len(ev) - 2] for ev in test])
            clip = tuple(cfg.protocol.clip)
            for model in (PersistenceBaseline(), ZeroAcceleration(), RecordedAcceleration()):
                table = evaluate_p0(model, test, cfg.protocol.warmup, device=device, clip=clip)
                metrics = {**pooled_rmse(table), "n_events": len(test), "n_samples": int(table["n"].sum())}
                if isinstance(model, PersistenceBaseline):
                    metrics["rms_jerk_dt"] = float(np.sqrt(np.mean(jerk**2)))
                    metrics["identity_error"] = abs(metrics["rms_jerk_dt"] - metrics["rmse_a"])
                    # the identity is exact where the recorded accelerations lie inside the clip
                    metrics["share_recorded_outside_clip"] = float(np.mean((recorded < clip[0]) | (recorded > clip[1])))
                write_result(
                    cfg.paths.results,
                    protocol="p0",
                    model=model.name,
                    family=model.family,
                    dataset=cfg.dataset.name,
                    data_version=version,
                    split=kind,
                    fold=fold,
                    part="test",
                    metrics=metrics,
                    config=run_config(cfg),
                    checklist=_checklist(cfg, version, kind, events),
                    per_event=table,
                )
                rows.append({"version": version, "split": kind, "model": model.name, **metrics})
    return rows


def run_calibration(cfg) -> list[dict]:
    """Per-event calibration (parameter spread) and global calibration on the training parts."""
    cal = calibration_config(cfg)
    fold = cfg.calibration.fold
    key = config_hash(run_config(cfg))
    rows = []
    for version in DATA_VERSIONS:
        events = load_events(cfg, version)
        out_dir = resolve_path(cfg.paths.calibration) / cfg.dataset.name / version
        out_dir.mkdir(parents=True, exist_ok=True)
        usable = _usable(list(events), cal.warmup)
        for name in cfg.calibration.models:
            model_cls = MECHANISTIC[name]
            started = time.time()
            row_file = out_dir / f"{name}_per_event.row.json"
            row = _resume(row_file, key)
            if row is None:
                sample = _sample(usable, cfg.calibration.per_event_sample, cfg.seed)
                table = calibrate_per_event(model_cls, sample, cal)
                table.to_parquet(out_dir / f"{name}_per_event.parquet", index=False)
                spread = parameter_spread(table, model_cls, cal)
                spread.update(data_version=version, dataset=cfg.dataset.name, config_hash=key)
                write_json(out_dir / f"{name}_spread.json", spread)
                row = {
                    "version": version, "model": name, "kind": "per_event", "n_events": len(sample),
                    "rmse_s": spread["rmse_s"]["median"], "rmse_v": spread["rmse_v"]["median"],
                    "collision_rate": spread["n_collided"] / max(1, spread["n_events"]),
                    "params": {p: s["median"] for p, s in spread["parameters"].items()},
                    "cv": {p: s["cv"] for p, s in spread["parameters"].items()},
                    "seconds": round(time.time() - started, 1),
                }
                write_json(row_file, {"config_hash": key, "row": row})
            rows.append(row)
            for kind in cfg.calibration.splits:
                if not split_path(cfg, kind).exists():
                    continue
                row_file = out_dir / f"{name}_global_{kind}-fold{fold}.row.json"
                row = _resume(row_file, key)
                if row is not None:
                    rows.append(row)
                    continue
                split = Split.load(split_path(cfg, kind))
                started = time.time()
                train = _sample(_usable(part_events(events, split, fold, "train"), cal.warmup), cfg.calibration.global_sample, cfg.seed)
                fit = calibrate_global(model_cls, train, cal)
                test = _usable(part_events(events, split, fold, "test"), cal.warmup)
                theta = [fit["params"][p] for p in model_cls.param_names]
                scores = score_events(model_cls, test, theta, cal)
                n = scores["n_samples"].sum()
                metrics = {
                    "rmse_s": float(np.sqrt((scores["rmse_s"] ** 2 * scores["n_samples"]).sum() / n)),
                    "rmse_v": float(np.sqrt((scores["rmse_v"] ** 2 * scores["n_samples"]).sum() / n)),
                    "collision_rate": float(scores["collided"].mean()),
                    "n_events": len(test),
                    "n_samples": int(n),
                }
                fit.update(data_version=version, dataset=cfg.dataset.name, split={"kind": kind, "fold": fold},
                           test=metrics, config_hash=key)
                write_json(out_dir / f"{name}_global_{kind}-fold{fold}.json", fit)
                write_result(
                    cfg.paths.results,
                    protocol="p1_whole_event",
                    model=name,
                    family="mechanistic",
                    dataset=cfg.dataset.name,
                    data_version=version,
                    split=kind,
                    fold=fold,
                    part="test",
                    metrics=metrics,
                    config=run_config(cfg),
                    checklist=_checklist(
                        cfg, version, kind, events,
                        mechanistic_calibration_protocol="Punzo-Zheng-Montanino 2021: closed loop, NRMSE(s)+NRMSE(v), "
                        "differential evolution, global parameters from the training part",
                        closed_loop_horizon="whole event",
                    ),
                    per_event=scores,
                    extra={"params": fit["params"], "train": {k: fit[k] for k in ("objective", "rmse_s", "rmse_v", "collision_rate", "n_events")}},
                )
                row = {
                    "version": version, "model": name, "kind": f"global/{kind}", "n_events": len(train),
                    "train_rmse_s": fit["rmse_s"], "train_rmse_v": fit["rmse_v"],
                    "rmse_s": metrics["rmse_s"], "rmse_v": metrics["rmse_v"],
                    "collision_rate": metrics["collision_rate"], "params": fit["params"],
                    "generations": fit["n_generations"], "seconds": round(time.time() - started, 1),
                }
                write_json(row_file, {"config_hash": key, "row": row})
                rows.append(row)
    write_json(resolve_path(cfg.paths.calibration) / cfg.dataset.name / "summary.json", rows)
    return rows
