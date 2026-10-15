"""Paired splits of experiment E2 (H5.2, decision R2; docs/experiments.md, section 2).

For fold ``k`` of the driver split the events of every test driver with at least two events
are shuffled and halved: the first ``n // 2`` are *leak* events, the rest *probe* events; a
driver with a single event gives a probe event only. Both arms are scored on the probe events
and share the validation part of the driver split. The clean arm trains on the training part
of the driver split; the leaky arm replaces a uniformly drawn subset of it, of the size of the
leak set, by the leak events, so that both arms train on the same number of events.

One generator, ``numpy.random.default_rng([seed, fold])`` with the seed of the driver split,
draws the halving (test drivers and their events in sorted order) and then the replaced
events, so the arms depend on the driver split and the fold only.

The events an arm neither trains on nor scores (the leak events in the clean arm, the replaced
events in the leaky arm) form its part ``unused``: the explicit parts of every fold are then a
partition of the events, as :func:`~trafficloop.data.splits.check_leakage` requires.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trafficloop import pipeline
from trafficloop.data.splits import Split, check_leakage, event_table
from trafficloop.utils import write_json

ARMS = {"clean": "paired-clean", "leaky": "paired-leaky"}  # arm -> split kind
STATS = ("n_probe", "n_leak", "n_test_drivers", "n_replaced", "share_probe_with_leak", "n_train")
VERSION = "reconstructed"  # data version of the events the script reads (docs/experiments.md, section 1)


def _followers(frame: pd.DataFrame) -> dict[str, str]:
    """event id -> follower id, from an event table indexed by event id (or with an event_id column)."""
    if "event_id" in frame.columns:
        frame = frame.set_index("event_id")
    return dict(zip(frame.index.astype(str), frame["follower_id"].astype(str)))


def paired_partitions(split: Split, frame: pd.DataFrame, fold: int) -> dict:
    """Parts of both arms in one fold of a driver split, and the statistics of the fold.

    ``frame`` is :func:`~trafficloop.data.splits.event_table` indexed by event id. Returns
    ``{"clean": parts, "leaky": parts, "stats": stats}``: ``parts`` maps ``train``, ``val``
    and ``test`` to sorted event ids, ``stats`` holds :data:`STATS`.
    """
    if split.kind != "driver":
        raise ValueError(f"paired splits are built from a driver split, not from a {split.kind} split")
    follower = _followers(frame)
    parts = split.partition(fold)
    missing = [e for ids in parts.values() for e in ids if e not in follower]
    if missing:
        raise ValueError(
            f"fold {fold}: {len(missing)} events of the split are not in the event table, e.g. {missing[:3]}"
        )
    rng = np.random.default_rng([split.seed, fold])
    drivers: dict[str, list[str]] = {}
    for event_id in sorted(parts["test"]):
        drivers.setdefault(follower[event_id], []).append(event_id)
    leak, probe, probe_with_leak = [], [], 0
    for driver in sorted(drivers):
        ids = drivers[driver]
        if len(ids) < 2:
            probe.extend(ids)
            continue
        shuffled = [ids[i] for i in rng.permutation(len(ids))]
        n_leak = len(ids) // 2
        leak.extend(shuffled[:n_leak])
        probe.extend(shuffled[n_leak:])
        probe_with_leak += len(ids) - n_leak
    train = sorted(parts["train"])
    if len(leak) > len(train):
        raise ValueError(f"fold {fold}: {len(leak)} leak events, but only {len(train)} training events to replace")
    replaced = {train[i] for i in rng.choice(len(train), size=len(leak), replace=False)} if leak else set()
    val, probe = sorted(parts["val"]), sorted(probe)
    stats = {
        "n_probe": len(probe),
        "n_leak": len(leak),
        "n_test_drivers": len(drivers),
        "n_replaced": len(replaced),
        "share_probe_with_leak": probe_with_leak / len(probe) if probe else 0.0,
        "n_train": len(train),
    }
    return {
        "clean": {"train": train, "val": val, "test": probe},
        "leaky": {"train": sorted((set(train) - replaced) | set(leak)), "val": list(val), "test": list(probe)},
        "stats": stats,
    }


def _note(fold: str, stats: dict) -> str:
    def text(value) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    return f"fold {fold}: " + " ".join(f"{name}={text(stats[name])}" for name in STATS)


def _build(split: Split, frame: pd.DataFrame) -> tuple[Split, Split, dict[str, dict]]:
    """Both arms and the statistics of every fold."""
    explicit: dict[str, dict[str, dict[str, list[str]]]] = {arm: {} for arm in ARMS}
    stats: dict[str, dict] = {}
    events = set(split.folds)
    for fold in range(split.n_folds):
        out = paired_partitions(split, frame, fold)
        stats[str(fold)] = out["stats"]
        for arm in ARMS:
            parts = out[arm]
            unused = sorted(events.difference(*parts.values()))
            explicit[arm][str(fold)] = {**parts, "unused": unused} if unused else dict(parts)
    notes = [_note(fold, s) for fold, s in stats.items()]
    origin = f"arm of the driver split of {split.dataset} (seed {split.seed}), docs/experiments.md section 2"
    clean, leaky = (
        Split(
            dataset=split.dataset,
            kind=kind,
            seed=split.seed,
            n_folds=split.n_folds,
            groups=dict(split.groups),
            folds=dict(split.folds),
            val={fold: list(ids) for fold, ids in split.val.items()},
            val_fraction=split.val_fraction,
            notes=[f"{kind} {origin}", *notes],
            driver_disjoint=arm == "clean",
            block_seconds=split.block_seconds,
            explicit=explicit[arm],
        )
        for arm, kind in ARMS.items()
    )
    return clean, leaky, stats


def paired_splits(split: Split, frame: pd.DataFrame) -> tuple[Split, Split]:
    """The clean and the leaky arm of a driver split (kinds ``paired-clean``, ``paired-leaky``)."""
    clean, leaky, _ = _build(split, frame)
    return clean, leaky


def pooled(stats: dict[str, dict]) -> dict:
    """Statistics summed over the folds (the test drivers of the folds are disjoint)."""
    out = {name: sum(s[name] for s in stats.values()) for name in ("n_probe", "n_leak", "n_test_drivers", "n_replaced")}
    with_leak = sum(round(s["share_probe_with_leak"] * s["n_probe"]) for s in stats.values())
    out["share_probe_with_leak"] = with_leak / out["n_probe"] if out["n_probe"] else 0.0
    return out


def build_paired_splits(cfg) -> dict:
    """Write both arms of a dataset next to its driver split, check the clean arm, write the report."""
    table = event_table(pipeline.load_events(cfg, VERSION))
    driver_path = pipeline.split_path(cfg, "driver")
    driver = Split.load(driver_path)
    clean, leaky, stats = _build(driver, table.set_index("event_id"))
    leakage = check_leakage(clean, table)  # the leaky arm fails this check by design and is not checked
    files = {arm.kind: arm.save(pipeline.split_path(cfg, arm.kind)).name for arm in (clean, leaky)}
    report = {
        "dataset": cfg.dataset.name,
        "data_version": VERSION,
        "driver_split": driver_path.name,
        "seed": driver.seed,
        "n_folds": driver.n_folds,
        "files": files,
        "folds": stats,
        "pooled": pooled(stats),
        "leakage_clean": leakage["folds"],
    }
    write_json(driver_path.with_name(f"{cfg.dataset.name}_paired_report.json"), report)
    return report
