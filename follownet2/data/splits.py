"""Splits: driver, site and regime level, plus the legacy random event-level split.

A split deals *groups* to folds (superset of the file format of cf-stability):

============  =========================  ===========================================
kind          group of an event          what the test fold measures
============  =========================  ===========================================
``event``     the event itself           legacy protocol; drivers leak into the test
``driver``    ``follower_id``            generalisation to unseen drivers
``site``      ``<dataset>/<site>``       generalisation to unseen sites
``regime``    free / steady / congested  generalisation to an unseen traffic regime
``block``     time block of a recording  generalisation to traffic that was not seen
============  =========================  ===========================================

For fold ``k`` the test part is fold ``k``. The rest is the development part, from which
the validation part is held out: by follower (kinds driver, site, regime) or by event
(kind event, which is leaky on purpose). Groups are shuffled with
``numpy.random.default_rng(seed)`` and dealt greedily so that folds are balanced in
samples. With fewer groups than folds the split is leave-one-group-out.

A regime split would inherit the driver leak of the event-level split, because one driver
is met in several regimes. The events of the development part whose follower also drives in
the test part are therefore left out of that fold (part ``unused``), unless this would
remove more than ``max_unused`` of the development samples; the split then stays
driver-overlapping and says so in its notes.

The driver split does not separate platoons: the leader of a test event is often the follower
of a training event, so its trajectory, the input of the model, was a training target. The
block split cuts every recording into blocks of ``block_seconds`` by the start time of the
events and leaves out of the development part every event that shares a vehicle, as follower
or as leader, with the test part.

The paired splits of experiment E2 (kinds ``paired-clean`` and ``paired-leaky``, built by
:mod:`follownet2.experiments.paired` from a driver split) are not dealt: they carry explicit
per-fold parts (``Split.explicit``), which :meth:`Split.partition` returns as they are. Their
groups are followers, so the leakage check treats them like a driver split: the clean arm
passes, the leaky arm fails by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from follownet2.utils import read_json, write_json

KINDS = ("event", "driver", "site", "regime", "block")
LEGACY = "legacy"  # the split published with a dataset, kept in meta["legacy_split"]
PARTS = ("train", "val", "test")
# kinds whose development part must not contain a follower of the test part
DRIVER_DISJOINT = ("driver", "site", "regime", "block")
# kinds whose development part must not contain any vehicle of the test part, in either role
VEHICLE_DISJOINT = ("block",)
# paired splits of E2 (decision R2): explicit parts derived from a driver split, groups are followers
PAIRED = ("paired-clean", "paired-leaky")


class LeakageError(AssertionError):
    pass


def group_key(frame: pd.DataFrame, kind: str, block_seconds: float | None = None) -> pd.Series:
    if kind in ("event", LEGACY):
        return frame["event_id"].astype(str)
    if kind == "block":
        if block_seconds is None or "group" not in frame.columns or frame["group"].isna().any():
            raise ValueError("block split needs block_seconds and the recording (group) of every event")
        block = np.floor(frame["t0"].astype(np.float64) / block_seconds).astype(np.int64).astype(str)
        return frame["dataset"].astype(str) + "/" + frame["site"].astype(str) + "/" + frame["group"].astype(str) + "/b" + block
    if kind == "driver" or kind in PAIRED:
        return frame["follower_id"].astype(str)
    if kind == "site":
        return frame["dataset"].astype(str) + "/" + frame["site"].astype(str)
    if kind == "regime":
        if "regime" not in frame.columns or frame["regime"].isna().any():
            raise ValueError("regime split needs a regime label for every event")
        return frame["regime"].astype(str)
    raise ValueError(f"unknown split kind {kind!r}")


def _deal(groups: pd.Series, weights: pd.Series, n_folds: int, rng: np.random.Generator) -> dict[str, int]:
    """Deal groups to folds: random order, heaviest first, always to the lightest fold."""
    size = weights.groupby(groups.to_numpy(), sort=True).sum()
    names = size.index.to_numpy()
    order = rng.permutation(len(names))
    order = order[np.argsort(-size.to_numpy()[order], kind="stable")]
    load = np.zeros(n_folds)
    out = {}
    for i in order:
        fold = int(np.argmin(load))
        out[str(names[i])] = fold
        load[fold] += size.iloc[i]
    return out


def _hold_out(groups: np.ndarray, weights: np.ndarray, fraction: float, rng: np.random.Generator) -> set[str]:
    """Random groups holding at least ``fraction`` of the weight (never all groups)."""
    size = pd.Series(weights).groupby(groups).sum()
    names = size.index.to_numpy()
    if len(names) < 2 or fraction <= 0:
        return set()
    order = rng.permutation(len(names))
    target = fraction * size.sum()
    cumulative = np.cumsum(size.to_numpy()[order])
    n_take = min(int(np.searchsorted(cumulative, target, side="left")) + 1, len(names) - 1)
    return {str(n) for n in names[order[:n_take]]}


@dataclass
class Split:
    dataset: str
    kind: str
    seed: int
    n_folds: int
    groups: dict[str, int]  # group -> test fold
    folds: dict[str, int]  # event_id -> test fold
    val: dict[str, list[str]]  # fold -> event ids of the validation part
    val_fraction: float
    notes: list[str] = field(default_factory=list)
    unused: dict[str, list[str]] = field(default_factory=dict)  # fold -> event ids left out
    driver_disjoint: bool = True
    block_seconds: float | None = None
    # fold -> {"train", "val", "test"} (and "unused" when not empty): parts given explicitly
    # (paired splits of E2); they take precedence over folds, val and unused. Files written
    # before this field existed load with the default.
    explicit: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def partition(self, fold: int = 0) -> dict[str, list[str]]:
        """Event ids of the parts ``train``, ``val``, ``test`` (and ``unused`` when not empty).

        A fold with explicit parts returns a copy of them.
        """
        if not 0 <= fold < self.n_folds:
            raise ValueError(f"fold {fold} outside 0..{self.n_folds - 1}")
        given = self.explicit.get(str(fold))
        if given is not None:
            parts = {name: list(given.get(name, ())) for name in PARTS}
            parts.update({name: list(ids) for name, ids in given.items() if name not in PARTS})
            return parts
        val = set(self.val[str(fold)])
        unused = set(self.unused.get(str(fold), ()))
        parts: dict[str, list[str]] = {name: [] for name in PARTS}
        if unused:
            parts["unused"] = []
        for event_id, test_fold in self.folds.items():
            if test_fold == fold:
                parts["test"].append(event_id)
            elif event_id in unused:
                parts["unused"].append(event_id)
            elif event_id in val:
                parts["val"].append(event_id)
            else:
                parts["train"].append(event_id)
        return parts

    def assignment(self, fold: int = 0) -> dict[str, str]:
        return {event_id: part for part, ids in self.partition(fold).items() for event_id in ids}

    def save(self, path: str | Path) -> Path:
        payload = dict(self.__dict__)
        if not payload["explicit"]:
            del payload["explicit"]  # the files of the dealt kinds keep their format
        return write_json(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "Split":
        return cls(**read_json(path))


def make_split(
    frame: pd.DataFrame,
    kind: str,
    n_folds: int = 5,
    seed: int = 0,
    val_fraction: float = 0.1875,
    dataset: str | None = None,
    max_unused: float = 0.5,
    block_seconds: float = 300.0,
) -> Split:
    """Build a split from the event table (:func:`event_table`).

    ``val_fraction`` is the share of the development samples held out for validation
    (0.1875 of 80 % gives 65 / 15 / 20 for five folds).
    """
    if kind in PAIRED:
        raise ValueError(f"a {kind} split is derived from a driver split: follownet2.experiments.paired")
    if frame["event_id"].duplicated().any():
        raise ValueError("duplicate event ids")
    frame = frame.sort_values("event_id", kind="stable").reset_index(drop=True)
    groups = group_key(frame, kind, block_seconds)
    weights = frame["n_samples"].astype(np.float64)
    notes = []
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError(f"a {kind} split needs at least 2 groups, found {n_groups}")
    if n_groups < n_folds:
        notes.append(f"only {n_groups} groups: {n_folds} folds reduced to leave-one-group-out")
        n_folds = n_groups
    rng = np.random.default_rng(seed)
    dealt = _deal(groups, weights, n_folds, rng)
    test_fold = groups.map(dealt).to_numpy()

    holder = frame["event_id"].astype(str) if kind == "event" else frame["follower_id"].astype(str)
    follower = frame["follower_id"].astype(str).to_numpy()
    leader = frame["leader_id"].astype(str).to_numpy()
    size = weights.to_numpy()
    seen = []
    for f in range(n_folds):
        test = test_fold == f
        if kind in VEHICLE_DISJOINT:
            vehicles = np.union1d(follower[test], leader[test])
            shared = np.isin(follower, vehicles) | np.isin(leader, vehicles)
        else:
            shared = np.isin(follower, follower[test])
        seen.append(shared & ~test)
    lost = max(size[s].sum() / max(size[test_fold != f].sum(), 1.0) for f, s in enumerate(seen))
    driver_disjoint = bool(kind in DRIVER_DISJOINT and lost <= max_unused)
    if kind in DRIVER_DISJOINT and not driver_disjoint:
        notes.append(
            f"not driver-disjoint: leaving out the followers of the test part would remove "
            f"{lost:.0%} of the development samples"
        )
    val, unused = {}, {}
    for fold in range(n_folds):
        dev = test_fold != fold
        if driver_disjoint and seen[fold].any():
            unused[str(fold)] = frame["event_id"][seen[fold]].astype(str).tolist()
            dev = dev & ~seen[fold]
        fold_rng = np.random.default_rng([seed, 1000 + fold])
        held = _hold_out(holder.to_numpy()[dev], weights.to_numpy()[dev], val_fraction, fold_rng)
        val[str(fold)] = frame["event_id"][dev & holder.isin(held).to_numpy()].astype(str).tolist()
    name = dataset or "+".join(sorted(frame["dataset"].astype(str).unique()))
    return Split(
        dataset=name,
        kind=kind,
        seed=seed,
        n_folds=n_folds,
        groups=dealt,
        folds=dict(zip(frame["event_id"].astype(str), (int(f) for f in test_fold))),
        val=val,
        val_fraction=val_fraction,
        notes=notes,
        unused=unused,
        driver_disjoint=driver_disjoint,
        block_seconds=block_seconds if kind == "block" else None,
    )


def make_legacy_split(frame: pd.DataFrame, dataset: str | None = None) -> Split:
    """The published split of a dataset (column ``legacy_split``: train, val or test).

    One fold. FollowNet publishes a random split by event without any id, so nothing can be
    said about drivers: the split is treated like the event-level split.
    """
    labels = frame["legacy_split"].astype(str)
    unknown = sorted(set(labels) - set(PARTS))
    if unknown:
        raise ValueError(f"legacy split labels {unknown} are not among {PARTS}")
    ids = frame["event_id"].astype(str)
    return Split(
        dataset=dataset or "+".join(sorted(frame["dataset"].astype(str).unique())),
        kind=LEGACY,
        seed=-1,
        n_folds=1,
        groups={},
        folds={e: 0 if part == "test" else -1 for e, part in zip(ids, labels)},
        val={"0": ids[(labels == "val").to_numpy()].tolist()},
        val_fraction=float((frame["n_samples"][labels == "val"].sum()) / max(frame["n_samples"][labels != "test"].sum(), 1)),
        notes=["published split, by event, without ids"],
        driver_disjoint=False,
    )


def check_leakage(split: Split, frame: pd.DataFrame, fold: int | None = None) -> dict:
    """Fail if a group of the split kind occurs on both sides of a fold.

    For a driver-level split this is the requirement of the protocol: no ``follower_id`` in
    both the training (or validation) part and the test part. The same holds for every
    split that declares itself driver-disjoint (site, and regime where possible). Kinds
    driver, site and regime additionally require the validation part to share no follower
    with the training part. The event-level split passes by construction and only reports
    how many test followers also drive in the training part. The paired splits of E2 are
    checked like a driver split: the clean arm passes, the leaky arm, whose training part
    holds events of the test drivers, fails by design.

    Returns diagnostics, including the share of test events whose *leader* is a follower
    of the training part (the leader trajectory is an input, not a target, so this is
    reported and not forbidden).
    """
    frame = frame.assign(event_id=frame["event_id"].astype(str)).set_index("event_id")
    missing = set(frame.index) ^ set(split.folds)
    if missing:
        raise LeakageError(f"split and event table disagree on {len(missing)} event ids")
    keys = group_key(frame.reset_index(), split.kind, split.block_seconds).set_axis(frame.index)
    report = {"kind": split.kind, "folds": {}}
    for k in range(split.n_folds) if fold is None else [fold]:
        parts = split.partition(k)
        ids = {name: pd.Index(parts[name]) for name in PARTS}
        n_assigned = sum(len(v) for v in parts.values())
        if n_assigned != len(frame) or len(set().union(*parts.values())) != n_assigned:
            raise LeakageError(f"fold {k}: the parts are not a partition of the events")
        dev_groups = set(keys.loc[ids["train"]]) | set(keys.loc[ids["val"]])
        shared = dev_groups & set(keys.loc[ids["test"]])
        strict = split.kind in DRIVER_DISJOINT or split.kind in PAIRED
        if strict and shared:
            raise LeakageError(f"fold {k}: {len(shared)} {split.kind} groups on both sides, e.g. {sorted(shared)[:3]}")
        followers = {name: set(frame.loc[ids[name], "follower_id"]) for name in PARTS}
        if strict and followers["train"] & followers["val"]:
            raise LeakageError(f"fold {k}: followers shared by the training and validation parts")
        test_followers = followers["test"]
        seen = test_followers & (followers["train"] | followers["val"])
        if (split.kind == "driver" or (strict and split.driver_disjoint)) and seen:
            raise LeakageError(f"fold {k}: {len(seen)} follower ids in train and test, e.g. {sorted(seen)[:3]}")
        test_leaders = frame.loc[ids["test"], "leader_id"]
        if split.kind in VEHICLE_DISJOINT and split.driver_disjoint:
            dev = ids["train"].append(ids["val"])
            on_test = set(frame.loc[ids["test"], "follower_id"]) | set(test_leaders)
            on_dev = set(frame.loc[dev, "follower_id"]) | set(frame.loc[dev, "leader_id"])
            if on_test & on_dev:
                raise LeakageError(f"fold {k}: {len(on_test & on_dev)} vehicles on both sides")
        report["folds"][str(k)] = {
            "n_train": len(ids["train"]),
            "n_val": len(ids["val"]),
            "n_test": len(ids["test"]),
            "n_unused": len(parts.get("unused", ())),
            "share_samples_test": float(frame.loc[ids["test"], "n_samples"].sum() / frame["n_samples"].sum()),
            "share_samples_val": float(frame.loc[ids["val"], "n_samples"].sum() / frame["n_samples"].sum()),
            "test_followers_seen_in_training": len(seen),
            "share_test_followers_seen": len(seen) / max(1, len(test_followers)),
            "share_test_events_with_training_follower_as_leader": float(
                test_leaders.isin(followers["train"] | followers["val"]).mean()
            ),
        }
    return report


def event_table(events) -> pd.DataFrame:
    """Event table of an EventSet with the regime column, as used by the split functions."""
    frame = events.events_frame().drop(columns=["meta_json"])
    frame["regime"] = [ev.meta.get("regime") for ev in events]
    frame["legacy_split"] = [ev.meta.get("legacy_split") for ev in events]
    frame["group"] = [ev.meta.get("group") for ev in events]
    frame["t0"] = [ev.meta.get("t0", 0.0) for ev in events]
    return frame
