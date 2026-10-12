"""Analyses of milestone 4: experiments E1-E4 (docs/experiments.md, section 3).

Functions of the run directories under ``root`` (``results/models``),
``<root>/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/``, and of DataFrames; nothing is
written here (``scripts/m4_report.py`` writes ``results/m4``). A missing run, protocol file or
table leaves its rows out (or NaN) instead of failing, so the analyses run while campaigns are
still incomplete.

Bootstrap conventions (1000 replicates by default, 95 % percentile intervals):

* **Drivers** (E1, P0 and P1 rankings of E3, E4): the multinomial counts of the followers are
  drawn once per dataset (per target dataset in E4) with :func:`follownet2.stats.draw_counts`
  from ``numpy.random.default_rng(seed)``, over the sorted union of the followers of every table
  involved, and applied to every model's table on the fast pooled path of
  :mod:`follownet2.stats.bootstrap` (common random numbers across models, metrics and horizons).
  For a table whose followers are that union, the replicates are exactly those of
  ``bootstrap_groups(table, seed=seed)``.
* **E2**: :func:`follownet2.stats.paired_difference` (one draw over the probe drivers, shared by
  both arms) and :func:`follownet2.stats.holm` over the models of a dataset, metric and subset.
* **Profiles and speeds** (P2 rankings, P3 ranking, composite): one Generator seeded with
  ``[seed, 1]`` draws the in-range profile counts, then the speed counts, as
  :func:`~follownet2.protocols.composite.composite_bootstrap` does; the P3 unstable fraction of a
  replicate is taken over the drawn speeds with an equilibrium (converged and tracking), which
  reproduces ``unstable_fraction`` of ``p3.json``.
* **Synthetic P4 scenarios** (the in-range ones): Generator ``[seed, 2]``; **empirical P4**: the
  drivers of the near-miss episodes, Generator ``[seed, 3]``.

A per-window table is used only when it reproduces the summary of the JSON it belongs to
(events, windows and samples, pooled RMSE within 1 %): in milestone 3 the transfer evaluation
overwrote the own ``p1.<table>.parquet`` of every run (decision M4-5), and such tables are
recognised and left out. All metrics are lower-is-better; in a ranking a model without a value
ranks last (in every replicate).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from follownet2.protocols.composite import INPUTS, PROFILE_COLUMNS, _aligned, _mean_over, _sum_over, composite_score
from follownet2.protocols.p5 import collect_transfer, transfer_matrix
from follownet2.stats.bootstrap import _Resampler, draw_counts, percentile_ci, pooled_statistic
from follownet2.stats.paired import holm, paired_difference
from follownet2.stats.ranks import kendall_tau, kendall_tau_bootstrap
from follownet2.utils import read_json, resolve_path

VERSION = "reconstructed"
GROUP = "follower_id"
REFERENCE = "persistence"
#: the eight-model subset of E2 and E4 (decision M4-2); persistence is evaluated next to it
SUBSET = ("idm", "gipps", "mlp", "lstm", "mlp_loop", "lstm_loop", "ddpg", "stability_mlp")
MAIN_DATASETS = ("ngsim", "follownet_highd")
LEAK_DATASETS = ("ngsim", "openacc")
TRANSFER_DATASETS = ("ngsim", "follownet_highd", "openacc", "waymo_hu", "follownet_ngsim", "tgsim")
SEEDS = (0, 1, 2)
FOLDS = (0, 1, 2, 3, 4)
CLEAN, LEAKY = "paired-clean", "paired-leaky"
HORIZONS = ("h10", "h30", "whole")
#: relative tolerance of the pooled RMSE when a table is checked against its summary
RTOL = 1e-2

#: E2 metrics: name -> (table, statistic); "p0" is test_p0.events.parquet, "whole"
#: test_closed_loop.events.parquet, "h10" p1.h10.parquet
E2_METRICS = {"p0_rmse_a": ("p0", "rmse_a"), "p1_whole_rmse_s": ("whole", "rmse_s"), "p1_h10_rmse_s": ("h10", "rmse_s")}
E2_SUBSETS = ("all", "affected")
#: E3 rankings: name -> column of :func:`e3_rankings`
RANKINGS = {
    "P0": "p0_rmse_a",
    "P1 h10": "p1_h10_rmse_s",
    "P1 h30": "p1_h30_rmse_s",
    "P1 whole": "p1_whole_rmse_s",
    "P2 concave": "p2_concave_growth_error",
    "P2 hysteresis": "p2_hysteresis_error",
    "P2 collisions": "p2_collisions_100",
    "P3 unstable": "p3_unstable_fraction",
    "P4 synthetic": "p4_collision_rate_synthetic",
    "P4 empirical": "p4_collision_rate_empirical",
    "composite": "composite",
}
#: the rankings between which the entries into and exits from the top three are listed
TOP3_SEQUENCE = ("P0", "P1 whole", "composite")
#: composite input -> column of :func:`e3_rankings`
COMPOSITE_COLUMNS = {
    "concave_growth_error": "p2_concave_growth_error",
    "hysteresis_error": "p2_hysteresis_error",
    "unstable_fraction": "p3_unstable_fraction",
    "collisions_100": "p2_collisions_100",
}
#: E3 driver-bootstrap columns: column -> (table, statistic)
DRIVER_COLUMNS = {
    "p0_rmse_a": ("p0", "rmse_a"),
    **{f"p1_{h}_rmse_{x}": (h, f"rmse_{x}") for h in HORIZONS for x in ("s", "v")},
}

E1_COLUMNS = ["dataset", "model", "family", "rmse_a", "ratio", "low", "high", "n_drivers",
              "seed_min", "seed_max", "n_seeds", "folds_min", "folds_max", "n_folds"]
E2_COLUMNS = ["dataset", "subset", "model", "family", "metric", "clean", "leaky", "inflation", "mean_difference", "low",
              "high", "p_bootstrap", "p_wilcoxon", "p_holm", "n_drivers", "n_affected_drivers", "n_events", "n_folds"]
E2_CLASS_COLUMNS = ["dataset", "subset", "metric", "family", "mean_inflation", "min_inflation", "max_inflation",
                    "mean_difference", "n_models"]
E2_UNPAIRED_COLUMNS = ["dataset", "model", "family", "metric", "driver", "event", "inflation", "difference",
                       "n_events_driver", "n_events_event"]
E3_METRICS = ["p0_rmse_a", "p1_h10_rmse_s", "p1_h10_rmse_v", "p1_h30_rmse_s", "p1_h30_rmse_v", "p1_whole_rmse_s",
              "p1_whole_rmse_v", "p2_concave_growth_error", "p2_hysteresis_error", "p2_collisions_100",
              "p3_unstable_fraction", "p4_collision_rate_synthetic", "p4_collision_rate_empirical", "composite"]
E3_COLUMNS = ["dataset", "model", "family", *E3_METRICS, "stale"]
KENDALL_COLUMNS = ["dataset", "ranking_a", "ranking_b", "tau", "low", "high", "p", "n_models", "tau_boot", "note"]
SPEED_COLUMNS = ["dataset", "horizon", "tau", "low", "high", "p", "n_models", "tau_boot", "note"]
TOP3_COLUMNS = ["dataset", "ranking", "top1", "top2", "top3", "values", "tied"]
E4_COLUMNS = ["model", "trained_on", "tested_on", "horizon", "rmse_s", "rmse_s_own", "ratio", "low", "high",
              "n_drivers", "note"]
E4_SUMMARY_COLUMNS = ["model", "horizon", "mean_degradation", "n_pairs", "worst_source", "worst_target", "worst_ratio"]
E4_TARGET_COLUMNS = ["tested_on", "horizon", "mean_degradation", "n_pairs", "n_models"]
STATUS_COLUMNS = ["campaign", "dataset", "runs", "detail"]

_PROTOCOL_FILE = re.compile(r"^p\d(_.+)?\.json$")


# ----------------------------------------------------------------------------- runs and files


def run_dir(root, dataset: str, label: str, split: str = "driver", fold: int = 0, seed: int = 0,
            version: str = VERSION) -> Path:
    """Directory of one run."""
    return Path(root) / dataset / version / f"{split}-fold{fold}" / label / f"seed{seed}"


def run_labels(root, dataset: str, split: str = "driver", fold: int = 0, seed: int = 0, version: str = VERSION) -> list[str]:
    """Sorted labels with a finished run (``train.json``)."""
    base = Path(root) / dataset / version / f"{split}-fold{fold}"
    return sorted(path.parent.parent.name for path in base.glob(f"*/seed{seed}/train.json"))


def run_datasets(root, split: str = "driver", fold: int = 0, seed: int = 0, version: str = VERSION) -> list[str]:
    """Sorted datasets with at least one finished run of the split, fold and seed."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and run_labels(root, p.name, split, fold, seed, version))


def _json(path: Path) -> dict | None:
    return read_json(path) if path.exists() else None


def _summary(directory: Path, name: str) -> dict | None:
    data = _json(directory / f"{name}.json")
    return None if data is None else data.get("summary")


def _num(mapping, *keys) -> float:
    """Nested value as a float; NaN when a key is missing or the value is null."""
    value = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return float("nan")
        value = value[key]
    try:
        return float("nan") if value is None else float(value)
    except (TypeError, ValueError):
        return float("nan")


def _family(directory: Path) -> str | None:
    train = _json(directory / "train.json")
    return None if train is None else train.get("family")


def _read(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame | None:
    return pd.read_parquet(path, columns=list(columns) if columns else None) if path.exists() else None


def stale_protocols(directory: Path) -> list[str]:
    """Protocol files of a run computed from another checkpoint than the current one: the
    provenance hash of their model differs from the configuration hash of ``train.json``."""
    train = _json(Path(directory) / "train.json")
    key = None if train is None else train.get("config_hash")
    out = []
    for path in sorted(Path(directory).glob("p*.json")):
        if not _PROTOCOL_FILE.match(path.name):
            continue
        provenance = (read_json(path).get("model") or {}).get("provenance") or {}
        if key and provenance.get("config_hash") and provenance["config_hash"] != key:
            out.append(path.stem)
    return out


def check_table(table: pd.DataFrame | None, summary: Mapping | None, statistic: str = "rmse_s", rtol: float = RTOL) -> str | None:
    """Why ``table`` does not reproduce ``summary`` (a P1 horizon block or a whole-event block), or None.

    Compared when the summary has them: ``n_events`` (distinct ``event_id``), ``n_windows``
    (rows), ``n_samples`` (sum of ``n``) and the pooled ``statistic`` (relative ``rtol``).
    """
    if table is None:
        return "no table"
    if not isinstance(summary, Mapping):
        return "no summary to check the table against"
    for key, value in (("n_events", lambda t: t["event_id"].nunique()), ("n_windows", len),
                       ("n_samples", lambda t: int(round(float(t["n"].sum()))))):
        expected = _num(summary, key)
        if np.isfinite(expected) and value(table) != int(expected):
            return f"{key} {value(table)} in the table, {int(expected)} in the summary"
    expected = _num(summary, statistic)
    if np.isfinite(expected) and len(table):
        value = pooled_statistic(table, [_source(statistic)])[statistic]
        if not abs(value - expected) <= rtol * abs(expected) + 1e-12:
            return f"pooled {statistic} {value:.6g} in the table, {expected:.6g} in the summary"
    return None


def own_p1_table(directory: Path, horizon: str, fallback: bool = True) -> tuple[pd.DataFrame | None, str | None]:
    """The P1 table of a run at ``horizon`` on its own test part, or ``(None, reason)``.

    ``p1.<horizon>.parquet`` when it reproduces ``p1.json``; for the whole event (``fallback``)
    otherwise ``test_closed_loop.events.parquet`` of the training run, checked against the
    whole-event block of ``p1.json`` (of ``train.json`` without ``p1.json``).
    """
    directory = Path(directory)
    summary = _summary(directory, "p1")
    block = None if summary is None else summary.get(horizon)
    table = _read(directory / f"p1.{horizon}.parquet")
    problem = check_table(table, block)
    if problem is None:
        return table, None
    if horizon == "whole" and fallback:
        if block is None:
            block = ((_json(directory / "train.json") or {}).get("test") or {}).get("closed_loop_whole_event")
        closed = _read(directory / "test_closed_loop.events.parquet")
        if check_table(closed, block) is None:
            return closed, None
    return None, f"p1.{horizon}.parquet: {problem}"


def transfer_table(directory: Path, tested_on: str, horizon: str) -> tuple[pd.DataFrame | None, str | None]:
    """The per-window table of the transfer evaluation of a run on ``tested_on``
    (``p1_<tested_on>.<horizon>.parquet``, checked against ``p1_<tested_on>.json``), or ``(None, reason)``."""
    directory = Path(directory)
    summary = _summary(directory, f"p1_{tested_on}")
    table = _read(directory / f"p1_{tested_on}.{horizon}.parquet")
    problem = check_table(table, None if summary is None else summary.get(horizon))
    return (table, None) if problem is None else (None, f"p1_{tested_on}.{horizon}.parquet: {problem}")


def _training_table(directory: Path, kind: str) -> tuple[pd.DataFrame | None, str | None]:
    """The per-event tables of the training run (``p0``: one step, ``whole``: closed loop) checked
    against ``train.json``."""
    directory = Path(directory)
    test = ((_json(directory / "train.json") or {}).get("test")) or {}
    if kind == "p0":
        table, block, statistic = _read(directory / "test_p0.events.parquet"), test.get("p0"), "rmse_a"
    else:
        table, block, statistic = _read(directory / "test_closed_loop.events.parquet"), test.get("closed_loop_whole_event"), "rmse_s"
    problem = check_table(table, block, statistic)
    name = "test_p0" if kind == "p0" else "test_closed_loop"
    return (table, None) if problem is None else (None, f"{name}.events.parquet: {problem}")


# ----------------------------------------------------------------------------- driver bootstrap


def _source(statistic: str) -> str:
    """Column of a per-event table behind a statistic of the default bootstrap statistic."""
    if statistic.startswith("rmse_"):
        return "sse_" + statistic[len("rmse_"):]
    if statistic == "collision_rate":
        return "collided"
    if statistic.endswith("_rate"):
        return statistic[: -len("_rate")]
    raise ValueError(f"no table column behind the statistic {statistic!r}")


def common_draw(tables: Iterable[pd.DataFrame | None], n_boot: int, seed, group: str = GROUP) -> tuple[pd.Index, np.ndarray]:
    """The sorted union of the groups of ``tables`` and one count matrix ``[n_boot, groups]`` over
    it, drawn with :func:`~follownet2.stats.draw_counts` from ``numpy.random.default_rng(seed)``."""
    labels = [t[group].astype(str) for t in tables if t is not None and len(t)]
    if not labels:
        raise ValueError("nothing to resample: no table with rows")
    universe = pd.Index(pd.concat(labels, ignore_index=True).unique()).sort_values()
    return universe, draw_counts(np.random.default_rng(seed), len(universe), n_boot)


def group_replicates(table: pd.DataFrame, statistic: str, universe: pd.Index, counts: np.ndarray, group: str = GROUP) -> np.ndarray:
    """``statistic`` (``rmse_<x>`` or ``collision_rate``) of ``table`` in every replicate of
    ``counts`` (the fast pooled path; groups of the universe without rows contribute nothing)."""
    table = table.assign(**{group: table[group].astype(str)})
    sampler = _Resampler(table, group, None, [_source(statistic)], universe)
    return sampler.apply(counts)[statistic].to_numpy(dtype=np.float64)


def _point(table: pd.DataFrame, statistic: str) -> float:
    return float(pooled_statistic(table, [_source(statistic)])[statistic])


def _interval(samples: np.ndarray) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=np.float64)
    return percentile_ci(samples[np.isfinite(samples)]) if np.isfinite(samples).any() else (float("nan"), float("nan"))


def _spread(values: Sequence[float]) -> tuple[float, float, int]:
    """Min and max of at least two finite values (NaN otherwise) and their number."""
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return float("nan"), float("nan"), len(finite)
    return float(min(finite)), float(max(finite)), len(finite)


# ----------------------------------------------------------------------------- E1


def _p0_rmse(directory: Path) -> float:
    return _num(_json(directory / "train.json"), "test", "p0", "rmse_a")


def _reference_on_probe(root, dataset: str, label: str, fold: int, reference: str) -> float:
    """P0 ``rmse_a`` of the reference on the probe events of the clean arm of ``fold``: from its
    own clean-arm run, else from its driver-split run of that fold restricted to the probe events
    (the probe events are a subset of that test part and the reference is not trained)."""
    own = run_dir(root, dataset, reference, CLEAN, fold)
    if (own / "train.json").exists():
        return _p0_rmse(own)
    probe = _read(run_dir(root, dataset, label, CLEAN, fold) / "test_p0.events.parquet", ["event_id"])
    full = _read(run_dir(root, dataset, reference, "driver", fold) / "test_p0.events.parquet")
    if probe is None or full is None:
        return float("nan")
    part = full[full["event_id"].astype(str).isin(set(probe["event_id"].astype(str)))]
    if part["event_id"].nunique() != probe["event_id"].nunique():
        return float("nan")
    return _point(part, "rmse_a")


def e1_ratios(root, dataset: str, labels: Sequence[str] | None = None, n_boot: int = 1000, seed: int = 0,
              reference: str = REFERENCE, subset: Sequence[str] = SUBSET) -> pd.DataFrame:
    """E1 (H5.1): ``RMSE_a(model) / RMSE_a(persistence)`` of P0 on fold 0 of the driver split.

    ``rmse_a`` and ``ratio`` from ``test_p0.events.parquet`` of the fold-0 seed-0 runs; ``low``,
    ``high``: 95 % interval of the ratio over ``n_boot`` driver replicates with common random
    numbers (the follower counts are drawn once per dataset over the union of the followers of
    every model's table and applied to the model's and the reference's table; see the module
    docstring). ``seed_min``, ``seed_max``: range of the ratio over the training seeds 0-2 of fold
    0 (``train.json``; NaN with fewer than two seeds, ``n_seeds`` counts them); ``folds_min``,
    ``folds_max``: range over the clean arms of folds 0-4 (probe events), for the models of
    ``subset`` (NaN with fewer than two folds). ``labels=None`` takes every finished run.
    """
    ref_dir = run_dir(root, dataset, reference)
    if not (ref_dir / "test_p0.events.parquet").exists():
        return pd.DataFrame(columns=E1_COLUMNS)
    labels = run_labels(root, dataset) if labels is None else list(labels)
    labels = [m for m in labels if (run_dir(root, dataset, m) / "test_p0.events.parquet").exists()]
    tables = {m: pd.read_parquet(run_dir(root, dataset, m) / "test_p0.events.parquet") for m in {reference, *labels}}
    universe, counts = common_draw(tables.values(), n_boot, seed)
    ref_samples = group_replicates(tables[reference], "rmse_a", universe, counts)
    ref_point = _point(tables[reference], "rmse_a")
    rows = []
    for label in labels:
        point = _point(tables[label], "rmse_a")
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = group_replicates(tables[label], "rmse_a", universe, counts) / ref_samples
        low, high = _interval(samples)
        seeds = []
        for s in SEEDS:
            directory = run_dir(root, dataset, label, seed=s)
            if (directory / "train.json").exists():
                ref = run_dir(root, dataset, reference, seed=s)
                seeds.append(_p0_rmse(directory) / (_p0_rmse(ref) if (ref / "train.json").exists() else ref_point))
        folds = []
        if label in subset or label == reference:
            for k in FOLDS:
                directory = run_dir(root, dataset, label, CLEAN, k)
                if (directory / "train.json").exists():
                    folds.append(_p0_rmse(directory) / _reference_on_probe(root, dataset, label, k, reference))
        seed_min, seed_max, n_seeds = _spread(seeds)
        folds_min, folds_max, n_folds = _spread(folds)
        rows.append({
            "dataset": dataset, "model": label, "family": _family(run_dir(root, dataset, label)), "rmse_a": point,
            "ratio": point / ref_point, "low": low, "high": high, "n_drivers": int(tables[label][GROUP].nunique()),
            "seed_min": seed_min, "seed_max": seed_max, "n_seeds": n_seeds,
            "folds_min": folds_min, "folds_max": folds_max, "n_folds": n_folds,
        })
    return pd.DataFrame(rows, columns=E1_COLUMNS)


# ----------------------------------------------------------------------------- E2


def affected_followers(dataset: str, splits_dir=None, events_dir=None, version: str = VERSION) -> dict[int, frozenset[str]]:
    """Fold -> the probe drivers of the paired split that are *affected* by the leak: followers of
    the probe events (``test`` of the leaky arm) with at least one event in the training part of
    the leaky arm of the same fold. Reads ``<splits_dir>/<dataset>_paired-leaky.json`` (default
    ``data/splits``) and the event index ``<events_dir>/<dataset>/<version>/events.parquet``
    (default ``data/events``) for the follower of every event."""
    splits_dir = resolve_path(splits_dir or "data/splits")
    events_dir = resolve_path(events_dir or "data/events")
    split = read_json(Path(splits_dir) / f"{dataset}_{LEAKY}.json")
    index = pd.read_parquet(Path(events_dir) / dataset / version / "events.parquet", columns=["event_id", GROUP])
    follower = dict(zip(index["event_id"].astype(str), index[GROUP].astype(str)))
    out = {}
    for fold, parts in (split.get("explicit") or {}).items():
        missing = [e for e in (*parts.get("train", ()), *parts.get("test", ())) if e not in follower]
        if missing:
            raise KeyError(f"{dataset} fold {fold}: {len(missing)} events of the split are not in the event index, e.g. {missing[:2]}")
        trained = {follower[e] for e in parts.get("train", ())}
        out[int(fold)] = frozenset(f for f in (follower[e] for e in parts.get("test", ())) if f in trained)
    if not out:
        raise ValueError(f"{dataset}_{LEAKY}.json has no explicit parts")
    return out


def _arm_table(root, dataset: str, label: str, arm: str, fold: int, kind: str) -> pd.DataFrame | None:
    directory = run_dir(root, dataset, label, arm, fold)
    if kind in ("p0", "whole"):
        return _training_table(directory, kind)[0]
    return own_p1_table(directory, kind, fallback=False)[0]


def e2_paired(root, dataset: str, labels: Sequence[str] = SUBSET, n_boot: int = 1000, seed: int = 0,
              affected: Mapping[int, Iterable[str]] | str | None = "auto") -> pd.DataFrame:
    """E2 (H5.2): paired inflation of the leaky arm on the probe events, pooled over the folds.

    For every metric of :data:`E2_METRICS` and model, the per-event tables of the folds present in
    both arms are concatenated (the test drivers of the folds are distinct) and compared with
    :func:`~follownet2.stats.paired_difference` (``clean - leaky``: positive when the leak lowers
    the error); ``inflation = (clean - leaky) / clean``; ``p_holm`` is the Holm adjustment of
    ``p_bootstrap`` over the models of a dataset, subset and metric. ``subset`` is ``all`` (every
    probe event) or ``affected`` (the probe events of the drivers of ``affected[fold]``, those with
    a leak event in the leaky training part; ``"auto"`` reads them with
    :func:`affected_followers`, ``None`` leaves the subset out).
    """
    if isinstance(affected, str):
        try:
            affected = affected_followers(dataset)
        except (OSError, KeyError, ValueError) as error:
            warnings.warn(f"E2 {dataset}: no affected subset ({type(error).__name__}: {error})", stacklevel=2)
            affected = None
    affected = None if affected is None else {int(k): frozenset(map(str, v)) for k, v in affected.items()}
    rows = []
    for metric, (kind, statistic) in E2_METRICS.items():
        blocks: dict[str, list[dict]] = {name: [] for name in E2_SUBSETS}
        for label in labels:
            clean, leaky, folds = [], [], []
            for k in FOLDS:
                a, b = (_arm_table(root, dataset, label, arm, k, kind) for arm in (CLEAN, LEAKY))
                if a is not None and b is not None and len(a) and len(b):
                    clean.append(a.assign(fold=k))
                    leaky.append(b.assign(fold=k))
                    folds.append(k)
            if not folds:
                continue
            clean_all, leaky_all = pd.concat(clean, ignore_index=True), pd.concat(leaky, ignore_index=True)
            for frame in (clean_all, leaky_all):
                frame[GROUP] = frame[GROUP].astype(str)
            hit = None
            if affected is not None and all(k in affected for k in folds):
                hit = lambda t: np.array([f in affected[k] for f, k in zip(t[GROUP], t["fold"])], dtype=bool)  # noqa: E731
            n_affected = int(clean_all.loc[hit(clean_all), GROUP].nunique()) if hit else float("nan")
            parts = {"all": (clean_all, leaky_all)}
            if hit is not None:
                parts["affected"] = (clean_all[hit(clean_all)], leaky_all[hit(leaky_all)])
            family = _family(run_dir(root, dataset, label, CLEAN, folds[0]))
            for name, (a, b) in parts.items():
                if not len(a) or not len(b):
                    continue
                result = paired_difference(a, b, GROUP, None, n_boot, seed, column=statistic)
                c, lk = _point(a, statistic), _point(b, statistic)
                blocks[name].append({
                    "dataset": dataset, "subset": name, "model": label, "family": family, "metric": metric,
                    "clean": c, "leaky": lk, "inflation": (c - lk) / c if c else float("nan"),
                    "mean_difference": result["mean_difference"], "low": result["low"], "high": result["high"],
                    "p_bootstrap": result["p_bootstrap"], "p_wilcoxon": result["p_wilcoxon"],
                    "n_drivers": int(result["n_groups"]), "n_affected_drivers": n_affected,
                    "n_events": int(a["event_id"].nunique()), "n_folds": len(folds),
                })
        for block in blocks.values():
            for row, p in zip(block, holm([row["p_bootstrap"] for row in block])):
                row["p_holm"] = float(p)
            rows += block
    frame = pd.DataFrame(rows, columns=E2_COLUMNS)
    order = {name: i for i, name in enumerate(E2_SUBSETS)}
    return frame.sort_values(["subset", "metric"], key=lambda s: s.map(order) if s.name == "subset" else s,
                             kind="stable").reset_index(drop=True)


def e2_class_means(table: pd.DataFrame) -> pd.DataFrame:
    """Mean, min and max inflation and the mean difference per dataset, subset, metric and family
    (mechanistic, network) of the table of :func:`e2_paired`."""
    if table.empty:
        return pd.DataFrame(columns=E2_CLASS_COLUMNS)
    out = (
        table.groupby(["dataset", "subset", "metric", "family"], sort=True)
        .agg(mean_inflation=("inflation", "mean"), min_inflation=("inflation", "min"), max_inflation=("inflation", "max"),
             mean_difference=("mean_difference", "mean"), n_models=("model", "nunique"))
        .reset_index()
    )
    return out[E2_CLASS_COLUMNS]


def _metric_value(directory: Path, metric: str) -> float:
    train = _json(directory / "train.json")
    if metric == "p0_rmse_a":
        return _num(train, "test", "p0", "rmse_a")
    if metric == "p1_whole_rmse_s":
        return _num(train, "test", "closed_loop_whole_event", "rmse_s")
    horizon = metric.split("_")[1]
    return _num(_summary(directory, "p1"), horizon, "rmse_s")


def e2_unpaired(root, dataset: str, labels: Sequence[str] = SUBSET) -> pd.DataFrame:
    """E2, unpaired comparison: event split against driver split on fold 0, each on its own test part.

    ``driver``, ``event``: pooled scores (P0 and the whole event from ``train.json``, P1 10 s from
    ``p1.json``); ``inflation = (driver - event) / driver``, ``difference = driver - event``
    (no test: the two test parts differ).
    """
    rows = []
    for label in labels:
        driver, event = run_dir(root, dataset, label, "driver", 0), run_dir(root, dataset, label, "event", 0)
        if not ((driver / "train.json").exists() and (event / "train.json").exists()):
            continue
        n = {name: _num(_json(d / "train.json"), "test", "closed_loop_whole_event", "n_events") for name, d in (("driver", driver), ("event", event))}
        for metric in E2_METRICS:
            d, e = _metric_value(driver, metric), _metric_value(event, metric)
            rows.append({
                "dataset": dataset, "model": label, "family": _family(driver), "metric": metric, "driver": d, "event": e,
                "inflation": (d - e) / d if d else float("nan"), "difference": d - e,
                "n_events_driver": n["driver"], "n_events_event": n["event"],
            })
    return pd.DataFrame(rows, columns=E2_UNPAIRED_COLUMNS)


# ----------------------------------------------------------------------------- E3


def _composite(frame: pd.DataFrame) -> np.ndarray:
    """Composite of the models (rows) of an E3 table from its in-range inputs; NaN without any input."""
    inputs = pd.DataFrame({name: frame[col].to_numpy(dtype=np.float64) for name, col in COMPOSITE_COLUMNS.items()},
                          index=pd.Index(frame["model"].to_numpy(), name="model"))[list(INPUTS)]
    if inputs.empty or inputs.isna().all(axis=None):
        return np.full(len(frame), np.nan)
    return composite_score(inputs)["composite"].reindex(inputs.index).to_numpy(dtype=np.float64)


def e3_rankings(root, dataset: str, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """E3: the metrics of every model on fold 0, seed 0 of the driver split (lower is better).

    P0 ``rmse_a`` and the whole event (``rmse_s``, ``rmse_v``) from ``train.json`` (the tables
    ``test_p0`` and ``test_closed_loop`` of the current checkpoint; the whole event of ``p1.json``
    is the same quantity), P1 at 10 s and 30 s from ``p1.json``, the in-range P2 keys
    (``*_in_range``), the P3 ``unstable_fraction``, the in-range synthetic and the empirical P4
    collision rates, and the composite (:func:`~follownet2.protocols.composite.composite_score`
    on the in-range inputs, as in milestone 3). ``stale`` lists the protocol files computed from
    an earlier checkpoint (:func:`stale_protocols`).
    """
    labels = run_labels(root, dataset) if labels is None else [m for m in labels if (run_dir(root, dataset, m) / "train.json").exists()]
    rows = []
    for label in labels:
        d = run_dir(root, dataset, label)
        train = _json(d / "train.json")
        p1, p2, p3, p4 = (_summary(d, name) for name in ("p1", "p2", "p3", "p4"))
        row = {"dataset": dataset, "model": label, "family": train.get("family"), "p0_rmse_a": _num(train, "test", "p0", "rmse_a")}
        for h in ("h10", "h30"):
            for x in ("s", "v"):
                row[f"p1_{h}_rmse_{x}"] = _num(p1, h, f"rmse_{x}")
        for x in ("s", "v"):
            row[f"p1_whole_rmse_{x}"] = _num(train, "test", "closed_loop_whole_event", f"rmse_{x}")
        row["p2_concave_growth_error"] = _num(p2, "concave_growth_error_in_range")
        row["p2_hysteresis_error"] = _num(p2, "hysteresis_error_in_range")
        row["p2_collisions_100"] = _num(p2, "collisions_100_in_range")
        row["p3_unstable_fraction"] = _num(p3, "unstable_fraction")
        row["p4_collision_rate_synthetic"] = _num(p4, "collision_rate_synthetic_in_range")
        row["p4_collision_rate_empirical"] = _num(p4, "collision_rate_empirical")
        row["stale"] = ",".join(stale_protocols(d))
        rows.append(row)
    frame = pd.DataFrame(rows, columns=[c for c in E3_COLUMNS if c != "composite"])
    frame["composite"] = _composite(frame) if len(frame) else []
    return frame[E3_COLUMNS]


def _needs(rankings: pd.DataFrame, column: str) -> list[str]:
    """Models with a finite value of ``column``."""
    values = rankings[column].to_numpy(dtype=np.float64)
    return [m for m, v in zip(rankings["model"], values) if np.isfinite(v)]


def _fill(models: Sequence[str], values: Mapping[str, np.ndarray], n_boot: int) -> dict[str, np.ndarray]:
    return {m: np.asarray(values[m], dtype=np.float64) if m in values else np.full(n_boot, np.nan) for m in models}


def _missing_reason(missing: Sequence[str], need: Sequence[str], problems: Mapping[str, str], what: str) -> str:
    example = missing[0]
    return f"{what} not usable for {len(missing)} of {len(need)} models (e.g. {example}: {problems.get(example, 'missing')})"


def _driver_unit(root, dataset, rankings, models, columns, n_boot, seed, samples, reasons) -> None:
    kinds = sorted({DRIVER_COLUMNS[c][0] for c in columns})
    tables: dict[str, dict[str, pd.DataFrame]] = {kind: {} for kind in kinds}
    problems: dict[str, dict[str, str]] = {kind: {} for kind in kinds}
    for model in models:
        d = run_dir(root, dataset, model)
        for kind in kinds:
            table, problem = _training_table(d, kind) if kind in ("p0", "whole") else own_p1_table(d, kind, fallback=False)
            if table is not None:
                tables[kind][model] = table
            else:
                problems[kind][model] = problem
    present = [t for kind in kinds for t in tables[kind].values()]
    if not present:
        for column in columns:
            if _needs(rankings, column):
                reasons[column] = "no per-event table"
        return
    universe, counts = common_draw(present, n_boot, seed)
    for column in columns:
        kind, statistic = DRIVER_COLUMNS[column]
        need = _needs(rankings, column)
        if not need:
            continue
        missing = [m for m in need if m not in tables[kind]]
        if missing:
            reasons[column] = _missing_reason(missing, need, problems[kind], "per-window table" if kind not in ("p0", "whole") else "per-event table")
            continue
        samples[column] = _fill(models, {m: group_replicates(tables[kind][m], statistic, universe, counts) for m in need}, n_boot)


def _profile_unit(root, dataset, rankings, models, n_boot, seed, samples, reasons, composite: bool) -> None:
    per_profile, per_speed, problems = {}, {}, {}
    for model in models:
        d = run_dir(root, dataset, model)
        profiles = _read(d / "p2.profiles.parquet")
        if profiles is not None:
            inside = profiles["in_range"].to_numpy(dtype=bool, na_value=False) if "in_range" in profiles else np.ones(len(profiles), bool)
            if "n_vehicles" in profiles:
                inside = inside & (profiles["n_vehicles"].to_numpy() == profiles["n_vehicles"].max())
            part = profiles[inside].rename(columns={"collisions": "collisions_100"})
            per_profile[model] = part[["profile", *PROFILE_COLUMNS]].assign(profile=part["profile"].astype(str))
        else:
            problems[model] = "no p2.profiles.parquet"
        speeds = _read(d / "p3.speeds.parquet")
        if speeds is not None:
            equilibrium = speeds["converged"].to_numpy(dtype=bool, na_value=False) & speeds["tracks"].to_numpy(dtype=bool, na_value=False)
            unstable = speeds["unstable_numerical"].to_numpy(dtype=np.float64, na_value=np.nan)
            per_speed[model] = pd.DataFrame({"v_e": speeds["v_e"].to_numpy(dtype=np.float64), "unstable": np.where(equilibrium, unstable, np.nan)})
        else:
            problems.setdefault(model, "no p3.speeds.parquet")
    # one Generator: the profile counts, then the speed counts (the order of composite_bootstrap)
    rng = np.random.default_rng([seed, 1])
    p2_columns = [c for c in ("p2_concave_growth_error", "p2_hysteresis_error", "p2_collisions_100") if _needs(rankings, c)]
    p2_need = sorted({m for c in p2_columns for m in _needs(rankings, c)})
    if p2_need:
        aligned, reason = None, None
        p2_missing = [m for m in p2_need if m not in per_profile]
        if p2_missing:
            reason = _missing_reason(p2_missing, p2_need, problems, "p2.profiles.parquet")
        else:
            try:
                aligned = _aligned(per_profile, "profile", PROFILE_COLUMNS, "per_profile")
            except (KeyError, ValueError) as error:
                reason = f"P2 profiles differ between models ({error})"
            if aligned is not None and not aligned["collisions_100"].shape[1]:
                aligned, reason = None, "no in-range P2 profile"
        if aligned is None:
            reasons.update(dict.fromkeys(p2_columns, reason))
        else:
            counts = draw_counts(rng, aligned["collisions_100"].shape[1], n_boot).astype(np.float64)
            values = {
                "p2_concave_growth_error": _mean_over(counts, aligned["concave_growth_error"]),
                "p2_hysteresis_error": _mean_over(counts, aligned["hysteresis_error"]),
                "p2_collisions_100": _sum_over(counts, aligned["collisions_100"]),
            }
            for column in p2_columns:
                samples[column] = _fill(models, {m: values[column][:, i] for i, m in enumerate(per_profile)}, n_boot)
    p3_need = _needs(rankings, "p3_unstable_fraction")
    if p3_need:
        p3_missing = [m for m in p3_need if m not in per_speed]
        if p3_missing:
            reasons["p3_unstable_fraction"] = _missing_reason(p3_missing, p3_need, problems, "p3.speeds.parquet")
        else:
            try:
                aligned = _aligned(per_speed, "v_e", ("unstable",), "per_speed")
                counts = draw_counts(rng, aligned["unstable"].shape[1], n_boot).astype(np.float64)
                matrix = _mean_over(counts, aligned["unstable"])
                samples["p3_unstable_fraction"] = _fill(models, {m: matrix[:, i] for i, m in enumerate(per_speed)}, n_boot)
            except (KeyError, ValueError) as error:
                reasons["p3_unstable_fraction"] = f"P3 speeds differ between models ({error})"
    if not composite or not _needs(rankings, "composite"):
        return
    inputs = list(COMPOSITE_COLUMNS.values())
    absent = [c for c in inputs if _needs(rankings, c) and c not in samples]
    if absent:
        reasons["composite"] = "no replicates of " + ", ".join(f"{c} ({reasons.get(c, 'missing')})" for c in absent)
        return
    stacked = {c: np.column_stack([samples[c][m] for m in models]) if c in samples else np.full((n_boot, len(models)), np.nan)
               for c in inputs}
    matrix = np.empty((n_boot, len(models)))
    for r in range(n_boot):
        frame = pd.DataFrame({"model": models, **{c: stacked[c][r] for c in inputs}})
        matrix[r] = _composite(frame)
    samples["composite"] = {m: matrix[:, i] for i, m in enumerate(models)}


def _scenario_unit(root, dataset, rankings, models, n_boot, seed, samples, reasons) -> None:
    column = "p4_collision_rate_synthetic"
    need = _needs(rankings, column)
    if not need:
        return
    frames, problems = {}, {}
    for model in need:
        scenarios = _read(run_dir(root, dataset, model) / "p4.scenarios.parquet")
        if scenarios is None:
            problems[model] = "no p4.scenarios.parquet"
            continue
        inside = scenarios["in_range"].to_numpy(dtype=bool, na_value=False) if "in_range" in scenarios else np.ones(len(scenarios), bool)
        part = scenarios[inside]
        frames[model] = pd.DataFrame({"event_id": part["event_id"].astype(str).to_numpy(),
                                      "collided": part["collided"].to_numpy(dtype=np.float64, na_value=np.nan)})
    missing = [m for m in need if m not in frames]
    if missing:
        reasons[column] = _missing_reason(missing, need, problems, "p4.scenarios.parquet")
        return
    try:
        aligned = _aligned(frames, "event_id", ("collided",), "scenarios")
    except (KeyError, ValueError) as error:
        reasons[column] = f"P4 scenarios differ between models ({error})"
        return
    counts = draw_counts(np.random.default_rng([seed, 2]), aligned["collided"].shape[1], n_boot).astype(np.float64)
    matrix = _mean_over(counts, aligned["collided"])
    samples[column] = _fill(models, {m: matrix[:, i] for i, m in enumerate(frames)}, n_boot)


def _episode_unit(root, dataset, rankings, models, n_boot, seed, samples, reasons) -> None:
    column = "p4_collision_rate_empirical"
    need = _needs(rankings, column)
    if not need:
        return
    tables = {m: _read(run_dir(root, dataset, m) / "p4.episodes.parquet", ["event_id", GROUP, "collided"]) for m in need}
    missing = [m for m, t in tables.items() if t is None]
    if missing:
        reasons[column] = _missing_reason(missing, need, {m: "no p4.episodes.parquet" for m in missing}, "p4.episodes.parquet")
        return
    universe, counts = common_draw(tables.values(), n_boot, [seed, 3])
    samples[column] = _fill(models, {m: group_replicates(t, "collision_rate", universe, counts) for m, t in tables.items()}, n_boot)


def e3_replicates(root, dataset: str, rankings: pd.DataFrame | None = None, n_boot: int = 1000, seed: int = 0,
                  columns: Sequence[str] | None = None) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    """Bootstrap replicates of the E3 metrics: ``{column: {model: [n_boot]}}`` and, for every
    column with values but without replicates, the reason.

    Units (module docstring): drivers for P0 and P1 (all horizons, ``rmse_s`` and ``rmse_v``),
    in-range profiles for P2, speeds with an equilibrium for P3, both for the composite, in-range
    scenarios for the synthetic P4 and episode drivers for the empirical P4. A model without a
    value has NaN replicates; a column is left out when a model with a value has no usable table.
    """
    rankings = e3_rankings(root, dataset) if rankings is None else rankings
    columns = list(E3_METRICS) if columns is None else list(columns)
    models = list(rankings["model"])
    samples: dict[str, dict[str, np.ndarray]] = {}
    reasons: dict[str, str] = {}
    if not models:
        return samples, reasons
    driver = [c for c in columns if c in DRIVER_COLUMNS]
    if driver:
        _driver_unit(root, dataset, rankings, models, driver, n_boot, seed, samples, reasons)
    if any(c in columns for c in (*COMPOSITE_COLUMNS.values(), "composite")):
        _profile_unit(root, dataset, rankings, models, n_boot, seed, samples, reasons, composite="composite" in columns)
    if "p4_collision_rate_synthetic" in columns:
        _scenario_unit(root, dataset, rankings, models, n_boot, seed, samples, reasons)
    if "p4_collision_rate_empirical" in columns:
        _episode_unit(root, dataset, rankings, models, n_boot, seed, samples, reasons)
    return {c: samples[c] for c in columns if c in samples}, {c: reasons[c] for c in columns if c in reasons}


def _last(values: Mapping[str, float] | Mapping[str, np.ndarray]) -> dict:
    """NaN -> +inf: a model without a value ranks last (lower is better)."""
    return {m: np.where(np.isnan(np.asarray(v, dtype=np.float64)), np.inf, np.asarray(v, dtype=np.float64)) for m, v in values.items()}


def _tau_row(rankings, samples, reasons, column_a, column_b) -> dict:
    models = list(rankings["model"])
    point = {c: dict(zip(models, rankings[c].to_numpy(dtype=np.float64))) for c in (column_a, column_b)}
    a, b = _last(point[column_a]), _last(point[column_b])
    tau, p = kendall_tau({m: float(v) for m, v in a.items()}, {m: float(v) for m, v in b.items()})
    row = {"tau": tau, "p": p, "n_models": len(models), "low": float("nan"), "high": float("nan"), "tau_boot": float("nan"), "note": ""}
    if column_a in samples and column_b in samples:
        boot = kendall_tau_bootstrap(_last(samples[column_a]), _last(samples[column_b]))
        row.update(low=boot["low"], high=boot["high"], tau_boot=boot["tau"])
    else:
        row["note"] = "; ".join(f"{c}: {reasons.get(c, 'no replicates')}" for c in (column_a, column_b) if c not in samples)
    return row


def _present(rankings: pd.DataFrame, column: str) -> bool:
    """A ranking exists when at least two models have a value."""
    return column in rankings and np.isfinite(rankings[column].to_numpy(dtype=np.float64)).sum() >= 2


def e3_kendall(root, dataset: str, n_boot: int = 1000, seed: int = 0, rankings: pd.DataFrame | None = None,
               replicates: tuple[dict, dict] | None = None) -> pd.DataFrame:
    """Kendall's tau-b between every pair of the rankings of :data:`RANKINGS` present for the dataset.

    ``tau`` and ``p`` (:func:`~follownet2.stats.kendall_tau`) on the point values, ``low`` and
    ``high`` the 95 % interval and ``tau_boot`` the mean of tau over the replicates of
    :func:`e3_replicates` (replicate ``r`` of one ranking against replicate ``r`` of the other;
    rankings of the same unit share the draw). A model without a value ranks last; ``note`` says
    why an interval is missing.
    """
    rankings = e3_rankings(root, dataset) if rankings is None else rankings
    if replicates is None:
        replicates = e3_replicates(root, dataset, rankings, n_boot, seed, columns=list(RANKINGS.values()))
    samples, reasons = replicates
    names = [name for name, column in RANKINGS.items() if _present(rankings, column)]
    rows = [{"dataset": dataset, "ranking_a": a, "ranking_b": b, **_tau_row(rankings, samples, reasons, RANKINGS[a], RANKINGS[b])}
            for a, b in combinations(names, 2)]
    return pd.DataFrame(rows, columns=KENDALL_COLUMNS)


def e3_spacing_vs_speed(root, dataset: str, n_boot: int = 1000, seed: int = 0, rankings: pd.DataFrame | None = None,
                        replicates: tuple[dict, dict] | None = None) -> pd.DataFrame:
    """H5.4: Kendall's tau-b between the ``rmse_s`` and the ``rmse_v`` ranking of P1 per horizon,
    with driver-bootstrap intervals (both from the same tables and draws)."""
    rankings = e3_rankings(root, dataset) if rankings is None else rankings
    if replicates is None:
        columns = [f"p1_{h}_rmse_{x}" for h in HORIZONS for x in ("s", "v")]
        replicates = e3_replicates(root, dataset, rankings, n_boot, seed, columns=columns)
    samples, reasons = replicates
    rows = []
    for h in HORIZONS:
        s, v = f"p1_{h}_rmse_s", f"p1_{h}_rmse_v"
        if _present(rankings, s) and _present(rankings, v):
            rows.append({"dataset": dataset, "horizon": h, **_tau_row(rankings, samples, reasons, s, v)})
    return pd.DataFrame(rows, columns=SPEED_COLUMNS)


def _ordered(block: pd.DataFrame, column: str) -> pd.DataFrame:
    """Models by value (missing values last), then by name."""
    values = block[column].to_numpy(dtype=np.float64)
    frame = pd.DataFrame({"model": block["model"].to_numpy(), "value": np.where(np.isnan(values), np.inf, values)})
    return frame.sort_values(["value", "model"], kind="stable").reset_index(drop=True)


def _top_set(block: pd.DataFrame, column: str, k: int = 3) -> set[str]:
    """Models within the first ``k`` positions, ties at position ``k`` included."""
    order = _ordered(block, column)
    if len(order) <= k:
        return set(order["model"])
    return set(order.loc[order["value"] <= order["value"].iloc[k - 1], "model"])


def _blocks(rankings: pd.DataFrame):
    if "dataset" in rankings and rankings["dataset"].notna().any():
        yield from rankings.groupby("dataset", sort=True)
    else:
        yield "", rankings


def e3_top3(rankings: pd.DataFrame) -> pd.DataFrame:
    """The first three models of every ranking present (ties broken by name); ``values`` their
    values, ``tied`` the further models tied with the third."""
    rows = []
    for dataset, block in _blocks(rankings):
        for name, column in RANKINGS.items():
            if column not in block or not np.isfinite(block[column].to_numpy(dtype=np.float64)).any():
                continue
            order = _ordered(block, column)
            top = list(order["model"].iloc[:3]) + [""] * max(0, 3 - len(order))
            third = order["value"].iloc[min(2, len(order) - 1)]
            tied = [m for m, v in zip(order["model"].iloc[3:], order["value"].iloc[3:]) if v == third]
            values = " / ".join("n/a" if not np.isfinite(v) else f"{v:.3f}" for v in order["value"].iloc[:3])
            rows.append({"dataset": dataset, "ranking": name, "top1": top[0], "top2": top[1], "top3": top[2],
                         "values": values, "tied": ", ".join(tied)})
    return pd.DataFrame(rows, columns=TOP3_COLUMNS)


def e3_top3_changes(rankings: pd.DataFrame, sequence: Sequence[str] = TOP3_SEQUENCE) -> list[str]:
    """The models that enter or leave the top three between consecutive rankings of ``sequence``
    (and between its first and last), ties at the third position counted in."""
    lines = []
    for dataset, block in _blocks(rankings):
        names = [n for n in sequence if RANKINGS[n] in block and np.isfinite(block[RANKINGS[n]].to_numpy(dtype=np.float64)).any()]
        pairs = list(zip(names, names[1:]))
        if len(names) > 2:
            pairs.append((names[0], names[-1]))
        for a, b in pairs:
            top_a, top_b = _top_set(block, RANKINGS[a]), _top_set(block, RANKINGS[b])
            enter, leave = sorted(top_b - top_a), sorted(top_a - top_b)
            prefix = f"{dataset}: " if dataset else ""
            lines.append(f"{prefix}{a} -> {b}: enter {', '.join(enter) or 'none'}; leave {', '.join(leave) or 'none'}")
    return lines


# ----------------------------------------------------------------------------- E4


def e4_transfer(root, labels: Sequence[str] | None = None, n_boot: int = 1000, seed: int = 0,
                horizons: Sequence[str] = ("h10", "whole"), version: str = VERSION, split: str = "driver", fold: int = 0) -> pd.DataFrame:
    """E4 (P5): relative degradation ``rmse_s(A -> B) / rmse_s(B -> B)`` per model and horizon.

    Point values: :func:`~follownet2.protocols.p5.collect_transfer` and
    :func:`~follownet2.protocols.p5.transfer_matrix` (seed 0). Interval: driver bootstrap over the
    test drivers of ``B``, one draw per target dataset shared by every model, source and horizon,
    on the per-window tables ``p1_<B>.<h>.parquet`` of the run trained on ``A`` and the own table
    of the run trained on ``B`` (:func:`own_p1_table`); every table must reproduce the summary of
    its JSON, otherwise the entry keeps its point value and ``note`` says why there is no interval.
    """
    runs = collect_transfer(Path(root), version, split, fold)
    if runs.empty:
        return pd.DataFrame(columns=E4_COLUMNS)
    if labels is not None:
        runs = runs[runs["model"].isin(list(labels))]
    runs = runs[runs["horizon"].isin(list(horizons))]
    rmse = runs.set_index(["model", "trained_on", "tested_on", "horizon"])["rmse_s"]
    entries = []
    for horizon in horizons:
        part = runs[runs["horizon"] == horizon]
        if part.empty:
            continue
        matrix = transfer_matrix(part, horizon)
        for row in part.itertuples(index=False):
            ratio = float("nan")
            if (row.model, row.trained_on) in matrix.index and row.tested_on in matrix.columns:
                ratio = float(matrix.loc[(row.model, row.trained_on), row.tested_on])
            entries.append({"model": row.model, "trained_on": row.trained_on, "tested_on": row.tested_on, "horizon": horizon,
                            "rmse_s": float(row.rmse_s), "rmse_s_own": float(rmse.get((row.model, row.tested_on, row.tested_on, horizon), np.nan)),
                            "ratio": ratio})
    if not entries:
        return pd.DataFrame(columns=E4_COLUMNS)
    frame = pd.DataFrame(entries)
    own: dict[tuple, tuple] = {}
    foreign: dict[tuple, tuple] = {}
    for row in frame.itertuples(index=False):
        key = (row.model, row.tested_on, row.horizon)
        if key not in own:
            own[key] = own_p1_table(run_dir(root, row.tested_on, row.model, split, fold, version=version), row.horizon)
        if row.trained_on != row.tested_on:
            directory = run_dir(root, row.trained_on, row.model, split, fold, version=version)
            foreign[(row.model, row.trained_on, row.tested_on, row.horizon)] = transfer_table(directory, row.tested_on, row.horizon)
    draws = {}
    for target in sorted(frame["tested_on"].unique()):
        tables = [t for (m, b, h), (t, _) in own.items() if b == target and t is not None]
        tables += [t for (m, a, b, h), (t, _) in foreign.items() if b == target and t is not None]
        if tables:
            draws[target] = common_draw(tables, n_boot, seed)
    low, high, n_drivers, notes = [], [], [], []
    for row in frame.itertuples(index=False):
        own_table, own_problem = own[(row.model, row.tested_on, row.horizon)]
        if row.trained_on == row.tested_on:
            other, other_problem = own_table, own_problem
        else:
            other, other_problem = foreign[(row.model, row.trained_on, row.tested_on, row.horizon)]
        if own_table is None or other is None or row.tested_on not in draws:
            interval, drivers = (float("nan"), float("nan")), float("nan")
            notes.append("; ".join(dict.fromkeys(p for p in (other_problem, own_problem) if p)) or "no table")
        else:
            universe, counts = draws[row.tested_on]
            with np.errstate(divide="ignore", invalid="ignore"):
                samples = group_replicates(other, "rmse_s", universe, counts) / group_replicates(own_table, "rmse_s", universe, counts)
            interval, drivers = _interval(samples), int(other[GROUP].nunique())
            notes.append("")
        low.append(interval[0])
        high.append(interval[1])
        n_drivers.append(drivers)
    frame["low"], frame["high"], frame["n_drivers"], frame["note"] = low, high, n_drivers, notes
    return frame.sort_values(["horizon", "model", "trained_on", "tested_on"], kind="stable").reset_index(drop=True)[E4_COLUMNS]


def _foreign(table: pd.DataFrame) -> pd.DataFrame:
    part = table[(table["trained_on"] != table["tested_on"])]
    return part[np.isfinite(part["ratio"].to_numpy(dtype=np.float64))]


def e4_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean relative degradation per model and horizon over its foreign pairs (``A != B``), and
    the pair with the largest degradation."""
    part = _foreign(table)
    rows = []
    for (model, horizon), block in part.groupby(["model", "horizon"], sort=True):
        worst = block.loc[block["ratio"].idxmax()]
        rows.append({"model": model, "horizon": horizon, "mean_degradation": float(block["ratio"].mean()), "n_pairs": len(block),
                     "worst_source": worst["trained_on"], "worst_target": worst["tested_on"], "worst_ratio": float(worst["ratio"])})
    return pd.DataFrame(rows, columns=E4_SUMMARY_COLUMNS)


def e4_targets(table: pd.DataFrame) -> pd.DataFrame:
    """Mean relative degradation per target dataset and horizon over the models and foreign sources."""
    part = _foreign(table)
    rows = [{"tested_on": target, "horizon": horizon, "mean_degradation": float(block["ratio"].mean()), "n_pairs": len(block),
             "n_models": int(block["model"].nunique())}
            for (target, horizon), block in part.groupby(["tested_on", "horizon"], sort=True)]
    return pd.DataFrame(rows, columns=E4_TARGET_COLUMNS)


# ----------------------------------------------------------------------------- campaigns


def campaign_status(root, version: str = VERSION) -> pd.DataFrame:
    """Finished runs (``train.json``) of the campaigns of docs/experiments.md, section 1, per dataset."""
    root = Path(root)
    datasets = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    rows = []
    for dataset in datasets:
        main = run_labels(root, dataset, version=version)
        if main:
            with_protocols = sum(all((run_dir(root, dataset, m, version=version) / f"{p}.json").exists() for p in ("p2", "p3", "p4")) for m in main)
            transfer = sum(len(list(run_dir(root, dataset, m, version=version).glob("p1_*.json"))) for m in main)
            stale = sum(bool(stale_protocols(run_dir(root, dataset, m, version=version))) for m in main)
            rows.append({"campaign": "main" if dataset in MAIN_DATASETS else "transfer (training)", "dataset": dataset, "runs": len(main),
                         "detail": f"P2-P4 for {with_protocols}, transfer P1 files {transfer}, runs with stale protocol files {stale}"})
        for name, split, folds, seeds in (("seeds", "driver", (0,), (1, 2)), ("paired", CLEAN, FOLDS, (0,)),
                                          ("paired", LEAKY, FOLDS, (0,)), ("event", "event", (0,), (0,))):
            counts = {(k, s): len(run_labels(root, dataset, split, k, s, version)) for k in folds for s in seeds}
            total = sum(counts.values())
            if total:
                detail = ", ".join(f"{split}-fold{k} seed{s}: {n}" for (k, s), n in counts.items())
                rows.append({"campaign": name, "dataset": dataset, "runs": total, "detail": detail})
    return pd.DataFrame(rows, columns=STATUS_COLUMNS)
