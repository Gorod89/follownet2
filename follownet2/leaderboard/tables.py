"""Tables of the leaderboard (docs/leaderboard.md, sections 1 and 2).

Per dataset, from the run directories ``<root>/<dataset>/reconstructed/<split>-fold<k>/<label>/seed<s>/``
validated by :func:`follownet2.leaderboard.checklist.validate_dataset`:

* ``main``: one row per admitted entry (driver split, fold 0, seed 0), sorted by the composite and
  then by the whole-event P1 spacing error; a column group without any value for the dataset is
  left out (P1 at 30 s without 30 s windows; P2-P4 and the composite for the datasets with P1
  only; the spread columns without a second seed or fold);
* ``leak``: the admitted runs under the ``event`` split, labelled "driver leak", never ranked;
* ``transfer``: per admitted model with ``p1_<other>.json`` and target, the whole-event relative
  degradation ``rmse_s(A -> B) / rmse_s(B -> B)``; its mean is :func:`transfer_means`;
* ``refused``: every run of the dataset whose checklist is incomplete, with its problems;
* ``provenance``: data version, reconstruction method, split, seeds, git revisions, protocol
  configuration hashes, bootstrap settings and date.

Statistics (1000 replicates, seed 0) reuse :mod:`follownet2.experiments.analysis`. P0 and P1:
one driver-bootstrap draw per dataset and split over the sorted union of the followers of every
per-event table used (:func:`~follownet2.experiments.analysis.common_draw`; common random numbers
across models, metrics and horizons), applied to the tables that reproduce their JSON summary
(:func:`~follownet2.experiments.analysis.own_p1_table` and the training tables); the P0 ratio of
a replicate is the model's ``rmse_a`` over the one of ``persistence``. Composite rank: the rank
distribution (:func:`~follownet2.stats.rank_distribution`) of the replicates of
:func:`composite_replicates`, the profile-and-speed bootstrap of
:func:`~follownet2.protocols.composite.composite_bootstrap` on the in-range profiles and the
speeds with an equilibrium (decision M4-6) extended by the training seeds: in every replicate each
model takes one of its admitted seeds at random; the point composite and its rank are those of
seed 0. Over the admitted models with a composite input. Transfer intervals:
:func:`~follownet2.experiments.analysis.e4_transfer`.
"""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from follownet2.experiments import analysis as an
from follownet2.leaderboard.checklist import (
    CLEAN,
    ENTRY,
    FOLD_SPREAD,
    LEAK,
    MAIN_FOLD,
    MAIN_SEED,
    MAIN_SPLIT,
    SEED_SPREAD,
    protocol_files,
    validate_dataset,
)
from follownet2.protocols.composite import INPUTS, PROFILE_COLUMNS, _aligned, _mean_over, _sum_over, composite_score
from follownet2.stats import draw_counts, rank_distribution
from follownet2.utils import read_json

N_BOOT, SEED = 1000, 0
REFERENCE = an.REFERENCE
FAMILIES = ("baseline", "mechanistic", "network")
#: driver-bootstrap metrics: column -> (per-event table, statistic)
DRIVER_METRICS = {
    "p0_rmse_a": ("p0", "rmse_a"),
    "p1_h10_rmse_s": ("h10", "rmse_s"),
    "p1_h30_rmse_s": ("h30", "rmse_s"),
    "p1_whole_rmse_s": ("whole", "rmse_s"),
}
#: training seeds (driver split, fold 0) whose runs may enter the composite bootstrap
COMPOSITE_SEEDS = (0, 1, 2)


def _ci(column: str) -> list[str]:
    return [column, f"{column}_low", f"{column}_high"]


#: column groups of the main table, in order; a group other than CORE is left out when it has no value
GROUPS: dict[str, list[str]] = {
    "id": ["model", "family"],
    "p0": _ci("p0_ratio"),
    "p1_h10": _ci("p1_h10_rmse_s"),
    "p1_h30": _ci("p1_h30_rmse_s"),
    "p1_whole": [*_ci("p1_whole_rmse_s"), "p1_whole_collision_rate"],
    "p2": ["p2_concave_growth_error", "p2_hysteresis_error", "p2_collisions_100"],
    "p3": ["p3_unstable_fraction", "p3_unstable_fraction_large"],
    "p4": ["p4_collision_rate_synthetic", "p4_collision_rate_empirical"],
    "composite": ["composite", "composite_rank", "composite_rank_low", "composite_rank_high"],
    "seeds": ["p0_ratio_seed_min", "p0_ratio_seed_max", "p1_whole_rmse_s_seed_min", "p1_whole_rmse_s_seed_max", "n_seeds"],
    "folds": ["p0_ratio_fold_min", "p0_ratio_fold_max", "p1_whole_rmse_s_fold_min", "p1_whole_rmse_s_fold_max", "n_folds"],
}
CORE = ("id", "p0", "p1_h10", "p1_whole")
MAIN_COLUMNS = [c for columns in GROUPS.values() for c in columns]
_LABELS = ("model", "family", "n_seeds", "n_folds")
LEAK_COLUMNS = ["model", "family", "label", "run", *_ci("p0_rmse_a"), *_ci("p1_h10_rmse_s"), *_ci("p1_h30_rmse_s"),
                *_ci("p1_whole_rmse_s"), "p1_whole_collision_rate", "p1_whole_rmse_s_driver"]
TRANSFER_COLUMNS = ["model", "family", "tested_on", "ratio", "low", "high", "rmse_s", "rmse_s_own", "note"]
TRANSFER_MEAN_COLUMNS = ["model", "family", "mean_degradation", "n_targets"]
REFUSED_COLUMNS = ["run", "model", "role", "label", "problems"]
RANK_COLUMNS = ["mean_rank", "rank_low", "rank_high", "p_first", "p_top3"]
_ROLE_ORDER = {ENTRY: 0, LEAK: 1, SEED_SPREAD: 2, FOLD_SPREAD: 3}


@dataclass
class Board:
    """The leaderboard of one dataset."""

    dataset: str
    main: pd.DataFrame
    leak: pd.DataFrame
    transfer: pd.DataFrame
    refused: pd.DataFrame
    runs: pd.DataFrame
    provenance: dict[str, str]
    composite_ranks: pd.DataFrame | None = None
    p1_ranks: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)
    n_boot: int = N_BOOT
    seed: int = SEED

    def counts(self) -> dict[str, int]:
        """Admitted and refused ranked entries, driver-leak runs and validated runs."""
        runs = self.runs
        admitted = runs["admitted"].astype(bool)
        entry, leak = runs["role"] == ENTRY, runs["role"] == LEAK
        return {"admitted": int((entry & admitted).sum()), "refused": int((entry & ~admitted).sum()),
                "leak_admitted": int((leak & admitted).sum()), "leak_refused": int((leak & ~admitted).sum()),
                "runs": len(runs), "runs_refused": int((~admitted).sum())}


# ----------------------------------------------------------------------------- driver bootstrap


def driver_replicates(directories: Mapping[str, Path], n_boot: int = N_BOOT, seed: int = SEED,
                      columns: Sequence[str] = tuple(DRIVER_METRICS)) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
    """Driver-bootstrap replicates ``{column: {key: [n_boot]}}`` of the runs ``{key: directory}``
    and, per column, why a run has none.

    The per-event tables (``test_p0`` and ``test_closed_loop`` of the training, checked against
    ``train.json``; ``p1.<h>.parquet`` checked against ``p1.json``) enter only when they reproduce
    their summary; one count matrix over the union of their followers serves every run and column.
    A table without rows (no 30 s window) gives no replicates and no problem.
    """
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    problems: dict[str, dict[str, str]] = {}
    for key, directory in directories.items():
        for column in columns:
            kind = DRIVER_METRICS[column][0]
            if kind in ("p0", "whole"):
                table, problem = an._training_table(directory, kind)
            else:
                table, problem = an.own_p1_table(directory, kind, fallback=False)
            if table is None:
                problems.setdefault(column, {})[key] = problem
            elif len(table):
                tables[(column, key)] = table
    samples: dict[str, dict[str, np.ndarray]] = {}
    if tables:
        universe, counts = an.common_draw(tables.values(), n_boot, seed)
        for (column, key), table in tables.items():
            samples.setdefault(column, {})[key] = an.group_replicates(table, DRIVER_METRICS[column][1], universe, counts)
    return samples, problems


def _intervals(frame: pd.DataFrame, column: str, samples: Mapping[str, np.ndarray], problems: Mapping[str, str],
               notes: list[str], what: str = "") -> None:
    """``<column>_low`` and ``<column>_high`` of the rows (index: keys of ``samples``) with a value."""
    low, high, missing = [], [], []
    for key, value in zip(frame.index, frame[column].to_numpy(dtype=np.float64)):
        replicates = samples.get(key)
        if replicates is None or not np.isfinite(value):
            low.append(np.nan)
            high.append(np.nan)
            if np.isfinite(value):
                missing.append(key)
            continue
        lo, hi = an._interval(replicates)
        low.append(lo)
        high.append(hi)
    frame[f"{column}_low"], frame[f"{column}_high"] = low, high
    if missing:
        names = [str(frame.at[k, "model"]) if "model" in frame else str(k) for k in missing]
        notes.append(f"{what}{column}: no interval for {', '.join(names)} ({problems.get(missing[0], 'no per-event table')})")


# ----------------------------------------------------------------------------- composite bootstrap


def composite_frames(directory) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """The composite inputs of one run, as in :func:`~follownet2.experiments.analysis.e3_replicates`:
    the in-range P2 profiles of the largest platoon (``profile`` and
    :data:`~follownet2.protocols.composite.PROFILE_COLUMNS`) and the P3 speeds with ``unstable``
    (NaN without an equilibrium, decision M4-6); None without ``p2.profiles.parquet`` or
    ``p3.speeds.parquet``."""
    directory = Path(directory)
    profiles, speeds = an._read(directory / "p2.profiles.parquet"), an._read(directory / "p3.speeds.parquet")
    if profiles is None or speeds is None:
        return None
    inside = profiles["in_range"].to_numpy(dtype=bool, na_value=False) if "in_range" in profiles else np.ones(len(profiles), bool)
    if "n_vehicles" in profiles:
        inside = inside & (profiles["n_vehicles"].to_numpy() == profiles["n_vehicles"].max())
    part = profiles[inside].rename(columns={"collisions": "collisions_100"})
    per_profile = part[["profile", *PROFILE_COLUMNS]].assign(profile=part["profile"].astype(str))
    equilibrium = speeds["converged"].to_numpy(dtype=bool, na_value=False) & speeds["tracks"].to_numpy(dtype=bool, na_value=False)
    unstable = speeds["unstable_numerical"].to_numpy(dtype=np.float64, na_value=np.nan)
    per_speed = pd.DataFrame({"v_e": speeds["v_e"].to_numpy(dtype=np.float64), "unstable": np.where(equilibrium, unstable, np.nan)})
    return per_profile, per_speed


def composite_replicates(root, dataset: str, runs: pd.DataFrame, models: Sequence[str], n_boot: int = N_BOOT, seed: int = SEED
                         ) -> tuple[dict[str, np.ndarray] | None, list[str], dict[str, list[int]]]:
    """Composite of ``models`` in ``n_boot`` replicates over profiles, speeds and training seeds.

    One Generator seeded ``[seed, 1]`` draws the counts of the in-range P2 profiles, then of the
    P3 speeds (one draw for all models, the order of
    :func:`~follownet2.experiments.analysis.e3_replicates`), then, per replicate and for each model
    independently, one of its seeds: the admitted runs of the driver split, fold 0, with seeds in
    :data:`COMPOSITE_SEEDS` that carry both tables (seed 0 alone when no other does). The
    replicate inputs are those of the drawn seed's run on the drawn profiles and speeds;
    :func:`~follownet2.protocols.composite.composite_score` ranks them within the models.

    Returns ``{model: [n_boot]}`` (None when the seed-0 tables are missing or differ between
    models), notes (the reason, or the seeds left out because their profiles or speeds differ) and
    the seeds used per model.
    """
    pool = runs[(runs["split"] == MAIN_SPLIT) & (runs["fold"] == MAIN_FOLD) & runs["seed"].isin(COMPOSITE_SEEDS)
                & runs["role"].isin([ENTRY, SEED_SPREAD]) & runs["admitted"].astype(bool)]
    frames: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for model in models:
        part = pool[pool["model"] == model].sort_values("seed")
        for s, directory in zip(part["seed"], part["directory"]):
            pair = composite_frames(directory)
            if pair is not None:
                frames[(model, int(s))] = pair
        if (model, MAIN_SEED) not in frames:
            return None, [f"{model}: no p2.profiles.parquet or p3.speeds.parquet in the seed-{MAIN_SEED} run"], {}
    reference = frames[(models[0], MAIN_SEED)]
    profiles, speeds = set(reference[0]["profile"]), set(reference[1]["v_e"])
    notes = []
    for key in list(frames):
        if set(frames[key][0]["profile"]) != profiles or set(frames[key][1]["v_e"]) != speeds:
            if key[1] == MAIN_SEED:
                return None, [f"the P2 profiles or P3 speeds of {key[0]} differ from those of {models[0]}"], {}
            notes.append(f"seed {key[1]} of {key[0]} left out of the composite bootstrap: its P2 profiles or P3 speeds differ")
            del frames[key]
    keys = list(frames)
    try:
        profile = _aligned({k: frames[k][0] for k in keys}, "profile", PROFILE_COLUMNS, "per_profile")
        speed = _aligned({k: frames[k][1] for k in keys}, "v_e", ("unstable",), "per_speed")
        rng = np.random.default_rng([seed, 1])
        profile_counts = draw_counts(rng, profile["collisions_100"].shape[1], n_boot).astype(np.float64)
        speed_counts = draw_counts(rng, speed["unstable"].shape[1], n_boot).astype(np.float64)
    except (KeyError, ValueError) as error:
        return None, [f"{type(error).__name__}: {error}"], {}
    seeds_of = {m: [s for k, s in keys if k == m] for m in models}
    column = {k: i for i, k in enumerate(keys)}
    pick = np.column_stack([np.asarray([column[(m, s)] for s in seeds_of[m]])[rng.integers(0, len(seeds_of[m]), size=n_boot)]
                            for m in models])
    inputs = {
        "concave_growth_error": _mean_over(profile_counts, profile["concave_growth_error"]),
        "hysteresis_error": _mean_over(profile_counts, profile["hysteresis_error"]),
        "unstable_fraction": _mean_over(speed_counts, speed["unstable"]),
        "collisions_100": _sum_over(profile_counts, profile["collisions_100"]),
    }
    rows = np.arange(n_boot)[:, None]
    chosen = {name: values[rows, pick] for name, values in inputs.items()}  # [n_boot, models]
    matrix = np.empty((n_boot, len(models)))
    for r in range(n_boot):
        table = pd.DataFrame({name: chosen[name][r] for name in INPUTS}, index=list(models))
        matrix[r] = composite_score(table)["composite"].reindex(list(models)).to_numpy(dtype=np.float64)
    return {m: matrix[:, i] for i, m in enumerate(models)}, notes, seeds_of


# ----------------------------------------------------------------------------- main table


def _whole(directory: Path, key: str = "rmse_s") -> float:
    return an._num(an._json(Path(directory) / "train.json"), "test", "closed_loop_whole_event", key)


def _admitted(runs: pd.DataFrame, role: str) -> pd.DataFrame:
    return runs[(runs["role"] == role) & runs["admitted"].astype(bool)]


def _seed_spread(runs: pd.DataFrame, labels: Sequence[str], reference: bool) -> dict[str, dict]:
    """Per model: range of the P0 ratio and the whole-event P1 error over its admitted training
    seeds of the driver split, fold 0 (the reference of the same seed, else seed 0, as in E1)."""
    seeds = runs[(runs["split"] == MAIN_SPLIT) & (runs["fold"] == MAIN_FOLD) & runs["role"].isin([ENTRY, SEED_SPREAD])
                 & runs["admitted"].astype(bool)]
    refs = seeds[seeds["model"] == REFERENCE]
    ref = {int(s): an._p0_rmse(d) for s, d in zip(refs["seed"], refs["directory"])}
    out = {}
    for label in labels:
        part = seeds[seeds["model"] == label]
        ratios, wholes = [], []
        for s, d in zip(part["seed"], part["directory"]):
            base = ref.get(s, ref.get(MAIN_SEED, np.nan)) if reference else np.nan
            ratios.append(an._p0_rmse(d) / base if base else np.nan)
            wholes.append(_whole(d))
        r_min, r_max, _ = an._spread(ratios)
        w_min, w_max, _ = an._spread(wholes)
        out[label] = {"p0_ratio_seed_min": r_min, "p0_ratio_seed_max": r_max, "p1_whole_rmse_s_seed_min": w_min,
                      "p1_whole_rmse_s_seed_max": w_max, "n_seeds": len(part)}
    return out


def _probe_reference(root, dataset: str, runs: pd.DataFrame, label: str, fold: int) -> float:
    """P0 ``rmse_a`` of the reference on the probe events of the clean arm of ``fold``
    (:func:`~follownet2.experiments.analysis._reference_on_probe`), NaN when the run it reads is refused."""
    def admitted(split: str) -> bool | None:
        row = runs[(runs["split"] == split) & (runs["fold"] == fold) & (runs["model"] == REFERENCE) & (runs["seed"] == MAIN_SEED)]
        return None if row.empty else bool(row["admitted"].iloc[0])

    own = admitted(CLEAN)
    if own is False or (own is None and admitted(MAIN_SPLIT) is not True):
        return float("nan")
    return an._reference_on_probe(root, dataset, label, fold, REFERENCE)


def _fold_spread(root, dataset: str, runs: pd.DataFrame, labels: Sequence[str]) -> dict[str, dict]:
    """Per model: range of the P0 ratio (against the reference on the same probe events, as in
    E1) and of the whole-event P1 error over the admitted clean arms of the paired split."""
    clean = _admitted(runs, FOLD_SPREAD)
    out = {}
    for label in labels:
        part = clean[clean["model"] == label]
        ratios, wholes = [], []
        for k, d in zip(part["fold"], part["directory"]):
            base = _probe_reference(root, dataset, runs, label, int(k))
            ratios.append(an._p0_rmse(d) / base if base else np.nan)
            wholes.append(_whole(d))
        r_min, r_max, _ = an._spread(ratios)
        w_min, w_max, _ = an._spread(wholes)
        out[label] = {"p0_ratio_fold_min": r_min, "p0_ratio_fold_max": r_max, "p1_whole_rmse_s_fold_min": w_min,
                      "p1_whole_rmse_s_fold_max": w_max, "n_folds": len(part)}
    return out


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    """By the composite, then the whole-event P1 spacing error (missing values last), then the name."""
    key = pd.DataFrame({"c": frame["composite"].astype(np.float64).fillna(np.inf),
                        "w": frame["p1_whole_rmse_s"].astype(np.float64).fillna(np.inf), "m": frame["model"]}, index=frame.index)
    return frame.loc[key.sort_values(["c", "w", "m"], kind="stable").index]


def _present(frame: pd.DataFrame) -> list[str]:
    keep = []
    for name, columns in GROUPS.items():
        values = [c for c in columns if c not in _LABELS]
        if name in CORE or (values and np.isfinite(frame[values].to_numpy(dtype=np.float64)).any()):
            keep += columns
    return keep


def main_table(root, dataset: str, runs: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED
               ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """The main table of the admitted entries, the rank distributions of the composite and of the
    whole-event P1 error (columns :data:`RANK_COLUMNS`, index model; None when not available)
    and the notes (intervals that could not be computed, missing reference)."""
    notes: list[str] = []
    labels = sorted(_admitted(runs, ENTRY)["model"])
    if not labels:
        return pd.DataFrame(columns=[c for g in CORE for c in GROUPS[g]]), None, None, ["no admitted entry"]
    directories = {m: an.run_dir(root, dataset, m) for m in labels}
    points = an.e3_rankings(root, dataset, labels).set_index("model", drop=False).reindex(labels)
    frame = pd.DataFrame({"model": labels, "family": points["family"].to_numpy()}, index=pd.Index(labels))
    samples, problems = driver_replicates(directories, n_boot, seed)

    reference = REFERENCE in labels
    p0 = points["p0_rmse_a"].astype(np.float64)
    ratio_samples: dict[str, np.ndarray] = {}
    ratio_problems = dict(problems.get("p0_rmse_a", {}))
    if reference:
        frame["p0_ratio"] = p0 / p0[REFERENCE]
        base = samples.get("p0_rmse_a", {}).get(REFERENCE)
        if base is None:
            ratio_problems = {m: f"{REFERENCE}: {problems.get('p0_rmse_a', {}).get(REFERENCE, 'no table')}" for m in labels}
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_samples = {m: s / base for m, s in samples.get("p0_rmse_a", {}).items()}
    else:
        frame["p0_ratio"] = np.nan
        notes.append(f"{REFERENCE} is not an admitted entry: no P0 ratio, and the ranking lacks the reference (checklist rule 4)")
    _intervals(frame, "p0_ratio", ratio_samples, ratio_problems, notes)
    for column in ("p1_h10_rmse_s", "p1_h30_rmse_s", "p1_whole_rmse_s"):
        frame[column] = points[column].astype(np.float64)
        _intervals(frame, column, samples.get(column, {}), problems.get(column, {}), notes)
    frame["p1_whole_collision_rate"] = [_whole(directories[m], "collision_rate") for m in labels]
    for column in ("p2_concave_growth_error", "p2_hysteresis_error", "p2_collisions_100", "p3_unstable_fraction",
                   "p4_collision_rate_synthetic", "p4_collision_rate_empirical"):
        frame[column] = points[column].astype(np.float64)
    frame["p3_unstable_fraction_large"] = [an._num(an._summary(directories[m], "p3"), "unstable_fraction_large") for m in labels]

    # composite over the admitted models with at least one input (P2, P3)
    for column in GROUPS["composite"]:
        frame[column] = np.nan
    inputs = list(an.COMPOSITE_COLUMNS.values())
    compared = [m for m, ok in zip(labels, np.isfinite(points[inputs].to_numpy(dtype=np.float64)).any(axis=1)) if ok]
    composite_ranks = None
    if compared:
        subset = points.loc[compared].reset_index(drop=True)
        values = pd.Series(an._composite(subset), index=compared)
        frame.loc[compared, "composite"] = values
        frame.loc[compared, "composite_rank"] = values.rank(method="average")
        replicates, composite_notes, seeds_of = composite_replicates(root, dataset, runs, compared, n_boot, seed)
        if replicates is not None:
            composite_ranks = rank_distribution(replicates)[RANK_COLUMNS]
            frame.loc[compared, "composite_rank_low"] = composite_ranks["rank_low"].reindex(compared)
            frame.loc[compared, "composite_rank_high"] = composite_ranks["rank_high"].reindex(compared)
            used = sorted({s for seeds in seeds_of.values() for s in seeds})
            several = sum(len(seeds) > 1 for seeds in seeds_of.values())
            notes.append(f"Composite rank interval: {n_boot} replicates over the in-range P2 profiles, the P3 speeds with an "
                         f"equilibrium and the training seeds (in every replicate each model takes one of its admitted runs with "
                         f"P2/P3 tables at random: seeds {_range(used)}, more than one seed for {several} of {len(compared)} "
                         f"models); the composite and its rank are those of seed {MAIN_SEED}.")
            notes += composite_notes
        else:
            notes.append(f"composite rank: no interval ({'; '.join(composite_notes) or 'no replicates'})")
        if len(compared) < len(labels):
            notes.append(f"no composite input (P2, P3) for {', '.join(m for m in labels if m not in compared)}: no composite")

    whole = samples.get("p1_whole_rmse_s", {})
    ranked = [m for m in labels if np.isfinite(frame.at[m, "p1_whole_rmse_s"])]
    p1_ranks = None
    if ranked and all(m in whole for m in ranked):
        p1_ranks = rank_distribution({m: whole[m] for m in ranked})[RANK_COLUMNS]
    elif ranked:
        notes.append("whole-event P1 rank: no rank distribution (driver replicates missing for "
                     f"{', '.join(m for m in ranked if m not in whole)})")

    spread: dict[str, dict] = {k: {} for k in labels}
    for label, values in _seed_spread(runs, labels, reference).items():
        spread[label].update(values)
    for label, values in _fold_spread(root, dataset, runs, labels).items():
        spread[label].update(values)
    for column in GROUPS["seeds"] + GROUPS["folds"]:
        frame[column] = [spread[m].get(column, np.nan) for m in labels]
    frame = _order(frame)
    return frame[_present(frame)].reset_index(drop=True), composite_ranks, p1_ranks, notes


# ----------------------------------------------------------------------------- driver leak, transfer, refused


def leak_table(root, dataset: str, runs: pd.DataFrame, main: pd.DataFrame | None = None, n_boot: int = N_BOOT,
               seed: int = SEED) -> tuple[pd.DataFrame, list[str]]:
    """The admitted runs under a leaky split (``event``), labelled "driver leak": P0 ``rmse_a``,
    P1 spacing errors with driver-bootstrap intervals (one draw per split and fold over its own
    test part) and, for comparison, the whole-event P1 error of the driver-split entry."""
    leak = _admitted(runs, LEAK)
    notes: list[str] = []
    if leak.empty:
        return pd.DataFrame(columns=LEAK_COLUMNS), notes
    driver = main.set_index("model")["p1_whole_rmse_s"] if main is not None and len(main) else pd.Series(dtype=np.float64)
    frames = []
    for (split, fold), part in leak.groupby(["split", "fold"], sort=True):
        directories = dict(zip(part["run"], part["directory"]))
        samples, problems = driver_replicates(directories, n_boot, seed)
        frame = pd.DataFrame({"model": part["model"].to_numpy(), "run": part["run"].to_numpy(), "label": part["label"].to_numpy()},
                             index=part["run"].to_numpy())
        frame["family"] = [an._family(d) for d in part["directory"]]
        frame["p0_rmse_a"] = [an._p0_rmse(d) for d in part["directory"]]
        for h in ("h10", "h30"):
            frame[f"p1_{h}_rmse_s"] = [an._num(an._summary(d, "p1"), h, "rmse_s") for d in part["directory"]]
        frame["p1_whole_rmse_s"] = [_whole(d) for d in part["directory"]]
        for column in DRIVER_METRICS:
            _intervals(frame, column, samples.get(column, {}), problems.get(column, {}), notes, f"driver leak {split}-fold{fold}: ")
        frame["p1_whole_collision_rate"] = [_whole(d, "collision_rate") for d in part["directory"]]
        frame["p1_whole_rmse_s_driver"] = frame["model"].map(driver).astype(np.float64)
        frames.append(frame)
    out = pd.concat(frames)
    out = out.assign(_w=out["p1_whole_rmse_s"].fillna(np.inf)).sort_values(["_w", "model", "run"], kind="stable")
    columns = [c for c in LEAK_COLUMNS if not (c.startswith("p1_h30") and not np.isfinite(out["p1_h30_rmse_s"].to_numpy(np.float64)).any())]
    return out[columns].reset_index(drop=True), notes


def transfer_table(e4: pd.DataFrame | None, dataset: str, main: pd.DataFrame, admitted: Collection[tuple[str, str]]) -> pd.DataFrame:
    """Whole-event relative degradation of the admitted models of ``dataset`` on every foreign
    target (rows of :func:`~follownet2.experiments.analysis.e4_transfer`); the ratio is left out
    when the run of the model trained on the target, its denominator, is not an admitted entry
    (``admitted``: pairs ``(dataset, model)``)."""
    if e4 is None or e4.empty or main.empty:
        return pd.DataFrame(columns=TRANSFER_COLUMNS)
    family = main.set_index("model")["family"]
    part = e4[(e4["horizon"] == "whole") & (e4["trained_on"] == dataset) & (e4["tested_on"] != dataset) & e4["model"].isin(family.index)]
    rows = []
    for row in part.itertuples(index=False):
        ratio, low, high, note = float(row.ratio), float(row.low), float(row.high), row.note if isinstance(row.note, str) else ""
        if (row.tested_on, row.model) not in admitted:
            ratio = low = high = float("nan")
            note = f"no admitted run of {row.model} trained on {row.tested_on} (the denominator)"
        rows.append({"model": row.model, "family": family[row.model], "tested_on": row.tested_on, "ratio": ratio, "low": low,
                     "high": high, "rmse_s": float(row.rmse_s), "rmse_s_own": float(row.rmse_s_own), "note": note})
    frame = pd.DataFrame(rows, columns=TRANSFER_COLUMNS)
    order = {m: i for i, m in enumerate(main["model"])}
    return frame.sort_values(["model", "tested_on"], key=lambda s: s.map(order) if s.name == "model" else s,
                             kind="stable").reset_index(drop=True)


def transfer_means(transfer: pd.DataFrame) -> pd.DataFrame:
    """Mean relative degradation per model over its targets with a ratio."""
    rows = []
    for model, block in transfer.groupby("model", sort=False):
        ratios = block["ratio"].to_numpy(dtype=np.float64)
        finite = ratios[np.isfinite(ratios)]
        rows.append({"model": model, "family": block["family"].iloc[0], "mean_degradation": float(finite.mean()) if finite.size else np.nan,
                     "n_targets": int(finite.size)})
    return pd.DataFrame(rows, columns=TRANSFER_MEAN_COLUMNS)


def refused_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Every refused run with its role and problems (ranked entries first)."""
    part = runs[~runs["admitted"].astype(bool)]
    frame = pd.DataFrame({"run": part["run"], "model": part["model"], "role": part["role"], "label": part["label"],
                          "problems": ["; ".join(p) for p in part["problems"]]}, columns=REFUSED_COLUMNS)
    order = frame["role"].map(_ROLE_ORDER).fillna(len(_ROLE_ORDER))
    return frame.assign(_o=order).sort_values(["_o", "run"], kind="stable").drop(columns="_o").reset_index(drop=True)


# ----------------------------------------------------------------------------- provenance


def _text(value) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{k} {_text(v)}" for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    return "n/a" if value is None else str(value)


def _counted(counter: Counter) -> str:
    if not counter:
        return "n/a"
    if len(counter) == 1:
        return next(iter(counter))
    return ", ".join(f"{value} ({n})" for value, n in sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _range(values: Iterable[int]) -> str:
    values = sorted(set(int(v) for v in values))
    if not values:
        return "none"
    if values == list(range(values[0], values[-1] + 1)) and len(values) > 2:
        return f"{values[0]}-{values[-1]}"
    return ", ".join(map(str, values))


def provenance(runs: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED, date: str | None = None) -> dict[str, str]:
    """Provenance footer of a dataset from its admitted ranked entries."""
    entries = _admitted(runs, ENTRY)
    fields = {k: Counter() for k in ("data_version", "reconstruction_method", "split_type", "seeds")}
    training, protocols = Counter(), Counter()
    hashes: dict[str, Counter] = {}
    for directory in entries["directory"]:
        train = an._json(Path(directory) / "train.json") or {}
        training[_text(train.get("git_revision"))] += 1
        for path in protocol_files(directory):
            data = read_json(path)
            block = data.get("checklist") or {}
            for key, counter in fields.items():
                counter[_text(block.get(key))] += 1
            protocols[_text(data.get("git_revision"))] += 1
            hashes.setdefault(path.stem, Counter())[_text(data.get("config_hash"))] += 1
    seeds = _admitted(runs, SEED_SPREAD)
    folds = _admitted(runs, FOLD_SPREAD)
    order = sorted(hashes, key=lambda s: (s.split("_", 1)[0], "_" in s, s))
    return {
        "data version": _counted(fields["data_version"]),
        "reconstruction method": _counted(fields["reconstruction_method"]),
        "split": f"{MAIN_SPLIT}, fold {MAIN_FOLD}, training seed {MAIN_SEED} (checklist split type: {_counted(fields['split_type'])})",
        "seeds": (f"checklist: {_counted(fields['seeds'])}; seed spread: "
                  f"{('training seeds ' + _range([MAIN_SEED, *seeds['seed']])) if len(seeds) else 'none'}; fold spread: "
                  f"{('clean arms of folds ' + _range(folds['fold'])) if len(folds) else 'none'}"),
        "git revision of the runs": f"training: {_counted(training)}; protocols: {_counted(protocols)}",
        "protocol configuration hashes": "; ".join(f"{stem}: {_counted(hashes[stem])}" for stem in order) or "n/a",
        "bootstrap": (f"{n_boot} replicates, seed {seed}: drivers for P0 and P1 (common random numbers), in-range P2 profiles, "
                      f"P3 speeds with an equilibrium and training seeds for the composite rank (Generator [{seed}, 1])"),
        "date": date or _dt.date.today().isoformat(),
    }


# ----------------------------------------------------------------------------- boards


def build_board(root, dataset: str, n_boot: int = N_BOOT, seed: int = SEED, runs: pd.DataFrame | None = None,
                e4: pd.DataFrame | None = None, admitted: Collection[tuple[str, str]] | None = None,
                date: str | None = None) -> Board:
    """The leaderboard of ``dataset``. ``runs`` (from :func:`validate_dataset`), ``e4`` (the
    whole-event rows of :func:`~follownet2.experiments.analysis.e4_transfer`) and ``admitted``
    (pairs ``(dataset, model)`` of the admitted entries of every dataset, the transfer
    denominators) are computed when not given."""
    root = Path(root)
    runs = validate_dataset(root, dataset) if runs is None else runs
    main, composite_ranks, p1_ranks, notes = main_table(root, dataset, runs, n_boot, seed)
    leak, leak_notes = leak_table(root, dataset, runs, main, n_boot, seed)
    has_transfer = any(path.name.startswith("p1_") for d in _admitted(runs, ENTRY)["directory"] for path in protocol_files(d))
    if has_transfer and e4 is None:
        e4 = an.e4_transfer(root, None, n_boot, seed, horizons=("whole",))
    if has_transfer and admitted is None:
        admitted = _admitted_entries(root, sorted(set(e4["tested_on"])) if e4 is not None and len(e4) else [])
    transfer = transfer_table(e4 if has_transfer else None, dataset, main, admitted or ())
    return Board(dataset=dataset, main=main, leak=leak, transfer=transfer, refused=refused_table(runs), runs=runs,
                 provenance=provenance(runs, n_boot, seed, date), composite_ranks=composite_ranks, p1_ranks=p1_ranks,
                 notes=notes + leak_notes, n_boot=n_boot, seed=seed)


def _admitted_entries(root, datasets: Iterable[str], validated: Mapping[str, pd.DataFrame] | None = None) -> set[tuple[str, str]]:
    out = set()
    for dataset in datasets:
        runs = validated[dataset] if validated is not None and dataset in validated else validate_dataset(root, dataset)
        out |= {(dataset, m) for m in _admitted(runs, ENTRY)["model"]}
    return out


def available_datasets(root) -> list[str]:
    """Datasets with at least one run directory under ``root``."""
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*/*/*/seed*"))) if root.is_dir() else []


def build_leaderboard(root, datasets: Sequence[str] | None = None, n_boot: int = N_BOOT, seed: int = SEED,
                      date: str | None = None) -> dict[str, Board]:
    """The boards of ``datasets`` (default: every dataset under ``root``). Every run directory of
    every dataset is validated (the transfer ratios need the admission of the runs on the targets)."""
    root = Path(root)
    available = available_datasets(root)
    selected = available if datasets is None else list(dict.fromkeys(datasets))
    unknown = [d for d in selected if d not in available]
    if unknown:
        raise ValueError(f"no runs of {', '.join(unknown)} under {root} (datasets with runs: {', '.join(available) or 'none'})")
    validated = {d: validate_dataset(root, d) for d in available}
    admitted = _admitted_entries(root, available, validated)
    has_transfer = any(path.name.startswith("p1_") for d in selected for directory in _admitted(validated[d], ENTRY)["directory"]
                       for path in protocol_files(directory))
    e4 = an.e4_transfer(root, None, n_boot, seed, horizons=("whole",)) if has_transfer else None
    return {d: build_board(root, d, n_boot, seed, runs=validated[d], e4=e4, admitted=admitted, date=date) for d in selected}


def summary_rows(boards: Mapping[str, Board], top: int = 3) -> pd.DataFrame:
    """One row per board: counts and the first models of the main table."""
    rows = []
    for dataset, board in boards.items():
        counts = board.counts()
        head = board.main["model"].head(top).tolist() if len(board.main) else []
        rows.append({"dataset": dataset, **counts, "ranked_by": "composite" if "composite" in board.main else "P1 whole",
                     "top": ", ".join(head), "transfer_models": int(board.transfer["model"].nunique()) if len(board.transfer) else 0})
    return pd.DataFrame(rows)

