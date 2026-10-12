"""Collect the experiments of milestone 4 into ``results/m4/REPORT.md`` and CSV tables.

    python scripts/m4_report.py [root=results/models] [out=results/m4] [n_boot=1000] [seed=0]
                                [splits=data/splits] [events=data/events] [models=main]

E1-E4 of docs/experiments.md, section 3 (``follownet2/experiments/analysis.py``), and E3b (composite
variants with realism inputs, their sensitivity and the Pareto view), from the run directories
``<root>/<dataset>/reconstructed/<split>-fold<k>/<label>/seed<s>/``. A campaign that has not run
(or not finished) is stated in the report; its tables stay empty. ``models=main`` (default)
restricts E1 and E3 to the reference models of the main campaign (``MODELS`` of
``scripts/m2.py``; the ablation runs of ``scripts/m4.py ablation`` share the directory of fold 0),
``models=all`` takes every finished run. Writes ``REPORT.md``, ``e1_ratios.csv``,
``e2_paired.csv``, ``e2_unpaired.csv``, ``e3_rankings.csv``, ``e3_kendall.csv`` (with the
spacing-against-speed rows), ``e3_composite_variants.csv``, ``e3_composite_sensitivity.csv``,
``e3_pareto.csv`` and ``e4_transfer.csv``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from follownet2.experiments import analysis as an
from follownet2.utils import REPO_ROOT, resolve_path

FAMILY_ORDER = {"baseline": 0, "mechanistic": 1, "network": 2}
P_COLUMNS = {"p", "p_bootstrap", "p_wilcoxon", "p_holm"}
#: E3 columns shown in the report (all of them are in e3_rankings.csv)
E3_SHOWN = {
    "p0_rmse_a": "P0", "p1_h10_rmse_s": "P1 h10", "p1_h30_rmse_s": "P1 h30", "p1_whole_rmse_s": "P1 whole",
    "p2_concave_growth_error": "P2 concave", "p2_hysteresis_error": "P2 hyst.", "p2_collisions_100": "P2 coll.",
    "p3_unstable_fraction": "P3 unst.", "p4_collision_rate_synthetic": "P4 syn.", "p4_collision_rate_empirical": "P4 emp.",
    "composite": "composite",
}
DEFAULTS = {"root": "results/models", "out": "results/m4", "n_boot": "1000", "seed": "0", "splits": "data/splits",
            "events": "data/events", "models": "main"}


def arguments(argv: list[str] | None) -> dict:
    args = dict(DEFAULTS)
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            raise SystemExit(f"unknown argument {item!r}; usage: m4_report.py " + " ".join(f"[{k}=...]" for k in DEFAULTS))
        args[key] = value
    if args["models"] not in ("main", "all"):
        raise SystemExit(f"models={args['models']!r}: use models=main or models=all")
    return {"root": resolve_path(args["root"]), "out": resolve_path(args["out"]), "n_boot": int(args["n_boot"]),
            "seed": int(args["seed"]), "splits": resolve_path(args["splits"]), "events": resolve_path(args["events"]),
            "models": args["models"]}


# ----------------------------------------------------------------------------- formatting


def fmt(value, column: str = "") -> str:
    """A cell: 3 decimals, counts as integers, p-values below 0.001 as ``<0.001``, missing ``n/a``."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        if column in P_COLUMNS and value < 0.001:
            return "<0.001"
        if column.startswith("rank"):  # average ranks and their percentiles
            return f"{value:.4g}"
        if value.is_integer() and (column.startswith("n_") or "collisions" in column or column in {"runs", "P2 coll."}):
            return str(int(value))
        return f"{value:.3f}"
    return str(value).replace("|", "/").replace("\n", " ")


def table(frame: pd.DataFrame, columns: list[str] | None = None, names: dict | None = None) -> str:
    """Markdown table of ``frame`` (``columns`` in that order, headers renamed by ``names``)."""
    frame = frame if columns is None else frame[columns]
    if frame.empty:
        return "(no rows)\n"
    headers = [str((names or {}).get(c, c)) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v, c) for v, c in zip(row, headers)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def by_family(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    order = frame["family"].map(FAMILY_ORDER).fillna(len(FAMILY_ORDER))
    return frame.assign(_order=order).sort_values(["_order", value, "model"], kind="stable").drop(columns="_order")


def interval(low: float, high: float) -> str:
    return "n/a" if not (np.isfinite(low) and np.isfinite(high)) else f"[{low:.3f}, {high:.3f}]"


# ----------------------------------------------------------------------------- campaigns


def main_labels() -> tuple[str, ...] | None:
    """Model labels of the main campaign (scripts/m2.py), when importable."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from m2 import MODELS
    except Exception:  # noqa: BLE001 - the counts of the campaign table and the label filter depend on it
        return None
    finally:
        sys.path.pop(0)
    return tuple(MODELS)


def main_models() -> int | None:
    """Number of model labels of the main campaign (scripts/m2.py), when importable."""
    labels = main_labels()
    return None if labels is None else len(labels)


def label_filter(which: str):
    """``labels_of(root, dataset)``: the labels of E1 and E3, ``None`` for every finished run
    (``models=all``, or when scripts/m2.py cannot be imported)."""
    known = main_labels() if which == "main" else None

    def labels_of(root: Path, dataset: str) -> list[str] | None:
        return None if known is None else [m for m in an.run_labels(root, dataset) if m in known]

    return labels_of


def completeness(root: Path) -> pd.DataFrame:
    """Expected and finished runs of every campaign of docs/experiments.md, section 1 (the main
    campaign counts the labels of scripts/m2.py only, when known: the ablations share its directory)."""
    known = main_labels()
    n_models = None if known is None else len(known)
    labels = (*an.SUBSET, an.REFERENCE)
    found = {
        "main": sum(len([m for m in an.run_labels(root, d) if known is None or m in known]) for d in an.MAIN_DATASETS),
        "seeds": sum(len(an.run_labels(root, d, seed=s)) for d in an.MAIN_DATASETS for s in (1, 2)),
        "paired": sum(len(set(an.run_labels(root, d, arm, k)) & set(an.SUBSET)) for d in an.LEAK_DATASETS
                      for arm in (an.CLEAN, an.LEAKY) for k in an.FOLDS),
        "event": sum(len(set(an.run_labels(root, d, "event", 0)) & set(an.SUBSET)) for d in an.LEAK_DATASETS),
        "transfer (training)": sum(len(set(an.run_labels(root, d)) & set(labels)) for d in an.TRANSFER_DATASETS
                                   if d not in an.MAIN_DATASETS),
        "transfer (P1 files)": sum(
            (an.run_dir(root, d, m) / f"p1_{o}.json").exists()
            for d in an.TRANSFER_DATASETS for m in labels for o in an.TRANSFER_DATASETS if o != d
        ),
    }
    expected = {
        "main": None if n_models is None else n_models * len(an.MAIN_DATASETS),
        "seeds": None if n_models is None else 2 * n_models * len(an.MAIN_DATASETS),
        "paired": 2 * len(an.FOLDS) * len(an.SUBSET) * len(an.LEAK_DATASETS),
        "event": len(an.SUBSET) * len(an.LEAK_DATASETS),
        "transfer (training)": len(labels) * (len(an.TRANSFER_DATASETS) - len(an.MAIN_DATASETS)),
        "transfer (P1 files)": len(labels) * len(an.TRANSFER_DATASETS) * (len(an.TRANSFER_DATASETS) - 1),
    }
    rows = []
    for name, n in found.items():
        total = expected[name]
        status = "not run" if n == 0 else ("complete" if total is not None and n >= total else "partial")
        rows.append({"campaign": name, "expected": total if total is not None else float("nan"), "found": n, "status": status})
    return pd.DataFrame(rows)


def section_campaigns(root: Path) -> tuple[str, pd.DataFrame]:
    done = completeness(root)
    text = "\n## Campaigns\n\n" + table(done)
    missing = done.loc[done["status"] != "complete", "campaign"].tolist()
    if missing:
        text += ("\nNot complete: " + ", ".join(f"`{m}` ({s})" for m, s in zip(done["campaign"], done["status"]) if m in missing)
                 + ". Tables that need them are marked *not available* below. `transfer (P1 files)` counts the "
                 "`p1_<other>.json` of SUBSET and persistence over the six datasets.\n")
    status = an.campaign_status(root)
    if not status.empty:
        text += "\nRuns found per dataset:\n\n" + table(status)
    stale = []
    for dataset in an.run_datasets(root):
        for label in an.run_labels(root, dataset):
            files = an.stale_protocols(an.run_dir(root, dataset, label))
            if files:
                stale.append(f"{dataset}/{label} ({', '.join(files)})")
    if stale:
        text += ("\nProtocol files computed from an earlier checkpoint than the current `train.json` (stale, decision "
                 "M4-5; their values are used and flagged until the protocols are recomputed): " + "; ".join(stale) + ".\n")
    return text, done


# ----------------------------------------------------------------------------- E1-E4


def section_e1(root: Path, n_boot: int, seed: int, labels_of=None) -> tuple[str, pd.DataFrame]:
    text = ("\n## E1: one-step error relative to persistence (H5.1)\n\n"
            f"`ratio` = RMSE_a(model) / RMSE_a(persistence) of P0 on fold 0 of the driver split (seed 0); `low`, `high`: 95 % "
            f"interval from {n_boot} driver-bootstrap replicates with common random numbers (the follower counts are drawn once "
            "per dataset and applied to every table). `seed_min`/`seed_max`: range over training seeds 0-2; `folds_min`/"
            "`folds_max`: range over the clean arms of folds 0-4 (probe events, SUBSET only); n/a with fewer than two runs.\n")
    datasets = [d for d in an.run_datasets(root) if (an.run_dir(root, d, an.REFERENCE) / "test_p0.events.parquet").exists()]
    if not datasets:
        return text + "\nNot available: no driver-split fold-0 run of persistence.\n", pd.DataFrame(columns=an.E1_COLUMNS)
    frames = []
    for dataset in datasets:
        frame = an.e1_ratios(root, dataset, labels_of(root, dataset) if labels_of else None, n_boot, seed)
        frames.append(frame)
        text += f"\n### {dataset}\n\n"
        below = frame[(frame["high"] < 1.0)]["model"].tolist()
        text += table(by_family(frame, "ratio"), ["model", "family", "rmse_a", "ratio", "low", "high", "n_drivers", "seed_min",
                                                  "seed_max", "n_seeds", "folds_min", "folds_max", "n_folds"])
        text += f"\nInterval entirely below 1 (better than persistence): {', '.join(below) if below else 'none'}.\n"
        if frame["n_seeds"].max() < 2:
            text += "Seed spread: not available (campaign `seeds` has not run).\n"
        if frame["n_folds"].max() < 2:
            text += "Fold spread: not available (campaign `paired` has not produced two clean folds with persistence on the probe events).\n"
    return text, pd.concat(frames, ignore_index=True)


def section_e2(root: Path, n_boot: int, seed: int, splits: Path, events: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    text = ("\n## E2: leakage inflation (H5.2)\n\n"
            "Paired design (decision R2): both arms scored on the same probe events of folds 0-4, pooled; `inflation` = (clean - "
            "leaky) / clean (positive: the leak lowers the error); `mean_difference` (clean - leaky), its interval and `p_bootstrap` "
            f"from `paired_difference` ({n_boot} driver replicates, common random numbers), `p_wilcoxon` the signed-rank test on the "
            "per-driver differences, `p_holm` the Holm adjustment of `p_bootstrap` over the models of a dataset, subset and metric. "
            "Subset `all`: every probe event; `affected`: the probe events of drivers with at least one event in the leaky "
            "training part of the same fold.\n")
    datasets = sorted(set(an.LEAK_DATASETS) | {d for k in an.FOLDS for d in an.run_datasets(root, an.CLEAN, k)})
    paired_frames, unpaired_frames = [], []
    for dataset in datasets:
        text += f"\n### {dataset}\n\n"
        affected, reason = None, ""
        try:
            affected = an.affected_followers(dataset, splits, events)
        except (OSError, KeyError, ValueError) as error:
            reason = f"{type(error).__name__}: {error}"
        frame = an.e2_paired(root, dataset, an.SUBSET, n_boot, seed, affected=affected)
        paired_frames.append(frame)
        if frame.empty:
            text += f"Paired comparison: not available (no fold with both arms of a SUBSET model under `{an.CLEAN}-fold<k>`; campaign `paired`).\n"
        else:
            folds = sorted({k for k in an.FOLDS for m in an.SUBSET if (an.run_dir(root, dataset, m, an.CLEAN, k) / "train.json").exists()})
            text += f"Folds with runs: {', '.join(map(str, folds))}; models: {frame['model'].nunique()} of {len(an.SUBSET)}.\n"
            if affected is None:
                text += f"Affected subset: not available ({reason}).\n"
            else:
                share = frame[(frame["metric"] == "p0_rmse_a")].pivot_table(index="model", columns="subset", values="n_events", aggfunc="first")
                if "affected" in share and "all" in share and len(share):
                    row = share.iloc[0]
                    text += (f"Affected probe events: {int(row['affected'])} of {int(row['all'])} ({row['affected'] / row['all']:.1%}) "
                             f"for {share.index[0]}, drivers {int(frame['n_affected_drivers'].dropna().iloc[0])}.\n")
            for subset in an.E2_SUBSETS:
                part = frame[frame["subset"] == subset]
                if part.empty:
                    continue
                text += f"\nSubset `{subset}`:\n\n" + table(part, ["model", "family", "metric", "clean", "leaky", "inflation", "mean_difference",
                                                                   "low", "high", "p_bootstrap", "p_wilcoxon", "p_holm", "n_drivers", "n_events", "n_folds"])
            text += "\nClass means:\n\n" + table(an.e2_class_means(frame), ["subset", "metric", "family", "mean_inflation", "min_inflation",
                                                                         "max_inflation", "mean_difference", "n_models"])
        unpaired = an.e2_unpaired(root, dataset, an.SUBSET)
        unpaired_frames.append(unpaired)
        if unpaired.empty:
            text += "\nUnpaired comparison (event split against driver split, fold 0): not available (no SUBSET model with runs of both splits; campaign `event`).\n"
        else:
            text += ("\nUnpaired comparison (event split against driver split on fold 0, each on its own test part; `inflation` = "
                     "(driver - event) / driver, no test):\n\n" + table(unpaired, ["model", "family", "metric", "driver", "event", "inflation",
                                                                                   "difference", "n_events_driver", "n_events_event"]))
    paired = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame(columns=an.E2_COLUMNS)
    unpaired = pd.concat(unpaired_frames, ignore_index=True) if unpaired_frames else pd.DataFrame(columns=an.E2_UNPAIRED_COLUMNS)
    return text, paired, unpaired


E3_PROTOCOLS = ("p2", "p3", "p4")


def e3_datasets(root: Path, labels_of=None, incomplete: dict | None = None) -> list[str]:
    """Datasets whose fold-0 runs carry the protocols P2-P4 (the main campaign): at least two runs with
    ``p2.json`` and every run of the labels (``labels_of``, default every finished run) with P2-P4. A
    dataset whose campaign is still evaluating (some runs without P2-P4) is left out and, when
    ``incomplete`` is given, entered there with the number of complete runs and the first missing ones."""
    out = []
    for dataset in an.run_datasets(root):
        labels = labels_of(root, dataset) if labels_of else None
        labels = an.run_labels(root, dataset) if labels is None else labels
        if sum((an.run_dir(root, dataset, m) / "p2.json").exists() for m in labels) < 2:
            continue
        missing = [m for m in labels if not all((an.run_dir(root, dataset, m) / f"{p}.json").exists() for p in E3_PROTOCOLS)]
        if missing:
            if incomplete is not None:
                incomplete[dataset] = f"{len(labels) - len(missing)} of {len(labels)} runs with P2-P4 (missing e.g. {', '.join(missing[:3])})"
            continue
        out.append(dataset)
    return out


def section_e3(root: Path, n_boot: int, seed: int, labels_of=None) -> tuple[str, pd.DataFrame, pd.DataFrame, dict]:
    """The E3 text, the rankings and Kendall tables, and ``{dataset: (rankings, replicates)}`` for E3b."""
    text = ("\n## E3: rankings across protocols (H5.3, H5.4)\n\n"
            "Point values of fold 0, seed 0 (lower is better; P2 and the synthetic P4 over the profiles and scenarios inside the "
            "speed range of the training data; P1 whole and P0 from `train.json`). Kendall's tau-b between every pair of rankings: "
            f"`tau` and `p` on the point values, `low`/`high` the 95 % interval and `tau_boot` the mean over {n_boot} replicates "
            "(drivers for P0, P1 and the empirical P4; in-range profiles for P2 and the composite, speeds with an equilibrium for P3, "
            "in-range scenarios for the synthetic P4). A model without a value ranks last.\n")
    incomplete: dict[str, str] = {}
    datasets = e3_datasets(root, labels_of, incomplete)
    missing = [d for d in an.MAIN_DATASETS if d not in datasets and d not in incomplete]
    if missing:
        text += f"\nNot available for {', '.join(missing)}: no fold-0 runs with P2-P4 (campaign `main`).\n"
    if incomplete:
        text += ("\nLeft out while their campaign is still evaluating (a ranking needs every model): "
                 + "; ".join(f"{d}: {why}" for d, why in incomplete.items()) + ".\n")
    rankings_frames, kendall_frames, cache = [], [], {}
    for dataset in datasets:
        started = time.time()
        rankings = an.e3_rankings(root, dataset, labels_of(root, dataset) if labels_of else None)
        replicates = an.e3_replicates(root, dataset, rankings, n_boot, seed)
        cache[dataset] = (rankings, replicates)
        kendall = an.e3_kendall(root, dataset, n_boot, seed, rankings=rankings, replicates=replicates)
        speed = an.e3_spacing_vs_speed(root, dataset, n_boot, seed, rankings=rankings, replicates=replicates)
        rankings_frames.append(rankings)
        kendall_frames += [kendall, speed.assign(ranking_a="P1 " + speed["horizon"] + " rmse_s",
                                                 ranking_b="P1 " + speed["horizon"] + " rmse_v").reindex(columns=an.KENDALL_COLUMNS)]
        text += f"\n### {dataset}\n\n" + table(by_family(rankings, "p1_whole_rmse_s"), ["model", "family", *E3_SHOWN, "stale"], E3_SHOWN)
        reasons = replicates[1]
        if reasons:
            text += "\nNo bootstrap replicates (point values and point tau only):\n\n" + "".join(f"* `{c}`: {r}\n" for c, r in reasons.items())
        text += "\nKendall tau between rankings:\n\n" + table(kendall, ["ranking_a", "ranking_b", "tau", "low", "high", "p", "tau_boot", "n_models"])
        text += "\nSpacing against speed (H5.4): tau between the `rmse_s` and the `rmse_v` ranking of P1:\n\n"
        text += table(speed, ["horizon", "tau", "low", "high", "p", "tau_boot", "n_models"])
        text += "\nTop three (`tied`: further models tied with the third):\n\n" + table(an.e3_top3(rankings), ["ranking", "top1", "top2", "top3", "values", "tied"])
        text += "\nEntries into and exits from the top three:\n\n" + "".join(f"* {line.split(': ', 1)[1]}\n" for line in an.e3_top3_changes(rankings))
        print(f"E3 {dataset}: {time.time() - started:.1f} s", flush=True)
    rankings = pd.concat(rankings_frames, ignore_index=True) if rankings_frames else pd.DataFrame(columns=an.E3_COLUMNS)
    kendall = pd.concat([k for k in kendall_frames if not k.empty], ignore_index=True) if any(not k.empty for k in kendall_frames) \
        else pd.DataFrame(columns=an.KENDALL_COLUMNS)
    return text, rankings, kendall, cache


# ----------------------------------------------------------------------------- E3b


def _rank_interval(low: float, high: float) -> str:
    """A rank interval at one decimal, as on the leaderboard."""
    return "n/a" if not (np.isfinite(low) and np.isfinite(high)) else f"[{low:.1f}, {high:.1f}]"


def _tau_text(row) -> str:
    """``tau [low, high]`` of tau(P0, composite)."""
    if not np.isfinite(row["tau_p0"]):
        return "n/a"
    return f"{row['tau_p0']:.3f}" + (f" {interval(row['tau_p0_low'], row['tau_p0_high'])}" if np.isfinite(row["tau_p0_low"]) else "")


def p0_sign_sentence(sensitivity: pd.DataFrame) -> tuple[str, bool | None]:
    """Whether tau(P0, composite) keeps the sign of v1 (the headline of E3) under every variant,
    and the range and the intervals over every composite of the sensitivity table; the second
    value is True when the sign is kept under every variant (None without a v1 value)."""
    rows = sensitivity.set_index("variant_or_leave_one_out")
    if "v1" not in rows.index or not np.isfinite(rows.at["v1", "tau_p0"]):
        return "Kendall tau of P0 against the composite: not available (no P0 or composite ranking).", None
    sign = np.sign(rows.at["v1", "tau_p0"])
    variants = [v for v in an.VARIANTS if v in rows.index]
    flipped = [v for v in variants if np.isfinite(rows.at[v, "tau_p0"]) and np.sign(rows.at[v, "tau_p0"]) != sign]
    listed = ", ".join(f"{v} {_tau_text(rows.loc[v])}" for v in variants)
    kept = not flipped
    text = (f"Kendall tau of P0 against the composite: {listed}. The sign of the headline (v1, "
            f"{'negative' if sign < 0 else 'positive' if sign > 0 else 'zero'}) is "
            + ("kept under every variant" if kept else f"reversed under {', '.join(flipped)}"))
    taus = rows["tau_p0"].to_numpy(dtype=np.float64)
    finite = np.isfinite(taus)
    if finite.any():
        same = int((np.sign(taus[finite]) == sign).sum())
        low, high = rows["tau_p0_low"].to_numpy(dtype=np.float64), rows["tau_p0_high"].to_numpy(dtype=np.float64)
        has = np.isfinite(low) & np.isfinite(high)
        excludes = has & ((high < 0) | (low > 0))
        open_rows = [label for label, h, e in zip(rows.index, has, excludes) if h and not e]
        text += (f"; over all {len(rows)} composites of the sensitivity table it lies between {np.nanmin(taus):.3f} and "
                 f"{np.nanmax(taus):.3f} with the sign of v1 in {same} of {int(finite.sum())}, and its 95 % interval excludes 0 "
                 f"in {int(excludes.sum())} of {int(has.sum())}" + (f" (not for {', '.join(open_rows)})" if open_rows else ""))
    return text + ".", kept


def section_e3b(root: Path, cache: dict, n_boot: int, seed: int) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    text = ("\n## E3b: composite variants and sensitivity\n\n"
            "The composite of E3 (`v1`: mean normalised rank of the concave-growth error, the hysteresis error, the unstable "
            "fraction and the 100-vehicle collisions) has no realism input: a follower that keeps a large gap and never reacts "
            "is stable and never collides. The variants add the whole-event P1 spacing RMSE (`v1+p1`), the relative "
            "fundamental-diagram error of P2 (`fd_error_rel`; `v1+fd`) or both (`v2`, six inputs, equal weights). `borda v2` "
            "is the Borda count of v2 (sum of n - rank, normalised to [0, 1] with 0 = best); with average ranks on ties it is an "
            "affine function of the rank sum, so it equals v2 by construction. Leave one out: v2 without each of its inputs; "
            "weights: v2 with weight 2 on the unstable fraction, on the collisions and on both (the stability and collision "
            "inputs). `fd_error_rel` is pooled over the samples of every P2 profile (the 20 OpenACC windows and the pulse: P2 "
            "has no in-range version), compares the density bins with at least 200 samples on both sides (`n_FD_bins`, which "
            "differs between models), and has no per-profile version to resample: it is held at its point value in the "
            f"replicates. Intervals ({n_boot} replicates, seed {seed}): the rank interval of v2 and the interval of tau(P0, "
            "composite) resample the in-range profiles and the speeds with an equilibrium, as in E3, and the drivers of the "
            "whole-event P1 input with the draw of P0 (common random numbers); the v1 row reproduces the P0-composite row of "
            "E3. Ranks: 1 = best, average ranks on ties. `top3 changes`: entries into and exits from the top three (ties at the "
            "third position counted in) against v1 for the variants and against v2 for Borda, leave one out and weights. Pareto "
            "view: `on front` marks the models that no other model matches or beats in both the whole-event P1 error and "
            "composite v2 (both lower is better; the P1 error is one of the six inputs of v2).\n")
    if not cache:
        empty = (pd.DataFrame(columns=an.VARIANTS_COLUMNS), pd.DataFrame(columns=an.SENSITIVITY_COLUMNS),
                 pd.DataFrame(columns=an.PARETO_COLUMNS))
        return text + "\nNot available: no dataset with E3 rankings.\n", *empty
    variants_frames, sensitivity_frames, pareto_frames, kept = [], [], [], {}
    for dataset, (rankings, replicates) in cache.items():
        variants = an.e3_composite_variants(root, dataset, rankings=rankings, replicates=replicates)
        sensitivity = an.e3_composite_sensitivity(root, dataset, rankings=rankings, replicates=replicates)
        pareto = an.pareto_table(root, dataset, rankings=rankings)
        variants_frames.append(variants)
        sensitivity_frames.append(sensitivity)
        pareto_frames.append(pareto)
        shown = variants.assign(rank_v2_ci=[_rank_interval(lo, hi) for lo, hi in zip(variants["rank_v2_low"], variants["rank_v2_high"])])
        text += f"\n### {dataset}\n\nComposite variants (best v2 first):\n\n"
        text += table(shown, ["model", "family", an.P1_INPUT, an.FD_INPUT, "fd_bins_compared", *(f"composite_{v}" for v in an.VARIANTS),
                              "composite_borda_v2", *(f"rank_{v}" for v in an.VARIANTS), "rank_v2_ci"],
                      {an.P1_INPUT: "P1 whole (m)", an.FD_INPUT: "FD rel.", "fd_bins_compared": "n_FD_bins",
                       **{f"composite_{v}": v for v in an.VARIANTS}, "composite_borda_v2": "borda v2",
                       **{f"rank_{v}": f"rank {v}" for v in an.VARIANTS}, "rank_v2_ci": "rank v2 [95%]"})
        top5 = ", ".join(variants["model"].head(5))
        text += f"\nTop five under v2: {top5}.\n"
        shown = sensitivity.assign(tau_p0_ci=[interval(lo, hi) for lo, hi in zip(sensitivity["tau_p0_low"], sensitivity["tau_p0_high"])])
        columns = ["variant_or_leave_one_out", "inputs", "kendall_tau_vs_v1", "kendall_tau_vs_v2", "top3", "top3_changes", "tau_p0", "tau_p0_ci"]
        if shown["note"].astype(str).str.len().gt(0).any():
            columns.append("note")
        text += "\nSensitivity (Kendall tau-b on the point values, a missing value last):\n\n"
        text += table(shown, columns, {"variant_or_leave_one_out": "composite", "kendall_tau_vs_v1": "tau vs v1",
                                       "kendall_tau_vs_v2": "tau vs v2", "top3_changes": "top3 changes", "tau_p0": "tau(P0)",
                                       "tau_p0_ci": "tau(P0) [95%]"})
        sentence, kept[dataset] = p0_sign_sentence(sensitivity)
        text += "\n" + sentence + "\n"
        shown = pareto.assign(front=np.where(pareto["on_front"].astype(bool), "yes", ""))
        text += "\nPareto view (by the whole-event P1 error):\n\n"
        text += table(shown, ["model", "family", an.P1_INPUT, "composite_v2", "unstable_fraction", "collisions_100", "front"],
                      {an.P1_INPUT: "P1 whole (m)", "composite_v2": "v2", "unstable_fraction": "P3 unst.", "collisions_100": "P2 coll.",
                       "front": "on front"})
        text += f"\nOn the front: {', '.join(pareto.loc[pareto['on_front'].astype(bool), 'model']) or 'none'}.\n"
    known = {d: k for d, k in kept.items() if k is not None}
    if known:
        reversed_ = [d for d, k in known.items() if not k]
        text += ("\nAcross datasets: the sign of tau(P0, composite) " + ("is kept under every variant on " + ", ".join(known)
                 if not reversed_ else "changes under at least one variant on " + ", ".join(reversed_)) + ".\n")
    return (text, pd.concat(variants_frames, ignore_index=True), pd.concat(sensitivity_frames, ignore_index=True),
            pd.concat(pareto_frames, ignore_index=True))


def section_e3c(root: Path) -> tuple[str, pd.DataFrame]:
    """E3c: the training ablations of the networks next to their reference entries (decision M4-7)."""
    text = ("\n## E3c: training ablations of the networks\n\n"
            "Campaign `ablation` (decision M4-7): rollout-loss horizons of 15 s and 30 s (`*_loop_h15`, `*_loop_h30`), "
            "a 3 s input history (`*_hist30`), a four-fold width (`*_wide`) and the stability penalty evaluated in a "
            "neighbourhood of every equilibrium (`stability_mlp_nbhd`); fold 0, seed 0 of the driver split, same training "
            "budget as the reference entries. `unstable` = numerical P3 fraction at 0.05 m/s, `large` at 1 m/s; collisions "
            "of P2 over the in-range profiles, P4 synthetic in range. The 15 s and 30 s horizons do not exist on "
            "FollowNet-highD (15 s events).\n")
    frames = []
    for dataset in an.run_datasets(root):
        frame = an.ablation_table(root, dataset)
        if frame.empty:
            continue
        frames.append(frame)
        text += f"\n### {dataset}\n\n" + table(frame, ["group", "model", "p0_rmse_a", "p1_h10_rmse_s", "p1_whole_rmse_s",
                                                      "p1_whole_collision_rate", "p2_collisions_100", "p3_n_equilibria",
                                                      "p3_unstable_fraction", "p3_unstable_fraction_large", "p3_max_gain",
                                                      "p4_collision_rate_synthetic", "p4_collision_rate_empirical"])
    if not frames:
        text += "\nNot available (campaign `ablation` has not run).\n"
        return text, pd.DataFrame()
    return text, pd.concat(frames, ignore_index=True)


def section_e4(root: Path, n_boot: int, seed: int) -> tuple[str, pd.DataFrame]:
    text = ("\n## E4: transfer (P5)\n\n"
            "Relative degradation `rmse_s(A -> B) / rmse_s(B -> B)` of P1 for SUBSET and persistence (rows: model and training "
            f"dataset A, columns: test dataset B; 1 = no loss); intervals from {n_boot} driver replicates over the test drivers of B "
            "(one draw per target, common to both tables), shown as `[low, high]` where the per-window tables of both evaluations exist.\n")
    labels = main_labels() or (*an.SUBSET, an.REFERENCE)  # every reference entry once the full matrix exists
    frame = an.e4_transfer(root, list(labels), n_boot, seed)
    if frame.empty:
        return text + "\nNot available: no `p1.json` of a SUBSET run.\n", frame
    sources = sorted(frame["trained_on"].unique())
    targets = sorted(frame["tested_on"].unique())
    if len(targets) < 2:
        text += "\nOnly one dataset with runs: no transfer pair.\n"
    missing = [d for d in an.TRANSFER_DATASETS if d not in sources]
    if missing:
        text += f"\nNo trained SUBSET runs on {', '.join(missing)} (campaign `transfer` has not run): the matrix covers {', '.join(sources)}.\n"
    foreign = frame[frame["trained_on"] != frame["tested_on"]]
    without = foreign["low"].isna().sum()
    if without:
        notes = foreign.loc[foreign["low"].isna(), "note"].iloc[0]
        text += (f"\n{without} of {len(foreign)} foreign entries have point values only: the per-window tables of the transfer "
                 f"evaluation (`p1_<B>.<h>.parquet`, written since decision M4-5) are missing or do not match their summary "
                 f"(e.g. {notes}).\n")
    for horizon in sorted(frame["horizon"].unique(), key=lambda h: an.HORIZONS.index(h) if h in an.HORIZONS else 9):
        part = frame[frame["horizon"] == horizon]
        cells = part.assign(cell=[
            f"{r:.3f} {interval(lo, hi)}" if np.isfinite(lo) and a != b else (f"{r:.3f}" if np.isfinite(r) else "n/a")
            for r, lo, hi, a, b in zip(part["ratio"], part["low"], part["high"], part["trained_on"], part["tested_on"])
        ])
        matrix = cells.pivot_table(index=["model", "trained_on"], columns="tested_on", values="cell", aggfunc="first").reset_index()
        matrix.columns.name = None
        text += f"\n### Horizon {horizon}\n\n" + table(matrix.fillna("n/a"))
    summary, targets_frame = an.e4_summary(frame), an.e4_targets(frame)
    if not summary.empty:
        text += "\nMean degradation per model over its foreign pairs:\n\n" + table(summary)
        text += "\nMean degradation per target dataset:\n\n" + table(targets_frame)
    return text, frame


def main(argv: list[str] | None = None) -> Path:
    args = arguments(argv)
    root, out, n_boot, seed = args["root"], args["out"], args["n_boot"], args["seed"]
    started = time.time()
    labels_of = label_filter(args["models"])
    scope = ("the reference models of the main campaign (`scripts/m2.py`) in E1 and E3" if main_labels() is not None and
             args["models"] == "main" else "every finished run in E1 and E3")
    text = ("# Milestone 4: experiments E1-E4\n\n"
            f"Generated by `scripts/m4_report.py` from `{root}` ({n_boot} bootstrap replicates, seed {seed}; {scope}); definitions in "
            "`docs/experiments.md`, section 3. Tables: 3 decimals, `n/a` = not available. Per-window tables are used only when they "
            "reproduce their JSON summary (decision M4-5: in milestone 3 the transfer evaluation overwrote the own `p1.<h>.parquet`).\n")
    campaigns, _ = section_campaigns(root)
    e1_text, e1 = section_e1(root, n_boot, seed, labels_of)
    e2_text, e2, e2u = section_e2(root, n_boot, seed, args["splits"], args["events"])
    e3_text, e3, e3k, cache = section_e3(root, n_boot, seed, labels_of)
    e3b_text, e3v, e3s, e3p = section_e3b(root, cache, n_boot, seed)
    e3c_text, e3c = section_e3c(root)
    e4_text, e4 = section_e4(root, n_boot, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(text + campaigns + e1_text + e2_text + e3_text + e3b_text + e3c_text + e4_text, encoding="utf-8")
    for name, frame in (("e1_ratios", e1), ("e2_paired", e2), ("e2_unpaired", e2u), ("e3_rankings", e3), ("e3_kendall", e3k),
                        ("e3_composite_variants", e3v), ("e3_composite_sensitivity", e3s), ("e3_pareto", e3p),
                        ("e3_ablations", e3c), ("e4_transfer", e4)):
        frame.to_csv(out / f"{name}.csv", index=False)
    print(f"report written to {out / 'REPORT.md'} ({time.time() - started:.0f} s): E1 {len(e1)} rows, E2 {len(e2)} paired and "
          f"{len(e2u)} unpaired rows, E3 {len(e3)} models and {len(e3k)} taus, E3b {len(e3s)} sensitivity rows, E4 {len(e4)} entries",
          flush=True)
    return out / "REPORT.md"


if __name__ == "__main__":
    main()
