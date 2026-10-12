"""Collect the experiments of milestone 4 into ``results/m4/REPORT.md`` and CSV tables.

    python scripts/m4_report.py [root=results/models] [out=results/m4] [n_boot=1000] [seed=0]
                                [splits=data/splits] [events=data/events]

E1-E4 of docs/experiments.md, section 3 (``follownet2/experiments/analysis.py``), from the run
directories ``<root>/<dataset>/reconstructed/<split>-fold<k>/<label>/seed<s>/``. A campaign that
has not run (or not finished) is stated in the report; its tables stay empty. Writes
``REPORT.md``, ``e1_ratios.csv``, ``e2_paired.csv``, ``e2_unpaired.csv``, ``e3_rankings.csv``,
``e3_kendall.csv`` (with the spacing-against-speed rows) and ``e4_transfer.csv``.
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
DEFAULTS = {"root": "results/models", "out": "results/m4", "n_boot": "1000", "seed": "0", "splits": "data/splits", "events": "data/events"}


def arguments(argv: list[str] | None) -> dict:
    args = dict(DEFAULTS)
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            raise SystemExit(f"unknown argument {item!r}; usage: m4_report.py " + " ".join(f"[{k}=...]" for k in DEFAULTS))
        args[key] = value
    return {"root": resolve_path(args["root"]), "out": resolve_path(args["out"]), "n_boot": int(args["n_boot"]),
            "seed": int(args["seed"]), "splits": resolve_path(args["splits"]), "events": resolve_path(args["events"])}


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


def main_models() -> int | None:
    """Number of model labels of the main campaign (scripts/m2.py), when importable."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from m2 import MODELS
    except Exception:  # noqa: BLE001 - only the expected counts of the campaign table depend on it
        return None
    finally:
        sys.path.pop(0)
    return len(MODELS)


def completeness(root: Path) -> pd.DataFrame:
    """Expected and finished runs of every campaign of docs/experiments.md, section 1."""
    n_models = main_models()
    labels = (*an.SUBSET, an.REFERENCE)
    found = {
        "main": sum(len(an.run_labels(root, d)) for d in an.MAIN_DATASETS),
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


def section_e1(root: Path, n_boot: int, seed: int) -> tuple[str, pd.DataFrame]:
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
        frame = an.e1_ratios(root, dataset, None, n_boot, seed)
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


def e3_datasets(root: Path) -> list[str]:
    """Datasets whose fold-0 runs carry the protocols P2-P4 (the main campaign)."""
    out = []
    for dataset in an.run_datasets(root):
        if sum((an.run_dir(root, dataset, m) / "p2.json").exists() for m in an.run_labels(root, dataset)) >= 2:
            out.append(dataset)
    return out


def section_e3(root: Path, n_boot: int, seed: int) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    text = ("\n## E3: rankings across protocols (H5.3, H5.4)\n\n"
            "Point values of fold 0, seed 0 (lower is better; P2 and the synthetic P4 over the profiles and scenarios inside the "
            "speed range of the training data; P1 whole and P0 from `train.json`). Kendall's tau-b between every pair of rankings: "
            f"`tau` and `p` on the point values, `low`/`high` the 95 % interval and `tau_boot` the mean over {n_boot} replicates "
            "(drivers for P0, P1 and the empirical P4; in-range profiles for P2 and the composite, speeds with an equilibrium for P3, "
            "in-range scenarios for the synthetic P4). A model without a value ranks last.\n")
    datasets = e3_datasets(root)
    missing = [d for d in an.MAIN_DATASETS if d not in datasets]
    if missing:
        text += f"\nNot available for {', '.join(missing)}: no fold-0 runs with P2-P4 (campaign `main`).\n"
    rankings_frames, kendall_frames = [], []
    for dataset in datasets:
        started = time.time()
        rankings = an.e3_rankings(root, dataset)
        replicates = an.e3_replicates(root, dataset, rankings, n_boot, seed)
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
    return text, rankings, kendall


def section_e4(root: Path, n_boot: int, seed: int) -> tuple[str, pd.DataFrame]:
    text = ("\n## E4: transfer (P5)\n\n"
            "Relative degradation `rmse_s(A -> B) / rmse_s(B -> B)` of P1 for SUBSET and persistence (rows: model and training "
            f"dataset A, columns: test dataset B; 1 = no loss); intervals from {n_boot} driver replicates over the test drivers of B "
            "(one draw per target, common to both tables), shown as `[low, high]` where the per-window tables of both evaluations exist.\n")
    frame = an.e4_transfer(root, [*an.SUBSET, an.REFERENCE], n_boot, seed)
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
    text = ("# Milestone 4: experiments E1-E4\n\n"
            f"Generated by `scripts/m4_report.py` from `{root}` ({n_boot} bootstrap replicates, seed {seed}); definitions in "
            "`docs/experiments.md`, section 3. Tables: 3 decimals, `n/a` = not available. Per-window tables are used only when they "
            "reproduce their JSON summary (decision M4-5: in milestone 3 the transfer evaluation overwrote the own `p1.<h>.parquet`).\n")
    campaigns, _ = section_campaigns(root)
    e1_text, e1 = section_e1(root, n_boot, seed)
    e2_text, e2, e2u = section_e2(root, n_boot, seed, args["splits"], args["events"])
    e3_text, e3, e3k = section_e3(root, n_boot, seed)
    e4_text, e4 = section_e4(root, n_boot, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text(text + campaigns + e1_text + e2_text + e3_text + e4_text, encoding="utf-8")
    for name, frame in (("e1_ratios", e1), ("e2_paired", e2), ("e2_unpaired", e2u), ("e3_rankings", e3), ("e3_kendall", e3k), ("e4_transfer", e4)):
        frame.to_csv(out / f"{name}.csv", index=False)
    print(f"report written to {out / 'REPORT.md'} ({time.time() - started:.0f} s): E1 {len(e1)} rows, E2 {len(e2)} paired and "
          f"{len(e2u)} unpaired rows, E3 {len(e3)} models and {len(e3k)} taus, E4 {len(e4)} entries", flush=True)
    return out / "REPORT.md"


if __name__ == "__main__":
    main()
