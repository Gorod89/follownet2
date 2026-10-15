"""Build the figures of the report (docs/leaderboard.md, section 4) into ``results/figures``.

    python scripts/make_report.py [root=results/models] [m4=results/m4] [out=results/figures]
                                  [p2_dataset=ngsim]

Reads the CSVs of milestone 4 under ``m4`` (``e1_ratios.csv``, ``e2_paired.csv``,
``e3_rankings.csv``, ``e3_pareto.csv``, ``e4_transfer.csv``) and the P2 tables of the run directories under ``root``
(``<root>/<dataset>/reconstructed/driver-fold0/<label>/seed0/``), draws every figure of
``trafficloop/figures/plots.py``, writes each as PNG (200 dpi) and PDF into ``out`` and lists the
files in ``out/README.md``, one line each. The P1 horizons come from ``e3_rankings.csv`` when it
carries them for a dataset, otherwise from the ``p1.json`` of its runs. A figure whose input is
missing is skipped and named; nothing is trained or evaluated.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from trafficloop.experiments import analysis as an  # noqa: E402
from trafficloop.figures import plots, style  # noqa: E402
from trafficloop.utils import REPO_ROOT, read_json, resolve_path  # noqa: E402

DEFAULTS = {"root": "results/models", "m4": "results/m4", "out": "results/figures", "p2_dataset": "ngsim"}
#: summary key of p1.json -> column of e3_rankings.csv
P1_COLUMNS = {"h10": "p1_h10_rmse_s", "h30": "p1_h30_rmse_s", "whole": "p1_whole_rmse_s"}
VEHICLE_COLUMNS = ["profile", "index", "std_v", "std_v_rec"]
PROFILE_COLUMNS = ["profile", "source", "n_vehicles", "in_range"]


def arguments(argv: Sequence[str] | None = None) -> dict:
    args = dict(DEFAULTS)
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if not sep or key not in DEFAULTS:
            raise SystemExit(f"unknown argument {item!r}; usage: make_report.py " + " ".join(f"[{k}=...]" for k in DEFAULTS))
        args[key] = value
    return {"root": resolve_path(args["root"]), "m4": resolve_path(args["m4"]), "out": resolve_path(args["out"]),
            "p2_dataset": args["p2_dataset"]}


# ----------------------------------------------------------------------------- loaders


def read_csv(m4: Path, name: str) -> pd.DataFrame | None:
    path = Path(m4) / name
    return pd.read_csv(path) if path.exists() else None


def run_family(directory: Path) -> str | None:
    """Family of a run: ``train.json``, else the model block of ``p2.json`` or ``p1.json``."""
    for name, keys in (("train.json", ("family",)), ("p2.json", ("model", "family")), ("p1.json", ("model", "family"))):
        path = directory / name
        if path.exists():
            value = read_json(path)
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if value:
                return str(value)
    return None


def load_p2(root: Path, dataset: str, models: Sequence[str] = plots.P2_MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The P2 ``vehicles`` and ``profiles`` tables of the driver-split runs (fold 0, seed 0) of
    ``models`` on ``dataset`` with ``model`` and ``family`` added; a run without them is left out."""
    vehicles, profiles = [], []
    for model in models:
        directory = an.run_dir(root, dataset, model)
        v_path, p_path = directory / "p2.vehicles.parquet", directory / "p2.profiles.parquet"
        if not (v_path.exists() and p_path.exists()):
            continue
        family = run_family(directory)
        vehicles.append(pd.read_parquet(v_path, columns=VEHICLE_COLUMNS).assign(model=model, family=family))
        profiles.append(pd.read_parquet(p_path, columns=PROFILE_COLUMNS).assign(model=model, family=family))
    if not vehicles:
        return (pd.DataFrame(columns=[*VEHICLE_COLUMNS, "model", "family"]),
                pd.DataFrame(columns=[*PROFILE_COLUMNS, "model", "family"]))
    return pd.concat(vehicles, ignore_index=True), pd.concat(profiles, ignore_index=True)


def p1_from_runs(root: Path, dataset: str) -> pd.DataFrame:
    """``summary.<h>.rmse_s`` of the ``p1.json`` of every driver-split run (fold 0, seed 0)."""
    rows = []
    for label in an.run_labels(root, dataset):
        directory = an.run_dir(root, dataset, label)
        if not (directory / "p1.json").exists():
            continue
        summary = read_json(directory / "p1.json").get("summary") or {}
        row = {"dataset": dataset, "model": label, "family": run_family(directory)}
        for key, column in P1_COLUMNS.items():
            value = (summary.get(key) or {}).get("rmse_s")
            row[column] = float("nan") if value is None else float(value)
        rows.append(row)
    return pd.DataFrame(rows, columns=["dataset", "model", "family", *P1_COLUMNS.values()])


def load_horizons(m4: Path, root: Path, datasets: Sequence[str] = plots.MAIN_DATASETS) -> tuple[pd.DataFrame, dict]:
    """P1 spacing RMSE at every horizon per dataset and model, and the source per dataset
    (``e3_rankings.csv`` when it carries the three columns and the dataset, else ``p1.json``)."""
    rankings = read_csv(m4, "e3_rankings.csv")
    usable = rankings is not None and set(P1_COLUMNS.values()) <= set(rankings.columns)
    frames, sources = [], {}
    for dataset in datasets:
        block = rankings[rankings["dataset"] == dataset] if usable else pd.DataFrame()
        if len(block):
            frames.append(block[["dataset", "model", "family", *P1_COLUMNS.values()]])
            sources[dataset] = "e3_rankings.csv"
        else:
            frames.append(p1_from_runs(root, dataset))
            sources[dataset] = "p1.json of the runs"
    frame = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else \
        pd.DataFrame(columns=["dataset", "model", "family", *P1_COLUMNS.values()])
    return frame, sources


# ----------------------------------------------------------------------------- figures


def _names(datasets: Sequence[str]) -> str:
    return " and ".join(style.dataset_label(d) for d in datasets)


def _source(source: str) -> str:
    """A source of :func:`load_horizons` in Markdown (file names as code)."""
    return f"`{source}`" if source.endswith(".csv") else source.replace("p1.json", "`p1.json`")


E1B_POOLED = {"h1_rmse_s": "1 s", "h10_rmse_s": "10 s", "h30_rmse_s": "30 s", "whole_rmse_s": "Whole event"}
E1B_MEDIAN = {f"{h}_median_rmse_s": label for h, label in
              (("h1", "1 s"), ("h10", "10 s"), ("h30", "30 s"), ("whole", "Whole event"))}


def figure_jobs(args: dict) -> list[tuple[str, Callable | None, str]]:
    """``(stem, builder or None when the input is missing, description)`` of every figure."""
    m4, root = args["m4"], args["root"]
    e1, e2 = read_csv(m4, "e1_ratios.csv"), read_csv(m4, "e2_paired.csv")
    e3, e4 = read_csv(m4, "e3_rankings.csv"), read_csv(m4, "e4_transfer.csv")
    pareto = read_csv(m4, "e3_pareto.csv")
    kendall = read_csv(m4, "e3_kendall.csv")
    e1b = read_csv(m4, "e1_horizons.csv")
    p2_dataset = args["p2_dataset"]
    vehicles, profiles = load_p2(root, p2_dataset)
    horizons, sources = load_horizons(m4, root)
    p2_models = [m for m in plots.P2_MODELS if m in set(vehicles["model"])]
    followers = int(pd.to_numeric(vehicles["index"]).max()) if len(vehicles) else 0
    no_h30 = [d for d in plots.MAIN_DATASETS
              if len(horizons) and horizons.loc[horizons["dataset"] == d, "p1_h30_rmse_s"].notna().sum() == 0]
    e4_datasets = sorted(set(e4["tested_on"])) if e4 is not None else []
    e4_models = [m for m in plots.SUBSET if e4 is not None and m in set(e4["model"])]
    runs = an.run_dir(root, p2_dataset, "<model>")
    runs = runs.relative_to(REPO_ROOT).as_posix() if runs.is_relative_to(REPO_ROOT) else runs.as_posix()
    return [
        ("e1_persistence_ratio_bars", None if e1 is None else lambda: plots.persistence_ratio_bars(e1),
         f"E1 (H5.1): P0 acceleration RMSE relative to persistence, {_names(plots.MAIN_DATASETS)}, driver split, "
         "fold 0, seed 0; bars from 1 on a log axis, 95 % driver-bootstrap intervals, models sorted by the ratio, "
         "persistence is the line at 1 (`results/m4/e1_ratios.csv`)."),
        ("e2_split_inflation_bars", None if e2 is None else lambda: plots.split_inflation_bars(e2),
         f"E2 (H5.2): paired inflation (clean - leaky) / clean in % on the affected probe events of folds 0-4, "
         f"{_names(plots.LEAK_DATASETS)}, P0 acceleration, P1 10 s and P1 whole-event spacing RMSE; whiskers: paired "
         "driver-bootstrap interval of clean - leaky divided by the clean error (`results/m4/e2_paired.csv`)."),
        ("e3_rank_bump_chart", None if e3 is None else lambda: plots.rank_bump_chart(e3),
         f"E3 (H5.3): rank of every model under P0, P1 10 s, P1 whole event and the composite, "
         f"{_names(plots.MAIN_DATASETS)}; rank 1 = lowest value, ties broken by the model name "
         "(`results/m4/e3_rankings.csv`)."),
        ("e3_pareto_scatter", None if pareto is None else lambda: plots.pareto_scatter(pareto),
         "E3b: realism against the traffic-level composite: whole-event P1 spacing RMSE (log axis) against composite v2 "
         "(mean normalised rank of the concave-growth and hysteresis errors, the unstable fraction, the 100-vehicle "
         "collisions, the whole-event P1 spacing RMSE and the relative fundamental-diagram error; 0 = best) per model, "
         f"{_names(plots.MAIN_DATASETS)}, family colours, every model labelled, the Pareto front of the two (both lower "
         "is better) as a step line, models on it with a dark rim (`results/m4/e3_pareto.csv`)."),
        ("e3_tau_summary", None if kendall is None else lambda: plots.tau_summary(kendall),
         "E3 across datasets: Kendall's tau between the one-step, the whole-event closed-loop and the composite "
         "rankings of the 24 entries, one group of bars per dataset with 95 % bootstrap intervals "
         "(`results/m4/e3_kendall.csv`)."),
        ("e4_transfer_heatmap", None if e4 is None else lambda: plots.transfer_heatmap(e4),
         f"E4 (P5): whole-event relative degradation rmse_s(A -> B) / rmse_s(B -> B) of {len(e4_models)} SUBSET "
         f"models over {len(e4_datasets)} datasets (rows trained on, columns tested on), diverging log colour "
         "scale centred at 1 and capped at 0.1 and 10, values printed (`results/m4/e4_transfer.csv`)."),
        ("p1_horizon_curves", None if horizons.empty else lambda: plots.horizon_curves(horizons),
         f"P1: spacing RMSE at 10 s, 30 s and the whole event per model, {_names(plots.MAIN_DATASETS)}, family "
         "colours, log axis, the three lowest and highest whole-event values labelled"
         + (f"; {_names(no_h30)} has no 30 s windows" if no_h30 else "")
         + (f" (from {_source(next(iter(sources.values())))})." if len(set(sources.values())) == 1 else
            " (" + "; ".join(f"{style.dataset_label(d)} from {_source(s)}" for d, s in sources.items()) + ").")),
        ("e1b_horizon_pooled", None if e1b is None else lambda: plots.horizon_curves(e1b, horizons=E1B_POOLED),
         f"E1b: pooled P1 spacing RMSE at 1 s (protocol p1s), 10 s, 30 s and the whole event per model, "
         f"{_names(plots.MAIN_DATASETS)}, family colours, log axis (`results/m4/e1_horizons.csv`)."),
        ("e1b_horizon_median", None if e1b is None else lambda: plots.horizon_curves(e1b, horizons=E1B_MEDIAN),
         f"E1b: median over windows (over events for the whole event) of the P1 spacing RMSE at 1 s, 10 s, 30 s and the "
         f"whole event per model, {_names(plots.MAIN_DATASETS)}, family colours, log axis (`results/m4/e1_horizons.csv`)."),
        ("p2_concave_growth_curves",
         None if not p2_models else lambda: plots.concave_growth_curves(vehicles, profiles, p2_models, dataset=p2_dataset),
         f"P2: speed standard deviation against the vehicle index (0 = leader) along the platoon of {followers} "
         f"followers for {', '.join(p2_models) or 'no model'} trained on {style.dataset_label(p2_dataset)}, one line "
         "per in-range OpenACC profile, recorded followers as black markers, log axis (`p2.vehicles.parquet` and "
         f"`p2.profiles.parquet` of `{runs}`)."),
    ]


def write_readme(out: Path, entries: Sequence[tuple[Path, str]]) -> Path:
    lines = [
        "# Figures of the report",
        "",
        "Written by `scripts/make_report.py` (docs/leaderboard.md, section 4; style in "
        "`trafficloop/figures/style.py`): 160 mm wide, 9 pt text, PNG at 200 dpi and PDF with the same content. "
        "Families: network blue, mechanistic vermillion, baseline bluish green (Okabe-Ito). One line per file:",
        "",
    ]
    for path, description in entries:
        kind = "PNG, 200 dpi" if path.suffix == ".png" else "PDF, vector"
        lines.append(f"- `{path.name}` ({kind}): {description}")
    readme = Path(out) / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = arguments(argv)
    out = args["out"]
    out.mkdir(parents=True, exist_ok=True)
    entries, skipped = [], []
    for stem, build, description in figure_jobs(args):
        if build is None:
            skipped.append(stem)
            continue
        for path in style.save(build(), out / stem):
            entries.append((path, description))
    readme = write_readme(out, entries)
    for path, _ in entries:
        print(f"{path.relative_to(out.parent) if out.parent in path.parents else path}  {path.stat().st_size / 1024:.0f} KiB")
    print(f"{readme.name}: {len(entries)} files listed")
    if skipped:
        print("skipped (input missing): " + ", ".join(skipped))
    return [path for path, _ in entries] + [readme]


if __name__ == "__main__":
    main()
