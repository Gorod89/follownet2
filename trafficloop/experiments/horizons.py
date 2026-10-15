"""E1b: closed-loop spacing error from 1 s windows to the whole event.

Per model and dataset: the pooled RMSE of the spacing (as in P1) and the median over windows
(over events for the whole event) at the horizons 1 s (protocol ``p1s``), 10 s, 30 s and the
whole event, the share of the whole-event summed squared error carried by the five followers
with the largest sums, and Kendall's tau-b between the one-step ranking (P0) and the ranking
at every horizon. The taus are point estimates; the driver-bootstrap intervals of the P1
horizons are in ``e3_kendall.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from trafficloop.experiments import analysis as an
from trafficloop.stats.ranks import kendall_tau

HORIZONS = ("h1", "h10", "h30", "whole")
TOP = 5
STATISTICS = ("rmse_s", "median_rmse_s")
COLUMNS = ["dataset", "model", "family", "p0_rmse_a",
           *(f"{h}_{k}" for h in HORIZONS for k in (*STATISTICS, "n_windows")),
           "whole_top5_share", "n_followers", "note"]
TAU_COLUMNS = ["dataset", "horizon", "statistic", "tau", "n_models"]


def short_p1_table(directory) -> tuple[pd.DataFrame | None, str | None]:
    """The 1 s window table of protocol ``p1s`` (``p1s.h1.parquet`` checked against ``p1s.json``)."""
    directory = Path(directory)
    summary = an._summary(directory, "p1s")
    table = an._read(directory / "p1s.h1.parquet")
    problem = an.check_table(table, None if summary is None else summary.get("h1"))
    return (table, None) if problem is None else (None, f"p1s.h1.parquet: {problem}")


def horizon_table(directory, horizon: str) -> tuple[pd.DataFrame | None, str | None]:
    return short_p1_table(directory) if horizon == "h1" else an.own_p1_table(directory, horizon)


def window_rmse(table: pd.DataFrame) -> np.ndarray:
    """Spacing RMSE of every window (row) of a P1 table."""
    n = table["n"].to_numpy(dtype=np.float64)
    return np.sqrt(table["sse_s"].to_numpy(dtype=np.float64) / np.where(n > 0, n, np.nan))


def top_share(table: pd.DataFrame, k: int = TOP, group: str = an.GROUP) -> float:
    """Share of the summed squared spacing error carried by the ``k`` groups with the largest sums."""
    sums = table.groupby(group)["sse_s"].sum().sort_values(ascending=False)
    total = float(sums.sum())
    return float(sums.iloc[:k].sum() / total) if total > 0 else float("nan")


def e1_horizons(root, dataset: str, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per model: pooled and median spacing errors per horizon, the top-five share."""
    if labels is None:
        labels = an.run_labels(root, dataset)
    else:
        labels = [m for m in labels if (an.run_dir(root, dataset, m) / "train.json").exists()]
    rows = []
    for label in labels:
        directory = an.run_dir(root, dataset, label)
        train = an._json(directory / "train.json") or {}
        row: dict = {"dataset": dataset, "model": label, "family": train.get("family"),
                     "p0_rmse_a": an._num(train, "test", "p0", "rmse_a"),
                     "whole_top5_share": float("nan"), "n_followers": 0}
        notes = []
        for h in HORIZONS:
            table, problem = horizon_table(directory, h)
            if table is None or len(table) == 0:
                row[f"{h}_rmse_s"] = row[f"{h}_median_rmse_s"] = float("nan")
                row[f"{h}_n_windows"] = 0
                notes.append(problem or f"{h}: empty table")
                continue
            n = table["n"].to_numpy(dtype=np.float64)
            row[f"{h}_rmse_s"] = float(np.sqrt(table["sse_s"].sum() / n.sum()))
            row[f"{h}_median_rmse_s"] = float(np.nanmedian(window_rmse(table)))
            row[f"{h}_n_windows"] = int(len(table))
            if h == "whole":
                row["whole_top5_share"] = top_share(table)
                row["n_followers"] = int(table[an.GROUP].nunique())
        row["note"] = "; ".join(notes)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def horizon_taus(frame: pd.DataFrame) -> pd.DataFrame:
    """Kendall's tau-b between the P0 ranking and the ranking at every horizon (pooled and median)."""
    rows = []
    for dataset, block in frame.groupby("dataset", sort=True):
        block = block.set_index("model")
        p0 = block["p0_rmse_a"]
        for h in HORIZONS:
            for statistic in STATISTICS:
                values = block[f"{h}_{statistic}"]
                keep = p0.notna() & values.notna()
                if int(keep.sum()) < 3:
                    continue
                tau, _ = kendall_tau(p0[keep].to_dict(), values[keep].to_dict())
                rows.append({"dataset": dataset, "horizon": h, "statistic": statistic, "tau": float(tau),
                             "n_models": int(keep.sum())})
    return pd.DataFrame(rows, columns=TAU_COLUMNS)
