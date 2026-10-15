"""P5: transfer between datasets.

A model trained on dataset ``A`` is scored with P1 on the test part of every other dataset
``B`` (same split kind, fold and data version); ``scripts/evaluate.py`` writes ``p1_<B>.json``
next to ``p1.json`` of the run. This module assembles the transfer matrix from those files:
the relative degradation ``rmse(A -> B) / rmse(B -> B)``, where ``B -> B`` is the run of the
same model label trained on ``B``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trafficloop.utils import read_json

HORIZON_KEYS = ("h10", "h30", "whole")


def collect_transfer(root: Path, version: str, split: str, fold: int, seed: int = 0) -> pd.DataFrame:
    """One row per (model, trained_on, tested_on, horizon) from ``p1.json`` and ``p1_<B>.json``."""
    rows = []
    for path in sorted(Path(root).glob(f"*/{version}/{split}-fold{fold}/*/seed{seed}/p1*.json")):
        trained_on = path.parents[4].name
        label = path.parents[1].name
        tested_on = trained_on if path.stem == "p1" else path.stem[len("p1_") :]
        summary = read_json(path)["summary"]
        for key in HORIZON_KEYS:
            if key in summary:
                rows.append({
                    "model": label, "trained_on": trained_on, "tested_on": tested_on, "horizon": key,
                    "rmse_s": summary[key]["rmse_s"], "rmse_v": summary[key]["rmse_v"],
                    "collision_rate": summary[key]["collision_rate"], "n_events": summary[key].get("n_events"),
                })
    return pd.DataFrame(rows)


def transfer_matrix(runs: pd.DataFrame, horizon: str = "whole", metric: str = "rmse_s") -> pd.DataFrame:
    """Relative degradation per model: rows ``(model, trained_on)``, columns ``tested_on``.

    The entry is ``metric(A -> B) / metric(B -> B)``; ``nan`` where the model has no run
    trained on ``B``. The diagonal is 1.
    """
    part = runs[runs["horizon"] == horizon]
    own = part[part["trained_on"] == part["tested_on"]].set_index(["model", "tested_on"])[metric]
    values = []
    for row in part.itertuples(index=False):
        base = own.get((row.model, row.tested_on), float("nan"))
        values.append({"model": row.model, "trained_on": row.trained_on, "tested_on": row.tested_on,
                       "ratio": getattr(row, metric) / base if base and base == base else float("nan")})
    frame = pd.DataFrame(values)
    if frame.empty:
        return frame
    return frame.pivot_table(index=["model", "trained_on"], columns="tested_on", values="ratio", aggfunc="first")
