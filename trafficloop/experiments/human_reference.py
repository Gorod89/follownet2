"""E3d: the composite with the human platoon reference of protocol ``p2h``.

Protocol P2 compares the simulated platoons with recorded followers behind OpenACC leaders (17 of
the 19 windows have ACC followers); ``p2h`` repeats the protocol behind the human-driven chains of
NGSIM and TGSIM and the two manual AstaZero runs (``docs/protocols.md``). This module reads both
summaries per model and builds the composite variants ``v1`` and ``v2`` from either reference
(``v1h``, ``v2h``: the human one), with Kendall's tau-b between the rankings as point estimates
(the driver-bootstrap intervals of the ACC composite are in ``e3_composite_variants.csv``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from trafficloop.experiments import analysis as an
from trafficloop.protocols.composite import FD_INPUT, INPUTS, P1_INPUT, VARIANTS, composite_score
from trafficloop.stats.ranks import kendall_tau

#: p2h summary key per composite input that comes from the platoon protocol
PLATOON_KEYS = {
    "concave_growth_error": "concave_growth_error_in_range",
    "hysteresis_error": "hysteresis_error_in_range",
    "collisions_100": "collisions_100_in_range",
    FD_INPUT: "fd_error_rel",
}
EXTRA_KEYS = ("n_profiles_in_range", "growth_exponent_20_in_range", "wave_speed_in_range")
COMPOSITES = ("v1", "v1h", "v2", "v2h")
COLUMNS = ["dataset", "model", "family", "p0_rmse_a", P1_INPUT, "unstable_fraction",
           *(f"acc_{k}" for k in PLATOON_KEYS), *(f"human_{k}" for k in PLATOON_KEYS),
           *(f"human_{k}" for k in EXTRA_KEYS), *(f"composite_{v}" for v in COMPOSITES),
           *(f"rank_{v}" for v in COMPOSITES), "note"]
TAU_COLUMNS = ["dataset", "ranking_a", "ranking_b", "tau", "p", "n_models"]
PAIRS = (("v1", "v1h"), ("v2", "v2h"), ("P0", "v1"), ("P0", "v1h"), ("P0", "v2"), ("P0", "v2h"), ("P1 whole", "v2h"))
TOP_COLUMNS = ["dataset", "ranking", "top1", "top2", "top3"]


def _inputs(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    """The composite inputs of ``frame`` under one reference (``acc`` or ``human``), named as in
    :mod:`trafficloop.protocols.composite`."""
    table = pd.DataFrame(index=frame.index)
    for name in INPUTS:
        table[name] = frame[f"{reference}_{name}"] if name in PLATOON_KEYS else frame[name]
    table[P1_INPUT] = frame[P1_INPUT]
    table[FD_INPUT] = frame[f"{reference}_{FD_INPUT}"]
    return table.astype(np.float64)


def _composite(table: pd.DataFrame, variant: str) -> np.ndarray:
    columns = VARIANTS[variant]
    if table.empty or table[list(columns)].isna().all(axis=None):
        return np.full(len(table), np.nan)
    return composite_score(table, columns)["composite"].sort_index().to_numpy(dtype=np.float64)


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="min", na_option="keep").to_numpy(dtype=np.float64)


def e3_human_reference(root, dataset: str, labels: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per model: the platoon inputs under both references, the four composites, ranks."""
    rankings = an.e3_rankings(root, dataset, labels)
    rows = []
    for _, r in rankings.iterrows():
        directory = an.run_dir(root, dataset, r["model"])
        p2, p2h = an._summary(directory, "p2"), an._summary(directory, "p2h")
        row: dict = {"dataset": dataset, "model": r["model"], "family": r["family"], "p0_rmse_a": r["p0_rmse_a"],
                     P1_INPUT: r["p1_whole_rmse_s"], "unstable_fraction": r["p3_unstable_fraction"]}
        for name, key in PLATOON_KEYS.items():
            row[f"acc_{name}"] = an._num(p2, key)
            row[f"human_{name}"] = an._num(p2h, key)
        for key in EXTRA_KEYS:
            row[f"human_{key}"] = an._num(p2h, key)
        row["note"] = "" if p2h is not None else "no p2h.json"
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    for variant in ("v1", "v2"):
        frame[f"composite_{variant}"] = _composite(_inputs(frame, "acc"), variant)
        frame[f"composite_{variant}h"] = _composite(_inputs(frame, "human"), variant)
    for variant in COMPOSITES:
        frame[f"rank_{variant}"] = _rank(frame[f"composite_{variant}"].to_numpy(dtype=np.float64))
    return frame[COLUMNS]


def _ranking(frame: pd.DataFrame, name: str) -> dict[str, float]:
    column = {"P0": "p0_rmse_a", "P1 whole": P1_INPUT}.get(name, f"composite_{name}")
    return dict(zip(frame["model"], frame[column].to_numpy(dtype=np.float64)))


def human_reference_taus(frame: pd.DataFrame) -> pd.DataFrame:
    """Kendall's tau-b between the rankings of :data:`PAIRS` per dataset (point estimates)."""
    rows = []
    for dataset, block in frame.groupby("dataset", sort=True):
        for a, b in PAIRS:
            ra, rb = _ranking(block, a), _ranking(block, b)
            keys = [k for k in ra if not (np.isnan(ra[k]) or np.isnan(rb[k]))]
            if len(keys) < 3:
                continue
            tau, p = kendall_tau({k: ra[k] for k in keys}, {k: rb[k] for k in keys})
            rows.append({"dataset": dataset, "ranking_a": a, "ranking_b": b, "tau": float(tau), "p": float(p),
                         "n_models": len(keys)})
    return pd.DataFrame(rows, columns=TAU_COLUMNS)


def human_reference_top3(frame: pd.DataFrame) -> pd.DataFrame:
    """The first three models of every composite per dataset (lowest composite first, ties by name)."""
    rows = []
    for dataset, block in frame.groupby("dataset", sort=True):
        for variant in COMPOSITES:
            order = block[["model", f"composite_{variant}"]].dropna().sort_values([f"composite_{variant}", "model"])
            top = list(order["model"][:3]) + [""] * 3
            rows.append({"dataset": dataset, "ranking": variant, "top1": top[0], "top2": top[1], "top3": top[2]})
    return pd.DataFrame(rows, columns=TOP_COLUMNS)
