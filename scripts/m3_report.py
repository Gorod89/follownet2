"""Collect the protocol results of milestone 3 into ``results/m3/REPORT.md``.

    python scripts/m3_report.py

Reads every run directory ``results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/``
with ``train.json`` and the protocol files ``p1.json`` ... ``p4.json`` and ``p1_<other>.json``.
"""

from __future__ import annotations

import pandas as pd

from follownet2.protocols.p5 import collect_transfer, transfer_matrix
from follownet2.utils import REPO_ROOT, read_json

ROOT = REPO_ROOT / "results" / "models"
OUT = REPO_ROOT / "results" / "m3"
FAMILY_ORDER = {"baseline": 0, "mechanistic": 1, "network": 2}
HORIZONS = ("h10", "h30", "whole")
#: composite inputs (name expected by composite_score -> summary key); the P2 inputs are taken
#: over the profiles inside the speed range of the training data
COMPOSITE_SOURCES = {
    "concave_growth_error": "concave_growth_error_in_range",
    "hysteresis_error": "hysteresis_error_in_range",
    "unstable_fraction": "unstable_fraction",
    "collisions_100": "collisions_100_in_range",
}


def _get(summary: dict | None, *keys, default=float("nan")):
    value = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return default if value is None else value


def collect() -> pd.DataFrame:
    rows = []
    for train in sorted(ROOT.rglob("train.json")):
        directory, r = train.parent, read_json(train)
        protocols = {p: read_json(directory / f"{p}.json")["summary"] for p in ("p1", "p2", "p3", "p4") if (directory / f"{p}.json").exists()}
        row = {
            "dataset": r["dataset"], "version": r["data_version"],
            "split": f'{r["split"]["kind"]}-{r["split"]["fold"]}', "seed": r["seed"],
            "model": r["model"], "family": r["family"], "directory": str(directory),
            "P0 RMSE_a": r["test"]["p0"]["rmse_a"],
        }
        p1 = protocols.get("p1")
        for key in HORIZONS:
            row[f"{key} RMSE_s"] = _get(p1, key, "rmse_s")
            row[f"{key} RMSE_v"] = _get(p1, key, "rmse_v")
            row[f"{key} coll"] = _get(p1, key, "collision_rate")
            row[f"{key} n"] = _get(p1, key, "n_events", default=0)
        p2 = protocols.get("p2")
        for key in ("concave_growth_error", "hysteresis_error", "growth_exponent_20", "growth_exponent_100",
                    "wave_speed", "wave_speed_pulse", "fd_error", "fd_error_rel", "collisions_20", "collisions_100", "settled_share",
                    "n_profiles", "n_profiles_in_range", "concave_growth_error_in_range", "hysteresis_error_in_range",
                    "growth_exponent_20_in_range", "growth_exponent_100_in_range", "wave_speed_in_range",
                    "collisions_20_in_range", "collisions_100_in_range", "settled_share_in_range"):
            row[key] = _get(p2, key)
        row["pulse_v0"] = _get(p2, "pulse", "v0")
        p3 = protocols.get("p3")
        for key in ("unstable_fraction", "unstable_fraction_1e3", "unstable_fraction_large", "unstable_fraction_analytic", "max_gain",
                    "max_gain_large", "n_collided_large", "peak_frequency", "n_equilibria", "agreement", "tracks_share"):
            row[key] = _get(p3, key)
        p4 = protocols.get("p4")
        for key in ("collision_rate_synthetic", "collision_rate_empirical", "min_ttc_q05", "min_ttc_q05_empirical", "min_ttc_q50", "min_ttc_q50_empirical",
                    "min_ttc_q50_recorded", "exceedance_4", "exceedance_4_empirical", "exceedance_4_recorded", "amplification_mean",
                    "n_scenarios", "n_episodes", "n_scenarios_in_range", "collision_rate_synthetic_in_range", "min_ttc_q50_in_range",
                    "n_collided_hold"):
            row[key] = _get(p4, key)
        rows.append(row)
    return pd.DataFrame(rows)


def table(frame: pd.DataFrame, digits: int = 3) -> str:
    fmt = lambda v: ("" if pd.isna(v) else f"{v:.{digits}f}") if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def kendall(block: pd.DataFrame, a: str, b: str) -> str | None:
    from scipy.stats import kendalltau

    part = block[[a, b]].dropna()
    if len(part) < 4:
        return None
    tau, p = kendalltau(part[a], part[b])
    return f"* {a} vs {b}: tau = {tau:.2f} (p = {p:.3f}, n = {len(part)})"


def composite_block(block: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Composite score of the block and, when the per-profile tables exist, its bootstrap."""
    from follownet2.protocols.composite import composite_bootstrap, composite_score

    inputs = block.set_index("model")[list(COMPOSITE_SOURCES.values())]
    inputs = inputs.rename(columns={source: name for name, source in COMPOSITE_SOURCES.items()})
    if inputs.isna().all(axis=None):
        return pd.DataFrame(), ""
    scored = composite_score(inputs)
    note = ""
    try:
        per_profile, per_speed = {}, {}
        for row in block.itertuples(index=False):
            profiles = pd.read_parquet(f"{row.directory}/p2.profiles.parquet")
            speeds = pd.read_parquet(f"{row.directory}/p3.speeds.parquet")
            if "in_range" in profiles.columns:
                profiles = profiles[profiles["in_range"].astype(bool)]
            if "n_vehicles" in profiles.columns:
                profiles = profiles[profiles["n_vehicles"] == profiles["n_vehicles"].max()]
            if {"converged", "tracks"} <= set(speeds.columns):  # equilibria only, as the summary
                speeds = speeds[speeds["converged"].fillna(False).astype(bool) & speeds["tracks"].fillna(False).astype(bool)]
            per_profile[row.model] = profiles.rename(columns={"collisions": "collisions_100"})[
                ["profile", "concave_growth_error", "hysteresis_error", "collisions_100"]
            ]
            per_speed[row.model] = speeds.rename(columns={"unstable_numerical": "unstable"})[["v_e", "unstable"]]
        boot = composite_bootstrap(per_profile, per_speed, n_boot=1000, seed=0)
        scored = scored.join(boot, rsuffix="_boot")
        note = ("\n`mean_rank`, `rank_low`, `rank_high` and `p_first` come from 1000 bootstrap replicates over the "
                "in-range P2 profiles and the P3 speeds (seed 0).\n")
    except Exception as error:  # noqa: BLE001 - the bootstrap is optional in the report
        note = f"\n(composite bootstrap not available: {type(error).__name__}: {error})\n"
    keep = [c for c in (*COMPOSITE_SOURCES, "composite", "mean_rank", "rank_low", "rank_high", "p_first", "p_top3") if c in scored.columns]
    return scored[keep].reset_index(), note


def main() -> None:
    frame = collect()
    if frame.empty:
        raise SystemExit(f"no run found under {ROOT}")
    text = "# Milestone 3: protocols P1-P5 on the reference models\n\n"
    text += ("Closed-loop horizons (P1), platoon (P2), string stability (P3), rare events (P4) and transfer "
             "(P5) of the models trained in milestone 2. `rank` orders a block by the whole-event spacing error; "
             "the composite score is the mean normalised rank over the concave-growth error, the hysteresis "
             "error, the numerical unstable fraction and the 100-vehicle collision count (lower is better).\n")
    for keys, block in frame.groupby(["dataset", "version", "split"], sort=True):
        block = block.copy()
        block["order"] = block["family"].map(FAMILY_ORDER)
        block["rank"] = block["whole RMSE_s"].rank(method="min").fillna(0).astype(int)
        block = block.sort_values(["order", "whole RMSE_s"])
        text += f"\n## {keys[0]}, {keys[1]}, split {keys[2]}\n\n### P1: closed loop by horizon\n\n"
        cols = ["model", "family", "P0 RMSE_a", "h10 RMSE_s", "h10 coll", "h30 RMSE_s", "h30 coll", "whole RMSE_s", "whole coll", "rank"]
        text += table(block[cols])
        n = {key: int(block[f"{key} n"].max()) for key in HORIZONS}
        text += f"\nEvents with windows: 10 s {n['h10']}, 30 s {n['h30']}, whole event {n['whole']}.\n"
        text += "\n### P2-P4: platoon, stability, rare events\n\n"
        n_range = block["n_profiles_in_range"].max()
        text += (f"P2 (platoon; `*` = over the {n_range:.0f} of {block['n_profiles'].max():.0f} profiles whose leader speeds lie inside the "
                 f"speed range of the training data, pulse from {block['pulse_v0'].max():.1f} m/s):\n\n")
        cols = ["model", "concave_growth_error_in_range", "hysteresis_error_in_range", "growth_exponent_20_in_range",
                "growth_exponent_100_in_range", "wave_speed_in_range", "wave_speed_pulse", "fd_error_rel",
                "collisions_20_in_range", "collisions_100_in_range", "settled_share_in_range", "concave_growth_error", "collisions_100"]
        shown = block[cols].rename(columns=lambda c: c.replace("_in_range", "*"))
        text += table(shown)
        text += "\nP3 (string stability; `unstable` = numerical fraction at 0.05 m/s, `1e3` the practical band, `large` the 1 m/s step):\n\n"
        cols = ["model", "n_equilibria", "unstable_fraction", "unstable_fraction_1e3", "unstable_fraction_large", "unstable_fraction_analytic",
                "max_gain", "max_gain_large", "n_collided_large"]
        text += table(block[cols])
        n_in = block["n_scenarios_in_range"].max()
        text += (f"\nP4 (rare events; `coll syn*` = collision rate over the {n_in:.0f} of 14 synthetic scenarios whose initial speed lies inside "
                 "the training speed range, `hold` = scenarios with a collision before the manoeuvre; the empirical suite is the TGSIM near-miss set):\n\n")
        cols = ["model", "collision_rate_synthetic", "collision_rate_synthetic_in_range", "n_collided_hold", "collision_rate_empirical",
                "min_ttc_q05", "min_ttc_q50", "min_ttc_q50_in_range", "min_ttc_q05_empirical", "min_ttc_q50_empirical",
                "exceedance_4", "exceedance_4_empirical", "amplification_mean"]
        shown = block[cols].rename(columns={"collision_rate_synthetic": "coll syn", "collision_rate_synthetic_in_range": "coll syn*",
                                            "n_collided_hold": "hold", "collision_rate_empirical": "coll emp"})
        text += table(shown)
        text += (f"\nRecorded drivers of the empirical suite: median min TTC {block['min_ttc_q50_recorded'].max():.2f} s, "
                 f"share of samples below -4 m/s^2 {block['exceedance_4_recorded'].max():.4f}.\n")
        scored, note = composite_block(block)
        if not scored.empty:
            text += "\n### Composite traffic-level score\n\n" + table(scored) + note
        lines = [
            kendall(block, "P0 RMSE_a", "h10 RMSE_s"), kendall(block, "h10 RMSE_s", "h30 RMSE_s"),
            kendall(block, "h30 RMSE_s", "whole RMSE_s"), kendall(block, "P0 RMSE_a", "whole RMSE_s"),
            kendall(block, "h10 RMSE_s", "h10 RMSE_v"), kendall(block, "h30 RMSE_s", "h30 RMSE_v"), kendall(block, "whole RMSE_s", "whole RMSE_v"),
        ]
        if not scored.empty:
            merged = block.merge(scored[["model", "composite"]], on="model", how="left")
            lines += [kendall(merged, "P0 RMSE_a", "composite"), kendall(merged, "whole RMSE_s", "composite")]
        text += "\nKendall tau between rankings:\n\n" + "\n".join(line for line in lines if line) + "\n"
    runs = collect_transfer(ROOT, "reconstructed", "driver", 0)
    if not runs.empty and runs["tested_on"].nunique() > 1:
        text += ("\n## P5: transfer\n\nRelative degradation of the closed-loop spacing error: `A->B` is the error of the model "
                 "trained on A and tested on the test part of B, divided by the error of the same model trained on B "
                 "(1 = no loss; below 1 the foreign model beats the native one).\n")
        for horizon in HORIZONS:
            matrix = transfer_matrix(runs, horizon)
            if matrix.empty or matrix.shape[1] < 2:
                continue
            pairs = {}
            for (model, trained), row in matrix.iterrows():
                for tested, value in row.items():
                    if tested != trained and pd.notna(value):
                        pairs.setdefault(model, {})[f"{trained}->{tested}"] = value
            compact = pd.DataFrame.from_dict(pairs, orient="index").sort_index()
            compact.index.name = "model"
            text += f"\n### Horizon {horizon}\n\n" + table(compact.reset_index())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    frame.drop(columns=["directory"]).to_csv(OUT / "runs.csv", index=False)
    if not runs.empty:
        runs.to_csv(OUT / "transfer.csv", index=False)
    print(f"{len(frame)} runs, report written")


if __name__ == "__main__":
    main()
