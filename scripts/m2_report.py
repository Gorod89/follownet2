"""Collect the training runs of milestone 2 into ``results/m2/REPORT.md``.

    python scripts/m2_report.py

Reads every ``results/models/<dataset>/<version>/<split>-fold<k>/<model>/seed<s>/train.json``.
"""

from __future__ import annotations

import pandas as pd

from follownet2.utils import REPO_ROOT, read_json

ROOT = REPO_ROOT / "results" / "models"
OUT = REPO_ROOT / "results" / "m2"
FAMILY_ORDER = {"baseline": 0, "mechanistic": 1, "network": 2}


def collect() -> pd.DataFrame:
    rows = []
    for path in sorted(ROOT.rglob("train.json")):
        r = read_json(path)
        p0, closed, log = r["test"]["p0"], r["test"]["closed_loop_whole_event"], r["training"]
        rows.append({
            "dataset": r["dataset"], "version": r["data_version"],
            "split": f'{r["split"]["kind"]}-{r["split"]["fold"]}', "seed": r["seed"],
            "model": r["model"], "family": r["family"], "parameters": r["n_parameters"],
            "minutes": log.get("seconds", 0.0) / 60.0,
            "val RMSE_a": log.get("best_val_rmse_a"),
            "P0 RMSE_a": p0["rmse_a"], "P0 RMSE_v": p0["rmse_v"], "P0 RMSE_s": p0["rmse_s"],
            "loop RMSE_s": closed["rmse_s"], "loop RMSE_v": closed["rmse_v"],
            "collisions": closed["collision_rate"], "test events": closed["n_events"],
        })
    return pd.DataFrame(rows)


def table(frame: pd.DataFrame, digits: int = 3) -> str:
    fmt = lambda v: ("" if pd.isna(v) else f"{v:.{digits}f}") if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def agreement(block: pd.DataFrame) -> str:
    """Kendall's tau between the one-step ranking and the closed-loop ranking of a block."""
    from scipy.stats import kendalltau

    newline = chr(10)
    lines = []
    for name, part in (("all models", block), ("learned models only", block[block["family"] == "network"])):
        if len(part) >= 4:
            tau, p = kendalltau(part["P0 RMSE_a"], part["loop RMSE_s"])
            lines.append(f"* Kendall tau between the one-step and the closed-loop ranking, {name}: {tau:.2f} (p = {p:.3f}, n = {len(part)})")
    return newline + newline.join(lines) + newline if lines else ""


def main() -> None:
    frame = collect()
    if frame.empty:
        raise SystemExit(f"no training run found under {ROOT}")
    text = "# Milestone 2: reference models\n\n"
    text += (
        "One-step scores (P0) and closed-loop scores over whole events on the test part. "
        "`ratio` is the one-step acceleration error relative to the persistence baseline of the "
        "same data; `rank` orders the models of a block by the closed-loop spacing error.\n"
    )
    for keys, block in frame.groupby(["dataset", "version", "split"], sort=True):
        block = block.copy()
        base = block.loc[block["model"] == "persistence", "P0 RMSE_a"]
        block["ratio"] = block["P0 RMSE_a"] / float(base.iloc[0]) if len(base) else float("nan")
        block["rank"] = block["loop RMSE_s"].rank(method="min").astype(int)
        block["order"] = block["family"].map(FAMILY_ORDER)
        block = block.sort_values(["order", "loop RMSE_s"])
        columns = ["model", "family", "parameters", "minutes", "P0 RMSE_a", "ratio", "P0 RMSE_v",
                   "loop RMSE_s", "loop RMSE_v", "collisions", "rank"]
        n = int(block["test events"].iloc[0])
        text += f"\n## {keys[0]}, {keys[1]}, split {keys[2]} ({n} test events)\n\n" + table(block[columns])
        text += agreement(block)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    frame.to_csv(OUT / "runs.csv", index=False)
    print(f"{len(frame)} runs, report written")


if __name__ == "__main__":
    main()
