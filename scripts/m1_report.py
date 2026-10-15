"""Collect the key numbers of milestone 1 into ``results/m1/REPORT.md``.

    python scripts/m1_report.py [dataset ...]

Reads the summaries written by ``scripts/m1.py`` (``results/m1/<dataset>_<step>.json``) and
the parameter spread files in ``data/calibration``.
"""

from __future__ import annotations

import sys

import pandas as pd

from trafficloop.models.mechanistic import MECHANISTIC
from trafficloop.utils import REPO_ROOT, read_json

ROOT = REPO_ROOT / "results" / "m1"
CALIBRATION = REPO_ROOT / "data" / "calibration"
AT_BOUND = 0.10  # share of per-event estimates at a bound that is worth a flag


def table(rows: list[dict], columns: list[str] | None = None, digits: int = 3) -> str:
    if not rows:
        return "_no rows_\n"
    frame = pd.DataFrame(rows)
    if columns:
        frame = frame[[c for c in columns if c in frame.columns]]
    fmt = lambda v: ("" if pd.isna(v) else f"{v:.{digits}f}") if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def load(dataset: str, step: str):
    path = ROOT / f"{dataset}_{step}.json"
    return read_json(path) if path.exists() else None


def datasets_section(names: list[str]) -> str:
    rows, rec, miss = [], [], []
    for name in names:
        b = load(name, "build")
        if b is None:
            continue
        per = b["events_per_follower"]
        rows.append({
            "dataset": name,
            "events": b["raw"]["n_events"],
            "followers": b["raw"]["n_followers"],
            "sites": b["raw"]["n_sites"],
            "hours": b["raw"]["duration_total_s"] / 3600.0,
            "median duration [s]": b["raw"]["duration_median_s"],
            "free / steady / congested": " / ".join(str(b["regimes"].get(r, 0)) for r in ("free", "steady", "congested")),
            "events per follower": per["mean"],
            "events of followers with several events": per["share_events_of_followers_with_several_events"],
        })
        r = b["reconstruction"]
        rec.append({
            "dataset": name,
            "failed": b["n_failed_reconstruction"],
            "gap change [m]": r["median_rms_gap_change"],
            "speed change [m/s]": r["median_rms_speed_change"],
            "acc. change [m/s2]": r["median_rms_acc_change"],
            "q99 max abs acc.: raw": r["q99_max_abs_acc_raw"],
            "rec.": r["q99_max_abs_acc"],
            "q99 max abs jerk: raw": r["q99_max_abs_jerk_raw"],
            "rec. ": r["q99_max_abs_jerk"],
            "gap bound active": r["share_gap_bound_active"],
        })
        if b.get("near_miss"):
            m = b["near_miss"]
            miss.append({
                "dataset": name,
                "episodes": m["n_events"],
                "followers": m["n_followers"],
                "with cut-in": m["with_cut_in"],
                **{f"trigger {k}": v for k, v in m["triggers"].items()},
                "median of the smallest TTC [s]": (m["min_ttc_quantiles"] or {}).get("0.5"),
            })
    return (
        "## 1. Events (raw version)\n\n" + table(rows, digits=2)
        + "\n## 2. Reconstruction (median change per event, 99 % quantile of the extremes)\n\n" + table(rec, digits=2)
        + "\n## 3. Near-miss episodes\n\n" + table(pd.DataFrame(miss).fillna(0).to_dict("records") if miss else [], digits=2)
    )


def splits_section(names: list[str]) -> str:
    rows = []
    for name in names:
        report = load(name, "splits") or {}
        for kind, r in report.items():
            if "skipped" in r:
                rows.append({"dataset": name, "split": kind, "folds": "skipped: " + r["skipped"]})
                continue
            rows.append({
                "dataset": name,
                "split": kind,
                "folds": r["n_folds"],
                "train / val / test / unused": f'{r["n_train"]} / {r["n_val"]} / {r["n_test"]} / {r["n_unused"]}',
                "driver-disjoint": r["driver_disjoint"],
                "test followers seen in training": r["share_test_followers_seen"],
                "test events led by a training follower": r["share_test_events_with_training_follower_as_leader"],
            })
    return "\n## 4. Splits (fold 0)\n\n" + table(rows)


def baselines_section(names: list[str]) -> str:
    rows = []
    for name in names:
        for r in load(name, "baselines") or []:
            rows.append({"dataset": name, **r})
    if not rows:
        return ""
    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(index=["dataset", "version", "split", "n_events"], columns="model", values=["rmse_a", "rmse_v", "rmse_s"])
    out = pd.DataFrame({
        "RMSE_a persistence": wide[("rmse_a", "persistence")],
        "RMSE_a constant speed": wide[("rmse_a", "constant_speed")],
        "ratio": wide[("rmse_a", "persistence")] / wide[("rmse_a", "constant_speed")],
        "RMSE_v persistence": wide[("rmse_v", "persistence")],
        "RMSE_v floor": wide[("rmse_v", "recorded_acceleration")],
        "RMSE_s persistence": wide[("rmse_s", "persistence")],
        "RMSE_s floor": wide[("rmse_s", "recorded_acceleration")],
    }).reset_index()
    identity = frame.loc[frame["model"] == "persistence", "identity_error"].max()
    return (
        "\n## 5. P0 baselines on the test part (fold 0)\n\n"
        "`floor` is the score of the recorded acceleration, i.e. the inconsistency of the record.\n"
        f"Largest deviation from the persistence identity over all rows: {identity:.2e} m/s^2.\n\n"
        + table(out.to_dict("records"), digits=4)
    )


def _at_bound(model: str, params: dict[str, float], tol: float = 0.01) -> str:
    cls = MECHANISTIC[model]
    flags = []
    for name, (low, high) in zip(cls.param_names, cls.bounds):
        if params[name] <= low + tol * (high - low):
            flags.append(f"{name} low")
        elif params[name] >= high - tol * (high - low):
            flags.append(f"{name} high")
    return ", ".join(flags)


def calibration_section(names: list[str]) -> str:
    spread, fits = [], []
    for name in names:
        for r in load(name, "calibration") or []:
            if r["kind"] == "per_event":
                path = CALIBRATION / name / r["version"] / f"{r['model']}_spread.json"
                detail = read_json(path)["parameters"] if path.exists() else {}
                flags = [
                    f"{p} {side} {d[key]:.2f}"
                    for p, d in detail.items()
                    for side, key in (("low", "share_at_lower"), ("high", "share_at_upper"))
                    if (d.get(key) or 0.0) >= AT_BOUND
                ]
                spread.append({
                    "dataset": name, "version": r["version"], "model": r["model"], "events": r["n_events"],
                    "RMSE_s": r["rmse_s"], "RMSE_v": r["rmse_v"], "collisions": r["collision_rate"],
                    "median (coefficient of variation)": ", ".join(
                        f"{k} {v:.2f} ({r['cv'][k]:.2f})" for k, v in r["params"].items()
                    ),
                    "share at a bound": ", ".join(flags),
                })
            else:
                fits.append({
                    "dataset": name, "version": r["version"], "model": r["model"], "split": r["kind"].split("/")[1],
                    "train events": r["n_events"], "train RMSE_s": r["train_rmse_s"], "test RMSE_s": r["rmse_s"],
                    "test RMSE_v": r["rmse_v"], "test collisions": r["collision_rate"],
                    "parameters": ", ".join(f"{k} {v:.2f}" for k, v in r["params"].items()),
                    "at a bound": _at_bound(r["model"], r["params"]),
                })
    text = "\n## 6. Calibration per event: median errors, parameter spread\n\n" + table(spread)
    text += "\n## 7. Global calibration on the training part, scored on the test part (whole event)\n\n" + table(fits)
    frame = pd.DataFrame(fits)
    if len(frame) and frame["split"].nunique() > 1:
        wide = frame.pivot_table(index=["dataset", "version", "model"], columns="split", values="test RMSE_s")
        if {"driver", "event"} <= set(wide.columns):
            wide["driver / event - 1"] = wide["driver"] / wide["event"] - 1.0
            text += "\n## 8. Test RMSE_s under the driver split relative to the event split\n\n"
            text += table(wide.reset_index().to_dict("records"))
    return text


def main() -> None:
    names = sys.argv[1:] or sorted({p.name.rsplit("_", 1)[0] for p in ROOT.glob("*_build.json")})
    text = "# Milestone 1: key numbers\n\nDatasets: " + ", ".join(names) + "\n\n"
    text += datasets_section(names) + splits_section(names) + baselines_section(names) + calibration_section(names)
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"written {ROOT / 'REPORT.md'}".encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
