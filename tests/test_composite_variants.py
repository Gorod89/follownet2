"""Composite variants with realism inputs (follownet2/protocols/composite.py), their E3b analyses
(follownet2/experiments/analysis.py, scripts/m4_report.py), the Pareto figure (follownet2/figures/plots.py,
scripts/make_report.py) and the v2 columns of the leaderboard, on hand-made tables and a fabricated
results tree: dataset ``alpha`` with five models, one of which (``gap``) keeps a large gap and never
reacts (stable, no collision, small platoon errors, the worst P1 and FD errors)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from follownet2.experiments import analysis as an  # noqa: E402
from follownet2.figures import plots, style  # noqa: E402
from follownet2.leaderboard import render, tables  # noqa: E402
from follownet2.protocols.composite import (  # noqa: E402
    FD_INPUT,
    INPUTS,
    P1_INPUT,
    VARIANTS,
    composite_bootstrap,
    composite_matrix,
    composite_score,
    pareto_front,
)

REPO = Path(__file__).resolve().parents[1]
N_BOOT = 150


# ----------------------------------------------------------------------------- composite_score


def gap_table() -> pd.DataFrame:
    """``gap`` wins all four v1 inputs but has the worst P1 and FD errors."""
    return pd.DataFrame(
        {
            "concave_growth_error": [0.20, 0.30, 0.50, 0.40, 0.60],
            "hysteresis_error": [40.0, 60.0, 90.0, 70.0, 80.0],
            "unstable_fraction": [0.0, 0.0, 1.0, 0.5, 1.0],
            "collisions_100": [0, 0, 300, 10, 500],
            P1_INPUT: [900.0, 4.5, 4.0, 5.0, 8.0],
            FD_INPUT: [9.0, 0.3, 2.0, 0.5, 1.0],
        },
        index=pd.Index(["gap", "idm", "mlp", "gipps", "lstm"], name="model"),
    )


def test_variants_extend_v1():
    assert VARIANTS["v1"] == INPUTS
    assert VARIANTS["v1+p1"] == (*INPUTS, P1_INPUT) and VARIANTS["v1+fd"] == (*INPUTS, FD_INPUT)
    assert VARIANTS["v2"] == (*INPUTS, P1_INPUT, FD_INPUT)
    table = gap_table()
    pd.testing.assert_frame_equal(composite_score(table), composite_score(table, "v1"))  # the default is v1
    with pytest.raises(KeyError, match="unknown composite variant"):
        composite_score(table, "v3")


def test_gap_keeping_model_leads_v1_but_not_v2():
    table = gap_table()
    v1 = composite_score(table, VARIANTS["v1"])
    assert v1.index[0] == "gap" and v1.loc["gap", "composite"] == pytest.approx(1.0 / 16.0)  # ties at 0 on two inputs
    v2 = composite_score(table, VARIANTS["v2"])
    assert list(v2.index) == ["idm", "gap", "gipps", "mlp", "lstm"]
    # gap: ranks 1, 1, 1.5, 1.5, 5, 5 -> (0 + 0 + 0.5 + 0.5 + 4 + 4) / (4 * 6); idm: 2, 2, 1.5, 1.5, 2, 1 -> 4 / 24
    assert v2.loc["gap", "composite"] == pytest.approx(9.0 / 24.0) and v2.loc["idm", "composite"] == pytest.approx(4.0 / 24.0)
    assert v2.loc["gap", f"rank_{P1_INPUT}"] == 5.0 and v2.loc["gap", f"nrank_{FD_INPUT}"] == 1.0
    for variant in ("v1+p1", "v1+fd"):
        assert composite_score(table, variant).index[0] != "gap"


def test_missing_inputs_rank_last():
    table = gap_table()
    table.loc["idm", FD_INPUT] = np.nan
    table.loc["mlp", P1_INPUT] = np.nan
    out = composite_score(table, "v2")
    assert out.loc["idm", f"rank_{FD_INPUT}"] == 5.0 and out.loc["mlp", f"rank_{P1_INPUT}"] == 5.0
    assert out.loc["gap", f"rank_{FD_INPUT}"] == 4.0  # the worst value ranks before a missing one
    both = table.assign(**{FD_INPUT: np.nan})
    assert composite_score(both, "v2")[f"rank_{FD_INPUT}"].tolist() == [3.0] * 5  # all missing: all tie last
    with pytest.raises(KeyError, match="composite inputs missing"):
        composite_score(table.drop(columns=FD_INPUT), "v2")


def test_borda_agrees_with_the_mean_normalised_rank():
    # ranks by hand (average ranks on ties): a 1, 2.5, 3; b 2, 2.5, 1.5; c 3, 1, 1.5
    table = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 5.0, 4.0], "z": [7.0, 6.0, 6.0]}, index=["a", "b", "c"])
    borda = composite_score(table, ("x", "y", "z"), method="borda")
    # Borda points sum(n - rank): a 2 + 0.5 + 0 = 2.5, b 1 + 0.5 + 1.5 = 3, c 0 + 2 + 1.5 = 3.5; maximum 3 * 2 = 6
    assert borda.loc[["a", "b", "c"], "borda"].tolist() == [2.5, 3.0, 3.5]
    assert borda.loc[["a", "b", "c"], "composite"].tolist() == pytest.approx([1 - 2.5 / 6, 1 - 3.0 / 6, 1 - 3.5 / 6])
    mean = composite_score(table, ("x", "y", "z"))
    assert list(borda.index) == list(mean.index) == ["c", "b", "a"]
    np.testing.assert_array_equal(borda["composite"].to_numpy(), mean["composite"].to_numpy())  # exactly equal
    assert "borda" not in mean.columns
    weights = {"x": 2.0}
    weighted = composite_score(table, ("x", "y", "z"), weights, method="borda")
    # weighted points: a 4 + 0.5 + 0 = 4.5, b 2 + 0.5 + 1.5 = 4, c 0 + 2 + 1.5 = 3.5 of 8
    assert weighted.loc[["a", "b", "c"], "borda"].tolist() == [4.5, 4.0, 3.5]
    np.testing.assert_array_equal(weighted["composite"].to_numpy(), composite_score(table, ("x", "y", "z"), weights)["composite"].to_numpy())
    assert list(weighted.index) == ["a", "b", "c"]  # the weight on x reverses the order
    on_real = gap_table()
    np.testing.assert_array_equal(composite_score(on_real, "v2", method="borda")["composite"].to_numpy(),
                                  composite_score(on_real, "v2")["composite"].to_numpy())


def test_weights_mapping_sequence_and_validation():
    table = gap_table()
    by_name = composite_score(table, "v2", {"unstable_fraction": 2.0, "collisions_100": 2.0})
    by_order = composite_score(table, "v2", [1, 1, 2, 2, 1, 1])
    pd.testing.assert_frame_equal(by_name, by_order)
    # weighted mean of the normalised ranks: gap (0 + 0 + 2 * 0.125 + 2 * 0.125 + 1 + 1) / 8
    assert by_name.loc["gap", "composite"] == pytest.approx(2.5 / 8.0)
    with pytest.raises(KeyError, match="not in the composite"):
        composite_score(table, "v1", {FD_INPUT: 2.0})
    for bad in ([1.0, 2.0], [1.0, -1.0, 1.0, 1.0], [0.0] * 4, [np.nan, 1.0, 1.0, 1.0]):
        with pytest.raises(ValueError):
            composite_score(table, "v1", bad)
    with pytest.raises(ValueError, match="unknown method"):
        composite_score(table, "v1", method="copeland")


def test_composite_matrix_equals_composite_score_row_by_row():
    rng = np.random.default_rng(4)
    columns = VARIANTS["v2"]
    inputs = {c: rng.integers(0, 4, size=(30, 6)).astype(float) for c in columns}  # many ties
    inputs[FD_INPUT][rng.random((30, 6)) < 0.2] = np.nan
    for method in ("mean_normalised_rank", "borda"):
        matrix = composite_matrix(inputs, columns, {"collisions_100": 2.0}, method)
        for r in range(30):
            table = pd.DataFrame({c: inputs[c][r] for c in columns})
            expected = composite_score(table, columns, {"collisions_100": 2.0}, method)["composite"].sort_index().to_numpy()
            np.testing.assert_array_equal(matrix[r], expected)


def test_pareto_front_hand_made():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 2.0, np.nan, 6.0]
    y = [5.0, 3.0, 3.0, 1.0, 0.5, 4.0, 3.0, 0.1, 0.5]
    # (3, 3) and (2.5, 4) are dominated by (2, 3); the two (2, 3) are equal and both on the front; NaN is
    # neither on the front nor dominates; (6, 0.5) is dominated by (5, 0.5)
    assert pareto_front(x, y).tolist() == [True, True, False, True, True, False, True, False, False]
    assert pareto_front([], []).tolist() == []
    with pytest.raises(ValueError):
        pareto_front([1.0, 2.0], [1.0])


# ----------------------------------------------------------------------------- composite_bootstrap


def bootstrap_inputs() -> tuple[dict, dict, dict, dict]:
    profiles = ["p1", "p2", "p3", "pulse"]
    errors = {"good": [0.1, 0.1, 0.2], "mid": [0.2, 0.9, 0.3], "twin": [0.2, 0.9, 0.3], "bad": [0.8, 0.2, 0.9]}
    loops = {"good": [1.0, 1.0, 1.0], "mid": [2.0, 5.0, 3.0], "twin": [2.0, 5.0, 3.0], "bad": [6.0, 2.0, 7.0]}
    crashes = {"good": [0, 0, 0, 0], "mid": [1, 1, 2, 1], "twin": [1, 1, 2, 1], "bad": [1, 3, 1, 2]}
    unstable = {"good": [0.0] * 6, "mid": [0, 0, 1, 1, np.nan, 0], "twin": [0, 0, 1, 1, np.nan, 0], "bad": [1, 1, np.nan, 0, 1, 1]}
    per_profile = {m: pd.DataFrame({"profile": profiles, "concave_growth_error": errors[m] + [np.nan],
                                    "hysteresis_error": loops[m] + [np.nan], "collisions_100": crashes[m]}) for m in errors}
    per_speed = {m: pd.DataFrame({"v_e": np.arange(5.0, 11.0), "unstable": unstable[m]}) for m in errors}
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"event_id": [f"e{i}" for i in range(16)], "follower_id": [f"d{i // 2}" for i in range(16)],
                         "n": rng.integers(50, 200, 16)})
    g = rng.gamma(2.0, 1.0, 16)
    quality = {"good": 3.0, "mid": 1.0, "bad": 1.1}
    per_event = {m: base.assign(sse_s=q * rng.gamma(2.0, 1.0, 16) * g * base["n"]) for m, q in quality.items()}
    per_event["twin"] = per_event["mid"].copy()
    fd = {FD_INPUT: {"good": 0.9, "mid": 0.2, "twin": 0.2, "bad": 0.4}}
    return per_profile, per_speed, per_event, fd


def test_composite_bootstrap_with_realism_inputs():
    per_profile, per_speed, per_event, fd = bootstrap_inputs()
    v1 = composite_bootstrap(per_profile, per_speed, n_boot=80, seed=3)
    # the extra arguments do not change the v1 replicates
    pd.testing.assert_frame_equal(v1, composite_bootstrap(per_profile, per_speed, n_boot=80, seed=3, columns="v1",
                                                          per_event=per_event, fixed=fd))
    v2 = composite_bootstrap(per_profile, per_speed, n_boot=80, seed=3, columns="v2", per_event=per_event, fixed=fd)
    pd.testing.assert_frame_equal(v2, composite_bootstrap(per_profile, per_speed, n_boot=80, seed=3, columns="v2",
                                                          per_event=per_event, fixed=fd))
    assert list(v2.columns) == list(v1.columns)
    # point: P2 means over the profiles, the unstable fraction over the speeds with a value, the pooled P1 error, FD as given
    point = pd.DataFrame({
        "concave_growth_error": {m: np.nanmean(f["concave_growth_error"]) for m, f in per_profile.items()},
        "hysteresis_error": {m: np.nanmean(f["hysteresis_error"]) for m, f in per_profile.items()},
        "unstable_fraction": {m: np.nanmean(f["unstable"]) for m, f in per_speed.items()},
        "collisions_100": {m: f["collisions_100"].sum() for m, f in per_profile.items()},
        P1_INPUT: {m: np.sqrt(t["sse_s"].sum() / t["n"].sum()) for m, t in per_event.items()},
        FD_INPUT: fd[FD_INPUT],
    })
    expected = composite_score(point, "v2")["composite"]
    np.testing.assert_allclose(v2["composite"], expected.reindex(v2.index).to_numpy())
    # twin = mid in every input: the drivers are drawn once for all models, so they tie in every replicate
    assert v2.loc["twin", "mean_rank"] == v2.loc["mid", "mean_rank"]
    assert v2.loc["twin", "composite_mean"] == v2.loc["mid", "composite_mean"]
    # FD alone has no resampling unit: the composite is the same in every replicate
    only_fd = composite_bootstrap(per_profile, per_speed, n_boot=40, seed=0, columns=(FD_INPUT,), fixed=fd)
    np.testing.assert_array_equal(only_fd["composite_low"], only_fd["composite_high"])
    only_p1 = composite_bootstrap(per_profile, per_speed, n_boot=40, seed=0, columns=(P1_INPUT,), per_event=per_event)
    assert (only_p1["composite_low"] < only_p1["composite_high"]).any()  # the drivers are resampled
    with pytest.raises(ValueError, match="per_event"):
        composite_bootstrap(per_profile, per_speed, n_boot=5, columns="v2", fixed=fd)
    with pytest.raises(KeyError, match="no source"):
        composite_bootstrap(per_profile, per_speed, n_boot=5, columns="v2", per_event=per_event)
    with pytest.raises(ValueError, match="lacks the models"):
        composite_bootstrap(per_profile, per_speed, n_boot=5, columns="v1+fd", fixed={FD_INPUT: {"good": 1.0}})


# ----------------------------------------------------------------------------- fabricated results tree

#: model -> (family, P0/P1 quality, concave, hysteresis, unstable per speed, collisions per profile, FD error)
WORLD = {
    "gap": ("baseline", 50.0, 0.2, 20.0, [0, 0, 0, 0], [0, 0, 0, 0], 4.0),
    "idm": ("mechanistic", 2.0, 0.3, 30.0, [0, 0, 0, 0], [0, 0, 0, 0], 0.3),
    "gipps": ("mechanistic", 3.0, 0.4, 25.0, [0, 0, 1, 0], [0, 0, 0, 0], 0.5),
    "mlp": ("network", 1.0, 0.6, 60.0, [1, 1, 1, 0], [3, 1, 0, 2], 1.5),
    "lstm": ("network", 1.5, 0.5, 80.0, [1, 1, 1, 1], [5, 2, 4, 1], 2.0),
}
DRIVERS = [f"d{i:02d}" for i in range(10)]
PROFILES = ["A", "B", "C", "pulse"]  # C outside the speed range


def pooled(table: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(table[column].sum() / table["n"].sum()))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_run(root: Path, label: str) -> None:
    family, q, concave, hysteresis, unstable, collisions, fd = WORLD[label]
    d = an.run_dir(root, "alpha", label)
    rng = np.random.default_rng(7)
    base = pd.DataFrame([{"event_id": f"{f}|L|{k}", "follower_id": f, "n": int(rng.integers(150, 300)), "g": float(rng.gamma(2.0, 1.0))}
                         for f in DRIVERS for k in range(2)])
    # one-step accuracy against the composite as in the data: the networks lead P0
    p0_quality = {"gap": 0.5, "idm": 5.0, "gipps": 4.0, "mlp": 0.3, "lstm": 0.1}[label]
    n, g = base["n"].to_numpy(np.float64), base["g"].to_numpy()
    p0 = base[["event_id", "follower_id"]].assign(n=n, sse_a=p0_quality * g * n)
    whole = base[["event_id", "follower_id"]].assign(n=base["n"], sse_s=q * g * n * (1.0 + 0.3 * (base.index % 3)), collided=False)
    block = {"rmse_s": pooled(whole, "sse_s"), "n_events": int(whole["event_id"].nunique()), "n_samples": int(whole["n"].sum())}
    write_json(d / "train.json", {"model": label, "family": family, "config_hash": f"h-{label}",
                                  "test": {"p0": {"rmse_a": pooled(p0, "sse_a")}, "closed_loop_whole_event": block}})
    p0.to_parquet(d / "test_p0.events.parquet", index=False)
    whole.to_parquet(d / "test_closed_loop.events.parquet", index=False)
    rows = []
    for i, profile in enumerate(PROFILES):
        for n_vehicles in (20, 100):
            rows.append({"profile": profile, "source": "synthetic" if profile == "pulse" else "openacc", "n_vehicles": n_vehicles,
                         "concave_growth_error": np.nan if profile == "pulse" else concave * (1 + 0.2 * i),
                         "hysteresis_error": np.nan if profile == "pulse" else hysteresis * (1 + 0.1 * i),
                         "collisions": collisions[i] if n_vehicles == 100 else 0, "in_range": profile != "C"})
    profiles = pd.DataFrame(rows)
    part = profiles[profiles["in_range"] & (profiles["n_vehicles"] == 100)]
    provenance = {"provenance": {"config_hash": f"h-{label}"}}
    write_json(d / "p2.json", {"protocol": "p2", "model": provenance, "summary": {
        "concave_growth_error_in_range": float(part["concave_growth_error"].mean()),
        "hysteresis_error_in_range": float(part["hysteresis_error"].mean()),
        "collisions_100_in_range": int(part["collisions"].sum()), "fd_error_rel": fd, "fd_bins_compared": 10 + len(label)}})
    profiles.to_parquet(d / "p2.profiles.parquet", index=False)
    speeds = pd.DataFrame({"v_e": [5.0, 10.0, 15.0, 20.0], "converged": True, "tracks": True, "unstable_numerical": [bool(u) for u in unstable]})
    write_json(d / "p3.json", {"protocol": "p3", "model": provenance, "summary": {"unstable_fraction": float(np.mean(unstable))}})
    speeds.to_parquet(d / "p3.speeds.parquet", index=False)
    write_json(d / "p4.json", {"protocol": "p4", "model": provenance, "summary": {}})  # present: the evaluation is complete


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> dict:
    top = tmp_path_factory.mktemp("e3b")
    root = top / "models"
    for label in WORLD:
        write_run(root, label)
    rankings = an.e3_rankings(root, "alpha")
    replicates = an.e3_replicates(root, "alpha", rankings, n_boot=N_BOOT, seed=1)
    return {"top": top, "root": root, "rankings": rankings, "replicates": replicates}


def test_composite_inputs_and_variants(world):
    root, rankings = world["root"], world["rankings"]
    inputs = an.composite_inputs(root, "alpha", rankings)
    assert list(inputs.columns) == an.INPUTS_COLUMNS and list(inputs["model"]) == list(rankings["model"])
    assert inputs.set_index("model").loc["idm", FD_INPUT] == 0.3 and inputs.set_index("model").loc["mlp", "fd_bins_compared"] == 13
    variants = an.e3_composite_variants(root, "alpha", rankings=rankings, replicates=world["replicates"])
    assert list(variants.columns) == an.VARIANTS_COLUMNS
    by_model = variants.set_index("model")
    # v1 is the composite of e3_rankings; Borda equals v2
    np.testing.assert_array_equal(by_model["composite_v1"].reindex(rankings["model"]).to_numpy(), rankings["composite"].to_numpy())
    np.testing.assert_array_equal(variants["composite_borda_v2"].to_numpy(), variants["composite_v2"].to_numpy())
    # the gap-keeping model leads v1 and loses the lead under v2
    assert by_model.loc["gap", "rank_v1"] == 1.0 and by_model.loc["gap", "rank_v2"] > 1.0
    assert variants["model"].iloc[0] == "idm" and variants["composite_v2"].is_monotonic_increasing
    assert np.isfinite(variants[["rank_v2_low", "rank_v2_high"]].to_numpy()).all()
    assert (variants["rank_v2_low"] <= variants["rank_v2_high"]).all()
    # the same replicates computed inside give the same intervals
    again = an.e3_composite_variants(root, "alpha", n_boot=N_BOOT, seed=1)
    pd.testing.assert_frame_equal(again, variants)


def test_sensitivity_rows_and_the_e3_headline(world):
    root, rankings, replicates = world["root"], world["rankings"], world["replicates"]
    table = an.e3_composite_sensitivity(root, "alpha", rankings=rankings, replicates=replicates)
    assert list(table.columns) == an.SENSITIVITY_COLUMNS
    labels = list(table["variant_or_leave_one_out"])
    leave = [f"v2 without {c}" for c in VARIANTS["v2"]]
    weights = ["v2, weight 2 on unstable_fraction", "v2, weight 2 on collisions_100",
               "v2, weight 2 on unstable_fraction and collisions_100"]
    assert labels == [*VARIANTS, "borda v2", *leave, *weights]
    rows = table.set_index("variant_or_leave_one_out")
    assert rows.loc["v1", "kendall_tau_vs_v1"] == pytest.approx(1.0) and rows.loc["v2", "kendall_tau_vs_v2"] == pytest.approx(1.0)
    assert rows.loc["borda v2", "kendall_tau_vs_v2"] == pytest.approx(1.0) and rows.loc["borda v2", "top3_changes"] == "vs v2: none"
    assert rows.loc["v1", "top3_changes"] == "reference" and rows.loc["v2", "top3_changes"] == "vs v1: none"
    assert an._top3_change(["a", "b", "c", "d"], np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1]), "v1") == "vs v1: enter a; leave d"
    assert an._top3_text(["a", "b", "c", "d"], np.array([1.0, 2, 3, 3])) == "a, b, c (tied with c: d)"
    assert all(len(rows.loc[label, "inputs"].split(", ")) == 5 for label in leave)
    assert rows.loc[weights[2], "inputs"] == "concave, hysteresis, unstable x2, collisions x2, P1 whole, FD"
    assert rows.loc["v1", "top3"].startswith("gap, ")
    # the v1 row reproduces the P0-composite row of E3 (same composite replicates, same driver draw)
    kendall = an.e3_kendall(root, "alpha", rankings=rankings, replicates=replicates).set_index(["ranking_a", "ranking_b"])
    headline = kendall.loc[("P0", "composite")]
    assert (rows.loc["v1", "tau_p0"], rows.loc["v1", "tau_p0_low"], rows.loc["v1", "tau_p0_high"]) == \
        pytest.approx((headline["tau"], headline["low"], headline["high"]))
    assert np.isfinite(table[["tau_p0", "tau_p0_low", "tau_p0_high"]].to_numpy()).all() and (table["note"] == "").all()
    # without driver replicates of the whole event the realism variants keep their point tau and lose the interval
    samples, reasons = replicates
    reduced = ({k: v for k, v in samples.items() if k != "p1_whole_rmse_s"}, {"p1_whole_rmse_s": "no table"})
    partial = an.e3_composite_sensitivity(root, "alpha", rankings=rankings, replicates=reduced).set_index("variant_or_leave_one_out")
    assert np.isnan(partial.loc["v2", "tau_p0_low"]) and "no table" in partial.loc["v2", "note"]
    assert partial.loc["v2", "tau_p0"] == rows.loc["v2", "tau_p0"] and np.isfinite(partial.loc["v1+fd", "tau_p0_low"])


def test_pareto_table(world):
    pareto = an.pareto_table(world["root"], "alpha", rankings=world["rankings"])
    assert list(pareto.columns) == an.PARETO_COLUMNS
    assert pareto[P1_INPUT].is_monotonic_increasing
    np.testing.assert_array_equal(pareto["on_front"].to_numpy(), pareto_front(pareto[P1_INPUT], pareto["composite_v2"]))
    assert pareto["on_front"].any() and not pareto.set_index("model").at["gap", "on_front"]  # the worst P1 error, not the best v2
    variants = an.e3_composite_variants(world["root"], "alpha", rankings=world["rankings"], replicates=world["replicates"])
    np.testing.assert_array_equal(pareto.set_index("model")["composite_v2"].reindex(variants["model"]).to_numpy(),
                                  variants["composite_v2"].to_numpy())


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_section_e3b(world):
    report = load("m4_report")
    cache = {"alpha": (world["rankings"], world["replicates"])}
    text, variants, sensitivity, pareto = report.section_e3b(world["root"], cache, N_BOOT, 1)
    assert "## E3b: composite variants and sensitivity" in text and "### alpha" in text
    assert "Top five under v2: idm, " in text and "Kendall tau of P0 against the composite: v1 " in text
    assert "On the front:" in text and "Across datasets: the sign of tau(P0, composite)" in text
    assert len(variants) == len(WORLD) and len(sensitivity) == 14 and len(pareto) == len(WORLD)
    sentence, kept = report.p0_sign_sentence(sensitivity)
    signs = np.sign(sensitivity.set_index("variant_or_leave_one_out").loc[list(VARIANTS), "tau_p0"])
    assert kept == bool((signs == signs["v1"]).all()) and ("kept under every variant" in sentence) == kept
    flipped = sensitivity.assign(tau_p0=np.where(sensitivity["variant_or_leave_one_out"] == "v2", -sensitivity["tau_p0"],
                                                 sensitivity["tau_p0"]))
    sentence, kept = report.p0_sign_sentence(flipped)
    assert not kept and "reversed under v2" in sentence
    empty = report.section_e3b(world["root"], {}, N_BOOT, 1)
    assert "Not available" in empty[0] and empty[1].empty


def test_report_writes_the_e3b_tables(world):
    report = load("m4_report")
    out = world["top"] / "m4"
    path = report.main([f"root={world['root']}", f"out={out}", "n_boot=30", "models=all",
                        f"splits={world['top'] / 'none'}", f"events={world['top'] / 'none'}"])
    text = path.read_text(encoding="utf-8")
    assert text.index("## E3:") < text.index("## E3b") < text.index("## E4")
    for name, columns in (("e3_composite_variants", an.VARIANTS_COLUMNS), ("e3_composite_sensitivity", an.SENSITIVITY_COLUMNS),
                          ("e3_pareto", an.PARETO_COLUMNS)):
        frame = pd.read_csv(out / f"{name}.csv")
        assert list(frame.columns) == columns and set(frame["dataset"]) == {"alpha"}
    # models=main keeps the labels of scripts/m2.py only: none of the fabricated ones
    labels_of = report.label_filter("main")
    assert labels_of(world["root"], "alpha") == ["gipps", "idm", "lstm", "mlp"]  # "gap" is not a reference model
    assert report.label_filter("all")(world["root"], "alpha") is None
    with pytest.raises(SystemExit, match="models="):
        report.arguments(["models=some"])


def test_report_leaves_out_a_dataset_whose_campaign_is_still_evaluating(world, tmp_path):
    report = load("m4_report")
    assert report.e3_datasets(world["root"]) == ["alpha"]
    copy = tmp_path / "models"
    shutil.copytree(world["root"], copy)
    (an.run_dir(copy, "alpha", "lstm") / "p4.json").unlink()  # lstm is being evaluated: P2 and P3 written, P4 not yet
    incomplete = {}
    assert report.e3_datasets(copy, None, incomplete) == []
    assert incomplete == {"alpha": "4 of 5 runs with P2-P4 (missing e.g. lstm)"}
    text, _, _, cache = report.section_e3(copy, 20, 0)
    assert cache == {} and "Left out while their campaign is still evaluating" in text and "alpha: 4 of 5 runs" in text


# ----------------------------------------------------------------------------- leaderboard


def test_leaderboard_v2_columns_follow_the_composite():
    assert tables.GROUPS["composite"][-2:] == ["composite_v2", "rank_v2"]
    assert tables.MAIN_COLUMNS.index("composite_v2") == tables.MAIN_COLUMNS.index("composite_rank_high") + 1
    main = pd.DataFrame({"model": ["idm", "gap"], "family": ["mechanistic", "baseline"], "composite": [0.25, 0.0],
                         "composite_rank": [2.0, 1.0], "composite_rank_low": [1.0, 1.0], "composite_rank_high": [2.0, 2.0],
                         "composite_v2": [0.2, 0.375], "rank_v2": [1.0, 2.5]})
    shown = render.display(main, render.MAIN_CELLS)
    headers = list(shown.columns)
    assert headers[headers.index("composite rank [95% CI]") + 1: headers.index("composite rank [95% CI]") + 3] == ["composite v2", "rank v2"]
    assert shown["composite v2"].tolist() == ["0.200", "0.375"] and shown["rank v2"].tolist() == ["1", "2.5"]
    assert "Composite v2" in tables.V2_NOTE and "fd_error_rel" in tables.V2_NOTE and "sorted by the composite" in tables.V2_NOTE
    v2 = an.variant_composite(gap_table().reset_index(), "v2")
    np.testing.assert_array_equal(v2, composite_score(gap_table(), "v2")["composite"].reindex(gap_table().index).to_numpy())


# ----------------------------------------------------------------------------- figure


def pareto_frame(with_front: bool = True) -> pd.DataFrame:
    rows = []
    for dataset, scale in (("ngsim", 1.0), ("follownet_highd", 0.4)):
        values = {"idm": (6.1, 0.27), "gipps": (6.0, 0.18), "newell": (8.4, 0.16), "mlp_loop": (5.6, 0.25), "mlp": (9.8, 0.64),
                  "lstm": (81.0, 0.75), "persistence": (828.0, 0.65), "ddpg": (8.0, 0.28)}
        families = {"idm": "mechanistic", "gipps": "mechanistic", "newell": "mechanistic", "persistence": "baseline"}
        for model, (x, y) in values.items():
            rows.append({"dataset": dataset, "model": model, "family": families.get(model, "network"),
                         P1_INPUT: x * scale if model != "persistence" else x * scale ** 2, "composite_v2": y})
    frame = pd.DataFrame(rows)
    if with_front:
        frame["on_front"] = np.concatenate([pareto_front(b[P1_INPUT], b["composite_v2"]) for _, b in frame.groupby("dataset", sort=False)])
    return frame


def step_vertices(front: pd.DataFrame) -> list[tuple[float, float]]:
    front = front.sort_values([P1_INPUT, "composite_v2"])
    out = []
    for k, (x, y) in enumerate(zip(front[P1_INPUT], front["composite_v2"])):
        if k:
            out.append((x, out[-1][1]))
        out.append((x, y))
    return out


def test_pareto_scatter_panels_labels_and_front(tmp_path):
    frame = pareto_frame()
    fig = plots.pareto_scatter(frame)
    assert isinstance(fig, Figure) and float(fig.get_size_inches()[0] * 25.4) == pytest.approx(style.WIDTH_MM)
    axes = [ax for ax in fig.axes if ax.get_title(loc="left")]
    assert [ax.get_title(loc="left") for ax in axes] == ["(a) NGSIM", "(b) FollowNet highD"]
    for ax, (dataset, block) in zip(axes, frame.groupby("dataset", sort=False)):
        assert ax.get_xscale() == "log"
        texts = sorted(t.get_text() for t in ax.texts)
        assert texts == sorted(block["model"])  # every point labelled once
        front = block[block["on_front"]]
        assert ax.get_title(loc="right").startswith(f"{len(front)} of {len(block)} models on the front")
        steps = [line for line in ax.lines if line.get_linestyle() == "-"]
        assert len(steps) == 1
        np.testing.assert_allclose(np.column_stack(steps[0].get_data()), np.array(step_vertices(front)))
        dotted = [line for line in ax.lines if line.get_linestyle() != "-"]
        assert len(dotted) == 2  # the boundary of the dominated region beyond the front
    assert ["baseline", "mechanistic", "network", "on the front", "Pareto front"] == [t.get_text() for t in fig.legends[0].get_texts()]
    # labels do not overlap each other
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        boxes = [t.get_window_extent(renderer) for t in ax.texts]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert plots._overlap(boxes[i].extents, boxes[j].extents) < 1.0, (ax.texts[i].get_text(), ax.texts[j].get_text())
    png, pdf = style.save(fig, tmp_path / "pareto")
    assert png.stat().st_size > 10_000 and pdf.stat().st_size > 5_000
    # without on_front the front is computed
    computed = plots.pareto_scatter(pareto_frame(with_front=False))
    first = [ax for ax in computed.axes if ax.get_title(loc="left")][0]
    assert first.get_title(loc="right") == axes[0].get_title(loc="right")
    with pytest.raises(ValueError, match="composite_v2"):
        plots.pareto_scatter(frame.drop(columns="composite_v2"))


def test_make_report_draws_the_pareto_figure(tmp_path):
    m4 = tmp_path / "m4"
    m4.mkdir()
    pareto_frame().to_csv(m4 / "e3_pareto.csv", index=False)
    out = tmp_path / "figures"
    paths = load("make_report").main([f"root={tmp_path / 'models'}", f"m4={m4}", f"out={out}"])
    assert {p.name for p in paths} == {"e3_pareto_scatter.png", "e3_pareto_scatter.pdf", "README.md"}
    lines = [line for line in (out / "README.md").read_text(encoding="utf-8").splitlines() if line.startswith("- `e3_pareto")]
    assert len(lines) == 2 and all("Pareto front" in line and "e3_pareto.csv" in line for line in lines)
