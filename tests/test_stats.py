"""Statistics package and composite score (protocol contract, sections 6 and 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from trafficloop.protocols.closed_loop import pooled_closed_loop
from trafficloop.protocols.composite import composite_bootstrap, composite_score
from trafficloop.stats import (
    bootstrap_groups,
    holm,
    kendall_tau,
    kendall_tau_bootstrap,
    paired_difference,
    percentile_ci,
    pooled_statistic,
    rank_distribution,
    summarise,
)
from trafficloop.stats import bootstrap as bootstrap_module


def event_table(seed: int = 0, n_groups: int = 15, max_rows: int = 4) -> pd.DataFrame:
    """A per-event table like the closed-loop protocol's, with several events per follower."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for k in range(int(rng.integers(1, max_rows + 1))):
            n = int(rng.integers(20, 300))
            rows.append(
                {
                    "event_id": f"e{g:02d}_{k}",
                    "follower_id": f"d{g:02d}",
                    "n": n,
                    "sse_s": n * rng.gamma(2.0, 1.0),
                    "sse_v": n * rng.gamma(2.0, 0.1),
                    "collided": bool(rng.random() < 0.2),
                    "first_collision": np.nan,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- bootstrap


def test_fast_path_equals_generic_path():
    table = event_table()
    fast = bootstrap_groups(table, n_boot=300, seed=4)
    assert list(fast.columns) == ["rmse_s", "rmse_v", "collision_rate"] and len(fast) == 300
    generic = bootstrap_groups(table, statistic=pooled_statistic, n_boot=300, seed=4)
    np.testing.assert_allclose(generic[fast.columns].to_numpy(), fast.to_numpy(), rtol=1e-12, atol=1e-12)
    # the closed-loop protocol's own statistic on the generic path, and its point estimate
    protocol = bootstrap_groups(table, statistic=pooled_closed_loop, n_boot=300, seed=4)
    np.testing.assert_allclose(protocol[fast.columns].to_numpy(), fast.to_numpy(), rtol=1e-12, atol=1e-12)
    point = pooled_closed_loop(table)
    assert pooled_statistic(table) == pytest.approx({k: point[k] for k in fast.columns}, rel=1e-12)
    # restricting the columns keeps the draws
    only_s = bootstrap_groups(table, n_boot=300, seed=4, columns=["sse_s"])
    assert list(only_s.columns) == ["rmse_s"]
    np.testing.assert_array_equal(only_s["rmse_s"], fast["rmse_s"])


def test_replicates_resample_whole_groups():
    # rmse of a alone 1, of b alone 3: a replicate is {a, a}, {b, b} or {a, b}
    table = pd.DataFrame({"follower_id": ["a", "a", "b"], "n": [10, 30, 20], "sse_s": [10.0, 30.0, 180.0]})
    out = bootstrap_groups(table, n_boot=100, seed=0)["rmse_s"].to_numpy()
    allowed = np.array([1.0, 3.0, np.sqrt(220.0 / 60.0)])
    hits = np.isclose(out[:, None], allowed[None, :], rtol=1e-12)
    assert hits.any(axis=1).all() and hits.any(axis=0).all()


def test_bootstrap_reproducible(monkeypatch):
    table = event_table(seed=1)
    first = bootstrap_groups(table, n_boot=200, seed=7)
    pd.testing.assert_frame_equal(first, bootstrap_groups(table, n_boot=200, seed=7))
    assert not np.allclose(first.to_numpy(), bootstrap_groups(table, n_boot=200, seed=8).to_numpy())
    # the replicates are drawn in blocks (memory bound): the block size does not change them
    monkeypatch.setattr(bootstrap_module, "BLOCK_ELEMENTS", 7)
    pd.testing.assert_frame_equal(first, bootstrap_groups(table, n_boot=200, seed=7))


def test_percentile_ci_known_samples():
    x = np.arange(101.0)
    assert percentile_ci(x, 0.9) == pytest.approx((5.0, 95.0))
    assert percentile_ci(x) == pytest.approx((2.5, 97.5))
    assert percentile_ci(np.r_[x, np.nan]) == pytest.approx((2.5, 97.5))  # NaN left out
    assert np.isnan(percentile_ci([np.nan])).all()
    with pytest.raises(ValueError):
        percentile_ci(x, 1.0)
    summary = summarise(pd.DataFrame({"a": x, "b": 2.0 * x}), level=0.9)
    assert list(summary.columns) == ["mean", "low", "high"]
    assert summary.loc["b"].tolist() == pytest.approx([100.0, 10.0, 190.0])


# ----------------------------------------------------------------------------- ranks


def test_rank_distribution_hand_made():
    rng = np.random.default_rng(0)
    samples = {"B": rng.uniform(1, 2, 400), "A": np.zeros(400), "C": rng.uniform(1, 2, 400), "D": np.full(400, 5.0)}
    out = rank_distribution(samples)
    assert list(out.columns) == ["mean_rank", "rank_low", "rank_high", "p_first", "p_top3"]
    assert out.index[0] == "A" and out.index[-1] == "D"
    assert out.loc["A"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])
    assert out.loc["D"].tolist() == pytest.approx([4.0, 4.0, 4.0, 0.0, 0.0])
    assert out.loc[["B", "C"], "p_first"].tolist() == [0.0, 0.0]
    assert out.loc["B", "mean_rank"] + out.loc["C", "mean_rank"] == pytest.approx(5.0)
    assert out["p_first"].sum() == pytest.approx(1.0) and out["p_top3"].sum() == pytest.approx(3.0)
    assert rank_distribution(samples, lower_is_better=False).index[0] == "D"
    # ties share their positions; a missing value ranks last
    tied = rank_distribution({"X": np.ones(10), "Y": np.ones(10), "Z": np.full(10, np.nan)})
    assert tied.loc["X", "mean_rank"] == 1.5 and tied.loc["X", "p_first"] == 0.5
    assert tied.loc["Z", "mean_rank"] == 3.0 and tied.loc["Z", "p_first"] == 0.0


def test_kendall_tau_identical_and_reversed():
    a = {"m1": 0.1, "m2": 0.5, "m3": 0.3, "m4": 0.9, "m5": 0.7}
    assert kendall_tau(a, dict(a))[0] == pytest.approx(1.0)
    assert kendall_tau(a, {k: -v for k, v in a.items()})[0] == pytest.approx(-1.0)
    # common keys with values in both only
    tau, p = kendall_tau(a, {**a, "extra": 3.0, "m5": np.nan})
    assert tau == pytest.approx(1.0) and p == pytest.approx(stats.kendalltau([1, 2, 3, 4], [1, 2, 3, 4]).pvalue)


def test_kendall_tau_bootstrap():
    rng = np.random.default_rng(2)
    a = {f"m{i}": rng.normal(i, 1.0, 50).round() for i in range(6)}  # rounding makes ties
    b = {f"m{i}": rng.normal(i, 1.0, 50).round() for i in range(6)}
    b["only_b"] = rng.normal(size=50)
    out = kendall_tau_bootstrap(a, b)
    assert out["n_models"] == 6
    per_replicate = [stats.kendalltau([a[k][r] for k in a], [b[k][r] for k in a]).statistic for r in range(50)]
    assert out["tau"] == pytest.approx(np.nanmean(per_replicate), abs=1e-12)
    assert out["low"] <= out["tau"] <= out["high"]
    same = kendall_tau_bootstrap(a, a)
    assert (same["tau"], same["low"], same["high"]) == pytest.approx((1.0, 1.0, 1.0))


# ----------------------------------------------------------------------------- paired


def test_paired_difference_constant_shift():
    table = event_table(seed=3, n_groups=20)
    table["x"] = np.random.default_rng(3).normal(size=len(table))
    shifted = table.assign(x=table["x"] - 0.5)
    out = paired_difference(
        table, shifted, statistic=lambda t: {"mean_x": t["x"].mean()}, column="mean_x", n_boot=200, seed=1
    )
    assert out["n_groups"] == 20
    assert out["mean_difference"] == pytest.approx(0.5, abs=1e-12)
    assert out["low"] - 1e-12 <= 0.5 <= out["high"] + 1e-12
    assert out["high"] - out["low"] < 1e-12  # the same rows in both tables in every replicate
    assert 0.0 < out["p_bootstrap"] <= 2.0 / (200 + 1)  # no replicate at or below 0
    assert out["p_wilcoxon"] < 1e-3


def test_paired_difference_common_random_numbers():
    table = event_table(seed=4, n_groups=12)
    same = paired_difference(table, table.copy(), n_boot=200, seed=2)
    assert (same["mean_difference"], same["low"], same["high"]) == (0.0, 0.0, 0.0)
    assert same["p_bootstrap"] == 1.0 and same["p_wilcoxon"] == 1.0 and same["n_groups"] == 12
    # rmse_s doubles in every replicate; each table's replicates are those of bootstrap_groups
    worse = table.assign(sse_s=4.0 * table["sse_s"])
    out = paired_difference(table, worse, n_boot=200, seed=2)
    reference = bootstrap_groups(table, n_boot=200, seed=2)["rmse_s"].to_numpy()
    assert out["mean_difference"] == pytest.approx(-reference.mean(), rel=1e-12)
    assert (out["low"], out["high"]) == pytest.approx(percentile_ci(-reference), rel=1e-12)
    assert 0.0 < out["p_bootstrap"] <= 2.0 / (200 + 1) and out["p_wilcoxon"] < 1e-3
    generic = paired_difference(table, worse, statistic=pooled_statistic, n_boot=200, seed=2)
    assert generic["mean_difference"] == pytest.approx(out["mean_difference"], rel=1e-12)
    # groups missing from one table are left out with a warning; fewer than 10: no Wilcoxon
    with pytest.warns(UserWarning, match="common groups"):
        partial = paired_difference(table, table[table["follower_id"] != "d00"], n_boot=20)
    assert partial["n_groups"] == 11
    small = table[table["follower_id"].isin([f"d{g:02d}" for g in range(9)])]
    assert np.isnan(paired_difference(small, small, n_boot=20)["p_wilcoxon"])


def test_holm_hand_computed():
    # sorted 0.01, 0.03, 0.04, 0.2 -> 4 * 0.01, 3 * 0.03, max(0.09, 2 * 0.04), 0.2
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(holm([0.5, 0.6]), [1.0, 1.0])  # capped at 1, monotone
    np.testing.assert_allclose(holm([0.01, np.nan, 0.02]), [0.02, np.nan, 0.02])
    assert holm([]).shape == (0,)


# ----------------------------------------------------------------------------- composite


def test_composite_score_hand_made():
    table = pd.DataFrame(
        {
            "concave_growth_error": [0.3, 0.1, np.nan, 0.2],
            "hysteresis_error": [2.0, 1.0, 4.0, 3.0],
            "unstable_fraction": [0.5, 0.0, 1.0, 0.0],
            "collisions_100": [2, 0, 10, 5],
        },
        index=pd.Index(["C", "A", "D", "B"], name="model"),
    )
    out = composite_score(table)
    assert list(out.index) == ["A", "B", "C", "D"]
    assert out.loc["D", "rank_concave_growth_error"] == 4.0  # NaN ranks last
    assert out.loc["A", "rank_unstable_fraction"] == out.loc["B", "rank_unstable_fraction"] == 1.5
    assert out.loc["B", "nrank_hysteresis_error"] == pytest.approx(2.0 / 3.0)
    np.testing.assert_allclose(out["composite"], [0.5 / 12, 5.5 / 12, 0.5, 1.0])
    assert composite_score(table.loc[["B"]])["composite"].tolist() == [0.0]


def composite_inputs() -> tuple[dict, dict]:
    profiles = ["p1", "p2", "p3", "pulse"]
    errors = {"good": [0.1, 0.1, 0.1], "mid": [0.2, 0.9, 0.3], "bad": [0.8, 0.2, 0.9]}
    loops = {"good": [1.0, 1.0, 1.0], "mid": [2.0, 5.0, 3.0], "bad": [6.0, 2.0, 7.0]}
    crashes = {"good": [0, 0, 0, 0], "mid": [1, 1, 2, 1], "bad": [1, 3, 1, 2]}
    unstable = {
        "good": [False] * 6,
        "mid": [False, False, True, True, np.nan, False],
        "bad": [True, True, np.nan, False, True, True],
    }
    per_profile = {
        m: pd.DataFrame(
            {
                "profile": profiles,
                "concave_growth_error": errors[m] + [np.nan],
                "hysteresis_error": loops[m] + [np.nan],
                "collisions_100": crashes[m],
            }
        )
        for m in errors
    }
    per_speed = {m: pd.DataFrame({"v_e": np.arange(5.0, 11.0), "unstable": unstable[m]}) for m in errors}
    per_profile["bad"] = per_profile["bad"].iloc[::-1]  # rows are matched by profile and speed
    per_speed["mid"] = per_speed["mid"].iloc[::-1]
    return per_profile, per_speed


def test_composite_bootstrap():
    per_profile, per_speed = composite_inputs()
    out = composite_bootstrap(per_profile, per_speed, n_boot=60, seed=5)
    pd.testing.assert_frame_equal(out, composite_bootstrap(per_profile, per_speed, n_boot=60, seed=5))
    assert not out.equals(composite_bootstrap(per_profile, per_speed, n_boot=60, seed=6))
    assert list(out.columns) == [
        "mean_rank", "rank_low", "rank_high", "p_first", "p_top3",
        "composite", "composite_mean", "composite_low", "composite_high",
    ]
    assert out.shape == (3, 9) and out.index[0] == "good"
    assert out.loc["good", "p_first"] == 1.0 and out["p_first"].sum() == pytest.approx(1.0)
    assert (out["composite_low"] <= out["composite_mean"]).all()
    assert (out["composite_mean"] <= out["composite_high"]).all()
    # point estimate: errors averaged over the OpenACC profiles, collisions summed over all,
    # unstable fraction over the speeds with an equilibrium
    summary = pd.DataFrame(
        {
            "concave_growth_error": {m: np.nanmean(f["concave_growth_error"]) for m, f in per_profile.items()},
            "hysteresis_error": {m: np.nanmean(f["hysteresis_error"]) for m, f in per_profile.items()},
            "unstable_fraction": {m: np.nanmean(f["unstable"].astype(float)) for m, f in per_speed.items()},
            "collisions_100": {m: f["collisions_100"].sum() for m, f in per_profile.items()},
        }
    )
    expected = composite_score(summary)["composite"]
    np.testing.assert_allclose(out["composite"], expected.reindex(out.index).to_numpy())
    assert out["composite"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="profile values differ"):
        composite_bootstrap({**per_profile, "bad": per_profile["bad"].iloc[1:]}, per_speed, n_boot=5)
