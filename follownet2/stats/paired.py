"""Paired comparison of two models on the same items (protocol contract, section 7, where the
module is called ``tests.py``)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from follownet2.stats.bootstrap import Statistic, _count_blocks, _Resampler, percentile_ci

#: fewest paired groups for which the Wilcoxon signed-rank test is reported
MIN_WILCOXON = 10


def _common_groups(table_a: pd.DataFrame, table_b: pd.DataFrame, group: str) -> pd.Index:
    for name, table in (("table_a", table_a), ("table_b", table_b)):
        if group not in table.columns:
            raise KeyError(f"group column {group!r} not in {name}")
    a, b = pd.Index(table_a[group].unique()), pd.Index(table_b[group].unique())
    common = a.intersection(b).sort_values()
    if len(common) == 0:
        raise ValueError("the two tables share no group")
    if len(common) < max(len(a), len(b)):
        warnings.warn(
            f"paired comparison over the {len(common)} common groups: {len(a) - len(common)} groups "
            f"only in table_a and {len(b) - len(common)} only in table_b are left out",
            stacklevel=3,
        )
    return common


def _wilcoxon(differences: np.ndarray) -> float:
    if differences.size < MIN_WILCOXON:
        return float("nan")
    if not np.any(differences):
        return 1.0  # identical on every group (scipy returns 1 with a warning)
    return float(stats.wilcoxon(differences).pvalue)


def paired_difference(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    group: str = "follower_id",
    statistic: Statistic | None = None,
    n_boot: int = 1000,
    seed: int | np.random.Generator = 0,
    column: str = "rmse_s",
) -> dict:
    """Bootstrap of ``statistic(table_a)[column] - statistic(table_b)[column]``.

    Both tables are resampled with the same group draws (common random numbers: one Generator
    seeded with ``seed`` drives both), over the groups present in both tables; with equal group
    sets the replicates of each table are those of :func:`bootstrap_groups` with the same seed.
    ``statistic=None`` is the default statistic of :mod:`follownet2.stats.bootstrap` (fast
    path). Returns ``mean_difference`` (mean over the replicates), ``low``, ``high`` (95 %
    percentile interval), ``p_bootstrap`` (two-sided, ``min(1, 2 min((k_le + 1) / (B + 1),
    (k_ge + 1) / (B + 1)))`` with ``k_le``, ``k_ge`` the replicates with ``d <= 0``, ``d >= 0``
    and ``B`` the replicates (``n_boot``, less any NaN replicate); never 0, at least
    ``2 / (B + 1)``), ``p_wilcoxon`` (signed-rank test on the per-group statistics, each group
    on its own rows; NaN with fewer than ``MIN_WILCOXON`` groups) and ``n_groups``.
    """
    universe = _common_groups(table_a, table_b, group)
    sampler_a = _Resampler(table_a, group, statistic, None, universe)
    sampler_b = _Resampler(table_b, group, statistic, None, universe)

    def pick(frame: pd.DataFrame) -> np.ndarray:
        if column not in frame.columns:
            raise KeyError(f"the statistic has no {column!r}; it gives {list(frame.columns)}")
        return frame[column].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    parts = [
        pick(sampler_a.apply(counts)) - pick(sampler_b.apply(counts))
        for counts in _count_blocks(rng, len(universe), n_boot)
    ]
    d = np.concatenate(parts)
    d = d[~np.isnan(d)]
    low, high = percentile_ci(d)
    if d.size:
        mean = float(d.mean())
        k_le, k_ge, b = int(np.sum(d <= 0)), int(np.sum(d >= 0)), d.size
        p_boot = float(min(1.0, 2.0 * min((k_le + 1) / (b + 1), (k_ge + 1) / (b + 1))))
    else:
        mean = p_boot = float("nan")
    per_group = pick(sampler_a.per_group()) - pick(sampler_b.per_group())
    return {
        "mean_difference": mean,
        "low": low,
        "high": high,
        "p_bootstrap": p_boot,
        "p_wilcoxon": _wilcoxon(per_group[~np.isnan(per_group)]),
        "n_groups": len(universe),
    }


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni (step-down) adjusted p-values, in the input order.

    The ``i``-th smallest of ``m`` p-values becomes ``max_{j <= i} min(1, (m - j + 1) p_(j))``.
    Missing p-values stay missing and do not count in ``m``.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    q = p[valid]
    m = q.size
    if m:
        order = np.argsort(q, kind="stable")
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * q[order]), 1.0)
        out[valid] = adjusted
    return out
