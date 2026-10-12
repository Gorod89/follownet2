"""Rank statistics of models over bootstrap replicates (protocol contract, section 7).

``samples[model]`` holds one statistic per replicate, replicate ``r`` of every model coming
from the same draw (common random numbers). In every replicate the models are ranked with
average ranks on ties; a missing value ranks last.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from follownet2.stats.bootstrap import percentile_ci


def _stack(samples: Mapping[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Model names and the matrix ``[replicates, models]``."""
    models = list(samples)
    if not models:
        raise ValueError("no model to rank")
    columns = [np.asarray(samples[m], dtype=np.float64).ravel() for m in models]
    if len({c.size for c in columns}) != 1:
        raise ValueError("every model needs the same number of replicates")
    if columns[0].size == 0:
        raise ValueError("no replicate")
    return models, np.column_stack(columns)


def replicate_ranks(matrix: np.ndarray, lower_is_better: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ranks of the models (columns) in every replicate (row): the average rank and the first
    and last position of each tie group. Missing values rank last."""
    key = np.asarray(matrix, dtype=np.float64)
    key = key if lower_is_better else -key
    key = np.where(np.isnan(key), np.inf, key)
    average, first, last = (stats.rankdata(key, method=m, axis=1) for m in ("average", "min", "max"))
    return average, first, last


def rank_distribution(samples: Mapping[str, np.ndarray], lower_is_better: bool = True) -> pd.DataFrame:
    """Distribution of the rank of every model over the replicates, best mean rank first.

    Columns: ``mean_rank``; ``rank_low``, ``rank_high``, the 2.5 and 97.5 percentiles of the
    average rank (linear interpolation); ``p_first`` and ``p_top3``, the share of replicates
    in which the model takes the first position or one of the first three, a tie group sharing
    its positions equally (ties broken at random, in expectation; consistent with the average
    rank, which is the expected position), so that ``p_first`` sums to 1 over the models.
    """
    models, matrix = _stack(samples)
    average, first, last = replicate_ranks(matrix, lower_is_better)

    def share(top: int) -> np.ndarray:
        return (np.clip(np.minimum(last, top) - first + 1, 0, None) / (last - first + 1)).mean(axis=0)

    low, high = np.percentile(average, [2.5, 97.5], axis=0)
    out = pd.DataFrame(
        {
            "mean_rank": average.mean(axis=0),
            "rank_low": low,
            "rank_high": high,
            "p_first": share(1),
            "p_top3": share(3),
        },
        index=pd.Index(models, name="model"),
    )
    return out.sort_values("mean_rank", kind="stable")


def kendall_tau(a: Mapping[str, float], b: Mapping[str, float]) -> tuple[float, float]:
    """Kendall's tau-b and its two-sided p-value between two metric maps.

    Over the keys present in both with a value in both (``scipy.stats.kendalltau``); NaN when
    fewer than two keys remain or one ranking is constant.
    """
    keys = [k for k in a if k in b and not (np.isnan(float(a[k])) or np.isnan(float(b[k])))]
    if len(keys) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant input: scipy warns and returns NaN
        result = stats.kendalltau([float(a[k]) for k in keys], [float(b[k]) for k in keys])
    return float(result.statistic), float(result.pvalue)


def _tau_b(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kendall's tau-b of every row pair of ``x``, ``y`` (``[replicates, models]``).

    Models with a missing value in a replicate are left out of that replicate.
    """
    i, j = np.triu_indices(x.shape[1], k=1)
    valid = ~(np.isnan(x) | np.isnan(y))
    pair = valid[:, i] & valid[:, j]
    sx = np.where(pair, (x[:, i] > x[:, j]).astype(np.int64) - (x[:, i] < x[:, j]), 0)
    sy = np.where(pair, (y[:, i] > y[:, j]).astype(np.int64) - (y[:, i] < y[:, j]), 0)
    concordance = (sx * sy).sum(axis=1)
    norm = np.sqrt((sx * sx).sum(axis=1).astype(np.float64) * (sy * sy).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(norm > 0, concordance / norm, np.nan)


def kendall_tau_bootstrap(samples_a: Mapping[str, np.ndarray], samples_b: Mapping[str, np.ndarray]) -> dict:
    """Kendall's tau-b between two rankings over the replicates.

    Replicate ``r`` compares the ``r``-th samples of the models present in both maps.
    Returns ``tau`` (mean over the replicates), ``low``, ``high`` (95 % percentile interval)
    and ``n_models``.
    """
    keys = [k for k in samples_a if k in samples_b]
    if len(keys) < 2:
        return {"tau": float("nan"), "low": float("nan"), "high": float("nan"), "n_models": len(keys)}
    _, x = _stack({k: samples_a[k] for k in keys})
    _, y = _stack({k: samples_b[k] for k in keys})
    if x.shape != y.shape:
        raise ValueError("both maps need the same number of replicates")
    tau = _tau_b(x, y)
    finite = tau[~np.isnan(tau)]
    low, high = percentile_ci(finite)
    return {
        "tau": float(finite.mean()) if finite.size else float("nan"),
        "low": low,
        "high": high,
        "n_models": len(keys),
    }
