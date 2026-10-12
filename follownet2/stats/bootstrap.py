"""Group bootstrap of the per-item protocol tables (protocol contract, section 7).

The protocol tables hold one row per event, window or episode with the sums that a pooled
score needs (``n``, ``sse_*``) and boolean outcomes (``collided``). The bootstrap resamples
the groups (by default the followers, i.e. the drivers) with replacement: a replicate draws
as many groups as there are, pools every row of the drawn groups, and a group drawn ``k``
times counts ``k`` times (its rows keep their group label). The draws are multinomial counts
of the groups from one ``numpy`` Generator seeded with ``seed``: the same seed and the same
groups give the same replicates, whatever the statistic.

The default statistic (``statistic=None``) is ``rmse_<x> = sqrt(sum(sse_<x>) / sum(n))`` for
every ``sse_<x>`` column plus the mean of every boolean column (``collided`` gives
``collision_rate``, any other column ``<column>_rate``; missing values are left out). It takes
the fast path: the table is summed per group once, and the sums of all replicates are the
product of the count matrix with the per-group sums. A user statistic ``statistic(table) ->
dict`` takes the generic path: the rows of the drawn groups are gathered and the statistic is
called once per replicate. Both paths use the same draws, so they agree to rounding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

Statistic = Callable[[pd.DataFrame], Mapping[str, float]]

#: replicates are drawn and combined in blocks of at most this many group counts (memory);
#: the draws do not depend on it (numpy draws the replicates of a block one after the other)
BLOCK_ELEMENTS = 1 << 22
#: names of the rates of boolean columns that differ from ``<column>_rate``
RATE_NAMES = {"collided": "collision_rate"}


def rate_name(column: str) -> str:
    """Name of the mean of a boolean column in the default statistic."""
    return RATE_NAMES.get(column, f"{column}_rate")


def default_columns(table: pd.DataFrame, columns: Sequence[str] | None = None) -> tuple[list[str], list[str]]:
    """The ``sse_*`` columns and the boolean columns that the default statistic summarises.

    ``columns`` restricts the choice; names in it that are neither (``n``, the group) are
    ignored.
    """
    names = list(table.columns) if columns is None else list(dict.fromkeys(columns))
    missing = [col for col in names if col not in table.columns]
    if missing:
        raise KeyError(f"columns not in the table: {missing}")
    sse = [col for col in names if isinstance(col, str) and col.startswith("sse_")]
    flags = [col for col in names if col not in sse and pd.api.types.is_bool_dtype(table[col])]
    if not sse and not flags:
        raise ValueError("the default statistic needs sse_* or boolean columns")
    if sse and "n" not in table.columns:
        raise KeyError("the pooled RMSE needs the sample counts in the column 'n'")
    return sse, flags


def _ratio(num, den) -> np.ndarray:
    """``num / den``, NaN where ``den`` is not positive."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pooled_statistic(table: pd.DataFrame, columns: Sequence[str] | None = None) -> dict[str, float]:
    """The default statistic on a whole table: the point estimate that the fast path resamples."""
    sse, flags = default_columns(table, columns)
    out: dict[str, float] = {}
    if sse:
        n = table["n"].to_numpy(dtype=np.float64).sum()
        for col in sse:
            out["rmse_" + col[4:]] = float(np.sqrt(_ratio(table[col].to_numpy(dtype=np.float64).sum(), n)))
    for col in flags:
        x = table[col].to_numpy(dtype=np.float64, na_value=np.nan)
        valid = ~np.isnan(x)
        out[rate_name(col)] = float(_ratio(x[valid].sum(), valid.sum()))
    return out


def group_universe(labels: pd.Series) -> pd.Index:
    """The sorted distinct group labels: column ``g`` of a count matrix is ``universe[g]``."""
    if labels.isna().any():
        raise ValueError(f"missing values in the group column {labels.name!r}")
    return pd.Index(labels.unique()).sort_values()


def _count_blocks(rng: np.random.Generator, n_groups: int, n_boot: int) -> Iterator[np.ndarray]:
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if n_groups < 1:
        raise ValueError("nothing to resample: no group")
    p = np.full(n_groups, 1.0 / n_groups)
    block = max(1, BLOCK_ELEMENTS // n_groups)
    for start in range(0, n_boot, block):
        yield rng.multinomial(n_groups, p, size=min(block, n_boot - start))


def draw_counts(rng: np.random.Generator, n_groups: int, n_boot: int) -> np.ndarray:
    """Counts ``[n_boot, n_groups]``: replicate ``r`` draws ``n_groups`` groups with replacement."""
    return np.concatenate(list(_count_blocks(rng, n_groups, n_boot)))


class _Resampler:
    """Replicate statistics of one table for given count matrices over a group universe.

    Rows whose group is not in ``universe`` are left out (paired comparisons).
    """

    def __init__(
        self,
        table: pd.DataFrame,
        group: str,
        statistic: Statistic | None = None,
        columns: Sequence[str] | None = None,
        universe: pd.Index | None = None,
    ):
        if group not in table.columns:
            raise KeyError(f"group column {group!r} not in the table")
        labels = table[group]
        if labels.isna().any():
            raise ValueError(f"missing values in the group column {group!r}")
        self.universe = group_universe(labels) if universe is None else universe
        self.n_groups = len(self.universe)
        codes = self.universe.get_indexer(labels)
        keep = codes >= 0
        if not keep.any():
            raise ValueError("no row belongs to the groups to resample")
        codes = codes[keep]
        self.statistic = statistic
        if statistic is None:
            sse, flags = default_columns(table, columns)
            self.names = ["rmse_" + col[4:] for col in sse] + [rate_name(col) for col in flags]
            self.n_sse, self.n_flags = len(sse), len(flags)
            values = [table["n"].to_numpy(dtype=np.float64)] if sse else []
            values += [table[col].to_numpy(dtype=np.float64) for col in sse]
            outcomes = [table[col].to_numpy(dtype=np.float64, na_value=np.nan) for col in flags]
            values += [np.nan_to_num(x, nan=0.0) for x in outcomes]
            values += [(~np.isnan(x)).astype(np.float64) for x in outcomes]
            # per-group sums [G, K]: n, sse_*, true outcomes, valid outcomes
            self.sums = np.stack(
                [np.bincount(codes, weights=x[keep], minlength=self.n_groups) for x in values], axis=1
            )
            nan = np.isnan(self.sums)
            self._nan = nan.astype(np.float64) if nan.any() else None
            self._clean = np.where(nan, 0.0, self.sums)
        else:
            data = table if columns is None else table[list(columns)]
            order = np.flatnonzero(keep)[np.argsort(codes, kind="stable")]
            self.data = data.iloc[order].reset_index(drop=True)  # rows sorted by group
            self.sizes = np.bincount(codes, minlength=self.n_groups)
            self.starts = np.cumsum(self.sizes) - self.sizes

    def _finish(self, sums: np.ndarray) -> pd.DataFrame:
        k, m = self.n_sse, self.n_flags
        parts = []
        if k:
            parts.append(np.sqrt(_ratio(sums[:, 1 : 1 + k], sums[:, :1])))
        first = 1 + k if k else 0
        if m:
            parts.append(_ratio(sums[:, first : first + m], sums[:, first + m : first + 2 * m]))
        return pd.DataFrame(np.concatenate(parts, axis=1), columns=self.names)

    def apply(self, counts: np.ndarray) -> pd.DataFrame:
        """One row of statistics per row of ``counts`` (``[b, n_groups]``)."""
        if self.statistic is None:
            weights = counts.astype(np.float64)
            sums = weights @ self._clean
            if self._nan is not None:  # a missing sum spoils the replicates that draw its group
                sums[(weights @ self._nan) > 0] = np.nan
            return self._finish(sums)
        records = []
        for row in counts:
            drawn = np.repeat(np.arange(self.n_groups), row)
            lengths = self.sizes[drawn]
            ends = np.cumsum(lengths)
            rows = np.arange(ends[-1]) - np.repeat(ends - lengths - self.starts[drawn], lengths)
            records.append(dict(self.statistic(self.data.iloc[rows].reset_index(drop=True))))
        return pd.DataFrame(records)

    def per_group(self) -> pd.DataFrame:
        """The statistic of every group on its own rows, indexed by the group labels."""
        if self.statistic is None:
            out = self._finish(self.sums)
        else:
            out = pd.DataFrame(
                [
                    dict(self.statistic(self.data.iloc[start : start + size].reset_index(drop=True)))
                    for start, size in zip(self.starts, self.sizes)
                ]
            )
        out.index = self.universe
        return out


def bootstrap_groups(
    table: pd.DataFrame,
    group: str = "follower_id",
    statistic: Statistic | None = None,
    n_boot: int = 1000,
    seed: int | np.random.Generator = 0,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Statistics of ``n_boot`` group-bootstrap replicates of ``table``, one row per replicate.

    ``statistic=None`` takes the fast path of the default statistic (module docstring);
    ``columns`` then restricts the ``sse_*`` and boolean columns it summarises. A user
    ``statistic(table) -> dict`` is called on every replicate table (rows of the drawn groups,
    fresh index); ``columns`` then restricts the columns that this table carries. The rows of
    a group drawn twice appear twice with the same labels: a statistic that aggregates rows by
    an id below the group (``event_collision_rate`` over the windows of an event) needs the
    table aggregated to that id first.
    """
    sampler = _Resampler(table, group, statistic, columns)
    rng = np.random.default_rng(seed)
    out = pd.concat([sampler.apply(c) for c in _count_blocks(rng, sampler.n_groups, n_boot)], ignore_index=True)
    out.index.name = "replicate"
    return out


def percentile_ci(samples, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap samples (linear interpolation, NaN samples left out)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1)")
    x = np.asarray(samples, dtype=np.float64).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    tail = 50.0 * (1.0 - level)
    low, high = np.percentile(x, [tail, 100.0 - tail])
    return float(low), float(high)


def summarise(samples: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Mean and percentile interval of every column of the replicates (one row per column)."""
    rows = {}
    for col in samples.columns:
        x = samples[col].to_numpy(dtype=np.float64)
        x = x[~np.isnan(x)]
        low, high = percentile_ci(x, level)
        rows[col] = {"mean": float(x.mean()) if x.size else float("nan"), "low": low, "high": high}
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["mean", "low", "high"])
    out.index.name = "statistic"
    return out
