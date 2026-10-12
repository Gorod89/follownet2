"""Composite traffic-level score (protocol contract, section 6).

Inputs per model, lower is better. The original composite (``v1``, :data:`INPUTS`) ranks
``concave_growth_error`` and ``hysteresis_error`` (P2, mean over the OpenACC profiles),
``unstable_fraction`` (P3, numerical, over the speeds with an equilibrium) and ``collisions_100``
(P2, summed over all profiles). It has no realism input: a model that keeps a large gap and never
reacts is stable and never collides, so it can lead. :data:`VARIANTS` add the two realism inputs
``p1_whole_rmse_s`` (P1, whole-event closed-loop spacing RMSE on the test drivers) and
``fd_error_rel`` (P2, relative fundamental-diagram error, pooled over the samples of every
profile): ``v1+p1``, ``v1+fd`` and both, ``v2``.

Each input is ranked within the compared set (average ranks on ties, a missing value last). The
composite of ``method="mean_normalised_rank"`` is the weighted mean of the normalised ranks
``(rank - 1) / (n - 1)``; ``method="borda"`` sums the weighted Borda points ``n - rank`` (column
``borda``, higher is better) and normalises them to ``1 - borda / ((n - 1) sum(w))`` in [0, 1].
Both composites are lower-is-better and, with average ranks on ties, equal: the Borda count is an
affine function of the rank sum, so the two methods order the models identically (the composite
is computed from the exact weighted rank sum and one division, so equal rank sums give equal
composites). The bootstrap resamples the P2 profiles and the P3 speeds with replacement and, for
``p1_whole_rmse_s``, the drivers, the same draw for all models, and recomputes the composite of
every replicate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from follownet2.stats.bootstrap import _Resampler, draw_counts, summarise
from follownet2.stats.ranks import rank_distribution

INPUTS = ("concave_growth_error", "hysteresis_error", "unstable_fraction", "collisions_100")
PROFILE_COLUMNS = ("concave_growth_error", "hysteresis_error", "collisions_100")
#: the realism inputs: whole-event P1 spacing RMSE and the relative fundamental-diagram error of P2
P1_INPUT, FD_INPUT = "p1_whole_rmse_s", "fd_error_rel"
#: the composites: name -> inputs (v1 is the composite of the protocol contract)
VARIANTS: dict[str, tuple[str, ...]] = {
    "v1": INPUTS,
    "v1+p1": (*INPUTS, P1_INPUT),
    "v1+fd": (*INPUTS, FD_INPUT),
    "v2": (*INPUTS, P1_INPUT, FD_INPUT),
}
METHODS = ("mean_normalised_rank", "borda")
_KNOWN = tuple(dict.fromkeys(c for columns in VARIANTS.values() for c in columns))


def _columns(columns: Sequence[str] | str) -> tuple[str, ...]:
    """The inputs of a composite: a sequence of column names or the name of a variant."""
    if isinstance(columns, str):
        if columns not in VARIANTS:
            raise KeyError(f"unknown composite variant {columns!r}; variants: {list(VARIANTS)}")
        return VARIANTS[columns]
    columns = tuple(columns)
    if not columns:
        raise ValueError("a composite needs at least one input")
    if len(set(columns)) != len(columns):
        raise ValueError(f"repeated composite inputs: {list(columns)}")
    return columns


def _weights(columns: Sequence[str], weights: Mapping[str, float] | Sequence[float] | None) -> np.ndarray:
    """Weights aligned with ``columns``: all 1 for None, a mapping (unnamed inputs 1) or a sequence."""
    if weights is None:
        w = np.ones(len(columns))
    elif isinstance(weights, Mapping):
        unknown = [k for k in weights if k not in columns]
        if unknown:
            raise KeyError(f"weights for inputs that are not in the composite: {unknown}")
        w = np.array([float(weights.get(c, 1.0)) for c in columns])
    else:
        w = np.asarray(list(weights), dtype=np.float64)
        if w.shape != (len(columns),):
            raise ValueError(f"{len(w)} weights for {len(columns)} inputs")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"weights must be finite, non-negative and not all zero: {w.tolist()}")
    return w


def _method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; methods: {list(METHODS)}")
    return method


def _combine(rank: np.ndarray, n: int, weights: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Composite (and Borda points) from the ranks ``[..., inputs]`` of ``n`` models.

    The weighted rank sum is exact for average ranks (multiples of 1/2) and integer weights, and
    the composite is that sum divided once, so that equal rank sums give equal composites.
    """
    shape = rank.shape[:-1]
    if n < 2:
        zero = np.zeros(shape)
        return zero, (zero.copy() if method == "borda" else None)
    scale = (n - 1) * float(weights.sum())
    if method == "borda":
        points = (n - rank) @ weights
        return (scale - points) / scale, points
    return ((rank - 1.0) @ weights) / scale, None


def composite_score(table: pd.DataFrame, columns: Sequence[str] | str = INPUTS,
                    weights: Mapping[str, float] | Sequence[float] | None = None,
                    method: str = "mean_normalised_rank") -> pd.DataFrame:
    """Ranks, normalised ranks and composite of the models (rows) of ``table``, best first.

    ``columns``: the inputs (a sequence of columns of ``table`` or the name of a variant of
    :data:`VARIANTS`; default v1); ``weights``: one per input (a sequence aligned with
    ``columns`` or a mapping, unnamed inputs 1; default equal weights); ``method``:
    ``"mean_normalised_rank"`` or ``"borda"`` (module docstring). Adds ``rank_<input>``,
    ``nrank_<input>`` (0 for a single model), ``borda`` (Borda method only) and ``composite`` to
    the columns of ``table`` (replacing earlier ones); the order of equal composites is that of
    ``table``. A missing input ranks last.
    """
    columns = _columns(columns)
    w = _weights(columns, weights)
    method = _method(method)
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise KeyError(f"composite inputs missing: {missing}")
    n = len(table)
    rank = table[list(columns)].astype(np.float64).rank(method="average", na_option="bottom", ascending=True)
    nrank = (rank - 1.0) / (n - 1) if n > 1 else rank * 0.0
    composite, borda = _combine(rank.to_numpy(dtype=np.float64), n, w, method)
    names = tuple(dict.fromkeys((*columns, *_KNOWN)))
    derived = [f"{kind}_{col}" for kind in ("rank", "nrank") for col in names] + ["borda", "composite"]
    parts = [table.drop(columns=derived, errors="ignore"), rank.add_prefix("rank_"), nrank.add_prefix("nrank_")]
    if borda is not None:
        parts.append(pd.Series(borda, index=table.index, name="borda"))
    parts.append(pd.Series(composite, index=table.index, name="composite"))
    return pd.concat(parts, axis=1).sort_values("composite", kind="stable")


def composite_matrix(inputs: Mapping[str, np.ndarray], columns: Sequence[str] | str = INPUTS,
                     weights: Mapping[str, float] | Sequence[float] | None = None,
                     method: str = "mean_normalised_rank") -> np.ndarray:
    """Composite of the models in every row (replicate) of ``inputs[column]`` (``[rows, models]``).

    Row ``r`` gives the composite of :func:`composite_score` on the table ``{column:
    inputs[column][r]}`` (same ranks, same arithmetic), for all rows at once.
    """
    columns = _columns(columns)
    w = _weights(columns, weights)
    method = _method(method)
    missing = [col for col in columns if col not in inputs]
    if missing:
        raise KeyError(f"composite inputs missing: {missing}")
    matrices = [np.atleast_2d(np.asarray(inputs[col], dtype=np.float64)) for col in columns]
    if len({m.shape for m in matrices}) != 1:
        raise ValueError("every input needs the same shape [rows, models]")
    ranks = np.stack([pd.DataFrame(m).rank(axis=1, method="average", na_option="bottom").to_numpy(dtype=np.float64)
                      for m in matrices], axis=-1)
    return _combine(ranks, matrices[0].shape[1], w, method)[0]


def pareto_front(x, y) -> np.ndarray:
    """Members of the Pareto front of two lower-is-better objectives.

    A point is on the front when no other point is at least as good in both objectives and
    strictly better in one (equal points are both on it). Points with a non-finite value are
    neither on the front nor dominate another.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be one-dimensional and of the same length")
    valid = np.isfinite(x) & np.isfinite(y)
    front = np.zeros(x.shape, dtype=bool)
    for i in np.flatnonzero(valid):
        dominated = valid & (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        front[i] = not dominated.any()
    return front


def _aligned(
    frames: Mapping[str, pd.DataFrame], key: str, columns: Sequence[str], what: str
) -> dict[str, np.ndarray]:
    """``{column: [models, items]}`` with the items (profiles, speeds) in a common order."""
    reference, first = None, None
    blocks: dict[str, list[np.ndarray]] = {col: [] for col in columns}
    for model, frame in frames.items():
        missing = [col for col in (key, *columns) if col not in frame.columns]
        if missing:
            raise KeyError(f"{what}[{model!r}] lacks the columns {missing}")
        if frame[key].duplicated().any():
            raise ValueError(f"{what}[{model!r}] repeats {key} values")
        labels = pd.Index(frame[key]).sort_values()
        if reference is None:
            reference, first = labels, model
        elif not labels.equals(reference):
            raise ValueError(f"{what}[{model!r}]: the {key} values differ from those of {first!r}")
        indexed = frame.set_index(key).reindex(reference)
        for col in columns:
            blocks[col].append(indexed[col].to_numpy(dtype=np.float64, na_value=np.nan))
    return {col: np.stack(rows) for col, rows in blocks.items()}


def _mean_over(counts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Count-weighted mean ``[replicates, models]`` of ``values [models, items]``, NaN left out."""
    valid = ~np.isnan(values)
    total = counts @ np.where(valid, values, 0.0).T
    weight = counts @ valid.T.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(weight > 0, total / weight, np.nan)


def _sum_over(counts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Count-weighted sum; NaN when a drawn item is NaN."""
    missing = np.isnan(values)
    total = counts @ np.where(missing, 0.0, values).T
    total[(counts @ missing.T.astype(np.float64)) > 0] = np.nan
    return total


def _inputs(profile_counts, speed_counts, profile, speed) -> dict[str, np.ndarray]:
    return {
        "concave_growth_error": _mean_over(profile_counts, profile["concave_growth_error"]),
        "hysteresis_error": _mean_over(profile_counts, profile["hysteresis_error"]),
        "unstable_fraction": _mean_over(speed_counts, speed["unstable"]),
        "collisions_100": _sum_over(profile_counts, profile["collisions_100"]),
    }


def _driver_replicates(per_event: Mapping[str, pd.DataFrame], models: Sequence[str], group: str,
                       rng: np.random.Generator, n_boot: int) -> tuple[np.ndarray, np.ndarray]:
    """Whole-event spacing RMSE ``[n_boot, models]`` over one draw of the drivers and its point value
    ``[models]``: the counts are drawn over the sorted union of the groups of every table and applied
    to each (common random numbers; the fast pooled path of :mod:`follownet2.stats.bootstrap`)."""
    missing = [m for m in models if m not in per_event]
    if missing:
        raise ValueError(f"per_event lacks the models {missing}")
    tables = []
    for model in models:
        table = per_event[model]
        absent = [c for c in (group, "n", "sse_s") if c not in table.columns]
        if absent:
            raise KeyError(f"per_event[{model!r}] lacks the columns {absent}")
        if not len(table):
            raise ValueError(f"per_event[{model!r}] has no rows")
        tables.append(table.assign(**{group: table[group].astype(str)}))
    universe = pd.Index(pd.concat([t[group] for t in tables], ignore_index=True).unique()).sort_values()
    counts = draw_counts(rng, len(universe), n_boot)
    samplers = [_Resampler(t, group, None, ["sse_s"], universe) for t in tables]
    replicates = np.column_stack([s.apply(counts)["rmse_s"].to_numpy(dtype=np.float64) for s in samplers])
    point = np.array([s.apply(np.ones((1, len(universe))))["rmse_s"].iloc[0] for s in samplers], dtype=np.float64)
    return replicates, point


def composite_bootstrap(
    per_profile: Mapping[str, pd.DataFrame],
    per_speed: Mapping[str, pd.DataFrame],
    n_boot: int = 1000,
    seed: int | np.random.Generator = 0,
    columns: Sequence[str] | str = INPUTS,
    per_event: Mapping[str, pd.DataFrame] | None = None,
    fixed: Mapping[str, Mapping[str, float]] | None = None,
    weights: Mapping[str, float] | Sequence[float] | None = None,
    method: str = "mean_normalised_rank",
    group: str = "follower_id",
) -> pd.DataFrame:
    """Rank distribution of a composite over bootstrap replicates of profiles, speeds and drivers.

    ``per_profile[model]``: one row per P2 profile with ``profile, concave_growth_error,
    hysteresis_error, collisions_100`` (the errors NaN for the synthetic pulse: the mean over
    the profiles leaves NaN out, the collision sum does not). ``per_speed[model]``: one row per
    P3 speed with ``v_e, unstable`` (NaN without an equilibrium). All models need the same
    profiles and speeds. ``columns``, ``weights`` and ``method`` choose the composite as in
    :func:`composite_score` (default: v1).

    Inputs beyond v1: ``p1_whole_rmse_s`` needs ``per_event[model]``, the per-event closed-loop
    table of the whole events (``test_closed_loop.events.parquet``: ``follower_id, n, sse_s``),
    whose drivers are resampled; inputs without a resampling unit are passed in ``fixed`` as
    ``{input: {model: value}}`` and held at that value in every replicate. ``fd_error_rel`` is
    such an input: P2 pools the samples of every profile into the bins of the fundamental diagram
    before it compares them with the training data, and the per-profile tables do not keep the bin
    sums, so the pooled error has no per-profile version to resample (the ``fd_error_rel`` column of
    ``p2.profiles.parquet`` is the error of one profile alone, a different statistic); its
    uncertainty is left out. ``fixed`` overrides the resampled value of any input it names.

    One Generator (``seed``) draws the profile counts, then the speed counts (one draw for all
    models; the v1 replicates do not depend on the other arguments), then, when
    ``p1_whole_rmse_s`` is an input, the counts of the drivers over the sorted union of the
    ``group`` values of every ``per_event`` table, applied to each table (common random numbers).
    Every replicate recomputes the inputs and the composite (:func:`composite_matrix`).

    Returns one row per model, best mean rank first: the columns of
    :func:`~follownet2.stats.ranks.rank_distribution` of the composite, ``composite`` (on all
    profiles, speeds and drivers), ``composite_mean``, ``composite_low`` and ``composite_high``
    (95 % percentile interval).
    """
    columns = _columns(columns)
    w = _weights(columns, weights)
    method = _method(method)
    models = list(per_profile)
    if not models:
        raise ValueError("no model")
    if set(per_speed) != set(models):
        raise ValueError("per_profile and per_speed must hold the same models")
    fixed = dict(fixed or {})
    for name, values in fixed.items():
        absent = [m for m in models if m not in values]
        if absent:
            raise ValueError(f"fixed[{name!r}] lacks the models {absent}")
    unknown = [c for c in columns if c not in fixed and c not in (*INPUTS, P1_INPUT)]
    if unknown:
        raise KeyError(f"no source for the inputs {unknown}: pass their values in `fixed` "
                       f"(fd_error_rel has no per-profile version to resample)")
    profile = _aligned(per_profile, "profile", PROFILE_COLUMNS, "per_profile")
    speed = _aligned({m: per_speed[m] for m in models}, "v_e", ("unstable",), "per_speed")
    n_profiles, n_speeds = profile["collisions_100"].shape[1], speed["unstable"].shape[1]

    rng = np.random.default_rng(seed)
    profile_counts = draw_counts(rng, n_profiles, n_boot).astype(np.float64)
    speed_counts = draw_counts(rng, n_speeds, n_boot).astype(np.float64)
    replicate = _inputs(profile_counts, speed_counts, profile, speed)
    point = {k: v[0] for k, v in _inputs(np.ones((1, n_profiles)), np.ones((1, n_speeds)), profile, speed).items()}
    if P1_INPUT in columns and P1_INPUT not in fixed:
        if per_event is None:
            raise ValueError("p1_whole_rmse_s needs the per-event tables `per_event`")
        replicate[P1_INPUT], point[P1_INPUT] = _driver_replicates(per_event, models, group, rng, n_boot)
    for name, values in fixed.items():
        held = np.array([float(values[m]) for m in models], dtype=np.float64)
        replicate[name], point[name] = np.tile(held, (n_boot, 1)), held

    matrix = composite_matrix({c: replicate[c] for c in columns}, columns, w, method)
    point_composite = composite_matrix({c: point[c][None, :] for c in columns}, columns, w, method)[0]

    samples = {model: matrix[:, i] for i, model in enumerate(models)}
    out = rank_distribution(samples, lower_is_better=True)
    spread = summarise(pd.DataFrame(samples))
    out["composite"] = pd.Series(point_composite, index=models).reindex(out.index)
    for name in ("mean", "low", "high"):
        out[f"composite_{name}"] = spread[name].reindex(out.index)
    return out
