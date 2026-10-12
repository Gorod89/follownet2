"""Composite traffic-level score (protocol contract, section 6).

Four inputs per model, lower is better: ``concave_growth_error`` and ``hysteresis_error`` (P2,
mean over the OpenACC profiles), ``unstable_fraction`` (P3, numerical, over the speeds with an
equilibrium) and ``collisions_100`` (P2, summed over all profiles). Each input is ranked within
the compared set (average ranks on ties, a missing value last) and the rank normalised to
``(rank - 1) / (n - 1)``; the composite is the mean of the four normalised ranks (lower is
better). The bootstrap resamples the P2 profiles and the P3 speeds with replacement, the same
draw for all models, and recomputes the composite of every replicate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from follownet2.stats.bootstrap import draw_counts, summarise
from follownet2.stats.ranks import rank_distribution

INPUTS = ("concave_growth_error", "hysteresis_error", "unstable_fraction", "collisions_100")
PROFILE_COLUMNS = ("concave_growth_error", "hysteresis_error", "collisions_100")


def composite_score(table: pd.DataFrame) -> pd.DataFrame:
    """Ranks, normalised ranks and composite of the models (rows) of ``table``, best first.

    Adds ``rank_<input>``, ``nrank_<input>`` (0 for a single model) and ``composite`` to the
    columns of ``table`` (replacing earlier ones); the order of equal composites is that of
    ``table``.
    """
    missing = [col for col in INPUTS if col not in table.columns]
    if missing:
        raise KeyError(f"composite inputs missing: {missing}")
    n = len(table)
    rank = table[list(INPUTS)].astype(np.float64).rank(method="average", na_option="bottom", ascending=True)
    nrank = (rank - 1.0) / (n - 1) if n > 1 else rank * 0.0
    derived = [f"{kind}_{col}" for kind in ("rank", "nrank") for col in INPUTS] + ["composite"]
    out = pd.concat(
        [
            table.drop(columns=derived, errors="ignore"),
            rank.add_prefix("rank_"),
            nrank.add_prefix("nrank_"),
            nrank.mean(axis=1).rename("composite"),
        ],
        axis=1,
    )
    return out.sort_values("composite", kind="stable")


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


def composite_bootstrap(
    per_profile: Mapping[str, pd.DataFrame],
    per_speed: Mapping[str, pd.DataFrame],
    n_boot: int = 1000,
    seed: int | np.random.Generator = 0,
) -> pd.DataFrame:
    """Rank distribution of the composite over bootstrap replicates of profiles and speeds.

    ``per_profile[model]``: one row per P2 profile with ``profile, concave_growth_error,
    hysteresis_error, collisions_100`` (the errors NaN for the synthetic pulse: the mean over
    the profiles leaves NaN out, the collision sum does not). ``per_speed[model]``: one row per
    P3 speed with ``v_e, unstable`` (NaN without an equilibrium). All models need the same
    profiles and speeds. Every replicate draws the profiles and the speeds with replacement
    (one draw for all models), recomputes the four inputs and calls :func:`composite_score`.

    Returns one row per model, best mean rank first: the columns of
    :func:`~follownet2.stats.ranks.rank_distribution` of the composite, ``composite`` (on all
    profiles and speeds), ``composite_mean``, ``composite_low`` and ``composite_high`` (95 %
    percentile interval).
    """
    models = list(per_profile)
    if not models:
        raise ValueError("no model")
    if set(per_speed) != set(models):
        raise ValueError("per_profile and per_speed must hold the same models")
    profile = _aligned(per_profile, "profile", PROFILE_COLUMNS, "per_profile")
    speed = _aligned({m: per_speed[m] for m in models}, "v_e", ("unstable",), "per_speed")
    n_profiles, n_speeds = profile["collisions_100"].shape[1], speed["unstable"].shape[1]

    def composites(inputs: dict[str, np.ndarray], r: int) -> np.ndarray:
        table = pd.DataFrame({col: inputs[col][r] for col in INPUTS}, index=models)
        return composite_score(table)["composite"].reindex(models).to_numpy()

    rng = np.random.default_rng(seed)
    profile_counts = draw_counts(rng, n_profiles, n_boot).astype(np.float64)
    speed_counts = draw_counts(rng, n_speeds, n_boot).astype(np.float64)
    inputs = _inputs(profile_counts, speed_counts, profile, speed)
    matrix = np.stack([composites(inputs, r) for r in range(n_boot)])
    point = composites(_inputs(np.ones((1, n_profiles)), np.ones((1, n_speeds)), profile, speed), 0)

    samples = {model: matrix[:, i] for i, model in enumerate(models)}
    out = rank_distribution(samples, lower_is_better=True)
    spread = summarise(pd.DataFrame(samples))
    out["composite"] = pd.Series(point, index=models).reindex(out.index)
    for name in ("mean", "low", "high"):
        out[f"composite_{name}"] = spread[name].reindex(out.index)
    return out
