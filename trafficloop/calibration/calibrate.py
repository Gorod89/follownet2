"""Calibration of the mechanistic models (protocol of Punzo, Zheng, Montanino 2021).

* Measure of performance: spacing and speed of a closed-loop rollout over the whole event
  (leader given, initial state observed), never the acceleration.
* Goodness of fit: ``J = NRMSE(s) + NRMSE(v)`` with
  ``NRMSE(y) = sqrt(mean((y_sim - y_obs)^2)) / sqrt(mean(y_obs^2))``.
* A rollout with a collision gets ``J = 10 + share of samples with s <= 0``, so that any
  collision is worse than any collision-free candidate while candidates stay ranked.
* Optimiser: batched differential evolution (:mod:`trafficloop.calibration.de`).

Two calibrations are produced: per event (one parameter vector per event; gives the
parameter spread and the best accuracy the model form can reach) and global (one vector
for a set of events, pooled errors plus ``10 * collision rate``; this is the model that is
scored in the benchmark, calibrated on the training part of a split only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch

from trafficloop.calibration.de import DEResult, differential_evolution
from trafficloop.data.batch import WARMUP, EventBatch, length_buckets
from trafficloop.data.schema import Event
from trafficloop.models.base import ACC_CLIP, MechanisticModel
from trafficloop.sim.rollout import RolloutErrors, rollout, rollout_errors

COLLISION_PENALTY = 10.0


@dataclass
class CalibrationConfig:
    popsize: int = 15
    maxiter: int = 200
    tol: float = 0.01
    seed: int = 0
    warmup: int = WARMUP
    clip: tuple[float, float] = ACC_CLIP
    batch_events: int = 512  # events per batched problem set (per-event calibration)
    max_rollouts: int = 200_000  # candidate x event rollouts evaluated at once (global calibration)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)  # overrides of the model defaults

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)


def model_bounds(model_cls: type[MechanisticModel], cfg: CalibrationConfig) -> torch.Tensor:
    pairs = [tuple(cfg.bounds.get(name, default)) for name, default in zip(model_cls.param_names, model_cls.bounds)]
    return torch.as_tensor(pairs, dtype=cfg.torch_dtype, device=cfg.device)


def _nrmse(sse: torch.Tensor, ss: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(sse / ss.clamp_min(1e-12))


def event_objective(err: RolloutErrors) -> torch.Tensor:
    """``J`` of every (candidate, event), shape ``[*lead, E]``."""
    fit = _nrmse(err.sse_s, err.ss_s) + _nrmse(err.sse_v, err.ss_v)
    crash = COLLISION_PENALTY + err.n_overlap / err.n.clamp_min(1)
    return torch.where(err.collided, crash.to(fit.dtype), fit)


def _event_table(events: Sequence[Event], names: Sequence[str], x: np.ndarray, err: RolloutErrors, de: DEResult | None):
    n = err.n.clamp_min(1).double()
    table = pd.DataFrame(
        {
            "event_id": [ev.event_id for ev in events],
            "follower_id": [ev.follower_id for ev in events],
            "dataset": [ev.dataset for ev in events],
            "site": [ev.site for ev in events],
            **{name: x[:, i] for i, name in enumerate(names)},
            "objective": event_objective(err).double().cpu().numpy(),
            "nrmse_s": _nrmse(err.sse_s, err.ss_s).double().cpu().numpy(),
            "nrmse_v": _nrmse(err.sse_v, err.ss_v).double().cpu().numpy(),
            "rmse_s": torch.sqrt(err.sse_s.double() / n).cpu().numpy(),
            "rmse_v": torch.sqrt(err.sse_v.double() / n).cpu().numpy(),
            "collided": err.collided.cpu().numpy(),
            "n_samples": err.n.cpu().numpy(),
        }
    )
    if de is not None:
        table["n_generations"] = de.n_generations.cpu().numpy()
        table["converged"] = de.converged.cpu().numpy()
    return table


def calibrate_per_event(
    model_cls: type[MechanisticModel], events: Sequence[Event], cfg: CalibrationConfig | None = None
) -> pd.DataFrame:
    """One parameter vector per event. Returns one row per event, in the input order."""
    cfg = cfg or CalibrationConfig()
    events = list(events)
    model = model_cls()
    bounds = model_bounds(model_cls, cfg)
    tables = []
    buckets = length_buckets([len(ev) for ev in events], cfg.batch_events)
    for b, index in enumerate(buckets):
        chunk = [events[i] for i in index]
        batch = EventBatch.from_events(chunk, device=cfg.device, dtype=cfg.torch_dtype)

        def objective(theta: torch.Tensor) -> torch.Tensor:  # [E, P, D] -> [E, P]
            result = rollout(model, batch, cfg.warmup, cfg.clip, params=theta.transpose(0, 1))
            return event_objective(rollout_errors(result, batch)).transpose(0, 1)

        de = differential_evolution(
            objective, bounds, n_problems=len(chunk), popsize=cfg.popsize, maxiter=cfg.maxiter, tol=cfg.tol,
            seed=cfg.seed + b,
        )
        final = rollout(model, batch, cfg.warmup, cfg.clip, params=de.x.unsqueeze(0))
        err = rollout_errors(final, batch)
        err = RolloutErrors(*(getattr(err, f).reshape(-1) for f in RolloutErrors.__dataclass_fields__))
        table = _event_table(chunk, model_cls.param_names, de.x.double().cpu().numpy(), err, de)
        table.index = index
        tables.append(table)
    return pd.concat(tables).sort_index().reset_index(drop=True)


def pooled_objective(parts: list[RolloutErrors]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pooled ``J`` over events for every candidate, shape ``[*lead]``."""
    total = lambda name: sum(getattr(p, name).double().sum(-1) for p in parts)  # noqa: E731
    n, n_events = total("n"), sum(p.collided.shape[-1] for p in parts)
    nrmse_s = torch.sqrt(total("sse_s") / total("ss_s"))
    nrmse_v = torch.sqrt(total("sse_v") / total("ss_v"))
    rate = sum(p.collided.double().sum(-1) for p in parts) / n_events
    stats = {
        "nrmse_s": nrmse_s,
        "nrmse_v": nrmse_v,
        "rmse_s": torch.sqrt(total("sse_s") / n),
        "rmse_v": torch.sqrt(total("sse_v") / n),
        "collision_rate": rate,
    }
    return nrmse_s + nrmse_v + COLLISION_PENALTY * rate, stats


def evaluate_params(
    model_cls: type[MechanisticModel], events: Sequence[Event], params: torch.Tensor, cfg: CalibrationConfig
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[RolloutErrors], list[np.ndarray]]:
    """Pooled objective of candidates ``params[P, D]`` on a set of events."""
    events = list(events)
    model = model_cls()
    params = params.to(device=cfg.device, dtype=cfg.torch_dtype)
    per_batch = max(1, cfg.max_rollouts // max(1, params.shape[0]))
    buckets = length_buckets([len(ev) for ev in events], per_batch)
    parts = []
    for index in buckets:
        batch = EventBatch.from_events([events[i] for i in index], device=cfg.device, dtype=cfg.torch_dtype)
        result = rollout(model, batch, cfg.warmup, cfg.clip, params=params.unsqueeze(1))
        parts.append(rollout_errors(result, batch))
    objective, stats = pooled_objective(parts)
    return objective, stats, parts, buckets


def calibrate_global(
    model_cls: type[MechanisticModel],
    events: Sequence[Event],
    cfg: CalibrationConfig | None = None,
    penalty: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict:
    """One parameter vector for all events (pooled errors).

    ``penalty`` maps candidates ``[P, D]`` to a term ``[P]`` added to the objective (the string-stability
    margin of a certified core, :func:`trafficloop.stability.certificate.stability_shortfall`); the result
    then reports ``objective`` with the term and ``fit_objective`` without it.
    """
    cfg = cfg or CalibrationConfig()
    events = list(events)
    bounds = model_bounds(model_cls, cfg)

    def objective(theta: torch.Tensor) -> torch.Tensor:  # [1, P, D] -> [1, P]
        value = evaluate_params(model_cls, events, theta[0], cfg)[0].to(theta.dtype)
        if penalty is not None:
            value = value + penalty(theta[0]).to(value)
        return value.unsqueeze(0)

    x0 = torch.as_tensor(model_cls.default, dtype=cfg.torch_dtype, device=cfg.device)
    de = differential_evolution(
        objective, bounds, n_problems=1, popsize=cfg.popsize, maxiter=cfg.maxiter, tol=cfg.tol, seed=cfg.seed, x0=x0
    )
    value, stats, _, _ = evaluate_params(model_cls, events, de.x, cfg)
    out = {
        "model": model_cls.name,
        "params": {name: float(v) for name, v in zip(model_cls.param_names, de.x[0].double().cpu())},
        "objective": float(value[0]),
        **{name: float(v[0]) for name, v in stats.items()},
        "n_events": len(events),
        "n_generations": int(de.n_generations[0]),
        "converged": bool(de.converged[0]),
    }
    if penalty is not None:
        out["fit_objective"] = out["objective"]
        out["objective"] = float(value[0] + penalty(de.x)[0].to(value))
    return out


def score_events(
    model_cls: type[MechanisticModel], events: Sequence[Event], params: Sequence[float], cfg: CalibrationConfig | None = None
) -> pd.DataFrame:
    """Closed-loop errors of one parameter vector, one row per event (input order)."""
    cfg = cfg or CalibrationConfig()
    events = list(events)
    theta = torch.as_tensor(params, dtype=cfg.torch_dtype, device=cfg.device).reshape(1, -1)
    _, _, parts, buckets = evaluate_params(model_cls, events, theta, cfg)
    tables = []
    for err, index in zip(parts, buckets):
        err = RolloutErrors(*(getattr(err, f).reshape(-1) for f in RolloutErrors.__dataclass_fields__))
        x = np.tile(np.asarray(params, dtype=np.float64), (len(index), 1))
        table = _event_table([events[i] for i in index], model_cls.param_names, x, err, None)
        table.index = index
        tables.append(table)
    return pd.concat(tables).sort_index().reset_index(drop=True)


def parameter_spread(table: pd.DataFrame, model_cls: type[MechanisticModel], cfg: CalibrationConfig | None = None) -> dict:
    """Spread of the per-event estimates: moments, quantiles, share at a bound, correlations.

    Events whose best candidate still collides are excluded and counted.
    """
    cfg = cfg or CalibrationConfig()
    names = list(model_cls.param_names)
    ok = table[~table["collided"]]
    bounds = model_bounds(model_cls, cfg).double().cpu().numpy()
    out: dict = {
        "model": model_cls.name,
        "n_events": int(len(table)),
        "n_used": int(len(ok)),
        "n_collided": int(table["collided"].sum()),
        "share_converged": float(table["converged"].mean()) if "converged" in table else None,
        "objective": _describe(ok["objective"].to_numpy()),
        "rmse_s": _describe(ok["rmse_s"].to_numpy()),
        "rmse_v": _describe(ok["rmse_v"].to_numpy()),
        "parameters": {},
    }
    for i, name in enumerate(names):
        x = ok[name].to_numpy(dtype=np.float64)
        low, high = bounds[i]
        edge = 1e-3 * (high - low)
        stats = _describe(x)
        stats["cv"] = float(stats["std"] / abs(stats["mean"])) if len(x) and stats["mean"] else None
        stats["share_at_lower"] = float(np.mean(x <= low + edge)) if len(x) else None
        stats["share_at_upper"] = float(np.mean(x >= high - edge)) if len(x) else None
        stats["bounds"] = [float(low), float(high)]
        out["parameters"][name] = stats
    if len(ok) > 2:
        corr = np.corrcoef(ok[names].to_numpy(dtype=np.float64), rowvar=False)
        out["correlation"] = {"names": names, "matrix": np.nan_to_num(corr).round(4).tolist()}
    return out


def _describe(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"mean": None, "std": None, "median": None, "q05": None, "q25": None, "q75": None, "q95": None}
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(q[2]),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "q75": float(q[3]),
        "q95": float(q[4]),
    }
