"""Physics-informed reconstruction of car-following events (data version "reconstructed").

Every vehicle trajectory is the solution of a convex quadratic programme whose feasible set
is the set of trajectories that the benchmark integrator can produce:

    minimise   w_x * sum (x[k] - x_obs[k])^2 + w_v * sum (v[k] - v_obs[k])^2
               + w_j * sum ((a[k+1] - a[k]) / dt)^2
               (the two data terms with the weights of a Huber loss: outlier step)
    subject to v[k+1] = v[k] + a[k] * dt,   x[k+1] = x[k] + v[k+1] * dt     (semi-implicit Euler)
               v[k] >= 0,   a_min <= a[k] <= a_max,   |a[k+1] - a[k]| <= j_max * dt
               x[k] <= x_lead[k] - s_min                                     (follower only)

so positions, speeds and accelerations are mutually consistent by construction, the
acceleration and the jerk are bounded, and the gap stays positive. The leader is
reconstructed first, the follower second with the gap constraint against the reconstructed
leader. The jerk weight follows from a cut-off frequency ``f_c``: without active constraints
the programme is a linear low-pass filter with gain 1/2 at ``f_c``,
``w_j = (w_x + w_v * wc^2) / wc^6`` with ``wc = 2 * pi * f_c``.
See docs/reconstruction.md.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from follownet2.data.schema import DT, Event

METHOD = "qp-semi-implicit-v1"
log = logging.getLogger(__name__)


@dataclass
class ReconstructionConfig:
    sigma_x: float | None = 0.5  # m, position noise (None: positions not used)
    sigma_v: float | None = None  # m/s, speed noise (None: speeds not used)
    cutoff_hz: float = 0.5
    a_min: float = -8.0
    a_max: float = 4.0
    jerk_max: float = 15.0
    s_min: float = 0.3  # m, smallest admissible net gap
    # outlier step: samples whose residual exceeds huber_c standard deviations get the weight
    # of a Huber loss, those beyond twice that are left out, and the programme is solved again
    huber_c: float = 2.5
    robust_iterations: int = 3
    dt: float = DT
    eps: float = 1e-6
    max_iter: int = 20000

    def __post_init__(self) -> None:
        if self.sigma_x is None and self.sigma_v is None:
            raise ValueError("at least one of sigma_x and sigma_v must be given")

    def weights(self) -> tuple[float, float, float]:
        w_x = 0.0 if self.sigma_x is None else 1.0 / self.sigma_x**2
        w_v = 0.0 if self.sigma_v is None else 1.0 / self.sigma_v**2
        wc = 2.0 * math.pi * self.cutoff_hz
        return w_x, w_v, (w_x + w_v * wc**2) / wc**6


def _difference(n: int) -> sp.csc_matrix:
    """First difference, shape ``(n - 1, n)``."""
    return sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csc")


def _solve(P, q, A, low, high, cfg: ReconstructionConfig) -> tuple[np.ndarray, str]:
    import osqp

    solver = osqp.OSQP()
    common = dict(verbose=False, eps_abs=cfg.eps, eps_rel=cfg.eps, max_iter=cfg.max_iter)
    try:  # osqp >= 1.0
        solver.setup(P, q, A, low, high, polishing=True, **common)
    except TypeError:  # osqp 0.6
        solver.setup(P, q, A, low, high, polish=True, **common)
    result = solver.solve()
    return np.asarray(result.x, dtype=np.float64), str(result.info.status)


def integrate(x0: float, v0: float, a: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-implicit Euler from ``(x0, v0)``; returns ``x, v`` and the realised acceleration."""
    n = len(a) + 1
    x, v = np.empty(n), np.empty(n)
    x[0], v[0] = x0, max(v0, 0.0)
    for k in range(n - 1):
        v[k + 1] = max(v[k] + a[k] * dt, 0.0)
        x[k + 1] = x[k] + v[k + 1] * dt
    return x, v, np.diff(v) / dt


def huber_weights(residual: np.ndarray, sigma: float | None, c: float) -> np.ndarray:
    """Weight 1 inside ``c * sigma``, the Huber weight ``c * sigma / |residual|`` up to
    ``2 c * sigma`` and 0 beyond: a block of tracking errors must not pull at all."""
    if sigma is None:
        return np.ones(len(residual))
    size = np.abs(residual) / sigma
    return np.where(size <= c, 1.0, np.where(size <= 2.0 * c, c / np.maximum(size, 1e-12), 0.0))


def reconstruct_trajectory(
    x_obs: np.ndarray,
    v_obs: np.ndarray,
    cfg: ReconstructionConfig,
    x_upper: np.ndarray | None = None,
) -> dict:
    """Reconstruct one trajectory; ``x_upper`` is the pointwise upper bound of the position.

    Returns ``x, v, a`` (``a`` has the length of ``x``, the last value repeats the one
    before), the solver status and the share of samples treated as outliers.
    """
    n = len(x_obs)
    dt = cfg.dt
    w_x, w_v, w_j = cfg.weights()
    origin = float(x_obs[0])
    x_rel = np.asarray(x_obs, dtype=np.float64) - origin
    v_obs = np.asarray(v_obs, dtype=np.float64)
    d_n, d_a = _difference(n), _difference(n - 1)
    shift = sp.eye(n, format="csc")[1:, :]  # selects v[k+1]
    zero = sp.csc_matrix
    jerk = 2.0 * w_j / dt**2 * (d_a.T @ d_a)

    rows = [
        sp.hstack([zero((n - 1, n)), d_n, -dt * sp.eye(n - 1)]),  # v[k+1] - v[k] - dt a[k] = 0
        sp.hstack([d_n, -dt * shift, zero((n - 1, n - 1))]),  # x[k+1] - x[k] - dt v[k+1] = 0
        sp.hstack([zero((n, n)), sp.eye(n), zero((n, n - 1))]),  # v >= 0
        sp.hstack([zero((n - 1, 2 * n)), sp.eye(n - 1)]),  # acceleration bounds
        sp.hstack([zero((n - 2, 2 * n)), d_a]),  # jerk bounds
    ]
    low = [np.zeros(2 * (n - 1)), np.zeros(n), np.full(n - 1, cfg.a_min), np.full(n - 2, -cfg.jerk_max * dt)]
    high = [np.zeros(2 * (n - 1)), np.full(n, np.inf), np.full(n - 1, cfg.a_max), np.full(n - 2, cfg.jerk_max * dt)]
    if x_upper is not None:
        rows.append(sp.hstack([sp.eye(n), zero((n, 2 * n - 1))]))
        low.append(np.full(n, -np.inf))
        high.append(np.asarray(x_upper, dtype=np.float64) - origin)
    A, low, high = sp.vstack(rows, format="csc"), np.concatenate(low), np.concatenate(high)

    weight_x, weight_v = np.ones(n), np.ones(n)
    for iteration in range(1 + max(cfg.robust_iterations, 0)):
        P = sp.block_diag([2.0 * w_x * sp.diags(weight_x), 2.0 * w_v * sp.diags(weight_v), jerk], format="csc")
        q = np.concatenate([-2.0 * w_x * weight_x * x_rel, -2.0 * w_v * weight_v * v_obs, np.zeros(n - 1)])
        z, status = _solve(P, q, A, low, high, cfg)
        if not np.all(np.isfinite(z)):
            return {"status": status, "ok": False}
        if iteration == cfg.robust_iterations:
            break
        new_x = huber_weights(z[:n] - x_rel, cfg.sigma_x, cfg.huber_c)
        new_v = huber_weights(z[n : 2 * n] - v_obs, cfg.sigma_v, cfg.huber_c)
        if np.allclose(new_x, weight_x, atol=1e-3) and np.allclose(new_v, weight_v, atol=1e-3):
            break
        weight_x, weight_v = new_x, new_v
    a = np.clip(z[2 * n :], cfg.a_min, cfg.a_max)
    # exact consistency: re-integrate the solver output with the benchmark scheme
    x, v, a = integrate(z[0], z[n], a, dt)
    return {
        "x": x + origin,
        "v": v,
        "a": np.append(a, a[-1]),
        "status": status,
        "ok": status in ("solved", "solved inaccurate", "solved_inaccurate"),
        "outliers": float(np.mean((weight_x < 1.0) | (weight_v < 1.0))),
    }


def reconstruct_event(event: Event, cfg: ReconstructionConfig | None = None) -> Event | None:
    """Reconstructed version of a raw event, or ``None`` when the solver fails.

    Sources that measure the gap directly (``meta["position_source"] == "gap"``: OpenACC,
    FollowNet) have no independent follower position, so the follower is fitted to
    ``x_lead_reconstructed - s_obs``; track-based sources are fitted to their own positions.
    """
    cfg = cfg or ReconstructionConfig()
    lead = reconstruct_trajectory(event.x_lead, event.v_lead, cfg)
    if not lead["ok"]:
        return None
    gap_based = event.meta.get("position_source", "track") == "gap"
    target = lead["x"] - event.s if gap_based else event.x_follower
    follower = reconstruct_trajectory(target, event.v, cfg, x_upper=lead["x"] - cfg.s_min)
    if not follower["ok"]:
        return None
    s = lead["x"] - follower["x"]
    short = cfg.s_min - float(np.min(s))
    if short > 0.0:
        # the solver meets the gap bound to its tolerance only: a rigid shift makes it exact
        follower["x"] = follower["x"] - short
        s = lead["x"] - follower["x"]
    jerk_raw = np.diff(event.a) / cfg.dt
    jerk = np.diff(follower["a"][:-1]) / cfg.dt
    meta = {
        **event.meta,
        "data_version": "reconstructed",
        "a_source": "reconstruction",
        "speeds": "reconstruction",
        "reconstruction": {
            "method": METHOD,
            **{k: v for k, v in asdict(cfg).items() if k not in ("eps", "max_iter", "dt")},
            "status": [lead["status"], follower["status"]],
            "outlier_share": [lead["outliers"], follower["outliers"]],
            "rms_gap_change": float(np.sqrt(np.mean((s - event.s) ** 2))),
            "rms_speed_change": float(np.sqrt(np.mean((follower["v"] - event.v) ** 2))),
            "rms_acc_change": float(np.sqrt(np.mean((follower["a"] - event.a) ** 2))),
            "max_abs_acc_raw": float(np.max(np.abs(event.a))),
            "max_abs_acc": float(np.max(np.abs(follower["a"]))),
            "max_abs_jerk_raw": float(np.max(np.abs(jerk_raw))),
            "max_abs_jerk": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
            "min_gap_raw": float(np.min(event.s)),
            "min_gap": float(np.min(s)),
            "gap_bound_active": bool(np.min(s) <= cfg.s_min + 1e-3),
        },
    }
    return Event(
        event_id=event.event_id,
        dataset=event.dataset,
        site=event.site,
        follower_id=event.follower_id,
        leader_id=event.leader_id,
        t=event.t.copy(),
        s=s,
        dv=follower["v"] - lead["v"],
        v=follower["v"],
        a=follower["a"],
        v_lead=lead["v"],
        x_lead=lead["x"],
        x_follower=follower["x"],
        meta=meta,
    )


def _work(args: tuple[Event, ReconstructionConfig]) -> Event | None:
    return reconstruct_event(*args)


def reconstruct_events(
    events: Iterable[Event], cfg: ReconstructionConfig | None = None, workers: int = 1, chunksize: int = 16
) -> tuple[list[Event], list[str]]:
    """Reconstruct many events; returns the reconstructed events and the ids that failed."""
    cfg = cfg or ReconstructionConfig()
    events = list(events)
    jobs = [(ev, cfg) for ev in events]
    out: list[Event | None] = []
    if workers > 1 and len(events) > chunksize:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(_work, jobs, chunksize=chunksize):
                    out.append(result)
        except BrokenProcessPool:
            # a worker was killed (the machine ran out of memory): the rest is done here
            log.warning("process pool broke after %d of %d events, continuing serially", len(out), len(jobs))
    out.extend(_work(job) for job in jobs[len(out):])
    done = [ev for ev in out if ev is not None]
    failed = [ev.event_id for ev, res in zip(events, out) if res is None]
    return done, failed


def consistency_residual(event: Event, dt: float = DT) -> dict[str, float]:
    """How far an event is from a trajectory of the benchmark integrator (0 when reconstructed)."""
    dv = event.v[1:] - event.v[:-1] - event.a[:-1] * dt
    dx = event.x_follower[1:] - event.x_follower[:-1] - event.v[1:] * dt
    return {"rms_speed_residual": float(np.sqrt(np.mean(dv**2))), "rms_position_residual": float(np.sqrt(np.mean(dx**2)))}


def summarise(events: Sequence[Event]) -> dict:
    """Aggregate reconstruction diagnostics for the manifest."""
    rec = [ev.meta["reconstruction"] for ev in events if "reconstruction" in ev.meta]
    if not rec:
        return {}
    col = lambda name: np.array([r[name] for r in rec], dtype=np.float64)  # noqa: E731
    return {
        "method": METHOD,
        "n_events": len(rec),
        "share_gap_bound_active": float(np.mean(col("gap_bound_active"))),
        "mean_outlier_share": float(np.mean([np.mean(r.get("outlier_share", [0.0])) for r in rec])),
        "share_solved_inaccurate": float(np.mean([any("inaccurate" in s for s in r["status"]) for r in rec])),
        "median_rms_gap_change": float(np.median(col("rms_gap_change"))),
        "median_rms_speed_change": float(np.median(col("rms_speed_change"))),
        "median_rms_acc_change": float(np.median(col("rms_acc_change"))),
        "q99_max_abs_acc_raw": float(np.quantile(col("max_abs_acc_raw"), 0.99)),
        "q99_max_abs_acc": float(np.quantile(col("max_abs_acc"), 0.99)),
        "q99_max_abs_jerk_raw": float(np.quantile(col("max_abs_jerk_raw"), 0.99)),
        "q99_max_abs_jerk": float(np.quantile(col("max_abs_jerk"), 0.99)),
    }
