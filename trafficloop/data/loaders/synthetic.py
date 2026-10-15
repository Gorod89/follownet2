"""Synthetic events: IDM followers behind a random leader, exact under the benchmark integrator.

Used by the tests and the sanity experiments (known parameters, known stability) and as a
stand-in when no dataset is on disk. With ``sigma_x = sigma_v = 0`` the events satisfy
``v[k+1] = v[k] + a[k]*dt`` and ``x[k+1] = x[k] + v[k+1]*dt`` to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficloop.data.schema import DT, Event

IDM_NOMINAL = (33.3, 1.5, 2.0, 1.0, 1.5)  # v0, T, s0, a, b


@dataclass
class SyntheticConfig:
    n_drivers: int = 40
    events_per_driver: int = 3
    n_sites: int = 2
    duration: tuple[float, float] = (20.0, 60.0)
    speed_range: tuple[float, float] = (3.0, 30.0)
    hold: tuple[float, float] = (5.0, 15.0)  # s, time between changes of the leader's target speed
    relax: float = 0.5  # 1/s, gain of the leader towards its target speed
    leader_acc: tuple[float, float] = (-3.0, 2.0)
    theta: tuple[float, ...] = IDM_NOMINAL
    theta_cv: float = 0.15  # coefficient of variation of the driver parameters (log-normal)
    sigma_x: float = 0.0  # m, position noise
    sigma_v: float = 0.0  # m/s, speed noise
    dataset: str = "synthetic"
    seed: int = 0
    dt: float = DT


def _idm(s, dv, v, theta):
    v0, T, s0, a, b = theta
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * np.sqrt(a * b)))
    return a * (1.0 - (max(v, 0.0) / v0) ** 4 - (s_star / max(s, 0.1)) ** 2)


def leader_profile(n: int, cfg: SyntheticConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    dt = cfg.dt
    v = np.empty(n)
    x = np.empty(n)
    v[0], x[0] = rng.uniform(*cfg.speed_range), 0.0
    target, change = v[0], 0
    for k in range(n - 1):
        if k >= change:
            target = rng.uniform(*cfg.speed_range)
            change = k + int(rng.uniform(*cfg.hold) / dt)
        acc = float(np.clip(cfg.relax * (target - v[k]), *cfg.leader_acc))
        v[k + 1] = max(v[k] + acc * dt, 0.0)
        x[k + 1] = x[k] + v[k + 1] * dt
    return x, v


def follow(x_lead: np.ndarray, v_lead: np.ndarray, theta, s_init: float, v_init: float, dt: float = DT):
    n = len(x_lead)
    x, v, a = np.empty(n), np.empty(n), np.empty(n)
    x[0], v[0] = x_lead[0] - s_init, v_init
    for k in range(n - 1):
        acc = float(np.clip(_idm(x_lead[k] - x[k], v[k] - v_lead[k], v[k], theta), -8.0, 4.0))
        v[k + 1] = max(v[k] + acc * dt, 0.0)
        x[k + 1] = x[k] + v[k + 1] * dt
        a[k] = (v[k + 1] - v[k]) / dt
    a[n - 1] = a[n - 2]
    return x, v, a


def make_events(cfg: SyntheticConfig | None = None, return_truth: bool = False):
    """Events of ``cfg.n_drivers`` drivers; with ``return_truth`` also the noise-free arrays.

    The measurement noise has its own random stream, so the underlying trajectories do not
    depend on the noise level.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    noise = np.random.default_rng([cfg.seed, 1])
    truths = []
    nominal = np.asarray(cfg.theta, dtype=np.float64)
    sigma = np.sqrt(np.log1p(cfg.theta_cv**2))
    events = []
    for d in range(cfg.n_drivers):
        theta = nominal * np.exp(sigma * rng.standard_normal(len(nominal)) - 0.5 * sigma**2)
        site = f"site{d % cfg.n_sites}"
        follower_id = f"{cfg.dataset}/{site}/{d}"
        for e in range(cfg.events_per_driver):
            n = int(rng.uniform(*cfg.duration) / cfg.dt) + 1
            x_lead, v_lead = leader_profile(n, cfg, rng)
            v_init = min(v_lead[0], 0.95 * theta[0])
            gap = (theta[2] + v_init * theta[1]) / np.sqrt(1.0 - (v_init / theta[0]) ** 4)
            x, v, a = follow(x_lead, v_lead, theta, gap * rng.uniform(0.9, 1.3), v_init, cfg.dt)
            truth = {"s": x_lead - x, "v": v, "a": a, "v_lead": v_lead, "x_lead": x_lead, "x_follower": x}
            if np.min(truth["s"]) <= 0.0:
                continue
            if cfg.sigma_x > 0 or cfg.sigma_v > 0:
                x_lead = x_lead + cfg.sigma_x * noise.standard_normal(n)
                x = np.minimum(x + cfg.sigma_x * noise.standard_normal(n), x_lead - 0.05)
                v_lead = np.clip(v_lead + cfg.sigma_v * noise.standard_normal(n), 0.0, None)
                v = np.clip(v + cfg.sigma_v * noise.standard_normal(n), 0.0, None)
                a = np.gradient(v, cfg.dt)
            truths.append(truth)
            leader = f"{cfg.dataset}/{site}/L{d}_{e}"
            events.append(
                Event(
                    event_id=f"{follower_id}|L{d}_{e}|0",
                    dataset=cfg.dataset,
                    site=site,
                    follower_id=follower_id,
                    leader_id=leader,
                    t=np.arange(n) * cfg.dt,
                    s=x_lead - x,
                    dv=v - v_lead,
                    v=v,
                    a=a,
                    v_lead=v_lead,
                    x_lead=x_lead,
                    x_follower=x,
                    meta={
                        "data_version": "raw",
                        "a_source": "simulation" if cfg.sigma_x == 0 and cfg.sigma_v == 0 else "gradient",
                        "position_source": "track",
                        "leader_length": 4.5,
                        "theta_true": [float(p) for p in theta],
                        "group": f"rec{d % cfg.n_sites}",
                        "t0": 100.0 * len(events),
                    },
                )
            )
    return (events, truths) if return_truth else events


POSITION_SOURCE = "track"


def load_events(raw_dir=None, dataset: str = "synthetic", **options) -> list[Event]:
    """Loader interface of the registry; ``options`` are fields of :class:`SyntheticConfig`."""
    return make_events(SyntheticConfig(dataset=dataset, **options))
