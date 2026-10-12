"""Synthetic scenarios: events with a prescribed leader and a placeholder follower.

Protocols P3 and P4 (and the platoon simulator of P2) drive a model behind a prescribed
leader. Every leader series starts with a *hold* phase at its initial speed ``v0``: the first
``WARMUP`` samples are the observed warm-up of the protocols, the rest of the hold lets the
follower settle at its own equilibrium gap, and the manoeuvre starts at the sample
``manoeuvre_start = round(hold / dt)``. The placeholder follower sits at the gap ``s_init``
and copies the leader speed; it provides the warm-up rows ``(s_init, 0, v0, 0)`` and is
never scored (a protocol scores collisions, gaps, speeds and accelerations of the rollout).

The leader position integrates like the data, ``x[k+1] = x[k] + v[k+1] dt`` (semi-implicit
Euler), so that ``EventBatch.from_events`` reproduces the speeds from the displacements.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from follownet2.data.batch import WARMUP
from follownet2.data.schema import DT, Event

#: hold phase before every manoeuvre [s], warm-up included
HOLD = 60.0
#: vehicle length of the synthetic scenarios [m] (density proxy, platoon spacing)
VEHICLE_LENGTH = 4.5


def initial_gap(v0: float) -> float:
    """Reference starting gap ``2 m + 1.5 s * v0``; the follower settles during the hold."""
    return 2.0 + 1.5 * float(v0)


def speed_from_segments(v0: float, segments: Sequence[tuple[float, float]], dt: float = DT) -> np.ndarray:
    """Speed series of piecewise constant acceleration, floored at zero.

    ``segments`` are ``(acceleration [m/s^2], duration [s])`` pairs; the series starts with the
    sample ``v0`` and has ``1 + sum(round(duration / dt))`` samples.
    """
    v = [float(v0)]
    for acc, duration in segments:
        for _ in range(int(round(duration / dt))):
            v.append(max(v[-1] + float(acc) * dt, 0.0))
    return np.asarray(v, dtype=np.float64)


def with_hold(v_manoeuvre: np.ndarray, hold: float = HOLD, dt: float = DT) -> tuple[np.ndarray, int]:
    """Prepend the hold phase at ``v_manoeuvre[0]``; returns the series and the manoeuvre start."""
    v_manoeuvre = np.asarray(v_manoeuvre, dtype=np.float64)
    n_hold = int(round(hold / dt))
    if n_hold < WARMUP + 1:
        raise ValueError(f"a hold of {hold} s is shorter than the warm-up of {WARMUP} samples")
    return np.concatenate([np.full(n_hold, v_manoeuvre[0]), v_manoeuvre]), n_hold


def positions(v: np.ndarray, x0: float = 0.0, dt: float = DT) -> np.ndarray:
    """Positions of a speed series under the integration scheme of the benchmark."""
    x = np.empty(len(v), dtype=np.float64)
    x[0] = x0
    x[1:] = x0 + np.cumsum(v[1:]) * dt
    return x


def scenario_event(
    v_lead: np.ndarray,
    s_init: float,
    event_id: str,
    hold: float = HOLD,
    dataset: str = "scenario",
    site: str = "synthetic",
    meta: dict | None = None,
    dt: float = DT,
) -> Event:
    """Event with the leader ``hold + v_lead`` and a placeholder follower at the gap ``s_init``.

    ``meta["manoeuvre_start"]`` is the index of ``v_lead[0]`` in the event (the hold ends there),
    ``meta["hold"]`` the hold in seconds. The event is not validated: it is a protocol input.
    """
    v, n_hold = with_hold(v_lead, hold, dt)
    if s_init <= 0:
        raise ValueError("the starting gap must be positive")
    n = len(v)
    x_lead = positions(v, 0.0, dt)
    a = np.empty(n, dtype=np.float64)
    a[:-1] = np.diff(v) / dt
    a[-1] = a[-2]
    return Event(
        event_id=event_id,
        dataset=dataset,
        site=site,
        follower_id=f"{dataset}/{event_id}",
        leader_id=f"{dataset}/{event_id}/leader",
        t=np.arange(n, dtype=np.float64) * dt,
        s=np.full(n, float(s_init)),
        dv=np.zeros(n),
        v=v.copy(),
        a=a,
        v_lead=v,
        x_lead=x_lead,
        x_follower=x_lead - float(s_init),
        meta={
            "data_version": "raw",
            "a_source": "scenario",
            "position_source": "scenario",
            "leader_length": VEHICLE_LENGTH,
            "manoeuvre_start": n_hold,
            "hold": float(hold),
            **(meta or {}),
        },
    )


def constant_speed_event(v0: float, duration: float, event_id: str, hold: float = HOLD, s_init: float | None = None, **kw) -> Event:
    """Leader at ``v0`` throughout: hold, then ``duration`` seconds more (equilibrium search)."""
    v = np.full(int(round(duration / DT)) + 1, float(v0))
    return scenario_event(v, initial_gap(v0) if s_init is None else s_init, event_id, hold, **kw)
