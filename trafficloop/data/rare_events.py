"""Empirical near-miss episodes for the rare-event suite (P4).

The FollowNet criteria (15 s behind one unchanged leader) remove most safety-critical
interactions: they are short and many start with a cut-in. Episodes are therefore mined
from the follower series themselves, with their own, weaker validity rules:

* the follower keeps its lane, the gap is positive, all values are finite;
* the leader MAY change: ``x_lead`` and ``v_lead`` are those of the instantaneous leader,
  so a cut-in is a downward jump of the gap, which a closed-loop replay reproduces through
  the leader displacement.

Triggers, evaluated on lightly smoothed speeds to keep measurement noise from firing them.
``ttc`` and ``drac`` fire only while the follower is faster than ``min_speed`` and the gap is
larger than ``min_gap``: in a creeping queue a gap of 2 m closing at 1 m/s has a time to
collision of 2 s without any danger, and gaps of a few centimetres at speed are artefacts of
the records (70 % of the first candidates in the Waymo pairs were of these two kinds).

============  ==========================================================  =============
trigger       condition (held for ``min_hold`` consecutive samples)        default
============  ==========================================================  =============
``ttc``       time to collision ``s / dv`` with ``dv = v - v_lead > 0``    below 2.0 s
``drac``      deceleration rate to avoid a crash ``dv^2 / (2 s)``          above 3.35 m/s^2
``brake``     leader acceleration                                          below -4 m/s^2
``cut_in``    the leader changes and the new time gap ``s / v``            below 1.0 s
============  ==========================================================  =============

An episode is the window ``[t* - pre, t* + post]`` around the most severe instant ``t*``
(smallest TTC, or the trigger instant), cut to the valid run and extended backwards when the
run ends before ``min_duration`` is reached (evasive lane change, end of the record). ``pre`` contains the 3 s of
observed history that every protocol gives to the models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from trafficloop.data.extraction import PairSeries, savgol_derivative
from trafficloop.data.schema import DT, Event

TRIGGERS = ("ttc", "drac", "brake", "cut_in")


@dataclass
class EpisodeConfig:
    ttc_below: float = 2.0
    drac_above: float = 3.35
    leader_acc_below: float = -4.0
    cut_in_gap_below: float = 1.0  # s, time gap right after a leader change
    min_speed: float = 3.0  # m/s, follower speed below which ttc and drac do not fire
    min_gap: float = 1.0  # m, gap below which ttc and drac do not fire
    min_hold: int = 3  # samples
    pre: float = 5.0
    post: float = 10.0
    min_duration: float = 6.0
    savgol_window: int = 11
    savgol_order: int = 2
    negative_speed_tol: float = 0.5
    triggers: tuple[str, ...] = TRIGGERS
    dt: float = DT


def time_to_collision(s: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """``s / dv`` where the follower closes in, infinity elsewhere."""
    out = np.full(len(s), np.inf)
    closing = dv > 1e-6
    out[closing] = s[closing] / dv[closing]
    return out


def drac(s: np.ndarray, dv: np.ndarray) -> np.ndarray:
    """Deceleration needed to match the leader's speed within the gap."""
    return np.where(dv > 0.0, dv * dv / (2.0 * np.maximum(s, 1e-6)), 0.0)


def _held(mask: np.ndarray, n: int) -> np.ndarray:
    """True where ``mask`` has been true for the last ``n`` samples."""
    if n <= 1:
        return mask.copy()
    run = np.convolve(mask.astype(np.int64), np.ones(n, dtype=np.int64), mode="full")[: len(mask)]
    return run >= n


def _valid_runs(pair: PairSeries, dt: float, tol: float = 0.0) -> list[tuple[int, int]]:
    n = len(pair.t)
    has_leader = np.array([lid is not None and lid == lid for lid in pair.leader_id], dtype=bool)
    with np.errstate(invalid="ignore"):
        ok = (
            has_leader
            & np.isfinite(pair.x_follower) & np.isfinite(pair.v) & np.isfinite(pair.x_lead) & np.isfinite(pair.v_lead)
            & (pair.x_lead - pair.x_follower > 0.0) & (pair.v >= -tol) & (pair.v_lead >= -tol)
        )
    brk = np.zeros(n, dtype=bool)
    brk[1:] = np.abs(np.diff(pair.t) - dt) > 1e-6
    if pair.lane_follower is not None:
        lane = np.asarray(pair.lane_follower, dtype=np.float64)
        brk[1:] |= (lane[1:] != lane[:-1]) & np.isfinite(lane[1:]) & np.isfinite(lane[:-1])
    runs, start = [], None
    for k in range(n):
        if not ok[k]:
            if start is not None:
                runs.append((start, k))
                start = None
        elif start is None:
            start = k
        elif brk[k]:
            runs.append((start, k))
            start = k
    if start is not None:
        runs.append((start, n))
    return runs


def _smooth(y: np.ndarray, jumps: np.ndarray, cfg: EpisodeConfig, deriv: int) -> np.ndarray:
    """Savitzky-Golay per piece between leader changes, so that a jump is not smeared."""
    out = np.empty(len(y))
    edges = [0, *np.flatnonzero(jumps).tolist(), len(y)]
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            out[a:b] = savgol_derivative(y[a:b], cfg.dt, cfg.savgol_window, cfg.savgol_order, deriv=deriv)
    return out


def mine_episodes(pair: PairSeries, cfg: EpisodeConfig | None = None, counters: dict | None = None) -> list[Event]:
    """Near-miss episodes of one follower, as raw events tagged in ``meta["episode"]``."""
    cfg = cfg or EpisodeConfig()
    counters = {} if counters is None else counters
    pre, post = int(round(cfg.pre / cfg.dt)), int(round(cfg.post / cfg.dt))
    min_samples = int(round(cfg.min_duration / cfg.dt)) + 1
    episodes: list[Event] = []
    for start, stop in _valid_runs(pair, cfg.dt, cfg.negative_speed_tol):
        if stop - start < min_samples:
            continue
        sl = slice(start, stop)
        leader = np.asarray(pair.leader_id[sl], dtype=object)
        change = np.zeros(stop - start, dtype=bool)
        change[1:] = leader[1:] != leader[:-1]
        s = (pair.x_lead[sl] - pair.x_follower[sl]).astype(np.float64)
        v = np.clip(pair.v[sl].astype(np.float64), 0.0, None)
        v_lead = np.clip(pair.v_lead[sl].astype(np.float64), 0.0, None)
        none = np.zeros(len(v), dtype=bool)
        v_s = np.clip(_smooth(v, none, cfg, 0), 0.0, None)
        vl_s = np.clip(_smooth(v_lead, change, cfg, 0), 0.0, None)
        al_s = _smooth(v_lead, change, cfg, 1)
        ttc = time_to_collision(s, v_s - vl_s)
        need = drac(s, v_s - vl_s)
        moving = (v_s >= cfg.min_speed) & (s >= cfg.min_gap)
        fired = {
            "ttc": _held((ttc < cfg.ttc_below) & moving, cfg.min_hold),
            "drac": _held((need > cfg.drac_above) & moving, cfg.min_hold),
            "brake": _held(al_s < cfg.leader_acc_below, cfg.min_hold),
            "cut_in": change & (s < cfg.cut_in_gap_below * np.maximum(v_s, 0.1)),
        }
        any_fired = np.zeros(len(s), dtype=bool)
        for name in cfg.triggers:
            any_fired |= fired[name]
        k = 0
        while k < len(s):
            if not any_fired[k]:
                k += 1
                continue
            # one episode per cluster of trigger instants that fall into the same window
            last = k
            while True:
                later = np.flatnonzero(any_fired[last + 1 : last + 1 + post])
                if len(later) == 0:
                    break
                last = last + 1 + int(later[-1])
            cluster = slice(k, last + 1)
            severity = np.where(moving[cluster], ttc[cluster], np.inf)
            worst = k + int(np.argmin(severity)) if np.isfinite(severity).any() else k
            b = min(len(s), worst + post + 1)
            # a run that ends soon after the critical instant is extended backwards
            a = max(0, min(worst - pre, b - min_samples))
            if b - a >= min_samples:
                episodes.append(_episode(pair, start + a, start + b, start + worst, cfg, counters, {
                    "triggers": [name for name in cfg.triggers if fired[name][cluster].any()],
                    "min_ttc": float(np.min(severity)) if np.isfinite(severity).any() else None,
                    "max_drac": float(np.max(np.where(moving[cluster], need[cluster], 0.0))),
                    "min_leader_acc": float(np.min(al_s[a:b])),
                    "cut_in": bool(change[a:b].any()),
                    "leader_changes": int(change[a:b].sum()),
                }))
            k = last + 1 + post
    return episodes


def _episode(pair: PairSeries, a: int, b: int, worst: int, cfg: EpisodeConfig, counters: dict, tags: dict) -> Event:
    sl = slice(a, b)
    n = b - a
    v = np.clip(pair.v[sl].astype(np.float64), 0.0, None)
    v_lead = np.clip(pair.v_lead[sl].astype(np.float64), 0.0, None)
    x_lead = pair.x_lead[sl].astype(np.float64)
    x_follower = pair.x_follower[sl].astype(np.float64)
    if pair.a is not None and np.all(np.isfinite(pair.a[sl])):
        acc, source = pair.a[sl].astype(np.float64), "dataset"
    else:
        acc, source = savgol_derivative(v, cfg.dt, cfg.savgol_window, cfg.savgol_order), "savgol"
    leader_id = str(pair.leader_id[worst])
    index = counters.get(pair.follower_id, 0)
    counters[pair.follower_id] = index + 1
    return Event(
        event_id=f"{pair.follower_id}|{leader_id.rsplit('/', 1)[-1]}|nm{index}",
        dataset=pair.dataset,
        site=pair.site,
        follower_id=pair.follower_id,
        leader_id=leader_id,
        t=np.arange(n, dtype=np.float64) * cfg.dt,
        s=x_lead - x_follower,
        dv=v - v_lead,
        v=v,
        a=acc,
        v_lead=v_lead,
        x_lead=x_lead,
        x_follower=x_follower,
        meta={
            **pair.meta,
            "t0": float(pair.t[a]),
            "a_source": source,
            "speeds": "measured",
            "data_version": "raw",
            "episode": {**tags, "t_trigger": float((worst - a) * cfg.dt)},
        },
    )


def mine_all(pairs: Iterable[PairSeries], cfg: EpisodeConfig | None = None) -> list[Event]:
    cfg = cfg or EpisodeConfig()
    counters: dict = {}
    out: list[Event] = []
    for pair in pairs:
        out.extend(mine_episodes(pair, cfg, counters))
    return out
