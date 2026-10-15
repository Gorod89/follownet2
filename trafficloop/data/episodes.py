"""Reconstruction and confirmation of the near-miss episodes (empirical suite of P4).

The episodes of :mod:`trafficloop.data.rare_events` are mined on the raw tracks and may contain
leader changes: ``x_lead`` and ``v_lead`` are those of the instantaneous leader, so a cut-in is
a downward jump of the leader position. :func:`trafficloop.data.reconstruction.reconstruct_event`
would smear such a jump, so an episode is reconstructed piecewise (docs/protocols.md, section 4):

* *leader jumps* are the samples ``k + 1`` with
  ``|x_lead[k+1] - x_lead[k] - 0.5 (v_lead[k] + v_lead[k+1]) dt| > 2 m``
  (the trapezoidal step is exact for the gap-based sources, whose leader position is the
  integral of its speed, and within a few centimetres for smooth tracks);
* every leader piece between two jumps with at least ``MIN_PIECE`` samples goes through
  :func:`reconstruct_trajectory`; shorter pieces (a leader seen for less than a second) stay raw;
* the follower goes through :func:`reconstruct_trajectory` with the upper bound
  ``x_lead_rec - s_min`` (a rigid shift makes the bound exact, as in ``reconstruct_event``);
* ``s``, ``dv`` and ``a`` are rebuilt; ``meta["data_version"] = "reconstructed"`` and
  ``meta["reconstruction"]`` carries the diagnostics (the keys of ``reconstruct_event``, so that
  :func:`trafficloop.data.reconstruction.summarise` applies, plus the jumps and pieces).

Leader jumps are detected from the positions only, so the gap-based sources without leader
changes (OpenACC, Waymo pairs, FollowNet: their leader position is the integral of its speed)
are unaffected and reconstructed in one piece, exactly like ``reconstruct_event``.

The triggers of :class:`EpisodeConfig` are then evaluated again on the reconstructed series
(:func:`confirm_triggers`): ``ttc`` and ``drac`` with ``min_speed``, ``min_gap`` and ``min_hold``,
``brake`` on the leader acceleration held ``min_hold`` samples, ``cut_in`` on the time gap at a
jump. ``meta["episode"]["confirmed_triggers"]`` lists the triggers that hold anywhere in the
episode, ``meta["episode"]["triggers_after_warmup"]`` those that hold at a sample index of at
least ``WARMUP`` (the end of the observed warm-up of the protocols), and
:func:`reconstruct_episodes` keeps the episodes whose second list is not empty: an episode whose
only critical instants are observed tests nothing in a rollout.

The reconstructed series are smooth, so the triggers are evaluated without the Savitzky-Golay
step of the miner. The leader acceleration is taken inside the pieces of at least ``MIN_PIECE``
samples only: the speed difference across a jump belongs to two vehicles, and a piece shorter
than a second is raw (it can neither be filtered nor hold a braking manoeuvre).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from typing import Iterable

import numpy as np

from trafficloop.data.rare_events import EpisodeConfig, _held, drac, time_to_collision
from trafficloop.data.reconstruction import METHOD, ReconstructionConfig, reconstruct_trajectory
from trafficloop.data.schema import DT, Event

log = logging.getLogger(__name__)

#: largest mismatch [m] between the leader displacement and its trapezoidal speed integral
JUMP_THRESHOLD = 2.0
#: shortest leader piece [samples] that is reconstructed; shorter pieces stay raw
MIN_PIECE = 10
#: end of the observed warm-up [samples], equal to ``trafficloop.data.batch.WARMUP`` (tested); not
#: imported from there because ``batch`` imports torch, which the worker processes do not need
WARMUP = 30


def leader_jumps(event: Event, threshold: float = JUMP_THRESHOLD, dt: float = DT) -> np.ndarray:
    """Boolean mask of the event's length, True at the first sample of every new leader piece.

    ``jumps[k + 1]`` is True when ``|x_lead[k+1] - x_lead[k] - 0.5 (v_lead[k] + v_lead[k+1]) dt|``
    exceeds ``threshold``; ``jumps[0]`` is always False.
    """
    out = np.zeros(len(event), dtype=bool)
    step = np.diff(event.x_lead) - 0.5 * (event.v_lead[:-1] + event.v_lead[1:]) * dt
    out[1:] = np.abs(step) > threshold
    return out


def _pieces(jumps: np.ndarray) -> list[tuple[int, int]]:
    edges = [0, *np.flatnonzero(jumps).tolist(), len(jumps)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _stored_jumps(event: Event) -> np.ndarray | None:
    """The jumps found on the raw series, when the event is a reconstructed episode."""
    samples = event.meta.get("reconstruction", {}).get("jump_samples")
    if samples is None:
        return None
    out = np.zeros(len(event), dtype=bool)
    out[np.asarray(samples, dtype=np.int64)] = True
    return out


def leader_acceleration(event: Event, jumps: np.ndarray, dt: float = DT) -> np.ndarray:
    """Leader acceleration ``(v_lead[k+1] - v_lead[k]) / dt`` inside the pieces of at least
    ``MIN_PIECE`` samples; NaN across a jump, in the short pieces and at the last sample."""
    out = np.full(len(event), np.nan)
    for a, b in _pieces(jumps):
        if b - a >= MIN_PIECE:
            out[a : b - 1] = np.diff(event.v_lead[a:b]) / dt
    return out


def _severity(event: Event, jumps: np.ndarray, cfg: EpisodeConfig) -> tuple[dict[str, np.ndarray], dict]:
    """Trigger masks of the series and the severity measures stored with the episode."""
    s, v = event.s, event.v
    dv = v - event.v_lead
    ttc = time_to_collision(s, dv)
    need = drac(s, dv)
    moving = (v >= cfg.min_speed) & (s >= cfg.min_gap)
    acc_lead = leader_acceleration(event, jumps, cfg.dt)
    with np.errstate(invalid="ignore"):
        braking = acc_lead < cfg.leader_acc_below  # NaN compares False
    fired = {
        "ttc": _held((ttc < cfg.ttc_below) & moving, cfg.min_hold),
        "drac": _held((need > cfg.drac_above) & moving, cfg.min_hold),
        "brake": _held(braking, cfg.min_hold),
        "cut_in": jumps & (s < cfg.cut_in_gap_below * np.maximum(v, 0.1)),
    }
    ttc_moving = np.where(moving, ttc, np.inf)
    measures = {
        "min_ttc": float(ttc_moving.min()) if np.isfinite(ttc_moving).any() else None,
        "max_drac": float(np.max(np.where(moving, need, 0.0))),
        "min_leader_acc": float(np.nanmin(acc_lead)) if np.isfinite(acc_lead).any() else None,
        "leader_jumps": int(jumps.sum()),
    }
    return fired, measures


def _holding(fired: dict[str, np.ndarray], cfg: EpisodeConfig, after: int = 0) -> list[str]:
    return [name for name in cfg.triggers if bool(fired[name][after:].any())]


def confirm_triggers(
    event: Event, episode_cfg: EpisodeConfig | None = None, jumps: np.ndarray | None = None, after: int = 0
) -> list[str]:
    """The triggers of ``episode_cfg.triggers`` that hold on the series of ``event`` at a sample
    index of at least ``after`` (0: anywhere; ``WARMUP``: after the observed warm-up).

    A trigger instant is a sample where the condition has held for ``min_hold`` samples (``ttc``,
    ``drac``, ``brake``) or a jump sample (``cut_in``). ``jumps`` defaults to the jumps stored by
    :func:`reconstruct_episode` (found on the raw series) and otherwise to :func:`leader_jumps`
    of the event itself.
    """
    cfg = episode_cfg or EpisodeConfig()
    if jumps is None:
        jumps = _stored_jumps(event)
    if jumps is None:
        jumps = leader_jumps(event, dt=cfg.dt)
    fired, _ = _severity(event, jumps, cfg)
    return _holding(fired, cfg, after)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def reconstruct_episode(
    event: Event,
    cfg: ReconstructionConfig | None = None,
    episode_cfg: EpisodeConfig | None = None,
    warmup: int = WARMUP,
) -> Event | None:
    """Reconstructed version of a raw near-miss episode, or ``None`` when a solve fails.

    The returned episode carries ``meta["episode"]["confirmed_triggers"]`` (anywhere, possibly
    empty), ``meta["episode"]["triggers_after_warmup"]`` (at a sample index ``>= warmup``) and
    the severity measures of the reconstructed series in ``meta["episode"]["reconstructed"]``.
    """
    cfg = cfg or ReconstructionConfig()
    episode_cfg = episode_cfg or EpisodeConfig()
    n = len(event)
    jumps = leader_jumps(event, dt=cfg.dt)
    x_lead, v_lead = np.empty(n), np.empty(n)
    status, outliers, raw_pieces = [], [], []
    for a, b in _pieces(jumps):
        if b - a >= MIN_PIECE:
            lead = reconstruct_trajectory(event.x_lead[a:b], event.v_lead[a:b], cfg)
            if not lead["ok"]:
                return None
            x_lead[a:b], v_lead[a:b] = lead["x"], lead["v"]
            status.append(lead["status"])
            outliers.append(lead["outliers"])
        else:
            x_lead[a:b], v_lead[a:b] = event.x_lead[a:b], np.clip(event.v_lead[a:b], 0.0, None)
            raw_pieces.append([a, b])
    gap_based = event.meta.get("position_source", "track") == "gap"
    target = x_lead - event.s if gap_based else event.x_follower
    follower = reconstruct_trajectory(target, event.v, cfg, x_upper=x_lead - cfg.s_min)
    if not follower["ok"]:
        return None
    s = x_lead - follower["x"]
    short = cfg.s_min - float(np.min(s))
    if short > 0.0:
        # the solver meets the gap bound to its tolerance only: a rigid shift makes it exact
        follower["x"] = follower["x"] - short
        s = x_lead - follower["x"]
    jerk_raw = np.diff(event.a) / cfg.dt
    jerk = np.diff(follower["a"][:-1]) / cfg.dt
    meta = {
        **event.meta,
        "data_version": "reconstructed",
        "a_source": "reconstruction",
        "speeds": "reconstruction",
        "reconstruction": {
            "method": METHOD,
            "piecewise_leader": True,
            **{k: v for k, v in asdict(cfg).items() if k not in ("eps", "max_iter", "dt")},
            "jump_threshold": JUMP_THRESHOLD,
            "min_piece": MIN_PIECE,
            "jump_samples": [int(k) for k in np.flatnonzero(jumps)],
            "raw_pieces": raw_pieces,
            "status": [*status, follower["status"]],
            "outlier_share": [*outliers, follower["outliers"]],
            "rms_gap_change": _rms(s - event.s),
            "rms_speed_change": _rms(follower["v"] - event.v),
            "rms_acc_change": _rms(follower["a"] - event.a),
            "rms_acc_raw": _rms(event.a),
            "rms_acc": _rms(follower["a"]),
            "max_abs_acc_raw": float(np.max(np.abs(event.a))),
            "max_abs_acc": float(np.max(np.abs(follower["a"]))),
            "max_abs_jerk_raw": float(np.max(np.abs(jerk_raw))),
            "max_abs_jerk": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
            "min_gap_raw": float(np.min(event.s)),
            "min_gap": float(np.min(s)),
            "gap_bound_active": bool(np.min(s) <= cfg.s_min + 1e-3),
        },
    }
    rec = Event(
        event_id=event.event_id,
        dataset=event.dataset,
        site=event.site,
        follower_id=event.follower_id,
        leader_id=event.leader_id,
        t=event.t.copy(),
        s=s,
        dv=follower["v"] - v_lead,
        v=follower["v"],
        a=follower["a"],
        v_lead=v_lead,
        x_lead=x_lead,
        x_follower=follower["x"],
        meta=meta,
    )
    fired, measures = _severity(rec, jumps, episode_cfg)
    rec.meta["episode"] = {
        **event.meta.get("episode", {}),
        "confirmed_triggers": _holding(fired, episode_cfg),
        "triggers_after_warmup": _holding(fired, episode_cfg, warmup),
        "reconstructed": measures,
    }
    return rec


def _work(job: tuple[Event, ReconstructionConfig, EpisodeConfig, int]) -> Event | None:
    return reconstruct_episode(*job)


def _count(names: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in names:
        out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items()))


def reconstruct_episodes(
    events: Iterable[Event],
    cfg: ReconstructionConfig | None = None,
    workers: int = 1,
    episode_cfg: EpisodeConfig | None = None,
    chunksize: int = 16,
    warmup: int = WARMUP,
) -> tuple[list[Event], dict]:
    """Reconstruct and confirm many episodes; returns the kept episodes and the report.

    An episode is kept (confirmed) when a trigger holds at a sample index ``>= warmup``. The
    report counts the episodes in (``n_in``), the failed solves (``n_failed``, ids in
    ``failed_events``), the reconstructed ones (``n_reconstructed = n_confirmed + n_only_warmup
    + n_unconfirmed``): kept, dropped because every confirmed trigger instant lies inside the
    warm-up, dropped without a confirmed trigger. Per trigger: mined on the raw series
    (``mined_triggers``), confirmed on the reconstructed one anywhere (``confirmed_triggers``) and
    after the warm-up (``triggers_after_warmup``), both counted over the kept episodes, and the
    share of the reconstructed episodes mined with a trigger in which it is confirmed anywhere
    (``confirmation_rate``) and after the warm-up (``confirmation_rate_after_warmup``). Finally
    the pooled rms acceleration of the follower before (``rms_acc_raw``) and after (``rms_acc``)
    over the reconstructed episodes.
    """
    cfg = cfg or ReconstructionConfig()
    episode_cfg = episode_cfg or EpisodeConfig()
    events = list(events)
    jobs = [(ev, cfg, episode_cfg, warmup) for ev in events]
    out: list[Event | None] = []
    if workers > 1 and len(events) > chunksize:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(_work, jobs, chunksize=chunksize):
                    out.append(result)
        except BrokenProcessPool:
            # a worker was killed (the machine ran out of memory): the rest is done here
            log.warning("process pool broke after %d of %d episodes, continuing serially", len(out), len(jobs))
    out.extend(_work(job) for job in jobs[len(out):])
    pairs = [(raw, rec) for raw, rec in zip(events, out) if rec is not None]
    tags = lambda ev, key: ev.meta.get("episode", {}).get(key, [])  # noqa: E731
    kept = [rec for _, rec in pairs if tags(rec, "triggers_after_warmup")]
    only_warmup = sum(bool(tags(rec, "confirmed_triggers")) and not tags(rec, "triggers_after_warmup") for _, rec in pairs)
    rates: dict[str, dict] = {"confirmed_triggers": {}, "triggers_after_warmup": {}}
    for name in episode_cfg.triggers:
        with_name = [rec for raw, rec in pairs if name in tags(raw, "triggers")]
        for key, rate in rates.items():
            hits = sum(name in tags(rec, key) for rec in with_name)
            rate[name] = float(hits / len(with_name)) if with_name else None
    raw_a = np.concatenate([raw.a for raw, _ in pairs]) if pairs else np.zeros(0)
    rec_a = np.concatenate([rec.a for _, rec in pairs]) if pairs else np.zeros(0)
    report = {
        "n_in": len(events),
        "n_failed": len(events) - len(pairs),
        "n_reconstructed": len(pairs),
        "n_confirmed": len(kept),
        "n_only_warmup": int(only_warmup),
        "n_unconfirmed": len(pairs) - len(kept) - int(only_warmup),
        "warmup": int(warmup),
        "mined_triggers": _count(name for ev in events for name in tags(ev, "triggers")),
        "confirmed_triggers": _count(name for ev in kept for name in tags(ev, "confirmed_triggers")),
        "triggers_after_warmup": _count(name for ev in kept for name in tags(ev, "triggers_after_warmup")),
        "confirmation_rate": rates["confirmed_triggers"],
        "confirmation_rate_after_warmup": rates["triggers_after_warmup"],
        "rms_acc_raw": _rms(raw_a),
        "rms_acc": _rms(rec_a),
        "failed_events": [ev.event_id for ev, rec in zip(events, out) if rec is None],
    }
    return kept, report
