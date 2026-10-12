"""P4: rare events, a synthetic scenario suite and the empirical near-miss episodes.

**Synthetic suite** (:func:`synthetic_suite`, version ``SUITE_VERSION``): every scenario holds
the leader at ``v0`` for ``hold`` seconds (warm-up included, the follower settles at its own
equilibrium) and starts the manoeuvre at ``meta["manoeuvre_start"]``:

* ``pulse_a<a>_v<v0>``: deceleration at ``a in {-4, -6, -8}`` by 10 m/s, 10 s at the low
  speed, acceleration at +1 m/s^2 back to ``v0``, 20 s at ``v0``, for ``v0 in {15, 25, 35}``;
  the last step of a ramp is shorter when 10 m/s is not a multiple of ``|a| dt``;
* ``stop_v30``: from 30 m/s at -4 m/s^2 to standstill, 20 s stopped;
* ``cutin_tau<tau>_v<v0>``: the leader drives at ``v0`` throughout; at the manoeuvre start a
  new leader (speed ``v0``) appears at the gap ``tau * v_f(k_c)`` in front of the follower's
  simulated position and the old leader disappears (:func:`rollout_cut_in`), then 30 s follow.

``meta["scenario"]`` holds ``kind`` (pulse, stop, cut_in), ``a``, ``v0`` and ``tau`` (``None``
where a value does not apply).

**Metrics** (:func:`evaluate_p4`) per scenario from the manoeuvre start ``k_c`` and per episode
from the end of the warm-up ``k_0 = warmup``: the window holds the simulated states
``k_c + 1 .. n - 1`` and the accelerations decided from the start on, ``a[k_c .. n - 2]``
(the acceleration ``a[k]`` is the one realised over the step ``k -> k + 1``):

* ``collided`` (``s <= 0`` at a state of the window), ``first_collision`` [s after the start,
  NaN without collision], ``min_gap``;
* ``min_ttc``: smallest ``s / dv`` over the states with ``dv > 0``, capped at ``TTC_MAX`` (100 s:
  never closing, or closing without a conflict);
  a collided follower has ``min_ttc = 0``, the value reached at the collision (a collided
  vehicle keeps driving through the leader, and ``s / dv`` of the overlap means nothing);
* ``min_acc``, ``exceedance_4`` and ``exceedance_6``: share of the accelerations of the window
  with ``a <= -4`` and ``a <= -6 m/s^2`` (a tolerance of ``EXCEEDANCE_TOL`` absorbs the float32
  rounding of an acceleration realised at a threshold), with the counts ``n_exceed_4``,
  ``n_exceed_6`` and ``n_samples`` for pooling;
* ``amplification = min_acc / min(a_lead)`` for the pulses and the stop;
* episodes: the same measures of the recorded follower over the same samples
  (``min_ttc_recorded``, ``min_acc_recorded``, ``exceedance_4_recorded``, ...).

The tables keep ``min_ttc = inf``; in the summary an infinite quantile is ``None`` (JSON has no
infinity): most items of the set never close in.

A collision during the hold of a scenario is reported apart (``collided_hold``): the metrics of
the manoeuvre are then those of a follower that started from an invalid state.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from follownet2.data.batch import A_PREV, WARMUP, EventBatch, S, V, length_buckets
from follownet2.data.rare_events import TRIGGERS, time_to_collision
from follownet2.data.schema import DT, Event
from follownet2.models.base import ACC_CLIP, CFModel
from follownet2.protocols.scenarios import HOLD, initial_gap, scenario_event, speed_from_segments
from follownet2.sim.rollout import RolloutResult, advance, rollout

SUITE_VERSION = "p4-v1"
PULSE_ACCELERATIONS = (-4.0, -6.0, -8.0)  # m/s^2
PULSE_SPEEDS = (15.0, 25.0, 35.0)  # m/s
PULSE_DROP = 10.0  # m/s, speed lost in a pulse
PULSE_LOW = 10.0  # s at the low speed
PULSE_RECOVERY = 1.0  # m/s^2, acceleration back to v0
PULSE_TAIL = 20.0  # s at v0 after the recovery
STOP_SPEED = 30.0
STOP_DECELERATION = -4.0
STOP_TAIL = 20.0  # s at standstill
CUT_IN_GAPS = (0.5, 1.0)  # s, time gap of the new leader
CUT_IN_SPEEDS = (15.0, 25.0)
CUT_IN_TAIL = 30.0  # s after the cut-in
EXCEEDANCE_TOL = 1e-3  # m/s^2
#: s; a time to collision above this is no conflict: ``min_ttc`` is capped here so that the
#: quantiles of the summary stay finite and readable
TTC_MAX = 100.0


def _ramp(delta_v: float, acc: float, dt: float = DT) -> list[tuple[float, float]]:
    """Segments that change the speed by ``delta_v`` at ``acc``; the last step is shorter."""
    steps = abs(delta_v) / (abs(acc) * dt)
    full = int(math.floor(steps + 1e-9))
    segments = [(float(acc), full * dt)] if full else []
    rest = abs(delta_v) - full * abs(acc) * dt
    if rest > 1e-9:
        segments.append((math.copysign(rest / dt, acc), dt))
    return segments


def pulse_speed(a: float, v0: float, dt: float = DT) -> np.ndarray:
    """Leader speed of a braking pulse (manoeuvre part, without the hold)."""
    return speed_from_segments(
        v0,
        [*_ramp(PULSE_DROP, a, dt), (0.0, PULSE_LOW), *_ramp(PULSE_DROP, PULSE_RECOVERY, dt), (0.0, PULSE_TAIL)],
        dt,
    )


def stop_speed(v0: float = STOP_SPEED, a: float = STOP_DECELERATION, dt: float = DT) -> np.ndarray:
    """Leader speed of the full stop (manoeuvre part, without the hold)."""
    return speed_from_segments(v0, [*_ramp(v0, a, dt), (0.0, STOP_TAIL)], dt)


def scenario_id(kind: str, v0: float, a: float | None = None, tau: float | None = None) -> str:
    if kind == "pulse":
        return f"pulse_a{a:g}_v{v0:g}"
    if kind == "stop":
        return f"stop_v{v0:g}"
    if kind == "cut_in":
        return f"cutin_tau{tau:.1f}_v{v0:g}"
    raise ValueError(f"unknown scenario kind {kind!r}")


def _scenario(kind: str, v: np.ndarray, v0: float, hold: float, a: float | None = None, tau: float | None = None) -> Event:
    sid = scenario_id(kind, v0, a, tau)
    meta = {
        "suite": SUITE_VERSION,
        "scenario": {
            "kind": kind,
            "a": None if a is None else float(a),
            "v0": float(v0),
            "tau": None if tau is None else float(tau),
        },
    }
    return scenario_event(v, initial_gap(v0), sid, hold=hold, meta=meta)


def synthetic_suite(hold: float = HOLD) -> list[Event]:
    """The 14 scenarios: 9 pulses, the full stop, 4 cut-ins (in this order)."""
    suite = [_scenario("pulse", pulse_speed(a, v0), v0, hold, a=a) for a in PULSE_ACCELERATIONS for v0 in PULSE_SPEEDS]
    suite.append(_scenario("stop", stop_speed(), STOP_SPEED, hold, a=STOP_DECELERATION))
    for tau in CUT_IN_GAPS:
        for v0 in CUT_IN_SPEEDS:
            v = speed_from_segments(v0, [(0.0, CUT_IN_TAIL)])
            suite.append(_scenario("cut_in", v, v0, hold, tau=tau))
    return suite


def scenario_kind(event: Event) -> str | None:
    return event.meta.get("scenario", {}).get("kind")


@torch.no_grad()
def rollout_cut_in(
    model: CFModel,
    batch: EventBatch,
    tau: float | Sequence[float] | torch.Tensor,
    warmup: int = WARMUP,
    clip: tuple[float, float] = ACC_CLIP,
    start: int | Sequence[int] | None = None,
) -> RolloutResult:
    """:func:`follownet2.sim.rollout.rollout` with a cut-in at the step ``k_c = start``.

    At ``k_c`` the leader displacement of the step is replaced by
    ``x_f(k_c) + tau v_f(k_c) + dx_lead[k_c] - x_lead(k_c) = tau v_f(k_c) - s(k_c) + dx_lead[k_c]``
    from the simulated state, so that the new leader stands ``tau v_f(k_c)`` in front of the
    simulated follower and moves like the recorded leader (``dx_lead[k_c] = v0 dt`` in the
    suite): the gap after the step is ``tau v_f(k_c) + (v0 - v_f(k_c + 1)) dt``. Afterwards the
    leader is the new one. ``tau`` and ``start`` are scalars or one value per event; ``start``
    defaults to the end of the hold, ``round(HOLD / dt)``.
    """
    n_steps, dt = batch.n_steps, batch.dt
    history = model.history
    if warmup + 1 < history:
        raise ValueError(f"warm-up of {warmup} samples is shorter than the history {history} of {model.name}")
    if n_steps < warmup + 2:
        raise ValueError(f"events of {n_steps} samples are too short for a warm-up of {warmup}")
    n_events = batch.n_events
    starts = [int(round(HOLD / dt))] * n_events if start is None else np.broadcast_to(np.asarray(start), (n_events,)).tolist()
    lengths = batch.lengths.tolist()
    for k_c, n in zip(starts, lengths):
        if not warmup <= k_c < n - 1:
            raise ValueError(f"cut-in step {k_c} outside the rollout [{warmup}, {n - 1}) of an event")
    state = batch.state().clone()
    tau = torch.as_tensor(tau, dtype=state.dtype, device=state.device).expand(n_events)
    cut = {
        k: torch.as_tensor([k_c == k for k_c in starts], device=state.device) for k in sorted(set(starts))
    }
    if warmup >= WARMUP - 1:
        model.begin(state[:, warmup - WARMUP + 1 : warmup + 1, :])
    acc = torch.zeros_like(state[..., 0])
    n_invalid = torch.zeros((), dtype=torch.long, device=state.device)
    for k in range(warmup, n_steps - 1):
        hist = state[:, k - history + 1 : k + 1, :]
        a = model(hist)
        dx = batch.dx_lead[:, k]
        if k in cut:
            dx = torch.where(cut[k], tau * state[:, k, V] - state[:, k, S] + dx, dx)
        row, n_bad = advance(a, state[:, k, S], state[:, k, V], dx, batch.v_lead[:, k + 1], clip[0], clip[1], dt)
        state[:, k + 1, :] = row
        acc[:, k] = row[:, A_PREV]
        n_invalid = n_invalid + n_bad
    steps = torch.arange(n_steps, device=state.device)
    sim_mask = batch.mask & (steps > warmup)
    hit = (state[..., S] <= 0.0) & sim_mask
    collided = hit.any(dim=-1)
    first = torch.where(hit, steps, torch.full_like(steps, n_steps)).amin(dim=-1)
    first = torch.where(collided, first, torch.full_like(first, -1))
    return RolloutResult(state[..., S], state[..., V], acc, sim_mask, collided, first, int(n_invalid), warmup)


def window_metrics(s: np.ndarray, dv: np.ndarray, a: np.ndarray, start: int, n: int, dt: float = DT) -> dict:
    """Measures of one follower over the states ``start + 1 .. n - 1`` and the accelerations
    ``start .. n - 2`` (arrays of at least ``n`` samples, ``dv = v - v_lead``)."""
    s_w, dv_w, a_w = s[start + 1 : n], dv[start + 1 : n], a[start : n - 1]
    hit = np.flatnonzero(s_w <= 0.0)
    ttc = time_to_collision(s_w, dv_w)
    n_samples = len(a_w)
    n4 = int(np.sum(a_w <= -4.0 + EXCEEDANCE_TOL))
    n6 = int(np.sum(a_w <= -6.0 + EXCEEDANCE_TOL))
    return {
        "collided": bool(len(hit)),
        "first_collision": float((hit[0] + 1) * dt) if len(hit) else math.nan,
        "min_gap": float(s_w.min()),
        "min_ttc": 0.0 if len(hit) else float(min(ttc.min(), TTC_MAX)),
        "min_acc": float(a_w.min()),
        "exceedance_4": n4 / n_samples,
        "exceedance_6": n6 / n_samples,
        "n_samples": n_samples,
        "n_exceed_4": n4,
        "n_exceed_6": n6,
    }


def _arrays(result: RolloutResult, batch: EventBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = result.s.double().cpu().numpy()
    v = result.v.double().cpu().numpy()
    a = result.a.double().cpu().numpy()
    return s, v - batch.v_lead.double().cpu().numpy(), a


def _scenario_rows(events: list[Event], result: RolloutResult, batch: EventBatch, warmup: int) -> list[dict]:
    s, dv, a = _arrays(result, batch)
    rows = []
    for i, ev in enumerate(events):
        n, k_c = len(ev), int(ev.meta.get("manoeuvre_start", warmup))
        info = ev.meta.get("scenario", {})
        kind = info.get("kind")
        a_lead = np.diff(ev.v_lead[k_c:]) / batch.dt
        min_a_lead = float(a_lead.min()) if len(a_lead) else math.nan
        metrics = window_metrics(s[i], dv[i], a[i], k_c, n, batch.dt)
        amplify = kind in ("pulse", "stop") and min_a_lead < 0.0
        rows.append(
            {
                "event_id": ev.event_id,
                "follower_id": ev.follower_id,
                "kind": kind,
                "a": math.nan if info.get("a") is None else float(info["a"]),
                "v0": math.nan if info.get("v0") is None else float(info["v0"]),
                "tau": math.nan if info.get("tau") is None else float(info["tau"]),
                "manoeuvre_start": k_c,
                "collided_hold": bool(np.any(s[i, warmup + 1 : k_c + 1] <= 0.0)),
                **metrics,
                "min_a_lead": min_a_lead,
                "amplification": metrics["min_acc"] / min_a_lead + 0.0 if amplify else math.nan,
            }
        )
    return rows


def _joined(names: list[str] | None) -> str | None:
    return None if names is None else ",".join(names)


def _episode_rows(events: list[Event], result: RolloutResult, batch: EventBatch, warmup: int) -> list[dict]:
    """One row per episode; the ``trigger_*`` flags follow the most specific list of the episode:
    triggers after the warm-up, else confirmed, else mined (raw episodes)."""
    s, dv, a = _arrays(result, batch)
    rows = []
    for i, ev in enumerate(events):
        n = len(ev)
        tags = ev.meta.get("episode", {})
        mined, confirmed, late = list(tags.get("triggers", [])), tags.get("confirmed_triggers"), tags.get("triggers_after_warmup")
        active = late if late is not None else confirmed if confirmed is not None else mined
        recorded = window_metrics(ev.s, ev.dv, ev.a, warmup, n, batch.dt)
        rows.append(
            {
                "event_id": ev.event_id,
                "follower_id": ev.follower_id,
                "dataset": ev.dataset,
                "site": ev.site,
                "data_version": ev.data_version,
                "triggers": ",".join(mined),
                "confirmed_triggers": _joined(confirmed),
                "triggers_after_warmup": _joined(late),
                **{f"trigger_{name}": name in active for name in TRIGGERS},
                **window_metrics(s[i], dv[i], a[i], warmup, n, batch.dt),
                **{f"{key}_recorded": recorded[key] for key in (
                    "collided", "min_gap", "min_ttc", "min_acc", "exceedance_4", "exceedance_6", "n_exceed_4", "n_exceed_6",
                )},
            }
        )
    return rows


SCENARIO_COLUMNS = [
    "event_id", "follower_id", "kind", "a", "v0", "tau", "manoeuvre_start", "collided_hold", "collided",
    "first_collision", "min_gap", "min_ttc", "min_acc", "exceedance_4", "exceedance_6", "n_samples",
    "n_exceed_4", "n_exceed_6", "min_a_lead", "amplification",
]
EPISODE_COLUMNS = [
    "event_id", "follower_id", "dataset", "site", "data_version", "triggers", "confirmed_triggers",
    "triggers_after_warmup", *[f"trigger_{name}" for name in TRIGGERS],
    "collided", "first_collision", "min_gap", "min_ttc", "min_acc", "exceedance_4", "exceedance_6",
    "n_samples", "n_exceed_4", "n_exceed_6",
    "collided_recorded", "min_gap_recorded", "min_ttc_recorded", "min_acc_recorded", "exceedance_4_recorded",
    "exceedance_6_recorded", "n_exceed_4_recorded", "n_exceed_6_recorded",
]


def quantile(values, q: float) -> float | None:
    """Linear-interpolation quantile that keeps ``inf`` (a quantile next to ``inf`` is ``inf``);
    ``None`` for no values."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    if len(x) == 0:
        return None
    pos = q * (len(x) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(x) - 1)
    w = pos - lo
    if not np.isfinite(x[lo]) or (w > 0.0 and not np.isfinite(x[hi])):
        return float(x[hi] if w > 0.0 else x[lo])
    return float(x[lo] + w * (x[hi] - x[lo]))


def _summary_quantile(values, q: float) -> float | None:
    """:func:`quantile` for the JSON summary: an infinite quantile is ``None``."""
    value = quantile(values, q)
    return value if value is None or math.isfinite(value) else None


def _share(table: pd.DataFrame, column: str) -> float | None:
    return float(table[column].mean()) if len(table) else None


def _pooled(table: pd.DataFrame, count: str, total: str = "n_samples") -> float | None:
    n = int(table[total].sum()) if len(table) else 0
    return float(table[count].sum() / n) if n else None


@torch.no_grad()
def evaluate_p4(
    model: CFModel,
    scenarios: Sequence[Event] | None,
    episodes: Sequence[Event] | None,
    warmup: int = WARMUP,
    clip: tuple[float, float] = ACC_CLIP,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    batch_events: int = 1024,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Score ``model`` on the synthetic scenarios (``None``: :func:`synthetic_suite`) and the
    near-miss episodes (may be empty). Returns the summary and the tables ``scenarios`` and
    ``episodes`` (one row each)."""
    scenarios = synthetic_suite() if scenarios is None else list(scenarios)
    episodes = [] if episodes is None else list(episodes)
    model = model.to(device).eval()
    n_invalid = 0
    rows: dict[str, dict] = {}
    cut_ins = [ev for ev in scenarios if scenario_kind(ev) == "cut_in"]
    prescribed = [ev for ev in scenarios if scenario_kind(ev) != "cut_in"]
    if prescribed:
        batch = EventBatch.from_events(prescribed, device=device, dtype=dtype)
        result = rollout(model, batch, warmup, clip)
        n_invalid += result.n_invalid
        rows.update((r["event_id"], r) for r in _scenario_rows(prescribed, result, batch, warmup))
    if cut_ins:
        batch = EventBatch.from_events(cut_ins, device=device, dtype=dtype)
        tau = [float(ev.meta["scenario"]["tau"]) for ev in cut_ins]
        start = [int(ev.meta["manoeuvre_start"]) for ev in cut_ins]
        result = rollout_cut_in(model, batch, tau, warmup, clip, start=start)
        n_invalid += result.n_invalid
        rows.update((r["event_id"], r) for r in _scenario_rows(cut_ins, result, batch, warmup))
    scen = pd.DataFrame([rows[ev.event_id] for ev in scenarios], columns=SCENARIO_COLUMNS)

    usable = [ev for ev in episodes if len(ev) >= warmup + 2]
    episode_rows: list[dict] = []
    for index in length_buckets([len(ev) for ev in usable], batch_events) if usable else []:
        chunk = [usable[i] for i in index]
        batch = EventBatch.from_events(chunk, device=device, dtype=dtype)
        result = rollout(model, batch, warmup, clip)
        n_invalid += result.n_invalid
        episode_rows.extend(zip(index.tolist(), _episode_rows(chunk, result, batch, warmup)))
    epi = pd.DataFrame([row for _, row in sorted(episode_rows, key=lambda item: item[0])], columns=EPISODE_COLUMNS)

    pulses = scen[scen["kind"].isin(["pulse", "stop"])]
    amplification = pulses["amplification"].to_numpy(dtype=np.float64)
    amplification = amplification[np.isfinite(amplification)]
    summary = {
        "collision_rate_synthetic": _share(scen, "collided"),
        "collision_rate_empirical": _share(epi, "collided"),
        **{f"min_ttc_q{q:02d}": _summary_quantile(scen["min_ttc"], q / 100) for q in (5, 25, 50)},
        **{f"min_ttc_q{q:02d}_empirical": _summary_quantile(epi["min_ttc"], q / 100) for q in (5, 25, 50)},
        "exceedance_4": _pooled(scen, "n_exceed_4"),
        "exceedance_6": _pooled(scen, "n_exceed_6"),
        "exceedance_4_empirical": _pooled(epi, "n_exceed_4"),
        "exceedance_6_empirical": _pooled(epi, "n_exceed_6"),
        "amplification_mean": float(amplification.mean()) if len(amplification) else None,
        "n_scenarios": int(len(scen)),
        "n_episodes": int(len(epi)),
        "n_episodes_skipped": int(len(episodes) - len(usable)),
        "n_collided_hold": int(scen["collided_hold"].sum()) if len(scen) else 0,
        "n_invalid": int(n_invalid),
        "min_ttc_q50_recorded": _summary_quantile(epi["min_ttc_recorded"], 0.5),
        "exceedance_4_recorded": _pooled(epi, "n_exceed_4_recorded"),
    }
    return summary, {"scenarios": scen, "episodes": epi}
