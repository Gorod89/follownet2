"""Platoon simulator of protocol P2 (docs/protocols.md, section 2).

Vehicle 0 is a prescribed leader: the manoeuvre speed series ``v_lead`` behind the hold phase of
:func:`~trafficloop.protocols.scenarios.with_hold`, positions by
:func:`~trafficloop.protocols.scenarios.positions`. Vehicles ``1..N`` are followers driven by the
model, follower ``i`` behind vehicle ``i - 1``. Every follower starts at the gap ``s_init`` and the
initial leader speed ``v0`` with the observed warm-up rows ``(s_init, 0, v0, 0)``;
``model.begin()`` receives them once, as in :func:`~trafficloop.sim.rollout.rollout`. From the
sample ``warmup`` on, one model call per step on the histories ``[N, H, 4]`` of all followers
gives their accelerations and all vehicles advance together with the step rule of ``rollout``::

    a[i, k]   = clip(model output), a non-finite output -> the lower clip (counted)
    v[i, k+1] = max(v[i, k] + a[i, k] dt, 0)
    x[i, k+1] = x[i, k] + v[i, k+1] dt
    s[i, k+1] = x[i-1, k+1] - x[i, k+1]

Positions are in a frame without vehicle lengths (``s[i] = x[i-1] - x[i]``): in a platoon of
vehicles of length ``L`` the front bumper of vehicle ``i`` is at ``x[i] - i L``. The gap is
integrated from the displacements, ``s[i, k+1] = s[i, k] + (x[i-1, k+1] - x[i-1, k]) - v[i, k+1] dt``,
as ``rollout`` integrates it from ``EventBatch.dx_lead``: the same in exact arithmetic, and accurate
in float32 whatever the distance travelled. Positions are accumulated in float64 at the end. A
collision is ``s <= 0`` at any simulated sample (after the warm-up, the hold included); a collided
follower keeps driving. Nothing is read back from the device inside the step loop.

Prefix property: a follower never depends on the vehicles behind it, so the first ``n`` followers
of a platoon of ``N > n`` are the platoon of ``n``. Bit for bit this needs every follower to be
computed by the same machine code whatever ``N``. Elementwise CPU kernels compute the tail of a
tensor that does not fill a pair of vector registers with scalar code, whose ``pow`` differs from
the vectorised one in the last bit (1-2 % of the IDM outputs at highway speeds on AVX2); the update
``v + a dt`` nearly always rounds such a difference away, but not always, and the block size
depends on the CPU. The model is therefore called on a batch padded to a multiple of
:data:`BATCH_MULTIPLE` followers (no measurable cost); the extra followers drive behind the platoon
and are dropped. Kernels whose blocking depends on the batch size (matrix products of networks)
can still differ in the last bit; P2 takes both platoon sizes from one simulation, so its numbers
do not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from trafficloop.data.batch import A_PREV, N_CHANNELS, WARMUP, S, V
from trafficloop.data.schema import DT
from trafficloop.models.base import ACC_CLIP, CFModel
from trafficloop.protocols.scenarios import HOLD, positions, with_hold

#: the follower batch of every model call is padded to a multiple of this size (prefix property)
BATCH_MULTIPLE = 64


@torch.jit.script
def platoon_advance(
    a: torch.Tensor,
    s: torch.Tensor,
    v: torch.Tensor,
    dx_lead: torch.Tensor,
    v_lead_next: torch.Tensor,
    low: float,
    high: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of all followers: the next rows ``(s, dv, v, a_prev)`` ``[M, 4]`` and the mask of
    non-finite outputs.

    ``dx_lead`` and ``v_lead_next`` (0-d) are the displacement of the prescribed leader over the
    step and its next speed; row ``i > 0`` follows row ``i - 1``. The clip and the non-finite rule
    are those of :func:`trafficloop.sim.rollout.advance`.
    """
    bad = ~torch.isfinite(a)
    a = torch.where(bad, torch.full_like(a, low), a).clamp(low, high)
    v_new = (v + a * dt).clamp_min(0.0)
    step = v_new * dt
    s_new = s + torch.cat([dx_lead.reshape(1), step[:-1]]) - step
    dv_new = v_new - torch.cat([v_lead_next.reshape(1), v_new[:-1]])
    row = torch.stack([s_new, dv_new, v_new, (v_new - v) / dt], dim=-1)
    return row, bad


@dataclass
class PlatoonResult:
    """Trajectories on the CPU (float64); row 0 of ``x``, ``v``, ``a`` is the leader."""

    x: np.ndarray  # [N + 1, T] positions in the frame without vehicle lengths
    v: np.ndarray  # [N + 1, T]
    a: np.ndarray  # [N + 1, T] acceleration realised over k -> k+1, zero where no step is simulated
    s: np.ndarray  # [N, T] gap of follower i (row i - 1) behind vehicle i - 1
    collided: np.ndarray  # [N] bool
    first_collision: np.ndarray  # [N] sample index, -1 without collision
    manoeuvre_start: int
    n_invalid: int  # non-finite model outputs of the N followers (replaced by the lower clip)
    invalid: np.ndarray  # [N] the same per follower
    warmup: int = WARMUP
    dt: float = DT

    @property
    def n_followers(self) -> int:
        return int(self.s.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.s.shape[1])

    def prefix(self, n: int) -> "PlatoonResult":
        """The leader and the first ``n`` followers (the platoon of ``n``)."""
        if not 1 <= n <= self.n_followers:
            raise ValueError(f"prefix of {n} followers out of a platoon of {self.n_followers}")
        invalid = self.invalid[:n]
        return PlatoonResult(
            self.x[: n + 1],
            self.v[: n + 1],
            self.a[: n + 1],
            self.s[:n],
            self.collided[:n],
            self.first_collision[:n],
            self.manoeuvre_start,
            int(invalid.sum()),
            invalid,
            self.warmup,
            self.dt,
        )


@torch.no_grad()
def simulate_platoon(
    model: CFModel,
    v_lead: np.ndarray,
    n_vehicles: int,
    s_init: float,
    hold: float = HOLD,
    clip: tuple[float, float] = ACC_CLIP,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PlatoonResult:
    """Drive ``n_vehicles`` followers behind the leader ``hold + v_lead`` (see the module docstring)."""
    n = int(n_vehicles)
    if n < 1:
        raise ValueError(f"a platoon needs at least one follower, got {n_vehicles}")
    if not (np.isfinite(s_init) and s_init > 0):
        raise ValueError(f"the starting gap must be positive, got {s_init}")
    v_lead = np.asarray(v_lead, dtype=np.float64)
    if v_lead.ndim != 1 or len(v_lead) == 0 or not np.all(np.isfinite(v_lead)):
        raise ValueError("v_lead must be a non-empty finite 1-d speed series")
    v_full, start = with_hold(v_lead, hold, DT)
    n_steps, warmup, history = len(v_full), WARMUP, model.history
    if warmup + 1 < history:
        raise ValueError(f"warm-up of {warmup} samples is shorter than the history {history} of {model.name}")
    x_lead = positions(v_full, 0.0, DT)
    model = model.to(device).eval()

    m = -(-n // BATCH_MULTIPLE) * BATCH_MULTIPLE
    v0 = float(v_full[0])
    state = torch.zeros((m, n_steps, N_CHANNELS), dtype=dtype, device=device)
    state[:, : warmup + 1] = torch.tensor([float(s_init), 0.0, v0, 0.0], dtype=dtype, device=device)
    dx_lead = torch.as_tensor(np.diff(x_lead), dtype=dtype, device=device)
    v_next = torch.as_tensor(v_full, dtype=dtype, device=device)
    model.begin(state[:, warmup - WARMUP + 1 : warmup + 1])
    invalid = torch.zeros(m, dtype=torch.long, device=device)
    low, high = float(clip[0]), float(clip[1])
    for k in range(warmup, n_steps - 1):
        a = model(state[:, k - history + 1 : k + 1])
        row, bad = platoon_advance(a, state[:, k, S], state[:, k, V], dx_lead[k], v_next[k + 1], low, high, DT)
        state[:, k + 1] = row
        invalid += bad

    sim = state[:n].to(device="cpu", dtype=torch.float64).numpy()
    counts = invalid[:n].cpu().numpy()
    s, v_f = sim[..., S], sim[..., V]
    a_f = np.zeros((n, n_steps))
    a_f[:, warmup : n_steps - 1] = sim[:, warmup + 1 :, A_PREV]
    a_lead = np.zeros(n_steps)
    a_lead[:-1] = np.diff(v_full) / DT
    x = np.empty((n + 1, n_steps))
    x[0] = x_lead
    x[1:, : warmup + 1] = x_lead[: warmup + 1] - float(s_init) * np.arange(1, n + 1, dtype=np.float64)[:, None]
    x[1:, warmup + 1 :] = x[1:, warmup : warmup + 1] + np.cumsum(v_f[:, warmup + 1 :], axis=1) * DT
    hit = s <= 0.0
    hit[:, : warmup + 1] = False
    collided = hit.any(axis=1)
    first = np.where(collided, hit.argmax(axis=1), -1)
    return PlatoonResult(
        x=x,
        v=np.vstack([v_full, v_f]),
        a=np.vstack([a_lead, a_f]),
        s=s,
        collided=collided,
        first_collision=first,
        manoeuvre_start=start,
        n_invalid=int(counts.sum()),
        invalid=counts,
    )
