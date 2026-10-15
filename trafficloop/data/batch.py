"""Padded tensor batches of events for batched one-step evaluation and rollouts.

State history convention (binding for every model and protocol)::

    hist[..., H, 4]   channels (s, dv, v, a_prev), rows ordered in time, last row = now

``a_prev[k] = a[k-1]`` is the last acceleration known at ``t[k]`` (``a_prev[0] = a[0]`` is a
placeholder: no protocol uses index 0 as a prediction instant). A row therefore holds only
information available at its own time, and the model output is the acceleration applied over
the following step, ``v[k+1] = v[k] + a[k]*dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from trafficloop.data.schema import DT, Event

S, DV, V, A_PREV = 0, 1, 2, 3
N_CHANNELS = 4
# samples of observed history given to every model before the first prediction (3 s)
WARMUP = 30


@dataclass
class EventBatch:
    event_ids: list[str]
    follower_ids: list[str]
    lengths: torch.Tensor  # [E] long
    mask: torch.Tensor  # [E, T] bool
    s: torch.Tensor  # [E, T]
    v: torch.Tensor
    a: torch.Tensor
    v_lead: torch.Tensor
    # leader displacement over the step k -> k+1; the gap is integrated from it, which keeps
    # rollouts accurate in float32 whatever the magnitude of the source coordinates
    dx_lead: torch.Tensor
    dt: float = DT

    @property
    def n_events(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.s.shape[1])

    @property
    def device(self) -> torch.device:
        return self.s.device

    @property
    def dv(self) -> torch.Tensor:
        return self.v - self.v_lead

    @property
    def a_prev(self) -> torch.Tensor:
        return torch.cat([self.a[:, :1], self.a[:, :-1]], dim=1)

    def state(self) -> torch.Tensor:
        """Observed state sequence ``[E, T, 4]`` with channels (s, dv, v, a_prev)."""
        return torch.stack([self.s, self.dv, self.v, self.a_prev], dim=-1)

    def to(self, device=None, dtype=None) -> "EventBatch":
        def move(x: torch.Tensor) -> torch.Tensor:
            return x.to(device=device, dtype=dtype if x.is_floating_point() else None)

        return EventBatch(
            self.event_ids,
            self.follower_ids,
            self.lengths.to(device),
            self.mask.to(device),
            move(self.s),
            move(self.v),
            move(self.a),
            move(self.v_lead),
            move(self.dx_lead),
            self.dt,
        )

    @classmethod
    def from_events(
        cls,
        events: Sequence[Event],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        start: Sequence[int] | None = None,
        length: int | None = None,
    ) -> "EventBatch":
        """Pad events to a common length.

        ``start`` and ``length`` cut a window ``[start, start + length)`` out of every event
        (windows running past the end of an event are shortened). Padding repeats the last
        valid sample and lets the leader continue at constant speed.
        """
        events = list(events)
        starts = np.zeros(len(events), dtype=np.int64) if start is None else np.asarray(start, dtype=np.int64)
        lengths = np.array(
            [len(ev) - s0 if length is None else min(length, len(ev) - s0) for ev, s0 in zip(events, starts)],
            dtype=np.int64,
        )
        if len(events) == 0 or lengths.min() < 2:
            raise ValueError("every event (window) needs at least 2 samples")
        n_max = int(lengths.max())
        arrays = {name: np.empty((len(events), n_max), dtype=np.float64) for name in ("s", "v", "a", "v_lead", "dx")}
        for i, (ev, s0, n) in enumerate(zip(events, starts, lengths)):
            sl = slice(s0, s0 + n)
            for name, src in (("s", ev.s), ("v", ev.v), ("a", ev.a), ("v_lead", ev.v_lead)):
                arrays[name][i, :n] = src[sl]
                arrays[name][i, n:] = src[s0 + n - 1]
            arrays["dx"][i, : n - 1] = np.diff(ev.x_lead[sl])
            arrays["dx"][i, n - 1 :] = ev.v_lead[s0 + n - 1] * DT
        mask = np.arange(n_max)[None, :] < lengths[:, None]
        as_tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)  # noqa: E731
        return cls(
            event_ids=[ev.event_id for ev in events],
            follower_ids=[ev.follower_id for ev in events],
            lengths=torch.as_tensor(lengths, device=device),
            mask=torch.as_tensor(mask, device=device),
            s=as_tensor(arrays["s"]),
            v=as_tensor(arrays["v"]),
            a=as_tensor(arrays["a"]),
            v_lead=as_tensor(arrays["v_lead"]),
            dx_lead=as_tensor(arrays["dx"]),
        )


def length_buckets(lengths: Sequence[int], batch_size: int) -> list[np.ndarray]:
    """Indices of events grouped into batches of similar length (longest first)."""
    order = np.argsort(-np.asarray(lengths), kind="stable")
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
