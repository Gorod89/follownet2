"""Training data: prediction instants of a set of events.

A *window* is one prediction instant ``(event, k)``: the model sees the observed history up
to ``t[k]`` and the target is ``a[k]``, the acceleration applied over the next step. The
instants are the ones the protocols score (``k >= WARMUP``), so a model is trained on what it
is evaluated on. Nothing is copied per window: the events stay in one padded tensor on the
device and the histories are gathered by index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from trafficloop.data.batch import WARMUP, EventBatch
from trafficloop.data.schema import Event


@dataclass
class WindowSet:
    batch: EventBatch
    state: torch.Tensor  # [E, T, 4] observed (s, dv, v, a_prev)
    event: torch.Tensor  # [N] event of every instant
    step: torch.Tensor  # [N] sample index k of every instant
    driver: torch.Tensor  # [E] index of the follower of every event
    driver_names: list[str]
    horizon: int  # steps after k that lie inside the event for every instant

    @classmethod
    def from_events(
        cls,
        events: Sequence[Event],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        horizon: int = 1,
        stride: int = 1,
        warmup: int = WARMUP,
    ) -> "WindowSet":
        """Instants ``warmup <= k <= len - 1 - horizon`` of every event, every ``stride`` samples.

        ``horizon`` steps after every instant lie inside its event: ``a[k .. k + horizon - 1]``
        are targets and the states up to ``k + horizon`` exist. Events too short for one
        instant contribute nothing.
        """
        events = list(events)
        batch = EventBatch.from_events(events, device=device, dtype=dtype)
        names = sorted({ev.follower_id for ev in events})
        index = {name: i for i, name in enumerate(names)}
        event, step = [], []
        for e, ev in enumerate(events):
            k = torch.arange(warmup, max(len(ev) - horizon, warmup), stride)
            event.append(torch.full_like(k, e))
            step.append(k)
        return cls(
            batch=batch,
            state=batch.state(),
            event=torch.cat(event).to(device),
            step=torch.cat(step).to(device),
            driver=torch.tensor([index[ev.follower_id] for ev in events], device=device),
            driver_names=names,
            horizon=horizon,
        )

    def __len__(self) -> int:
        return int(self.event.shape[0])

    @property
    def device(self) -> torch.device:
        return self.state.device

    def history(self, index: torch.Tensor, rows: int = WARMUP) -> torch.Tensor:
        """Observed histories ``[B, rows, 4]`` ending at the instants ``index``."""
        k = self.step[index].unsqueeze(-1) + torch.arange(1 - rows, 1, device=self.device)
        return self.state[self.event[index].unsqueeze(-1), k]

    def _ahead(self, values: torch.Tensor, index: torch.Tensor, first: int, count: int) -> torch.Tensor:
        k = self.step[index].unsqueeze(-1) + torch.arange(first, first + count, device=self.device)
        return values[self.event[index].unsqueeze(-1), k]

    def target(self, index: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """Recorded accelerations ``a[k .. k + horizon - 1]``, shape ``[B, horizon]``."""
        self._check(horizon)
        return self._ahead(self.batch.a, index, 0, horizon)

    def future(self, index: torch.Tensor, horizon: int) -> dict[str, torch.Tensor]:
        """What a closed-loop rollout over ``horizon`` steps needs and is compared with.

        ``dx_lead[:, h]`` and ``v_lead[:, h]`` drive step ``h`` (leader displacement over the
        step and leader speed at its end); ``s``, ``v`` and ``a`` are the recorded follower at
        the end of the step and the acceleration over it. All ``[B, horizon]``.
        """
        self._check(horizon)
        return {
            "dx_lead": self._ahead(self.batch.dx_lead, index, 0, horizon),
            "v_lead": self._ahead(self.batch.v_lead, index, 1, horizon),
            "s": self._ahead(self.batch.s, index, 1, horizon),
            "v": self._ahead(self.batch.v, index, 1, horizon),
            "a": self._ahead(self.batch.a, index, 0, horizon),
        }

    def _check(self, horizon: int) -> None:
        if horizon > self.horizon:
            raise ValueError(f"the windows were built for a horizon of {self.horizon} steps, {horizon} asked")

    def driver_of(self, index: torch.Tensor) -> torch.Tensor:
        return self.driver[self.event[index]]

    def batches(
        self, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True, limit: int | None = None
    ) -> Iterator[torch.Tensor]:
        """Index tensors of at most ``batch_size`` instants; ``limit`` caps the number of batches."""
        n = len(self)
        order = torch.randperm(n, generator=generator).to(self.device) if shuffle else torch.arange(n, device=self.device)
        for i, start in enumerate(range(0, n, batch_size)):
            if limit is not None and i >= limit:
                return
            yield order[start : start + batch_size]

    def statistics(self, sample: int = 1_000_000, seed: int = 0) -> dict[str, torch.Tensor]:
        """Mean and standard deviation of the four input channels and of the target."""
        gen = torch.Generator().manual_seed(seed)
        index = torch.randperm(len(self), generator=gen)[:sample].to(self.device)
        rows = self.state[self.event[index], self.step[index]]
        target = self.batch.a[self.event[index], self.step[index]]
        return {
            "x_mean": rows.mean(0),
            "x_std": rows.std(0).clamp_min(1e-6),
            "a_mean": target.mean(),
            "a_std": target.std().clamp_min(1e-6),
        }
