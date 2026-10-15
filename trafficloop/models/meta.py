"""Meta-learned LSTM follower that adapts its output layer to the event at hand (MetaFollower style).

Chen et al. (2024), "MetaFollower: Adaptable personalized autonomous car following", train a
car-following network by model-agnostic meta-learning (MAML, Finn, Abbeel, Levine 2017) on
the events of many drivers, so that a few trajectories of a new driver suffice to adapt it
to that driver; their network combines an LSTM with the IDM.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict

import torch
from torch import nn

from trafficloop.data.batch import A_PREV
from trafficloop.models.network import NetworkModel
from trafficloop.train.windows import WindowSet

CHUNK = 1 << 16  # histories per pass of the LSTM; bounds the memory of long evaluation batches


class MAMLLSTM(NetworkModel):
    """LSTM body shared by all drivers and a linear head adapted to every event.

    The body is an LSTM of ``hidden`` units over the last ``history`` normalised rows; the
    head is linear in its last hidden state, and its weight and bias are the meta-learned
    initialisation. :meth:`begin` adapts the head separately for each event of the observed
    warm-up: the support pairs are (rows ``j - history + 1 .. j``, ``a[j]``) for every ``j``
    whose target lies inside the warm-up (``j = 9 .. 28`` for 30 rows; ``a[j]`` is the
    ``a_prev`` of row ``j + 1``), and the head takes ``inner_steps`` gradient steps of the mean
    squared error in normalised units. The step size of a task is ``inner_lr / lambda_max``,
    with ``lambda_max`` the largest eigenvalue of the Hessian ``(2 / n) X^T X`` of this error
    (``X``: the support features and a column of ones), so that every step descends for
    ``inner_lr < 2`` whatever the scale of the features. The gradient is linear in the head,
    so the steps are computed in closed form, batched over the events. The adapted heads are
    kept until the next :meth:`begin` and applied to all histories of their event; without
    them, or when a call has another number of events than the last :meth:`begin`, the
    meta-learned head is used.

    Meta-training (:meth:`fit`): a task is a driver. Every meta-step draws ``meta_batch``
    drivers; the support of a driver is one block of ``n_support`` consecutive instants of
    one of its events, like the warm-up at test time, and the query ``n_query`` of its other
    instants, drawn without replacement (drivers without such a block or with fewer than
    ``n_support + n_query`` instants are left out). The head is adapted on the support pairs
    and the query error of the adapted head is minimised. Every ``val_every`` meta-steps the
    one-step error with adaptation is measured on the validation drivers (support: the first
    ``n_support`` instants of a driver, query: the others), and the best model is kept. An
    "epoch" of the log is one such validation.

    Differences from the paper: only the output layer is adapted, not the whole network
    (the inner loop of ANIL, Raghu et al. 2020), with closed-form gradient steps whose size is
    scaled by the largest curvature of the support error of each task; the outer update is
    first order: the adapted head is treated as ``head - sum of the inner steps`` with the
    steps detached, so the outer gradient reaches the body through the query features only
    and the head initialisation through the identity; there is no IDM component; a task is a
    driver whose support is a block of consecutive instants, and at test time the support is
    the observed warm-up of the event itself, not other trajectories of the driver.
    """

    name = "maml_lstm"
    adaptive = True

    def __init__(
        self,
        history: int = 10,
        use_a_prev: bool = False,
        hidden: int = 64,
        inner_steps: int = 3,
        inner_lr: float = 0.5,
        meta_batch: int = 32,
        n_support: int = 20,
        n_query: int = 40,
        meta_steps: int = 2000,
        val_every: int = 100,
    ) -> None:
        super().__init__(history, use_a_prev)
        self.inner_steps, self.inner_lr = int(inner_steps), float(inner_lr)
        self.meta_batch, self.n_support, self.n_query = int(meta_batch), int(n_support), int(n_query)
        self.meta_steps, self.val_every = int(meta_steps), int(val_every)
        self.kwargs.update(
            hidden=int(hidden), inner_steps=self.inner_steps, inner_lr=self.inner_lr, meta_batch=self.meta_batch,
            n_support=self.n_support, n_query=self.n_query, meta_steps=self.meta_steps, val_every=self.val_every,
        )
        self.body = nn.LSTM(self.n_inputs, int(hidden), batch_first=True)
        self.head = nn.Linear(int(hidden), 1)
        #: adapted heads ``[E, hidden + 1]`` (weights, bias) of the events of the last :meth:`begin`
        self.adapted: torch.Tensor | None = None

    # ------------------------------------------------------------------ model
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Last hidden state ``[B, hidden]`` of the body for normalised inputs ``[B, H, C]``."""
        return torch.cat([self.body(part)[1][0][-1] for part in x.split(CHUNK)])

    def head_vector(self) -> torch.Tensor:
        """The meta-learned head as one vector ``[hidden + 1]`` (weights, bias)."""
        return torch.cat([self.head.weight[0], self.head.bias])

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)

    def adapt(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Heads ``[T, hidden + 1]`` after the inner steps on the pairs ``features [T, n, hidden]``, ``target [T, n]``.

        The step of task ``t`` is ``inner_lr / lambda_max[t]`` times the gradient of its mean
        squared error. The steps are constants (first order): the result depends on the
        meta-learned head through the identity and on nothing else that has a gradient.
        """
        start = self.head_vector()
        with torch.no_grad():
            x = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
            hessian = x.transpose(1, 2) @ x * (2.0 / x.shape[1])
            # the CUDA solver handles matrices larger than 32 x 32 one task at a time (seconds per
            # thousand tasks): the CPU solves the whole batch at once, the result moves back
            lambda_max = torch.linalg.eigvalsh(hessian.cpu())[:, -1:].to(hessian.device)
            size = self.inner_lr / lambda_max  # the ones column makes lambda_max >= 2
            head = start.detach().expand(x.shape[0], -1)
            total = torch.zeros_like(head)
            for _ in range(self.inner_steps):
                residual = torch.einsum("tnf,tf->tn", x, head) - target
                step = size * torch.einsum("tnf,tn->tf", x, residual) * (2.0 / x.shape[1])
                head = head - step
                total = total + step
        return start - total

    @staticmethod
    def apply_heads(features: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """Outputs ``[T, n]`` of the heads ``[T, hidden + 1]`` on the features ``[T, n, hidden]``."""
        return torch.einsum("tnf,tf->tn", features, heads[:, :-1]) + heads[:, -1:]

    def begin(self, hist: torch.Tensor) -> None:
        """Adapt the head to every event on the (history, acceleration) pairs of its observed warm-up."""
        hist = hist.to(self.x_mean.dtype)
        windows = hist.unfold(-2, self.history, 1).transpose(-1, -2)[:, :-1]  # [E, rows - history, history, 4]
        target = hist[:, self.history :, A_PREV] / self.a_std  # a[j] is the a_prev of row j + 1
        with torch.no_grad():
            features = self.features(self.inputs(windows)).reshape(*windows.shape[:2], -1)
        self.adapted = self.adapt(features, target)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        batch = hist.shape[:-2]
        features = self.features(self.inputs(hist.to(self.x_mean.dtype)))
        if self.adapted is not None and len(batch) > 0 and batch[0] == self.adapted.shape[0]:
            out = self.apply_heads(features.reshape(batch[0], -1, features.shape[-1]), self.adapted)
        else:
            out = self.head(features)
        return (out * self.a_std).reshape(batch).to(hist.dtype)

    # --------------------------------------------------------------- training
    @staticmethod
    def _groups(windows: WindowSet) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Instants sorted by driver (index order inside a driver), their rank inside the driver, the counts."""
        driver = windows.driver_of(torch.arange(len(windows), device=windows.device))
        order = torch.argsort(driver, stable=True)
        counts = torch.bincount(driver, minlength=len(windows.driver_names))
        start = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(windows), device=windows.device) - start[driver[order]]
        return order, rank, counts

    def _tasks(self, windows: WindowSet) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per usable driver: its instants (in index order) and the first positions of its support blocks.

        A block is ``n_support`` consecutive instants of one event; the instants of an event
        are contiguous in the index, so a block lies in one event when its ends do.
        """
        order, _, counts = self._groups(windows)
        tasks = []
        for instants, events in zip(order.split(counts.tolist()), windows.event[order].split(counts.tolist())):
            if len(instants) < self.n_support + self.n_query:
                continue
            starts = torch.nonzero(events[: len(events) - self.n_support + 1] == events[self.n_support - 1 :])
            if len(starts) > 0:
                tasks.append((instants.cpu(), starts.squeeze(-1).cpu()))
        return tasks

    def _draw_tasks(self, tasks: list[tuple[torch.Tensor, torch.Tensor]], gen: torch.Generator) -> torch.Tensor:
        """Instants ``[T, n_support + n_query]`` of ``meta_batch`` drivers: a support block, then the query."""
        rows = []
        for d in torch.randperm(len(tasks), generator=gen)[: self.meta_batch].tolist():
            instants, starts = tasks[d]
            first = int(starts[torch.randint(len(starts), (1,), generator=gen)])
            others = torch.cat([instants[:first], instants[first + self.n_support :]])
            query = others[torch.randperm(len(others), generator=gen)[: self.n_query]]
            rows.append(torch.cat([instants[first : first + self.n_support], query]))
        return torch.stack(rows)

    @torch.no_grad()
    def _validate(self, val: WindowSet, batch_size: int) -> tuple[float, float]:
        """One-step RMSE [m/s^2] on the query instants of the validation drivers, with and without adaptation."""
        self.eval()
        order, rank, counts = self._groups(val)
        driver = val.driver_of(order)
        eligible = counts[driver] > self.n_support
        support = order[eligible & (rank < self.n_support)].reshape(-1, self.n_support)
        query = order[eligible & (rank >= self.n_support)]
        if len(query) == 0:
            raise ValueError(f"no validation driver has more than n_support={self.n_support} instants")
        row = torch.full_like(counts, -1)
        row[val.driver_of(support[:, 0])] = torch.arange(len(support), device=val.device)
        needed = torch.cat([support.reshape(-1), query])
        features = torch.cat(
            [self.features(self.inputs(val.history(part, self.history))) for part in needed.split(batch_size)]
        )
        target = val.target(needed)[:, 0].to(features.dtype) / self.a_std
        n_s = support.numel()
        heads = self.adapt(features[:n_s].reshape(*support.shape, -1), target[:n_s].reshape(support.shape))
        q_features, q_target = features[n_s:], target[n_s:]
        adapted = self.apply_heads(q_features.unsqueeze(1), heads[row[val.driver_of(query)]]).squeeze(1)
        without = self.head(q_features).squeeze(-1)
        rmse = lambda pred: float(torch.sqrt(torch.mean((pred - q_target) ** 2)) * self.a_std)  # noqa: E731
        return rmse(adapted), rmse(without)

    def fit(self, train: WindowSet, val: WindowSet, cfg) -> dict:
        """First-order meta-training over the drivers of ``train``; keeps the best validation state."""
        gen = torch.Generator().manual_seed(cfg.seed)
        started = time.time()
        self.to(train.device)
        self.adapted = None
        self.set_scale(train.statistics(seed=cfg.seed))
        tasks = self._tasks(train)
        if not tasks:
            raise ValueError(
                f"no training driver has {self.n_support} consecutive instants in one event and "
                f"{self.n_support + self.n_query} instants in all"
            )
        optimiser = torch.optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.meta_steps)
        best, best_without, best_state = float("inf"), float("inf"), copy.deepcopy(self.state_dict())
        history, total, n_steps = [], 0.0, 0
        for step in range(1, self.meta_steps + 1):
            self.train()
            index = self._draw_tasks(tasks, gen).to(train.device)
            features = self.features(self.inputs(train.history(index.reshape(-1), self.history)))
            features = features.reshape(*index.shape, -1)
            target = (train.target(index.reshape(-1))[:, 0].to(features.dtype) / self.a_std).reshape(index.shape)
            heads = self.adapt(features[:, : self.n_support], target[:, : self.n_support])
            predicted = self.apply_heads(features[:, self.n_support :], heads)
            loss = torch.mean((predicted - target[:, self.n_support :]) ** 2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
            n_steps += 1
            if step % self.val_every == 0 or step == self.meta_steps:
                error, without = self._validate(val, cfg.batch_size)
                history.append({"step": step, "train_loss": total / n_steps, "val_rmse_a": error,
                                "val_rmse_a_without_adaptation": without})
                total, n_steps = 0.0, 0
                if error < best - 1e-6:
                    best, best_without, best_state = error, without, copy.deepcopy(self.state_dict())
        self.load_state_dict(best_state)
        self.eval()
        return {
            "best_val_rmse_a": best,
            "val_rmse_a_without_adaptation": best_without,
            "epochs_run": len(history),
            "meta_steps": self.meta_steps,
            "n_tasks": len(tasks),
            "history": history,
            "seconds": round(time.time() - started, 1),
            "n_parameters": sum(p.numel() for p in self.parameters()),
            "config": asdict(cfg),
        }
