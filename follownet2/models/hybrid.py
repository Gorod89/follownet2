"""Physics-hybrid networks (model contract, section 2): PIDL, PERL, stability-regularised MLP.

All three are :class:`~follownet2.models.network.NetworkModel` s: physical units at the
interface, normalised units inside. Their physics is the benchmark IDM
(:func:`~follownet2.models.mechanistic.idm_acc`, ``delta = 4``) with parameters inside
``IDM.bounds``, reached from an unconstrained space through a sigmoid.
"""

from __future__ import annotations

import time
from typing import Iterator, Sequence

import torch
from torch import Tensor, nn

from follownet2.data.batch import DV, S, V
from follownet2.models.base import ACC_CLIP
from follownet2.models.mechanistic import IDM, idm_acc
from follownet2.models.network import NetworkModel, mlp
from follownet2.stability import analytic
from follownet2.train.trainer import TrainConfig, fit_supervised
from follownet2.train.windows import WindowSet

#: instants drawn for the quantiles of the collocation box
QUANTILE_SAMPLE = 1_000_000


def _unconstrained(theta: Tensor, bounds: Tensor) -> Tensor:
    """Logits of parameters inside ``bounds [D, 2]``; inverse of :func:`_bounded`."""
    return torch.logit((theta - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]))


def _bounded(raw: Tensor, bounds: Tensor) -> Tensor:
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.sigmoid(raw)


def _idm(rows: Tensor, theta: Tensor) -> Tensor:
    """IDM acceleration of state rows ``[..., C >= 3]`` (channels s, dv, v first)."""
    return idm_acc(rows[..., S], rows[..., DV], rows[..., V], *theta.to(rows.dtype).unbind(-1))


def _rows(windows: WindowSet, index: Tensor) -> Tensor:
    """Current state rows ``[B, 4]`` of the instants ``index``."""
    return windows.state[windows.event[index], windows.step[index]]


def _endless(windows: WindowSet, batch_size: int, generator: torch.Generator) -> Iterator[Tensor]:
    """Minibatches of instants, epoch after epoch."""
    if len(windows) == 0:
        raise ValueError("no training windows")
    while True:
        yield from windows.batches(batch_size, generator)


@torch.no_grad()
def _physics_rmse(windows: WindowSet, theta: Tensor, cfg: TrainConfig) -> float:
    """One-step RMSE of the IDM alone [m/s^2] on the batches that :func:`validate` scores."""
    gen = torch.Generator().manual_seed(cfg.seed)
    total, count = 0.0, 0
    for index in windows.batches(cfg.batch_size, gen, limit=cfg.val_batches):
        total += float(torch.sum((_idm(_rows(windows, index), theta) - windows.target(index)[:, 0]) ** 2))
        count += len(index)
    return (total / max(count, 1)) ** 0.5


class _MemorylessMLP(NetworkModel):
    """MLP (tanh) on the current row ``(s, dv, v)``: the network of PIDL and of the stability MLP."""

    memoryless = True

    def __init__(self, hidden: Sequence[int]) -> None:
        super().__init__(history=1)
        self.kwargs = {"hidden": [int(h) for h in hidden]}
        self.net = mlp([self.n_inputs, *self.kwargs["hidden"], 1])

    def core(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1)).squeeze(-1)


class PIDL(_MemorylessMLP):
    """Physics-informed deep learning (Mo, Shi, Di 2021): a network ``a = N(s, dv, v)`` fitted to
    the data and to the IDM, whose parameters are learned jointly.

    Loss, in the normalised units of the network::

        alpha * mean((N - a_recorded)^2) + (1 - alpha) * mean((N - IDM_theta)^2 at collocation states)

    ``n_collocation`` states are drawn in every training step, uniformly in the box between
    the 1 % and 99 % quantiles of ``(s, dv, v)`` of the training windows (taken once at the
    start of :meth:`fit` and kept in the buffer ``collocation_box``). ``theta`` is an
    ``nn.Parameter`` in an unconstrained space, mapped into ``IDM.bounds`` by a sigmoid and
    initialised at ``IDM.default``; it enters the physics term only. The prediction is the
    network alone: ``alpha = 1`` is a plain network, ``alpha = 0`` a network that copies the
    IDM and never reads a recorded acceleration (``theta`` then keeps its initial value:
    without data it is not identified and would only drift after the network).

    Differences from the paper: the IDM is clipped to the protocol range ``[-8, 4]`` m/s^2 in
    the physics term (in the corners of the box, small gap at high speed, it demands hundreds
    of m/s^2 of braking, which would dominate the loss); the collocation box is the quantile
    box of the training windows; the IDM parameters share the optimiser (AdamW), learning
    rate, gradient clipping and early stopping on the one-step validation error of the
    network (the shared trainer).
    """

    name = "pidl"

    def __init__(self, hidden: Sequence[int] = (64, 64), alpha: float = 0.7, n_collocation: int = 4096) -> None:
        super().__init__(hidden)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        self.alpha, self.n_collocation = float(alpha), int(n_collocation)
        self.kwargs.update(alpha=self.alpha, n_collocation=self.n_collocation)
        bounds = IDM.bounds_tensor(torch.float32)
        self.register_buffer("idm_bounds", bounds, persistent=False)
        self.idm_raw = nn.Parameter(_unconstrained(torch.tensor(IDM.default), bounds))
        self.register_buffer("collocation_box", torch.zeros(2, 3))  # rows: low, high of (s, dv, v)
        self._rng: torch.Generator | None = None  # collocation draws, seeded in fit

    def idm_theta(self) -> Tensor:
        """Current IDM parameters ``(v0, T, s0, a, b)``, differentiable."""
        return _bounded(self.idm_raw, self.idm_bounds)

    def physics_parameters(self) -> dict[str, float]:
        return dict(zip(IDM.param_names, self.idm_theta().detach().cpu().tolist()))

    def physics_loss(self) -> Tensor:
        """Squared difference between network and clipped IDM at fresh collocation states."""
        low, high = self.collocation_box
        u = torch.rand(self.n_collocation, 3, generator=self._rng, device=low.device, dtype=low.dtype)
        states = low + (high - low) * u
        theta = self.idm_theta()
        if self.alpha == 0.0:  # without data nothing identifies theta: it would only chase the network
            theta = theta.detach()
        target = _idm(states, theta).clamp(*ACC_CLIP) / self.a_std
        hist = torch.cat([states, torch.zeros_like(states[:, :1])], dim=-1).unsqueeze(1)  # a_prev = 0, unused
        return torch.mean((self.core(self.inputs(hist)) - target) ** 2)

    def training_loss(self, windows: WindowSet, index: Tensor) -> Tensor:
        loss = torch.zeros((), dtype=self.a_std.dtype, device=self.a_std.device)
        if self.alpha > 0.0:
            loss = loss + self.alpha * super().training_loss(windows, index)
        if self.alpha < 1.0:
            loss = loss + (1.0 - self.alpha) * self.physics_loss()
        return loss

    def fit(self, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
        gen = torch.Generator().manual_seed(cfg.seed)
        index = torch.randperm(len(train), generator=gen)[:QUANTILE_SAMPLE].to(train.device)
        rows = _rows(train, index)[:, :3]
        levels = torch.tensor([0.01, 0.99], dtype=rows.dtype, device=rows.device)
        self.collocation_box.copy_(torch.quantile(rows, levels, dim=0))
        self._rng = torch.Generator(device=train.device).manual_seed(cfg.seed)
        try:
            log = fit_supervised(self, train, val, cfg)
        finally:
            self._rng = None
        log["physics"] = {"params": self.physics_parameters()}
        return log


class PERL(NetworkModel):
    """Physics-enhanced residual learning (Long et al. 2023): ``a = IDM(s, dv, v; theta) + R(hist)``.

    ``R`` is an LSTM over the last ``history`` rows with a linear head initialised at zero,
    so that the model equals the IDM until ``R`` is trained. :meth:`fit` has two stages:

    1. :meth:`fit_physics`: least squares of the one-step IDM acceleration on the training
       windows by Adam (``physics_steps`` minibatches, learning rate ``physics_lr`` with a
       cosine decay) on unconstrained parameters mapped into ``IDM.bounds`` by a sigmoid,
       starting from ``IDM.default``; ``theta`` is a buffer afterwards;
    2. the supervised trainer fits ``R`` to the recorded acceleration minus the IDM.

    The log holds both stages; ``physics`` has the parameters and the one-step RMSE of the
    IDM alone on training and validation windows.

    Differences from the paper: the physics model is the benchmark IDM calibrated by Adam on
    the one-step squared error of the acceleration; the residual is trained on the one-step
    acceleration only (longer horizons come from the closed loop of the protocols) with the
    shared trainer (AdamW, early stopping on the one-step validation error).
    """

    name = "perl"

    def __init__(
        self,
        history: int = 10,
        use_a_prev: bool = False,
        hidden: int = 64,
        physics_steps: int = 500,
        physics_lr: float = 0.05,
    ) -> None:
        super().__init__(history, use_a_prev)
        self.physics_steps, self.physics_lr = int(physics_steps), float(physics_lr)
        self.kwargs.update(hidden=int(hidden), physics_steps=self.physics_steps, physics_lr=self.physics_lr)
        self.lstm = nn.LSTM(self.n_inputs, int(hidden), batch_first=True)
        self.head = nn.Linear(int(hidden), 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("theta", torch.tensor(IDM.default))  # (v0, T, s0, a, b), set by fit_physics

    def core(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

    def physics(self, hist: Tensor) -> Tensor:
        """IDM acceleration of the last row of ``hist`` with the parameters of the first stage,
        inside the acceleration range of the benchmark."""
        return _idm(hist[..., -1, :], self.theta).clamp(*ACC_CLIP)

    def forward(self, hist: Tensor) -> Tensor:
        return self.physics(hist) + super().forward(hist)

    def training_loss(self, windows: WindowSet, index: Tensor) -> Tensor:
        hist = windows.history(index)
        target = (windows.target(index)[:, 0] - self.physics(hist)) / self.a_std
        return torch.mean((self.core(self.inputs(hist)) - target) ** 2)

    def fit_physics(self, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
        """First stage: IDM parameters by least squares of the one-step acceleration."""
        started = time.time()
        self.to(train.device)
        gen = torch.Generator().manual_seed(cfg.seed)
        bounds = IDM.bounds_tensor(train.state.dtype, train.device)
        raw = _unconstrained(torch.as_tensor(IDM.default, dtype=bounds.dtype, device=bounds.device), bounds)
        raw.requires_grad_(True)
        optimiser = torch.optim.Adam([raw], lr=self.physics_lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(self.physics_steps, 1))
        for _, index in zip(range(self.physics_steps), _endless(train, cfg.batch_size, gen)):
            # clipped like every rollout: a few states with a small gap must not decide the fit
            error = _idm(_rows(train, index), _bounded(raw, bounds)).clamp(*ACC_CLIP) - train.target(index)[:, 0]
            loss = torch.mean(error**2)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
        self.theta.copy_(_bounded(raw.detach(), bounds))
        return {
            "params": dict(zip(IDM.param_names, self.theta.cpu().tolist())),
            "train_rmse_a": _physics_rmse(train, self.theta, cfg),
            "val_rmse_a": _physics_rmse(val, self.theta, cfg),
            "steps": self.physics_steps,
            "seconds": round(time.time() - started, 1),
        }

    def fit(self, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
        physics = self.fit_physics(train, val, cfg)
        log = fit_supervised(self, train, val, cfg)
        log["seconds"] = round(log["seconds"] + physics["seconds"], 1)
        log["physics"] = physics
        return log


class StabilityRegularisedMLP(_MemorylessMLP):
    """Memoryless MLP ``a = f(s, dv, v)`` trained with the analytic string-stability penalty.

    Loss: the one-step squared error in normalised units plus ``weight`` times
    :func:`~follownet2.stability.analytic.stability_penalty` of the current network, evaluated
    once per training step at ``penalty_speeds`` equilibrium speeds evenly spaced from 5 to
    30 m/s with the margin ``margin`` [1/s^2]. ``weight = 0`` is the plain memoryless MLP. The
    log of :meth:`fit` reports the share of string-unstable equilibria of the trained network
    (``unstable_fraction``) and its final penalty.
    """

    name = "stability_mlp"

    def __init__(
        self,
        hidden: Sequence[int] = (64, 64),
        weight: float = 1.0,
        margin: float = 0.0,
        penalty_speeds: int = 26,
        neighbourhood: float = 0.0,
    ) -> None:
        super().__init__(hidden)
        self.weight, self.margin, self.neighbourhood = float(weight), float(margin), float(neighbourhood)
        self.kwargs.update(
            weight=self.weight, margin=self.margin, penalty_speeds=int(penalty_speeds), neighbourhood=self.neighbourhood
        )
        self.speeds = tuple(torch.linspace(5.0, 30.0, int(penalty_speeds), dtype=torch.float64).tolist())

    def stability_penalty(self) -> Tensor:
        """Penalty of the current network at the penalty speeds, differentiable in its parameters.

        ``neighbourhood > 0`` adds the margin at perturbed states around every equilibrium
        (finite-perturbation proxy, :func:`~follownet2.stability.analytic.stability_penalty`).
        """
        return analytic.stability_penalty(self, self.speeds, self.margin, self.neighbourhood)

    def unstable_fraction(self) -> float:
        """Share of the equilibria at the penalty speeds with a negative string-stability margin."""
        return analytic.unstable_fraction(self, self.speeds)

    def training_loss(self, windows: WindowSet, index: Tensor) -> Tensor:
        loss = super().training_loss(windows, index)
        if self.weight == 0.0:
            return loss
        return loss + self.weight * self.stability_penalty()

    def fit(self, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
        log = fit_supervised(self, train, val, cfg)
        log["unstable_fraction"] = self.unstable_fraction()
        log["stability_penalty"] = float(self.stability_penalty().detach())
        return log
