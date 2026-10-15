"""Base class of the learned models.

A learned model works in physical units at its interface, ``acc = model(state_history)``,
and in normalised units inside. Subclasses implement :meth:`NetworkModel.core`, which maps
normalised inputs ``[B, H, C]`` to normalised accelerations ``[B]``, and may replace
:meth:`NetworkModel.training_loss` (mixture likelihood, physics term, stability penalty ...).

Inputs. The reference implementations read ``(s, dv, v)`` like the models of the literature.
``use_a_prev`` adds the last acceleration as a fourth channel; such a model can copy the
persistence baseline, which is why both variants exist.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from trafficloop.data.batch import A_PREV, DV, S, V
from trafficloop.models.base import CFModel
from trafficloop.train.windows import WindowSet


class NetworkModel(CFModel):
    family = "network"
    name = "network"

    def __init__(self, history: int = 10, use_a_prev: bool = False) -> None:
        self.history = int(history)
        super().__init__()
        channels = (S, DV, V, A_PREV) if use_a_prev else (S, DV, V)
        self.kwargs: dict[str, Any] = {"history": int(history), "use_a_prev": bool(use_a_prev)}
        self.register_buffer("channels", torch.tensor(channels))
        self.register_buffer("x_mean", torch.zeros(len(channels)))
        self.register_buffer("x_std", torch.ones(len(channels)))
        self.register_buffer("a_std", torch.ones(()))

    @property
    def n_inputs(self) -> int:
        return int(self.channels.shape[0])

    def set_scale(self, statistics: dict[str, torch.Tensor]) -> None:
        """Take the normalisation from the statistics of the training windows."""
        self.x_mean.copy_(statistics["x_mean"][self.channels.cpu()].to(self.x_mean))
        self.x_std.copy_(statistics["x_std"][self.channels.cpu()].to(self.x_std))
        self.a_std.copy_(statistics["a_std"].to(self.a_std))

    def inputs(self, hist: torch.Tensor) -> torch.Tensor:
        """Normalised inputs ``[B, H, C]`` of histories ``[..., rows, 4]`` with ``rows >= H``."""
        x = hist[..., -self.history :, :].index_select(-1, self.channels).to(self.x_mean.dtype)
        x = (x - self.x_mean) / self.x_std
        return x.reshape(-1, self.history, self.n_inputs)

    def core(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        """Normalised acceleration ``[B]`` from normalised inputs ``[B, H, C]``."""
        raise NotImplementedError

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        out = self.core(self.inputs(hist)) * self.a_std
        return out.reshape(hist.shape[:-2]).to(hist.dtype)

    # ------------------------------------------------------------------ training
    def training_loss(self, windows: WindowSet, index: torch.Tensor) -> torch.Tensor:
        """Loss of one minibatch; the default is the one-step squared error in normalised units."""
        predicted = self.core(self.inputs(windows.history(index)))
        target = (windows.target(index)[:, 0] / self.a_std).to(predicted.dtype)
        return torch.mean((predicted - target) ** 2)

    @torch.no_grad()
    def validation_error(self, windows: WindowSet, index: torch.Tensor) -> torch.Tensor:
        """Sum of squared one-step acceleration errors [m^2/s^4], the criterion of early stopping."""
        return torch.sum((self(windows.history(index)) - windows.target(index)[:, 0]) ** 2)

    def fit(self, train: WindowSet, val: WindowSet, cfg) -> dict:
        """Train on ``train`` with early stopping on ``val``; returns the training log."""
        from trafficloop.train.trainer import fit_supervised

        return fit_supervised(self, train, val, cfg)


def mlp(sizes: list[int], activation: type[nn.Module] = nn.Tanh, last: bool = False) -> nn.Sequential:
    """Fully connected stack; ``last`` keeps the activation after the final layer."""
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if last or i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)
