"""Model interface of the benchmark: ``acc = model(state_history)``.

``state_history`` has shape ``[..., H, 4]`` with channels ``(s, dv, v, a_prev)`` (see
:mod:`follownet2.data.batch`), rows ordered in time and the last row being the present.
The output has shape ``[...]`` and is the acceleration applied over the next 0.1 s step.
Models written for the three-channel interface of the sibling repositories are wrapped by
:class:`ThreeChannelAdapter`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import nn

from follownet2.data.batch import WARMUP

ACC_CLIP = (-8.0, 4.0)
S_FLOOR = 0.1  # m, keeps interaction terms finite after a collision


class CFModel(nn.Module):
    """Base class. Subclasses set ``history`` and implement ``forward``."""

    name: str = "model"
    family: str = "network"  # "baseline" | "mechanistic" | "network"
    #: rows of history consumed; must not exceed the warm-up of the protocols
    history: int = 1
    #: True when the output depends on the current (s, dv, v) only, so that the analytic
    #: string-stability criterion applies
    memoryless: bool = False
    #: steps after a prediction instant that training needs inside the event
    train_horizon: int = 1
    #: True when :meth:`begin` changes the model for the events at hand (few-shot adaptation)
    adaptive: bool = False

    def __init__(self) -> None:
        super().__init__()
        if self.history > WARMUP:
            raise ValueError(f"{self.name}: history {self.history} exceeds the protocol warm-up {WARMUP}")
        if not hasattr(self, "kwargs"):
            self.kwargs: dict = {}

    def begin(self, hist: torch.Tensor) -> None:
        """Called by every protocol once per batch of events, before the first prediction.

        ``hist`` is the observed warm-up ``[E, WARMUP, 4]`` that ends at the initial condition.
        Models that adapt to the driver at hand do it here, from these rows only; afterwards
        they are called with histories ``[E, ..., H, 4]`` of the same ``E`` events. The
        default does nothing.
        """

    def fit(self, train, val, cfg) -> dict:
        """Train on the windows ``train`` with ``val`` for model selection; returns the log.

        The baselines have nothing to fit.
        """
        return {}

    def checkpoint(self) -> dict:
        """Everything needed to build the model again: name, constructor arguments, state."""
        return {"name": self.name, "kwargs": dict(self.kwargs), "state": self.state_dict()}

    def restore(self, state: dict) -> None:
        self.load_state_dict(state)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_dist(self, hist: torch.Tensor):
        """Optional predictive distribution of the acceleration (``torch.distributions``)."""
        raise NotImplementedError(f"{self.name} is a point model")

    @property
    def is_distributional(self) -> bool:
        return type(self).predict_dist is not CFModel.predict_dist


class MechanisticModel(CFModel):
    """Closed-form model with a parameter vector.

    ``params`` has shape ``[..., D]`` and broadcasts against the batch dimensions of the
    history, which is what the batched calibration relies on (one candidate per leading
    index).
    """

    family = "mechanistic"
    param_names: Sequence[str] = ()
    #: default calibration bounds, one (low, high) pair per parameter
    bounds: Sequence[tuple[float, float]] = ()
    #: parameter value used when nothing has been calibrated (literature default)
    default: Sequence[float] = ()

    def __init__(self, params: Sequence[float] | torch.Tensor | None = None) -> None:
        super().__init__()
        values = torch.as_tensor(self.default if params is None else params, dtype=torch.float64)
        if values.shape[-1] != len(self.param_names):
            raise ValueError(f"{self.name}: expected {len(self.param_names)} parameters, got {values.shape[-1]}")
        self.register_buffer("params", values)
        self.kwargs = {"params": [float(v) for v in values.reshape(-1)]}

    def acc(self, hist: torch.Tensor, params: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self.acc(hist, self.params.to(hist.dtype))

    def with_params(self, params: Sequence[float] | torch.Tensor) -> "MechanisticModel":
        return type(self)(params)

    def param_dict(self) -> dict[str, float]:
        flat = self.params.reshape(-1, len(self.param_names))[0]
        return {name: float(value) for name, value in zip(self.param_names, flat)}

    @classmethod
    def bounds_tensor(cls, dtype=torch.float64, device=None) -> torch.Tensor:
        return torch.as_tensor(cls.bounds, dtype=dtype, device=device)


class ThreeChannelAdapter(CFModel):
    """Wrap a callable that expects histories ``[..., H, 3]`` with channels (s, dv, v)."""

    name = "external"

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], history: int = 1, name: str = "external") -> None:
        self.history = history
        self.name = name
        super().__init__()
        self.fn = fn

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        out = self.fn(hist[..., -self.history :, :3])
        return out.reshape(hist.shape[:-2])
