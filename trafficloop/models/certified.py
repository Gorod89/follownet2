"""Certified hybrid of cf-stability: a frozen IDM core plus a bounded residual with a certified Jacobian bound.

Port of ``ResidualIDM`` (``cf_stability/models/residual_idm.py``) with the certified budget of
decision D86 of cf-stability (:mod:`trafficloop.stability.certificate`). Conventions of both
repositories agree: net gap ``s``, ``dv = v - v_lead`` (positive when closing in), SI units; the
residual reads ``(s, dv, v)`` of the last row through the same fixed physical scaler.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm

from trafficloop.data.batch import DV, S, V
from trafficloop.models.base import ACC_CLIP, TrainingContext
from trafficloop.models.mechanistic import IDM, idm_acc
from trafficloop.models.network import NetworkModel, mlp
from trafficloop.stability import certificate
from trafficloop.train.trainer import TrainConfig, fit_supervised
from trafficloop.train.windows import WindowSet

#: fixed scales of ``(s, dv, v)`` [m, m/s, m/s] of the residual's input (cf-stability ``PHYSICAL_SCALE``):
#: the Jacobian bound of the residual must not depend on the dataset
PHYSICAL_SCALE = (20.0, 2.0, 10.0)


def _antisymmetric_init(hidden: nn.Linear, out: nn.Linear) -> None:
    """Duplicate the units of ``hidden`` and give ``out`` opposite weights on the copies: output 0 for every input.

    Spectral normalisation divides a weight by its norm, so scaling the weights cannot make the
    initial output small; cancelling pairs of units can.
    """
    half = hidden.out_features // 2
    with torch.no_grad():
        hidden.weight[half:] = hidden.weight[:half]
        hidden.bias[half:] = hidden.bias[:half]
        out.weight[:, half:] = -out.weight[:, :half]
        out.bias.zero_()


class CertifiedResidualIDM(NetworkModel):
    """``a = IDM(s, dv, v; theta) + r_max * tanh(lipschitz * g(x / scale))`` on the last row, with a certificate.

    ``theta`` is a frozen IDM core from the pooled calibration of the training part (buffer);
    ``g`` is a tanh MLP ``3 -> hidden -> 1`` with spectral normalisation on every linear layer,
    initialised antisymmetrically so that the untrained residual is zero; ``x / scale`` is the
    fixed physical scaler ``PHYSICAL_SCALE``. Then ``|r| <= r_max`` and
    ``|dr/dx_j| <= r_max * lipschitz * prod(layer norms) / scale_j`` (:meth:`jacobian_bound`).
    :meth:`before_training` chooses the core and sets ``lipschitz`` by the certified budget of
    D86 (:func:`~trafficloop.stability.certificate.enforce_budget`), in this order:

    1. the pooled IDM of the training part, i.e. the core of the ``idm`` entry, with ``r_max``;
    2. the pooled IDM calibrated with the string-stability margin ``stability_margin`` (the term
       of decision D72 of cf-stability) with ``r_max``;
    3. that core with ``fallback_r_max``, the amplitude of D86.

    The first step whose a priori certificate admits a budget is kept; when none does, training
    stops with ``ValueError``. :meth:`fit` runs the supervised trainer and checks the a priori
    certificate of the trained weights; the record goes to ``train.json["certificate"]``.
    ``r_max`` and ``lipschitz`` in force are buffers (``residual_scale``), so a checkpoint keeps
    them; ``kwargs`` stay the configuration.

    Differences from cf-stability: the core is the benchmark IDM (gap floored at 0.1 m instead of
    0.01 m, identical wherever an equilibrium can lie) computed in the dtype of the input (float64
    in P3), the residual in the dtype of the network; the training is the one-step loss of the
    benchmark (prediction clipped to ``[-8, 4]``, as in P0) instead of the rollout loss of
    cf-stability (its D63), with the shared trainer and budget; the core comes from the
    benchmark's calibration with the fallbacks above instead of one configured core; the
    certificate covers the spacings ``0.5 .. 300 m`` (:mod:`trafficloop.stability.certificate`).
    """

    name = "residual_idm_certified"
    memoryless = True

    def __init__(
        self,
        hidden: Sequence[int] = (64, 64),
        r_max: float = 0.3,
        safety: float = 0.9,
        stability_margin: float = 0.2,
        fallback_r_max: float = 0.1,
    ) -> None:
        super().__init__(history=1)
        hidden = [int(h) for h in hidden]
        if hidden[-1] % 2:
            raise ValueError(f"the last hidden layer needs an even width, got {hidden[-1]}")
        if not 0.0 < safety <= 1.0:
            raise ValueError(f"safety must lie in (0, 1], got {safety}")
        if not (r_max > 0.0 and fallback_r_max > 0.0):
            raise ValueError(f"r_max and fallback_r_max must be positive, got {r_max} and {fallback_r_max}")
        self.kwargs = {
            "hidden": hidden,
            "r_max": float(r_max),
            "safety": float(safety),
            "stability_margin": float(stability_margin),
            "fallback_r_max": float(fallback_r_max),
        }
        self.x_std.copy_(torch.tensor(PHYSICAL_SCALE))  # x_mean stays 0: inputs(hist) is x / PHYSICAL_SCALE
        self.g = mlp([self.n_inputs, *hidden, 1])
        linears = self._linears()
        _antisymmetric_init(linears[-2], linears[-1])
        for layer in linears:
            spectral_norm(layer)
        self.register_buffer("theta", torch.tensor(IDM.default, dtype=torch.float64))  # (v0, T, s0, a, b)
        self.register_buffer("residual_scale", torch.tensor([float(r_max), 1.0], dtype=torch.float64))
        #: core, attempts and budget chosen by :meth:`before_training` (not part of a checkpoint)
        self.setup: dict | None = None

    # ------------------------------------------------------------------ parameters
    @property
    def r_max(self) -> float:
        """Amplitude of the residual in force [m/s^2]."""
        return float(self.residual_scale[0])

    @r_max.setter
    def r_max(self, value: float) -> None:
        with torch.no_grad():
            self.residual_scale[0] = float(value)

    @property
    def lipschitz(self) -> float:
        """Factor of ``g`` inside the tanh; :func:`~trafficloop.stability.certificate.enforce_budget` sets it."""
        return float(self.residual_scale[1])

    @lipschitz.setter
    def lipschitz(self, value: float) -> None:
        with torch.no_grad():
            self.residual_scale[1] = float(value)

    @property
    def physical_scale(self) -> Tensor:
        """Scales ``[3]`` of ``(s, dv, v)`` of the residual's input, float64."""
        return self.x_std.double()

    def core_params(self) -> dict[str, float]:
        return dict(zip(IDM.param_names, self.theta.detach().double().cpu().tolist()))

    def set_core(self, params: Mapping[str, float] | Sequence[float]) -> None:
        """Set the IDM core ``(v0, T, s0, a, b)``, by name or in this order."""
        values = [float(params[n]) for n in IDM.param_names] if isinstance(params, Mapping) else list(params)
        with torch.no_grad():
            self.theta.copy_(torch.as_tensor(values, dtype=torch.float64))

    def _linears(self) -> list[nn.Linear]:
        return [m for m in self.g if isinstance(m, nn.Linear)]

    def layer_norms(self) -> list[float]:
        """Exact spectral norms (SVD) of the effective weights of ``g``, read in evaluation mode."""
        was_training = self.training
        self.eval()  # reading a weight in training mode would run a power iteration
        try:
            with torch.no_grad():
                return [float(torch.linalg.matrix_norm(m.weight.double(), ord=2)) for m in self._linears()]
        finally:
            self.train(was_training)

    def jacobian_bound(self) -> Tensor:
        """Bound ``[3]`` of ``|dr/d(s, dv, v)|``: ``r_max * lipschitz * prod(layer_norms) / scale``."""
        return self.r_max * self.lipschitz * math.prod(self.layer_norms()) / self.physical_scale.cpu()

    def set_scale(self, statistics: dict[str, Tensor]) -> None:
        """Only the scale of the loss comes from the training data; the input scaler stays physical."""
        self.a_std.copy_(statistics["a_std"].to(self.a_std))

    # --------------------------------------------------------------------- model
    def core_acc(self, hist: Tensor) -> Tensor:
        """Acceleration of the IDM core at the last row, in the dtype of ``hist``."""
        now = hist[..., -1, :]
        v0, T, s0, a, b = self.theta.to(hist.dtype).unbind(-1)
        return idm_acc(now[..., S], now[..., DV], now[..., V], v0, T, s0, a, b)

    def residual(self, hist: Tensor) -> Tensor:
        """``r_max * tanh(lipschitz * g(x / scale))`` of the last row, in the dtype of the network."""
        g = self.g(self.inputs(hist).flatten(1)).squeeze(-1)
        scale = self.residual_scale.to(g.dtype)
        return (scale[0] * torch.tanh(scale[1] * g)).reshape(hist.shape[:-2])

    def forward(self, hist: Tensor) -> Tensor:
        return self.core_acc(hist) + self.residual(hist).to(hist.dtype)

    # ------------------------------------------------------------------ training
    def training_loss(self, windows: WindowSet, index: Tensor) -> Tensor:
        """One-step squared error in units of the acceleration spread, the prediction clipped to ``[-8, 4]``
        (the core alone brakes far harder than that at small gaps, where the residual can do nothing)."""
        predicted = self(windows.history(index)).clamp(*ACC_CLIP)
        target = windows.target(index)[:, 0].to(predicted.dtype)
        return torch.mean(((predicted - target) / self.a_std.to(predicted.dtype)) ** 2)

    @torch.no_grad()
    def validation_error(self, windows: WindowSet, index: Tensor) -> Tensor:
        """Sum of squared one-step errors of the clipped prediction [m^2/s^4], as P0 scores it."""
        return torch.sum((self(windows.history(index)).clamp(*ACC_CLIP) - windows.target(index)[:, 0]) ** 2)

    def before_training(self, context: TrainingContext) -> None:
        """Choose the core and certify the budget: the pooled core and the core with the stability margin are
        both certified at ``r_max`` and the one with the larger admissible budget is kept; the margin core at
        ``fallback_r_max`` only when both admit none; ``ValueError`` when no step admits a budget."""
        r_max, margin, fallback = (self.kwargs[k] for k in ("r_max", "stability_margin", "fallback_r_max"))
        with_margin = "pooled with stability margin"
        steps = dict.fromkeys([("pooled", None, r_max), (with_margin, margin, r_max), (with_margin, margin, fallback)])
        cores: dict[float | None, dict] = {}
        attempts: list[dict[str, Any]] = []
        candidates: list[tuple[float, str, float | None, float]] = []
        for kind, core_margin, amplitude in steps:  # unique steps, in order
            if amplitude != r_max and candidates:
                break  # the fallback amplitude only when both cores admit no budget at r_max
            if core_margin not in cores:
                penalty = None if core_margin is None else partial(certificate.stability_shortfall, margin=core_margin)
                cores[core_margin] = context.calibrate(IDM, penalty=penalty)
            self.set_core(cores[core_margin]["params"])
            self.r_max, self.lipschitz = amplitude, 1.0
            attempt = {"core": kind, "stability_margin": core_margin, "r_max": amplitude, "params": self.core_params()}
            try:
                budget = certificate.enforce_budget(self, self.kwargs["safety"])
            except ValueError as err:
                attempts.append({**attempt, "admissible": 0.0, "error": str(err)})
                continue
            attempts.append({**attempt, "admissible": budget["admissible"]})
            candidates.append((budget["admissible"], kind, core_margin, amplitude))
        if not candidates:
            tried = "; ".join(f"{a['core']} with r_max {a['r_max']:g}: {a['error']}" for a in attempts)
            raise ValueError(f"{self.name} on {context.dataset}: no core admits a certified residual budget; {tried}")
        admissible, kind, core_margin, amplitude = max(candidates, key=lambda c: c[0])  # the larger budget wins
        self.set_core(cores[core_margin]["params"])
        self.r_max, self.lipschitz = amplitude, 1.0
        budget = certificate.enforce_budget(self, self.kwargs["safety"])
        self.setup = {
            "core": _core_record(kind, core_margin, cores[core_margin], self.theta),
            "attempts": attempts,
            "rule": "the core with the larger admissible budget at r_max; fallback_r_max only when both admit none",
            **budget,
            "layer_norms_at_enforcement": self.layer_norms(),
        }

    def fit(self, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
        """The supervised trainer, after :meth:`before_training`; the log carries the a priori certificate of
        the trained weights (``certificate``), which must hold."""
        if self.setup is None:
            raise RuntimeError(f"{self.name}: before_training(context) must set the core and the certified budget first")
        log = fit_supervised(self, train, val, cfg)
        record = self.certificate_record()
        if not record["holds"]:
            failing = [v for v, ok in zip(record["speeds"], record["holds_per_speed"]) if not ok]
            raise RuntimeError(f"{self.name}: the a priori certificate of the trained model fails at v = {failing} m/s")
        log["certificate"] = record
        return log

    def certificate_record(self) -> dict[str, Any]:
        """The a priori certificate of the current weights as plain JSON values.

        ``holds``: at every speed of the grid the certificate holds or no spacing is feasible (no
        equilibrium can exist there). With the entries of :meth:`before_training` when it ran.
        """
        cert = certificate.certificate_a_priori(self, keep_r_max=True)
        holds = cert["holds"] | ~cert["feasible"]
        margin = cert["guaranteed_margin"]
        worst = int(torch.nan_to_num(margin, nan=torch.inf).argmin())
        record = {
            "decision": "a priori certificate of cf-stability (D71, D86): lipschitz = safety * admissible / r_max",
            "holds": bool(holds.all()),
            **(self.setup or {}),
            "r_max": cert["r_max"],
            "lipschitz": cert["lipschitz"],
            "budget": cert["budget"],
            "bounds": cert["bounds"],
            "layer_norms": cert["layer_norms"],
            "speeds": cert["v"],
            "holds_per_speed": holds,
            "n_holds": cert["n_holds"],
            "n_feasible": int(cert["feasible"].sum()),
            "n_speeds": len(cert["v"]),
            "guaranteed_margin_min": float(margin[worst]) if bool(torch.isfinite(margin).any()) else None,
            "v_worst": float(cert["v"][worst]) if bool(torch.isfinite(margin).any()) else None,
            "guaranteed_margin": margin,
            "s_low": cert["s_low"],
            "s_high": cert["s_high"],
            "s_worst": cert["s_worst"],
            "within": cert["within"],
            "max_budget": cert["max_budget"],
            "spacing_range": [certificate.S_MIN, certificate.S_MAX],
            "n_scan": certificate.N_SCAN,
        }
        return certificate.to_json(record)


def _core_record(kind: str, margin: float | None, fit: Mapping[str, Any], theta: Tensor) -> dict[str, Any]:
    """The chosen core: how it was calibrated, its parameters, the fit and its smallest margin over the grid."""
    keys = ("objective", "fit_objective", "rmse_s", "rmse_v", "collision_rate", "n_events", "n_generations", "converged")
    m = certificate.core_margin(theta.detach().cpu())
    return {
        "kind": kind,
        "stability_margin": margin,
        "params": dict(zip(IDM.param_names, theta.detach().double().cpu().tolist())),
        "margin_min": float(torch.nan_to_num(m, nan=certificate.NO_EQUILIBRIUM).min()),
        "calibration": {k: fit[k] for k in keys if k in fit},
    }
