"""Models of the benchmark by name, and their checkpoints.

A checkpoint is a ``torch`` file with the name of the model, its constructor arguments, its
state and a free dictionary of provenance (dataset, data version, split, seed, config hash,
training log). Loading builds the model from the arguments and restores the state; no code
is stored in the file.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from trafficloop.models.base import CFModel

MODELS: dict[str, str] = {
    "persistence": "trafficloop.models.persistence:PersistenceBaseline",
    "constant_speed": "trafficloop.models.persistence:ZeroAcceleration",
    "idm": "trafficloop.models.mechanistic:IDM",
    "gipps": "trafficloop.models.mechanistic:Gipps",
    "newell": "trafficloop.models.mechanistic:Newell",
    "knn": "trafficloop.models.knn:KNN",
    "mlp": "trafficloop.models.sequence:MLP",
    "gru": "trafficloop.models.sequence:GRU",
    "lstm": "trafficloop.models.sequence:LSTM",
    "seq2seq": "trafficloop.models.sequence:Seq2Seq",
    "mdn_lstm": "trafficloop.models.sequence:MDNLSTM",
    "pidl": "trafficloop.models.hybrid:PIDL",
    "perl": "trafficloop.models.hybrid:PERL",
    "stability_mlp": "trafficloop.models.hybrid:StabilityRegularisedMLP",
    "residual_idm_certified": "trafficloop.models.certified:CertifiedResidualIDM",
    "ddpg": "trafficloop.models.policy:DDPG",
    "gail": "trafficloop.models.policy:GAIL",
    "maml_lstm": "trafficloop.models.meta:MAMLLSTM",
    "diffusion": "trafficloop.models.diffusion:DiffusionCF",
}
#: mechanistic models with the median of the per-event estimates instead of the pooled fit
MEDIAN_SUFFIX = "_median"


def model_class(name: str) -> type[CFModel]:
    base = name[: -len(MEDIAN_SUFFIX)] if name.endswith(MEDIAN_SUFFIX) else name
    if base not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODELS)}")
    module, attribute = MODELS[base].split(":")
    return getattr(importlib.import_module(module), attribute)


def create(name: str, **kwargs: Any) -> CFModel:
    model = model_class(name)(**kwargs)
    model.name = name
    return model


def save_model(model: CFModel, path: str | Path, provenance: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**model.checkpoint(), "provenance": provenance or {}}
    payload["state"] = {k: v.detach().cpu() for k, v in payload["state"].items()}
    torch.save(payload, path)
    return path


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[CFModel, dict]:
    """The model of a checkpoint in evaluation mode, and its provenance."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = create(payload["name"], **payload["kwargs"])
    floating = [v.dtype for v in payload["state"].values() if v.is_floating_point()]
    if floating and all(d == torch.float64 for d in floating):
        model = model.double()  # a model trained in double precision is not rounded
    model.restore(payload["state"])
    return model.to(device).eval(), payload.get("provenance", {})
