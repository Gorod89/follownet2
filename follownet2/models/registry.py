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

from follownet2.models.base import CFModel

MODELS: dict[str, str] = {
    "persistence": "follownet2.models.persistence:PersistenceBaseline",
    "constant_speed": "follownet2.models.persistence:ZeroAcceleration",
    "idm": "follownet2.models.mechanistic:IDM",
    "gipps": "follownet2.models.mechanistic:Gipps",
    "newell": "follownet2.models.mechanistic:Newell",
    "knn": "follownet2.models.knn:KNN",
    "mlp": "follownet2.models.sequence:MLP",
    "gru": "follownet2.models.sequence:GRU",
    "lstm": "follownet2.models.sequence:LSTM",
    "seq2seq": "follownet2.models.sequence:Seq2Seq",
    "mdn_lstm": "follownet2.models.sequence:MDNLSTM",
    "pidl": "follownet2.models.hybrid:PIDL",
    "perl": "follownet2.models.hybrid:PERL",
    "stability_mlp": "follownet2.models.hybrid:StabilityRegularisedMLP",
    "ddpg": "follownet2.models.policy:DDPG",
    "gail": "follownet2.models.policy:GAIL",
    "maml_lstm": "follownet2.models.meta:MAMLLSTM",
    "diffusion": "follownet2.models.diffusion:DiffusionCF",
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
