"""Result files: one JSON per model, protocol, dataset, data version and split.

``results/<protocol>/<dataset>/<version>/<split>-fold<k>/<model>.json`` holds the metrics,
the config hash, the data version tag and the reporting checklist; the per-event table
next to it (``<model>.events.parquet``) is what the driver-level bootstrap resamples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from follownet2.utils import config_hash, git_revision, resolve_path, write_json

CHECKLIST_FIELDS = (
    "data_version",
    "reconstruction_method",
    "split_type",
    "mechanistic_calibration_protocol",
    "closed_loop_horizon",
    "stability_test",
    "rare_event_suite",
    "seeds",
)


def empty_checklist() -> dict[str, Any]:
    return {name: None for name in CHECKLIST_FIELDS}


def result_dir(root: str | Path, protocol: str, dataset: str, version: str, split: str, fold: int) -> Path:
    return resolve_path(root) / protocol / dataset / version / f"{split}-fold{fold}"


def write_result(
    root: str | Path,
    *,
    protocol: str,
    model: str,
    family: str,
    dataset: str,
    data_version: str,
    split: str,
    fold: int,
    part: str,
    metrics: dict[str, Any],
    config: Any,
    checklist: dict[str, Any] | None = None,
    per_event: pd.DataFrame | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    directory = result_dir(root, protocol, dataset, data_version, split, fold)
    payload = {
        "protocol": protocol,
        "model": model,
        "family": family,
        "dataset": dataset,
        "data_version": data_version,
        "split": {"kind": split, "fold": fold, "part": part},
        "metrics": metrics,
        "config_hash": config_hash(config),
        "git_revision": git_revision(),
        "checklist": {**empty_checklist(), **(checklist or {})},
        **(extra or {}),
    }
    if per_event is not None:
        directory.mkdir(parents=True, exist_ok=True)
        per_event.to_parquet(directory / f"{model}.events.parquet", index=False)
        payload["per_event"] = f"{model}.events.parquet"
    return write_json(directory / f"{model}.json", payload)
