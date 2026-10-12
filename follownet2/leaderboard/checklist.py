"""Checklist gate of the leaderboard (docs/leaderboard.md, section 1; docs/checklist.md).

A run directory is ``<root>/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/`` with
``train.json`` and the protocol files ``p1.json`` ... ``p4.json`` and ``p1_<other>.json``.
:func:`validate_entry` lists the problems of one run (empty when it is admitted):

* every protocol file present has a ``checklist`` block; a run without any is refused;
* ``data_version``, ``reconstruction_method``, ``split_type``,
  ``mechanistic_calibration_protocol`` and ``seeds`` are non-empty in every block (inside a
  mapping such as ``seeds`` every entry must be stated);
* ``closed_loop_horizon`` is non-empty in ``p1.json``, ``p2.json``, ``p4.json`` and every
  ``p1_<other>.json``, ``stability_test`` in ``p3.json``, ``rare_event_suite`` in ``p4.json``;
* the data version and the split type of every block are those of the run (its directory);
* ``train.json`` carries ``config_hash`` and ``git_revision`` and every protocol file names the
  same model configuration hash (``model.provenance.config_hash``).

Runs under the ``event`` split are labelled "driver leak" (checklist rule 2).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from follownet2.experiments.analysis import _PROTOCOL_FILE, VERSION
from follownet2.utils import read_json

#: fields every checklist block states
COMMON_FIELDS = ("data_version", "reconstruction_method", "split_type", "mechanistic_calibration_protocol", "seeds")
#: further fields per protocol (``p1_<other>`` is a P1 result)
PROTOCOL_FIELDS = {
    "p1": ("closed_loop_horizon",),
    "p2": ("closed_loop_horizon",),
    "p3": ("stability_test",),
    "p4": ("closed_loop_horizon", "rare_event_suite"),
}
DRIVER_LEAK = "driver leak"
#: splits whose results carry the "driver leak" label
LEAK_SPLITS = ("event",)
#: the runs ranked by the leaderboard of a dataset
MAIN_SPLIT, MAIN_FOLD, MAIN_SEED = "driver", 0, 0
CLEAN = "paired-clean"
#: roles of a run in the leaderboard of its dataset
ENTRY, SEED_SPREAD, FOLD_SPREAD, LEAK, UNUSED = "ranked entry", "seed spread", "fold spread", DRIVER_LEAK, "not used"
RUN_COLUMNS = ["run", "dataset", "version", "split", "fold", "model", "seed", "role", "label", "admitted", "problems", "directory"]

_SPLIT = re.compile(r"^(?P<split>.+)-fold(?P<fold>\d+)$")
_SEED = re.compile(r"^seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class Run:
    """Identity of a run directory, read from its path."""

    directory: Path
    dataset: str
    version: str
    split: str
    fold: int
    label: str
    seed: int

    @property
    def name(self) -> str:
        return f"{self.split}-fold{self.fold}/{self.label}/seed{self.seed}"

    @property
    def role(self) -> str:
        if self.version != VERSION:
            return UNUSED
        if self.split == MAIN_SPLIT and self.fold == MAIN_FOLD:
            return ENTRY if self.seed == MAIN_SEED else SEED_SPREAD
        if self.seed == MAIN_SEED and (self.split == MAIN_SPLIT or self.split == CLEAN):
            return FOLD_SPREAD  # folds 1-4 of the driver split (full test parts), or the clean arms of E2
        if self.split in LEAK_SPLITS:
            return LEAK
        return UNUSED


def parse_run(directory) -> Run | None:
    """The identity of ``directory`` or None when the path does not follow the run layout."""
    directory = Path(directory)
    parts = directory.parts
    if len(parts) < 5:
        return None
    seed, split = _SEED.match(parts[-1]), _SPLIT.match(parts[-3])
    if seed is None or split is None:
        return None
    return Run(directory, parts[-5], parts[-4], split["split"], int(split["fold"]), parts[-2], int(seed["seed"]))


def entry_label(directory) -> str:
    """``"driver leak"`` for a run under a leaky split (``event``), else ``""``."""
    run = parse_run(directory)
    return DRIVER_LEAK if run is not None and run.split in LEAK_SPLITS else ""


def protocol_files(directory) -> list[Path]:
    """``p<k>.json`` and ``p<k>_<other>.json`` of a run, sorted."""
    return sorted(p for p in Path(directory).glob("p*.json") if _PROTOCOL_FILE.match(p.name))


def required_fields(name: str) -> tuple[str, ...]:
    """Checklist fields that must be non-empty in the block of protocol file ``name`` (stem or file name)."""
    stem = Path(name).name.removesuffix(".json")
    return COMMON_FIELDS + PROTOCOL_FIELDS.get(stem.split("_", 1)[0], ())


def _empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (Mapping, list, tuple, set)):
        return len(value) == 0
    return False


def _field(file: str, mapping: Mapping, field: str) -> list[str]:
    if field not in mapping:
        return [f"{file}: {field} missing"]
    value = mapping[field]
    if _empty(value):
        return [f"{file}: {field} empty"]
    if isinstance(value, Mapping):
        return [f"{file}: {field}.{key} empty" for key, item in value.items() if _empty(item)]
    return []


def _load(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, f"{path.name} missing"
    try:
        data = read_json(path)
    except (OSError, ValueError) as error:
        return None, f"{path.name}: unreadable ({type(error).__name__}: {error})"
    if not isinstance(data, Mapping):
        return None, f"{path.name}: not a JSON object"
    return dict(data), None


def validate_entry(directory) -> list[str]:
    """Problems of the checklist of the run in ``directory`` (empty when the run is admitted).

    Every problem names the file and the field, e.g. ``"p3.json: stability_test empty"`` or
    ``"p2.json: data_version raw differs from the run's reconstructed"``.
    """
    directory = Path(directory)
    run = parse_run(directory)
    problems = []
    if run is None:
        problems.append("the directory does not follow <dataset>/<version>/<split>-fold<k>/<label>/seed<s>")
    train, problem = _load(directory / "train.json")
    if train is None:
        problems.append(problem)
    else:
        for key in ("config_hash", "git_revision"):
            problems += _field("train.json", train, key)
    model_hash = None if train is None or _empty(train.get("config_hash")) else str(train["config_hash"])
    files = protocol_files(directory)
    if not files:
        problems.append("no protocol file")
    for path in files:
        data, problem = _load(path)
        if data is None:
            problems.append(problem)
            continue
        block = data.get("checklist")
        if not isinstance(block, Mapping) or not block:
            problems.append(f"{path.name}: no checklist block")
        else:
            for field in required_fields(path.name):
                problems += _field(path.name, block, field)
            if run is not None:
                for field, expected in (("data_version", run.version), ("split_type", run.split)):
                    value = block.get(field)
                    if not _empty(value) and str(value) != expected:
                        problems.append(f"{path.name}: {field} {value} differs from the run's {expected}")
        model = data.get("model")
        provenance = model.get("provenance") if isinstance(model, Mapping) else None
        named = provenance.get("config_hash") if isinstance(provenance, Mapping) else None
        if _empty(named):
            problems.append(f"{path.name}: model.provenance.config_hash missing")
        elif model_hash is not None and str(named) != model_hash:
            problems.append(f"{path.name}: model.provenance.config_hash {named} differs from train.json's {model_hash}")
    return problems


def find_runs(root, dataset: str) -> list[Run]:
    """Every run directory of ``dataset`` under ``root`` (any version, split, fold, label and seed)."""
    base = Path(root) / dataset
    runs = [parse_run(p) for p in base.glob("*/*/*/seed*") if p.is_dir()]
    return sorted((r for r in runs if r is not None), key=lambda r: (r.version, r.split, r.fold, r.label, r.seed))


def validate_dataset(root, dataset: str) -> pd.DataFrame:
    """One row per run directory of ``dataset``: identity, role in the leaderboard, label,
    ``admitted`` and the list of ``problems`` (:func:`validate_entry`)."""
    rows = []
    for run in find_runs(root, dataset):
        problems = validate_entry(run.directory)
        rows.append({"run": run.name, "dataset": run.dataset, "version": run.version, "split": run.split, "fold": run.fold,
                     "model": run.label, "seed": run.seed, "role": run.role, "label": entry_label(run.directory),
                     "admitted": not problems, "problems": problems, "directory": run.directory})
    return pd.DataFrame(rows, columns=RUN_COLUMNS)
