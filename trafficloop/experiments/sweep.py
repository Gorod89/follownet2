"""Experiment E5: hyperparameter sweep of a reference network with two selection criteria.

A sweep trains ``n`` configurations of one model (``lstm``, ``gru`` or ``mlp``) on one dataset,
data version, split, fold and seed, and selects among them on the validation part of the split
by two criteria: the one-step criterion ``val_rmse_a`` (the validation RMSE of the acceleration
that the early stopping keeps, ``best_val_rmse_a`` of ``train.json``) and the closed-loop
criterion ``val_rmse_s_10s`` (the spacing RMSE of the 10 s windows of P1 on the validation
events). The reference configuration and the two selected runs are then scored with the
protocols on the test part (``docs/experiments.md``, section 4).

The runs are ordinary run directories of ``scripts/train.py`` and ``scripts/evaluate.py`` with
their own label, ``sweep_<model>_<kk>``, and their own results root (the hydra key
``paths.results``, default ``results/sweeps``):
``<root>/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/``. The leaderboard and the
milestone reports scan ``results/models`` only, so they never see a sweep run. The files of a
sweep are in ``<root>/<dataset>/<model>/``: ``configs.json``, ``sweep.csv``, ``selection.json``,
``failures.json`` and ``REPORT.md``; every validated run directory receives ``sweep.json``.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trafficloop import pipeline
from trafficloop.data.splits import Split
from trafficloop.experiments.analysis import _json, _num, _summary, stale_protocols
from trafficloop.stats.ranks import kendall_tau
from trafficloop.train import run as training
from trafficloop.utils import REPO_ROOT, config_hash, git_revision, read_json, resolve_path, write_json

#: models with a search space: the one-step reference networks (none of them has a dropout layer)
MODELS = ("lstm", "gru", "mlp")
WIDTHS = (32, 64, 128, 256)  # units of the recurrent layer, or of every hidden layer of the MLP
MLP_LAYERS = (2, 3)
HISTORIES = (10, 20, 30)  # samples
LR_RANGE = (1e-4, 3e-3)  # log-uniform, rounded to three significant digits
WEIGHT_DECAYS = (0.0, 1e-5, 1e-4)
BATCH_SIZES = (1024, 4096)
A_PREV = (False, True)
#: hyperparameter -> hydra key of scripts/train.py; the rest of the budget is conf/train.yaml
KEYS = {
    "hidden": "model.kwargs.hidden",
    "history": "model.kwargs.history",
    "use_a_prev": "model.kwargs.use_a_prev",
    "lr": "train.optim.lr",
    "weight_decay": "train.optim.weight_decay",
    "batch_size": "train.optim.batch_size",
}
HYPERPARAMETERS = tuple(KEYS)
#: keys the sweep sets itself; a further override must not change them (nor a node above them)
RESERVED = (
    "dataset", "model", "model.label", "paths.results", "train.version", "train.split", "train.fold",
    "train.seed", "protocols", *KEYS.values(), "model.optim.lr", "model.optim.weight_decay", "model.optim.batch_size",
)
PROTOCOLS = ("p1", "p1s", "p2", "p3", "p4")
STAGES = ("train", "validate", "select", "evaluate", "report")
ROOT = "results/sweeps"
HORIZON = 10.0  # s, windows of the closed-loop criterion
VAL_EVENTS = 2000  # validation events of the closed-loop criterion, at most
TIMEOUT = 2 * 3600  # s per run, as scripts/m2.py
ATTEMPTS = 3  # a failed run (the GPU is shared) is repeated, as in scripts/m4.py
#: selection -> criterion (lower is better)
CRITERIA = {"one_step": "val_rmse_a", "closed_loop": "val_rmse_s_10s"}
ROLES = {"reference": "reference", "one_step": "one-step selection", "closed_loop": "closed-loop selection"}
COLUMNS = (
    "index", "label", "hidden", "history", "use_a_prev", "lr", "weight_decay", "batch_size",
    "val_rmse_a", "val_rmse_s_10s", "val_collision_rate_10s", "val_rmse_a_p0", "val_rmse_v_10s",
    "epochs", "seconds", "n_parameters", "status",
)
#: protocol scores of an evaluated run (test part) -> (protocol file, keys in its summary)
SCORES = {
    "p1_h10_rmse_s": ("p1", ("h10", "rmse_s")),
    "p1_whole_rmse_s": ("p1", ("whole", "rmse_s")),
    "p2_collisions_100_in_range": ("p2", ("collisions_100_in_range",)),
    "p3_unstable_fraction": ("p3", ("unstable_fraction",)),
}
#: runs a script with hydra overrides; None on success, else the last exit status
Runner = Callable[[str, Sequence[str]], "str | None"]


class SweepError(ValueError):
    """A sweep that cannot run as asked (conflicting files, reserved keys, missing stages)."""


def _reserved(key: str) -> bool:
    return any(key == name or name.startswith(key + ".") for name in RESERVED)


@dataclass(frozen=True)
class Sweep:
    """``n`` configurations of ``model`` on one dataset, data version, split, fold and seed.

    ``seed`` is the training seed of every run and the seed of the sampling; ``root`` is the
    results root of the runs and of the files of the sweep; ``overrides`` are further hydra
    overrides of every run (a smaller budget for a smoke test).
    """

    dataset: str = "ngsim"
    model: str = "lstm"
    n: int = 16
    seed: int = 0
    version: str = "reconstructed"
    split: str = "driver"
    fold: int = 0
    root: str = ROOT
    overrides: tuple[str, ...] = ()
    protocols: tuple[str, ...] = PROTOCOLS
    val_events: int = VAL_EVENTS

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise SweepError(f"model {self.model!r} has no search space; sweeps exist for {', '.join(MODELS)}")
        if self.n < 1:
            raise SweepError("n must be at least 1: configuration 0 is the reference")
        if self.val_events < 1:
            raise SweepError("val_events must be at least 1")
        for item in self.overrides:
            key, sep, _ = item.partition("=")
            if not sep or _reserved(key.lstrip("+~")):
                raise SweepError(f"override {item!r}: not key=value, or a key that the sweep sets itself")

    @property
    def directory(self) -> Path:
        """``<root>/<dataset>/<model>``: the files of the sweep."""
        return resolve_path(self.root) / self.dataset / self.model

    def run_dir(self, label: str) -> Path:
        """Directory of a run, as :func:`trafficloop.train.run.model_dir` builds it."""
        return (
            resolve_path(self.root) / "models" / self.dataset / self.version / f"{self.split}-fold{self.fold}"
            / label / f"seed{self.seed}"
        )

    def identity(self) -> dict:
        """What the files of the sweep belong to; one directory holds one sweep."""
        return {"dataset": self.dataset, "model": self.model, "seed": self.seed, "version": self.version,
                "split": self.split, "fold": self.fold}


# ----------------------------------------------------------------------------- configurations


def compose_config(name: str, overrides: Sequence[str]):
    """``conf/<name>.yaml`` with hydra overrides, as ``scripts/train.py`` or ``scripts/evaluate.py`` sees it."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        return compose(config_name=name, overrides=list(overrides))


def run_label(model: str, index: int) -> str:
    return f"sweep_{model}_{index:02d}"


def reference_config(model: str) -> dict:
    """Hyperparameters of the reference run: ``conf/model/<model>.yaml`` over ``conf/train.yaml``."""
    cfg = compose_config("train", [f"model={model}"])
    kwargs = pipeline._plain(cfg.model.kwargs)
    optim = {**pipeline._plain(cfg.train.optim), **pipeline._plain(cfg.model.get("optim"))}
    hidden = kwargs["hidden"]
    return {
        "hidden": [int(h) for h in hidden] if isinstance(hidden, list) else int(hidden),
        "history": int(kwargs["history"]),
        "use_a_prev": bool(kwargs["use_a_prev"]),
        "lr": float(optim["lr"]),
        "weight_decay": float(optim["weight_decay"]),
        "batch_size": int(optim["batch_size"]),
    }


def _model_code(model: str) -> int:
    """Stable integer of a model name for the seed sequence (``hash`` of a string is salted)."""
    return zlib.crc32(model.encode("utf-8"))


def sample_config(model: str, seed: int, index: int) -> dict:
    """Configuration ``index >= 1``, drawn from its own generator ``[seed, model, index]``: it
    depends neither on ``n`` nor on the other configurations."""
    rng = np.random.default_rng([int(seed), _model_code(model), int(index)])
    width = int(rng.choice(WIDTHS))
    hidden = [width] * int(rng.choice(MLP_LAYERS)) if model == "mlp" else width
    low, high = np.log(LR_RANGE)
    return {
        "hidden": hidden,
        "history": int(rng.choice(HISTORIES)),
        "use_a_prev": bool(rng.choice(A_PREV)),
        "lr": float(f"{math.exp(rng.uniform(low, high)):.3g}"),
        "weight_decay": float(rng.choice(WEIGHT_DECAYS)),
        "batch_size": int(rng.choice(BATCH_SIZES)),
    }


def sample_configs(model: str, n: int, seed: int) -> list[dict]:
    """Configurations ``0 .. n-1`` with index and label; 0 is the reference configuration."""
    if model not in MODELS:
        raise SweepError(f"model {model!r} has no search space; sweeps exist for {', '.join(MODELS)}")
    draws = [reference_config(model), *(sample_config(model, seed, i) for i in range(1, n))]
    return [{"index": i, "label": run_label(model, i), **config} for i, config in enumerate(draws)]


def search_space(model: str) -> dict:
    hidden = {"width": list(WIDTHS)}
    if model == "mlp":
        hidden.update(layers=list(MLP_LAYERS), rule="one width for every hidden layer")
    return {
        "hidden": hidden,
        "history": list(HISTORIES),
        "lr": {"log_uniform": list(LR_RANGE), "significant_digits": 3},
        "weight_decay": list(WEIGHT_DECAYS),
        "batch_size": list(BATCH_SIZES),
        "use_a_prev": list(A_PREV),
        "budget": "epochs, patience and batches per epoch of conf/train.yaml",
    }


def write_configs(sweep: Sweep) -> list[dict]:
    """Sample the configurations and write ``configs.json``; refuse to mix two sweeps in one directory."""
    configs = sample_configs(sweep.model, sweep.n, sweep.seed)
    payload = {"experiment": "E5", **sweep.identity(), "n": sweep.n, "space": search_space(sweep.model), "configs": configs}
    path = sweep.directory / "configs.json"
    if path.exists():
        old = read_json(path)
        changed = [key for key, value in sweep.identity().items() if old.get(key) != value]
        common = min(len(old.get("configs", [])), len(configs))
        if not changed and old.get("configs", [])[:common] != configs[:common]:
            changed.append("configs")
        if changed:
            raise SweepError(
                f"{path} belongs to another sweep ({', '.join(changed)} differ); pass another root=... "
                "or remove the directory"
            )
    write_json(path, payload)
    return configs


def _literal(value) -> str:
    """Hydra override literal of a plain value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_literal(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def run_overrides(sweep: Sweep, config: Mapping) -> list[str]:
    """Hydra overrides of ``scripts/train.py`` and ``scripts/evaluate.py`` for one configuration."""
    return [
        f"dataset={sweep.dataset}",
        f"model={sweep.model}",
        f"++model.label={config['label']}",
        f"paths.results='{Path(sweep.root).as_posix()}'",
        f"train.version={sweep.version}",
        f"train.split={sweep.split}",
        f"train.fold={sweep.fold}",
        f"train.seed={sweep.seed}",
        *(f"{key}={_literal(config[name])}" for name, key in KEYS.items()),
        *sweep.overrides,
    ]


def hidden_text(hidden) -> str:
    """``64`` for a recurrent layer, ``64x64`` for the hidden layers of an MLP."""
    return "x".join(str(h) for h in hidden) if isinstance(hidden, (list, tuple)) else str(hidden)


# ----------------------------------------------------------------------------- bookkeeping


def call_script(script: str, args: Sequence[str], timeout: float = TIMEOUT, attempts: int = ATTEMPTS) -> str | None:
    """Run ``scripts/<script>`` in a subprocess; ``None`` on success, else the last exit status."""
    path = REPO_ROOT / "scripts" / script
    code: int | str = "not run"
    for attempt in range(1, attempts + 1):
        started = time.time()
        try:  # a CUDA call can hang under memory pressure on the shared GPU: cut and retry
            code = subprocess.call([sys.executable, str(path), *args], timeout=timeout)
        except subprocess.TimeoutExpired:
            code = "timeout"
        print(f"### {script} {' '.join(args)}: attempt {attempt}, exit {code}, {time.time() - started:.0f} s", flush=True)
        if code == 0:
            return None
    return str(code)


def record_failures(sweep: Sweep, stage: str, failures: Sequence[Mapping]) -> None:
    """``failures.json``: the failures of the last run of every stage (an empty list: none)."""
    path = sweep.directory / "failures.json"
    data = _json(path) or {}
    data[stage] = list(failures)
    write_json(path, data)


def _listed(failures: Sequence[Mapping]) -> str:
    return "none" if not failures else ", ".join(f"{f['label']} ({f['status']})" for f in failures)


# ----------------------------------------------------------------------------- stage train


def is_trained(sweep: Sweep, config: Mapping) -> bool:
    """A finished run with the configuration hash that ``scripts/train.py`` computes (and no ``train.force``)."""
    path = sweep.run_dir(config["label"]) / "train.json"
    if not path.exists():
        return False
    cfg = compose_config("train", run_overrides(sweep, config))
    if cfg.train.get("force", False):
        return False
    return read_json(path).get("config_hash") == config_hash(pipeline.run_config(cfg))


def stage_train(sweep: Sweep, configs: Sequence[Mapping], runner: Runner = call_script) -> list[dict]:
    """One ``scripts/train.py`` process per configuration; finished runs are skipped."""
    failures, ran, kept = [], 0, 0
    for config in configs:
        if is_trained(sweep, config):
            kept += 1
            continue
        status = runner("train.py", run_overrides(sweep, config))
        if status is None and not (sweep.run_dir(config["label"]) / "train.json").exists():
            status = "no train.json"
        if status is None:
            ran += 1
        else:
            failures.append({"label": config["label"], "status": status})
    record_failures(sweep, "train", failures)
    print(f"train: {len(configs)} configurations, {ran} trained now, {kept} finished before; failed: {_listed(failures)}",
          flush=True)
    return failures


# ----------------------------------------------------------------------------- stage validate


def validation_events(cfg, max_events: int | None = VAL_EVENTS) -> list:
    """Usable events of the validation part of the run's split and fold (``test_events`` of
    ``trafficloop/protocols/evaluate.py`` on the part ``val``), at most ``max_events`` of them drawn
    by the sampler of the training with the run seed, in split order. With
    ``train.max_val_events`` equal to ``max_events`` they are the events of the validation windows
    of the early stopping, unless a validation event is shorter than the warm-up plus three
    samples (the training needs one sample more; on NGSIM they are the same 2,000 events)."""
    t = cfg.train
    events = pipeline.load_events(cfg, t.version)
    split = Split.load(pipeline.split_path(cfg, t.split))
    usable = pipeline._usable(pipeline.part_events(events, split, int(t.fold), "val"), int(cfg.protocol.warmup))
    return training._sample(usable, max_events, int(t.seed))


def validation_settings(sweep: Sweep, cfg) -> dict:
    """What the validation criteria depend on besides the checkpoint (hashed into ``sweep.json``)."""
    return {
        "criterion": "E5 validation",
        "part": "val",
        "horizon_s": HORIZON,
        "max_events": int(sweep.val_events),
        "warmup": int(cfg.protocol.warmup),
        "clip": [float(c) for c in cfg.protocol.clip],
        "version": sweep.version,
        "split": sweep.split,
        "fold": int(sweep.fold),
        "seed": int(sweep.seed),
        "precision": "full",  # TF32 off, as in the protocols
    }


def validation_criteria(model, events: Sequence, cfg, device: str) -> dict:
    """The closed-loop criterion (10 s windows of P1) and, for comparison, the one-step error of P0
    on the same validation events (every configuration on the same instants)."""
    import torch

    from trafficloop.protocols.p0 import evaluate_p0, pooled_rmse
    from trafficloop.protocols.p1 import evaluate_horizon, pooled_horizon

    warmup, clip = int(cfg.protocol.warmup), tuple(float(c) for c in cfg.protocol.clip)
    windows = pooled_horizon(evaluate_horizon(model, events, HORIZON, warmup, clip, int(cfg.p1.batch_windows), device))
    one_step = pooled_rmse(evaluate_p0(model, events, warmup, device=device, dtype=torch.float32, clip=clip))
    return {
        "val_rmse_s_10s": windows["rmse_s"],
        "val_rmse_v_10s": windows["rmse_v"],
        "val_collision_rate_10s": windows["collision_rate"],
        "val_event_collision_rate_10s": windows["event_collision_rate"],
        "val_n_windows_10s": windows["n_windows"],
        "val_n_events": len(events),
        "val_rmse_a_p0": one_step["rmse_a"],
    }


def _validated(directory: Path, key: str) -> bool:
    """``sweep.json`` exists for these settings and for the current checkpoint."""
    done, train = _json(directory / "sweep.json"), _json(directory / "train.json")
    return bool(done and train) and done.get("config_hash") == key and (
        (done.get("model") or {}).get("config_hash") == train.get("config_hash")
    )


def validate_runs(sweep: Sweep, configs: Sequence[Mapping], cfg) -> list[dict]:
    """Write ``sweep.json`` into every trained run whose criteria are missing or stale."""
    import torch

    from trafficloop.models.registry import load_model

    key = config_hash(validation_settings(sweep, cfg))
    todo, failures = [], []
    for config in configs:
        directory = sweep.run_dir(config["label"])
        if not (directory / "train.json").exists() or _validated(directory, key):
            continue
        if not (directory / "model.pt").exists():
            failures.append({"label": config["label"], "status": "no checkpoint (model.pt)"})
            continue
        todo.append(config)
    if not todo:
        return failures
    # full float32 as in the protocols (trafficloop/protocols/evaluate.py): no TF32 noise
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(int(cfg.train.threads))
    device = pipeline._device(cfg.train.device)
    events = validation_events(cfg, sweep.val_events)
    for config in todo:
        directory = sweep.run_dir(config["label"])
        started = time.time()
        try:
            model, _ = load_model(directory / "model.pt", device)
            criteria = validation_criteria(model, events, cfg, device)
        except Exception as error:  # the GPU is shared: record the run and go on with the next one
            failures.append({"label": config["label"], "status": f"{type(error).__name__}: {error}"})
            continue
        train = read_json(directory / "train.json")
        criteria["val_rmse_a"] = float(train["training"]["best_val_rmse_a"])
        write_json(directory / "sweep.json", {
            "experiment": "E5",
            "label": config["label"],
            "hyperparameters": {name: config[name] for name in HYPERPARAMETERS},
            "config": validation_settings(sweep, cfg),
            "config_hash": key,
            "model": {"label": config["label"], "class": sweep.model, "config_hash": train.get("config_hash")},
            "criteria": criteria,
            "git_revision": git_revision(),
            "seconds": round(time.time() - started, 1),
        })
        print(f"validate: {config['label']} val_rmse_a {criteria['val_rmse_a']:.4f}, "
              f"val_rmse_s_10s {criteria['val_rmse_s_10s']:.4f} ({time.time() - started:.0f} s)", flush=True)
    return failures


def collect(sweep: Sweep, configs: Sequence[Mapping]) -> pd.DataFrame:
    """One row per configuration: hyperparameters, both criteria, budget used and status
    (``missing``: no run; ``trained``: no current ``sweep.json``; ``validated``)."""
    rows = []
    for config in configs:
        directory = sweep.run_dir(config["label"])
        train, done = _json(directory / "train.json"), _json(directory / "sweep.json")
        fresh = bool(train and done) and (done.get("model") or {}).get("config_hash") == train.get("config_hash")
        criteria = done.get("criteria", {}) if fresh else {}
        rows.append({
            "index": config["index"],
            "label": config["label"],
            "hidden": hidden_text(config["hidden"]),
            **{name: config[name] for name in HYPERPARAMETERS if name != "hidden"},
            "val_rmse_a": _num(train, "training", "best_val_rmse_a"),
            **{name: _num(criteria, name)
               for name in ("val_rmse_s_10s", "val_collision_rate_10s", "val_rmse_a_p0", "val_rmse_v_10s")},
            "epochs": _num(train, "training", "epochs_run"),
            "seconds": _num(train, "training", "seconds"),
            "n_parameters": _num(train, "n_parameters"),
            "status": "missing" if train is None else ("validated" if criteria else "trained"),
        })
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    for column in ("epochs", "n_parameters"):
        frame[column] = frame[column].astype("Int64")
    return frame


def write_table(sweep: Sweep, frame: pd.DataFrame) -> Path:
    """``sweep.csv``: the rows of :func:`collect`."""
    path = sweep.directory / "sweep.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def stage_validate(sweep: Sweep, configs: Sequence[Mapping]) -> pd.DataFrame:
    """Both criteria of every trained run (``sweep.json``) and the table ``sweep.csv``."""
    cfg = compose_config("evaluate", run_overrides(sweep, configs[0]))
    failures = validate_runs(sweep, configs, cfg)
    record_failures(sweep, "validate", failures)
    frame = collect(sweep, configs)
    path = write_table(sweep, frame)
    counts = frame["status"].value_counts().to_dict()
    print(f"validate: {counts.get('validated', 0)} validated, {counts.get('trained', 0)} trained without criteria, "
          f"{counts.get('missing', 0)} missing; failed: {_listed(failures)}; {path}", flush=True)
    return frame


# ----------------------------------------------------------------------------- stage select


def _entry(row: Mapping, config: Mapping) -> dict:
    return {
        "index": int(row["index"]),
        "label": str(row["label"]),
        "hyperparameters": {name: config[name] for name in HYPERPARAMETERS},
        **{name: _num(row, name) for name in ("val_rmse_a", "val_rmse_s_10s", "val_collision_rate_10s", "val_rmse_a_p0")},
    }


def select(frame: pd.DataFrame, configs: Sequence[Mapping]) -> dict:
    """The validated configurations with the smallest one-step and the smallest closed-loop
    criterion (ties: the lower index), and Kendall's tau-b between the two criteria."""
    a = frame["val_rmse_a"].to_numpy(dtype=np.float64)
    s = frame["val_rmse_s_10s"].to_numpy(dtype=np.float64)
    candidates = frame[np.isfinite(a) & np.isfinite(s)]
    if candidates.empty:
        raise SweepError("no validated run: run the stages train and validate first")
    by_index = {int(c["index"]): c for c in configs}
    out: dict = {}
    for role, column in CRITERIA.items():
        best = candidates.loc[candidates[column].astype(float).idxmin()].to_dict()
        out[role] = {"criterion": column, **_entry(best, by_index[int(best["index"])])}
    tau, p = kendall_tau(dict(zip(candidates["label"], candidates["val_rmse_a"].astype(float))),
                         dict(zip(candidates["label"], candidates["val_rmse_s_10s"].astype(float))))
    out["coincide"] = out["one_step"]["label"] == out["closed_loop"]["label"]
    out["n_validated"] = int(len(candidates))
    out["kendall_tau_criteria"] = {"tau": tau, "p": p}
    reference = frame[frame["index"] == 0]
    out["reference"] = _entry(reference.iloc[0].to_dict(), by_index[0]) if len(reference) else None
    return out


def stage_select(sweep: Sweep, configs: Sequence[Mapping]) -> dict:
    """``selection.json`` from the criteria in the run directories."""
    selection = {"experiment": "E5", **sweep.identity(), "n": sweep.n, **select(collect(sweep, configs), configs)}
    write_json(sweep.directory / "selection.json", selection)
    one, closed = selection["one_step"], selection["closed_loop"]
    print(f"select: one step -> {one['label']} (val_rmse_a {one['val_rmse_a']:.4f}), closed loop -> "
          f"{closed['label']} (val_rmse_s_10s {closed['val_rmse_s_10s']:.4f}); "
          f"{'the same run' if selection['coincide'] else 'different runs'}", flush=True)
    return selection


# ----------------------------------------------------------------------------- stage evaluate


def evaluation_roles(selection: Mapping | None, configs: Sequence[Mapping]) -> dict[str, list[str]]:
    """label -> roles of the runs scored with the protocols: configuration 0 and the two selections."""
    roles: dict[str, list[str]] = {configs[0]["label"]: [ROLES["reference"]]}
    for role in CRITERIA:
        entry = (selection or {}).get(role)
        if entry:
            roles.setdefault(entry["label"], []).append(ROLES[role])
    return roles


def stage_evaluate(sweep: Sweep, configs: Sequence[Mapping], runner: Runner = call_script) -> list[dict]:
    """``scripts/evaluate.py`` with the protocols of the sweep on configuration 0 and the selected runs."""
    selection = _json(sweep.directory / "selection.json")
    if selection is None:
        raise SweepError(f"no {sweep.directory / 'selection.json'}: run the stage select first")
    cfg = compose_config("evaluate", run_overrides(sweep, configs[0]))
    unknown = [name for name in sweep.protocols if name not in cfg]
    if unknown:
        raise SweepError(f"protocols {unknown} are not defined in conf/evaluate.yaml")
    by_label = {c["label"]: c for c in configs}
    failures = []
    for label in evaluation_roles(selection, configs):
        if label not in by_label:
            failures.append({"label": label, "status": "not among the configurations of this n"})
        elif not (sweep.run_dir(label) / "model.pt").exists():
            failures.append({"label": label, "status": "not trained"})
        else:
            args = [*run_overrides(sweep, by_label[label]), f"protocols=[{','.join(sweep.protocols)}]"]
            status = runner("evaluate.py", args)
            if status is not None:
                failures.append({"label": label, "status": status})
    record_failures(sweep, "evaluate", failures)
    print(f"evaluate: protocols {','.join(sweep.protocols)}; failed: {_listed(failures)}", flush=True)
    return failures


# ----------------------------------------------------------------------------- stage report


def protocol_scores(directory: Path) -> dict:
    """Test-part scores of an evaluated run: P0 from ``train.json`` and the summaries of P1-P3."""
    scores = {"p0_rmse_a": _num(_json(directory / "train.json"), "test", "p0", "rmse_a")}
    for column, (name, keys) in SCORES.items():
        scores[column] = _num(_summary(directory, name), *keys)
    scores["stale"] = ",".join(stale_protocols(directory))
    return scores


def _fmt(value, digits: int = 4) -> str:
    """Table cell: ``-`` for a missing value, integers as such, floats with ``digits`` significant digits."""
    if value is None or (not isinstance(value, (str, list, tuple)) and pd.isna(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        return value or "-"
    number = float(value)
    return f"{number:.{digits}g}" if math.isfinite(number) else "-"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(r) + " |" for r in rows)]


def _shown(path: Path) -> str:
    """``path`` relative to the repository when it lies inside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


HYPER_HEADER = ("hidden", "history", "a_prev", "lr", "weight decay", "batch")


def _hyper_cells(config: Mapping) -> list[str]:
    return [hidden_text(config["hidden"]), str(config["history"]), "yes" if config["use_a_prev"] else "no",
            _fmt(config["lr"], 3), _fmt(config["weight_decay"], 3), str(config["batch_size"])]


def report_markdown(sweep: Sweep, configs: Sequence[Mapping], frame: pd.DataFrame, selection: Mapping | None,
                    scores: Mapping[str, Mapping], failures: Mapping | None = None) -> str:
    """``REPORT.md``: the selections, every configuration and the protocols of the evaluated runs."""
    counts = frame["status"].value_counts().to_dict()
    lines = [
        f"# E5: hyperparameter sweep of `{sweep.model}` on `{sweep.dataset}`",
        "",
        f"Written by `scripts/sweep.py` (design: `docs/experiments.md`, section 4). Data version `{sweep.version}`, "
        f"`{sweep.split}` split, fold {sweep.fold}, seed {sweep.seed}; {len(configs)} configurations (0 = the reference "
        f"configuration of `conf/model/{sweep.model}.yaml`), {len(frame) - counts.get('missing', 0)} trained, "
        f"{counts.get('validated', 0)} validated. Runs: `{_shown(sweep.run_dir(f'sweep_{sweep.model}_<kk>'))}`.",
        "",
    ]
    reference = _json(sweep.run_dir(configs[0]["label"]) / "train.json")
    if reference:
        training_log = reference.get("training", {})
        budget = training_log.get("config", {})
        lines += [
            f"Training budget of every run (`train.json` of configuration 0): `epochs` {budget.get('epochs')}, "
            f"`batches_per_epoch` {budget.get('batches_per_epoch')}, `patience` {budget.get('patience')}; "
            f"{training_log.get('n_train_events')} training events.",
            "",
        ]
    lines += [
        "Both criteria come from the validation part of the split, never from the test part. `val_rmse_a`: the "
        "one-step RMSE of the acceleration kept by the early stopping (`best_val_rmse_a` of `train.json`, m/s^2). "
        f"`val_rmse_s_10s`: the spacing RMSE of the closed-loop 10 s windows of P1 on at most {sweep.val_events} "
        "validation events (m). `val_rmse_a_p0`: the one-step RMSE of P0 on the same events, every configuration on "
        "the same instants (the windows of the early stopping depend on the batch size).",
        "",
        "## Selection",
        "",
    ]
    by_index = {int(c["index"]): c for c in configs}
    if selection:
        rows = []
        for role, column in (("one_step", "val_rmse_a"), ("closed_loop", "val_rmse_s_10s"), ("reference", None)):
            entry = selection.get(role)
            if not entry or entry["index"] not in by_index:
                continue
            name = f"{ROLES[role]} (`{column}`)" if column else "reference (configuration 0)"
            rows.append([name, f"{entry['index']:02d}", *_hyper_cells(by_index[entry["index"]]),
                         _fmt(entry.get("val_rmse_a")), _fmt(entry.get("val_rmse_a_p0")), _fmt(entry.get("val_rmse_s_10s"))])
        lines += _table(["Selection", "#", *HYPER_HEADER, "val_rmse_a", "val_rmse_a_p0", "val_rmse_s_10s"], rows)
        tau = selection.get("kendall_tau_criteria") or {}
        same = "the same configuration" if selection.get("coincide") else "different configurations"
        lines += [
            "",
            f"The two criteria select {same}. Kendall's tau-b between them over the {selection.get('n_validated')} "
            f"validated configurations: {_fmt(tau.get('tau'), 3)} (p = {_fmt(tau.get('p'), 2)}).",
            "",
        ]
    else:
        lines += ["No selection yet (stage select).", ""]
    roles = evaluation_roles(selection, configs) if selection else {}
    marks = {label: ", ".join(r) for label, r in roles.items()}
    rows = [
        [f"{config['index']:02d}", *_hyper_cells(config), _fmt(row["val_rmse_a"]), _fmt(row["val_rmse_a_p0"]),
         _fmt(row["val_rmse_s_10s"]), _fmt(row["val_collision_rate_10s"], 3), _fmt(row["epochs"]), _fmt(row["seconds"], 3),
         row["status"], marks.get(config["label"], "")]
        for row, config in zip(frame.to_dict("records"), configs)
    ]
    lines += ["## Configurations", ""]
    lines += _table(["#", *HYPER_HEADER, "val_rmse_a", "val_rmse_a_p0", "val_rmse_s_10s", "val coll. 10 s", "epochs",
                     "fit s", "status", "role"], rows)
    lines += [
        "",
        "## Protocols of the evaluated runs (test part)",
        "",
        "P0 `rmse_a` from `train.json`; P1 spacing RMSE of the 10 s windows and of the whole event (`p1.json`); P2 "
        "collisions of the 100-vehicle platoons on the in-range profiles (`collisions_100_in_range`); P3 "
        "`unstable_fraction`. `stale`: protocol files computed from another checkpoint.",
        "",
    ]
    rows = [[label, ", ".join(r), *(_fmt(scores.get(label, {}).get(c)) for c in ("p0_rmse_a", *SCORES)),
             scores.get(label, {}).get("stale") or "-"] for label, r in roles.items()]
    if rows:
        lines += _table(["Run", "Role", "P0 rmse_a", "P1 10 s rmse_s", "P1 whole rmse_s", "P2 collisions",
                         "P3 unstable fraction", "stale"], rows)
    else:
        lines += ["No evaluated run yet (stages select and evaluate)."]
    listed = [(stage, f) for stage, items in (failures or {}).items() for f in items]
    if listed:
        lines += ["", "## Failures of the last run of every stage", ""]
        lines += _table(["Stage", "Run", "Status"], [[stage, f["label"], str(f["status"])] for stage, f in listed])
    return "\n".join(lines) + "\n"


def stage_report(sweep: Sweep, configs: Sequence[Mapping]) -> Path:
    """``REPORT.md`` from the run directories, ``selection.json`` and ``failures.json``."""
    frame = collect(sweep, configs)
    selection = _json(sweep.directory / "selection.json")
    roles = evaluation_roles(selection, configs) if selection else {}
    scores = {label: protocol_scores(sweep.run_dir(label)) for label in roles}
    text = report_markdown(sweep, configs, frame, selection, scores, _json(sweep.directory / "failures.json"))
    path = sweep.directory / "REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"report: {path}", flush=True)
    return path


def run_stages(sweep: Sweep, stages: Sequence[str] = STAGES, runner: Runner = call_script) -> None:
    """Write ``configs.json`` and run the stages in the given order."""
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SweepError(f"unknown stages {unknown}; the stages are {', '.join(STAGES)} or all")
    configs = write_configs(sweep)
    for stage in stages:
        started = time.time()
        if stage == "train":
            stage_train(sweep, configs, runner)
        elif stage == "validate":
            stage_validate(sweep, configs)
        elif stage == "select":
            stage_select(sweep, configs)
        elif stage == "evaluate":
            stage_evaluate(sweep, configs, runner)
        else:
            stage_report(sweep, configs)
        print(f"### stage {stage}: {time.time() - started:.0f} s", flush=True)
