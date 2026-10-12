"""Campaigns of milestone 4 (docs/experiments.md, section 1).

    python scripts/m4.py main [datasets=ngsim,follownet_highd] [models=idm,lstm]
    python scripts/m4.py seeds | paired | event | transfer | protocols

Every run is a subprocess of ``scripts/train.py`` or ``scripts/evaluate.py``; a failed run
(the GPU is shared) is repeated up to three times, finished runs are skipped by their
configuration hash, so a campaign can be started again after an interruption.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from m2 import MODELS, TIMEOUT

SUBSET = ("idm", "gipps", "mlp", "lstm", "mlp_loop", "lstm_loop", "ddpg", "stability_mlp")
MAIN_DATASETS = ("ngsim", "follownet_highd")
LEAK_DATASETS = ("ngsim", "openacc")
TRANSFER_DATASETS = ("ngsim", "follownet_highd", "openacc", "waymo_hu", "follownet_ngsim", "tgsim")
PROTOCOLS = ("p1", "p2", "p3", "p4")
SCRIPTS = Path(__file__).parent


def run(script: str, *args: str) -> bool:
    for attempt in range(1, 4):
        started = time.time()
        try:  # a CUDA call can hang under memory pressure on the shared GPU: cut and retry
            code = subprocess.call([sys.executable, str(SCRIPTS / script), *args], timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            code = "timeout"
        print(f"### {script} {' '.join(args)}: attempt {attempt}, exit {code}, {time.time() - started:.0f} s", flush=True)
        if code == 0:
            return True
    return False


def train(dataset: str, model: str, split: str = "driver", fold: int = 0, seed: int = 0) -> bool:
    return run("train.py", f"dataset={dataset}", f"model={model}", f"train.split={split}", f"train.fold={fold}", f"train.seed={seed}")


def protocols(dataset, model, names=PROTOCOLS, transfer=(), split="driver", fold=0, seed=0) -> bool:
    return run(
        "evaluate.py", f"dataset={dataset}", f"model={model}", f"train.split={split}", f"train.fold={fold}", f"train.seed={seed}",
        f"protocols=[{','.join(names)}]", f"p1.transfer_datasets=[{','.join(transfer)}]",
    )


def campaign_main(datasets, models, failed) -> None:
    for dataset in datasets:
        for model in models:
            if not train(dataset, model):
                failed.append(("main", dataset, model))
        others = [d for d in MAIN_DATASETS if d != dataset]
        for model in models:
            if not protocols(dataset, model, PROTOCOLS, others):
                failed.append(("main-protocols", dataset, model))


def campaign_seeds(datasets, models, failed) -> None:
    for seed in (1, 2):
        for dataset in datasets:
            for model in models:
                if not train(dataset, model, seed=seed):
                    failed.append(("seeds", dataset, model, seed))
                elif not protocols(dataset, model, PROTOCOLS, (), seed=seed):
                    failed.append(("seeds-protocols", dataset, model, seed))


def campaign_paired(datasets, models, failed) -> None:
    for dataset in datasets:
        for fold in range(5):
            for arm in ("paired-clean", "paired-leaky"):
                for model in models:
                    if not train(dataset, model, split=arm, fold=fold):
                        failed.append(("paired", dataset, arm, fold, model))
                    elif not protocols(dataset, model, ("p1",), (), split=arm, fold=fold):
                        failed.append(("paired-p1", dataset, arm, fold, model))


def campaign_event(datasets, models, failed) -> None:
    for dataset in datasets:
        for model in models:
            if not train(dataset, model, split="event"):
                failed.append(("event", dataset, model))
            elif not protocols(dataset, model, ("p1",), (), split="event"):
                failed.append(("event-p1", dataset, model))


def campaign_transfer(datasets, models, failed) -> None:
    labels = [*models, "persistence"] if "persistence" not in models else list(models)
    for dataset in datasets:
        if dataset in MAIN_DATASETS:
            continue  # trained by the main campaign
        for model in labels:
            if not train(dataset, model):
                failed.append(("transfer-train", dataset, model))
    for dataset in datasets:
        others = [d for d in TRANSFER_DATASETS if d != dataset]
        for model in labels:
            if not protocols(dataset, model, ("p1",), others):
                failed.append(("transfer-p1", dataset, model))


def campaign_protocols(datasets, models, failed) -> None:
    """Protocols only (after a change of a protocol), fold 0, seed 0."""
    for dataset in datasets:
        others = [d for d in MAIN_DATASETS if d != dataset]
        for model in models:
            if not protocols(dataset, model, PROTOCOLS, others):
                failed.append(("protocols", dataset, model))


def campaign_recompute(datasets, models, failed, names=("p3",)) -> None:
    """Every existing run directory of the datasets (all splits, folds, seeds): the protocols
    ``names`` are recomputed where their configuration or the checkpoint changed, skipped
    otherwise (``scripts/evaluate.py`` compares the hashes)."""
    import re

    root = Path(__file__).resolve().parents[1] / "results" / "models"
    pattern = re.compile(r"^(?P<split>.+)-fold(?P<fold>\d+)$")
    for dataset in datasets:
        for train_json in sorted((root / dataset / "reconstructed").glob("*/*/seed*/train.json")):
            seed_dir = train_json.parent
            label, split_fold = seed_dir.parent.name, seed_dir.parent.parent.name
            match = pattern.match(split_fold)
            if match is None or (models and label not in models):
                continue
            seed = int(seed_dir.name[len("seed"):])
            if not protocols(dataset, label, names, (), match["split"], int(match["fold"]), seed):
                failed.append(("recompute", dataset, split_fold, label, seed))


from follownet2.leaderboard.tables import ABLATION_LABELS as ABLATIONS  # one list, never ranked


def campaign_ablation(datasets, models, failed) -> None:
    """Training ablations of the networks (rollout horizon, history, width, penalty neighbourhood)."""
    for dataset in datasets:
        for model in models:
            if not train(dataset, model):
                failed.append(("ablation", dataset, model))
            elif not protocols(dataset, model, PROTOCOLS, ()):
                failed.append(("ablation-protocols", dataset, model))


def campaign_folds(datasets, models, failed) -> None:
    """Folds 1-4 of the driver split for every model (fold 0 is the main campaign), P1-P4 included."""
    for fold in range(1, 5):
        for dataset in datasets:
            for model in models:
                if not train(dataset, model, fold=fold):
                    failed.append(("folds", dataset, fold, model))
                elif not protocols(dataset, model, PROTOCOLS, (), fold=fold):
                    failed.append(("folds-protocols", dataset, fold, model))


CAMPAIGNS = {
    "main": (campaign_main, MAIN_DATASETS, MODELS),
    "ablation": (campaign_ablation, MAIN_DATASETS, ABLATIONS),
    "folds": (campaign_folds, MAIN_DATASETS, MODELS),
    "recompute": (campaign_recompute, TRANSFER_DATASETS, ()),  # P3 of every run directory
    "seeds": (campaign_seeds, MAIN_DATASETS, MODELS),
    "paired": (campaign_paired, LEAK_DATASETS, SUBSET),
    "event": (campaign_event, LEAK_DATASETS, SUBSET),
    "transfer": (campaign_transfer, TRANSFER_DATASETS, SUBSET),
    "protocols": (campaign_protocols, MAIN_DATASETS, MODELS),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in CAMPAIGNS:
        raise SystemExit(f"usage: m4.py <{'|'.join(CAMPAIGNS)}> [datasets=a,b] [models=a,b]")
    function, datasets, models = CAMPAIGNS[sys.argv[1]]
    names = None
    for arg in sys.argv[2:]:
        if arg.startswith("datasets="):
            datasets = tuple(arg.split("=", 1)[1].split(","))
        elif arg.startswith("models="):
            models = tuple(arg.split("=", 1)[1].split(","))
        elif arg.startswith("protocols="):  # recompute campaign: which protocols to recompute
            names = tuple(arg.split("=", 1)[1].split(","))
    failed: list = []
    if names is not None and function is campaign_recompute:
        function(datasets, models, failed, names)
    else:
        function(datasets, models, failed)
    print("### failed:", failed if failed else "none", flush=True)


if __name__ == "__main__":
    main()
