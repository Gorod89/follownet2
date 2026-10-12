"""Build the paired splits of E2 (decision R2) from the driver split of a dataset.

    python scripts/make_paired_splits.py dataset=<name> [key=value ...]

Writes ``<dataset>_paired-clean.json``, ``<dataset>_paired-leaky.json`` and
``<dataset>_paired_report.json`` next to the driver split (docs/experiments.md, section 2).
The leakage check runs on the clean arm only: the leaky arm fails it by design.
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from follownet2.experiments.paired import build_paired_splits


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    report = build_paired_splits(cfg)
    show(f"paired splits: {cfg.dataset.name} ({', '.join(report['files'].values())})",
         [{"fold": int(fold), **stats} for fold, stats in report["folds"].items()])
    show("pooled over the folds", report["pooled"])
    seen = {fold: f["test_followers_seen_in_training"] for fold, f in report["leakage_clean"].items()}
    show("clean arm: leakage check passed, test followers seen in training per fold", seen)


if __name__ == "__main__":
    main()
