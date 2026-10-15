"""Train one model and score it on the test part.

    python scripts/train.py dataset=<name> model=<name> [train.split=driver train.fold=0 ...]
    python scripts/train.py dataset=<name> model=<name> rescore=true     # scores only, no training
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from trafficloop.train.run import label, rescore_model, train_model


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    summary = rescore_model(cfg) if cfg.get("rescore", False) else train_model(cfg)
    summary["training"].pop("history", None)
    show(f"train: {label(cfg)} on {cfg.dataset.name}", summary)


if __name__ == "__main__":
    main()
