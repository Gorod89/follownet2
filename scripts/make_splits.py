"""Build the event, driver, site and regime splits and run the leakage check.

    python scripts/make_splits.py dataset=<name> [key=value ...]
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from trafficloop.pipeline import build_splits


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    show("build_splits: " + cfg.dataset.name, build_splits(cfg))


if __name__ == "__main__":
    main()
