"""Calibrate IDM, Gipps and Newell per event and globally on the training parts.

    python scripts/calibrate_mechanistic.py dataset=<name> [key=value ...]
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from follownet2.pipeline import run_calibration


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    show("run_calibration: " + cfg.dataset.name, run_calibration(cfg))


if __name__ == "__main__":
    main()
