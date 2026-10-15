"""P0 for persistence, constant speed and the recorded acceleration.

    python scripts/run_baselines.py dataset=<name> [key=value ...]
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from trafficloop.pipeline import run_p0_baselines


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    show("run_p0_baselines: " + cfg.dataset.name, run_p0_baselines(cfg))


if __name__ == "__main__":
    main()
