"""Extract events of one dataset and write the raw and the reconstructed version.

    python scripts/build_events.py dataset=<name> [key=value ...]
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from follownet2.pipeline import build_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    show("build_dataset: " + cfg.dataset.name, build_dataset(cfg))


if __name__ == "__main__":
    main()
