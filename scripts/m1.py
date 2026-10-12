"""Milestone 1 for one dataset: events (both versions), splits, baselines, calibration.

    python scripts/m1.py dataset=<name> [steps=[build,splits,baselines,calibration]]
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from follownet2 import pipeline
from follownet2.utils import resolve_path, write_json

STEPS = {
    "build": pipeline.build_dataset,
    "splits": pipeline.build_splits,
    "baselines": pipeline.run_p0_baselines,
    "calibration": pipeline.run_calibration,
}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    summary = {}
    for step in cfg.get("steps", list(STEPS)):
        summary[step] = STEPS[step](cfg)
        show(f"{step}: {cfg.dataset.name}", summary[step])
        write_json(resolve_path(cfg.paths.results) / "m1" / f"{cfg.dataset.name}_{step}.json", summary[step])


if __name__ == "__main__":
    main()
