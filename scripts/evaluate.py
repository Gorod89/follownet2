"""Run the protocols on one trained run.

    python scripts/evaluate.py dataset=<name> model=<name> ["protocols=[p1,p3]"] [force=true]
    python scripts/evaluate.py dataset=ngsim model=lstm "p1.transfer_datasets=[follownet_highd]"

The run is `results/models/<dataset>/<train.version>/<train.split>-fold<train.fold>/<label>/seed<train.seed>`;
every protocol writes `<p>.json` and `<p>.<table>.parquet` there (finished results are kept).
"""

from __future__ import annotations

import hydra
from _common import show
from omegaconf import DictConfig

from trafficloop.protocols.evaluate import evaluate_run


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    results = evaluate_run(cfg)
    show(f"evaluate: {results['model']} on {results['dataset']}", results)


if __name__ == "__main__":
    main()
