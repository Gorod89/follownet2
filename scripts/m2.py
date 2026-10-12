"""Train and score the reference models on one dataset, one process per model.

    python scripts/m2.py dataset=<name> [models=mlp,lstm,...] [key=value ...]

Every model is a run of ``scripts/train.py``; a failed run (the GPU is shared) is repeated up
to three times, and finished runs with the same configuration are not repeated.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

MODELS = (
    "persistence", "constant_speed", "idm", "idm_median", "gipps", "gipps_median", "newell", "newell_median",
    "knn", "mlp", "gru", "lstm", "seq2seq", "mdn_lstm", "pidl", "perl", "stability_mlp",
    "ddpg", "gail", "maml_lstm", "diffusion", "mlp_loop", "gru_loop", "lstm_loop",
)
#: seconds per run; the longest regular run (maml_lstm on NGSIM) takes 30 minutes
TIMEOUT = 2 * 3600


def main() -> None:
    models, overrides = list(MODELS), []
    for arg in sys.argv[1:]:
        if arg.startswith("models="):
            models = arg.split("=", 1)[1].split(",")
        else:
            overrides.append(arg)
    script = str(Path(__file__).with_name("train.py"))
    failed = []
    for model in models:
        for attempt in range(1, 4):
            started = time.time()
            try:  # a CUDA call can hang under memory pressure on the shared GPU: cut and retry
                code = subprocess.call([sys.executable, script, f"model={model}", *overrides], timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                code = "timeout"
            print(f"### {model}: attempt {attempt}, exit {code}, {time.time() - started:.0f} s", flush=True)
            if code == 0:
                break
        else:
            failed.append(model)
    print("### failed:", failed if failed else "none", flush=True)


if __name__ == "__main__":
    main()
