"""Run the protocols on the trained reference models of one dataset, one process per model.

    python scripts/m3.py dataset=<name> [models=mlp,lstm,...] ["protocols=[p1,p3]"] [key=value ...]

Every model is a run of ``scripts/evaluate.py``; a failed run (the GPU is shared) is repeated
up to three times, and finished protocol results with the same configuration are kept.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from m2 import MODELS


def main() -> None:
    models, overrides = list(MODELS), []
    for arg in sys.argv[1:]:
        if arg.startswith("models="):
            models = arg.split("=", 1)[1].split(",")
        else:
            overrides.append(arg)
    script = str(Path(__file__).with_name("evaluate.py"))
    failed = []
    for model in models:
        for attempt in range(1, 4):
            started = time.time()
            code = subprocess.call([sys.executable, script, f"model={model}", *overrides])
            print(f"### {model}: attempt {attempt}, exit {code}, {time.time() - started:.0f} s", flush=True)
            if code == 0:
                break
        else:
            failed.append(model)
    print("### failed:", failed if failed else "none", flush=True)


if __name__ == "__main__":
    main()
