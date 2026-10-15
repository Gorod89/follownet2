"""Experiment E5: hyperparameter sweep of one reference network with two selection criteria.

    python scripts/sweep.py dataset=ngsim model=lstm n=16 seed=0 [stage=all]
    python scripts/sweep.py dataset=ngsim model=mlp n=16 stage=train|validate|select|evaluate|report
                            [version=reconstructed] [split=driver] [fold=0] [root=results/sweeps]
                            [protocols=p1,p1s,p2,p3,p4] [val_events=2000] [<hydra override> ...]

Stages (``docs/experiments.md``, section 4): ``train`` (one ``scripts/train.py`` process per
configuration, label ``sweep_<model>_<kk>``; finished runs are skipped), ``validate`` (both
criteria on the validation part: ``sweep.json`` per run and ``sweep.csv``), ``select``
(``selection.json``), ``evaluate`` (``scripts/evaluate.py`` on configuration 0 and the two
selected runs) and ``report`` (``REPORT.md``); ``all`` runs them in this order, a comma list
runs several. ``seed`` is the training seed and the seed of the sampling. The files of the sweep
are written to ``<root>/<dataset>/<model>/``, the runs to ``<root>/models/...``, never to
``results/models``. Arguments with a dot (``train.optim.epochs=1``) are hydra overrides of every
run; the keys the sweep sets itself are refused.
"""

from __future__ import annotations

import sys

from trafficloop.experiments.sweep import PROTOCOLS, ROOT, STAGES, VAL_EVENTS, Sweep, SweepError, run_stages

DEFAULTS = {"dataset": "ngsim", "model": "lstm", "n": "16", "seed": "0", "stage": "all", "version": "reconstructed",
            "split": "driver", "fold": "0", "root": ROOT, "protocols": ",".join(PROTOCOLS), "val_events": str(VAL_EVENTS)}


def _items(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.strip("[]").split(",") if v.strip())


def arguments(argv: list[str] | None = None) -> tuple[Sweep, tuple[str, ...]]:
    """The sweep and the stages of a command line."""
    args, overrides = dict(DEFAULTS), []
    for item in sys.argv[1:] if argv is None else argv:
        key, sep, value = item.partition("=")
        if sep and ("." in key or key.startswith(("+", "~"))):
            overrides.append(item)
        elif sep and key in DEFAULTS:
            args[key] = value
        else:
            raise SystemExit(f"unknown argument {item!r}; usage: sweep.py " + " ".join(f"[{k}=...]" for k in DEFAULTS)
                             + " [<hydra override with a dot> ...]")
    stages = STAGES if args["stage"] == "all" else _items(args["stage"])
    try:
        sweep = Sweep(
            dataset=args["dataset"], model=args["model"], n=int(args["n"]), seed=int(args["seed"]),
            version=args["version"], split=args["split"], fold=int(args["fold"]), root=args["root"],
            overrides=tuple(overrides), protocols=_items(args["protocols"]), val_events=int(args["val_events"]),
        )
    except (SweepError, ValueError) as error:
        raise SystemExit(str(error)) from None
    return sweep, stages


def main(argv: list[str] | None = None) -> None:
    sweep, stages = arguments(argv)
    try:
        run_stages(sweep, stages)
    except SweepError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
