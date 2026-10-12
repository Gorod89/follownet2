"""Reconstruct and confirm the near-miss episodes of one dataset (empirical suite of P4).

    python scripts/reconstruct_episodes.py dataset=<name> [max_events=<n>] [out=<tag>] [workers=<n>] [key=value ...]

Reads ``data/events/<name>_nearmiss/raw``, reconstructs every episode piecewise with the
dataset's reconstruction settings (``pipeline.reconstruction_config``), re-evaluates the
triggers of the ``episodes`` section on the reconstructed series and writes the episodes with a
confirmed trigger at or after the end of the warm-up (``protocol.warmup``) to
``data/events/<name>_nearmiss/reconstructed`` with the report in the manifest.

``max_events=<n>`` draws a sample of ``n`` episodes (seeded by ``seed``); ``out=<tag>`` writes to
``reconstructed_<tag>`` instead (``out=sample`` for a trial run). Both are plain arguments, not
hydra keys: they are taken off the command line before hydra parses it.
"""

from __future__ import annotations

import sys
import time

import hydra
from _common import show
from omegaconf import DictConfig

# the process pool of the reconstruction re-imports this module in every worker: the heavy
# imports (torch through the pipeline) happen inside run()
ARGS: dict = {"max_events": None, "out": None}


def _take_args(argv: list[str]) -> list[str]:
    rest = []
    for arg in argv:
        key, _, value = arg.partition("=")
        if key in ARGS and value:
            ARGS[key] = int(value) if key == "max_events" else value
        else:
            rest.append(arg)
    return rest


def episode_config(cfg):
    from follownet2.data.rare_events import EpisodeConfig
    from follownet2.utils import to_plain

    mining = dict(to_plain(cfg.get("episodes")) or {})
    mining.pop("enabled", None)
    if "triggers" in mining:
        mining["triggers"] = tuple(mining["triggers"])
    return EpisodeConfig(**mining)


def run(cfg, max_events: int | None = None, out: str | None = None) -> dict:
    from dataclasses import asdict

    import numpy as np

    from follownet2 import pipeline
    from follownet2.data.episodes import JUMP_THRESHOLD, MIN_PIECE, reconstruct_episodes
    from follownet2.data.reconstruction import summarise
    from follownet2.data.schema import EventSet, events_dir
    from follownet2.utils import config_hash, git_revision, require_free_space, resolve_path

    name = cfg.dataset.name + pipeline.NEAR_MISS_SUFFIX
    root = resolve_path(cfg.paths.events)
    source = events_dir(root, name, "raw")
    events = list(EventSet.from_parquet(source))
    n_total = len(events)
    if max_events is not None and len(events) > max_events:
        index = np.sort(np.random.default_rng([cfg.seed, 41]).choice(len(events), size=max_events, replace=False))
        events = [events[i] for i in index]
    rec_cfg = pipeline.reconstruction_config(cfg)
    episode_cfg = episode_config(cfg)
    warmup = int(cfg.protocol.warmup)
    started = time.time()
    done, report = reconstruct_episodes(events, rec_cfg, workers=int(cfg.workers), episode_cfg=episode_cfg, warmup=warmup)
    seconds = time.time() - started
    report["seconds"] = round(seconds, 1)
    report["seconds_per_episode"] = seconds / max(1, len(events))
    report["workers"] = int(cfg.workers)
    episodes = EventSet(done)
    episodes.validate()
    config = {
        "reconstruction": asdict(rec_cfg),
        "episodes": asdict(episode_cfg),
        "jump_threshold": JUMP_THRESHOLD,
        "min_piece": MIN_PIECE,
        "warmup": warmup,
        "sample": None if max_events is None else {"max_events": max_events, "seed": cfg.seed, "n_total": n_total},
    }
    manifest = {
        "dataset": name,
        "source": source.relative_to(root).as_posix(),
        "config": config,
        "config_hash": config_hash(config),
        "git_revision": git_revision(),
        "report": report,
        "reconstruction": summarise(done),
    }
    if out is None:
        directory = pipeline.write_events(cfg, episodes, name, "reconstructed", manifest)
    else:
        directory = root / name / f"reconstructed_{out}"
        n_samples = sum(len(ev) for ev in episodes)
        columns = 6 if cfg.storage.compact else 8
        require_free_space(directory, n_samples * columns * 8, reserve=int(cfg.storage.reserve_mb) << 20)
        episodes.to_parquet(directory, manifest, compact=cfg.storage.compact)
    return {"directory": str(directory), **report, "reconstruction": manifest["reconstruction"]}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    show(f"reconstruct_episodes: {cfg.dataset.name}", run(cfg, **ARGS))


if __name__ == "__main__":
    sys.argv[1:] = _take_args(sys.argv[1:])
    main()
