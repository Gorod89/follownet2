"""Dataset loaders by name.

A loader module exposes one of

* ``load_pairs(raw_dir, **options) -> Iterable[PairSeries]``: follower series with their
  instantaneous leader; events are cut by :mod:`follownet2.data.extraction`;
* ``load_events(raw_dir, **options) -> list[Event]``: events already in schema form.

and the constant ``POSITION_SOURCE`` ("track": vehicles have their own longitudinal
coordinate; "gap": the source measures the gap and positions are integrated from speeds).
"""

from __future__ import annotations

import importlib
from types import ModuleType

LOADERS = {
    "synthetic": "follownet2.data.loaders.synthetic",
    "follownet": "follownet2.data.loaders.follownet",
    "openacc": "follownet2.data.loaders.openacc",
    "ngsim": "follownet2.data.loaders.ngsim",
    "waymo_hu": "follownet2.data.loaders.waymo_hu",
    "tgsim": "follownet2.data.loaders.tgsim",
    "ute": "follownet2.data.loaders.ute",
    "highd": "follownet2.data.loaders.highd",
    "exid": "follownet2.data.loaders.exid",
    "citysim": "follownet2.data.loaders.citysim",
}


def get_loader(name: str) -> ModuleType:
    if name not in LOADERS:
        raise KeyError(f"unknown loader {name!r}; known: {sorted(LOADERS)}")
    module = importlib.import_module(LOADERS[name])
    if not (hasattr(module, "load_pairs") or hasattr(module, "load_events")):
        raise AttributeError(f"{LOADERS[name]} defines neither load_pairs nor load_events")
    return module
