"""Dataset loaders by name.

A loader module exposes one of

* ``load_pairs(raw_dir, **options) -> Iterable[PairSeries]``: follower series with their
  instantaneous leader; events are cut by :mod:`trafficloop.data.extraction`;
* ``load_events(raw_dir, **options) -> list[Event]``: events already in schema form.

and the constant ``POSITION_SOURCE`` ("track": vehicles have their own longitudinal
coordinate; "gap": the source measures the gap and positions are integrated from speeds).
"""

from __future__ import annotations

import importlib
from types import ModuleType

LOADERS = {
    "synthetic": "trafficloop.data.loaders.synthetic",
    "follownet": "trafficloop.data.loaders.follownet",
    "openacc": "trafficloop.data.loaders.openacc",
    "ngsim": "trafficloop.data.loaders.ngsim",
    "waymo_hu": "trafficloop.data.loaders.waymo_hu",
    "tgsim": "trafficloop.data.loaders.tgsim",
    "ute": "trafficloop.data.loaders.ute",
    "highd": "trafficloop.data.loaders.highd",
    "exid": "trafficloop.data.loaders.exid",
    "citysim": "trafficloop.data.loaders.citysim",
}


def get_loader(name: str) -> ModuleType:
    if name not in LOADERS:
        raise KeyError(f"unknown loader {name!r}; known: {sorted(LOADERS)}")
    module = importlib.import_module(LOADERS[name])
    if not (hasattr(module, "load_pairs") or hasattr(module, "load_events")):
        raise AttributeError(f"{LOADERS[name]} defines neither load_pairs nor load_events")
    return module
