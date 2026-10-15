"""Bridges to the sibling repositories cf-stability, cf-identifiability and cf-uncertainty.

TrafficLoop keeps its own implementation of the shared interfaces (see docs/decisions.md,
S1) and stays interchangeable with the siblings at two levels:

* data: event directories have the layout of ``cf_stability.data.schema.EventSet``; the
  converters below translate in-memory objects in both directions;
* models: any callable ``acc = model(state_history)`` with the three-channel history
  ``(s, dv, v)`` of the siblings is wrapped by :func:`wrap_model`.

The sibling packages are imported lazily from their checkouts next to this repository.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from trafficloop.data.schema import ARRAY_FIELDS, Event, EventSet
from trafficloop.models.base import ThreeChannelAdapter
from trafficloop.utils import REPO_ROOT

SIBLINGS = {
    "cf_stability": "cf-stability",
    "cf_ident": "cf-identifiability",
    "cf_uncertainty": "cf-uncertainty",
}


def sibling(package: str):
    """Import a sibling package, from its checkout next to this repository if necessary."""
    try:
        return importlib.import_module(package)
    except ImportError:
        checkout = REPO_ROOT.parent / SIBLINGS[package]
        if not (checkout / package).is_dir():
            return None
        sys.path.insert(0, str(checkout))
        try:
            return importlib.import_module(package)
        except ImportError:
            return None
        finally:
            sys.path.remove(str(checkout))


def sibling_status() -> dict[str, str | None]:
    """Location of every sibling package that can be imported (for manifests and reports)."""
    out = {}
    for package in SIBLINGS:
        module = sibling(package)
        out[package] = None if module is None else str(Path(module.__file__).parent)
    return out


def to_cf_stability(events: EventSet):
    schema = importlib.import_module("cf_stability.data.schema") if sibling("cf_stability") else None
    if schema is None:
        raise ImportError("cf_stability is not available")
    fields = ("event_id", "dataset", "site", "follower_id", "leader_id", *ARRAY_FIELDS)
    return schema.EventSet(schema.Event(**{f: getattr(ev, f) for f in fields}, meta=dict(ev.meta)) for ev in events)


def from_cf_stability(events, data_version: str = "raw") -> EventSet:
    """Convert a cf-stability EventSet; events without a version tag get ``data_version``."""
    fields = ("event_id", "dataset", "site", "follower_id", "leader_id", *ARRAY_FIELDS)
    out = []
    for ev in events:
        meta = {"data_version": data_version, **ev.meta}
        out.append(Event(**{f: getattr(ev, f) for f in fields}, meta=meta))
    return EventSet(out)


def to_cf_ident(events: EventSet):
    """Column store of cf-identifiability (integer ids, CSR pointers); returns it with the id maps."""
    if sibling("cf_ident") is None:
        raise ImportError("cf_ident is not available")
    schema = importlib.import_module("cf_ident.schema")
    drivers = {name: i for i, name in enumerate(sorted({ev.follower_id for ev in events}))}
    lengths = np.array([len(ev) for ev in events], dtype=np.int64)
    columns = {name: np.concatenate([getattr(ev, name) for ev in events]) for name in ("t", "s", "dv", "v", "a", "v_lead", "x_lead")}
    columns["x"] = np.concatenate([ev.x_follower for ev in events])
    meta = pd.DataFrame(
        {
            "event_id": np.arange(len(events)),
            "source_event_id": [ev.event_id for ev in events],
            "follower_id": [ev.follower_id for ev in events],
            "data_version": [ev.data_version for ev in events],
        }
    )
    store = schema.EventSet(
        event_id=np.arange(len(events), dtype=np.int64),
        driver_id=np.array([drivers[ev.follower_id] for ev in events], dtype=np.int64),
        ptr=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        meta=meta,
        **columns,
    )
    return store, drivers


def wrap_model(fn: Callable, history: int = 1, name: str = "external") -> ThreeChannelAdapter:
    """Benchmark model from a sibling model that maps ``[..., H, 3]`` histories to accelerations."""
    return ThreeChannelAdapter(fn, history=history, name=name)
