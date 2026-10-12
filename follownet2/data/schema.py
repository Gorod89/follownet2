"""Car-following Event schema and parquet storage.

The schema is wire-compatible with ``cf_stability.data.schema`` (same arrays, ids and
directory layout), so event directories are interchangeable between the repositories.
FollowNet-2 adds one mandatory piece of metadata: the data version tag.

Conventions (binding for the whole project, see docs/data_layout.md):

* SI units, 10 Hz (``DT = 0.1`` s), float64 arrays.
* ``dv = v - v_lead`` (approach rate, positive when closing in).
* ``x_lead`` is the longitudinal position of the leader's REAR bumper and
  ``x_follower`` the position of the follower's FRONT bumper, so the net gap is
  ``s = x_lead - x_follower`` exactly.
* ``a[k]`` is the acceleration at ``t[k]``. In the reconstructed data version it is the
  acceleration applied over ``[t[k], t[k+1]]`` by the semi-implicit Euler scheme,
  ``v[k+1] = v[k] + a[k]*dt`` and ``x[k+1] = x[k] + v[k+1]*dt``, with ``a[-1] = a[-2]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

DT = 0.1
ARRAY_FIELDS = ("t", "s", "dv", "v", "a", "v_lead", "x_lead", "x_follower")
ID_FIELDS = ("dataset", "site", "follower_id", "leader_id")
DATA_VERSIONS = ("raw", "reconstructed")

# redundant columns, recomputed bit for bit on reading (s = x_lead - x_follower, dv = v - v_lead)
DERIVED_FIELDS = ("s", "dv")
SAMPLES_FILE = "samples.parquet"
EVENTS_FILE = "events.parquet"
MANIFEST_FILE = "manifest.json"


class EventValidationError(ValueError):
    pass


@dataclass
class Event:
    """One car-following event on the common 10 Hz grid."""

    event_id: str
    dataset: str
    site: str
    follower_id: str
    leader_id: str
    t: np.ndarray
    s: np.ndarray
    dv: np.ndarray
    v: np.ndarray
    a: np.ndarray
    v_lead: np.ndarray
    x_lead: np.ndarray
    x_follower: np.ndarray
    # free-form, JSON-serialisable extras; ``data_version`` is mandatory
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ARRAY_FIELDS:
            setattr(self, name, np.ascontiguousarray(getattr(self, name), dtype=np.float64))

    def __len__(self) -> int:
        return int(self.t.shape[0])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def data_version(self) -> str:
        return self.meta.get("data_version", "")

    @property
    def regime(self) -> str | None:
        return self.meta.get("regime")

    def validate(self, dt: float = DT, atol: float = 1e-6) -> None:
        """Raise EventValidationError if an invariant of the schema is violated."""
        n = len(self)
        if n < 2:
            raise EventValidationError(f"{self.event_id}: fewer than 2 samples")
        if self.data_version not in DATA_VERSIONS:
            raise EventValidationError(
                f"{self.event_id}: meta['data_version'] is {self.data_version!r}, expected one of {DATA_VERSIONS}"
            )
        for name in ARRAY_FIELDS:
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise EventValidationError(f"{self.event_id}: {name} has shape {arr.shape}, expected ({n},)")
            if not np.all(np.isfinite(arr)):
                raise EventValidationError(f"{self.event_id}: {name} contains non-finite values")
        if np.max(np.abs(np.diff(self.t) - dt)) > atol:
            raise EventValidationError(f"{self.event_id}: t is not uniform with step {dt}")
        if np.min(self.s) <= 0.0:
            raise EventValidationError(f"{self.event_id}: non-positive spacing (min {np.min(self.s):.3f})")
        if np.min(self.v) < -atol or np.min(self.v_lead) < -atol:
            raise EventValidationError(f"{self.event_id}: negative speed")
        if np.max(np.abs(self.dv - (self.v - self.v_lead))) > atol:
            raise EventValidationError(f"{self.event_id}: dv != v - v_lead")
        if np.max(np.abs(self.s - (self.x_lead - self.x_follower))) > atol:
            raise EventValidationError(f"{self.event_id}: s != x_lead - x_follower")


class EventSet:
    """Ordered collection of events with parquet round-trip.

    On disk an EventSet is a directory with ``samples.parquet`` (long format, one
    row per time sample, keyed by ``event_id``), ``events.parquet`` (one row per
    event: ids, n_samples, duration, data_version, meta_json) and ``manifest.json``.
    """

    def __init__(self, events: Iterable[Event] = ()) -> None:
        self.events: list[Event] = list(events)
        self._index: dict[str, int] = {}
        for i, ev in enumerate(self.events):
            if ev.event_id in self._index:
                raise ValueError(f"duplicate event_id {ev.event_id}")
            self._index[ev.event_id] = i

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)

    def __getitem__(self, key: int | str) -> Event:
        if isinstance(key, str):
            return self.events[self._index[key]]
        return self.events[key]

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._index

    def subset(self, event_ids: Iterable[str]) -> "EventSet":
        return EventSet(self.events[self._index[e]] for e in event_ids)

    def validate(self, dt: float = DT) -> None:
        for ev in self.events:
            ev.validate(dt=dt)

    @property
    def data_version(self) -> str:
        """The common version tag; raises if the set mixes versions."""
        versions = {ev.data_version for ev in self.events}
        if len(versions) > 1:
            raise ValueError(f"EventSet mixes data versions: {sorted(versions)}")
        return versions.pop() if versions else ""

    # ------------------------------------------------------------------ tables
    def events_frame(self) -> pd.DataFrame:
        rows = [
            {
                "event_id": ev.event_id,
                "dataset": ev.dataset,
                "site": ev.site,
                "follower_id": ev.follower_id,
                "leader_id": ev.leader_id,
                "n_samples": len(ev),
                "duration": ev.duration,
                "data_version": ev.data_version,
                "meta_json": json.dumps(ev.meta, sort_keys=True, default=str),
            }
            for ev in self.events
        ]
        columns = ["event_id", *ID_FIELDS, "n_samples", "duration", "data_version", "meta_json"]
        return pd.DataFrame(rows, columns=columns)

    def samples_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(columns=["event_id", *ARRAY_FIELDS])
        ids = np.concatenate([np.full(len(ev), i, dtype=np.int64) for i, ev in enumerate(self.events)])
        data = {"event_id": pd.Categorical.from_codes(ids, categories=[ev.event_id for ev in self.events])}
        for name in ARRAY_FIELDS:
            data[name] = np.concatenate([getattr(ev, name) for ev in self.events])
        return pd.DataFrame(data)

    def summary(self) -> dict:
        n = np.array([len(ev) for ev in self.events], dtype=np.int64)
        dur = np.array([ev.duration for ev in self.events], dtype=np.float64)
        return {
            "n_events": int(len(self.events)),
            "n_samples": int(n.sum()) if len(n) else 0,
            "n_followers": len({ev.follower_id for ev in self.events}),
            "n_sites": len({(ev.dataset, ev.site) for ev in self.events}),
            "duration_total_s": float(dur.sum()) if len(dur) else 0.0,
            "duration_median_s": float(np.median(dur)) if len(dur) else 0.0,
            "duration_min_s": float(dur.min()) if len(dur) else 0.0,
            "duration_max_s": float(dur.max()) if len(dur) else 0.0,
        }

    # ----------------------------------------------------------------- storage
    def to_parquet(self, directory: str | Path, manifest: dict | None = None, compact: bool = False) -> Path:
        """Write the event directory.

        ``compact`` leaves out the two derived columns, which saves space and loses
        nothing: they are recomputed exactly on reading. Compact directories are read by
        this package only; without ``compact`` the layout is the one shared with
        cf-stability.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        samples = self.samples_frame()
        if compact:
            samples = samples.drop(columns=list(DERIVED_FIELDS))
        floats = [c for c in samples.columns if c not in ("event_id", "t")]
        samples.to_parquet(
            directory / SAMPLES_FILE,
            index=False,
            compression="zstd",
            compression_level=9,
            use_dictionary=["event_id", "t"],
            use_byte_stream_split=floats,
        )
        self.events_frame().to_parquet(directory / EVENTS_FILE, index=False, compression="zstd")
        payload = {
            "schema_version": 1,
            "layout": "compact" if compact else "full",
            "dt": DT,
            "data_version": self.data_version,
            "summary": self.summary(),
            **(manifest or {}),
        }
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        (directory / MANIFEST_FILE).write_text(text, encoding="utf-8")
        return directory

    @classmethod
    def from_parquet(cls, directory: str | Path, default_version: str | None = None) -> "EventSet":
        """Load an event directory.

        ``default_version`` tags events written by a repository that does not store the
        version tag (cf-stability); without it such events fail ``validate()``.
        """
        directory = Path(directory)
        meta = pd.read_parquet(directory / EVENTS_FILE)
        samples = pd.read_parquet(directory / SAMPLES_FILE)
        samples["event_id"] = samples["event_id"].astype(str)
        # rows of one event are contiguous and time-ordered by construction
        arrays = {
            name: samples[name].to_numpy(dtype=np.float64) for name in ARRAY_FIELDS if name in samples.columns
        }
        if "s" not in arrays:
            arrays["s"] = arrays["x_lead"] - arrays["x_follower"]
        if "dv" not in arrays:
            arrays["dv"] = arrays["v"] - arrays["v_lead"]
        ids = samples["event_id"].to_numpy()
        empty = np.array([], dtype=np.int64)
        bounds = np.flatnonzero(ids[1:] != ids[:-1]) + 1 if len(ids) else empty
        starts = np.concatenate([[0], bounds]).astype(np.int64) if len(ids) else empty
        stops = np.concatenate([bounds, [len(ids)]]).astype(np.int64) if len(ids) else empty
        span = {ids[a]: (a, b) for a, b in zip(starts, stops)}
        if len(span) != len(starts):
            raise ValueError(f"{directory}: rows of an event are not contiguous in {SAMPLES_FILE}")
        events = []
        for row in meta.itertuples(index=False):
            a, b = span[row.event_id]
            ev_meta = json.loads(row.meta_json)
            if "data_version" not in ev_meta:
                version = getattr(row, "data_version", None) or default_version
                if version:
                    ev_meta["data_version"] = version
            events.append(
                Event(
                    event_id=row.event_id,
                    dataset=row.dataset,
                    site=row.site,
                    follower_id=row.follower_id,
                    leader_id=row.leader_id,
                    meta=ev_meta,
                    **{name: arrays[name][a:b].copy() for name in ARRAY_FIELDS},
                )
            )
        return cls(events)


def events_dir(root: str | Path, dataset: str, version: str) -> Path:
    """``<root>/<dataset>/<version>``, the directory of one dataset in one data version."""
    if version not in DATA_VERSIONS:
        raise ValueError(f"unknown data version {version!r}")
    return Path(root) / dataset / version
