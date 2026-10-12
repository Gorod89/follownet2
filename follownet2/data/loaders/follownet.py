"""FollowNet car-following event sets (Chen et al. 2023, Sci Data 10:828).

Source: figshare collection 10.6084/m9.figshare.c.6777810 (CC0), one ``.npy`` file per subset
(``<stem>_car_fol_event_set.npy``) and per published split (``<stem>_<split>_data.npy``);
docs/data_layout.md (part A) has the verified layout.

* HighD: dense ``float64`` array ``(N, 4, 375)`` at 25 Hz (every event is a 15 s window).
* NGSIM I-80, Waymo, Lyft, SPMD DAS1 and DAS2: pickled object array ``(N, 4)`` whose element
  ``[i, c]`` is a 1-D ``float64`` array; the length differs between events; 10 Hz.
* Channels (axis 1): 0 spacing [m], 1 follower speed [m/s], 2 relative speed
  ``v_leader - v_follower`` [m/s], 3 leader speed [m/s].
* There are no ids, time stamps, lanes or vehicle lengths of any kind.

Decisions:

* ``np.load(..., allow_pickle=True)`` is never called. Object arrays are unpickled by
  :class:`SafeUnpickler`, which admits only the globals of numpy array reconstruction. A
  pickled file is read as a whole (at most 225 MB, SPMD); dense files are memory-mapped.
* Channel 2 has the opposite sign of the benchmark's ``dv`` and is never stored: it only
  cross-checks the leader speed, ``|ch3 - ch1 - ch2| <= 1e-6``. Failing events are dropped
  and counted in :func:`last_report`.
* HighD is interpolated linearly to the 10 Hz grid ``t = 0, 0.1, ...`` up to the last source
  time: 150 samples, which the extraction counts as 15 s like FollowNet does.
* Every event is its own group: time starts at 0 in each. The pseudo ids come from the
  position in the file, ``<dataset>/<subset>/e<index:06d>`` for the follower and ``l<index>``
  for the leader; events of the split files carry the split, ``train.e000012``.
* The spacing is used as published and ``meta["gap_definition"]`` says what it is: the Waymo
  subset holds the centre-to-centre distance of Hu et al. (2022), verified by matching
  1439 of its 1440 events; the NGSIM subset holds the front-to-front distance (its median
  per speed class agrees within 0.7 m with the front-to-front distance of our extraction from
  the raw table and exceeds the net gap by 3.9 to 5.2 m, the leader length); for the other
  subsets the definition is not documented.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.lib import format as npy_format

from follownet2.data.extraction import PairSeries, positions_from_gap, to_grid
from follownet2.data.schema import DT
from follownet2.utils import resolve_path

POSITION_SOURCE = "gap"
DEFAULT_RAW_DIR = "../cf-stability/data/raw/follownet"
COLLECTION = "https://doi.org/10.6084/m9.figshare.c.6777810"
SPLITS = ("train", "val", "test")
CROSS_CHECK_ATOL = 1e-6  # m/s; the published files meet |ch3 - ch1 - ch2| = 0 exactly
GAP_UNDOCUMENTED = "as published, bumper or front-to-front not documented"
GAP_CENTRE = "centre to centre as in Hu et al. (2022), not converted: vehicle lengths are not published"
GAP_FRONT = "front to front (NGSIM space headway), not converted: vehicle lengths are not published"


@dataclass(frozen=True)
class Subset:
    event_set: str | None  # combined event set; Lyft has none on figshare
    split_stem: str  # the split files are <split_stem>_<split>_data.npy
    rate_hz: float
    gap_definition: str = GAP_UNDOCUMENTED


SUBSETS: dict[str, Subset] = {
    "highd": Subset("HighD_car_fol_event_set.npy", "HighD", 25.0),
    "ngsim": Subset("NGSIM_I_80_car_fol_event_set.npy", "NGSIM_I_80", 10.0, GAP_FRONT),
    "waymo": Subset("Waymo_filter_car_fol_event_set.npy", "Waymo", 10.0, GAP_CENTRE),
    "lyft": Subset(None, "Lyft", 10.0),
    "spmd1": Subset("SPMD_das1_car_fol_event_set.npy", "SPMD1", 10.0),
    "spmd2": Subset("SPMD_das2_car_fol_event_set.npy", "SPMD2", 10.0),
}

# An ndarray pickles as (_reconstruct, ...): taking the callable from __reduce__ gives the
# function of the installed numpy without importing numpy.core, deprecated since numpy 2.
_RECONSTRUCT = np.empty(0).__reduce__()[0]
_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): _RECONSTRUCT,  # numpy < 2, all published files
    ("numpy._core.multiarray", "_reconstruct"): _RECONSTRUCT,  # numpy >= 2
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}
_HEADER_READERS = {(1, 0): npy_format.read_array_header_1_0, (2, 0): npy_format.read_array_header_2_0}

_REPORT: dict = {}


class SafeUnpickler(pickle.Unpickler):
    """Unpickler that rebuilds numpy arrays and nothing else.

    A pickle can import and call any global it names. The FollowNet object arrays need only
    the array reconstruction globals; every other global is refused before it is imported.
    """

    def find_class(self, module: str, name: str):
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a FollowNet file") from None


def read_event_array(path: Path) -> np.ndarray:
    """The array of a FollowNet file: ``(N, 4, T)`` floats or ``(N, 4)`` objects."""
    with open(path, "rb") as fh:
        version = npy_format.read_magic(fh)
        if version not in _HEADER_READERS:
            raise ValueError(f"{path}: unsupported .npy format version {version}")
        shape, _, dtype = _HEADER_READERS[version](fh)
        array = SafeUnpickler(fh).load() if dtype.hasobject else None
    if array is None:
        array = np.load(path, mmap_mode="r")  # allow_pickle keeps its default, False
    layout = len(shape) == (2 if dtype.hasobject else 3) and shape[1:2] == (4,)
    if not (layout and isinstance(array, np.ndarray) and array.dtype == dtype and array.shape == shape):
        raise ValueError(f"{path}: expected an (N, 4) object or (N, 4, T) float array, found {shape} {dtype}")
    return array


def event_channels(array: np.ndarray, index: int, source: str = "") -> tuple[np.ndarray, ...]:
    """Spacing, follower speed, relative speed and leader speed of one event."""
    channels = tuple(np.asarray(array[index, c], dtype=np.float64) for c in range(4))
    if any(c.ndim != 1 for c in channels) or len({len(c) for c in channels}) != 1:
        raise ValueError(f"{source}: event {index} does not hold four 1-D channels of equal length")
    return channels


def cross_check_residual(v: np.ndarray, rel: np.ndarray, v_lead: np.ndarray) -> float:
    """``max |v_lead - v - rel|`` over the finite samples (inf when there is none)."""
    residual = np.abs(v_lead - v - rel)
    residual = residual[np.isfinite(residual)]
    return float(residual.max()) if len(residual) else math.inf


def last_report() -> dict:
    """What the last :func:`load_pairs` run read and dropped (filled while it is consumed)."""
    return {key: list(value) if isinstance(value, list) else value for key, value in _REPORT.items()}


def load_pairs(
    raw_dir: str | Path | None = None,
    *,
    subset: str = "highd",
    files: str = "event_set",
    dataset: str | None = None,
    max_groups: int | None = None,
) -> Iterator[PairSeries]:
    """One :class:`PairSeries` per FollowNet event, on the 10 Hz grid.

    ``files="event_set"`` reads the combined event set; ``files="splits"`` reads the published
    train, val and test files and writes the split to ``meta["legacy_split"]``. ``dataset``
    defaults to ``"follownet_" + subset``; ``max_groups`` limits the number of events read.
    """
    if subset not in SUBSETS:
        raise ValueError(f"unknown FollowNet subset {subset!r}; known: {sorted(SUBSETS)}")
    if files not in ("event_set", "splits"):
        raise ValueError(f"files must be 'event_set' or 'splits', not {files!r}")
    spec = SUBSETS[subset]
    if files == "event_set" and spec.event_set is None:
        raise ValueError(f"FollowNet subset {subset!r} has no event set on figshare; use files='splits'")
    root = resolve_path(DEFAULT_RAW_DIR if raw_dir is None else raw_dir)
    if files == "event_set":
        parts: list[tuple[str | None, str]] = [(None, spec.event_set)]
    else:
        parts = [(split, f"{spec.split_stem}_{split}_data.npy") for split in SPLITS]
    missing = [name for _, name in parts if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"FollowNet subset {subset!r}: {missing} not found in {root}; expected the figshare files "
            f"<raw_dir>/{' and <raw_dir>/'.join(name for _, name in parts)} from {COLLECTION}"
        )
    dataset = dataset or f"follownet_{subset}"
    _REPORT.clear()
    _REPORT.update(
        dataset=dataset,
        subset=subset,
        files=[name for _, name in parts],
        events_read=0,
        events_yielded=0,
        dropped_cross_check=0,
        dropped_events=[],
        max_residual=0.0,
    )
    prefix = f"{dataset}/{subset}/"
    for split, name in parts:
        array = read_event_array(root / name)
        tag = f"{split}." if split else ""
        meta = {
            "source_file": name,
            "native_rate_hz": spec.rate_hz,
            "ids": "pseudo",
            "gap_definition": spec.gap_definition,
            **({"legacy_split": split} if split else {}),
        }
        for index in range(array.shape[0]):
            if max_groups is not None and _REPORT["events_read"] >= max_groups:
                return
            _REPORT["events_read"] += 1
            s, v, rel, v_lead = event_channels(array, index, name)
            residual = cross_check_residual(v, rel, v_lead)
            if not residual <= CROSS_CHECK_ATOL:
                _REPORT["dropped_cross_check"] += 1
                _REPORT["dropped_events"].append(f"{name}#{index}")
                continue
            _REPORT["max_residual"] = max(_REPORT["max_residual"], residual)
            _REPORT["events_yielded"] += 1
            t, cols = to_grid(np.arange(len(s)) / spec.rate_hz, {"s": s, "v": v, "v_lead": v_lead})
            x_lead, x_follower = positions_from_gap(cols["v_lead"], cols["s"])
            yield PairSeries(
                dataset=dataset,
                site=subset,
                follower_id=f"{prefix}{tag}e{index:06d}",
                t=t,
                x_follower=x_follower,
                v=cols["v"],
                leader_id=np.full(len(t), f"{prefix}{tag}l{index:06d}", dtype=object),
                x_lead=x_lead,
                v_lead=cols["v_lead"],
                meta={**meta, "source_index": index},
            )
