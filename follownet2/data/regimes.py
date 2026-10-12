"""Traffic regime of an event from a density proxy.

The density seen by one follower over an event is, by Edie's generalised definition,
the time spent divided by the area swept in the (x, t) plane,

    k = 1000 / mean(s + L_lead)      [veh/km per lane]

with ``s`` the net gap and ``L_lead`` the leader length. Default thresholds are the
level-of-service boundaries of the Highway Capacity Manual for basic freeway segments:
C/D at 26 pc/mi/ln (16.2 veh/km) and E/F at 45 pc/mi/ln (28.0 veh/km):

* ``free``       k < 16.2         (LOS A-C)
* ``steady``     16.2 <= k < 28.0 (LOS D-E, dense but not broken down)
* ``congested``  k >= 28.0        (LOS F)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from follownet2.data.schema import Event

REGIMES = ("free", "steady", "congested")


@dataclass
class RegimeConfig:
    free_below: float = 16.2  # veh/km/lane
    congested_from: float = 28.0
    default_length: float = 4.5  # m, used when the source does not give the leader length


def density_proxy(event: Event, cfg: RegimeConfig | None = None) -> float:
    cfg = cfg or RegimeConfig()
    length = event.meta.get("leader_length")
    if length is None or not np.isfinite(length) or length <= 0:
        length = cfg.default_length
    return 1000.0 / float(np.mean(event.s) + length)


def regime_of(density: float, cfg: RegimeConfig | None = None) -> str:
    cfg = cfg or RegimeConfig()
    if density < cfg.free_below:
        return "free"
    return "steady" if density < cfg.congested_from else "congested"


def assign_regimes(events: Iterable[Event], cfg: RegimeConfig | None = None) -> dict[str, int]:
    """Write ``density_proxy``, ``mean_speed`` and ``regime`` into the meta of every event."""
    cfg = cfg or RegimeConfig()
    counts = {name: 0 for name in REGIMES}
    for ev in events:
        k = density_proxy(ev, cfg)
        ev.meta["density_proxy"] = round(k, 4)
        ev.meta["mean_speed"] = round(float(np.mean(ev.v)), 4)
        ev.meta["regime"] = regime_of(k, cfg)
        counts[ev.meta["regime"]] += 1
    return counts
