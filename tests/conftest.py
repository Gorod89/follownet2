from __future__ import annotations

import numpy as np
import pytest
import torch

from trafficloop.data.loaders.synthetic import SyntheticConfig, make_events
from trafficloop.data.regimes import assign_regimes
from trafficloop.data.schema import EventSet


@pytest.fixture(scope="session", autouse=True)
def single_thread():
    """Small networks on a machine shared with other jobs: one thread is many times faster."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="session")
def clean_events():
    """Noise-free IDM events with identical driver parameters (exact under the integrator)."""
    events = make_events(SyntheticConfig(n_drivers=8, events_per_driver=3, theta_cv=0.0, seed=3))
    assign_regimes(events)
    return events


@pytest.fixture(scope="session")
def driver_events():
    """Heterogeneous drivers, several events each, three sites."""
    events = make_events(SyntheticConfig(n_drivers=30, events_per_driver=4, n_sites=3, theta_cv=0.2, seed=5))
    assign_regimes(events)
    return EventSet(events)


@pytest.fixture(scope="session")
def noisy_events():
    cfg = SyntheticConfig(n_drivers=6, events_per_driver=1, theta_cv=0.0, sigma_x=0.3, sigma_v=0.2, seed=7)
    return make_events(cfg, return_truth=True)


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))
