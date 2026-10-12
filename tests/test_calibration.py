"""Batched differential evolution against SciPy, and parameter recovery."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import differential_evolution as scipy_de

from follownet2.calibration.calibrate import (
    CalibrationConfig,
    calibrate_global,
    calibrate_per_event,
    parameter_spread,
    score_events,
)
from follownet2.calibration.de import differential_evolution, latin_hypercube
from follownet2.data.batch import EventBatch
from follownet2.data.loaders.synthetic import IDM_NOMINAL, SyntheticConfig, make_events
from follownet2.models.mechanistic import IDM, Gipps, Newell
from follownet2.sim.rollout import rollout, rollout_errors

CPU = dict(device="cpu", dtype="float64")


def rosenbrock(x):
    return ((1.0 - x[..., :-1]) ** 2 + 100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2).sum(-1)


def test_latin_hypercube_has_one_point_per_stratum():
    gen = torch.Generator().manual_seed(0)
    sample = latin_hypercube(3, 20, 4, gen, dtype=torch.float64)
    strata = torch.floor(sample * 20).long()
    assert bool((strata.sort(dim=1).values == torch.arange(20)[None, :, None]).all())


def test_de_matches_scipy_on_a_test_function():
    bounds = torch.tensor([[-2.0, 2.0]] * 3, dtype=torch.float64)
    ours = differential_evolution(rosenbrock, bounds, n_problems=8, maxiter=400, tol=1e-8, seed=1)
    reference = scipy_de(lambda x: float(rosenbrock(torch.as_tensor(x))), bounds.tolist(), seed=1, tol=1e-8,
                         maxiter=400, polish=False, updating="deferred")
    assert reference.fun < 1e-6
    assert float(ours.fun.max()) < 1e-6
    np.testing.assert_allclose(ours.x.numpy(), np.ones((8, 3)), atol=1e-2)


def test_de_is_reproducible_and_respects_bounds():
    bounds = torch.tensor([[0.5, 3.0], [-1.0, 0.0]], dtype=torch.float64)
    sphere = lambda x: ((x - torch.tensor([0.0, -2.0])) ** 2).sum(-1)  # optimum outside the box
    a = differential_evolution(sphere, bounds, n_problems=4, seed=5)
    b = differential_evolution(sphere, bounds, n_problems=4, seed=5)
    assert torch.equal(a.x, b.x) and bool(a.converged.all())
    np.testing.assert_allclose(a.x.numpy(), np.tile([0.5, -1.0], (4, 1)), atol=1e-2)
    assert bool(((a.population >= bounds[:, 0]) & (a.population <= bounds[:, 1])).all())


def test_idm_objective_agrees_with_scipy(clean_events):
    """The batched optimiser reaches the objective value SciPy reaches on the same event."""
    event = min(clean_events, key=len)
    cfg = CalibrationConfig(maxiter=100, tol=1e-3, **CPU)
    ours = calibrate_per_event(IDM, [event], cfg)
    batch = EventBatch.from_events([event])

    def objective(x):  # SciPy's vectorised interface: x has shape (D, S)
        theta = torch.as_tensor(np.atleast_2d(x.T), dtype=torch.float64).unsqueeze(1)
        err = rollout_errors(rollout(IDM(), batch, params=theta), batch)
        fit = torch.sqrt(err.sse_s / err.ss_s) + torch.sqrt(err.sse_v / err.ss_v)
        value = torch.where(err.collided, 10.0 + err.n_overlap / err.n, fit)
        return value.reshape(-1).numpy()

    reference = scipy_de(objective, list(IDM.bounds), seed=0, tol=1e-3, maxiter=100, polish=False,
                         updating="deferred", vectorized=True)
    found = ours[list(IDM.param_names)].to_numpy()[0]
    assert ours["objective"][0] == pytest.approx(float(objective(found.reshape(-1, 1))[0]), rel=1e-9)
    assert ours["objective"][0] < 0.02 and reference.fun < 0.02
    assert abs(ours["objective"][0] - reference.fun) < 0.01


def test_per_event_calibration_recovers_the_spacing_parameters():
    events = make_events(SyntheticConfig(n_drivers=12, events_per_driver=1, theta_cv=0.0, duration=(40.0, 45.0), seed=2))
    table = calibrate_per_event(IDM, events, CalibrationConfig(maxiter=120, tol=1e-3, batch_events=7, **CPU))
    assert list(table["event_id"]) == [ev.event_id for ev in events]
    assert not table["collided"].any() and table["rmse_s"].max() < 0.3
    # the time gap is the best identified IDM parameter
    assert np.median(np.abs(table["T"] / IDM_NOMINAL[1] - 1.0)) < 0.05
    spread = parameter_spread(table, IDM, CalibrationConfig(**CPU))
    assert spread["n_used"] == len(events) and set(spread["parameters"]) == set(IDM.param_names)
    assert spread["parameters"]["T"]["cv"] < 0.1
    assert len(spread["correlation"]["matrix"]) == 5


def test_global_calibration_recovers_the_generating_model():
    events = make_events(SyntheticConfig(n_drivers=10, events_per_driver=1, theta_cv=0.0, duration=(30.0, 40.0), seed=4))
    cfg = CalibrationConfig(maxiter=120, tol=1e-3, **CPU)
    out = calibrate_global(IDM, events, cfg)
    assert out["collision_rate"] == 0.0 and out["rmse_s"] < 0.2 and out["rmse_v"] < 0.1
    assert out["params"]["T"] == pytest.approx(IDM_NOMINAL[1], rel=0.05)
    assert out["params"]["s0"] == pytest.approx(IDM_NOMINAL[2], rel=0.25)
    scores = score_events(IDM, events, [out["params"][n] for n in IDM.param_names], cfg)
    pooled = np.sqrt((scores["rmse_s"] ** 2 * scores["n_samples"]).sum() / scores["n_samples"].sum())
    assert pooled == pytest.approx(out["rmse_s"], rel=1e-9)


@pytest.mark.parametrize("cls", [Gipps, Newell])
def test_delay_models_calibrate_without_collisions(cls, clean_events):
    cfg = CalibrationConfig(maxiter=60, tol=1e-2, **CPU)
    before = score_events(cls, clean_events[:6], cls.default, cfg)
    table = calibrate_per_event(cls, clean_events[:6], cfg)
    assert not table["collided"].any()
    assert (table["objective"] <= before["objective"] + 1e-12).all()
    # the events come from an IDM: the delay models fit them with a structural error
    assert table["rmse_s"].median() < 0.5 * before["rmse_s"].median()
    bounds = np.asarray(cls.bounds)
    theta = table[list(cls.param_names)].to_numpy()
    assert np.all(theta >= bounds[:, 0] - 1e-9) and np.all(theta <= bounds[:, 1] + 1e-9)


def test_collisions_are_penalised():
    events = make_events(SyntheticConfig(n_drivers=3, events_per_driver=1, theta_cv=0.0, seed=9))
    reckless = score_events(IDM, events, (45.0, 0.3, 0.5, 4.0, 0.5), CalibrationConfig(**CPU))
    careful = score_events(IDM, events, IDM_NOMINAL, CalibrationConfig(**CPU))
    assert careful["objective"].max() < 1e-6
    crashed = reckless[reckless["collided"]]
    assert (crashed["objective"] >= 10.0).all() and (reckless.loc[~reckless["collided"], "objective"] < 10.0).all()
