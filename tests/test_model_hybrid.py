"""Physics-hybrid models: PIDL, PERL, stability-regularised MLP (model contract, sections 2 to 4).

Training runs in float32, the dtype of the training script, on one thread: the networks are
small, and on a shared CPU more threads only wait for each other.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from trafficloop.data.batch import WARMUP
from trafficloop.data.loaders.synthetic import IDM_NOMINAL, SyntheticConfig, make_events
from trafficloop.models import registry
from trafficloop.models.hybrid import PERL, PIDL, StabilityRegularisedMLP
from trafficloop.models.mechanistic import IDM, idm_acc
from trafficloop.models.network import NetworkModel
from trafficloop.sim.differentiable import rollout_windows
from trafficloop.stability import analytic
from trafficloop.train.trainer import TrainConfig
from trafficloop.train.windows import WindowSet
from trafficloop.utils import REPO_ROOT

NAMES = ("pidl", "perl", "stability_mlp")
THETA_PIDL = (33.3, 1.2, 2.0, 1.0, 1.5)  # the IDM default with T = 1.2 s instead of 1.5 s
THETA_PERL = (30.0, 1.2, 3.0, 1.5, 2.0)
# the IDM default, which is also the synthetic nominal driver, is string unstable from 0.91 to 18.61 m/s
THETA_UNSTABLE = IDM_NOMINAL
BOX = torch.tensor([[5.0, -3.0, 2.0], [80.0, 3.0, 30.0]])  # (s, dv, v)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def windows(theta, seed: int, n_drivers: int = 30, theta_cv: float = 0.0, horizon: int = 1, dtype=torch.float32) -> WindowSet:
    cfg = SyntheticConfig(n_drivers=n_drivers, events_per_driver=2, theta=theta, theta_cv=theta_cv, duration=(20.0, 35.0), seed=seed)
    return WindowSet.from_events(make_events(cfg), horizon=horizon, device="cpu", dtype=dtype)


def splits(theta, theta_cv: float = 0.0) -> tuple[WindowSet, WindowSet, WindowSet]:
    """Training, validation and held-out windows of disjoint drivers; the held-out ones allow 100-step rollouts."""
    return (
        windows(theta, 1, theta_cv=theta_cv),
        windows(theta, 2, n_drivers=10, theta_cv=theta_cv),
        windows(theta, 3, n_drivers=10, theta_cv=theta_cv, horizon=100),
    )


def config(epochs: int, lr: float = 3e-3) -> TrainConfig:
    return TrainConfig(epochs=epochs, batch_size=512, lr=lr, batches_per_epoch=None, patience=epochs, device="cpu", seed=0)


def rmse(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.double() ** 2)))


def one_step_rmse(model, w: WindowSet) -> float:
    index = torch.arange(len(w))
    with torch.no_grad():
        return rmse(model(w.history(index)) - w.target(index)[:, 0])


def constant_speed_rmse(w: WindowSet) -> float:
    return rmse(w.target(torch.arange(len(w)))[:, 0])


def box_states(n: int, box: torch.Tensor, seed: int | None) -> torch.Tensor:
    """Histories ``[n, 1, 4]`` of states drawn uniformly in ``box [2, 3]`` of (s, dv, v).

    ``seed=None`` draws from the global generator, like the collocation of an unfitted PIDL.
    """
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    u = torch.rand(n, 3, generator=generator, dtype=box.dtype)
    x = box[0] + (box[1] - box[0]) * u
    return torch.cat([x, torch.zeros_like(x[:, :1])], dim=-1).unsqueeze(1)


def idm(hist: torch.Tensor, theta) -> torch.Tensor:
    now = hist[..., -1, :]
    return idm_acc(now[..., 0], now[..., 1], now[..., 2], *torch.as_tensor(theta, dtype=hist.dtype).unbind(-1))


# ------------------------------------------------------------------------ trained models
@pytest.fixture(scope="module")
def pidl_fit():
    train, val, test = splits(THETA_PIDL)
    torch.manual_seed(0)
    model = registry.create("pidl")
    return model, model.fit(train, val, config(30)), test


@pytest.fixture(scope="module")
def perl_fit():
    """Heterogeneous drivers: the pooled IDM leaves a residual to learn."""
    train, val, test = splits(THETA_PERL, theta_cv=0.15)
    torch.manual_seed(0)
    model = registry.create("perl")
    return model, model.fit(train, val, config(8)), test


@pytest.fixture(scope="module")
def stability_fits():
    train, val, test = splits(THETA_UNSTABLE)
    fits = {}
    for weight in (0.0, 1.0):
        torch.manual_seed(0)
        model = registry.create("stability_mlp", weight=weight)
        fits[weight] = (model, model.fit(train, val, config(30)))
    return fits, test


@pytest.fixture(scope="module")
def trained(pidl_fit, perl_fit, stability_fits):
    fits, test = stability_fits
    return {"pidl": pidl_fit, "perl": perl_fit, "stability_mlp": (*fits[1.0], test)}


# ---------------------------------------------------------------------- common contract
@pytest.mark.parametrize("name", NAMES)
def test_interface(name):
    torch.manual_seed(1)
    model = registry.create(name)
    model.set_scale(windows(IDM_NOMINAL, 5, n_drivers=4).statistics())
    if isinstance(model, PERL):
        nn.init.normal_(model.head.weight, std=0.5)  # the zero head would hide the network
    model.eval()
    hist = windows(IDM_NOMINAL, 6, n_drivers=4, dtype=torch.float64).history(torch.arange(24))
    out = model(hist)
    assert out.shape == (24,) and out.dtype == torch.float64
    assert torch.equal(model(hist.reshape(4, 6, WARMUP, 4)).reshape(-1), out)
    single = model(hist.float())
    assert single.dtype == torch.float32
    torch.testing.assert_close(single.double(), out, rtol=1e-4, atol=1e-4)
    changed = hist.clone()
    changed[:, : -model.history] += 50.0  # rows before the history of the model
    changed[..., 3] += 5.0  # a_prev is not an input
    assert torch.equal(model(changed), out)
    assert torch.equal(model(hist), out)  # deterministic in evaluation mode
    if isinstance(model, PERL):
        reading = registry.create("perl", use_a_prev=True)
        nn.init.normal_(reading.head.weight, std=0.5)
        assert not torch.equal(reading(changed), reading(hist))
    else:
        assert model.memoryless and model.history == 1


@pytest.mark.parametrize("name", NAMES)
def test_configuration_lists_every_constructor_argument(name):
    conf = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "conf" / "model" / f"{name}.yaml"))
    arguments = inspect.signature(registry.model_class(name)).parameters
    assert conf["name"] == name and set(conf["kwargs"]) == set(arguments) and conf["optim"] == {}
    assert registry.create(name, **conf["kwargs"]).kwargs == registry.create(name).kwargs == conf["kwargs"]


@pytest.mark.parametrize("name", NAMES)
def test_checkpoint_round_trip(name, trained, tmp_path):
    model, _, test = trained[name]
    hist = test.history(torch.arange(300))
    loaded, _ = registry.load_model(registry.save_model(model, tmp_path / "model.pt"))
    assert loaded.name == name and loaded.kwargs == model.kwargs
    state = model.state_dict()
    assert all(torch.equal(value, state[key]) for key, value in loaded.state_dict().items())
    assert torch.equal(loaded(hist), model.eval()(hist))


@pytest.mark.parametrize("name, fraction", [("pidl", 0.1), ("perl", 0.35), ("stability_mlp", 0.1)])
def test_learning_and_rollout(name, fraction, trained):
    model, log, test = trained[name]
    assert {"best_val_rmse_a", "epochs_run", "seconds", "n_parameters", "history"} <= set(log)
    assert one_step_rmse(model, test) < fraction * constant_speed_rmse(test)
    with torch.no_grad():
        sim = rollout_windows(model, test, torch.arange(0, len(test), 29), 100)
    assert all(bool(torch.isfinite(sim[key]).all()) for key in ("s", "v", "a"))


# ------------------------------------------------------------------------------- PIDL
def test_pidl_moves_the_idm_parameters_towards_the_generating_ones(pidl_fit):
    model, log, _ = pidl_fit
    learned = log["physics"]["params"]
    assert learned == model.physics_parameters()
    start, truth = IDM.default[1], THETA_PIDL[1]
    assert abs(learned["T"] - truth) < abs(start - truth) and abs(learned["T"] - truth) < 0.15 * truth
    assert all(lo <= learned[name] <= hi for name, (lo, hi) in zip(IDM.param_names, IDM.bounds))


def test_pidl_with_alpha_zero_copies_the_idm_without_reading_a_recorded_acceleration():
    train, val, test = splits(IDM_NOMINAL)
    noise = torch.randn(train.batch.a.shape, generator=torch.Generator().manual_seed(0))
    train.batch.a = noise  # the recorded accelerations of the training windows are destroyed
    torch.manual_seed(0)
    model = registry.create("pidl", alpha=0.0)
    model.fit(train, val, config(30, lr=1e-2))
    assert model.physics_parameters() == pytest.approx(dict(zip(IDM.param_names, IDM.default)), rel=1e-5)
    hist = box_states(20_000, model.collocation_box, seed=7)
    with torch.no_grad():
        assert rmse(model(hist) - idm(hist, IDM.default).clamp(-8.0, 4.0)) < 0.1  # random states of the box
        observed = test.history(torch.arange(len(test)))
        assert rmse(model(observed) - idm(observed, IDM.default)) < 0.1  # random states of held-out events


def test_pidl_loss_is_the_weighted_sum_of_the_data_and_physics_terms():
    train = windows(IDM_NOMINAL, 8, n_drivers=4)
    index = torch.arange(256)
    for alpha in (0.0, 0.7, 1.0):
        torch.manual_seed(2)
        model = PIDL(alpha=alpha)
        model.set_scale(train.statistics())
        model.collocation_box.copy_(BOX)
        torch.manual_seed(3)
        loss = model.training_loss(train, index)
        torch.manual_seed(3)
        data, physics = NetworkModel.training_loss(model, train, index), model.physics_loss()
        torch.testing.assert_close(loss, alpha * data + (1.0 - alpha) * physics)
        loss.backward()
        # the IDM parameters learn from the physics term only, and not at all without data
        assert (model.idm_raw.grad is None) == (alpha in (0.0, 1.0))


def test_pidl_physics_term_compares_the_network_with_the_clipped_idm():
    torch.manual_seed(2)
    model = PIDL()
    model.set_scale(windows(IDM_NOMINAL, 8, n_drivers=4).statistics())
    model.collocation_box.copy_(BOX)
    torch.manual_seed(5)
    physics = model.physics_loss()
    torch.manual_seed(5)
    hist = box_states(model.n_collocation, BOX, seed=None)
    exact = idm(hist, model.idm_theta().detach())
    assert float(exact.min()) < -8.0  # in the corners of the box the IDM brakes harder than the protocols allow
    with torch.no_grad():
        expected = torch.mean(((model(hist) - exact.clamp(-8.0, 4.0)) / model.a_std) ** 2)
    torch.testing.assert_close(physics.detach(), expected)


# ------------------------------------------------------------------------------- PERL
@pytest.fixture(scope="module")
def idm_followers():
    """Noise-free followers of one IDM with parameters other than the initial ones."""
    return splits(THETA_PERL)


def test_perl_equals_the_idm_until_the_residual_is_trained(idm_followers):
    train, val, test = idm_followers
    model = registry.create("perl")
    physics = model.fit_physics(train, val, config(1))
    assert physics["params"] == dict(zip(IDM.param_names, model.theta.tolist()))
    hist = test.history(torch.arange(len(test)))
    with torch.no_grad():
        assert float((model(hist) - idm(hist, model.theta)).abs().max()) < 1e-6
    # noise-free followers of one IDM: the least squares finds it
    assert physics["params"] == pytest.approx(dict(zip(IDM.param_names, THETA_PERL)), rel=1e-3)
    assert physics["train_rmse_a"] < 1e-3 and physics["val_rmse_a"] < 1e-3
    assert not any(name == "theta" for name, _ in model.named_parameters())


def test_perl_residual_is_not_worse_than_the_physics_alone(perl_fit):
    model, log, test = perl_fit
    assert set(log["physics"]) == {"params", "train_rmse_a", "val_rmse_a", "steps", "seconds"}
    assert log["physics"]["params"] == dict(zip(IDM.param_names, model.theta.tolist()))
    index = torch.arange(len(test))
    hist, target = test.history(index), test.target(index)[:, 0]
    with torch.no_grad():
        assert rmse(model(hist) - target) <= rmse(model.physics(hist) - target)


def test_perl_residual_stays_small_on_idm_followers(idm_followers):
    train, val, test = idm_followers
    torch.manual_seed(0)
    model = registry.create("perl")
    model.fit(train, val, config(4))
    hist = test.history(torch.arange(len(test)))
    with torch.no_grad():
        assert rmse(model(hist) - model.physics(hist)) < 0.1


# --------------------------------------------------------------------- stability MLP
def test_the_penalty_lowers_the_unstable_fraction_at_a_small_cost(stability_fits):
    fits, test = stability_fits
    (plain, plain_log), (stable, stable_log) = fits[0.0], fits[1.0]
    assert analytic.unstable_fraction(IDM(THETA_UNSTABLE)) > 0.5  # the generating law
    assert plain_log["unstable_fraction"] == plain.unstable_fraction() > 0.25
    assert stable_log["unstable_fraction"] < plain_log["unstable_fraction"]
    assert stable_log["stability_penalty"] < plain_log["stability_penalty"]
    assert one_step_rmse(stable, test) <= 1.5 * one_step_rmse(plain, test)


def test_the_penalty_enters_the_loss_once_per_training_step(monkeypatch):
    calls = []
    penalty = analytic.stability_penalty

    def counting(model, speeds, margin, *args):
        calls.append((len(speeds), margin, model.training))
        return penalty(model, speeds, margin)

    monkeypatch.setattr(analytic, "stability_penalty", counting)
    train = windows(IDM_NOMINAL, 9, n_drivers=4)
    torch.manual_seed(0)
    model = StabilityRegularisedMLP(margin=0.01, penalty_speeds=11)
    log = model.fit(train, train, TrainConfig(epochs=2, batch_size=256, batches_per_epoch=3, device="cpu", seed=0))
    assert calls[:-1] == [(11, 0.01, True)] * 6 and calls[-1] == (11, 0.01, False)  # 6 steps, then the log
    assert model.speeds == pytest.approx(tuple(torch.linspace(5.0, 30.0, 11).tolist()))
    assert {"unstable_fraction", "stability_penalty"} <= set(log)
    calls.clear()
    StabilityRegularisedMLP(weight=0.0).training_loss(train, torch.arange(128))
    assert calls == []


def test_the_loss_adds_the_weighted_penalty_of_the_current_network(stability_fits):
    fits, _ = stability_fits
    model = StabilityRegularisedMLP(weight=2.0)
    model.load_state_dict(fits[0.0][0].state_dict())  # the plain network, unstable at low speeds
    train, index = windows(IDM_NOMINAL, 9, n_drivers=4), torch.arange(512)
    penalty = model.stability_penalty()
    data = NetworkModel.training_loss(model, train, index)
    assert float(penalty.detach()) > 0.0
    loss = model.training_loss(train, index)
    torch.testing.assert_close(loss, data + 2.0 * penalty)
    weight = model.net[0].weight
    (with_penalty,), (without,) = torch.autograd.grad(loss, weight), torch.autograd.grad(data, weight)
    assert bool(torch.isfinite(with_penalty).all()) and not torch.allclose(with_penalty, without)
