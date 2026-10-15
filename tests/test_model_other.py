"""knn, maml_lstm, diffusion and the empirical distribution (docs/model_contract.md, sections 2 and 4).

Every learned model is trained once per module on synthetic IDM followers with a small budget.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from scipy import stats

from trafficloop.data.batch import A_PREV, WARMUP, EventBatch
from trafficloop.data.loaders.synthetic import SyntheticConfig, make_events
from trafficloop.models import registry
from trafficloop.models.diffusion import DiffusionCF, normal_quantiles
from trafficloop.models.dist import Empirical
from trafficloop.models.knn import KNN
from trafficloop.models.meta import MAMLLSTM
from trafficloop.models.persistence import ZeroAcceleration
from trafficloop.protocols.p0 import evaluate_p0, pooled_rmse
from trafficloop.sim.differentiable import rollout_windows
from trafficloop.sim.rollout import rollout
from trafficloop.train.trainer import TrainConfig
from trafficloop.train.windows import WindowSet
from trafficloop.utils import REPO_ROOT

NOISE = 0.4  # m/s^2, standard deviation of the acceleration noise in the data of the diffusion model


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """The networks are small: on a machine shared with other jobs one thread is the fastest."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def synthetic(n_drivers: int, events_per_driver: int, seed: int, theta_cv: float = 0.0) -> list:
    cfg = SyntheticConfig(n_drivers=n_drivers, events_per_driver=events_per_driver, theta_cv=theta_cv,
                          duration=(20.0, 35.0), seed=seed)
    return make_events(cfg)


def with_noise(events: list, seed: int) -> list:
    """The events with Gaussian noise of standard deviation NOISE added to the recorded acceleration."""
    rng = np.random.default_rng(seed)
    return [dataclasses.replace(ev, a=ev.a + NOISE * rng.standard_normal(len(ev))) for ev in events]


def p0_rmse(model, events) -> float:
    return pooled_rmse(evaluate_p0(model, events))["rmse_a"]


def check_interface(model, hist: torch.Tensor) -> torch.Tensor:
    """Shapes, dtypes, leading dimensions, determinism; rows before the history and a_prev are ignored."""
    out = model(hist)
    assert out.shape == hist.shape[:-2] and out.dtype == hist.dtype and bool(torch.isfinite(out).all())
    assert torch.equal(model(hist), out)
    assert torch.equal(model(hist.double()), out.double())
    stacked = model(hist.reshape(3, -1, *hist.shape[-2:]))
    assert stacked.shape == (3, hist.shape[0] // 3) and torch.equal(stacked.reshape(-1), out)
    assert model(hist[0]).shape == ()
    changed = hist.clone()
    changed[:, : -model.history] += 100.0
    changed[..., A_PREV] += 5.0
    assert torch.equal(model(changed), out)
    return out


def a_prev_changes_the_output(model, hist: torch.Tensor) -> bool:
    changed = hist.clone()
    changed[..., A_PREV] += 1.0
    return not torch.allclose(model(changed), model(hist))


def round_trip(model, path, hist: torch.Tensor):
    loaded, provenance = registry.load_model(registry.save_model(model, path, {"seed": 0}))
    assert type(loaded) is type(model) and loaded.kwargs == model.kwargs and provenance == {"seed": 0}
    assert not loaded.training and torch.equal(loaded(hist), model.eval()(hist))
    return loaded


def finite_rollout(model, events: list) -> dict:
    windows = WindowSet.from_events(events, horizon=100)
    with torch.no_grad():
        sim = rollout_windows(model, windows, torch.arange(0, len(windows), 40), 100)
    assert all(bool(torch.isfinite(sim[key]).all()) for key in ("s", "v", "a"))
    return sim


# --------------------------------------------------------------------- configuration
@pytest.mark.parametrize("name", ["knn", "maml_lstm", "diffusion"])
def test_configuration_lists_every_constructor_argument(name):
    cfg = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "conf" / "model" / f"{name}.yaml"))
    parameters = inspect.signature(registry.model_class(name)).parameters.values()
    defaults = {p.name: list(p.default) if isinstance(p.default, tuple) else p.default for p in parameters}
    assert cfg["name"] == name and cfg["kwargs"] == defaults and cfg["optim"] == {}
    model = registry.create(name, **cfg["kwargs"])
    assert model.name == name and model.kwargs == defaults and model.history <= WARMUP


# ------------------------------------------------------------------ empirical distribution
def test_empirical_distribution_of_normal_samples():
    mu, sigma = 1.5, 2.0
    gen = torch.Generator().manual_seed(0)
    dist = Empirical(mu + sigma * torch.randn(3, 4000, generator=gen, dtype=torch.float64))
    assert dist.batch_shape == (3,) and dist.n_samples == 4000
    np.testing.assert_allclose(dist.mean, mu, atol=0.12)
    np.testing.assert_allclose(dist.stddev, sigma, rtol=0.04)
    x = torch.tensor([-1.0, 1.5, 4.0], dtype=torch.float64)  # one value per batch element
    z = ((x - mu) / sigma).numpy()
    np.testing.assert_allclose(dist.cdf(x), stats.norm.cdf(z), atol=0.03)
    q = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    np.testing.assert_allclose(dist.icdf(q), mu + sigma * stats.norm.ppf(q.numpy()), atol=0.15)
    # the kernel estimate is the normal density smoothed by the kernel
    h = dist.bandwidth.numpy()
    np.testing.assert_allclose(h, 0.9 * sigma * 4000**-0.2, rtol=0.05)
    np.testing.assert_allclose(dist.log_prob(x), stats.norm.logpdf(x.numpy(), mu, np.sqrt(sigma**2 + h**2)), atol=0.1)
    crps = sigma * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / math.sqrt(math.pi))
    np.testing.assert_allclose(dist.crps(x), crps, rtol=0.04)
    draws = dist.sample(gen, sample_shape=(2000,))
    assert draws.shape == (2000, 3) and bool(torch.isin(draws[:, 1], dist.samples[1]).all())
    np.testing.assert_allclose(draws.mean(0), mu, atol=0.2)
    assert dist.icdf(0.5).shape == (3,) and dist.cdf(torch.zeros(5, 1, dtype=torch.float64)).shape == (5, 3)


def test_empirical_distribution_of_a_two_point_law():
    a, b, m = -1.0, 3.0, 50
    order = torch.randperm(2 * m, generator=torch.Generator().manual_seed(0))
    dist = Empirical(torch.cat([torch.full((m,), a), torch.full((m,), b)]).double()[order])
    assert dist.batch_shape == () and float(dist.mean) == 1.0 and float(dist.stddev) == 2.0
    x = torch.tensor([-2.0, -1.0, 0.0, 3.0, 5.0], dtype=torch.float64)
    assert dist.cdf(x).tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]
    q = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    assert dist.icdf(q).tolist() == [a, a, 0.5 * (a + b), b, b]
    xn = x.numpy()
    np.testing.assert_allclose(dist.crps(x), 0.5 * (np.abs(a - xn) + np.abs(b - xn)) - 0.25 * (b - a), atol=1e-12)
    h = 0.9 * 2.0 * (2 * m) ** -0.2  # sd = 2 is below IQR / 1.34 = 2.99
    density = 0.5 * stats.norm.pdf(xn, a, h) + 0.5 * stats.norm.pdf(xn, b, h)
    np.testing.assert_allclose(dist.log_prob(x), np.log(density), rtol=1e-10)
    draws = dist.sample(torch.Generator().manual_seed(1), sample_shape=(4000,))
    assert set(draws.unique().tolist()) == {a, b} and abs(float((draws == b).double().mean()) - 0.5) < 0.03
    point = Empirical(torch.full((2, 1), 0.5, dtype=torch.float64))  # a point mass: CRPS = absolute error
    np.testing.assert_allclose(point.crps(torch.tensor([1.0, -1.0])), [0.5, 1.5])
    assert bool(torch.isfinite(point.log_prob(0.5)).all()) and point.bandwidth.tolist() == [1e-3, 1e-3]


# ----------------------------------------------------------------------------------- knn
@pytest.fixture(scope="module")
def clean():
    """Identical IDM drivers without noise: training, validation and held-out events."""
    return synthetic(30, 2, 21), synthetic(6, 1, 22), synthetic(8, 1, 23)


@pytest.fixture(scope="module")
def knn(clean):
    train, val, _ = clean
    model = KNN()
    log = model.fit(WindowSet.from_events(train), WindowSet.from_events(val), TrainConfig(device="cpu", seed=0))
    return model, log


def reference_knn(k: int, n: int = 500, **kwargs) -> KNN:
    """A float64 KNN with random reference instants (normalised) and a known scale."""
    gen = torch.Generator().manual_seed(4)
    model = KNN(k=k, **kwargs).double()
    model.set_scale({
        "x_mean": torch.tensor([20.0, 0.5, 15.0, 0.0]), "x_std": torch.tensor([10.0, 2.0, 5.0, 1.0]),
        "a_std": torch.tensor(0.8),
    })
    state = model.state_dict()
    state["x_ref"] = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    state["a_ref"] = torch.randn(n, generator=gen, dtype=torch.float64)
    model.restore(state)
    return model


def physical(model: KNN, rows: torch.Tensor) -> torch.Tensor:
    """Histories ``[B, 1, 4]`` whose normalised last row is ``rows [B, 3]``; a_prev is arbitrary."""
    hist = torch.zeros(len(rows), 1, 4, dtype=torch.float64)
    hist[:, 0, :3] = rows * model.x_std + model.x_mean
    hist[:, 0, A_PREV] = 7.0
    return hist


def test_knn_interface_learning_and_rollout(knn, clean):
    model, log = knn
    train, _, test = clean
    hist = WindowSet.from_events(test).history(torch.arange(0, 1200, 100))
    check_interface(model, hist)
    assert {"best_val_rmse_a", "epochs_run", "seconds", "n_parameters", "history"} <= set(log)
    assert log["n_parameters"] == 0 and log["n_reference"] == len(WindowSet.from_events(train))  # fewer than 50000
    assert p0_rmse(model, test) < 0.3 * p0_rmse(ZeroAcceleration(), test)
    finite_rollout(model, test)
    with_a_prev = KNN(n_reference=3000, use_a_prev=True)
    with_a_prev.fit(WindowSet.from_events(train), WindowSet.from_events(clean[1]), TrainConfig(device="cpu", seed=0))
    assert with_a_prev.x_ref.shape == (3000, 4) and a_prev_changes_the_output(with_a_prev, hist)


def test_knn_returns_the_neighbours_and_matches_brute_force():
    model = reference_knn(k=1)
    rows = model.x_ref[[3, 99, 450]]
    torch.testing.assert_close(model(physical(model, rows)), model.a_ref[[3, 99, 450]] * model.a_std, rtol=0, atol=1e-12)

    model = reference_knn(k=5)
    query = torch.randn(300, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    x, r, a = query.numpy(), model.x_ref.numpy(), model.a_ref.numpy()
    distance = np.sqrt(((x[:, None, :] - r[None, :, :]) ** 2).sum(-1))
    nearest = np.argsort(distance, axis=1)[:, :5]
    weight = 1.0 / (np.take_along_axis(distance, nearest, axis=1) + 1e-6)
    expected = (weight * a[nearest]).sum(1) / weight.sum(1) * float(model.a_std)
    np.testing.assert_allclose(model(physical(model, query)).numpy(), expected, rtol=0, atol=1e-6)

    chunked = KNN(k=5, max_pairs=7 * 500).double()
    chunked.restore(model.state_dict())
    assert torch.equal(chunked(physical(model, query)), model(physical(model, query)))

    # a reference point twice with two accelerations, queried exactly: no NaN, the mean of both
    state = model.state_dict()
    state["x_ref"][1], state["a_ref"][:2] = state["x_ref"][0], torch.tensor([1.0, -0.5], dtype=torch.float64)
    model.restore(state)
    out = model(physical(model, model.x_ref[:1]))
    assert bool(torch.isfinite(out).all())
    assert float(out[0]) == pytest.approx(0.25 * float(model.a_std), abs=1e-4)


def test_knn_checkpoint_round_trip(clean, tmp_path):
    train, val, test = clean
    model = KNN(k=7, n_reference=700)
    model.fit(WindowSet.from_events(train), WindowSet.from_events(val), TrainConfig(device="cpu", seed=3))
    hist = WindowSet.from_events(test).history(torch.arange(0, 1000, 50))
    loaded = round_trip(model, tmp_path / "knn.pt", hist)
    assert loaded.x_ref.shape == (700, 3) and loaded.kwargs["n_reference"] == 700
    default = KNN()  # default arguments: empty buffers
    default.restore(model.state_dict())
    assert default.x_ref.shape == (700, 3) and default.a_ref.shape == (700,)
    same_k = KNN(k=7)
    same_k.restore(model.state_dict())
    assert torch.equal(same_k(hist), model(hist))
    with pytest.raises(RuntimeError, match="reference"):
        KNN()(hist)


# ------------------------------------------------------------------------------ maml_lstm
@pytest.fixture(scope="module")
def drivers():
    """Heterogeneous IDM drivers (theta_cv = 0.3): training, validation and held-out drivers."""
    return synthetic(40, 3, 31, theta_cv=0.3), synthetic(10, 2, 32, theta_cv=0.3), synthetic(16, 1, 33, theta_cv=0.3)


@pytest.fixture(scope="module")
def maml(drivers):
    train, val, _ = drivers
    torch.manual_seed(0)  # the initial weights are drawn when the model is built
    model = MAMLLSTM(meta_batch=16, meta_steps=300, val_every=50)
    log = model.fit(WindowSet.from_events(train), WindowSet.from_events(val), TrainConfig(device="cpu", seed=0))
    return model, log


def warm_up(events: list, dtype=torch.float32) -> torch.Tensor:
    """The observed warm-up ``[E, 30, 4]`` that the protocols give to ``begin``."""
    return EventBatch.from_events(events, dtype=dtype).state()[:, 1 : WARMUP + 1]


def test_maml_adaptation_improves_on_held_out_drivers(maml, drivers):
    model, log = maml
    test = drivers[2]
    assert {"best_val_rmse_a", "val_rmse_a_without_adaptation", "epochs_run", "seconds", "n_parameters",
            "history"} <= set(log)
    assert log["epochs_run"] == 6 and log["n_tasks"] == 40
    with_adaptation = p0_rmse(model, test)
    model.adaptive = False  # the protocol then asks the meta-learned head for every instant
    try:
        without = p0_rmse(model, test)
    finally:
        del model.adaptive
    assert with_adaptation < without
    assert with_adaptation < 0.5 * p0_rmse(ZeroAcceleration(), test)
    sim = finite_rollout(model, test)
    assert model.adapted.shape[0] == sim["s"].shape[0]  # rollout_windows adapts to its instants


def test_maml_interface_and_fallback_to_the_meta_learned_head(maml, drivers):
    model, _ = maml
    model.adapted = None
    hist = WindowSet.from_events(drivers[2]).history(torch.arange(0, 1200, 100))
    out = check_interface(model, hist)  # the meta-learned head
    model.begin(warm_up(drivers[2][:4]))
    assert model.adapted.shape == (4, 65)
    assert torch.equal(model(hist), out)  # 12 histories, not 4 events: the meta-learned head, no error
    per_event = hist.reshape(4, 3, WARMUP, 4)  # three histories of each of the four events
    adapted = model(per_event)
    assert adapted.shape == (4, 3) and torch.equal(model(per_event), adapted)
    assert not torch.allclose(adapted.reshape(-1), out)
    for j in range(3):
        torch.testing.assert_close(model(per_event[:, j]), adapted[:, j], rtol=0, atol=1e-6)
    changed = per_event.clone()
    changed[..., : -model.history, :] += 100.0
    changed[..., A_PREV] += 5.0
    assert torch.equal(model(changed), adapted)
    assert a_prev_changes_the_output(MAMLLSTM(use_a_prev=True), hist)


def test_maml_inner_steps_descend_whatever_the_scale_of_the_features():
    model = MAMLLSTM()
    gen = torch.Generator().manual_seed(0)
    features = 1.0 + 0.1 * torch.randn(2, 20, 64, generator=gen)  # similar rows of large norm, like a warm-up
    target = torch.randn(2, 20, generator=gen)
    x = torch.cat([features, torch.ones(2, 20, 1)], dim=-1)
    assert float(torch.linalg.eigvalsh(x.transpose(1, 2) @ x * (2.0 / 20))[:, -1].min()) > 2.0 / 0.1
    error = lambda heads: torch.mean((model.apply_heads(features, heads) - target) ** 2, dim=-1)  # noqa: E731
    start = model.head_vector().detach().expand(2, -1)
    assert bool((error(model.adapt(features, target).detach()) < error(start)).all())
    gradient = torch.einsum("tnf,tn->tf", x, model.apply_heads(features, start) - target) * (2.0 / 20)
    assert bool((error(start - 0.1 * gradient) > error(start)).all())  # a fixed step of 0.1 would diverge


def test_maml_tasks_are_a_support_block_and_other_instants_of_one_driver(drivers):
    windows = WindowSet.from_events(drivers[0])
    model = MAMLLSTM(meta_batch=8)
    tasks = model._tasks(windows)
    assert len(tasks) == 40
    index = model._draw_tasks(tasks, torch.Generator().manual_seed(0))
    assert index.shape == (8, 60)
    for row in index:
        support = row[:20]
        assert len(set(row.tolist())) == 60 and len(set(windows.driver_of(row).tolist())) == 1
        assert len(set(windows.event[support].tolist())) == 1
        assert torch.equal(windows.step[support], windows.step[support[0]] + torch.arange(20))
    # drivers without a block of n_support instants in one event are left out
    long_blocks = MAMLLSTM(n_support=250, n_query=10)._tasks(windows)
    per_event = torch.bincount(windows.event)
    expected = {int(windows.driver[e]) for e in range(len(per_event)) if per_event[e] >= 250}
    assert 0 < len(long_blocks) < 40
    assert {int(windows.driver_of(instants[:1])) for instants, _ in long_blocks} == expected


def test_maml_begin_uses_only_the_given_rows(maml, drivers):
    model, _ = maml
    batch = EventBatch.from_events(drivers[2][:5], dtype=torch.float32)
    model.begin(batch.state()[:, 1 : WARMUP + 1])
    heads = model.adapted.clone()
    # the protocols pass the rows 1 .. 30: nothing before or after them matters
    later = slice(WARMUP + 1, None)
    other = dataclasses.replace(batch, s=batch.s.clone(), v=batch.v.clone(), a=batch.a.clone(),
                                v_lead=batch.v_lead.clone())
    for name in ("s", "v", "v_lead"):
        getattr(other, name)[:, 0] += 3.0
        getattr(other, name)[:, later] += 3.0
    other.a[:, WARMUP:] += 2.0  # a[30] is the a_prev of row 31
    first = rollout(model, batch).a[:, WARMUP]
    assert torch.equal(rollout(model, other).a[:, WARMUP], first)
    assert torch.equal(model.adapted, heads)
    # a target inside the warm-up of one event changes the head of that event only
    inside = batch.state()[:, 1 : WARMUP + 1].clone()
    inside[2, 20, A_PREV] += 1.0
    model.begin(inside)
    assert not torch.allclose(model.adapted[2], heads[2])
    torch.testing.assert_close(model.adapted[[0, 1, 3, 4]], heads[[0, 1, 3, 4]], rtol=0, atol=1e-6)


def test_maml_heads_differ_between_events_and_protocols_use_them(maml, drivers):
    model, _ = maml
    batch = EventBatch.from_events(drivers[2][:8])
    result = rollout(model, batch)
    assert result.n_invalid == 0 and bool(torch.isfinite(result.s).all())
    heads = model.adapted
    assert heads.shape == (8, 65)
    distance = torch.cdist(heads, heads) + 1e9 * torch.eye(8)
    assert float(distance.min()) > 1e-4
    assert float((heads - model.head_vector().detach()).norm(dim=-1).min()) > 1e-4
    # the first simulated step uses the adapted heads, not the meta-learned one
    hist = batch.state()[:, WARMUP - model.history + 1 : WARMUP + 1]
    with torch.no_grad():  # as in the rollout: the kernels of the LSTM differ in the last bit with autograd
        torch.testing.assert_close(result.a[:, WARMUP], model(hist).clamp(-8.0, 4.0), rtol=0, atol=1e-6)
        meta = model(torch.cat([hist, hist[:1]]))[:8]
    assert not torch.allclose(result.a[:, WARMUP], meta.clamp(-8.0, 4.0))
    table = evaluate_p0(model, drivers[2][:8])
    assert model.adapted.shape == (8, 65) and bool(np.isfinite(table["sse_a"]).all())
    # every adapted head fits the pairs of its warm-up better than the meta-learned head
    warm = warm_up(drivers[2][:8], torch.float64)
    model.begin(warm)
    pairs = warm.unfold(-2, model.history, 1).transpose(-1, -2)[:, :-1]  # [8, 20, history, 4]
    target = warm[:, model.history :, A_PREV]
    adapted = torch.mean((model(pairs) - target) ** 2, dim=-1)
    model.adapted = None
    assert bool((adapted < torch.mean((model(pairs) - target) ** 2, dim=-1)).all())


def test_maml_checkpoint_round_trip(maml, drivers, tmp_path):
    model, _ = maml
    model.adapted = None
    hist = WindowSet.from_events(drivers[2]).history(torch.arange(0, 600, 50))
    loaded = round_trip(model, tmp_path / "maml.pt", hist)
    warm = warm_up(drivers[2][:4])
    model.begin(warm)
    loaded.begin(warm)
    assert torch.equal(loaded(hist.reshape(4, 3, WARMUP, 4)), model(hist.reshape(4, 3, WARMUP, 4)))


# ------------------------------------------------------------------------------ diffusion
@pytest.fixture(scope="module")
def noisy(clean):
    """The clean events with acceleration noise: the conditional standard deviation is NOISE."""
    return tuple(with_noise(events, seed) for events, seed in zip(clean, (1, 2, 3)))


@pytest.fixture(scope="module")
def diffusion(noisy):
    train, val, _ = noisy
    torch.manual_seed(0)
    model = DiffusionCF()
    cfg = TrainConfig(epochs=12, batch_size=512, lr=2e-3, batches_per_epoch=None, patience=12, device="cpu", seed=0)
    log = model.fit(WindowSet.from_events(train), WindowSet.from_events(val), cfg)
    return model, log


def test_diffusion_interface_determinism_and_batch_independence(diffusion, noisy):
    model, _ = diffusion
    windows = WindowSet.from_events(noisy[2])
    hist = windows.history(torch.arange(0, 1200, 100))
    out = check_interface(model, hist)
    others = windows.history(torch.arange(1500, 1600))
    torch.testing.assert_close(model(torch.cat([others[:7], hist[5:6], others[7:]]))[7], out[5], rtol=0, atol=1e-5)
    torch.testing.assert_close(model(hist[5:7]), out[5:7], rtol=0, atol=1e-5)
    draw = lambda h: model.sample(h, torch.Generator().manual_seed(7)).detach()  # noqa: E731
    assert draw(hist).shape == (12,) and torch.equal(draw(hist), draw(hist))
    assert float(draw(hist[:1].expand(40, -1, -1)).std()) > 0.1  # independent noise per history
    dist = model.predict_dist(hist.reshape(3, 4, WARMUP, 4))
    assert isinstance(dist, Empirical) and dist.batch_shape == (3, 4) and dist.n_samples == 64
    assert dist.samples.dtype == hist.dtype and a_prev_changes_the_output(DiffusionCF(use_a_prev=True), hist)


def test_diffusion_learns_and_its_loss_decreases(diffusion, noisy):
    model, log = diffusion
    losses = [h["train_loss"] for h in log["history"]]
    assert losses[-1] < 0.7 * losses[0]
    test = noisy[2]
    constant = p0_rmse(ZeroAcceleration(), test)
    assert p0_rmse(model, test) < NOISE + 0.1 * constant  # the noise is the floor
    finite_rollout(model, test)


def test_diffusion_spread_matches_the_noise_and_forward_is_the_mean_of_the_samples(diffusion, noisy):
    model, _ = diffusion
    windows = WindowSet.from_events(noisy[2])
    hist = windows.history(torch.arange(0, len(windows), 10))
    with torch.no_grad():
        draws = model.sample(hist.expand(64, *hist.shape), torch.Generator().manual_seed(1))
        bias = float(torch.mean(model(hist) - draws.mean(0)))
    assert abs(float(draws.std(0).mean()) / NOISE - 1.0) < 0.35
    quantiles = normal_quantiles(model.n_forward_samples)  # the initial noise of forward
    assert torch.equal(quantiles, -quantiles.flip(0)) and abs(bias) < 0.02


def test_diffusion_distribution_beats_a_point_mass_at_the_mean(diffusion, noisy):
    model, _ = diffusion
    train, test = WindowSet.from_events(noisy[0]), WindowSet.from_events(noisy[2])
    index = torch.arange(0, len(test), 5)
    target = test.target(index)[:, 0]
    with torch.no_grad():
        dist = model.predict_dist(test.history(index))
    assert isinstance(dist, Empirical) and dist.batch_shape == target.shape
    mean = train.target(torch.arange(len(train)))[:, 0].mean()
    point = Empirical(mean.expand(len(index), 1))
    assert float(dist.crps(target).mean()) < 0.5 * float(point.crps(target).mean())


def test_diffusion_checkpoint_round_trip(diffusion, noisy, tmp_path):
    model, _ = diffusion
    hist = WindowSet.from_events(noisy[2]).history(torch.arange(0, 600, 50))
    loaded = round_trip(model, tmp_path / "diffusion.pt", hist)
    assert torch.equal(loaded.predict_dist(hist).samples, model.predict_dist(hist).samples)
