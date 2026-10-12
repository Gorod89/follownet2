"""Reference sequence models: MLP, GRU, LSTM, sequence to sequence, mixture density LSTM.

The learning tests train on synthetic IDM followers (CPU, small budget). With identical
noise-free drivers the acceleration is a function of the current state, so a model that
learns anything must beat the constant-speed follower by far.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.distributions import MixtureSameFamily, Normal

from follownet2.data.batch import A_PREV, WARMUP
from follownet2.data.loaders.synthetic import SyntheticConfig, make_events
from follownet2.data.schema import Event
from follownet2.models import registry
from follownet2.models.sequence import SIGMA_FLOOR, Seq2Seq
from follownet2.sim.differentiable import rollout_windows
from follownet2.train.trainer import TrainConfig
from follownet2.train.windows import WindowSet
from follownet2.utils import REPO_ROOT

NAMES = ("mlp", "gru", "lstm", "seq2seq", "mdn_lstm")
BUDGET = TrainConfig(epochs=6, batch_size=128, lr=3e-3, batches_per_epoch=80, patience=6, device="cpu", seed=0)
#: required one-step RMSE on held-out events, as a fraction of that of the constant-speed follower
LEARNED = 0.25
#: non-default arguments of every model, for the checkpoint round trip
OTHER_KWARGS = {
    "mlp": {"history": 4, "use_a_prev": True, "hidden": [16, 8]},
    "gru": {"history": 6, "use_a_prev": True, "hidden": 16},
    "lstm": {"history": 6, "use_a_prev": False, "hidden": 16},
    "seq2seq": {"history_rows": 6, "delay": 2, "use_a_prev": True, "hidden": 16, "out_steps": 4, "teacher_forcing": 0.25},
    "mdn_lstm": {"history": 6, "use_a_prev": True, "hidden": 16, "n_components": 3},
}


def idm_events(seed: int, n_drivers: int, theta_cv: float = 0.0) -> list[Event]:
    """Two noise-free events of 20 to 35 s per driver."""
    cfg = SyntheticConfig(n_drivers=n_drivers, events_per_driver=2, theta_cv=theta_cv, duration=(20.0, 35.0), seed=seed)
    return make_events(cfg)


def with_acceleration_noise(events: list[Event], sigma: float, seed: int) -> list[Event]:
    """The same events with white noise of standard deviation ``sigma`` added to the acceleration."""
    rng = np.random.default_rng(seed)
    return [dataclasses.replace(ev, a=ev.a + sigma * rng.standard_normal(len(ev))) for ev in events]


def build(name: str, statistics: dict, **kwargs):
    """Model with seeded initial weights and the scale of the training windows, in evaluation mode."""
    torch.manual_seed(0)
    model = registry.create(name, **kwargs)
    model.set_scale(statistics)
    return model.eval()


@torch.no_grad()
def one_step_rmse(model, windows: WindowSet) -> float:
    index = torch.arange(len(windows))
    return float(torch.sqrt(torch.mean((model(windows.history(index)) - windows.target(index)[:, 0]) ** 2)))


def constant_speed_rmse(windows: WindowSet) -> float:
    return float(torch.sqrt(torch.mean(windows.target(torch.arange(len(windows)))[:, 0] ** 2)))


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Train on one thread and restore the setting afterwards.

    The models are small: one thread trains them as fast as many on an idle CPU, while many
    threads stall on a CPU shared with other jobs.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def train():
    return WindowSet.from_events(idm_events(41, 30), horizon=10)  # the decoder of seq2seq is trained on 10 steps


@pytest.fixture(scope="module")
def val():
    return WindowSet.from_events(idm_events(42, 4))


@pytest.fixture(scope="module")
def held_out():
    return WindowSet.from_events(idm_events(43, 10), horizon=100)  # room for the closed-loop check


@pytest.fixture(scope="module")
def statistics(train):
    return train.statistics()


@pytest.fixture(scope="module")
def trained(train, val):
    """``trained(name)``: the model and its training log after ``fit`` with the budget, built once."""
    cache = {}

    def get(name: str):
        if name not in cache:
            torch.manual_seed(0)  # the initial weights are drawn when the model is built
            model = registry.create(name)
            cache[name] = model, model.fit(train, val, BUDGET)
        return cache[name]

    return get


# --------------------------------------------------------------------------- interface
@pytest.mark.parametrize("name", NAMES)
def test_defaults_are_small_and_read_the_specified_history(name):
    model = registry.create(name)
    assert model.name == name and model.family == "network"
    assert sum(p.numel() for p in model.parameters()) < 100_000
    assert model.history == (15 if name == "seq2seq" else 10) and not model.kwargs["use_a_prev"]
    assert model.train_horizon == (10 if name == "seq2seq" else 1)


@pytest.mark.parametrize("name", NAMES)
def test_interface(name, train, statistics):
    model = build(name, statistics)
    hist = train.history(torch.arange(0, 2400, 100)).double()
    out = model(hist)
    assert out.shape == (24,) and out.dtype == torch.float64 and bool(torch.isfinite(out).all())
    assert torch.equal(model(hist.float()), out.float())  # computed in the dtype of the model
    stacked = model(hist.reshape(4, 6, WARMUP, 4))
    assert stacked.shape == (4, 6) and torch.equal(stacked.reshape(-1), out)
    assert torch.equal(model(hist[:, -model.history :]), out)
    changed = hist.clone()
    changed[:, : -model.history] += 100.0  # rows before the history of the model
    changed[..., A_PREV] += 5.0  # not an input by default
    assert torch.equal(model(changed), out)

    with_a_prev = build(name, statistics, use_a_prev=True)
    shifted = hist.clone()
    shifted[..., A_PREV] += 1.0
    assert not torch.allclose(with_a_prev(shifted), with_a_prev(hist))


@pytest.mark.parametrize("name", NAMES)
def test_evaluation_is_deterministic_and_independent_of_the_batch(name, train, statistics):
    model = build(name, statistics)
    hist = train.history(torch.arange(0, 3000, 50))
    out = model(hist)
    assert torch.equal(model(hist), out)
    order = torch.randperm(len(hist), generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(torch.cat([model(hist[order[:7]]), model(hist[order[7:]])]), out[order])
    torch.testing.assert_close(torch.stack([model(h) for h in hist[:5]]), out[:5])


@pytest.mark.parametrize("name", NAMES)
def test_checkpoint_round_trip(name, train, statistics, tmp_path):
    model = build(name, statistics, **OTHER_KWARGS[name])
    path = registry.save_model(model, tmp_path / name / "model.pt", {"seed": 0})
    loaded, provenance = registry.load_model(path)
    assert provenance == {"seed": 0} and loaded.name == name and not loaded.training
    assert loaded.kwargs == OTHER_KWARGS[name] and loaded.history == model.history
    assert loaded.train_horizon == model.train_horizon
    hist = train.history(torch.arange(0, 3000, 50))
    assert torch.equal(loaded(hist), model(hist))
    if name == "seq2seq":
        assert torch.equal(loaded.predict_sequence(hist), model.predict_sequence(hist))
    if name == "mdn_lstm":
        assert torch.equal(loaded.predict_dist(hist).stddev, model.predict_dist(hist).stddev)


@pytest.mark.parametrize("name", NAMES)
def test_configuration_lists_every_constructor_argument_with_its_default(name):
    conf = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "conf" / "model" / f"{name}.yaml"))
    assert set(conf) == {"name", "kwargs", "optim"} and conf["name"] == name
    assert set(conf["kwargs"]) == set(inspect.signature(registry.model_class(name)).parameters)
    assert conf["kwargs"] == registry.create(name).kwargs
    assert set(conf["optim"]) <= {field.name for field in dataclasses.fields(TrainConfig)}


# ---------------------------------------------------------------------------- learning
@pytest.mark.parametrize("name", NAMES)
def test_learns_noise_free_idm_followers(name, trained, held_out):
    model, log = trained(name)
    assert {"best_val_rmse_a", "epochs_run", "seconds", "n_parameters", "history"} <= set(log)
    assert log["n_parameters"] == sum(p.numel() for p in model.parameters()) and not model.training
    assert one_step_rmse(model, held_out) < LEARNED * constant_speed_rmse(held_out)
    with torch.no_grad():
        sim = rollout_windows(model, held_out, torch.arange(0, len(held_out), 25), 100)
    assert all(bool(torch.isfinite(sim[key]).all()) for key in ("s", "v", "a"))


def test_last_acceleration_input_pays_off_where_the_acceleration_is_smooth():
    """The acceleration of IDM followers is smooth, so ``a_prev`` is almost the answer (persistence).

    With heterogeneous drivers the same ``(s, dv, v)`` leads to different accelerations; the
    last acceleration tells how the driver at hand responds, and the model that reads it must
    exploit that within the same budget.
    """
    train = WindowSet.from_events(idm_events(51, 30, theta_cv=0.3))
    val = WindowSet.from_events(idm_events(52, 4, theta_cv=0.3))
    held_out = WindowSet.from_events(idm_events(53, 10, theta_cv=0.3))
    errors = {}
    for use_a_prev in (False, True):
        torch.manual_seed(0)
        model = registry.create("mlp", use_a_prev=use_a_prev)
        model.fit(train, val, BUDGET)
        errors[use_a_prev] = one_step_rmse(model, held_out)
    assert errors[True] < 0.5 * errors[False]


# ----------------------------------------------------------------------------- seq2seq
def test_seq2seq_does_not_read_the_rows_of_the_reaction_delay(train, statistics):
    model = build("seq2seq", statistics)
    hist = train.history(torch.arange(0, 3000, 50))
    out, sequence = model(hist), model.predict_sequence(hist)
    changed = hist.clone()
    changed[:, -5:] += torch.randn(changed[:, -5:].shape, generator=torch.Generator().manual_seed(0))
    assert torch.equal(model(changed), out) and torch.equal(model.predict_sequence(changed), sequence)
    changed[:, -6, :3] += 1.0  # the last row the encoder reads
    assert not torch.allclose(model(changed), out)

    immediate = Seq2Seq(delay=0)
    immediate.load_state_dict(model.state_dict())  # the delay moves the rows read, not the weights
    immediate.eval()
    assert immediate.history == 10
    assert torch.equal(immediate(hist[:, :-5]), out)
    changed = hist.clone()
    changed[:, -1, :3] += 1.0
    assert not torch.allclose(immediate(changed), immediate(hist))


def test_seq2seq_sequence_starts_with_forward_and_is_decoded_step_by_step(train, statistics):
    model = build("seq2seq", statistics)
    hist = train.history(torch.arange(0, 3000, 50)).reshape(4, 15, WARMUP, 4).double()
    sequence = model.predict_sequence(hist)
    assert sequence.shape == (4, 15, 10) and sequence.dtype == torch.float64
    assert torch.equal(sequence[..., 0], model(hist))
    shorter = Seq2Seq(out_steps=4)
    shorter.load_state_dict(model.state_dict())
    assert shorter.train_horizon == 4
    assert torch.equal(shorter.eval().predict_sequence(hist), sequence[..., :4])


def test_seq2seq_teacher_forcing(train, val, statistics):
    index = torch.arange(0, 3000, 30)
    hist, target = train.history(index), train.target(index, 10)

    free = build("seq2seq", statistics, teacher_forcing=0.0).train()
    free_loss = free.training_loss(train, index)
    torch.testing.assert_close(free_loss, torch.mean(((free.predict_sequence(hist) - target) / free.a_std) ** 2))

    forced = build("seq2seq", statistics, teacher_forcing=1.0).train()  # the same weights
    forced_loss = forced.training_loss(train, index)
    assert torch.equal(forced.training_loss(train, index), forced_loss) and not torch.equal(forced_loss, free_loss)
    mixed = build("seq2seq", statistics).train()
    assert not torch.equal(mixed.training_loss(train, index), mixed.training_loss(train, index))

    short = TrainConfig(epochs=2, batch_size=128, batches_per_epoch=3, val_batches=2, device="cpu", seed=7)
    runs = [build("seq2seq", statistics) for _ in range(2)]
    logs = [model.fit(train, val, short) for model in runs]
    assert logs[0]["best_val_rmse_a"] == logs[1]["best_val_rmse_a"]
    assert all(torch.equal(a, b) for a, b in zip(runs[0].state_dict().values(), runs[1].state_dict().values()))


# ---------------------------------------------------------------------------- mdn_lstm
def test_mdn_predictive_distribution(train, statistics):
    model = build("mdn_lstm", statistics)
    hist = train.history(torch.arange(0, 3000, 50)).reshape(4, 15, WARMUP, 4).double()
    with torch.no_grad():
        dist, out = model.predict_dist(hist), model(hist)
    assert isinstance(dist, MixtureSameFamily) and dist.batch_shape == (4, 15)
    assert dist.log_prob(torch.zeros(4, 15, dtype=torch.float64)).shape == (4, 15)
    assert dist.mean.dtype == torch.float64
    torch.testing.assert_close(dist.mean, out)
    assert float(dist.component_distribution.stddev.min()) >= SIGMA_FLOOR * float(model.a_std) * (1 - 1e-6)


def test_mdn_samples_follow_the_predictive_distribution(train, statistics):
    model = build("mdn_lstm", statistics)
    with torch.no_grad():  # components that differ in weight, location and scale, so that a wrong draw shows
        logits, means, raw_std = [1.0, -1.0, 0.5, -0.5, 0.0], [-2.0, -1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, -1.0, -2.0]
        model.head.bias.copy_(torch.tensor(logits + means + raw_std))
    hist = train.history(torch.tensor([0, 1500]))
    few = hist.expand(50, *hist.shape)
    draws = model.sample(few, torch.Generator().manual_seed(1))
    assert draws.shape == (50, 2) and draws.dtype == torch.float32
    assert torch.equal(model.sample(few, torch.Generator().manual_seed(1)), draws)
    assert torch.equal(model.sample(few.double(), torch.Generator().manual_seed(1)), draws.double())
    # Kolmogorov-Smirnov distance between many draws and the mixture, for both histories: a correct
    # sampler exceeds 0.012 with a probability below 1e-4; standard deviations left in normalised
    # units give about 0.02, a wrong component 0.1
    n = 40_000
    draws = model.sample(hist.expand(n, *hist.shape), torch.Generator().manual_seed(0))
    with torch.no_grad():
        cdf = model.predict_dist(hist).cdf(draws.sort(dim=0).values)
    empirical = (torch.arange(n).unsqueeze(-1) + 0.5) / n
    assert float((cdf - empirical).abs().max()) < 0.012


def test_mdn_recovers_the_level_of_acceleration_noise():
    """White noise on the recorded acceleration only: the predictive spread must match its level."""
    sigma = 0.3
    train = WindowSet.from_events(with_acceleration_noise(idm_events(41, 30), sigma, 1))
    val = WindowSet.from_events(with_acceleration_noise(idm_events(42, 4), sigma, 2))
    held_out = WindowSet.from_events(with_acceleration_noise(idm_events(43, 10), sigma, 3))
    torch.manual_seed(0)
    model = registry.create("mdn_lstm")
    model.fit(train, val, BUDGET)
    index = torch.arange(len(held_out))
    target = held_out.target(index)[:, 0]
    with torch.no_grad():
        dist = model.predict_dist(held_out.history(index))
        predicted_std = float(dist.variance.mean().sqrt())
        nll = float(-dist.log_prob(target).mean())
    assert abs(predicted_std - sigma) < 0.3 * sigma
    assert nll < float(-Normal(target.mean(), target.std()).log_prob(target).mean())
