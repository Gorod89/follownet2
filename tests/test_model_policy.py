"""Policy learners: the reward of Zhu et al. (2020), the replay buffer, DDPG and GAIL."""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from omegaconf import OmegaConf
from scipy import stats

from follownet2.data.batch import WARMUP
from follownet2.data.loaders.synthetic import SyntheticConfig, make_events
from follownet2.models import registry
from follownet2.models.persistence import ZeroAcceleration
from follownet2.models.policy import (
    DDPG, GAIL, HEADWAY_MU, HEADWAY_SIGMA, ReplayBuffer, episode_statistics, zhu_reward,
)
from follownet2.sim.differentiable import StepInfo, rollout_windows
from follownet2.train.trainer import TrainConfig
from follownet2.train.windows import WindowSet
from follownet2.utils import REPO_ROOT

F32 = dict(device="cpu", dtype=torch.float32)
#: small networks for the interface tests
SMALL = {"ddpg": dict(hidden=[16, 16]), "gail": dict(hidden=16, disc_hidden=[16, 16])}


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Networks this small run fastest on one thread, above all on a machine that other jobs keep busy."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def train():
    events = make_events(SyntheticConfig(n_drivers=24, events_per_driver=2, theta_cv=0.1, seed=31))
    return WindowSet.from_events(events, horizon=100, **F32)


@pytest.fixture(scope="module")
def val():
    """Events of other drivers, for validation and for the held-out scores."""
    events = make_events(SyntheticConfig(n_drivers=8, events_per_driver=2, theta_cv=0.1, seed=32))
    return WindowSet.from_events(events, horizon=100, **F32)


def build(name: str, train: WindowSet, **kwargs):
    model = registry.create(name, **{**SMALL[name], **kwargs})
    model.set_scale(train.statistics())
    return model.eval()


def constant_speed_rmse(windows: WindowSet) -> float:
    return float(torch.sqrt(torch.mean(windows.target(torch.arange(len(windows)))[:, 0] ** 2)))


# ------------------------------------------------------------------------------ reward
def step(s, dv, v, jerk=0.0, collided=False) -> StepInfo:
    """Environment step quantities of the given states (float64, broadcast to one dimension)."""
    s, dv, v, jerk = torch.broadcast_tensors(
        *(torch.as_tensor(x, dtype=torch.float64).reshape(-1) for x in (s, dv, v, jerk))
    )
    zero = torch.zeros_like(s)
    return StepInfo(
        s=s, v=v, dv=dv, v_lead=v - dv, a=zero, jerk=jerk, collided=torch.as_tensor(collided).expand_as(s),
        s_obs=zero, v_obs=zero, a_obs=zero,
    )


def test_zhu_reward_at_known_states():
    # 20 m at 20 m/s (headway 1 s) without jerk: only the TTC term differs between these states
    base = float(zhu_reward(step(20.0, 0.0, 20.0)))  # not closing in: TTC infinite
    assert float(zhu_reward(step(20.0, 10.0, 20.0))) - base == pytest.approx(math.log(0.5))  # TTC 2 s
    assert float(zhu_reward(step(20.0, 40.0, 20.0))) - base == pytest.approx(math.log(0.125))  # TTC 0.5 s
    assert float(zhu_reward(step(20.0, 5.0, 20.0))) - base == pytest.approx(0.0, abs=1e-12)  # TTC 4 s: log 1
    assert float(zhu_reward(step(20.0, 4.0, 20.0))) == base  # TTC 5 s
    assert float(zhu_reward(step(20.0, -3.0, 20.0))) == base  # opening gap

    # headway term alone: the log-normal density of Zhu et al., largest at exp(mu - sigma^2)
    h = torch.linspace(0.2, 4.0, 3801, dtype=torch.float64)
    density = zhu_reward(step(20.0 * h, -1.0, 20.0))
    reference = stats.lognorm(s=HEADWAY_SIGMA, scale=math.exp(HEADWAY_MU)).pdf(h.numpy())
    torch.testing.assert_close(density, torch.from_numpy(reference), rtol=1e-12, atol=1e-12)
    mode = math.exp(HEADWAY_MU - HEADWAY_SIGMA**2)
    assert float(h[density.argmax()]) == pytest.approx(mode, abs=1e-3)
    standing = float(zhu_reward(step(0.1 * mode, -1.0, 0.0)))  # the headway of a standing follower uses 0.1 m/s
    assert standing == pytest.approx(float(density.max()), rel=1e-5)

    # jerk term: -jerk^2 / 3600, -1 at the largest jerk of the action range (60 m/s^3)
    assert float(zhu_reward(step(20.0, 0.0, 20.0, jerk=60.0))) - base == pytest.approx(-1.0)
    assert float(zhu_reward(step(20.0, 0.0, 20.0, jerk=-30.0))) - base == pytest.approx(-0.25)

    # a collision gives -1 in total, whatever the other terms
    crashed = zhu_reward(step([-0.5, 0.0, 3.0], 5.0, 20.0, jerk=60.0, collided=True))
    assert crashed.tolist() == [-1.0, -1.0, -1.0]


# ------------------------------------------------------------------------ replay buffer
def transitions(first: int, n: int) -> tuple[torch.Tensor, ...]:
    """Transitions numbered ``first .. first + n - 1``; every third one is terminal."""
    k = torch.arange(first, first + n, dtype=torch.float32)
    return k[:, None].repeat(1, 2), k / 100.0, k, k[:, None].repeat(1, 2) + 0.5, k.long() % 3 == 0


def test_replay_buffer_wraps_around_samples_reproducibly_and_keeps_terminal_flags():
    buffer = ReplayBuffer(capacity=10, state_size=2)
    buffer.add(*transitions(0, 6))
    assert len(buffer) == 6 and buffer.position == 6
    buffer.add(*transitions(6, 7))  # 13 transitions in 10 slots: the oldest three are overwritten
    assert len(buffer) == 10 and buffer.position == 3
    assert buffer.reward.tolist() == [10, 11, 12, 3, 4, 5, 6, 7, 8, 9]
    buffer.add(*transitions(13, 25))  # more than the capacity: the newest ten are kept, in ring order
    assert buffer.position == 8 and sorted(buffer.reward.tolist()) == list(range(28, 38))
    assert buffer.reward[8] == 28 and buffer.reward[7] == 37

    first = buffer.sample(256, torch.Generator().manual_seed(4))
    again = buffer.sample(256, torch.Generator().manual_seed(4))
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert not torch.equal(first[2], buffer.sample(256, torch.Generator().manual_seed(5))[2])
    state, action, reward, next_state, terminal = first
    assert set(reward.tolist()) == set(range(28, 38))  # every stored transition, nothing else
    assert torch.equal(state[:, 0], reward) and torch.equal(action, reward / 100.0)
    assert torch.equal(next_state[:, 1], reward + 0.5)
    assert torch.equal(terminal, reward.long() % 3 == 0) and bool(terminal.any()) and not bool(terminal.all())


# ------------------------------------------------------------------------- interface
@pytest.mark.parametrize("name", ["ddpg", "gail"])
def test_interface(name, train):
    model = build(name, train)
    hist = train.history(torch.arange(12))
    out = model(hist)
    assert out.shape == (12,) and out.dtype == torch.float32
    assert torch.equal(model(hist), out)  # deterministic in evaluation mode
    stacked = model(hist.reshape(3, 4, WARMUP, 4))
    assert stacked.shape == (3, 4) and torch.allclose(stacked.reshape(-1), out, atol=1e-6)
    double = model(hist.double())
    assert double.dtype == torch.float64 and torch.allclose(double, out.double(), atol=1e-5)
    changed = hist.clone()
    changed[:, : -model.history] += 100.0  # rows before the history of the model
    changed[..., 3] += 5.0  # a_prev is not an input
    assert torch.equal(model(changed), out)
    with_a_prev = build(name, train, use_a_prev=True)
    assert with_a_prev.n_inputs == 4 and not torch.equal(with_a_prev(changed), with_a_prev(hist))


def test_ddpg_actions_are_bounded_and_the_default_policy_is_memoryless(train):
    model = build("ddpg", train, action_limit=2.0)
    extreme = train.history(torch.arange(64)) * torch.linspace(-50.0, 50.0, 64)[:, None, None]
    with torch.no_grad():
        assert float(model(extreme).abs().max()) <= 2.0
    assert DDPG().memoryless and not DDPG(history=2).memoryless and not DDPG(use_a_prev=True).memoryless
    assert DDPG().history == 1 and GAIL().history == 10
    assert DDPG().train_horizon == 100 and GAIL().train_horizon == 50


@pytest.mark.parametrize("name", ["ddpg", "gail"])
def test_checkpoint_round_trip(name, train, tmp_path):
    model = build(name, train, use_a_prev=True)
    hist = train.history(torch.arange(50))
    loaded, _ = registry.load_model(registry.save_model(model, tmp_path / name / "model.pt"))
    assert loaded.kwargs == model.kwargs and not loaded.training
    assert torch.equal(loaded(hist), model(hist))


def test_gail_distribution_and_samples(train):
    model = build("gail", train, init_std=0.4)
    hist = train.history(torch.arange(6)).reshape(2, 3, WARMUP, 4)
    dist = model.predict_dist(hist)
    assert isinstance(dist, torch.distributions.Normal) and dist.batch_shape == (2, 3)
    assert torch.equal(dist.mean, model(hist)) and torch.allclose(dist.stddev, torch.full((2, 3), 0.4))
    one = model.sample(hist, torch.Generator().manual_seed(7))
    assert one.shape == (2, 3) and torch.equal(one, model.sample(hist, torch.Generator().manual_seed(7)))
    draws = model.sample(hist.expand(4000, 2, 3, WARMUP, 4), torch.Generator().manual_seed(8))
    assert torch.allclose(draws.mean(0), dist.mean, atol=0.03)
    assert torch.allclose(draws.std(0), dist.stddev, rtol=0.06)


@pytest.mark.parametrize("name", ["ddpg", "gail"])
def test_configuration_lists_every_constructor_argument_with_its_default(name):
    conf = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "conf" / "model" / f"{name}.yaml"))
    assert conf["name"] == name and conf["optim"] == {}
    assert conf["kwargs"] == registry.create(name).kwargs


# -------------------------------------------------------------------------- training
@pytest.mark.parametrize(
    "name, budget",
    [
        ("ddpg", dict(n_envs=16, train_horizon=20, iterations=2, updates_per_iteration=4, buffer_size=500)),
        ("gail", dict(n_envs=16, train_horizon=20, iterations=2, bc_epochs=1)),
    ],
)
def test_fit_is_deterministic_for_a_seed(name, budget, train, val):
    cfg = TrainConfig(batch_size=64, batches_per_epoch=3, val_batches=2, device="cpu")
    runs = []
    for seed in (3, 3, 4):
        torch.manual_seed(0)  # the initial weights are drawn when the model is built
        model = registry.create(name, **SMALL[name], **budget)
        state = torch.get_rng_state()
        log = model.fit(train, val, dataclasses.replace(cfg, seed=seed))
        assert torch.equal(torch.get_rng_state(), state)  # the global random state is left alone
        assert not model.training and log["epochs_run"] == 2 and len(log["history"]) == 2
        runs.append((model.state_dict(), log))
    (first, log), (again, log_again), (other, _) = runs
    assert first.keys() == again.keys() and all(torch.equal(first[k], again[k]) for k in first)
    assert log["history"] == log_again["history"] and log["best_val_rmse_a"] == log_again["best_val_rmse_a"]
    assert not all(torch.equal(first[k], other[k]) for k in first)  # another seed, another result


@pytest.fixture(scope="module")
def ddpg(train, val):
    torch.manual_seed(0)  # the initial weights are drawn when the model is built
    model = DDPG(n_envs=64, iterations=60, updates_per_iteration=100, buffer_size=50_000)
    log = model.fit(train, val, TrainConfig(batch_size=256, seed=0, device="cpu"))
    return model, log


def test_ddpg_learns_safe_efficient_and_comfortable_following(ddpg, train, val):
    """The policy is judged by the reward of Zhu et al., its collisions and its time headway.

    The reward does not imitate the driver: the one-step error against the recorded
    acceleration is NOT expected to be small (it is larger than that of a constant-speed
    follower here) and is only checked to be what the log reports.
    """
    model, log = ddpg
    index = torch.randperm(len(val), generator=torch.Generator().manual_seed(1))[:512]
    torch.manual_seed(0)
    untrained = DDPG()
    untrained.set_scale(train.statistics())
    learned = episode_statistics(model, val, index, 100)
    assert learned["reward"] > episode_statistics(untrained.eval(), val, index, 100)["reward"]
    assert learned["reward"] > episode_statistics(ZeroAcceleration(), val, index, 100)["reward"]
    assert learned["collision_rate"] < 0.05
    assert 0.8 < learned["headway"] < 2.5
    # the kept policy is the best on the validation episodes, and its one-step error is reported
    history = log["history"]
    assert log["val_reward"] == max(h["val_reward"] for h in history) == history[log["best_iteration"]]["val_reward"]
    assert {"reward", "collision_rate", "headway", "critic_loss", "val_reward"} <= history[0].keys()
    error = model.validation_error(val, torch.arange(len(val))) / len(val)
    assert log["best_val_rmse_a"] == pytest.approx(float(error.sqrt()), rel=1e-5)
    with torch.no_grad():
        sim = rollout_windows(model, val, index, 100)
    assert all(bool(torch.isfinite(sim[key]).all()) for key in ("s", "v", "a"))


@pytest.fixture(scope="module")
def gail(train, val):
    """GAIL after a small budget, starting from a policy much noisier than the drivers.

    The recorded accelerations are an almost deterministic function of the state, and the
    mean of the cloned policy is close to it: with the default ``init_std=0.5`` the
    discriminator of such a short run may stay at chance and has nothing to show.
    """
    torch.manual_seed(0)  # the initial weights are drawn when the model is built
    model = GAIL(hidden=32, n_envs=64, iterations=25, init_std=1.5)
    log = model.fit(train, val, TrainConfig(batch_size=512, batches_per_epoch=None, seed=0, device="cpu"))
    return model, log


def test_gail_imitates_the_driver(gail, val):
    model, log = gail
    assert log["best_val_rmse_a"] < 0.4 * constant_speed_rmse(val)
    # the policy became harder to tell apart: the final discriminator recognises its pairs less
    # often than those of the first iteration (the accuracy logged after the first iteration
    # comes from a discriminator trained for two passes only and is no yardstick)
    history = log["history"]
    assert history[-1]["d_acc_generator"] < log["d_acc_initial_generator"]
    assert history[-1]["std"] < history[0]["std"]
    assert {"d_loss", "d_acc_expert", "d_acc_generator", "reward", "collision_rate"} <= history[0].keys()
    with torch.no_grad():
        sim = rollout_windows(model, val, torch.arange(0, len(val), 9), 100)
    assert all(bool(torch.isfinite(sim[key]).all()) for key in ("s", "v", "a"))
