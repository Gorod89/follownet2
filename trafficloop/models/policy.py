"""Policy-learning reference models: DDPG (Zhu et al. 2020) and GAIL (Kuefler et al. 2017 style).

Both learn in closed loop: the policy drives the follower in
:class:`~trafficloop.sim.differentiable.FollowEnv` behind the recorded leaders of the training
windows. At the interface they are models like the others, ``acc = model(hist)``, with the
inputs normalised by the statistics of the training windows (:meth:`NetworkModel.set_scale`).
``validation_error`` stays the one-step squared error of ``forward`` against the recorded
acceleration, so that ``best_val_rmse_a`` of their logs is comparable with the supervised
models; neither of them minimises it.

Everything random in ``fit`` (initial weights of the networks built there, episode starts,
exploration, minibatches) comes from ``cfg.seed``, drawn from generators on the device of
the windows; the global random state is left unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from dataclasses import asdict
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from trafficloop.models.base import CFModel
from trafficloop.models.network import NetworkModel, mlp
from trafficloop.sim.differentiable import FollowEnv, StepInfo
from trafficloop.train.trainer import TrainConfig, validate
from trafficloop.train.windows import WindowSet

#: time to collision [s] up to which the safety term of Zhu et al. (2020) is active
TTC_MAX = 4.0
#: log-normal distribution of the time headway [s] of Zhu et al. (2020), fitted to NGSIM
HEADWAY_MU, HEADWAY_SIGMA = 0.4226, 0.4365
#: square of the largest jerk of the action range, (2 * 3 m/s^2 / 0.1 s)^2 [m^2/s^6]
JERK_SCALE = 3600.0
#: reward of a step that ends in a collision; the episode ends there
COLLISION_REWARD = -1.0
#: speed [m/s] below which the time headway is taken at this speed
HEADWAY_SPEED_FLOOR = 0.1


def zhu_reward(info: StepInfo) -> torch.Tensor:
    """Reward of Zhu et al. (2020) for one environment step, shape ``[B]``.

    ``F_ttc + F_headway + F_jerk``: ``F_ttc = log(TTC / 4)`` for ``0 <= TTC <= 4 s`` and 0
    otherwise, with ``TTC = s / dv`` when closing in (``dv > 0``) and infinite otherwise;
    ``F_headway`` the log-normal density (``HEADWAY_MU``, ``HEADWAY_SIGMA``) at the time
    headway ``s / max(v, 0.1)``; ``F_jerk = -jerk^2 / 3600``. A step that ends in a
    collision gets -1 in total.
    """
    s = info.s
    ttc = torch.where(info.dv > 0, s / info.dv, torch.full_like(s, math.inf))
    f_ttc = torch.where((ttc >= 0) & (ttc <= TTC_MAX), torch.log(ttc / TTC_MAX), torch.zeros_like(s))
    headway = s / info.v.clamp_min(HEADWAY_SPEED_FLOOR)
    h = headway.clamp_min(1e-12)  # the density is zero for h <= 0; the floor only keeps the log finite
    density = torch.exp(-((torch.log(h) - HEADWAY_MU) ** 2) / (2.0 * HEADWAY_SIGMA**2)) / (
        h * HEADWAY_SIGMA * math.sqrt(2.0 * math.pi)
    )
    f_headway = torch.where(headway > 0, density, torch.zeros_like(s))
    f_jerk = -(info.jerk**2) / JERK_SCALE
    return torch.where(info.collided, torch.full_like(s, COLLISION_REWARD), f_ttc + f_headway + f_jerk)


def _summary(
    alive: torch.Tensor, reward: torch.Tensor, collided: torch.Tensor, gap: torch.Tensor, speed: torch.Tensor
) -> dict[str, float]:
    """Mean episode reward, collision rate and mean time headway of steps stacked to ``[T, B]``.

    ``alive`` marks the steps inside an episode (up to and including a collision). The time
    headway ``s / v`` is averaged over the steps that end without a collision and with the
    follower moving (``v > 0.1 m/s``): at a standstill it is not defined, and a few followers
    stopped far behind their leader would otherwise dominate the mean.
    """
    moving = alive & ~collided & (speed > HEADWAY_SPEED_FLOOR)
    return {
        "reward": float(torch.where(alive, reward, torch.zeros_like(reward)).sum(0).mean()),
        "collision_rate": float(collided.any(0).float().mean()),
        "headway": float((gap[moving] / speed[moving]).mean()),
    }


@torch.no_grad()
def episode_statistics(model: CFModel, windows: WindowSet, index: torch.Tensor, horizon: int) -> dict[str, float]:
    """Mean episode reward of :func:`zhu_reward`, collision rate and mean time headway [s] of ``model``.

    ``model`` drives the follower in :class:`FollowEnv` for ``horizon`` steps from every
    instant of ``index``; an episode ends at its first collision.
    """
    env = FollowEnv(windows, horizon)
    obs = env.reset(index)
    steps = []
    for _ in range(horizon):
        alive = env.alive.clone()
        obs, info, _ = env.step(model(obs))
        steps.append((alive, zhu_reward(info), info.collided, info.s, info.v))
    return _summary(*(torch.stack(x) for x in zip(*steps)))


class ReplayBuffer:
    """Ring buffer of transitions ``(state, action, reward, next_state, terminal)`` on one device.

    New transitions overwrite the oldest ones once the buffer is full. ``terminal`` marks the
    transitions that end their episode for good (a collision): their value target does not
    bootstrap.
    """

    def __init__(
        self, capacity: int, state_size: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        self.capacity = int(capacity)
        self.state = torch.zeros(self.capacity, state_size, device=device, dtype=dtype)
        self.next_state = torch.zeros_like(self.state)
        self.action = torch.zeros(self.capacity, device=device, dtype=dtype)
        self.reward = torch.zeros_like(self.action)
        self.terminal = torch.zeros(self.capacity, device=device, dtype=torch.bool)
        self.position = 0  # slot of the next transition
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminal: torch.Tensor,
    ) -> None:
        """Append a batch of transitions (first dimension); of more than ``capacity`` the newest are kept."""
        n = int(state.shape[0])
        skip = max(n - self.capacity, 0)
        slots = (self.position + skip + torch.arange(n - skip, device=self.state.device)) % self.capacity
        for store, values in (
            (self.state, state), (self.action, action), (self.reward, reward),
            (self.next_state, next_state), (self.terminal, terminal),
        ):
            store[slots] = values[skip:].to(store.dtype)
        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        """``batch_size`` transitions drawn uniformly with replacement.

        Returns their states, actions, rewards, next states and terminal flags.
        """
        index = torch.randint(self.size, (batch_size,), generator=generator, device=generator.device)
        index = index.to(self.state.device)
        return self.state[index], self.action[index], self.reward[index], self.next_state[index], self.terminal[index]


@contextlib.contextmanager
def _seeded(seed: int) -> Iterator[None]:
    """Networks built inside draw their initial weights from ``seed``; the global random state is restored."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        yield


def _minibatches(
    n: int, batch_size: int, generator: torch.Generator, limit: int | None = None
) -> Iterator[torch.Tensor]:
    """Shuffled index tensors of at most ``batch_size`` covering ``range(n)``; ``limit`` caps their number."""
    order = torch.randperm(n, generator=generator, device=generator.device)
    starts = range(0, n, batch_size)
    for start in starts if limit is None else starts[:limit]:
        yield order[start : start + batch_size]


def _step(optimiser: torch.optim.Optimizer, loss: torch.Tensor, grad_clip: float) -> None:
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_([p for group in optimiser.param_groups for p in group["params"]], grad_clip)
    optimiser.step()


@torch.no_grad()
def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t, s in zip(target.parameters(), source.parameters()):
        t.lerp_(s, tau)


class DDPG(NetworkModel):
    """Deep deterministic policy gradient follower of Zhu, Wang, Pu, Hu, Wang, Ke (2020).

    The policy maximises the reward of :func:`zhu_reward` (safety, efficiency, comfort), not
    the likelihood of the recorded driver. Actor and critic are MLPs (``hidden``, ReLU) on the
    normalised state, the last ``history`` rows ``(s, dv, v)`` (the last row, as in the
    paper, by default); the action is ``action_limit * tanh(.)`` m/s^2 and the critic reads
    (state, action / action_limit). ``forward`` is the actor without noise.

    One training iteration runs ``n_envs`` episodes of ``train_horizon`` steps in
    :class:`FollowEnv` from instants drawn at random from the training windows, with Gaussian
    exploration noise whose standard deviation decays linearly from ``noise`` to
    ``noise_final`` m/s^2 over the iterations (noisy actions are clipped to the action
    range), adds their transitions to a replay buffer of ``buffer_size``, and makes
    ``updates_per_iteration`` gradient steps on ``cfg.batch_size`` transitions: Adam with
    ``actor_lr`` and ``critic_lr``, target networks updated softly with ``tau``, discount
    ``gamma``. A collision ends its episode and its transition does not bootstrap; a
    transition cut by the horizon does. After every iteration the policy drives
    ``cfg.batch_size`` validation episodes drawn once from the validation windows, and the
    policy with the best mean validation reward is kept. ``best_val_rmse_a`` is the one-step
    error of that policy against the recorded acceleration: reported, not used for selection.

    Differences from the paper:

    * no safety check: the paper overrides unsafe actions by a collision-avoidance rule in
      training and testing, here the learned policy alone is scored; the reward of -1 for a
      collision, which ends the episode, is the benchmark's;
    * episodes are 10 s segments from random instants of the training windows, many in
      parallel, instead of whole NGSIM car-following events;
    * the network sizes, learning rates, replay buffer, minibatch, target rate, discount and
      the Gaussian exploration noise with a linear decay are the benchmark's settings; the
      paper's values were not checked (its networks and exploration process may differ);
    * the inputs are normalised by the statistics of the training windows, and the policy is
      selected by the mean reward on held-out validation episodes.
    """

    name = "ddpg"

    def __init__(
        self,
        history: int = 1,
        use_a_prev: bool = False,
        hidden: Sequence[int] = (64, 64),
        action_limit: float = 3.0,
        train_horizon: int = 100,
        n_envs: int = 256,
        buffer_size: int = 200_000,
        noise: float = 0.3,
        noise_final: float = 0.05,
        tau: float = 0.005,
        gamma: float = 0.95,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        iterations: int = 200,
        updates_per_iteration: int = 100,
    ) -> None:
        super().__init__(history, use_a_prev)
        self.hidden = [int(n) for n in hidden]
        self.action_limit, self.noise, self.noise_final = float(action_limit), float(noise), float(noise_final)
        self.tau, self.gamma = float(tau), float(gamma)
        self.actor_lr, self.critic_lr = float(actor_lr), float(critic_lr)
        self.train_horizon, self.n_envs, self.buffer_size = int(train_horizon), int(n_envs), int(buffer_size)
        self.iterations, self.updates_per_iteration = int(iterations), int(updates_per_iteration)
        self.kwargs.update(
            hidden=self.hidden, action_limit=self.action_limit, train_horizon=self.train_horizon,
            n_envs=self.n_envs, buffer_size=self.buffer_size, noise=self.noise, noise_final=self.noise_final,
            tau=self.tau, gamma=self.gamma, actor_lr=self.actor_lr, critic_lr=self.critic_lr,
            iterations=self.iterations, updates_per_iteration=self.updates_per_iteration,
        )
        # a law a = f(s, dv, v): the analytic string-stability criterion applies
        self.memoryless = self.history == 1 and not use_a_prev
        self.actor = mlp([self.state_size, *self.hidden, 1], nn.ReLU)

    @property
    def state_size(self) -> int:
        return self.history * self.n_inputs

    def state(self, hist: torch.Tensor) -> torch.Tensor:
        """Normalised states ``[B, state_size]`` of histories ``[B, rows, 4]``."""
        return self.inputs(hist.to(self.x_mean.dtype)).flatten(1)

    def policy(self, state: torch.Tensor) -> torch.Tensor:
        """Action of the actor divided by ``action_limit``, in (-1, 1), for normalised states."""
        return torch.tanh(self.actor(state)).squeeze(-1)

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_limit * self.policy(x.flatten(1)) / self.a_std

    def fit(self, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
        """Train by DDPG on episodes behind the leaders of ``train``; keep the best policy on ``val``.

        Of ``cfg`` it uses ``batch_size`` (minibatch of the updates and number of validation
        episodes), ``grad_clip``, ``seed`` and, for ``best_val_rmse_a``, ``val_batches``; the
        learning rates are constructor arguments.
        """
        started = time.time()
        device = train.device
        self.to(device)
        self.set_scale(train.statistics(seed=cfg.seed))
        dtype = self.x_mean.dtype
        gen = torch.Generator(device=device).manual_seed(cfg.seed)
        with _seeded(cfg.seed):
            critic = mlp([self.state_size + 1, *self.hidden, 1], nn.ReLU).to(device=device, dtype=dtype)
        actor_target, critic_target = copy.deepcopy(self.actor), copy.deepcopy(critic)
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
        buffer = ReplayBuffer(self.buffer_size, self.state_size, device, dtype)
        env = FollowEnv(train, self.train_horizon)
        val_index = torch.randperm(len(val), generator=gen, device=device)[: cfg.batch_size]

        def value(net: nn.Module, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return net(torch.cat([state, action.unsqueeze(-1)], dim=-1)).squeeze(-1)

        best, best_state, best_iteration, history = -math.inf, copy.deepcopy(self.state_dict()), -1, []
        for iteration in range(self.iterations):
            self.train()
            sigma = self.noise + (self.noise_final - self.noise) * iteration / max(self.iterations - 1, 1)
            start = torch.randint(len(train), (self.n_envs,), generator=gen, device=device)
            explored = self._explore(env, start, sigma, buffer, gen)
            critic_loss = torch.zeros((), device=device, dtype=dtype)
            for _ in range(self.updates_per_iteration):
                state, action, reward, next_state, terminal = buffer.sample(cfg.batch_size, gen)
                with torch.no_grad():
                    ahead = value(critic_target, next_state, torch.tanh(actor_target(next_state)).squeeze(-1))
                    target = torch.where(terminal, reward, reward + self.gamma * ahead)
                loss = F.mse_loss(value(critic, state, action), target)
                _step(critic_opt, loss, cfg.grad_clip)
                _step(actor_opt, -value(critic, state, self.policy(state)).mean(), cfg.grad_clip)
                _soft_update(critic_target, critic, self.tau)
                _soft_update(actor_target, self.actor, self.tau)
                critic_loss += loss.detach()
            self.eval()
            scores = episode_statistics(self, val, val_index, self.train_horizon)
            history.append({
                "iteration": iteration, "noise": sigma, **explored,
                "critic_loss": float(critic_loss) / max(self.updates_per_iteration, 1),
                **{f"val_{key}": score for key, score in scores.items()},
            })
            if scores["reward"] > best:
                best, best_state, best_iteration = scores["reward"], copy.deepcopy(self.state_dict()), iteration
        self.load_state_dict(best_state)
        self.eval()
        final = episode_statistics(self, val, val_index, self.train_horizon)
        return {
            "best_val_rmse_a": validate(self, val, cfg),
            **{f"val_{key}": score for key, score in final.items()},
            "best_iteration": best_iteration,
            "epochs_run": len(history),
            "history": history,
            "seconds": round(time.time() - started, 1),
            "n_parameters": sum(p.numel() for p in self.parameters()),
            "config": asdict(cfg),
        }

    @torch.no_grad()
    def _explore(
        self, env: FollowEnv, start: torch.Tensor, sigma: float, buffer: ReplayBuffer, gen: torch.Generator
    ) -> dict[str, float]:
        """Noisy episodes from the instants ``start``: their transitions go to ``buffer``, their statistics back."""
        obs = env.reset(start)
        state = self.state(obs)
        steps = []
        for _ in range(self.train_horizon):
            alive = env.alive.clone()
            noise = torch.randn(len(start), generator=gen, device=gen.device, dtype=state.dtype)
            action = (self.policy(state) + sigma / self.action_limit * noise).clamp(-1.0, 1.0)
            obs, info, _ = env.step(self.action_limit * action.to(obs.dtype))
            next_state = self.state(obs)
            reward = zhu_reward(info).to(state.dtype)
            steps.append((alive, state, action, reward, next_state, info.collided, info.s, info.v))
            state = next_state
        alive, state, action, reward, next_state, collided, gap, speed = (torch.stack(x) for x in zip(*steps))
        buffer.add(state[alive], action[alive], reward[alive], next_state[alive], collided[alive])
        return _summary(alive, reward, collided, gap, speed)


class _Recurrent(nn.Module):
    """GRU over the history rows and a linear head on its last hidden state (zero initial state)."""

    def __init__(self, n_inputs: int, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(n_inputs, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gru(x)[0][:, -1]).squeeze(-1)


def _advantages(
    reward: torch.Tensor, value: torch.Tensor, terminal: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """Generalised advantage estimates ``[T, B]`` (Schulman et al. 2016).

    ``value`` has ``T + 1`` rows, the last for the state after the final step: an episode cut
    by the horizon bootstraps from it, a terminal step (collision) does not.
    """
    advantage = torch.empty_like(reward)
    running = torch.zeros_like(reward[0])
    for t in range(reward.shape[0] - 1, -1, -1):
        keep = (~terminal[t]).to(reward.dtype)
        running = reward[t] + gamma * keep * value[t + 1] - value[t] + gamma * lam * keep * running
        advantage[t] = running
    return advantage


class GAIL(NetworkModel):
    """Generative adversarial imitation of the driver in the style of Kuefler, Morton, Wheeler, Kochenderfer (2017).

    Policy: Gaussian over the acceleration; its mean comes from a GRU (``hidden`` units) over
    the normalised last ``history`` rows ``(s, dv, v)``, its standard deviation is one
    learnable number (``init_std`` m/s^2 at the start) for every state. Expert pairs are the
    recorded (history, acceleration) of the training windows; generator pairs come from
    ``n_envs`` episodes of ``train_horizon`` steps of the stochastic policy in
    :class:`FollowEnv`, from instants drawn at random from the training windows, per
    iteration. The discriminator, an MLP (``disc_hidden``, tanh, sigmoid output) on the
    normalised last row and the action divided by the standard deviation of the recorded
    accelerations, gives the probability ``D`` that a pair is the driver's; it is trained by
    binary cross-entropy with targets ``1 - label_smoothing`` (expert) and
    ``label_smoothing`` (generator), ``d_steps`` passes over the generator pairs of the
    iteration, each minibatch with as many expert pairs drawn at random. The policy is
    rewarded with ``-log(1 - D + 1e-8)`` and updated by PPO: clipped ratio (``clip``),
    ``ppo_epochs`` passes over the pairs of the iteration, generalised advantage estimation
    (``gamma``, ``lam``) with a value network of the same shape as the policy mean, entropy
    bonus ``entropy``. A collision ends its episode and does not bootstrap. Before the
    adversarial phase, ``bc_epochs`` epochs of behaviour cloning fit the mean to the recorded
    accelerations (the supervised one-step loss).

    ``forward`` is the mean of the policy, :meth:`sample` a draw from it and
    :meth:`predict_dist` the Gaussian itself. The policy after the last iteration is kept;
    ``best_val_rmse_a`` is its one-step error on the validation windows.

    Differences from the paper:

    * PPO with a value network replaces the TRPO step of the paper;
    * the behaviour-cloning warm start is not part of the paper (``bc_epochs=0`` switches it
      off);
    * car following behind a recorded leader, observation the ``(s, dv, v)`` history and
      action the acceleration, instead of the multi-lane highway simulation of the paper with
      its feature vector (ego state, lane geometry, range sensor) and its two actions
      (acceleration and turn rate);
    * the sizes (GRU of 64, discriminator [64, 64], horizon, batch), the label smoothing and
      the discriminator schedule are the benchmark's settings; the paper's were not checked.
    """

    name = "gail"

    def __init__(
        self,
        history: int = 10,
        use_a_prev: bool = False,
        hidden: int = 64,
        init_std: float = 0.5,
        disc_hidden: Sequence[int] = (64, 64),
        train_horizon: int = 50,
        n_envs: int = 256,
        iterations: int = 150,
        ppo_epochs: int = 4,
        clip: float = 0.2,
        gamma: float = 0.95,
        lam: float = 0.95,
        entropy: float = 1e-3,
        d_steps: int = 2,
        label_smoothing: float = 0.1,
        bc_epochs: int = 2,
    ) -> None:
        super().__init__(history, use_a_prev)
        self.hidden, self.disc_hidden = int(hidden), [int(n) for n in disc_hidden]
        self.train_horizon, self.n_envs, self.iterations = int(train_horizon), int(n_envs), int(iterations)
        self.ppo_epochs, self.clip, self.gamma, self.lam = int(ppo_epochs), float(clip), float(gamma), float(lam)
        self.entropy, self.d_steps, self.label_smoothing = float(entropy), int(d_steps), float(label_smoothing)
        self.bc_epochs = int(bc_epochs)
        self.kwargs.update(
            hidden=self.hidden, init_std=float(init_std), disc_hidden=self.disc_hidden,
            train_horizon=self.train_horizon, n_envs=self.n_envs, iterations=self.iterations,
            ppo_epochs=self.ppo_epochs, clip=self.clip, gamma=self.gamma, lam=self.lam, entropy=self.entropy,
            d_steps=self.d_steps, label_smoothing=self.label_smoothing, bc_epochs=self.bc_epochs,
        )
        self.policy = _Recurrent(self.n_inputs, self.hidden)
        self.log_std = nn.Parameter(torch.tensor(math.log(init_std)))

    @property
    def std(self) -> torch.Tensor:
        """Standard deviation of the policy [m/s^2], the same for every state."""
        return self.log_std.exp()

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    def predict_dist(self, hist: torch.Tensor) -> Normal:
        """The Gaussian policy over the acceleration in m/s^2, batch shape ``hist.shape[:-2]``."""
        mean = self(hist)
        return Normal(mean, self.std.to(mean.dtype).expand_as(mean), validate_args=False)

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """One draw of the policy per history [m/s^2], shape ``hist.shape[:-2]``.

        The normal numbers come from ``generator`` on its own device, so the draws do not
        depend on the device of the model.
        """
        mean = self(hist)
        noise = torch.randn(mean.shape, generator=generator, device=generator.device).to(mean)
        return mean + self.std.to(mean.dtype) * noise

    def fit(self, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
        """Behaviour cloning, then adversarial imitation of the recorded drivers of ``train``.

        Of ``cfg`` it uses ``lr`` (Adam, every network), ``batch_size`` (minibatches of the
        cloning, the discriminator and PPO), ``batches_per_epoch`` (cloning), ``grad_clip``,
        ``seed`` and, for ``best_val_rmse_a``, ``val_batches``.

        The log holds per iteration the discriminator loss, its accuracy on the expert and on
        the generator pairs after its update, the mean reward, the collision rate and the
        standard deviation of the policy. ``d_acc_initial_generator`` is the accuracy of the
        final discriminator on the generator pairs of the first iteration: next to
        ``history[-1]["d_acc_generator"]`` it measures, with one discriminator, how much harder
        to tell apart the policy became. The accuracy logged after the first iteration cannot
        serve for that: two passes leave the discriminator near chance, and against the
        over-dispersed initial policy it labels most pairs of both kinds as expert pairs.
        """
        started = time.time()
        device = train.device
        self.to(device)
        self.set_scale(train.statistics(seed=cfg.seed))
        dtype = self.x_mean.dtype
        gen = torch.Generator(device=device).manual_seed(cfg.seed)
        with _seeded(cfg.seed):
            value = _Recurrent(self.n_inputs, self.hidden).to(device=device, dtype=dtype)
            disc = mlp([self.n_inputs + 1, *self.disc_hidden, 1]).to(device=device, dtype=dtype)
        self.train()
        bc_loss = self._clone(train, cfg, gen)
        policy_opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        value_opt = torch.optim.Adam(value.parameters(), lr=cfg.lr)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg.lr)
        env = FollowEnv(train, self.train_horizon)
        history, first = [], None
        for iteration in range(self.iterations):
            start = torch.randint(len(train), (self.n_envs,), generator=gen, device=device)
            x, action, log_prob, alive, collided = self._generate(env, start, gen)
            pairs = self._pairs(x[:-1, :, -1], action)  # [T, B, C + 1], every step
            generated = pairs[alive]
            first = generated if first is None else first
            index = torch.randint(len(train), (len(generated),), generator=gen, device=device)
            rows = self.inputs(train.history(index, self.history).to(dtype))[:, -1]
            expert = self._pairs(rows, train.target(index)[:, 0])
            d_loss = self._discriminate(disc, disc_opt, expert, generated, cfg, gen)
            with torch.no_grad():
                p_expert = torch.sigmoid(disc(expert)).squeeze(-1)
                p_all = torch.sigmoid(disc(pairs)).squeeze(-1)
                reward = -torch.log(1.0 - p_all + 1e-8)
                values = value(x.flatten(0, 1)).view(x.shape[:2])
                advantage = _advantages(reward, values, collided, self.gamma, self.lam)
            returns = advantage + values[:-1]
            self._ppo(value, policy_opt, value_opt, x[:-1][alive], action[alive], log_prob[alive],
                      advantage[alive], returns[alive], cfg, gen)
            history.append({
                "iteration": iteration,
                "d_loss": d_loss,
                "d_acc_expert": float((p_expert > 0.5).float().mean()),
                "d_acc_generator": float((p_all[alive] < 0.5).float().mean()),
                "reward": float(reward[alive].mean()),
                "collision_rate": float(collided.any(0).float().mean()),
                "std": float(self.std.detach()),
            })
        self.eval()
        with torch.no_grad():
            initial = math.nan if first is None else float((disc(first) < 0.0).float().mean())
        return {
            "best_val_rmse_a": validate(self, val, cfg),
            "d_acc_initial_generator": initial,
            "bc_loss": bc_loss,
            "epochs_run": len(history),
            "history": history,
            "seconds": round(time.time() - started, 1),
            "n_parameters": sum(p.numel() for p in self.parameters()),
            "config": asdict(cfg),
        }

    def _pairs(self, row: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Discriminator inputs: normalised last rows ``[..., C]`` and the actions [m/s^2] divided by ``a_std``."""
        return torch.cat([row, (action / self.a_std).unsqueeze(-1).to(row.dtype)], dim=-1)

    def _clone(self, train: WindowSet, cfg: TrainConfig, gen: torch.Generator) -> list[float]:
        """Behaviour-cloning warm start; returns the mean one-step loss of every epoch."""
        optimiser = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        losses = []
        for _ in range(self.bc_epochs):
            total, count = torch.zeros((), device=train.device), 0
            for index in _minibatches(len(train), cfg.batch_size, gen, cfg.batches_per_epoch):
                loss = self.training_loss(train, index)
                _step(optimiser, loss, cfg.grad_clip)
                total, count = total + loss.detach(), count + 1
            losses.append(float(total) / max(count, 1))
        return losses

    @torch.no_grad()
    def _generate(self, env: FollowEnv, start: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        """Episodes of the stochastic policy from the instants ``start``.

        Returns the normalised inputs ``[T + 1, B, history, C]`` (the last entry is the state
        after the final step), the actions [m/s^2] and their log-probabilities, the mask of
        the steps inside an episode and the collision flags, each ``[T, B]``.
        """
        std = self.std
        inputs, steps = [], []
        obs = env.reset(start)
        for _ in range(self.train_horizon):
            x = self.inputs(obs.to(self.x_mean.dtype))
            mean = self.core(x) * self.a_std
            action = mean + std * torch.randn(mean.shape, generator=gen, device=gen.device, dtype=mean.dtype)
            alive = env.alive.clone()
            obs, info, _ = env.step(action.to(obs.dtype))
            inputs.append(x)
            log_prob = Normal(mean, std.expand_as(mean), validate_args=False).log_prob(action)
            steps.append((action, log_prob, alive, info.collided))
        inputs.append(self.inputs(obs.to(self.x_mean.dtype)))
        return (torch.stack(inputs), *(torch.stack(column) for column in zip(*steps)))

    def _discriminate(
        self, disc: nn.Module, optimiser: torch.optim.Optimizer, expert: torch.Tensor, generated: torch.Tensor,
        cfg: TrainConfig, gen: torch.Generator,
    ) -> float:
        """``d_steps`` passes of binary cross-entropy with smoothed labels; returns the mean loss."""
        total, count = torch.zeros((), device=expert.device, dtype=expert.dtype), 0
        for _ in range(self.d_steps):
            for index in _minibatches(len(generated), cfg.batch_size, gen):
                logits = disc(torch.cat([expert[index], generated[index]])).squeeze(-1)
                target = torch.full_like(logits, self.label_smoothing)
                target[: len(index)] = 1.0 - self.label_smoothing
                loss = F.binary_cross_entropy_with_logits(logits, target)
                _step(optimiser, loss, cfg.grad_clip)
                total, count = total + loss.detach(), count + 1
        return float(total) / max(count, 1)

    def _ppo(
        self, value: nn.Module, policy_opt: torch.optim.Optimizer, value_opt: torch.optim.Optimizer,
        x: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, advantage: torch.Tensor,
        returns: torch.Tensor, cfg: TrainConfig, gen: torch.Generator,
    ) -> None:
        """``ppo_epochs`` passes of the clipped surrogate (policy) and squared error (value) over the pairs."""
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        for _ in range(self.ppo_epochs):
            for index in _minibatches(len(action), cfg.batch_size, gen):
                mean = self.core(x[index]) * self.a_std
                dist = Normal(mean, self.std.expand_as(mean), validate_args=False)
                ratio = torch.exp(dist.log_prob(action[index]) - log_prob[index])
                surrogate = torch.minimum(
                    ratio * advantage[index], ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage[index]
                )
                _step(policy_opt, -surrogate.mean() - self.entropy * dist.entropy().mean(), cfg.grad_clip)
                _step(value_opt, F.mse_loss(value(x[index]), returns[index]), cfg.grad_clip)
