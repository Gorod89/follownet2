"""Reference neural car-following models: MLP, GRU, LSTM, sequence to sequence, mixture density LSTM.

All of them map the normalised history rows ``(s, dv, v)`` (plus ``a_prev`` with
``use_a_prev=True``) to the acceleration over the next step and are trained by
:func:`follownet2.train.trainer.fit_supervised`. The point models minimise the one-step
squared error, :class:`Seq2Seq` the squared error of a sequence of accelerations and
:class:`MDNLSTM` the negative log-likelihood of a Gaussian mixture. Early stopping uses the
one-step error of ``forward`` for all of them, so that the selected epochs are comparable.

The recurrent layers start from a zero state at every call: the output for one history does
not depend on the other histories of the batch or on earlier calls.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, MixtureSameFamily, Normal

from follownet2.models.network import NetworkModel, mlp
from follownet2.train.windows import WindowSet

#: lower bound of the standard deviations of the mixture components, in normalised units
SIGMA_FLOOR = 1e-3


class MLP(NetworkModel):
    """Multilayer perceptron on the flattened last ``history`` rows.

    ``a = N(rows k - history + 1 .. k)``, hidden layers ``[64, 64]`` with tanh and one linear
    output: the feed-forward neural car-following model of the literature, given a short
    memory instead of the current state only.

    Differences from the papers: the sizes are the benchmark's instead of being tuned on the
    papers' data, and the output is the acceleration over the next step, the quantity every
    model of the benchmark predicts (papers of this family also predict the speed or the gap).
    """

    name = "mlp"

    def __init__(self, history: int = 10, use_a_prev: bool = False, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__(history, use_a_prev)
        self.kwargs["hidden"] = [int(n) for n in hidden]
        self.net = mlp([self.history * self.n_inputs, *self.kwargs["hidden"], 1])

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)


class RecurrentModel(NetworkModel):
    """One recurrent layer over the history rows and a linear head on its last hidden state."""

    layer: type[nn.RNNBase]

    def __init__(self, history: int = 10, use_a_prev: bool = False, hidden: int = 64) -> None:
        super().__init__(history, use_a_prev)
        self.kwargs["hidden"] = int(hidden)
        self.rnn = self.layer(self.n_inputs, int(hidden), batch_first=True)
        self.head = nn.Linear(int(hidden), 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Last hidden state ``[B, hidden]``; the layer starts from a zero state at every call."""
        return self.rnn(x)[0][:, -1]

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


class GRU(RecurrentModel):
    """GRU car-following model: one GRU layer of 64 units over the last 10 rows, linear head.

    The recurrent model with a memory of the recent states, as in Wang et al. (2018).

    Differences from the paper: the sizes are the benchmark's instead of being tuned on the
    paper's data, and the output is the acceleration over the next step, the quantity every
    model of the benchmark predicts.
    """

    name = "gru"
    layer = nn.GRU


class LSTM(RecurrentModel):
    """LSTM car-following model: one LSTM layer of 64 units over the last 10 rows, linear head.

    The recurrent model of Huang et al. (2018) and of the later LSTM car-following studies.

    Differences from the paper: the sizes are the benchmark's instead of being tuned on the
    paper's data, and the output is the acceleration over the next step, the quantity every
    model of the benchmark predicts.
    """

    name = "lstm"
    layer = nn.LSTM


class Seq2Seq(NetworkModel):
    """Sequence to sequence model with a reaction delay (Ma and Qu 2020).

    The encoder LSTM reads the ``history_rows`` rows that end ``delay`` samples before now,
    rows ``[-(history_rows + delay) : -delay]`` of the history (``history = history_rows +
    delay``); the last ``delay`` rows, the reaction time, are not used. The decoder LSTM
    starts from the final encoder state and a zero input and produces the accelerations
    ``a[k], ..., a[k + out_steps - 1]``, each step fed with the output of the step before.
    ``forward`` returns the first one (receding horizon), :meth:`predict_sequence` all of them.
    Loss: squared error over all outputs, which needs windows ``train_horizon = out_steps``
    steps long.

    Training uses scheduled teacher forcing at a constant rate: the input of every decoder
    step after the first is, independently per window and step, the recorded previous
    acceleration with probability ``teacher_forcing`` and the decoder's own previous output
    otherwise (gradients flow through the fed-back outputs). The draws come from a generator
    seeded with ``cfg.seed`` in :meth:`fit`. At evaluation the decoder is always fed with its
    own output.

    Differences from the paper: the sizes (one layer of ``hidden`` units in the encoder and
    the decoder, 10 input rows, 10 outputs) and the delay of 0.5 s are the benchmark's
    defaults, not tuned per dataset; the first decoder input is zero, so that no acceleration
    channel is read unless ``use_a_prev=True``; the scheduled teacher forcing above is a
    choice of this implementation.
    """

    name = "seq2seq"

    def __init__(
        self,
        history_rows: int = 10,
        delay: int = 5,
        use_a_prev: bool = False,
        hidden: int = 64,
        out_steps: int = 10,
        teacher_forcing: float = 0.5,
    ) -> None:
        if history_rows < 1 or delay < 0 or out_steps < 1:
            raise ValueError("seq2seq needs history_rows >= 1, delay >= 0 and out_steps >= 1")
        super().__init__(int(history_rows) + int(delay), use_a_prev)
        self.history_rows, self.delay, self.out_steps = int(history_rows), int(delay), int(out_steps)
        self.teacher_forcing = float(teacher_forcing)
        self.train_horizon = self.out_steps
        self.kwargs = {
            "history_rows": self.history_rows,
            "delay": self.delay,
            "use_a_prev": bool(use_a_prev),
            "hidden": int(hidden),
            "out_steps": self.out_steps,
            "teacher_forcing": self.teacher_forcing,
        }
        self.encoder = nn.LSTM(self.n_inputs, int(hidden), batch_first=True)
        self.decoder = nn.LSTMCell(1, int(hidden))
        self.head = nn.Linear(int(hidden), 1)
        self._teacher: torch.Generator | None = None

    def decode(
        self,
        x: torch.Tensor,
        steps: int,
        recorded: torch.Tensor | None = None,
        forced: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalised accelerations ``[B, steps]`` from normalised inputs ``[B, history, C]``.

        Where ``forced[:, j]`` is true, step ``j + 1`` is fed with ``recorded[:, j]`` instead
        of the output of step ``j`` (teacher forcing); both are ``[B, steps]``.
        """
        _, (h, c) = self.encoder(x[:, : self.history_rows])
        h, c = h[0], c[0]
        previous = x.new_zeros(x.shape[0], 1)
        out = []
        for j in range(steps):
            h, c = self.decoder(previous, (h, c))
            out.append(self.head(h))
            previous = out[-1]
            if forced is not None:
                previous = torch.where(forced[:, j : j + 1], recorded[:, j : j + 1], previous)
        return torch.cat(out, dim=1)

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x, 1)[:, 0]

    def predict_sequence(self, hist: torch.Tensor) -> torch.Tensor:
        """Accelerations ``[..., out_steps]`` in m/s^2 over the next ``out_steps`` steps."""
        out = self.decode(self.inputs(hist.to(self.x_mean.dtype)), self.out_steps) * self.a_std
        return out.reshape(*hist.shape[:-2], self.out_steps).to(hist.dtype)

    def training_loss(self, windows: WindowSet, index: torch.Tensor) -> torch.Tensor:
        target = windows.target(index, self.out_steps) / self.a_std
        # drawn on the CPU, so that the draws do not depend on the device of the windows
        forced = (torch.rand(target.shape, generator=self._teacher) < self.teacher_forcing).to(target.device)
        predicted = self.decode(self.inputs(windows.history(index)), self.out_steps, target, forced)
        return torch.mean((predicted - target) ** 2)

    def fit(self, train: WindowSet, val: WindowSet, cfg) -> dict:
        """As :meth:`NetworkModel.fit`; the teacher-forcing draws come from ``cfg.seed``."""
        self._teacher = torch.Generator().manual_seed(cfg.seed)
        try:
            return super().fit(train, val, cfg)
        finally:
            self._teacher = None


class MDNLSTM(LSTM):
    """Mixture density network (Bishop 1994) on the encoder of :class:`LSTM`.

    The head maps the last hidden state to ``n_components`` Gaussians over the normalised
    acceleration: mixture logits, means and standard deviations ``softplus(.) + 1e-3``.
    Loss: negative log-likelihood of the recorded acceleration. ``forward`` is the mean of
    the mixture, :meth:`predict_dist` the mixture in m/s^2 and :meth:`sample` one draw from
    it per history.

    Differences from the paper: Bishop parameterises the standard deviations by an
    exponential; the softplus with a floor keeps the likelihood bounded on noise-free data,
    where a component would otherwise collapse onto the targets.
    """

    name = "mdn_lstm"

    def __init__(self, history: int = 10, use_a_prev: bool = False, hidden: int = 64, n_components: int = 5) -> None:
        super().__init__(history, use_a_prev, hidden)
        self.kwargs["n_components"] = int(n_components)
        self.head = nn.Linear(int(hidden), 3 * int(n_components))  # replaces the point head of LSTM

    def mixture(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Logits, means and standard deviations ``[B, n_components]`` in normalised units."""
        logits, mean, raw = self.head(self.encode(x)).chunk(3, dim=-1)
        return logits, mean, F.softplus(raw) + SIGMA_FLOOR

    def core(self, x: torch.Tensor) -> torch.Tensor:
        logits, mean, _ = self.mixture(x)
        return torch.sum(torch.softmax(logits, dim=-1) * mean, dim=-1)

    def training_loss(self, windows: WindowSet, index: torch.Tensor) -> torch.Tensor:
        logits, mean, std = self.mixture(self.inputs(windows.history(index)))
        target = windows.target(index)[:, 0] / self.a_std
        return -torch.mean(gaussian_mixture(logits, mean, std).log_prob(target))

    def predict_dist(self, hist: torch.Tensor) -> MixtureSameFamily:
        """Predictive mixture over the acceleration in m/s^2, batch shape ``hist.shape[:-2]``."""
        logits, mean, std = self.mixture(self.inputs(hist.to(self.x_mean.dtype)))
        shape = (*hist.shape[:-2], logits.shape[-1])
        logits, mean, std = (t.reshape(shape).to(hist.dtype) for t in (logits, mean * self.a_std, std * self.a_std))
        return gaussian_mixture(logits, mean, std)

    @torch.no_grad()
    def sample(self, hist: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """One acceleration [m/s^2] per history drawn from :meth:`predict_dist`, shape ``hist.shape[:-2]``.

        The uniform and normal numbers come from ``generator`` on its own device, so the draws
        do not depend on the device of the model.
        """
        logits, mean, std = self.mixture(self.inputs(hist.to(self.x_mean.dtype)))
        u = torch.rand(len(logits), 1, generator=generator, device=generator.device).to(logits)
        z = torch.randn(len(logits), 1, generator=generator, device=generator.device).to(logits)
        cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1)
        component = (u > cdf).sum(-1, keepdim=True).clamp(max=logits.shape[-1] - 1)
        draw = (mean.gather(-1, component) + std.gather(-1, component) * z) * self.a_std
        return draw.reshape(hist.shape[:-2]).to(hist.dtype)


def gaussian_mixture(logits: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> MixtureSameFamily:
    """Mixture of univariate Gaussians over the last dimension of the parameters.

    The parameters are valid by construction and are not checked: a check synchronises the
    GPU at every call and would turn a diverged model into an exception instead of the
    non-finite outputs that the protocols count.
    """
    return MixtureSameFamily(
        Categorical(logits=logits, validate_args=False), Normal(mean, std, validate_args=False), validate_args=False
    )
