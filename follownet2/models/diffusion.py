"""Conditional denoising diffusion model of the next acceleration.

A DDPM (Ho, Jain, Abbeel 2020) with the cosine noise schedule of Nichol and Dhariwal (2021),
conditioned on a GRU encoding of the history and sampled with DDIM (Song, Meng, Ermon 2021).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from follownet2.models.dist import Empirical
from follownet2.models.network import NetworkModel, mlp
from follownet2.train.windows import WindowSet

CLIP = 5.0  # bound of the samples in standard deviations of the acceleration
MAX_ROWS = 1 << 16  # (history, noise) pairs per pass of the sampler; bounds the memory of long batches


def cosine_alpha_bar(n_steps: int, offset: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Signal fractions ``alpha_bar[t]``, ``t = 0 .. n_steps - 1``, of the cosine schedule (float64)."""
    t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((t + offset) / (1.0 + offset) * math.pi / 2.0) ** 2
    beta = (1.0 - f[1:] / f[:-1]).clamp(max=max_beta)
    return torch.cumprod(1.0 - beta, dim=0)


def normal_quantiles(n: int) -> torch.Tensor:
    """The ``n`` quantiles of the standard normal distribution at the probabilities ``(i + 0.5) / n`` (float64)."""
    return torch.special.ndtri((torch.arange(n, dtype=torch.float64) + 0.5) / n)


def step_embedding(n_steps: int, dim: int) -> torch.Tensor:
    """Sinusoidal embedding ``[n_steps, 2 * (dim // 2)]`` of the diffusion step (float64)."""
    half = dim // 2
    freq = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float64) / half)
    angle = torch.arange(n_steps, dtype=torch.float64).unsqueeze(-1) * freq
    return torch.cat([angle.sin(), angle.cos()], dim=-1)


class DiffusionCF(NetworkModel):
    """Conditional DDPM over the next acceleration, sampled with DDIM.

    The condition is the last hidden state of a GRU of ``hidden`` units over the last
    ``history`` normalised rows. The denoiser is an MLP (hidden layers ``denoiser``, SiLU) on
    the noisy acceleration, a sinusoidal embedding of the diffusion step and the condition,
    and predicts the noise. Accelerations are diffused in units of their standard deviation
    over ``n_steps`` steps of the cosine schedule; the loss is the squared error of the noise
    at a step drawn uniformly (the simple loss of Ho et al.).

    Sampling is deterministic DDIM (``eta = 0``) over the ``n_sample_steps`` steps
    ``i * n_steps // n_sample_steps`` (the uniform spacing of the implementation of Song et
    al.: with 10 of 50 steps the sampler starts at step 45), and the samples are clipped to
    ``[-5, 5]`` standard deviations at the end.

    * ``sample(hist, generator)``: one draw per history, with independent noise;
    * ``forward``: the mean of ``n_forward_samples`` draws whose initial noise is not random:
      the normal quantiles at the probabilities ``(i + 0.5) / n`` (symmetric, mean zero), the
      same for every history, so that the output is a function of the history alone,
      whatever the rest of the batch and the device;
    * ``predict_dist``: :class:`Empirical` of ``n_dist_samples`` draws from the normal
      quantiles in the same way (stratified draws).

    Differences from the papers: DDIM starts at step 45 of 50 from standard normal noise
    instead of the last step, whose signal fraction (1e-6) would multiply the error of the
    predicted noise by about 150 in the first update; the signal left at step 45 is ignored,
    which shrinks the samples slightly towards zero (by 3 % for a Gaussian of 0.3 standard
    deviations with the exact denoiser); the samples are clipped once at the end, not at every
    step; the accelerations are scaled but not centred; the point estimate and the predictive
    distribution start from fixed normal quantiles instead of random noise.
    """

    name = "diffusion"

    def __init__(
        self,
        history: int = 10,
        use_a_prev: bool = False,
        hidden: int = 64,
        denoiser: Sequence[int] = (128, 128),
        step_features: int = 32,
        n_steps: int = 50,
        n_sample_steps: int = 10,
        n_forward_samples: int = 8,
        n_dist_samples: int = 64,
    ) -> None:
        super().__init__(history, use_a_prev)
        if not 1 <= n_sample_steps <= n_steps:
            raise ValueError(f"n_sample_steps must lie in [1, n_steps = {n_steps}], got {n_sample_steps}")
        denoiser = [int(n) for n in denoiser]
        self.n_steps, self.n_sample_steps = int(n_steps), int(n_sample_steps)
        self.n_forward_samples, self.n_dist_samples = int(n_forward_samples), int(n_dist_samples)
        self.kwargs.update(
            hidden=int(hidden), denoiser=denoiser, step_features=int(step_features), n_steps=self.n_steps,
            n_sample_steps=self.n_sample_steps, n_forward_samples=self.n_forward_samples,
            n_dist_samples=self.n_dist_samples,
        )
        table = step_embedding(self.n_steps, int(step_features))
        self.encoder = nn.GRU(self.n_inputs, int(hidden), batch_first=True)
        self.denoiser = mlp([1 + table.shape[-1] + int(hidden), *denoiser, 1], activation=nn.SiLU)
        # functions of the constructor arguments: rebuilt, not stored in checkpoints
        self.register_buffer("alpha_bar", cosine_alpha_bar(self.n_steps).float(), persistent=False)
        self.register_buffer("step_table", table.float(), persistent=False)
        #: diffusion steps visited by the sampler, in sampling order
        self.sample_steps = [i * self.n_steps // self.n_sample_steps for i in range(self.n_sample_steps)][::-1]

    # ------------------------------------------------------------------ model
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Condition ``[B, hidden]`` of normalised inputs ``[B, H, C]``."""
        return self.encoder(x)[1][-1]

    def denoise(self, noisy: torch.Tensor, step: torch.Tensor | int, condition: torch.Tensor) -> torch.Tensor:
        """Predicted noise ``[N]`` of the noisy normalised accelerations ``[N]`` at the diffusion steps ``step``."""
        embedded = self.step_table[step].expand(noisy.shape[0], -1)
        return self.denoiser(torch.cat([noisy.unsqueeze(-1), embedded, condition], dim=-1)).squeeze(-1)

    def ddim(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Normalised accelerations ``[N]`` from the initial noise ``[N]`` (DDIM, ``eta = 0``)."""
        x = noise
        for i, step in enumerate(self.sample_steps):
            ab = self.alpha_bar[step]
            last = i + 1 == len(self.sample_steps)
            ab_next = torch.ones_like(ab) if last else self.alpha_bar[self.sample_steps[i + 1]]
            eps = self.denoise(x, step, condition)
            x0 = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()
            x = ab_next.sqrt() * x0 + (1.0 - ab_next).sqrt() * eps
        return x.clamp(-CLIP, CLIP)

    def _draws(self, hist: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Accelerations ``[..., n]`` [m/s^2] of histories ``[..., H, 4]``, in the dtype of the model.

        ``noise`` is the initial noise, ``[B, n]`` for the ``B`` histories of the flattened
        batch or ``[n]`` shared by all of them.
        """
        x = self.inputs(hist.to(self.x_mean.dtype))
        noise = noise.to(x).expand(x.shape[0], -1)
        n = noise.shape[-1]
        rows = max(1, MAX_ROWS // n)
        out = [
            self.ddim(z.reshape(-1), self.encode(part).repeat_interleave(n, dim=0)).reshape(-1, n)
            for part, z in zip(x.split(rows), noise.split(rows))
        ]
        return (torch.cat(out) * self.a_std).reshape(*hist.shape[:-2], n)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self._draws(hist, normal_quantiles(self.n_forward_samples)).mean(-1).to(hist.dtype)

    def sample(self, hist: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """One draw of the acceleration [m/s^2] per history, independent noise per history."""
        device = generator.device if generator is not None else self.x_mean.device
        noise = torch.randn(hist.shape[:-2].numel(), 1, generator=generator, device=device, dtype=self.x_mean.dtype)
        return self._draws(hist, noise).squeeze(-1).to(hist.dtype)

    def predict_dist(self, hist: torch.Tensor) -> Empirical:
        return Empirical(self._draws(hist, normal_quantiles(self.n_dist_samples)).to(hist.dtype))

    # --------------------------------------------------------------- training
    def training_loss(self, windows: WindowSet, index: torch.Tensor) -> torch.Tensor:
        condition = self.encode(self.inputs(windows.history(index, self.history)))
        x0 = windows.target(index)[:, 0].to(condition.dtype) / self.a_std
        step = torch.randint(0, self.n_steps, x0.shape, device=x0.device)
        noise = torch.randn_like(x0)
        ab = self.alpha_bar[step]
        noisy = ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
        return torch.mean((self.denoise(noisy, step, condition) - noise) ** 2)
