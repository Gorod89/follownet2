"""Empirical distribution of samples: the predictive distribution of the sampling models.

It offers the part of the ``torch.distributions`` interface that the protocols use (``mean``,
``stddev``, ``batch_shape``, ``sample``, ``cdf``, ``icdf``, ``log_prob``) and the continuous
ranked probability score, which an empirical distribution has in closed form.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

BANDWIDTH_FLOOR = 1e-3


class Empirical:
    """Distribution given by ``samples [..., K]``; the last dimension indexes the samples.

    * ``mean`` and ``stddev`` are those of the empirical measure (the variance divides by ``K``);
    * ``cdf`` is the empirical distribution function ``P(X <= x)``;
    * ``icdf`` interpolates linearly between the order statistics (position ``q (K - 1)``, the
      default of ``numpy.quantile``);
    * ``log_prob`` is a Gaussian kernel density estimate with the bandwidth of Silverman's rule
      of thumb, ``0.9 min(sd, IQR / 1.34) K^(-1/5)`` per batch element (``sd`` alone when the
      interquartile range is zero), floored at ``1e-3``;
    * ``crps`` is the exact continuous ranked probability score of the empirical measure,
      ``E|X - x| - E|X - X'| / 2`` with ``X, X'`` drawn independently from the samples.
    """

    def __init__(self, samples: torch.Tensor) -> None:
        if samples.dim() < 1 or samples.shape[-1] < 1:
            raise ValueError("an empirical distribution needs a last dimension with at least one sample")
        self.samples = samples
        self._sorted: torch.Tensor | None = None

    @property
    def batch_shape(self) -> torch.Size:
        return self.samples.shape[:-1]

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[-1])

    @property
    def mean(self) -> torch.Tensor:
        return self.samples.mean(-1)

    @property
    def stddev(self) -> torch.Tensor:
        return self.samples.std(-1, correction=0)

    @property
    def sorted(self) -> torch.Tensor:
        """The samples in ascending order along the last dimension."""
        if self._sorted is None:
            self._sorted = self.samples.sort(-1).values
        return self._sorted

    def _value(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=self.samples.dtype, device=self.samples.device)

    def sample(self, generator: torch.Generator | None = None, sample_shape: Sequence[int] = ()) -> torch.Tensor:
        """One of the stored samples per batch element, drawn uniformly; shape ``sample_shape + batch_shape``."""
        shape = torch.Size(sample_shape) + self.batch_shape
        device = generator.device if generator is not None else self.samples.device
        index = torch.randint(self.n_samples, shape, generator=generator, device=device).to(self.samples.device)
        return self.samples.expand(*shape, self.n_samples).gather(-1, index.unsqueeze(-1)).squeeze(-1)

    def cdf(self, x) -> torch.Tensor:
        x = self._value(x)
        return (self.samples <= x.unsqueeze(-1)).to(self.samples.dtype).mean(-1)

    def icdf(self, q) -> torch.Tensor:
        q = self._value(q)
        shape = torch.broadcast_shapes(q.shape, self.batch_shape)
        position = (q.expand(shape) * (self.n_samples - 1)).clamp(0.0, self.n_samples - 1.0)
        lower = position.floor()
        index = lower.long().unsqueeze(-1)
        ordered = self.sorted.expand(*shape, self.n_samples)
        left = ordered.gather(-1, index).squeeze(-1)
        right = ordered.gather(-1, (index + 1).clamp(max=self.n_samples - 1)).squeeze(-1)
        return left + (position - lower) * (right - left)

    @property
    def bandwidth(self) -> torch.Tensor:
        """Kernel bandwidth of :meth:`log_prob` per batch element (Silverman's rule of thumb)."""
        sd = self.stddev
        iqr = (self.icdf(0.75) - self.icdf(0.25)) / 1.34
        spread = torch.where(iqr > 0, torch.minimum(sd, iqr), sd)
        return (0.9 * spread * self.n_samples ** (-0.2)).clamp_min(BANDWIDTH_FLOOR)

    def log_prob(self, x) -> torch.Tensor:
        x = self._value(x)
        h = self.bandwidth
        z = (x.unsqueeze(-1) - self.samples) / h.unsqueeze(-1)
        kernel = torch.logsumexp(-0.5 * z * z, dim=-1)
        return kernel - torch.log(h) - 0.5 * math.log(2.0 * math.pi) - math.log(self.n_samples)

    def crps(self, x) -> torch.Tensor:
        x = self._value(x)
        k = self.n_samples
        # sum over all ordered pairs of |x_i - x_j| from the order statistics: 2 sum_i (2i - K + 1) x_(i)
        rank = torch.arange(k, dtype=self.samples.dtype, device=self.samples.device)
        pair_mean = 2.0 * (self.sorted * (2.0 * rank - k + 1.0)).sum(-1) / k**2
        return (self.samples - x.unsqueeze(-1)).abs().mean(-1) - 0.5 * pair_mean
