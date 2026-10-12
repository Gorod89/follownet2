"""Batched differential evolution in torch.

Solves ``N`` independent box-constrained problems at once, which is what the per-event
calibration needs (one problem per event, all candidates of all problems evaluated in one
batched rollout). Strategy and defaults follow ``scipy.optimize.differential_evolution``:
best1bin, population ``popsize * D``, dithered mutation U(0.5, 1), recombination 0.7, Latin
hypercube initialisation, convergence when ``std(f) <= atol + tol * |mean(f)|``. Differences:
the population is updated once per generation (SciPy's ``updating='deferred'``) and there is
no final polishing step. SciPy is the reference in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class DEResult:
    x: torch.Tensor  # [N, D] best parameter vector of every problem
    fun: torch.Tensor  # [N]
    n_generations: torch.Tensor  # [N] generation at which the problem converged (or maxiter)
    converged: torch.Tensor  # [N] bool
    population: torch.Tensor  # [N, P, D]
    energies: torch.Tensor  # [N, P]


def latin_hypercube(n_problems: int, n_pop: int, n_dim: int, generator: torch.Generator, **kw) -> torch.Tensor:
    """Points in the unit cube, one per stratum in every dimension, shape ``[N, P, D]``."""
    device = kw.get("device")
    jitter = torch.rand(n_problems, n_pop, n_dim, generator=generator, **kw)
    order = torch.rand(n_problems, n_pop, n_dim, generator=generator, **kw).argsort(dim=1)
    return (order.to(jitter.dtype) + jitter).to(device) / n_pop


@torch.no_grad()
def differential_evolution(
    objective: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    n_problems: int = 1,
    popsize: int = 15,
    maxiter: int = 200,
    tol: float = 0.01,
    atol: float = 0.0,
    mutation: tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int = 0,
    x0: torch.Tensor | None = None,
    callback: Callable[[int, torch.Tensor], None] | None = None,
) -> DEResult:
    """Minimise ``objective(theta[N, P, D]) -> f[N, P]`` inside ``bounds[D, 2]``.

    ``x0`` (shape ``[D]`` or ``[N, D]``) replaces the first member of the initial population,
    e.g. by literature parameters.
    """
    device, dtype = bounds.device, bounds.dtype
    n_dim = bounds.shape[0]
    n_pop = max(popsize * n_dim, 5)
    low, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    kw = {"device": device, "dtype": dtype}

    unit = latin_hypercube(n_problems, n_pop, n_dim, gen, **kw)
    if x0 is not None:
        unit[:, 0, :] = ((x0.to(**kw) - low) / span).clamp(0.0, 1.0).expand(n_problems, n_dim)
    energies = objective(low + unit * span)
    energies = torch.nan_to_num(energies, nan=float("inf"))

    members = torch.arange(n_pop, device=device).expand(n_problems, n_pop)
    rows = torch.arange(n_problems, device=device).unsqueeze(-1)
    converged = torch.zeros(n_problems, dtype=torch.bool, device=device)
    n_generations = torch.full((n_problems,), maxiter, dtype=torch.long, device=device)

    for generation in range(1, maxiter + 1):
        best = unit[rows.squeeze(-1), energies.argmin(dim=1)]  # [N, D]
        r1 = torch.randint(0, n_pop - 1, (n_problems, n_pop), generator=gen, device=device)
        r1 = r1 + (r1 >= members)
        r2 = torch.randint(0, n_pop - 2, (n_problems, n_pop), generator=gen, device=device)
        lo, hi = torch.minimum(members, r1), torch.maximum(members, r1)
        r2 = r2 + (r2 >= lo)
        r2 = r2 + (r2 >= hi)
        scale = mutation[0] + (mutation[1] - mutation[0]) * torch.rand(n_problems, 1, 1, generator=gen, **kw)
        mutant = best.unsqueeze(1) + scale * (unit[rows, r1] - unit[rows, r2])
        outside = (mutant < 0.0) | (mutant > 1.0)
        mutant = torch.where(outside, torch.rand(mutant.shape, generator=gen, **kw), mutant)
        cross = torch.rand(n_problems, n_pop, n_dim, generator=gen, **kw) < recombination
        forced = torch.randint(0, n_dim, (n_problems, n_pop, 1), generator=gen, device=device)
        cross.scatter_(2, forced, True)
        trial = torch.where(cross, mutant, unit)

        trial_energies = torch.nan_to_num(objective(low + trial * span), nan=float("inf"))
        accept = (trial_energies <= energies) & ~converged.unsqueeze(-1)
        unit = torch.where(accept.unsqueeze(-1), trial, unit)
        energies = torch.where(accept, trial_energies, energies)

        finite = torch.isfinite(energies).all(dim=1)
        spread = torch.where(finite, energies.std(dim=1, correction=0), torch.full_like(energies[:, 0], float("inf")))
        mean = torch.where(finite, energies.mean(dim=1).abs(), torch.zeros_like(spread))
        done = (spread <= atol + tol * mean) & ~converged
        n_generations = torch.where(done, torch.full_like(n_generations, generation), n_generations)
        converged |= done
        if callback is not None:
            callback(generation, energies)
        if bool(converged.all()):
            break

    arg = energies.argmin(dim=1)
    flat = rows.squeeze(-1)
    return DEResult(
        x=low + unit[flat, arg] * span,
        fun=energies[flat, arg],
        n_generations=n_generations,
        converged=converged,
        population=low + unit * span,
        energies=energies,
    )
