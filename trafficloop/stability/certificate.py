"""A priori string-stability certificate of the certified residual IDM (port of cf-stability, decision D86).

The model (:class:`~trafficloop.models.certified.CertifiedResidualIDM`) is ``f = f_idm + r`` with a
frozen IDM core and a residual ``|r| <= r_max`` whose derivatives are bounded,
``|dr/dx_j| <= B_j`` for ``x = (s, dv, v)`` (``model.jacobian_bound()``). At an equilibrium the
partial derivatives of the hybrid are those of the core plus ``d_j`` with ``|d_j| <= B_j``; the
minimum of the string-stability criterion ``m = f_v^2 + 2 f_v f_dv - 2 f_s`` (``dv = v - v_lead``,
:mod:`trafficloop.stability.analytic`) over that box is a lower bound of the margin of the hybrid
(the guaranteed margin), and the box gives guaranteed signs of ``f_s`` and ``f_v + f_dv``.

A priori form (decisions D71 and D86 of cf-stability, ``cf_stability/stability/certificate.py``):
an equilibrium of the hybrid at the speed ``v`` lies where ``|f_idm(s, 0, v)| <= r_max`` (the
residual must cancel the core there), an interval ``I(v)`` of spacings because ``f_idm`` rises
with ``s``. The certificate holds at ``v`` when the guaranteed margin is ``>= 0`` and both signs
are guaranteed at every spacing of ``I(v)``; it depends on the weights of the residual only
through the product of the spectral norms of its layers, which spectral normalisation keeps at
1, so it survives training. :func:`enforce_budget` sets ``lipschitz = safety * b / r_max`` with
``b`` the largest ``r_max * lipschitz`` for which the certificate holds at every speed.

The mathematics is that of cf-stability: closed-form guaranteed margin, interval ends in closed
form, minimum over ``I(v)`` by a scan of ``n_scan`` spacings refined ``N_ZOOM`` times, budget by
doubling and bisection. Differences of the port: (1) the spacings are clamped to
``[S_MIN, S_MAX] = [0.5, 300] m``, the range in which this benchmark looks for equilibria
(:func:`~trafficloop.stability.analytic.equilibrium_gap`, and P3 counts a gap below 300 m), instead of
``[1, 200] m`` of the equilibrium search of cf-stability, so that the certificate covers every
equilibrium the protocols can find; (2) the core is a parameter vector of the benchmark IDM
(:func:`~trafficloop.models.mechanistic.idm_acc`, ``delta = 4``), whose derivatives
:func:`core_partials` takes by automatic differentiation as cf-stability does; (3) only the a priori
form is ported (the form at the equilibria and the spacing band of the training data are not
used here). Everything runs in float64 on a CPU copy of the model in evaluation mode.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from trafficloop.models.mechanistic import IDM
from trafficloop.stability.analytic import SPEEDS, idm_partials, partials, string_margin

if TYPE_CHECKING:
    from trafficloop.models.certified import CertifiedResidualIDM

#: spacings that can hold an equilibrium [m] (cf-stability: 1 and 200, see the module docstring)
S_MIN, S_MAX = 0.5, 300.0
#: rescans of the minimum around the best spacing, two grid cells wide each
N_ZOOM = 3
#: spacings per speed of the first scan
N_SCAN = 400
#: halvings of the budget interval
N_BISECT = 60
#: exponent of the IDM of the benchmark
DELTA = 4.0
#: weight of the shortfall of the margin in the objective of a core with a stability margin (D72)
STABILITY_WEIGHT = 10.0
#: margin counted at a speed without equilibrium of the core (D72)
NO_EQUILIBRIUM = -10.0

Speeds = Sequence[float] | Tensor


def _speeds(v: Speeds) -> Tensor:
    return torch.as_tensor(v, dtype=torch.float64).reshape(-1).cpu()


def _core(theta: Sequence[float] | Tensor) -> Tensor:
    return torch.as_tensor(theta, dtype=torch.float64).reshape(-1).cpu()


def _float64(model: "CertifiedResidualIDM") -> "CertifiedResidualIDM":
    return copy.deepcopy(model).cpu().double().eval()


# ----------------------------------------------------------------- box of the derivatives
def guaranteed_margin(f_s: Tensor, f_dv: Tensor, f_v: Tensor, bounds: Tensor) -> Tensor:
    """Exact minimum of ``(f_v + d_v)^2 + 2 (f_v + d_v)(f_dv + d_dv) - 2 (f_s + d_s)`` over ``|d_j| <= B_j``.

    ``bounds [..., 3]`` (order ``s, dv, v``) broadcasts against the derivatives. ``d_s = B_s``;
    ``p^2 + 2 p q`` (``p = f_v + d_v``, ``q = f_dv + d_dv``) is linear in ``q``, so its minimum lies on
    an edge of ``q``, and convex in ``p``, so there it lies at ``p = -q`` clamped to the interval of ``p``.
    """
    b_s, b_dv, b_v = bounds[..., 0], bounds[..., 1], bounds[..., 2]
    p_lo, p_hi = f_v - b_v, f_v + b_v
    edges = []
    for q in (f_dv - b_dv, f_dv + b_dv):
        p = torch.minimum(torch.maximum(-q, p_lo), p_hi)
        edges.append(p * p + 2.0 * p * q)
    return torch.minimum(*edges) - 2.0 * (f_s + b_s)


def guarantees(f_s: Tensor, f_dv: Tensor, f_v: Tensor, bounds: Tensor) -> dict[str, Tensor]:
    """Guaranteed margin and signs over the box of ``bounds``; NaN derivatives never hold."""
    margin = guaranteed_margin(f_s, f_dv, f_v, bounds)
    fs_positive = f_s - bounds[..., 0] > 0
    sum_negative = f_v + f_dv + bounds[..., 1] + bounds[..., 2] < 0
    return {
        "guaranteed_margin": margin,
        "fs_positive": fs_positive,
        "sum_negative": sum_negative,
        "holds": (margin >= 0) & fs_positive & sum_negative,
    }


# ---------------------------------------------------------------------- the IDM core
def core_partials(theta: Sequence[float] | Tensor, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(f_s, f_dv, f_v)`` of the benchmark IDM with the parameters ``theta = (v0, T, s0, a, b)`` at the
    states ``(s, 0, v)``, by automatic differentiation in float64 (:func:`~trafficloop.stability.analytic.partials`)."""
    return partials(IDM(_core(theta)), s, 0.0, v)


def core_margin(theta: Sequence[float] | Tensor, v: Speeds = SPEEDS) -> Tensor:
    """Closed-form margin ``[n]`` of the core at its own equilibria, NaN for ``v >= v0`` (no equilibrium)."""
    return string_margin(*idm_partials(_speeds(v), _core(theta)))


def stability_shortfall(
    params: Tensor, margin: float, speeds: Speeds = SPEEDS, weight: float = STABILITY_WEIGHT
) -> Tensor:
    """``weight * sum_v relu(margin - M(v))`` for candidate IDM parameters ``[P, 5]``, shape ``[P]``.

    The term of decision D72 of cf-stability that a calibration adds to its objective to obtain a
    core with a string-stability margin: ``M(v)`` is the closed-form margin of the IDM at its
    equilibrium of the speed ``v`` (:func:`~trafficloop.stability.analytic.idm_partials`), counted as
    ``NO_EQUILIBRIUM`` where ``v >= v0``. Computed in float64, returned in the dtype of ``params``.
    """
    v = torch.as_tensor(speeds, dtype=torch.float64, device=params.device).reshape(-1)
    m = string_margin(*idm_partials(v, params.double()[:, None, :]))
    m = torch.nan_to_num(m, nan=NO_EQUILIBRIUM).clamp(min=NO_EQUILIBRIUM)
    return (weight * torch.clamp(float(margin) - m, min=0.0).sum(-1)).to(params.dtype)


def feasible_ends(theta: Sequence[float] | Tensor, v: Tensor, r_max: Tensor | float) -> dict[str, Tensor]:
    """Ends of ``I(v) = {s > 0 : |f_idm(s, 0, v)| <= r_max}`` before the clamping to ``[S_MIN, S_MAX]``, and
    whether an equilibrium of the hybrid must exist in it. All entries ``[n]``.

    ``f_idm(s, 0, v) = a (1 - (v/v0)^delta) - a (s*/s)^2`` with ``s* = s0 + v T`` rises with ``s`` towards
    ``a (1 - (v/v0)^delta)``; ``(s*/s)^2 = 1 - (v/v0)^delta -+ r_max / a`` gives ``s_low`` and ``s_high``, ``inf``
    where the right side is not positive (``s_low``: no feasible spacing; ``s_high``: no upper end).
    ``bounded`` (``s_high`` finite): every residual with ``|r| <= r_max`` gives the hybrid an equilibrium in
    ``I(v)`` (intermediate values); otherwise none need exist. ``within``: ``S_MIN <= s_low`` and
    ``s_high <= S_MAX``, so the clamped interval of the certificate is all of ``I(v)``.
    """
    v0, T, s0, a, _ = _core(theta).unbind()
    s_star = s0 + v * T
    free = 1.0 - (v / v0) ** DELTA
    low_arg, high_arg = free + r_max / a, free - r_max / a
    s_low = torch.where(low_arg > 0, s_star / torch.sqrt(low_arg.clamp(min=1e-300)), torch.inf)
    s_high = torch.where(high_arg > 0, s_star / torch.sqrt(high_arg.clamp(min=1e-300)), torch.inf)
    return {
        "s_low": s_low,
        "s_high": s_high,
        "limit_margin": a * free - r_max,
        "bounded": high_arg > 0,
        "within": (s_low >= S_MIN) & (s_high <= S_MAX),
    }


def _feasible_interval(theta: Tensor, v: Tensor, r_max: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Spacings in ``[S_MIN, S_MAX]`` with ``|f_idm(s, 0, v)| <= r_max``: the ends of :func:`feasible_ends`
    clamped, and whether the interval is not empty."""
    ends = feasible_ends(theta, v, r_max)
    s_low, s_high = ends["s_low"].clamp(min=S_MIN), ends["s_high"].clamp(max=S_MAX)
    return s_low, s_high, s_low <= s_high


def a_priori(theta: Tensor, v: Tensor, r_max: Tensor, bounds: Tensor, n_scan: int = N_SCAN) -> dict[str, Tensor]:
    """Minimum over the feasible spacings of the guaranteed margin, and the signs at every one of them.

    ``theta [5]`` is the core, ``v``, ``r_max [n]`` and ``bounds [n, 3]`` float64 on the CPU. The spacings
    are scanned on ``n_scan`` points and the scan is repeated ``N_ZOOM`` times around the minimum; the
    ends of the interval are scan points, which makes the signs exact (the IDM derivatives are monotone
    in ``s``). ``holds`` is false where no spacing is feasible.
    """
    s_low, s_high, feasible = _feasible_interval(theta, v, r_max)
    lo, hi = torch.where(feasible, s_low, S_MIN), torch.where(feasible, s_high, S_MIN)
    frac = torch.linspace(0.0, 1.0, n_scan, dtype=torch.float64)
    n = len(v)
    rows = torch.arange(n)
    best, best_s = torch.full((n,), torch.inf, dtype=torch.float64), torch.full((n,), torch.nan, dtype=torch.float64)
    signs: dict[str, Tensor] = {}
    for scan in range(N_ZOOM + 1):
        s = lo[:, None] + (hi - lo)[:, None] * frac
        f = core_partials(theta, s.reshape(-1), v.repeat_interleave(n_scan))
        g = guarantees(*(x.reshape(n, n_scan) for x in f), bounds[:, None, :])
        value, k = g["guaranteed_margin"].min(dim=1)
        better = value < best
        best, best_s = torch.where(better, value, best), torch.where(better, s[rows, k], best_s)
        if scan == 0:
            signs = {name: g[name].all(dim=1) for name in ("fs_positive", "sum_negative")}
        step = (hi - lo) / (n_scan - 1)
        centre = s[rows, k]
        lo, hi = torch.maximum(centre - step, lo), torch.minimum(centre + step, hi)
    nan = torch.full_like(best, torch.nan)
    margin = torch.where(feasible, best, nan)
    fs_positive, sum_negative = signs["fs_positive"] & feasible, signs["sum_negative"] & feasible
    return {
        "s_low": torch.where(feasible, s_low, nan),
        "s_high": torch.where(feasible, s_high, nan),
        "feasible": feasible,
        "guaranteed_margin": margin,
        "s_worst": torch.where(feasible, best_s, nan),
        "fs_positive": fs_positive,
        "sum_negative": sum_negative,
        "holds": (margin >= 0) & fs_positive & sum_negative,
    }


def largest(holds_at: Callable[[Tensor], Tensor], n: int) -> Tensor:
    """Largest ``beta >= 0`` per entry with ``holds_at(beta)`` true (monotone), 0 where it fails at 0."""
    zero = torch.zeros(n, dtype=torch.float64)
    feasible = holds_at(zero)
    lo, hi = zero.clone(), torch.ones(n, dtype=torch.float64)
    for _ in range(64):  # doubling until the certificate fails everywhere
        grow = feasible & holds_at(hi)
        if not grow.any():
            break
        lo, hi = torch.where(grow, hi, lo), torch.where(grow, 2.0 * hi, hi)
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        ok = holds_at(mid)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    return torch.where(feasible, lo, zero)


# ----------------------------------------------------------------------- the model
def residual_bounds(model: "CertifiedResidualIDM") -> Tensor:
    """Bounds ``[3]`` of ``|dr/d(s, dv, v)|`` (``model.jacobian_bound()``, float64 on the CPU)."""
    return torch.as_tensor(model.jacobian_bound(), dtype=torch.float64).cpu()


def max_residual_budget(
    model: "CertifiedResidualIDM", v: Speeds = SPEEDS, *, n_scan: int = N_SCAN, keep_r_max: bool = False
) -> Tensor:
    """Largest ``r_max * lipschitz`` per speed ``[n]`` for which the a priori certificate would hold.

    The bounds scale with the product (the layer norms of the weights stay). The feasible spacings grow
    with ``r_max``: by default ``r_max`` and ``lipschitz`` scale together (the ratio of the model is kept);
    with ``keep_r_max`` the feasible spacings are those of the model's ``r_max`` and only ``lipschitz``
    scales, which is the budget of a model whose ``r_max`` is fixed (D86). A budget without any feasible
    spacing counts as holding (no equilibrium can exist there).
    """
    m, v = _float64(model), _speeds(v)
    theta = m.theta.detach().cpu()
    unit = math.prod(m.layer_norms()) / m.physical_scale.detach().cpu()  # bounds per unit of r_max * lipschitz
    r_max = m.r_max
    ratio = r_max / m.lipschitz if m.lipschitz > 0 else math.inf

    def holds_at(beta: Tensor) -> Tensor:
        amplitude = torch.full_like(beta, r_max) if keep_r_max else torch.sqrt(beta * ratio)
        out = a_priori(theta, v, amplitude, beta[:, None] * unit, n_scan)
        return out["holds"] | ~out["feasible"]

    return largest(holds_at, len(v))


def enforce_budget(
    model: "CertifiedResidualIDM", safety: float, v: Speeds = SPEEDS, n_scan: int = N_SCAN
) -> dict[str, Any]:
    """Certified budget of D86: ``b`` = the largest ``r_max * lipschitz`` with the model's ``r_max`` for
    which the a priori certificate holds at every speed of ``v``; sets ``lipschitz = safety * b / r_max``.

    Raises ``ValueError`` when ``b = 0``: the core is then not string stable (or its signs are not
    guaranteed) at some spacing that ``r_max`` admits as an equilibrium, and a core with a stability
    margin (:func:`stability_shortfall`) or a smaller ``r_max`` is needed.
    """
    if not 0.0 < safety <= 1.0:
        raise ValueError(f"safety must lie in (0, 1], got {safety}")
    speeds = _speeds(v)
    per_speed = max_residual_budget(model, speeds, n_scan=n_scan, keep_r_max=True)
    admissible = float(per_speed.min())
    if not admissible > 0.0:
        failing = [float(x) for x in speeds[per_speed <= 0.0]]
        core = {k: round(x, 4) for k, x in model.core_params().items()}
        raise ValueError(
            f"no admissible residual budget: with r_max = {model.r_max:g} the a priori certificate fails for every "
            f"lipschitz > 0 at v = {failing} m/s (IDM core {core}); a core with a stability margin or a smaller "
            f"r_max is needed"
        )
    model.lipschitz = safety * admissible / model.r_max
    return {"admissible": admissible, "safety": float(safety), "r_max": model.r_max, "lipschitz": model.lipschitz}


def certificate_a_priori(
    model: "CertifiedResidualIDM", v: Speeds = SPEEDS, n_scan: int = N_SCAN, *, keep_r_max: bool = False
) -> dict[str, Any]:
    """A priori certificate of ``model`` per speed: the entries of :func:`a_priori` (``holds``,
    ``guaranteed_margin``, ``s_low``, ``s_high``, ``s_worst``, ``feasible``, the signs), ``within`` of
    :func:`feasible_ends`, the residual (``bounds``, ``layer_norms``, ``r_max``, ``lipschitz``, their
    product ``budget``), the core, ``n_holds`` and ``max_budget`` (:func:`max_residual_budget`)."""
    m, v = _float64(model), _speeds(v)
    theta = m.theta.detach().cpu()
    r_max = torch.full_like(v, m.r_max)
    out = a_priori(theta, v, r_max, residual_bounds(m).expand(len(v), 3), n_scan)
    return {
        "v": v,
        "bounds": residual_bounds(m),
        "layer_norms": m.layer_norms(),
        "r_max": m.r_max,
        "lipschitz": m.lipschitz,
        "budget": m.r_max * m.lipschitz,
        "idm_params": m.core_params(),
        **out,
        "within": feasible_ends(theta, v, r_max)["within"],
        "n_holds": int(out["holds"].sum()),
        "max_budget": max_residual_budget(m, v, n_scan=n_scan, keep_r_max=keep_r_max),
    }


def to_json(value: Any) -> Any:
    """Plain JSON values: tensors and arrays as (nested) lists, non-finite numbers as ``None``."""
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, np.generic):
        return to_json(value.item())
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
