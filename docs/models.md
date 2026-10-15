# Reference models

Seventeen models of the specification plus the variants the benchmark adds. All share one
interface (`docs/model_contract.md`): `acc = model(history)` with the history
`(s, dv, v, a_prev)` in SI units, `dv = v - v_lead`. The implementations are the benchmark's
own; none is the code of the authors. Where a constant of a paper could not be checked it is
marked *unverified*.

## Overview

| Entry | Class | Family | Inputs | Parameters | Training |
|---|---|---|---|---|---|
| `persistence` | `PersistenceBaseline` | baseline | last acceleration | 0 | none |
| `constant_speed` | `ZeroAcceleration` | baseline | none | 0 | none |
| `idm`, `gipps`, `newell` | `IDM`, `Gipps`, `Newell` | mechanistic | state; delay models: 3 s of history | 5, 6, 3 | pooled calibration on the training part (C1, C2) |
| `idm_median`, `gipps_median`, `newell_median` | same | mechanistic | same | same | median of the per-event estimates of the training part (R4) |
| `knn` | `KNN` | network | last row | 0 (50,000 reference states) | sampling |
| `mlp` | `MLP` | network | 1 s | 6,209 | one step |
| `gru`, `lstm` | `GRU`, `LSTM` | network | 1 s | 13,313; 17,729 | one step |
| `mlp_loop`, `gru_loop`, `lstm_loop` | same | network | 1 s | same | one step plus rollout over 5 s |
| `seq2seq` | `Seq2Seq` | network | 1 s, ending 0.5 s ago | 34,881 | 10 steps ahead |
| `mdn_lstm` | `MDNLSTM` | network, distribution | 1 s | 18,639 | likelihood |
| `pidl` | `PIDL` | hybrid | last row | MLP + 5 | data and physics terms |
| `perl` | `PERL` | hybrid | 1 s | LSTM + 5 | physics first, then residual |
| `stability_mlp` | `StabilityRegularisedMLP` | network | last row | MLP | one step plus stability penalty |
| `residual_idm_certified` | `CertifiedResidualIDM` | hybrid | last row | 4,481 + 5 (frozen) | certified budget, then one step |
| `ddpg` | `DDPG` | policy | last row | 4,481 (actor) | reward of Zhu et al. (2020) |
| `gail` | `GAIL` | policy, distribution | 1 s | 13,314 (policy) | adversarial imitation |
| `maml_lstm` | `MAMLLSTM` | network, adaptive | 1 s, adapts on 3 s | 17,729 | meta-learning over drivers |
| `diffusion` | `DiffusionCF` | network, distribution | 1 s | 42,433 | denoising |

None of the learned models reads the last acceleration by default (`use_a_prev=False`), as in
the literature. With one second of speeds at 10 Hz a network can compute the past
acceleration itself; this is why the one-step error of `mlp` is below that of persistence.

## Differences from the papers

| Entry | Paper | Differences |
|---|---|---|
| `idm` | Treiber, Hennecke, Helbing (2000) | none; `delta = 4`; the gap is floored at 0.1 m inside the formula |
| `gipps` | Gipps (1981) | evaluated every 0.1 s on the state delayed by the reaction time, with linear interpolation between samples; braking rates as positive numbers |
| `newell` | Newell (2002) | bumper positions, so `d` is the net gap at standstill; no acceleration parameter, the acceleration is bounded by the clip of the rollout |
| `knn` | He, Zheng, Guan (2015) | target: the next acceleration; weights `1 / (distance + 1e-6)`; states scaled by their standard deviations; reference set: random sample of training instants |
| `mlp`, `gru`, `lstm` | FollowNet baselines (Chen et al. 2023) | output: acceleration of the next step instead of the next speed; sizes fixed by the benchmark; the variants `*_loop` add the rollout term, the plain ones are trained on one step only |
| `seq2seq` | Ma and Qu (2020) | reaction delay fixed at 0.5 s; first decoder input zero; teacher forcing with probability 0.5; applied with a receding horizon |
| `mdn_lstm` | mixture density network (Bishop 1994) on an LSTM | standard deviations by softplus with a floor; the point prediction is the mean of the mixture |
| `pidl` | Mo, Shi, Di (2021) | IDM parameters learned jointly inside the calibration bounds; the physics term uses the IDM clipped to `[-8, 4]`; collocation states drawn anew at every step from the range of the training data; `alpha = 0.7` |
| `perl` | Long et al. (2023) | the physics model is the IDM fitted by least squares of the one-step acceleration (clipped); the residual LSTM starts at zero and is trained on the one-step acceleration |
| `stability_mlp` | penalty of plan P1 (cf-stability) | own implementation of the analytic criterion (`trafficloop/stability/analytic.py`); the sibling repository had no stability module at the time |
| `residual_idm_certified` | `ResidualIDM` of cf-stability with the certified budget of its decision D86 | core from the benchmark's pooled calibration with two fallbacks; one-step loss instead of the rollout loss; certificate over the gaps 0.5 to 300 m; see the section below |
| `ddpg` | Zhu et al. (2020) | reward terms as in the paper; *unverified*: network sizes, exploration noise (Gaussian here), discount, learning rates; no separate collision-avoidance check; episodes of 10 s; the policy with the best validation reward is kept |
| `gail` | Kuefler et al. (2017) | PPO with a value network instead of TRPO; behaviour-cloning warm start of two epochs; car following with the history `(s, dv, v)` and one action instead of a multi-lane simulator with lidar features and two actions |
| `maml_lstm` | MetaFollower (Chen et al. 2024) | only the linear head adapts (ANIL); first-order outer update; step size of the inner loop divided by the largest eigenvalue of the task; no IDM part; tasks are drivers; at test time the support is the observed warm-up of the event |
| `diffusion` | DDPM (Ho et al. 2020), cosine schedule, DDIM | scalar target (acceleration of the next step); 50 steps, sampling with 10 DDIM steps; the point prediction is the mean of 8 samples started from the normal quantiles |

## Findings of milestone 2 about the models

* **Adaptation of `maml_lstm`.** With a fixed inner step of 0.1 the inner loop diverged for
  tasks whose feature matrix has a large eigenvalue (above `2 / 0.1`), and the support drawn
  at random did not match the consecutive warm-up used at test time: adaptation made the
  validation error worse in three of three seeds. With the normalised step and block
  supports it helps in all seeds (validation 0.266 against 0.281 without adaptation).
* **Point prediction of `diffusion`.** Eight random common noise values had a mean of -0.20,
  which shifted every prediction by -0.07 m/s^2; with the normal quantiles the bias is
  0.0025 m/s^2.
* **Parameters of `pidl`.** The jointly learned IDM parameters are weakly identified: with a
  true time gap of 2.0 s the estimate stops at 1.56 s, with 1.0 s at 1.16 to 1.19 s, while
  the network reproduces the data. The parameters of a PIDL model must not be read as driver
  parameters.
* **IDM defaults.** The IDM with `(v0, T, s0, a, b) = (33.3, 1.5, 2, 1, 1.5)` is string
  unstable between 0.9 and 18.6 m/s by the analytic criterion (14 of the 26 speeds of the
  grid 5 to 30 m/s). The synthetic drivers of the tests are of this kind.

## Certified hybrid (`residual_idm_certified`)

The certified hybrid of the sibling repository cf-stability (`ResidualIDM` with the budget of
its decision D86), ported to the interface of the benchmark in `trafficloop/models/certified.py`
and `trafficloop/stability/certificate.py` (decision M2-8):

```
a = IDM(s, dv, v; theta) + r_max * tanh(lipschitz * g(s / 20, dv / 2, v / 10))
```

on the last row of the history. `theta` is a frozen IDM core (benchmark IDM, `delta = 4`); `g`
is a tanh MLP `3 -> 64 -> 64 -> 1` with spectral normalisation on every linear layer and an
antisymmetric initialisation (duplicated hidden units with opposite output weights), so that
the untrained residual is zero and training starts at the core; the input scaler is the fixed
physical one of cf-stability. Then `|r| <= r_max` and `|dr/dx_j| <= r_max * lipschitz *
prod(layer norms) / scale_j` for every dataset.

**Certificate.** At an equilibrium the residual cancels the core, so the equilibrium gap of the
speed `v` lies in `I(v) = {s : |IDM(s, 0, v)| <= r_max}`. The a priori certificate holds at `v`
when the string-stability criterion of the model contract (section 3, `dv = v - v_lead`),
minimised over every gap of `I(v)` (clamped to 0.5-300 m, the range in which the benchmark
looks for equilibria) and over every residual derivative inside the bounds, is non-negative,
and `f_s > 0`, `f_v + f_dv < 0` are guaranteed. It sees the weights only through the product
of the layer norms, which spectral normalisation keeps at 1, so training cannot break it.
`lipschitz = 0.9 b / r_max`, with `b` the largest `r_max * lipschitz` for which the
certificate holds at every speed of 5-30 m/s. For a memoryless law linearised at an
equilibrium, the speed transfer function `H` of P3 under the integrator of the benchmark
satisfies `|den|^2 - |num|^2 = c (2 dt^2 (m - dt f_s f_v) + 4 c (1 + dt (f_v + f_dv)))` with
`c = 1 - cos(2 pi f dt)`, so `|H| <= 1` at every frequency when `m - dt f_s f_v >= 0` and
`f_v + f_dv >= -1/dt`: with `m >= 0`, `f_s > 0` and `f_v <= 0` the first holds, the second
holds for any realistic law. A certified model therefore shows no unstable equilibrium in P3,
up to the numerical tolerance of the protocol.

**Core.** Before training (`before_training`, called by the training entry) the model tries,
first possibility first: (1) the pooled IDM of the training part with `r_max = 0.3 m/s^2`, the
amplitude of the certified configuration of cf-stability (E4, D86), the same calibration (sample,
settings, seed) as the `idm` entry and therefore its core; (2) the pooled IDM calibrated with a
string-stability margin of at least 0.2 s^-2 at its equilibria of 5-30 m/s (the objective term of
cf-stability's decision D72) with `r_max = 0.3`; (3) that core with `r_max = 0.1 m/s^2`
(`fallback_r_max`). Steps 1 and 2 are both certified and the core with the larger admissible
budget is kept (a pooled core that admits a budget too small for the residual to act, as on NGSIM
with 0.0018, loses against the margin core); step 3 only when both admit none;
when none does, the training stops. Then the supervised trainer runs with the budget of the
networks on the one-step error of the prediction clipped to `[-8, 4]`, and the certificate of
the trained weights is checked; `train.json["certificate"]` holds the core (kind, parameters,
calibration, smallest margin), every attempt, `admissible`, `safety`, `r_max`, `lipschitz`, the
bounds, the layer norms at enforcement and after training, and per speed the verdict, the
guaranteed margin, the interval of gaps, the worst gap and the largest budget.

Budgets at fold 0 of the driver split, seed 0, computed by the entry's own steps at the two
amplitudes 1 and 0.3 m/s^2 (the entry uses 0.3; with 1 the budgets are a tenth as large and the
residual degenerates to an offset, which is why the amplitude of D86 was kept; `b`, the
largest `r_max * lipschitz`; "none" lists the speeds where the certificate fails for every
`lipschitz > 0`; the pooled core is that of the `idm` entry, bit for bit):

| Dataset | pooled core `(v0, T, s0, a, b)` | `b`, `r_max = 1` | core with margin 0.2 | `b`, `r_max = 1` | `b`, `r_max = 0.3` | step kept |
|---|---|---|---|---|---|---|
| `ngsim` | (23.88, 1.450, 2.319, 1.091, 0.541) | none (5-11 m/s) | (38.14, 1.474, 2.713, 1.851, 0.514) | 0.0319 | 0.339 | 2 |
| `follownet_highd` | (29.94, 0.750, 1.599, 0.327, 0.570) | none (5-20) | (35.38, 0.978, 2.776, 1.557, 0.501) | none (5-12) | 0.160 | 3 |
| `openacc` | (44.25, 0.871, 9.722, 1.638, 0.523) | none (5-20) | (44.52, 0.973, 8.930, 3.598, 0.501) | 0.0126 | 0.151 | 2 |
| `waymo_hu` | (44.41, 1.536, 1.161, 1.172, 0.500) | none (5-20) | (37.71, 1.294, 3.260, 1.759, 0.504) | 0.0079 | 0.303 | 2 |
| `follownet_ngsim` | (28.87, 1.754, 4.876, 0.938, 0.504) | none (5-14) | (36.50, 1.788, 5.669, 2.102, 0.503) | 0.0237 | 0.341 | 2 |
| `tgsim` | (43.80, 1.449, 1.789, 0.821, 0.509) | none (5-22) | (39.89, 1.447, 3.083, 1.854, 0.503) | 0.0784 | 0.330 | 2 |

No pooled core admits a budget at `r_max = 1`, even where it is string stable at all its own
equilibria (`waymo_hu`, smallest margin 0.08 s^-2): with `a` near or below `r_max` a residual
of 1 m/s^2 can hold an equilibrium far from the core's own one, at gaps where the IDM is string
unstable. The core with the margin is chosen automatically on every dataset; the amplitude of
D86 is needed on FollowNet-highD, as in cf-stability.


Realised choice of the entry (`r_max = 0.3 m/s^2`, fold 0 of the driver split, seed 0; `b` the admissible
budget of each core at that amplitude, the larger one is kept; P1 whole-event spacing RMSE next to the
`idm` entry's; P2 collisions of the 100-vehicle platoons in range):

| Dataset | `b`, pooled core | `b`, margin core | `lipschitz` kept | core kept `(v0, T, s0, a, b)` | P3 unstable fraction (equilibria) | P1 whole / `idm` [m] | P2 collisions |
|---|---|---|---|---|---|---|---|
| `ngsim` | 0.0018 | 0.339 | 1.02 | (38.1, 1.47, 2.71, 1.85, 0.51) | 0.00 (26) | 6.44 / 6.11 | 0 |
| `follownet_highd` | 0.0000 | 0.160 | 0.48 | (35.4, 0.98, 2.78, 1.56, 0.50) | 0.00 (26) | 6.01 / 4.01 | 0 |
| `openacc` | 0.0000 | 0.151 | 0.45 | (44.5, 0.97, 8.93, 3.60, 0.50) | 0.00 (26) | 11.70 / 11.19 | 0 |
| `waymo_hu` | 0.1200 | 0.303 | 0.91 | (37.7, 1.29, 3.26, 1.76, 0.50) | 0.00 (26) | 5.79 / 5.52 | 0 |
| `follownet_ngsim` | 0.0000 | 0.341 | 1.02 | (36.5, 1.79, 5.67, 2.10, 0.50) | 0.00 (26) | 6.23 / 5.82 | 0 |
| `tgsim` | 0.0000 | 0.330 | 0.99 | (39.9, 1.45, 3.08, 1.85, 0.50) | 0.00 (26) | 8.43 / 7.83 | 0 |

The margin core wins on every dataset: the pooled core admits no budget on four of them and a budget
too small to act on NGSIM (0.0018) and on the Waymo pairs (0.12). The certified entry is string stable at
every one of its 26 equilibria on every dataset and never collides in the platoons, and its whole-event
spacing error is 5 to 50 percent above that of the `idm` entry: the certificate costs trajectory fidelity.

**The core with the margin (step 2).** The same pooled calibration as the `idm` entry (the same
2,000 training events, differential evolution, settings and seed), with one more term in the
objective: `10 * sum_v relu(0.2 - M(v))` over `v = 5, ..., 30 m/s`, `M(v)` the closed-form
margin of the IDM at its equilibrium, counted -10 where `v >= v0` (decision D72 of
cf-stability). On NGSIM the pooled core is string stable at its equilibria (margins 0.07-0.15
s^-2 at 5-23 m/s) but has none above `v0 = 23.9 m/s`, and with `a = 1.09 m/s^2` a residual of
1 m/s^2 can hold an equilibrium anywhere between 6.9 and 33.6 m at 5 m/s (its own equilibrium:
9.6 m) and at any gap beyond 26 m at 20 m/s. The core with the margin, `(38.14, 1.474, 2.713,
1.851, 0.514)`, has a margin of at least 0.200 at every speed of the grid: a higher desired
speed (38.1 m/s, so every speed of the grid has an equilibrium), a larger maximal acceleration
(1.85 m/s^2, which shrinks the gaps an equilibrium can take to 8.1-14.9 m at 5 m/s and
26.6-51.9 m at 20 m/s), slightly larger `T` and `s0`, `b` near its lower bound as before. It
fits the training sample less well (spacing RMSE 6.08 m against 5.87 m, objective 0.471
against 0.453).

**What the budget leaves to the residual (NGSIM).** With `b = 0.339` and `r_max = 0.3` the
`lipschitz` in force is 1.018; the slope bounds of the residual are 0.0153 s^-2 (gap),
0.153 s^-1 (relative speed) and 0.0305 s^-1 (speed), so the residual can use its whole
amplitude of 0.3 m/s^2 across the state range of the data. At the amplitude 1 m/s^2 of the first run the
budget was 0.032 and the residual degenerated to a constant offset (decision M2-8).

**Differences from cf-stability.** The conventions agree (net gap, `dv = v - v_lead`, SI
units), so nothing changes sign. The core is the benchmark IDM (gap floored at 0.1 m instead of
0.01 m, identical wherever an equilibrium can lie), computed in the dtype of the input (float64
in P3); the residual computes in float32. The core comes from the benchmark's pooled calibration
with the fallbacks above instead of one configured core. Training is the one-step loss of the
benchmark instead of the rollout loss of cf-stability (its D63), and there is no fine-tuning on
a second dataset. The certificate covers the gaps 0.5-300 m instead of 1-200 m; on the range of
cf-stability the port reproduces its margins, interval ends and budgets to 1e-12.

**NGSIM results** (reconstructed, driver split fold 0, seed 0; the margin core with `b = 0.339` at
`r_max = 0.3`): after training the certificate holds at all 26 speeds (guaranteed margin at least
0.021 s^-2; product of the layer norms 1.054 at enforcement, 1.000 after training).
Against the `idm` entry:

| | `residual_idm_certified` | `idm` |
|---|---|---|
| P0 `rmse_a` [m/s^2], ratio to persistence | 2.97, 33.3 | 2.29, 25.7 |
| P1 `rmse_s` 10 s / 30 s / whole event [m] | 3.91 / 5.32 / 6.44 | 3.36 / 5.05 / 6.11 |
| P1 `rmse_v` 10 s / 30 s / whole event [m/s] | 0.94 / 0.87 / 0.94 | 0.84 / 0.81 / 0.88 |
| P1 collisions | 0 | 0 |
| P3 equilibria, unstable fraction (1e-5, 1e-3, 1 m/s) | 26, 0 / 0 / 0 | 19, 0 / 0 / 0 |
| P3 largest gain | 1 + 6e-11 | 1 + 6e-11 |

The `idm` core has no equilibrium above its desired speed of 23.9 m/s; the core with the margin
has one at every speed of the grid. The price of the certificate on NGSIM is the core with the
margin (its accuracy): the validation error fell from 2.85 to 2.85 m/s^2 in 26 epochs and the
whole-event spacing error is 5 percent above the `idm` entry's.

