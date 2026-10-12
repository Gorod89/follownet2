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
| `stability_mlp` | penalty of plan P1 (cf-stability) | own implementation of the analytic criterion (`follownet2/stability/analytic.py`); the sibling repository had no stability module at the time |
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
