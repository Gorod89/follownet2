# Physics-informed reconstruction (data version `reconstructed`)

Implementation: `trafficloop/data/reconstruction.py`, method tag `qp-semi-implicit-v1`
(stored in `meta["reconstruction"]["method"]` of every reconstructed event).

## Why

Every dataset of the benchmark exists in two versions.

* `raw`: positions and speeds as measured (resampled to 10 Hz where needed), acceleration as
  provided by the source or, if the source has none, the Savitzky-Golay derivative of the
  follower speed. Raw series are not mutually consistent: integrating the recorded
  acceleration does not give the recorded speed, and integrating the speed does not give the
  recorded position. NGSIM in addition contains accelerations and jerks that no vehicle can
  produce.
* `reconstructed`: the trajectory closest to the measurements among those that the benchmark
  integrator can produce with bounded acceleration and jerk and a positive gap.

The reconstructed NGSIM of Montanino and Punzo (2015) is no longer hosted anywhere, so the
reconstruction is part of this repository and is applied to all datasets in the same way.
It follows the same idea as Montanino and Punzo (2015) and Makridis and Kouvelas (2023),
remove what is physically impossible and keep the trajectory close to the measurement, but it
is a different algorithm. Results obtained on this version must be reported as obtained with
`qp-semi-implicit-v1`, not as obtained on "reconstructed NGSIM".

## Method

For one vehicle with measured positions `x_obs[k]` and speeds `v_obs[k]`, `k = 0..n-1`,
`dt = 0.1 s`, the unknowns are `x[k]`, `v[k]` and the accelerations `a[k]`, `k = 0..n-2`:

```
minimise    w_x * sum_k (x[k] - x_obs[k])^2
          + w_v * sum_k (v[k] - v_obs[k])^2
          + w_j * sum_k ((a[k+1] - a[k]) / dt)^2

subject to  v[k+1] = v[k] + a[k] * dt              (1)  semi-implicit Euler,
            x[k+1] = x[k] + v[k+1] * dt            (2)  the integrator of all rollouts
            v[k] >= 0                              (3)
            a_min <= a[k] <= a_max                 (4)  default -8 and 4 m/s^2
            |a[k+1] - a[k]| <= j_max * dt          (5)  default 15 m/s^3
            x[k] <= x_lead[k] - s_min              (6)  follower only, default 0.3 m
```

This is a convex quadratic programme with sparse banded matrices; it is solved with OSQP
(tolerances 1e-6, polishing on). The two data terms carry the weights of a Huber loss:
after a solution the samples whose residual exceeds 2.5 standard deviations of the assumed
noise are weighted down, those beyond 5 are left out, and the programme is solved again, at
most three times (outlier step). Steps:

1. **Leader.** Programme (1)-(5) on the leader trajectory.
2. **Follower.** Programme (1)-(6) with the reconstructed leader in (6).
   *Track sources* (NGSIM, TGSIM, highD, ...) fit the follower to its own measured position.
   *Gap sources* (OpenACC, FollowNet, Waymo pairs) have no independent follower position:
   the gap is what was measured, so the follower is fitted to
   `x_lead_reconstructed - s_obs`, which makes the position term the squared residual of the gap.
3. **Exact consistency.** The solver meets (1), (2) and (6) to its tolerance only. The
   solution is therefore re-integrated from `(x[0], v[0])` with the clipped accelerations by
   the benchmark scheme, and the follower is shifted back rigidly if the smallest gap is
   below `s_min`. After this step (1)-(2) hold to machine precision (residuals below 1e-12
   in the tests) and (6) holds exactly.
4. The event keeps its id, its time grid and its regime label. `a[n-1]` repeats `a[n-2]`.

### Weights

`w_x = 1 / sigma_x^2` and `w_v = 1 / sigma_v^2` are the inverse noise variances of the two
measurements (`null` switches a term off; NGSIM speeds are derived from the positions and
are not used). The jerk weight is set through a cut-off frequency `f_c`. Without active
inequality constraints the programme is a linear filter; in the frequency domain the
estimate of the position is

```
x_hat(w) = (w_x * x_obs(w) + w_v * w^2 * x_v(w)) / (w_x + w_v * w^2 + w_j * w^6)
```

(`x_v` the position implied by the measured speed). The filter passes half of the signal
at `wc = 2 * pi * f_c` when

```
w_j = (w_x + w_v * wc^2) / wc^6 .
```

Defaults: `f_c = 0.5 Hz` for video-based sources, `1.0 Hz` for sources with instrumented
vehicles; per-dataset values are in `conf/dataset/*.yaml: reconstruction`.

### What "physically consistent" means here

The feasible set of the programme is exactly the set of trajectories that a model can
produce in a closed-loop rollout of the benchmark. Consequences:

* a model that outputs the recorded acceleration reproduces the recorded speed, position
  and gap exactly, so the error floor of P0 and P1 is zero on this version (on raw data the
  floor is the inconsistency of the record, reported as model `recorded_acceleration`);
* P0 scores are proportional on this version, `RMSE_v = dt * RMSE_a` and
  `RMSE_s = dt^2 * RMSE_a`, so rankings by spacing and by speed coincide in P0 and can
  only differ in closed loop.

## Diagnostics

Per event, in `meta["reconstruction"]`: solver status of both programmes, root mean square
change of gap, speed and acceleration, largest absolute acceleration and jerk before and
after, smallest gap before and after, whether the gap bound is active. Per dataset, in
`manifest.json` of the reconstructed directory: aggregate of the above and the list of
events for which the solver failed (they are absent from the reconstructed version; splits
are defined on the raw version and shared by both).

## Validation

`tests/test_reconstruction.py`, on synthetic IDM followers with known noise-free
trajectories and added measurement noise (`sigma_x = 0.3 m`, `sigma_v = 0.2 m/s`):

| Property | Test |
|---|---|
| bounds on speed, acceleration, jerk and gap hold | `test_constraints_hold` |
| result is a trajectory of the integrator (residual < 1e-12) | `test_result_is_a_trajectory_of_the_benchmark_integrator` |
| error against the truth drops: gap and speed below 60 %, acceleration below 25 % of the raw error | `test_noise_is_reduced` |
| noise-free data change by less than 5 cm, 0.05 m/s, 0.15 m/s^2 | `test_clean_data_are_left_almost_unchanged` |
| a record with overlapping vehicles gets a positive gap | `test_gap_stays_positive_when_the_record_overlaps` |
| bounds become active on an impossible record | `test_bounds_bind_on_an_impossible_record` |
| gap sources are fitted to the measured gap | `test_gap_based_sources_fit_the_measured_gap` |
| parallel and serial runs are identical | `test_parallel_run_equals_serial_run` |

## Results on the datasets of milestone 1

Root mean square over all samples; `bound` is the share of events whose largest
acceleration or jerk lies on the bound.

| Dataset | rms acceleration raw / rec. [m/s^2] | rms jerk raw / rec. [m/s^3] | bound: acceleration | bound: jerk |
|---|---|---|---|---|
| ngsim | 1.41 / 0.94 | 10.06 / 1.70 | 1.3 % | 5.5 % |
| follownet_ngsim | 0.79 / 0.84 | 1.18 / 1.37 | 0 | 0 |
| openacc | 0.45 / 0.47 | 0.31 / 0.38 | 0.3 % | 0.5 % |
| waymo_hu | 0.66 / 0.66 | 0.58 / 0.53 | 0 | 0 |
| follownet_highd | 0.36 / 0.37 | 0.17 / 0.20 | 0.01 % | 0.01 % |

Raw NGSIM is the only source that the reconstruction changes substantially: 19.5 % of its
samples have a jerk above 10 m/s^3 and 5.1 % an acceleration above 3 m/s^2 (the published
acceleration is clipped at 3.41 m/s^2). After the reconstruction it is still rougher than the
FollowNet events of the same road, which come from the reconstruction of Montanino and
Punzo (rms jerk 1.70 against 1.18 m/s^3), and 1.3 % of its events use the full braking
bound of 8 m/s^2, a sign of position outliers that a bound alone does not remove. Sources
that are smooth already become slightly rougher, because the recorded acceleration is a
smoothed derivative and the reconstructed one the exact difference of the speeds.

## Choice of the cut-off for NGSIM

600 events drawn at random, `sigma_x = 0.5 m`, positions only, outlier step on:

| `f_c` [Hz] | rms position residual [m] | rms acceleration [m/s^2] | rms jerk [m/s^3] | events on the acceleration bound | events on the jerk bound |
|---|---|---|---|---|---|
| 0.25 | 0.17 | 0.63 | 0.49 | 0 | 0 |
| **0.35** | 0.12 | 0.76 | 0.91 | 0 | 0 |
| 0.50 | 0.08 | 0.94 | 1.73 | 1.5 % | 7.2 % |
| 0.70 | 0.07 | 1.14 | 2.88 | 6.3 % | 26.7 % |

Reference: the FollowNet events of I-80, which come from the reconstruction of Montanino and
Punzo, have an rms acceleration of 0.79 m/s^2 and an rms jerk of 1.18 m/s^3. NGSIM uses 0.35 Hz
(`conf/dataset/ngsim.yaml`). The residual stays far below `sigma_x` at every cut-off: the
published positions are smoothed already, so what is left of their error is slow and cannot be
removed by a filter. Without the outlier step the table is the same within 0.01: 0.05 % of
the samples are treated as outliers.

## Limitations

* Each event is reconstructed on its own. A vehicle that is the follower of one event and
  the leader of another gets two slightly different trajectories; platoon consistency
  across more than two vehicles (step 4 of Montanino and Punzo) is not enforced.
* The acceleration bounds do not depend on the speed or on the vehicle class.
* `sigma_x`, `sigma_v` and `f_c` are set per dataset from the literature, not estimated from
  the data. Position-only sources leave substantial noise in the acceleration at
  `f_c = 0.5 Hz` (about 0.4 m/s^2 for `sigma_x = 0.3 m` on the synthetic data).
* The outlier step removes errors beyond 5 standard deviations of the assumed noise; smaller
  errors are followed as far as the cut-off and the bounds allow.
  `meta["reconstruction"]["max_abs_acc"]`, `max_abs_jerk` and `outlier_share` describe
  every event.
* Macroscopic validation (flow and wave speed before and after) is not done yet; it
  belongs to the fundamental-diagram metric of P2.
* Near-miss episodes are kept in the raw version only: a cut-in is a jump of the leader
  position, which this smoother would smear.
