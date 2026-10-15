# Protocol contract (milestone 3)

Binding for every protocol module under `trafficloop/protocols/`, the platoon simulator
`trafficloop/sim/platoon.py` and the statistics package `trafficloop/stats/`. The model side
is `docs/model_contract.md`; the data side `docs/data_layout.md` and `docs/loader_contract.md`.

## 0. Shared conventions

* SI units, `dt = 0.1 s`, `dv = v - v_lead` (positive when closing in), net gap `s`.
* Every rollout uses `trafficloop.sim.rollout.rollout` (or the identical step rule): the
  acceleration is clipped to `ACC_CLIP = (-8, 4)`, non-finite outputs are replaced by the lower
  clip and counted, `v[k+1] = max(v[k] + a[k] dt, 0)`, `x[k+1] = x[k] + v[k+1] dt`, a collision is
  `s <= 0` at any simulated sample. The first `WARMUP = 30` samples are observed; `model.begin()`
  is called once per batch with the observed warm-up (`rollout` does it).
* Synthetic scenarios are built with `trafficloop/protocols/scenarios.py`: a prescribed leader
  with a *hold* phase of `HOLD = 60 s` at its initial speed (warm-up included), the manoeuvre
  starting at `meta["manoeuvre_start"]`; the placeholder follower starts at the gap
  `initial_gap(v0) = 2 + 1.5 v0` and settles at its own equilibrium during the hold. Metrics
  of a scenario are computed from the manoeuvre start on, unless stated otherwise.
* Every protocol module exposes one entry `evaluate_<p>(model, ..., device, dtype) ->
  (summary: dict, tables: dict[str, pd.DataFrame])`. The summary holds JSON-serialisable
  scalars only (floats, ints, bools, short lists); the tables hold the per-item rows
  (events, windows, vehicles, speeds, scenarios) that the bootstrap and the figures use.
  The caller (`scripts/evaluate.py`) writes `<p>.json` and `<p>.<table>.parquet` into the run
  directory `results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/` with the
  provenance of the checkpoint, the protocol configuration and its hash.
* Determinism: the same model, inputs and device give identical numbers; nothing draws
  random numbers except where a `seed` argument says so.
* Device: `model.to(device)`; tensors are built on that device; nothing is read back inside
  a stepping loop (the GPU is shared; see `sim/rollout.py`). Default `dtype` float32 for
  rollouts and float64 for the equilibrium and gain computations of P3. Precision: the
  runner switches TF32 off (`torch.backends.cudnn.allow_tf32`,
  `torch.backends.cuda.matmul.allow_tf32`) and runs P3 on the CPU: with TF32 the recurrent
  layers add noise of about 1e-3 m/s^2 to the acceleration, the settle criterion of P3 never
  holds (an LSTM with 15 equilibria on the CPU had none on the GPU), and the platoon
  collisions of P2 move by about 7 %. The keys `p3.device` and `precision: full` of
  `conf/evaluate.yaml` are part of the result hashes.
* Tests pin one torch thread (`tests/conftest.py`) and use the synthetic IDM events of the
  fixtures or the models `idm`, `persistence`, `constant_speed` and a tiny `mlp`.

## 1. P1: closed loop over horizons (`protocols/p1.py`)

Horizons 10 s and 30 s as windows cut from the events, plus the whole event
(`protocols/closed_loop.py`). A window of horizon `h` samples starts at `k0 = WARMUP + j h`,
`j = 0, 1, ...`, needs `k0 + h <= len(event) - 1` (full windows only, stride = horizon), and
is rolled from the observed state at `k0` with the observed history before it. Scores per
horizon: pooled `rmse_s`, `rmse_v` over the simulated samples, `collision_rate` over windows,
`event_collision_rate` (events with at least one colliding window), `n_windows`, `n_events`,
`n_samples`. Rows: `event_id, follower_id, start, n, sse_s, sse_v, collided, first_collision`.

**P1s** (protocol `p1s`, runner `_p1`, block `p1s` of `conf/evaluate.yaml`): the same windows
at the horizon 1 s (10 samples, stride 1 s) on the same test events, between the one step of
P0 and the 10 s of P1, without the whole-event rollout (`whole: false`; it is in `p1.json`).
`p1s.json` has the one summary key `h1` with the scores of a P1 horizon (and `horizon_s`), the
table `p1s.h1.parquet` the rows above; E1b reads both (`docs/experiments.md`).

## 2. P2: platoon (`sim/platoon.py`, `protocols/profiles.py`, `protocols/p2.py`)

**Simulator** `simulate_platoon(model, v_lead, n_vehicles, s_init, hold=HOLD, clip, device,
dtype) -> PlatoonResult`: vehicle 0 is the leader (prescribed speed series after the hold,
positions by `scenarios.positions`), vehicles `1..N` are followers driven by the model; all
followers start at the gap `s_init` and the leader speed, with the warm-up rows
`(s_init, 0, v0, 0)`. One model call per step on the histories `[N, H, 4]` of all followers;
all vehicles advance together: `v_new = clamp(v + a dt, 0)`, `x_new = x + v_new dt`,
`s_new[i] = x_new[i-1] - x_new[i]`. Result arrays `x, v, a` of shape `[N + 1, T]`, `s` of
shape `[N, T]`, `collided [N]`, `first_collision [N]`, `manoeuvre_start`, `n_invalid`.
A collided vehicle keeps driving (its gap may be negative), exactly as in `rollout`.
Because a follower never depends on the vehicles behind it, the 20-vehicle platoon is the
first 20 followers of the 100-vehicle simulation: P2 simulates 100 followers once and reports
both prefixes.

**Profiles** (`profiles.py`): `LeaderProfile(name, source, v: ndarray, t0, followers:
list[RecordedFollower], vehicle_length)` with `RecordedFollower(index, v, s, driver)`
(`index` 1 = first follower, `s` its gap, `driver` `"ACC"` or `"Human"`).
`openacc_profiles(raw_dir, window=300.0)` reads the ten AstaZero platoon files
(`AstaZero/ASta_*.csv`, five vehicles, 13 to 28 m/s) and the ZalaZone dynamic-platform
files (`ZalaZone/dynamic_part*.csv`, a leader and up to ten ACC followers at 8 to 17 m/s
performing deceleration-acceleration perturbations; the ten windows with the largest
leader-speed variance are kept). Files: 10 Hz, five metadata rows then a header; columns
`Time, Speed<k>, ..., IVS<k>`, `IVS<k>` the gap between vehicles `k` and `k + 1`,
`Driver<k>` when present; speed in m/s. From every file the window of `window` seconds,
among starts every 10 s, with the largest variance of the leader speed `Speed1` is taken,
subject to every vehicle's speed being above 0.5 m/s, every `IVS` finite and at least four
recorded followers in the window. The recorded followers of the window are the empirical
reference (ACC vehicles except in the two manual AstaZero laps; the `driver` tag says
which). Vehicle length: the campaign's `*_veh_specifications.csv` when it gives one,
otherwise `VEHICLE_LENGTH`. Profiles are cached as `data/scenarios/openacc_profiles.json`
(arrays as lists, selection rule and SHA-256 of the source files) by
`save_profiles`/`load_profiles`. `braking_pulse(v0, v_low, decel=-3, accel=1.5, hold_low=10,
tail=240)` is the synthetic pulse: hold (by the simulator), deceleration to `v_low`,
`hold_low` seconds there, acceleration back to `v0`, `tail` seconds at `v0` (the wave
travels about 2 s per vehicle, so 100 followers need a tail above 210 s). The runner takes
`v0` from the training data (95 % quantile of the follower speeds, at most 25 m/s) and
`v_low = 0.4 v0`, so that the pulse lies inside the speed range the model was trained on.

**Speed range**: a profile is *in range* when its leader speed stays within
`[q01 - 1, q99 + 1]` m/s of the follower speeds of the training data; every metric is
reported over all profiles and, with the suffix `_in_range`, over the in-range profiles and
the pulse. The composite score uses the in-range values: half of the AstaZero windows
start above 24 m/s, where NGSIM-trained models (congested data, desired speed of the
calibrated IDM 23.9 m/s) cannot keep up with the leader.

**Starting gap**: every profile starts the platoon at the model's own equilibrium gap at
`v0 = v[0]`, found by a two-vehicle settle before the platoon run (constant-speed scenario,
hold 120 s, the convergence rule of P3; `s_init = initial_gap(v0)` and
`s_init_converged = False` when the model does not settle). Reason: with `s_init != s_e`
follower `i` has to fall back by `i (s_e - s_init)` relative to the leader, which no hold
absorbs for 100 vehicles. The 60 s hold of the platoon stays.

**Metrics** (`evaluate_p2(model, profiles, source_events, n_vehicles=(20, 100), ...)`), per
profile over the manoeuvre window, for the prefixes 20 and 100:

1. Speed dispersion `std_v[n]` of every vehicle `n = 0..N` over the window. *Concave-growth
   error*: RMS over the recorded followers `n = 1..n_rec` of `std_v_sim[n] - std_v_rec[n]`
   [m/s], averaged over the recorded profiles (`concave_growth_error`; every profile but the
   synthetic pulse: the OpenACC windows here, the human windows in P2h). Growth exponent
   `beta`: least-squares slope of `log std_v[n]` on `log n` over the followers with
   `std_v > 0` (`growth_exponent_20`, `growth_exponent_100`); concave growth means
   `0 < beta < 1`.
2. *Hysteresis loop area*: signed shoelace area of the closed polygon `(s_n(t), v_n(t))` over
   the window, per follower [m^2/s]. *Hysteresis error*: RMS over the recorded followers of
   `area_sim[n] - area_rec[n]`, averaged over the recorded profiles (`hysteresis_error`);
   `hysteresis_area_20/100`: mean of `|area|` over the prefix, all profiles. Profiles may
   differ in length and in `n_rec`: every metric is computed per profile over its own window.
3. *Wave speed*: the deepest leader deceleration of the window (`t_min = argmin v_0`,
   `v_before = max v_0` over the 30 s before it, threshold `(v_before + v_0(t_min)) / 2`);
   `t_n` is the first crossing of the threshold by vehicle `n` after the leader's crossing,
   `x_n(t_n)` its physical position (`x_n - n L`); least squares `x = c t + d` over the
   leader and the crossing followers of the prefix (at least 5) gives the wave speed [m/s]
   (negative = backwards) and `n_crossed`. `wave_speed` and `wave_speed_pulse` are fitted
   over the first 20 followers so that models are comparable (a damped wave changes speed
   along the platoon); `wave_speed_100`, `wave_speed_pulse_100` use every crossing vehicle
   of the 100-platoon. A profile whose deepest deceleration is not complete inside the
   window has no fit (`nan`, left out of the mean; `n_wave_profiles` counts the fitted
   ones). Positive values are physical in free flow (OpenACC windows at 13 to 28 m/s lie on
   the rising branch of the fundamental diagram).
4. *Fundamental-diagram error*: for every simulated sample of followers `1..N` in the
   window, density `k = 1000 / (s + L)` [veh/km] and flow `q = k v 3.6` [veh/h], `L` the
   profile's vehicle length; the same for every sample of `source_events` (the events of
   the dataset and version the model was trained on, `L` = `meta["leader_length"]` or
   4.5 m). Bins of 5 veh/km from 0 to 150; bins with at least 200 samples on both sides are
   compared: `fd_error` = RMS over bins of the mean-flow difference [veh/h], `fd_error_rel`
   = RMS of the relative difference, `fd_bins_compared`. Pooled over all profiles.
5. *Collisions*: `collisions_20`, `collisions_100` = followers with `s <= 0` at any simulated
   sample, summed over all profiles (OpenACC and pulse).
6. `settled_share`: followers whose `|a| < 0.05 m/s^2` over the last 5 s of the hold.

Summary keys: `concave_growth_error, hysteresis_error, growth_exponent_20,
growth_exponent_100, hysteresis_area_20, hysteresis_area_100, wave_speed, wave_speed_100,
wave_speed_pulse, wave_speed_pulse_100, n_wave_profiles, fd_error, fd_error_rel,
fd_bins_compared, collisions_20, collisions_100, n_profiles, settled_share,
s_init_converged_share, n_invalid`, the in-range copies `n_profiles_in_range,
concave_growth_error_in_range, hysteresis_error_in_range, growth_exponent_20_in_range,
growth_exponent_100_in_range, wave_speed_in_range, collisions_20_in_range,
collisions_100_in_range, settled_share_in_range`, and `pulse` (its parameters) and
`speed_range` (added by the runner). Tables: `profiles` (one row per profile and prefix with
every metric, `s_init`, `s_init_converged`, `n_recorded`, `in_range`) and `vehicles`
(profile, index, std_v, std_v_rec, area, area_rec, t_cross, x_cross, collided, settled).

### P2h: human platoon reference (`data/platoons.py`, `scripts/build_human_profiles.py`)

The recorded followers of the OpenACC windows are ACC vehicles in 17 of the 19 windows, so the
empirical reference of P2 is essentially ACC-driven. Protocol `p2h` is P2 unchanged (runner
`_p2`, block `p2h` of `conf/evaluate.yaml` equal to `p2` except `profiles`) on a second profile
set, `data/scenarios/human_profiles.json`, whose followers are all human drivers. The summary
keys and tables are those of P2, so `p2h.json` and `p2.json` compare key by key; the cache is
never built implicitly (`leader_profiles` builds the OpenACC cache only).

**Sources**: the tracks frames of the NGSIM loader (us-101 and i-80, three 15-minute files each;
`ngsim.load_tracks`, positions, net gaps, lanes and `preceding` as in the events) and of the TGSIM
loader (I-90/I-94 stationary and moving, I-294 L1 and L2, I-395; `tgsim.load_tracks`, leaders
derived on the 10 Hz grid as for the events), plus the two manual AstaZero windows copied from the
OpenACC cache under their names (`ASta_040719_platoon3`, `ASta_040719_platoon10`).

**Rules** (`data/platoons.py`):

1. *Chain*: a leader and at least four (`MIN_FOLLOWERS`) followers in one lane; the leader of
   follower `k` is vehicle `k - 1` at every sample, present at the same sample in the same lane, at
   a raw net gap in `[0.5, 120)` m (the valid samples of the events, D1); nobody changes lanes;
   every vehicle is present throughout. Followers are the loader's followers (NGSIM: autos in the
   mainline lanes) that are not flagged automated (TGSIM); any vehicle may lead (a truck, an
   automated vehicle: `meta["windows"][i]["leader_automated"]`).
2. *Window*: inside the chain's common interval the longest stretch of 60 to 300 s
   (`round(T / dt) + 1` samples) whose leader speed is at least 3 m/s at its first sample and
   spans at least 3 m/s; ties by the largest leader-speed variance, then the earliest start.
3. *Selection* per site: ranked by the number of followers, then the duration; a window sharing a
   vehicle with one already chosen is skipped; at most 20 per site.
4. *Reconstruction* (`docs/reconstruction.md`, settings of the dataset: NGSIM `sigma_x = 0.5 m`,
   `f_c = 0.35 Hz`; TGSIM `sigma_x = 0.3 m`, `f_c = 0.5 Hz`; positions only; bounds -8 / 4 m/s^2,
   15 m/s^3, `s_min = 0.3 m`) on the front bumpers: the leader first, then every follower against
   its reconstructed predecessor (`x_front <= x_front_prev - L_prev - s_min`). Each vehicle gets one
   trajectory, so the chain is consistent along the platoon (the events are reconstructed pair by
   pair, `docs/reconstruction.md`, Limitations). Rule 2 is checked again on the reconstructed
   leader; a failing window is replaced by the next candidate.

Profile: `v` the reconstructed leader speed; follower `k` its speed and the gap
`s_k = x_front[k-1] - L[k-1] - x_front[k]`, `driver = "Human"`; `t0` the start in seconds of the
recording; `vehicle_length` the mean length of the chain; `source =
<dataset>/<site>/<group>/lane<lane>/leader<id>/frame<frame>` (NGSIM `frame_id`, TGSIM 0.1 s step).
The cache has the layout of the OpenACC cache; `meta` holds the rules, the reconstruction
settings, the SHA-256 of the source files, the counts per site and one record per window (vehicle
ids, classes, lengths, leader speeds, raw and reconstructed gaps, solver diagnostics).

Windows as built on 2026-10-14 (96 track windows, 1,156 recorded followers, 1.9 h; plus AstaZero):

| site | windows | followers min / median / max | duration [s] | leader speed range [m/s] | leader speeds [m/s] |
|---|---|---|---|---|---|
| ngsim/us-101 | 20 | 11 / 12 / 16 | 60.0 / 62.0 / 78.2 | 9.2 / 12.5 / 14.9 | 0.0 to 15.3 |
| ngsim/i-80 | 20 | 8 / 10 / 21 | 60.1 / 67.6 / 85.3 | 3.8 / 7.0 / 11.5 | 0.0 to 11.5 |
| tgsim/i90_94_stationary | 20 | 11 / 14 / 23 | 60.5 / 62.0 / 90.9 | 3.6 / 6.1 / 11.0 | 0.0 to 11.0 |
| tgsim/i90_94_moving | 20 | 9 / 11 / 46 | 60.3 / 63.6 / 102.6 | 3.4 / 6.5 / 13.8 | 0.0 to 15.4 |
| tgsim/i294_l1 | 15 | 4 / 5 / 19 | 60.5 / 76.6 / 185.3 | 4.7 / 14.5 / 18.6 | 0.0 to 38.7 |
| tgsim/i395 | 1 | 4 | 63.3 | 6.9 | 4.0 to 10.9 |
| openacc/AstaZero | 2 | 4 | 300.0 | 9.0 / 14.5 | 13.3 to 27.7 |

I-294 L2 has no window (its tracks last at most 49 s). Durations, followers and speed ranges
are min / median / max.

**Differences from the OpenACC windows**: human followers (the AstaZero leader is on ACC); 60 to
185 s instead of 300 s, because a chain stays together in a field of view of a few hundred metres
for about a minute (NGSIM vehicles stay 50 to 90 s in the median, lane changes cut the chains)
and the rule prefers many followers to long windows; 4 to 46 recorded followers instead of 4 to
10; congested and stop-and-go traffic (55 of the 96 leaders drop below 0.5 m/s), so the waves
travel backwards; trajectories reconstructed (the OpenACC profiles keep the raw speeds and IVS
gaps); vehicle lengths from the data (4.3 to 14.9 m per chain, TGSIM bounding boxes of up to
20 m) instead of 4.5 m.
Most windows lie inside the speed range of NGSIM-trained models (91 of 99 profiles, pulse
included, for the NGSIM IDM); the AstaZero windows and the fast I-294 windows (up to 38.7 m/s, as
recorded) do not.

**What the numbers mean**: `concave_growth_error` and `hysteresis_error` measure the distance to
human followers (the per-follower references are `std_v_rec` and `area_rec` of the `vehicles`
table); `wave_speed` is the backward wave of congested traffic. A window of about a minute
holds the disturbance for the first 20 to 30 followers only (about 2 s per vehicle), so the
100-vehicle keys (`growth_exponent_100`, `wave_speed_100`, `hysteresis_area_100`) say how far the
disturbance travels within the window rather than how it grows; compare P2h with P2 on the
recorded-follower errors, the 20-vehicle keys and the collisions. `fd_error` compares with the
training data of the model, as in P2.

## 3. P3: string stability (`protocols/p3.py`)

Speeds `v_e = 5, 6, ..., 30 m/s` (`stability.analytic.SPEEDS`).

**Equilibrium**: a constant-speed scenario (`scenarios.constant_speed_event`) with
`hold = 120 s`, starting gap the analytic `equilibrium_gap` for memoryless models when it is
finite, else `initial_gap(v_e)`. Converged when, over the last 10 s of the hold,
`max |a| < 2e-3 m/s^2`, `|v - v_e| < 0.01 m/s` and `0 < s < 300 m`; `s_e` is the mean gap
over those 10 s. A speed counts as an equilibrium when it converged *and* the follower
tracks the small step below (`n_equilibria`; `n_converged` and `n_tracks` are reported too):
a follower that settles but never returns to the leader's speed (persistence, the
constant-speed baseline) has no equilibrium in the string sense. The fractions, `agreement`
and the maximal gains are taken over the equilibria (`null` when there is none);
`n_collided_large` and `tracks_share` are over the converged speeds.

**Numerical frequency response**: after the hold the leader steps to `v_e + delta` and, in a
second scenario, to `v_e - delta` (`delta = 0.05 m/s`, the linear regime) and holds for
`response = 300 s`. The impulse response is `h[k] = (v_f[k_s + k + 1] - v_f[k_s + k]) / delta`,
`k_s` the step index, averaged over the two signs (the even-order nonlinearity cancels); the
gain is `G(f) = |sum_k h[k] exp(-2 pi i f k dt)|` on `f = 0` and a log grid of 200 frequencies
from 0.002 to 2 Hz. Per speed: `max_gain = max_f G(f)`, `peak_frequency`, `tracks` (the
follower speed ends within 5 % of the step of the new leader speed), `unstable_numerical =
max_gain > 1 + tol` with `tol = 1e-5`, and `unstable_1e3 = max_gain > 1 + 1e-3` (the practical
band: at least 0.1 % growth per vehicle). A second pass with `delta_large = 1.0 m/s` (same
settle, both signs) measures the finite-perturbation response: `max_gain_large`,
`unstable_large = max_gain_large > 1 + 1e-3`, `collided_large` (the follower hits the leader
after the step: a runaway). The linear verdict agrees with the analytic criterion; the
finite-amplitude pass exposes models whose stability holds only at the exact equilibrium
(the stability-regularised MLP of NGSIM: unstable fraction 0.35 at 0.05 m/s with a maximal gain of 1.0001, 0.92 at 1 m/s). For a linear follower this is the exact transfer function of the
speed perturbation along a homogeneous platoon; string instability (`G > 1` at low
frequencies) appears as an overshoot of the step response. Near the analytic boundary the
excess gain is only about `m^2 / (8 f_s^2)` (1e-4 one grid step inside the unstable band of
the IDM defaults), so the verdict uses the tight tolerance, which lies above the noise floor
of float64 rollouts (`1 + 4e-12`) and of models computing in float32 (`1 + 1e-6`); the loose
band is reported next to it because it moves the boundary by 1 to 2 m/s.

**Analytic** (memoryless models only): `equilibrium_margin`, `partials` at `s_e` (from
`stability/analytic.py`); `unstable_analytic = margin < 0`.

Summary: `unstable_fraction` (numerical at `tol`, over the speeds with an equilibrium),
`unstable_fraction_1e3` (practical band), `unstable_fraction_large` and `max_gain_large`
(finite perturbation, with `n_collided_large`), `unstable_fraction_analytic` (`null` unless
memoryless), `max_gain` (over speeds), `peak_frequency` (at the maximum), `n_equilibria`,
`n_speeds`, `agreement` (share of the speeds with an equilibrium where both verdicts agree;
`null` unless memoryless), `tracks_share`, `tol`, `n_invalid`. Table `speeds`: `v_e, s_e,
s_e_analytic, converged, tracks, max_gain, peak_frequency, gain_0.005, gain_0.01, gain_0.02,
gain_0.05, gain_0.1, gain_0.2, gain_0.5, max_gain_large, collided_large, margin, f_s, f_dv,
f_v, unstable_numerical, unstable_1e3, unstable_large, unstable_analytic` (nullable
booleans where there is no equilibrium or no analytic criterion).

Sanity check: for the IDM the numerical `G(f)` must match the discrete-time closed form `H(z)`
of the integrator (the continuous form `|H(i w)|^2 = (f_s^2 + f_dv^2 w^2) / ((f_s - w^2)^2 +
(f_dv + f_v)^2 w^2)`, `w = 2 pi f`, partials from `idm_partials`, departs from it by more than
1 % above 0.04 Hz) within 1 % for `f <= 0.5 Hz` at every speed with an equilibrium, and the
two verdicts agree on more than 95 % of those speeds at the default tolerance, for the
parameter sets `(33.3, 1.5, 2.0, 1.0, 1.5)` (analytic boundary 18.6 m/s, numerical 18.3 at
`tol = 1e-5` and 16.8 at `1e-3`; gains of at most 1.007), `(22.4, 0.95, 3.49, 1.24, 2.48)`
(boundary 15.2 m/s, gain up to 1.09) and `(23.9, 1.43, 2.05, 1.05, 0.53)` (stable everywhere
below `v0`). `persistence` and `constant_speed` never track a step (`G(0) = 0`): the
protocol reports them without error.

## 4. P4: rare events (`protocols/p4.py`, `data/rare_events.py`)

**Synthetic suite** (`synthetic_suite(hold=HOLD) -> list[Event]`, ids in brackets):

* leader pulses for `a in {-4, -6, -8} m/s^2` and `v0 in {15, 25, 35} m/s`: deceleration at `a`
  by 10 m/s, 10 s at the low speed, acceleration at `+1 m/s^2` back to `v0`, 20 s at `v0`
  (`pulse_a-4_v15`, ...);
* full stop from free flow: `v0 = 30`, `-4 m/s^2` to standstill, 20 s stopped (`stop_v30`);
* cut-in for `tau in {0.5, 1.0} s` and `v0 in {15, 25}`: at the manoeuvre start a new leader
  appears at the gap `tau * v_f(k_c)` in front of the follower's *simulated* position and
  drives at `v0` for 30 s; the old leader disappears (`cutin_tau0.5_v15`, `cutin_tau1.0_v25`,
  ...). Implemented by `rollout_cut_in`: the step loop of `sim/rollout.py` with
  `dx_lead[k_c]` replaced by `x_f(k_c) + tau v_f(k_c) + v0 dt - x_lead(k_c)` from the
  simulated state at `k_c`. The pulses by -6 and -8 m/s^2 end with one shorter step so that
  the low speed is exactly `v0 - 10`.

**Empirical suite**: near-miss episodes mined on the raw tracks (M1), reconstructed and
confirmed by `trafficloop/data/episodes.py` (`reconstruct_episodes(events, cfg, workers) ->
(list[Event], report)`): leader jumps are the samples where `|x_lead[k+1] - x_lead[k] - 0.5
(v_lead[k] + v_lead[k+1]) dt| > 2 m` (positions only, so gap-based sources without leader
changes are unaffected); every leader piece between jumps with at least 10 samples goes
through `reconstruct_trajectory` (shorter pieces stay raw), the follower through
`reconstruct_trajectory` with the upper bound `x_lead_rec - s_min`; `s`, `dv`, `a` are
rebuilt, `meta["data_version"] = "reconstructed"`, `meta["reconstruction"]` carries the
diagnostics. The triggers of `EpisodeConfig` are evaluated again on the reconstructed
series (TTC and DRAC with `min_speed`, `min_gap` and `min_hold`; leader acceleration below
`leader_acc_below` held `min_hold` samples, inside reconstructed pieces only; cut-in gap below
`cut_in_gap_below * v` at a jump); `meta["episode"]["confirmed_triggers"]` lists the ones
that hold anywhere and `meta["episode"]["triggers_after_warmup"]` those that fire at or
after sample `WARMUP`; only episodes with a non-empty second list are kept (an episode whose
critical instants all lie in the observed warm-up tests nothing in a rollout).
`scripts/reconstruct_episodes.py dataset=<name> [max_events=<n>] [out=<tag>]` writes
`data/events/<name>_nearmiss/reconstructed[_<tag>]` with the counts in the manifest.

**Metrics** (`evaluate_p4(model, scenarios, episodes, ...)`), per scenario or episode from
the manoeuvre start (scenarios) or the end of the warm-up (episodes): `collided`,
`first_collision` [s after the start], `min_ttc` (`s / dv` over the samples with `dv > 0`,
capped at `TTC_MAX = 100 s`, which also stands for "never closing"; 0 when the follower
collided), `min_acc` of the follower,
`exceedance_4` and `exceedance_6` (share of the simulated samples with `a <= -4` and
`a <= -6 m/s^2`), `amplification = min_acc / min(a_lead)` for pulses and the stop. For the
episodes also the recorded follower's `min_ttc_recorded`, `min_acc_recorded`,
`exceedance_4_recorded`. Infinite quantiles are `null` in the summary (JSON) and `inf` in
the tables.

Summary: `collision_rate_synthetic`, `collision_rate_empirical`, `min_ttc_q05/q25/q50`
(synthetic and `_empirical`), `exceedance_4`, `exceedance_6` (pooled samples, synthetic and
`_empirical`), `amplification_mean`, `n_scenarios`, `n_episodes`, `n_invalid`, and the
recorded references `min_ttc_q50_recorded`, `exceedance_4_recorded`. Tables: `scenarios`
(one row per scenario) and `episodes` (one row per episode with `event_id, follower_id`,
the triggers and all metrics).

## 5. P5: transfer (`protocols/p5.py`)

A model trained on dataset `A` is scored with P1 on the test part (same split kind and
fold, same version) of every other dataset `B`; `scripts/evaluate.py` writes `p1_<B>.json`.
`transfer_matrix(runs)` assembles the relative degradation
`rmse_s(A -> B) / rmse_s(B -> B)` per horizon, where `B -> B` is the same model label trained
on `B` (`nan` when missing).

## 6. Composite traffic-level score (`protocols/composite.py`)

Inputs per model: `concave_growth_error` (P2), `hysteresis_error` (P2), `unstable_fraction`
(P3, numerical), `collisions_100` (P2). Ranks within the compared set (average ranks on
ties, lower is better), normalised to `(rank - 1) / (n - 1)`; the composite is their mean
(lower is better). This is variant `v1`, the primary score of the leaderboard. Variants
(`trafficloop.protocols.composite.VARIANTS`): `v1+p1` adds the whole-event P1 spacing
error, `v1+fd` the fundamental-diagram error, `v2` both (a realism component next to the
traffic-level inputs); `composite_score` also takes weights and a Borda rank sum. The
sensitivity analysis of E3 (`e3_composite_sensitivity`: every variant, leave-one-input-out,
doubled safety weights) reports Kendall's tau of every composite against the one-step
ranking and the top three; the leaderboard shows `v2` next to `v1`. Uncertainty: the four inputs are simulation outputs without a driver
dimension, so the bootstrap resamples the P2 profiles (concave-growth, hysteresis and
collision inputs) and the P3 speeds (unstable fraction) with replacement, 1000 times with a
seed, and reports the rank distribution of every model; seeds of the training add to it in
milestone 4.

## 7. Statistics (`trafficloop/stats/`)

* `bootstrap.py`: `bootstrap_groups(table, group, statistic, n_boot=1000, seed=0)` resamples
  the groups (followers) with replacement and returns one row of statistics per replicate;
  `pooled_rmse` statistics from `sse_*` and `n` columns use the fast path (per-group sums
  times multinomial counts); `percentile_ci(samples, level=0.95)`.
* `ranks.py`: `rank_distribution(samples: dict[model, ndarray], lower_is_better=True)` with
  mean rank, 95 % interval, `p_first`, `p_top3`; `kendall_tau(a, b)` for two metric maps and
  `kendall_tau_bootstrap(samples_a, samples_b)` with a percentile interval.
* `paired.py`: `paired_difference(table_a, table_b, group, statistic, n_boot, seed)` with
  identical group draws for both tables (common random numbers), returning the mean
  difference, its interval and the two-sided bootstrap p-value
  `min(1, 2 min((k_le + 1) / (B + 1), (k_ge + 1) / (B + 1)))` (never exactly zero), plus the
  Wilcoxon signed-rank p-value on the per-group values; `holm(pvalues)` returns the
  step-down adjusted p-values.
* Reproducibility: the same seed gives the same replicates.

## 8. Tests

Every module has a test file: determinism (two calls, identical outputs; batch composition
does not change a vehicle's result), the prefix property of the platoon, the IDM sanity of
P3 (sanity check), the cut-in gap (`s` right after the cut-in equals `tau v_f` within 1 %), the
constant-speed follower colliding in the full stop and the IDM not colliding in the
`-4 m/s^2` pulse from 15 m/s, the episode reconstruction keeping a leader jump and reducing
the rms acceleration of noisy tracks, the bootstrap reproducibility, the Holm adjustment on
a hand-computed example and the fast pooled-RMSE path equalling the generic one. P2h
(`tests/test_platoons.py`): a lane change and an invalid gap cut a fabricated chain, automated
vehicles lead but never follow, the window rules, the selection never reuses a vehicle, the
profile schema round trip, an NGSIM file whose truck leads but does not follow, and P2 on two
human windows of unequal length (a follower equal to the simulated one has zero error).
