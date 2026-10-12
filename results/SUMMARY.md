# FollowNet-2: what the benchmark found

Summary of milestones 1-5. Numbers are from `results/m4/REPORT.md` (driver split, fold 0,
seed 0, full training parts, reconstructed data; intervals are 95 % driver bootstraps with
1000 replicates unless stated) and `results/leaderboard/`. Protocols: `docs/protocols.md`;
experiments: `docs/experiments.md`; every decision and deviation: `docs/decisions.md`.

## The benchmark in one paragraph

Six datasets with car-following events on the common 10 Hz schema (NGSIM, FollowNet-HighD,
OpenACC, Waymo, FollowNet-NGSIM, TGSIM; 55,000 events), each in a raw and a
physics-reconstructed version, with event-, driver-, site-, regime- and block-level splits
and a leakage check. Twenty-four reference entries: two baselines (persistence, constant
speed), three mechanistic models in two calibrations each (pooled, median of per-event),
and sixteen learned models from kNN to diffusion, including closed-loop-trained variants.
Five protocols: one-step prediction (P0), closed loop over 10 s, 30 s and the whole event
(P1), platoons of 100 followers behind recorded OpenACC leaders and a braking pulse (P2),
string stability by the analytic criterion and a step-response gain test (P3), a rare-event
suite of pulses, a full stop, cut-ins and 4,473 confirmed near-miss episodes from TGSIM (P4),
and transfer between datasets (P5). A composite traffic-level score ranks the models on
the concave-growth error, the hysteresis error, the unstable fraction and the platoon
collisions.

## Headline findings

1. **One-step accuracy is the wrong yardstick (H5.1, H5.3).** On NGSIM only the diffusion
   model beats the persistence baseline at one step (ratio 0.62 [0.57, 0.67]); every other
   model is worse, the MLP by 3 %, the LSTM by 30 %, the mechanistic models 19 to 75 times.
   On FollowNet-HighD nothing beats persistence (best: MLP, 2.5 times). The one-step
   ranking is *anti*-correlated with the traffic-level composite: Kendall tau -0.52
   [-0.60, -0.44] on NGSIM and -0.51 [-0.56, -0.43] on HighD. The top three under P0
   (diffusion, MLP, persistence) are replaced under the composite by mechanistic models on
   NGSIM (Newell median, Newell, IDM median) and by DDPG, IDM and IDM median on HighD.
2. **Closed-loop accuracy depends on the horizon, and on the dataset.** On NGSIM the
   one-step networks match the mechanistic models at 10 s (LSTM 3.07 m, IDM 3.36 m) and
   diverge afterwards (LSTM 81.6 m over the whole event, IDM 6.1 m); the closed-loop-trained
   variants lead at every horizon (mlp_loop 2.48 / 5.64 m at 10 s / whole event). Kendall
   tau between the P0 and the whole-event ranking is -0.30 on NGSIM; between 10 s and the
   whole event 0.34. On HighD (15 s events, free flow) the learned models lead everywhere
   (seq2seq 3.16 m, mlp_loop 3.17 m, IDM 4.0 m), and the same models are string unstable
   at every equilibrium and collide in 1,400-1,800 of 1,900 platoon followers: the
   whole-event ranking is anti-correlated with the composite there (tau -0.35).
3. **Trajectory imitation does not transfer to traffic-level behaviour.** Every learned
   model except DDPG is string unstable at (almost) every equilibrium (gains 1.1 to 7;
   DDPG at 6 % of the HighD equilibria and 50 % on NGSIM); the stability-regularised MLP
   is linearly stable on NGSIM (analytic margins agree with the step test at 0.05 m/s) but
   amplifies perturbations of 1 m/s at every speed. In 100-vehicle platoons on the
   in-range OpenACC leaders the mechanistic models and DDPG never collide; the learned
   models collide in 300 to 1,000 (NGSIM) and 800 to 1,800 (HighD) of the followers.
   In the rare-event suite the mechanistic models never collide; the NGSIM networks collide
   in 40 to 100 % of the synthetic scenarios (partly outside their speed range) and in
   8 to 36 % of the TGSIM near-miss episodes, against 1.5 to 7 % for IDM and Gipps.
4. **Driver leakage inflates little (H5.2, paired design).** Moving half of the events of
   every test driver into the training part changes the one-step error by at most 5 %
   (NGSIM: MLP 5.4 %, IDM 2.8 %, LSTM 0.1 %; OpenACC, where every probe driver leaks about
   95 events: lstm_loop 5.4 %, networks on average 1.8 %), the 10 s closed-loop error by at
   most 4 %, and the whole-event error only for the one-step MLP (25 %, 8 % on the affected
   events). Differences are often significant after Holm but small; the unpaired
   event-against-driver comparison moves by the same order in both directions. Driver
   identity is not what these models memorise.
5. **Rankings by spacing and by speed agree (H5.4 not supported).** Kendall tau between the
   P1 rankings by spacing and by speed error is 0.83 to 0.88 at every horizon on both
   datasets.
6. **Transfer is asymmetric.** Trained on NGSIM (congested), the networks degrade 5 to 15
   times on HighD and up to 43 times on OpenACC (LSTM); mechanistic models 1.4 to 4 times;
   DDPG transfers everywhere within 20 % (mean degradation 1.01, worst 1.17). Trained on
   HighD (free flow), models hold up on NGSIM (0.4 to 1.7 times). Ratios below 1 on TGSIM
   and NGSIM mean that the native one-step networks diverge in closed loop on their own
   long events (TGSIM: LSTM 542 m, MLP 445 m over the whole event; their closed-loop-trained
   variants 7 to 11 m), not that the foreign model is good.
7. **The data versions matter.** Raw NGSIM acceleration is noise (persistence 71 % of a
   constant-speed follower), raw TGSIM acceleration is unusable (99 % quantile 18 m/s^2);
   the reconstruction makes both consistent with the integration scheme. Near-miss triggers
   mined on raw tracks survive the reconstruction in 84 % of the TGSIM episodes but only
   38 % of the NGSIM ones (leader braking: 6 %).

## Composite leaderboard (top eight per dataset)

| NGSIM | composite | concave error* [m/s] | hysteresis error* | unstable fraction | collisions_100* |
|---|---|---|---|---|---|
| newell_median | 0.125 | 0.42 | 92 | 0.00 | 0 |
| newell | 0.168 | 0.43 | 98 | 0.00 | 0 |
| idm_median | 0.174 | 0.40 | 92 | 0.65 | 0 |
| gipps | 0.196 | 0.44 | 91 | 0.88 | 0 |
| ddpg | 0.261 | 0.47 | 111 | 0.50 | 0 |
| mlp_loop | 0.293 | 0.37 | 78 | 1.00 | 548 |
| idm | 0.342 | 0.55 | 111 | 0.00 | 0 |
| gru_loop | 0.348 | 0.45 | 88 | 0.95 | 597 |

| FollowNet-HighD | composite | concave error* [m/s] | hysteresis error* | unstable fraction | collisions_100* |
|---|---|---|---|---|---|
| ddpg | 0.130 | 0.64 | 311 | 0.06 | 433 |
| idm | 0.158 | 0.75 | 332 | 0.64 | 0 |
| idm_median | 0.185 | 0.47 | 227 | 1.00 | 81 |
| newell | 0.288 | 0.74 | 640 | 0.00 | 713 |
| newell_median | 0.288 | 0.88 | 407 | 0.00 | 636 |
| gipps | 0.299 | 0.87 | 404 | 1.00 | 0 |
| gail | 0.359 | 0.86 | 274 | 1.00 | 815 |
| knn | 0.402 | 0.75 | 492 | 1.00 | 781 |

`*` in-range profiles (leader speeds inside the training speed range). Composite rank
intervals (95 %, bootstrap over profiles, speeds and the three training seeds):
NGSIM newell_median 1 [1, 2], newell 2 [2, 4], idm_median 3 [1, 4], gipps 4 [3, 6.5],
ddpg 5 [4, 12]; HighD ddpg 1 [1, 4], idm 2 [1, 4], idm_median 3 [1, 5]. The full tables
with intervals, the rank-stability plots and the refused entries (none: all 376 run
directories pass the checklist) are in `results/leaderboard/`.

## Caveats

* Whole-event closed-loop errors of the one-step networks on NGSIM are dominated by a few
  drivers (five drivers carry 97 % of the squared error of the LSTM); their intervals are
  wide and the bootstrap mean lies below the point value.
* OpenACC has 25 drivers: its E2 intervals are coarse and its whole-event numbers noisy.
* The synthetic rare-event suite and half of the AstaZero leader windows lie above the
  speed range of the congested NGSIM data; in-range views are reported next to the full
  ones and feed the composite.
* The empirical platoon reference (concave growth, hysteresis) consists of ACC-equipped
  followers (two human-driven AstaZero laps excepted); the growth exponent of a human
  platoon is not measured here.
* The mechanistic models are calibrated on a 2,000-event sample with one objective; the
  networks use one training budget per family (decision M2-4). Single fold for the 24-model
  tables, three seeds; five folds for the eight-model subset.
* FollowNet-HighD events define spacing as `dhw` and have no driver ids (driver split =
  event split there); the NGSIM near-miss set is not used for scoring.

## Reproduction

```
python scripts/m1.py dataset=<name>            # events, splits, baselines, calibration
python scripts/m2.py dataset=<name>            # reference models
python scripts/m4.py main|seeds|paired|event|transfer
python scripts/m3_report.py; python scripts/m4_report.py
python scripts/make_leaderboard.py; python scripts/make_report.py
```

## Comparison with the published FollowNet baselines

FollowNet (Chen et al., Scientific Data 2023) reports, on the same highD and NGSIM I-80
events that this benchmark re-uses, the spacing MSE and the collision rate of five
baselines on a random 70/15/15 event split, rolled over the whole 15 s event. Our rollouts
start after the 3 s warm-up and are scored on the test part of the driver split (on the
FollowNet events this is a random event split, since they carry no identities); the
per-event mean of the spacing MSE is the comparable quantity.

| model | highD, published | highD, ours | NGSIM I-80, published | NGSIM I-80, ours |
|---|---|---|---|---|
| IDM | 27.5 (0 ‰) | 16.0 (0 ‰) | 55.0 (0 ‰) | 35.2 (0 ‰) |
| Gipps / GHR | 23.8 (0 ‰, GHR) | 18.6 (0 ‰, Gipps) | 62.2 (0 ‰, GHR) | 33.6 (0 ‰, Gipps) |
| NN / MLP | 24.1 (39.9 ‰) | 18.6 (7.2 ‰) | 29.0 (3.5 ‰) | 87.6 (36.6 ‰) |
| LSTM | 22.9 (3.7 ‰) | 10.9 (0.8 ‰) | 23.9 (17.3 ‰) | 55.8 (2.6 ‰) |
| mlp_loop / lstm_loop | --- | 10.0 / 10.2 | --- | 29.3 / 31.0 |
| DDPG | 38.5 (0 ‰) | 134.2 (0.4 ‰) | 46.9 (0 ‰) | 187.1 (0 ‰) |

Orders of magnitude agree on highD; our calibrated IDM and our LSTM are better than the
published ones, our one-step MLP and LSTM are worse on the NGSIM I-80 events (383 test
events against their 1,930 events, and long events where one-step networks diverge), and
our DDPG keeps a larger spacing than the record because its reward (Zhu et al. 2020) does
not compare the simulated with the observed spacing, which the FollowNet DDPG does.
