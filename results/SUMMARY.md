# TrafficLoop: what the benchmark found

Summary of milestones 1-5 and of the additions after them (protocols P1s and P2h, analyses
E1b, E3d and E5, the certified entry). Numbers are from `results/m4/REPORT.md` (driver
split, fold 0, seed 0, full training parts, reconstructed data; intervals are 95 % driver
bootstraps with 1000 replicates unless stated), `results/sweeps/` (E5) and
`results/leaderboard/`. Protocols: `docs/protocols.md`; experiments: `docs/experiments.md`;
every decision and deviation: `docs/decisions.md`.

## The benchmark in one paragraph

Six datasets with car-following events on the common 10 Hz schema (NGSIM, FollowNet-HighD,
OpenACC, Waymo, FollowNet-NGSIM, TGSIM; 55,000 events), each in a raw and a
physics-reconstructed version, with event-, driver-, site-, regime- and block-level splits
and a leakage check. Twenty-five reference entries: two baselines (persistence, constant
speed), three mechanistic models in two calibrations each (pooled, median of per-event),
sixteen learned models from kNN to diffusion, including closed-loop-trained variants, and a
certified hybrid (an IDM core plus a bounded residual network whose string stability is
certified a priori). The protocols: one-step prediction (P0), closed loop over 1 s windows
(P1s) and over 10 s, 30 s and the whole event (P1), platoons of 100 followers behind
recorded OpenACC leaders and a braking pulse (P2) and behind 98 platoon windows with human
followers from NGSIM, TGSIM and AstaZero (P2h), string stability by the analytic criterion
and a step-response gain test (P3), a rare-event suite of pulses, a full stop, cut-ins and
4,473 confirmed near-miss episodes from TGSIM (P4), and transfer between datasets (P5). A
composite traffic-level score ranks the models on the concave-growth error, the hysteresis
error, the unstable fraction and the platoon collisions.

## Headline findings

1. **One-step accuracy is the wrong yardstick (H5.1, H5.3).** On NGSIM only the diffusion
   model beats the persistence baseline at one step (ratio 0.62 [0.57, 0.67]); every other
   model is worse, the MLP by 3 %, the LSTM by 30 %, the mechanistic models 19 to 75 times.
   On FollowNet-HighD nothing beats persistence (best: MLP, 2.5 times). The one-step
   ranking is *anti*-correlated with the traffic-level composite: Kendall tau -0.52
   [-0.60, -0.45] on NGSIM, -0.53 [-0.56, -0.44] on HighD and -0.41 to -0.64 on the four
   other datasets. The sign holds under every composite of the sensitivity analysis (realism
   inputs added, a Borda count, each input left out, doubled safety weights): all 84
   intervals of tau(P0, variant), 14 variants on six datasets, lie below zero (highest upper
   bound -0.013). The top three under P0 (NGSIM: diffusion, persistence, MLP; HighD:
   persistence, MLP, mlp_loop) are replaced under the composite by mechanistic models on
   NGSIM (Newell median, Newell, IDM median) and by the certified hybrid, DDPG and IDM on
   HighD.
2. **Closed-loop accuracy depends on the horizon, and on the dataset.** On NGSIM the
   one-step networks match the mechanistic models at 10 s (LSTM 3.07 m, IDM 3.36 m) and
   diverge afterwards (LSTM 81.6 m over the whole event, IDM 6.1 m); the closed-loop-trained
   variants lead at 10 s, 30 s and over the whole event (mlp_loop 2.48 / 5.64 m at 10 s /
   whole event). Kendall tau between the P0 and the whole-event ranking is -0.31 on NGSIM;
   between 10 s and the whole event 0.35. On HighD (15 s events, free flow) the learned
   models lead everywhere (gru_loop 3.14 m, seq2seq 3.16 m, mlp_loop 3.17 m, IDM 4.0 m), and
   five of the six best (all but seq2seq, which has no equilibrium) are string unstable at
   every equilibrium and collide in 1,076 to 1,794 of the 1,900 platoon followers: the
   whole-event ranking is anti-correlated with the composite there (tau -0.36).
3. **Trajectory imitation does not transfer to traffic-level behaviour.** Of the 17
   learned entries, 8 (FollowNet-NGSIM) to 14 (OpenACC) are string unstable at every
   equilibrium P3 finds (gains 1.06 to 28) and up to five more at 64 to 96 % of them
   (entries without an equilibrium, one or two per dataset, are not counted); DDPG is
   unstable at 6 % of the HighD equilibria and 42 % on NGSIM (0 to 65 % over the six
   datasets), the certified hybrid at none (finding 11). The stability-regularised MLP is
   unstable at 35 % of its NGSIM equilibria at the verdict tolerance, but with gains of at
   most 1.0001 (none in the 0.1 % band), and amplifies perturbations of 1 m/s at 92 % of
   them. In 100-vehicle platoons on the in-range OpenACC leaders the mechanistic models,
   DDPG and the certified hybrid never collide on NGSIM, while the other learned models
   collide in 272 to 998 of the 1,000 followers (seq2seq 40); on HighD only the pooled IDM,
   Gipps and the certified hybrid never collide (DDPG 433, the other four mechanistic
   calibrations 81 to 1,123), and the other learned models collide in 181 to 1,794 of the
   1,900 followers (seq2seq 62). In the synthetic rare-event scenarios the mechanistic
   models and the certified hybrid never collide on NGSIM, DDPG collides in 1 of the 14 and
   the other 15 NGSIM networks in 14 to 100 % (13 of them in 64 % or more; partly outside
   their speed range); in the TGSIM near-miss episodes the NGSIM networks collide in 8 to
   36 %, against 1.5 to 7 % for IDM and Gipps (NGSIM and HighD) and 3.9 % for the certified
   hybrid.
4. **Driver leakage inflates little (H5.2, paired design).** Moving half of the events of
   every test driver into the training part changes the one-step error by about 5 % at most
   (NGSIM: MLP 5.4 %, IDM 2.8 %, LSTM 0.1 %; OpenACC, where every probe driver leaks about
   95 events: lstm_loop 5.4 %, networks on average 1.8 %), the 10 s closed-loop error by at
   most 4 % on NGSIM and 7 % on OpenACC, and the whole-event error notably only for the
   one-step MLP on NGSIM (25 %, 8 % on the affected events; the OpenACC changes of up to
   11 % are not significant after Holm). Differences are often significant after Holm but
   small; the unpaired event-against-driver comparison moves by the same order in both
   directions. Driver identity is not what these models memorise.
5. **Rankings by spacing and by speed agree (H5.4 not supported).** Kendall tau between the
   P1 rankings by spacing and by speed error is 0.83 to 0.87 at every horizon on NGSIM and
   HighD, 0.79 to 0.89 on all six datasets.
6. **Transfer is asymmetric.** Trained on NGSIM (congested), the networks degrade up to 13
   times over the whole event on HighD (GRU 13.0, gru_loop 13.3, diffusion 12.5, LSTM 11.7)
   and up to 51 times on OpenACC (gru_loop; LSTM 43), the mechanistic models 1.4 to 3.8
   times on HighD and up to 14 times on OpenACC. DDPG transfers everywhere within 20 % (mean
   degradation 1.01 over its 30 foreign pairs, worst 1.17), the certified hybrid within 65 %
   (mean 1.19, worst 1.65). Trained on HighD (free flow), every model holds up on NGSIM over
   10 s windows (0.5 to 1.6 times). Ratios below 1 on TGSIM and NGSIM mean that the native
   one-step networks diverge in closed loop on their own long events (TGSIM: LSTM 542 m, MLP
   444 m over the whole event; their closed-loop-trained variants 7 to 11 m), not that the
   foreign model is good.
7. **The data versions matter.** Raw NGSIM acceleration is noise (persistence 71 % of a
   constant-speed follower), raw TGSIM acceleration is unusable (99 % quantile 18 m/s^2);
   the reconstruction makes both consistent with the integration scheme. Near-miss triggers
   mined on raw tracks survive the reconstruction in 84 % of the TGSIM episodes but only
   38 % of the NGSIM ones (leader braking: 6 %).
8. **The closed-loop divergence sits in a few followers (E1b).** At 1 s the closed-loop
   ranking still follows the one-step ranking (Kendall tau with P0 0.85 to 0.95 on the six
   datasets, 0.84 to 0.92 with medians over windows); at 10 s it is 0.23 to 0.55 (medians
   0.11 to 0.63). Over the whole event it turns negative with the pooled RMSE on the four
   datasets with 30 s windows (NGSIM -0.31, FollowNet-NGSIM -0.16, OpenACC -0.23, TGSIM
   -0.40; HighD 0.50 and Waymo 0.09 on their short events) but is near zero or positive with
   medians over events (NGSIM -0.15, FollowNet-NGSIM -0.07, OpenACC 0.31, TGSIM 0.08). The
   divergence of the one-step networks in closed loop is concentrated in a few followers: on
   NGSIM five followers carry 99.0 and 99.6 % of the squared whole-event error of the LSTM
   and the GRU, whose median errors over events (4.7 and 4.8 m) are close to the IDM's
   (4.0 m).
9. **Tuning buys one-step accuracy, not stability (E5).** Sixteen configurations each of
   the LSTM and the MLP on NGSIM (width, depth of the MLP, history, learning rate, weight
   decay, batch size, the previous acceleration as input), selected on the validation part by
   the one-step error and by the closed-loop 10 s spacing error. For the LSTM both criteria
   pick the same configuration (256 units, 3 s history, previous acceleration as input):
   one-step error 0.0119 m/s^2 on the test part against 0.116 for the reference
   configuration, whole-event error 6.5 m against 81.7 m, but string unstable at every
   equilibrium, and 614 of the 1,000 in-range platoon followers collide. For the MLP the
   criteria disagree (Kendall tau between them over the 16 configurations -0.23; LSTM 0.18):
   the two selections have whole-event errors of 44.9 and 46.7 m against 9.8 m for the
   reference, and both are unstable at every equilibrium (424 and 599 collisions). With the
   previous acceleration as input the one-step error falls far below that of persistence
   (0.0077 to 0.0119 against 0.089 m/s^2): one-step accuracy is cheap.
10. **A human platoon reference changes little (P2h, E3d).** Behind the 98 windows of P2h
    with human followers (40 NGSIM and 56 TGSIM chains, two manual AstaZero runs) the
    composite ranks the entries much as with the ACC-driven OpenACC reference: Kendall tau
    0.64 to 0.77 between v1 and v1h and 0.60 to 0.83 between v2 and v2h over the six
    datasets, and the one-step ranking stays anti-correlated with it (tau(P0, v1h) -0.48 to
    -0.54, tau(P0, v2h) -0.27 to -0.57). Under the human reference the first three are
    mechanistic entries or the certified hybrid on five of the six datasets; on
    FollowNet-HighD DDPG (v1h) and GAIL (v2h) take the third place.
11. **The certified hybrid buys string stability with trajectory fidelity.**
    `residual_idm_certified` (decision M2-8, `docs/models.md`) is string stable at every one
    of its 26 equilibria on every dataset and never collides in the platoons (P2, P2h) or in
    the synthetic scenarios; its whole-event spacing error is 5 to 50 % above the IDM's
    (NGSIM 6.44 against 6.11 m, HighD 6.01 against 4.01, OpenACC 11.70 against 11.19, Waymo
    5.79 against 5.52, FollowNet-NGSIM 6.23 against 5.82, TGSIM 8.43 against 7.83) and its
    one-step ratio 17 to 125, like those of the mechanistic models. Composite rank 1 of 25
    on FollowNet-HighD, 3 on Waymo, 5 on TGSIM, 8 on OpenACC, 10 on NGSIM and 13 on
    FollowNet-NGSIM (v2: 2, 3, 4, 3, 9 and 7.5); mean transfer degradation 1.19, second to
    DDPG (1.01) among the non-baseline entries. The core with a stability margin wins the
    budget on every dataset: the pooled IDM core admits none on four datasets and too little
    for the residual to act on NGSIM (0.0018) and Waymo (0.12), against 0.15 to 0.34 for the
    margin core (lipschitz 0.45 to 1.02 at r_max = 0.3 m/s^2).

## Composite leaderboard (top eight per dataset)

From `results/m4/e3_rankings.csv` (25 entries, sorted by the composite, ties by the
whole-event P1 error):

| NGSIM | composite | concave error* [m/s] | hysteresis error* | unstable fraction | collisions_100* |
|---|---|---|---|---|---|
| newell_median | 0.130 | 0.42 | 92 | 0.00 | 0 |
| newell | 0.172 | 0.42 | 98 | 0.00 | 0 |
| idm_median | 0.182 | 0.40 | 92 | 0.65 | 0 |
| gipps | 0.224 | 0.44 | 91 | 0.88 | 0 |
| ddpg | 0.266 | 0.47 | 105 | 0.42 | 0 |
| mlp_loop | 0.312 | 0.37 | 78 | 1.00 | 548 |
| idm | 0.339 | 0.55 | 111 | 0.00 | 0 |
| gru_loop | 0.344 | 0.45 | 87 | 0.74 | 603 |

| FollowNet-HighD | composite | concave error* [m/s] | hysteresis error* | unstable fraction | collisions_100* |
|---|---|---|---|---|---|
| residual_idm_certified | 0.115 | 0.66 | 358 | 0.00 | 0 |
| ddpg | 0.146 | 0.64 | 274 | 0.06 | 433 |
| idm | 0.177 | 0.75 | 332 | 0.64 | 0 |
| idm_median | 0.198 | 0.47 | 227 | 1.00 | 81 |
| newell | 0.312 | 0.74 | 640 | 0.00 | 713 |
| newell_median | 0.312 | 0.88 | 407 | 0.00 | 636 |
| gipps | 0.323 | 0.87 | 404 | 1.00 | 0 |
| gail | 0.385 | 0.86 | 297 | 1.00 | 816 |

`*` in-range profiles (leader speeds inside the training speed range). Composite rank
intervals (95 %, bootstrap over profiles, speeds and the three training seeds) from
`results/leaderboard/`, which ranks the same 25 entries): NGSIM newell_median 1 [1.0, 2.0], newell 2 [2.0, 3.5], idm_median 3 [1.0, 4.0], gipps 4 [3.0, 7.0], ddpg 5 [4.0, 10.0];
HighD residual_idm_certified 1 [1.0, 4.0], ddpg 2 [1.0, 5.0], idm 3 [1.0, 5.0]. First three on the
other datasets (25 entries, seed 0): OpenACC newell, idm_median, gipps; TGSIM newell,
gipps, ddpg; Waymo gipps, idm, residual_idm_certified; FollowNet-NGSIM gipps, perl, then
lstm_loop and idm_median tied. The full tables with intervals, the rank-stability plots
and the refused entries (none of the 660 run directories) are in `results/leaderboard/`.

## Caveats

* Whole-event closed-loop errors of the one-step networks on NGSIM are dominated by a few
  followers (finding 8); their intervals are wide (LSTM 81.6 m [10.2, 140.6]) and the
  bootstrap mean lies below the point value.
* OpenACC has 25 drivers: its E2 intervals are coarse and its whole-event numbers noisy.
* The synthetic rare-event suite and half of the AstaZero leader windows lie above the
  speed range of the congested NGSIM data; in-range views are reported next to the full
  ones and feed the composite.
* The empirical platoon reference of P2 consists of ACC-equipped followers (two manual
  AstaZero runs excepted). P2h has human followers, but its track windows last 60 to 185 s
  with 4 to 46 recorded followers: its 100-vehicle keys say how far a disturbance travels
  within the window rather than how it grows, and the growth exponent of a long human
  platoon is still not measured.
* The mechanistic models are calibrated on a 2,000-event sample with one objective; the
  networks use one training budget per family (decision M2-4). The tables are fold 0 and
  seed 0; seeds 1-2 and folds 1-4 of the driver split exist for every entry on NGSIM and
  HighD (seed and fold spread of the leaderboard), the four other datasets have fold 0 and
  seed 0 only, and E1 reports five clean-arm folds for the eight-model subset.
* E5 tunes two networks on one dataset, one fold and one seed; its runs live in
  `results/sweeps/` and are never ranked.
* FollowNet-HighD events define spacing as `dhw` and have no driver ids (driver split =
  event split there); the NGSIM near-miss set is not used for scoring.

## Reproduction

```
python scripts/m1.py dataset=<name>            # events, splits, baselines, calibration
python scripts/build_human_profiles.py         # human platoon windows of p2h (data/scenarios/human_profiles.json)
python scripts/m2.py dataset=<name>            # reference models
python scripts/m4.py main|seeds|folds|paired|event|transfer|ablation
python scripts/m4.py main|seeds|folds|transfer models=residual_idm_certified   # the certified entry
python scripts/m4.py protocols datasets=openacc,waymo_hu,follownet_ngsim,tgsim  # every protocol of every entry there
python scripts/evaluate.py dataset=<name> model=<label> "protocols=[p1s,p2h]"   # the two newer protocols of one run
python scripts/sweep.py dataset=ngsim model=lstm n=16                          # E5 (and model=mlp)
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
