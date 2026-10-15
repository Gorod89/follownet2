# E5: hyperparameter sweep of `lstm` on `ngsim`

Written by `scripts/sweep.py` (design: `docs/experiments.md`, section 4). Data version `reconstructed`, `driver` split, fold 0, seed 0; 16 configurations (0 = the reference configuration of `conf/model/lstm.yaml`), 16 trained, 16 validated. Runs: `results/sweeps/models/ngsim/reconstructed/driver-fold0/sweep_lstm_<kk>/seed0`.

Training budget of every run (`train.json` of configuration 0): `epochs` 40, `batches_per_epoch` 200, `patience` 6; 9461 training events.

Both criteria come from the validation part of the split, never from the test part. `val_rmse_a`: the one-step RMSE of the acceleration kept by the early stopping (`best_val_rmse_a` of `train.json`, m/s^2). `val_rmse_s_10s`: the spacing RMSE of the closed-loop 10 s windows of P1 on at most 2000 validation events (m). `val_rmse_a_p0`: the one-step RMSE of P0 on the same events, every configuration on the same instants (the windows of the early stopping depend on the batch size).

## Selection

| Selection | # | hidden | history | a_prev | lr | weight decay | batch | val_rmse_a | val_rmse_a_p0 | val_rmse_s_10s |
|---|---|---|---|---|---|---|---|---|---|---|
| one-step selection (`val_rmse_a`) | 01 | 256 | 30 | yes | 0.000127 | 0.0001 | 4096 | 0.01161 | 0.01194 | 2.418 |
| closed-loop selection (`val_rmse_s_10s`) | 01 | 256 | 30 | yes | 0.000127 | 0.0001 | 4096 | 0.01161 | 0.01194 | 2.418 |
| reference (configuration 0) | 00 | 64 | 10 | no | 0.001 | 0 | 4096 | 0.1161 | 0.1163 | 3.066 |

The two criteria select the same configuration. Kendall's tau-b between them over the 16 validated configurations: 0.183 (p = 0.35).

## Configurations

| # | hidden | history | a_prev | lr | weight decay | batch | val_rmse_a | val_rmse_a_p0 | val_rmse_s_10s | val coll. 10 s | epochs | fit s | status | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | 64 | 10 | no | 0.001 | 0 | 4096 | 0.1161 | 0.1163 | 3.066 | 0.0189 | 40 | 25.9 | validated | reference |
| 01 | 256 | 30 | yes | 0.000127 | 0.0001 | 4096 | 0.01161 | 0.01194 | 2.418 | 0.0175 | 40 | 81.9 | validated | one-step selection, closed-loop selection |
| 02 | 64 | 10 | yes | 0.000301 | 0.0001 | 4096 | 0.01236 | 0.0136 | 3.072 | 0.0542 | 40 | 24.8 | validated |  |
| 03 | 32 | 10 | no | 0.000266 | 0.0001 | 1024 | 0.3333 | 0.3321 | 3.113 | 0.0201 | 40 | 20.3 | validated |  |
| 04 | 32 | 10 | no | 0.0028 | 1e-05 | 1024 | 0.09465 | 0.09335 | 3.727 | 0.0344 | 40 | 22.8 | validated |  |
| 05 | 256 | 10 | no | 0.00185 | 1e-05 | 1024 | 0.06435 | 0.06379 | 4.839 | 0.0608 | 40 | 22.5 | validated |  |
| 06 | 32 | 10 | no | 0.000396 | 0 | 1024 | 0.2701 | 0.2685 | 3.543 | 0.0641 | 40 | 19.9 | validated |  |
| 07 | 256 | 20 | no | 0.000696 | 1e-05 | 4096 | 0.03302 | 0.03271 | 3.321 | 0.0501 | 40 | 55.9 | validated |  |
| 08 | 64 | 10 | no | 0.00223 | 0 | 4096 | 0.07391 | 0.07398 | 3.612 | 0.0521 | 40 | 24.4 | validated |  |
| 09 | 256 | 30 | no | 0.0005 | 1e-05 | 4096 | 0.03449 | 0.03326 | 5.645 | 0.0343 | 40 | 82.9 | validated |  |
| 10 | 32 | 20 | no | 0.000257 | 1e-05 | 1024 | 0.2366 | 0.2348 | 3.656 | 0.0332 | 40 | 26.7 | validated |  |
| 11 | 32 | 30 | yes | 0.000111 | 0 | 1024 | 0.03934 | 0.03907 | 3.205 | 0.0592 | 40 | 22.5 | validated |  |
| 12 | 32 | 20 | no | 0.000271 | 1e-05 | 1024 | 0.2317 | 0.2297 | 3.655 | 0.0343 | 40 | 22.8 | validated |  |
| 13 | 256 | 10 | no | 0.000289 | 0.0001 | 1024 | 0.1534 | 0.1525 | 3.031 | 0.0241 | 40 | 26 | validated |  |
| 14 | 32 | 30 | yes | 0.000101 | 0.0001 | 4096 | 0.03739 | 0.03921 | 3.126 | 0.0537 | 40 | 41.1 | validated |  |
| 15 | 64 | 10 | yes | 0.000146 | 0.0001 | 1024 | 0.02471 | 0.02482 | 2.916 | 0.047 | 40 | 38.5 | validated |  |

## Protocols of the evaluated runs (test part)

P0 `rmse_a` from `train.json`; P1 spacing RMSE of the 10 s windows and of the whole event (`p1.json`); P2 collisions of the 100-vehicle platoons on the in-range profiles (`collisions_100_in_range`); P3 `unstable_fraction`. `stale`: protocol files computed from another checkpoint.

| Run | Role | P0 rmse_a | P1 10 s rmse_s | P1 whole rmse_s | P2 collisions | P3 unstable fraction | stale |
|---|---|---|---|---|---|---|---|
| sweep_lstm_00 | reference | 0.1157 | 3.073 | 81.72 | 716 | 0.9333 | - |
| sweep_lstm_01 | one-step selection, closed-loop selection | 0.01185 | 2.405 | 6.467 | 614 | 1 | - |
