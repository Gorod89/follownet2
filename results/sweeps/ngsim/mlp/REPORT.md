# E5: hyperparameter sweep of `mlp` on `ngsim`

Written by `scripts/sweep.py` (design: `docs/experiments.md`, section 4). Data version `reconstructed`, `driver` split, fold 0, seed 0; 16 configurations (0 = the reference configuration of `conf/model/mlp.yaml`), 16 trained, 16 validated. Runs: `results/sweeps/models/ngsim/reconstructed/driver-fold0/sweep_mlp_<kk>/seed0`.

Training budget of every run (`train.json` of configuration 0): `epochs` 40, `batches_per_epoch` 200, `patience` 6; 9461 training events.

Both criteria come from the validation part of the split, never from the test part. `val_rmse_a`: the one-step RMSE of the acceleration kept by the early stopping (`best_val_rmse_a` of `train.json`, m/s^2). `val_rmse_s_10s`: the spacing RMSE of the closed-loop 10 s windows of P1 on at most 2000 validation events (m). `val_rmse_a_p0`: the one-step RMSE of P0 on the same events, every configuration on the same instants (the windows of the early stopping depend on the batch size).

## Selection

| Selection | # | hidden | history | a_prev | lr | weight decay | batch | val_rmse_a | val_rmse_a_p0 | val_rmse_s_10s |
|---|---|---|---|---|---|---|---|---|---|---|
| one-step selection (`val_rmse_a`) | 09 | 64x64 | 20 | yes | 0.00293 | 0.0001 | 4096 | 0.006702 | 0.007235 | 2.889 |
| closed-loop selection (`val_rmse_s_10s`) | 10 | 64x64x64 | 20 | yes | 0.00287 | 0.0001 | 1024 | 0.00754 | 0.007971 | 2.85 |
| reference (configuration 0) | 00 | 64x64 | 10 | no | 0.001 | 0 | 4096 | 0.09213 | 0.09195 | 4.054 |

The two criteria select different configurations. Kendall's tau-b between them over the 16 validated configurations: -0.233 (p = 0.23).

## Configurations

| # | hidden | history | a_prev | lr | weight decay | batch | val_rmse_a | val_rmse_a_p0 | val_rmse_s_10s | val coll. 10 s | epochs | fit s | status | role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | 64x64 | 10 | no | 0.001 | 0 | 4096 | 0.09213 | 0.09195 | 4.054 | 0.0322 | 20 | 8 | validated | reference |
| 01 | 256x256 | 30 | no | 0.00186 | 1e-05 | 1024 | 0.0954 | 0.09533 | 3.527 | 0.082 | 40 | 17.5 | validated |  |
| 02 | 128x128 | 10 | yes | 0.000216 | 1e-05 | 1024 | 0.01178 | 0.01179 | 9.184 | 0.217 | 40 | 16.8 | validated |  |
| 03 | 32x32 | 30 | no | 0.00028 | 0 | 1024 | 0.1303 | 0.1293 | 3.353 | 0.0884 | 40 | 16.5 | validated |  |
| 04 | 32x32x32 | 10 | yes | 0.000662 | 0.0001 | 1024 | 0.01242 | 0.01242 | 4.941 | 0.101 | 40 | 16.8 | validated |  |
| 05 | 256x256 | 10 | no | 0.000675 | 0 | 1024 | 0.09168 | 0.09127 | 3.998 | 0.0602 | 17 | 7.1 | validated |  |
| 06 | 32x32x32 | 30 | yes | 0.00029 | 0.0001 | 4096 | 0.01496 | 0.01612 | 4.991 | 0.149 | 40 | 20.3 | validated |  |
| 07 | 64x64x64 | 20 | yes | 0.00227 | 1e-05 | 1024 | 0.02653 | 0.02631 | 16.81 | 0.575 | 15 | 6.3 | validated |  |
| 08 | 256x256 | 30 | no | 0.000285 | 1e-05 | 1024 | 0.1001 | 0.09989 | 3.477 | 0.0713 | 40 | 17.6 | validated |  |
| 09 | 64x64 | 20 | yes | 0.00293 | 0.0001 | 4096 | 0.006702 | 0.007235 | 2.889 | 0.0353 | 40 | 16.6 | validated | one-step selection |
| 10 | 64x64x64 | 20 | yes | 0.00287 | 0.0001 | 1024 | 0.00754 | 0.007971 | 2.85 | 0.0465 | 40 | 18 | validated | closed-loop selection |
| 11 | 32x32 | 30 | no | 0.00228 | 0.0001 | 4096 | 0.1058 | 0.106 | 3.33 | 0.0635 | 40 | 17 | validated |  |
| 12 | 64x64x64 | 20 | yes | 0.00237 | 1e-05 | 4096 | 0.009752 | 0.01072 | 6.631 | 0.173 | 28 | 12.2 | validated |  |
| 13 | 32x32 | 20 | yes | 0.00115 | 1e-05 | 1024 | 0.01125 | 0.01132 | 4.756 | 0.145 | 40 | 16.7 | validated |  |
| 14 | 64x64x64 | 10 | no | 0.000163 | 0.0001 | 1024 | 0.08436 | 0.08398 | 3.733 | 0.0716 | 40 | 17.9 | validated |  |
| 15 | 256x256x256 | 30 | yes | 0.000117 | 1e-05 | 1024 | 0.01278 | 0.01252 | 6.943 | 0.138 | 40 | 18.5 | validated |  |

## Protocols of the evaluated runs (test part)

P0 `rmse_a` from `train.json`; P1 spacing RMSE of the 10 s windows and of the whole event (`p1.json`); P2 collisions of the 100-vehicle platoons on the in-range profiles (`collisions_100_in_range`); P3 `unstable_fraction`. `stale`: protocol files computed from another checkpoint.

| Run | Role | P0 rmse_a | P1 10 s rmse_s | P1 whole rmse_s | P2 collisions | P3 unstable fraction | stale |
|---|---|---|---|---|---|---|---|
| sweep_mlp_00 | reference | 0.09212 | 4.109 | 9.787 | 860 | 1 | - |
| sweep_mlp_09 | one-step selection | 0.007651 | 2.988 | 44.92 | 424 | 1 | - |
| sweep_mlp_10 | closed-loop selection | 0.008363 | 2.907 | 46.73 | 599 | 1 | - |
