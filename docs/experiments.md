# Experiments (milestone 4)

Binding for `scripts/m4.py`, `follownet2/experiments/` and `scripts/m4_report.py`. The
protocols are `docs/protocols.md`; the models `docs/models.md`.

## 1. Runs and campaigns

A run is one model label on one dataset, data version (`reconstructed` throughout), split
kind, fold and seed, stored as
`results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/` with `train.json`
(training log, provenance, P0 and whole-event scores on the test part), the per-event
tables `test_p0.events.parquet` and `test_closed_loop.events.parquet`, and the protocol
files `p1.json` ... `p4.json`, `p1_<other>.json` with their tables. Runs are resumable:
`scripts/train.py` skips a finished run with the same configuration hash, and
`scripts/evaluate.py` recomputes a protocol only when its configuration or the checkpoint
changed.

`SUBSET = (idm, gipps, mlp, lstm, mlp_loop, lstm_loop, ddpg, stability_mlp)` is the
eight-model subset of E2 and E4; `persistence` is evaluated next to it at no cost.
Training uses the full training part (`train.max_train_events: null`); the mechanistic
calibration keeps its 2,000-event sample (decision C5).

| Campaign (`scripts/m4.py <campaign>`) | Runs | Purpose |
|---|---|---|
| `main` | all 24 labels, `ngsim` and `follownet_highd`, driver split, fold 0, seed 0; then P1-P4 and P1 on the other dataset | E1, E3 |
| `seeds` | the same with seeds 1 and 2 | seed spread of E1 and E3 |
| `paired` | SUBSET on `ngsim` and `openacc`, splits `paired-clean` and `paired-leaky`, folds 0-4, seed 0 | E2 (paired design, decision R2) |
| `event` | SUBSET on `ngsim` and `openacc`, split `event`, fold 0, seed 0 | E2 (unpaired comparison) |
| `transfer` | SUBSET on `openacc`, `waymo_hu`, `follownet_ngsim`, `tgsim` (driver split, fold 0, seed 0) and P1 of every SUBSET run on the test parts of all six datasets | E4 |
| `ablation` | the training ablations of decision M4-7 on `ngsim` and `follownet_highd` (fold 0, seed 0), P1-P4 | E3c |
| `folds` | folds 1-4 of the driver split for all 24 labels on `ngsim` and `follownet_highd`, P1-P4 | fold spread of E1 and E3 |
| `recompute [protocols=p2,p4]` | every existing run directory of the six datasets: the named protocols are recomputed where their configuration hash or the checkpoint changed (after a change of a protocol, such as the precision rule of `docs/protocols.md` section 0) | maintenance |

## 2. Paired splits (E2)

`scripts/make_paired_splits.py dataset=<name>` builds `data/splits/<dataset>_paired-clean.json`
and `<dataset>_paired-leaky.json` from the driver split of the dataset. For fold `k`:

* the test drivers of the driver split are taken; the events of every test driver with at
  least two events are shuffled (seed = split seed, fold) and halved: the first half are
  *leak* events, the rest *probe* events; single-event drivers give probe events only;
* the clean arm trains on the training part of the driver split and is scored on the probe
  events; the leaky arm replaces a random subset of the clean training part, of the size of
  the leak set, by the leak events, so both arms train on the same number of events;
  validation parts are identical;
* both arms are `Split` objects with explicit per-fold partitions (`Split.explicit`,
  fold -> `{train, val, test}`), kind `paired-clean` / `paired-leaky`, the seed of the driver
  split and five folds; `Split.partition` returns the explicit parts when they exist.

The report of the script (and `notes` of the splits) states per fold `n_probe`, `n_leak`,
`n_test_drivers`, `n_replaced` and the share of probe events whose driver has a leak event
(the share of the probe set that can profit from the leak).

## 3. Analyses

* **E1 (H5.1)**: per dataset and model the ratio `RMSE_a(model) / RMSE_a(persistence)` of
  P0 with a 95 % interval from 1000 driver-bootstrap replicates drawn once per dataset and
  applied to both tables (common random numbers), from `test_p0.events.parquet` of the
  fold-0 runs; the spread over seeds 0-2 (min and max of the ratio) next to it; for SUBSET
  the ratio on folds 0-4 from the clean arms (probe events).
* **E2 (H5.2)**: per dataset and SUBSET model the paired inflation
  `(clean - leaky) / clean` of P0 `rmse_a`, P1 whole-event `rmse_s` and P1 10 s `rmse_s` on
  the probe events pooled over the five folds, with `stats.paired.paired_difference`
  (bootstrap over drivers with common random numbers, Wilcoxon) and Holm over the models of
  a dataset; class means (mechanistic, network); next to it the unpaired comparison: event
  split against driver split on fold 0, each on its own test part.
* **E3 (H5.3, H5.4)**: per dataset the rankings of all models under P0 (`rmse_a`), P1 at
  10 s, 30 s and whole event (`rmse_s`, and `rmse_v` for H5.4), P2 (`concave_growth_error`,
  `hysteresis_error`, `collisions_100`, in range), P3 (`unstable_fraction`), P4
  (`collision_rate_synthetic_in_range`, `collision_rate_empirical`) and the composite;
  Kendall tau between every pair of rankings with 95 % bootstrap intervals (drivers for
  P0, P1 and the empirical P4; profiles and speeds for P2, P3 and the composite); the
  spacing-against-speed tau per horizon; the top three of every ranking and the models that
  enter or leave the top three between P0, P1 whole event and the composite.
* **E4 (P5)**: the transfer matrices of SUBSET (whole event and 10 s): relative degradation
  `rmse_s(A -> B) / rmse_s(B -> B)` with driver-bootstrap intervals, the mean degradation per
  model over the foreign targets, and the mean per target dataset.

`scripts/m4_report.py` writes `results/m4/REPORT.md` and one CSV per table
(`results/m4/e1_ratios.csv`, `e2_paired.csv`, `e2_unpaired.csv`, `e3_rankings.csv`,
`e3_kendall.csv`, `e4_transfer.csv`). Figures belong to milestone 5.
