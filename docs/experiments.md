# Experiments (milestone 4)

Binding for `scripts/m4.py`, `trafficloop/experiments/` and `scripts/m4_report.py`. The
protocols are `docs/protocols.md`; the models `docs/models.md`.

## 1. Runs and campaigns

A run is one model label on one dataset, data version (`reconstructed` throughout), split
kind, fold and seed, stored as
`results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/` with `train.json`
(training log, provenance, P0 and whole-event scores on the test part), the per-event
tables `test_p0.events.parquet` and `test_closed_loop.events.parquet`, and the protocol
files `p1.json` ... `p4.json`, `p1s.json`, `p2h.json` and `p1_<other>.json` with their
tables. Runs are resumable: `scripts/train.py` skips a finished run with the same
configuration hash, and `scripts/evaluate.py` recomputes a protocol only when its
configuration or the checkpoint changed.

`SUBSET = (idm, gipps, mlp, lstm, mlp_loop, lstm_loop, ddpg, stability_mlp)` is the
eight-model subset of E2 and the default of the `transfer` campaign (E4); `persistence` is
evaluated next to it at no cost.
Training uses the full training part (`train.max_train_events: null`); the mechanistic
calibration keeps its 2,000-event sample (decision C5).

"The protocols" below are `PROTOCOLS = (p1, p1s, p2, p2h, p3, p4)` of `scripts/m4.py`, the
default list of `conf/evaluate.yaml`: P1, its 1 s windows `p1s`, P2, the human platoon
reference `p2h`, P3 and P4. `p2h` reads `data/scenarios/human_profiles.json`, which
`scripts/build_human_profiles.py` builds once; no campaign builds it. Every campaign takes
`datasets=a,b` and `models=a,b` in place of the datasets and labels of its row.

| Campaign (`scripts/m4.py <campaign>`) | Runs | Purpose |
|---|---|---|
| `main` | all 25 labels, `ngsim` and `follownet_highd`, driver split, fold 0, seed 0; then the protocols and P1 on the other dataset | E1, E1b, E3, E3d |
| `seeds` | the same with seeds 1 and 2 | seed spread of E1 and E3 |
| `paired` | SUBSET on `ngsim` and `openacc`, splits `paired-clean` and `paired-leaky`, folds 0-4, seed 0 | E2 (paired design, decision R2) |
| `event` | SUBSET on `ngsim` and `openacc`, split `event`, fold 0, seed 0 | E2 (unpaired comparison) |
| `transfer` | SUBSET on `openacc`, `waymo_hu`, `follownet_ngsim`, `tgsim` (driver split, fold 0, seed 0) and P1 of every SUBSET run on the test parts of all six datasets; E4 has been run for all 25 labels (`models=`) | E4 |
| `ablation` | the training ablations of decision M4-7 on `ngsim` and `follownet_highd` (fold 0, seed 0), the protocols | E3c |
| `folds` | folds 1-4 of the driver split for all 25 labels on `ngsim` and `follownet_highd`, the protocols | fold spread of E1 and E3 |
| `protocols` | the protocols of the existing fold-0, seed-0 runs of all 25 labels on `ngsim` and `follownet_highd`, P1 also on the test part of the other one; `datasets=openacc,waymo_hu,follownet_ngsim,tgsim` does the same for the four other datasets (runs trained by `scripts/m2.py`), with P1 on both `ngsim` and `follownet_highd` | E1b, E3 and E3d on all six datasets; after a change of a protocol |
| `recompute [protocols=p2,p4]` | every existing run directory of the six datasets: the named protocols (default `p3`) are recomputed where their configuration hash or the checkpoint changed (after a change of a protocol, such as the precision rule of `docs/protocols.md` section 0; `protocols=p1s` added the 1 s windows to every run) | maintenance |

The certified entry `residual_idm_certified` (decision M2-8) joined after these campaigns
had run: `main`, `seeds`, `folds` and `transfer` with `models=residual_idm_certified` (the
last one trains it on the four other datasets), then `protocols` with the four other
datasets. The campaigns over all labels (`main`, `seeds`, `folds`, `protocols`) include it
from now on (`MODELS` of `scripts/m2.py`).

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
* **E1b**: per dataset and model the pooled P1 spacing RMSE and the median over windows (over
  events for the whole event) at 1 s (protocol `p1s`), 10 s, 30 s and the whole event, the share of
  the whole-event summed squared error carried by the five followers with the largest sums, and
  Kendall's tau-b between the P0 ranking and every horizon's ranking (point estimates;
  `trafficloop/experiments/horizons.py`).
* **E3d**: the composite variants v1 and v2 built from the human platoon reference of protocol
  `p2h` (`v1h`, `v2h`: the P2 inputs replaced by those of `p2h`, the P1 whole-event error and the P3
  unstable fraction unchanged) next to the ACC-referenced ones, Kendall's tau-b between the
  rankings and the first three models of every composite (point estimates;
  `trafficloop/experiments/human_reference.py`).
* **E4 (P5)**: the transfer matrices of the reference entries with `p1_<other>.json` files
  (`MODELS` of `scripts/m2.py`, all 25; whole event and 10 s): relative degradation
  `rmse_s(A -> B) / rmse_s(B -> B)` with driver-bootstrap intervals, the mean degradation per
  model over the foreign targets, and the mean per target dataset.

`scripts/m4_report.py` writes `results/m4/REPORT.md` and one CSV per table
(`results/m4/e1_ratios.csv`, `e2_paired.csv`, `e2_unpaired.csv`, `e3_rankings.csv`,
`e3_kendall.csv`, `e3_composite_variants.csv`, `e3_composite_sensitivity.csv`,
`e3_pareto.csv`, `e3_ablations.csv`, `e4_transfer.csv`, `e1_horizons.csv`,
`e1_horizon_taus.csv`, `e3_human_reference.csv`, `e3_human_taus.csv`, `e3_human_top3.csv`).
Figures belong to milestone 5.

## 4. E5: hyperparameter sweep with two selection criteria

Binding for `scripts/sweep.py` and `trafficloop/experiments/sweep.py` (decision M4-8). The
question: does a tuned reference network escape the traffic-level failures of E3, and does the
configuration chosen by the one-step validation error differ from the one chosen by a
closed-loop validation error?

```bash
python scripts/sweep.py dataset=ngsim model=lstm n=16 seed=0      # stage=all
python scripts/sweep.py dataset=ngsim model=mlp n=16 stage=train   # train | validate | select | evaluate | report
```

**Design.** A sweep is one model (`lstm`, `gru` or `mlp`) on one dataset, data version
(`reconstructed`), split kind (`driver`), fold (0) and seed (0); the seed is the training seed
of every run and the seed of the sampling. `n` configurations are drawn at random;
configuration 0 is always the reference configuration (`conf/model/<model>.yaml` over
`conf/train.yaml`). Configuration `i >= 1` comes from its own generator
`numpy.random.default_rng([seed, crc32(model), i])`, so it depends neither on `n` nor on the
other configurations: raising `n` extends a sweep. Every configuration has the training budget
of `conf/train.yaml` (at most 40 epochs of 200 batches, patience 6): a batch of 1024 sees a
quarter of the windows per epoch that a batch of 4096 sees.

| Hyperparameter | Values |
|---|---|
| hidden units | `lstm`, `gru`: one recurrent layer of 32, 64, 128 or 256 units; `mlp`: 2 or 3 hidden layers of one width, 32, 64, 128 or 256 |
| history | 10, 20 or 30 samples |
| learning rate | log-uniform on [1e-4, 3e-3], three significant digits |
| weight decay (AdamW) | 0, 1e-5 or 1e-4 |
| batch size | 1024 or 4096 |
| `use_a_prev` | false or true |

No dropout: none of the three model classes has a dropout layer.

**Runs.** One `scripts/train.py` process per configuration (hydra overrides of `model.kwargs`
and `train.optim`) with the label `sweep_<model>_<kk>` and the results root
`paths.results=results/sweeps`:
`results/sweeps/models/<dataset>/<version>/<split>-fold<k>/sweep_<model>_<kk>/seed<s>/`. The
leaderboard and the milestone reports read `results/models` only and never see a sweep run. A
finished run (same configuration hash) is skipped; a failed run is repeated up to three times
with a timeout of 2 h per attempt and listed in `failures.json`.

**Criteria**, on the validation part of the split only:

* one step, `val_rmse_a`: `best_val_rmse_a` of `train.json`, the validation RMSE of the
  acceleration at the epoch the early stopping keeps;
* closed loop, `val_rmse_s_10s`: the spacing RMSE of the 10 s windows of P1
  (`evaluate_horizon` and `pooled_horizon` of `trafficloop/protocols/p1.py`) on the usable
  validation events, at most 2,000 drawn with the sampler of the training (on NGSIM fold 0 the
  2,000 events of the early stopping), in full float32 (TF32 off) as in the protocols; the 10 s
  collision rate and speed RMSE are stored next to it;
* for comparison, `val_rmse_a_p0`: P0 on the same events. The early stopping scores the first
  `val_batches x batch_size` windows of a fixed permutation, so `val_rmse_a` of a batch of 1024
  comes from a quarter of the windows of a batch of 4096; `val_rmse_a_p0` scores every
  configuration on the same instants.

The *one-step selection* is the validated configuration with the smallest `val_rmse_a`, the
*closed-loop selection* the one with the smallest `val_rmse_s_10s` (ties: the lower index); the
two may coincide. `scripts/evaluate.py` then runs P1, P1s, P2, P3 and P4 on the test part for
configuration 0 and both selections (`protocols=...` chooses others).

**Outputs** in `results/sweeps/<dataset>/<model>/`:

| File | Content |
|---|---|
| `configs.json` | the search space and the `n` configurations (index, label, hyperparameters); a sweep with another seed, data version, split or fold in the same directory is refused (`root=...` gives it its own) |
| `sweep.csv` | one row per configuration: hyperparameters, `val_rmse_a`, `val_rmse_s_10s`, `val_collision_rate_10s`, `val_rmse_a_p0`, `val_rmse_v_10s`, epochs run, fit seconds, parameters, status |
| `selection.json` | both selections with their criteria and hyperparameters, whether they coincide, Kendall's tau-b between the two criteria over the validated configurations, configuration 0 |
| `failures.json` | the failed runs of the last call of every stage |
| `REPORT.md` | the selections, every configuration, and for the evaluated runs P0 `rmse_a`, the P1 spacing RMSE of the 10 s windows and of the whole event, the in-range P2 collisions (`collisions_100_in_range`) and the P3 `unstable_fraction` (test part) |

Every validated run directory holds `sweep.json` (the criteria, their settings and hash, and
the configuration hash of the checkpoint), recomputed when the checkpoint or the settings
change. Further hydra overrides of every run may be appended (`train.optim.epochs=1 ...` for a
smoke test, with `root=` pointing elsewhere); overrides of the keys the sweep sets are refused.
