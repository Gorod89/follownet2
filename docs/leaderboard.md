# Leaderboard and report (milestone 5)

Binding for `follownet2/leaderboard/`, `follownet2/figures/`, `scripts/make_leaderboard.py`
and `scripts/make_report.py`. Runs and their files are described in `docs/experiments.md`
(section 1), the metrics in `docs/protocols.md`, the checklist in `docs/checklist.md`.

## 1. Entries

An *entry* is one run directory `results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/`
with `train.json` and the protocol files `p1.json` ... `p4.json` (`p1_<other>.json` for
transfer). The leaderboard of a dataset lists the entries of the driver split, fold 0, seed 0;
seeds 1-2 and the folds of the paired clean arms give the spread columns. An entry is
*admitted* only when its checklist is complete:

* every protocol file present has a `checklist` block;
* the fields `data_version`, `reconstruction_method`, `split_type`,
  `mechanistic_calibration_protocol`, `seeds` are non-empty in every block;
* `closed_loop_horizon` is non-empty in `p1.json`, `p2.json`, `p4.json` and every
  `p1_<other>.json`; `stability_test` in `p3.json`; `rare_event_suite` in `p4.json`;
* the data version of every block equals the version of the run, and the split type the
  split of the run;
* `train.json` carries the provenance (`config_hash`, `git_revision`) and the protocol files
  name the same model configuration hash (`model.provenance.config_hash`).

`follownet2/leaderboard/checklist.py`: `validate_entry(directory) -> list[str]` (problems;
empty when admitted). Refused entries are listed in a separate table with their problems;
they never appear in a ranking. Entries under the `event` split are labelled "driver leak"
and listed apart from the driver-split ranking (checklist rule 2).

## 2. Tables

Per dataset (`results/leaderboard/<dataset>.md`, plus `results/leaderboard/LEADERBOARD.md`
with every dataset and `results/leaderboard/index.html`, one self-contained page):

1. **Main table**, one row per admitted model, sorted by the composite (then by the
   whole-event P1 spacing error): `model, family, P0 ratio [low, high], P1 10 s RMSE_s
   [low, high], P1 30 s RMSE_s [low, high] (when the dataset has 30 s windows), P1 whole
   RMSE_s [low, high], P1 whole collision rate, P2 concave-growth error*, P2 hysteresis
   error*, P2 collisions_100*, P3 unstable fraction (and the 1 m/s fraction), P4
   collision rate synthetic* / empirical, composite, composite rank [low, high]`, where
   `*` marks the in-range P2/P4 values and `[low, high]` are 95 % intervals: driver
   bootstrap for P0/P1 (per-event tables, common random numbers across models), the
   profile-and-speed bootstrap of `composite_bootstrap` for the composite rank. The seed
   spread (min-max over seeds 0-2) of the P0 ratio and the whole-event P1 error are two
   further columns when the seeds exist.
2. **Transfer table** for the models with `p1_<other>.json`: relative degradation per
   target (whole event) and its mean.
3. **Refused entries** with the problems.
4. **Provenance footer**: data version, reconstruction method, split, seeds, git revision
   of the runs, protocol configuration hashes, date.

Intervals come from `follownet2.stats` (`bootstrap_groups`, `percentile_ci`,
`rank_distribution`) and `follownet2.protocols.composite` with 1000 replicates and seed 0;
the per-event tables are used only when they reproduce their JSON summary (as in
`follownet2/experiments/analysis.py`, which may be reused).

## 3. Rank-stability plot

`results/leaderboard/rank_stability_<dataset>.png` (and `.pdf`): for every admitted model
the composite rank on the x axis with its 95 % interval (dot and whiskers, models sorted by
the mean rank), and next to it the same for the whole-event P1 rank from the driver
bootstrap; families in distinct colours (baseline, mechanistic, network), a legend, the
dataset and the number of replicates in the title.

## 4. Figures of the report (`scripts/make_report.py` -> `results/figures/`)

PNG (200 dpi) and PDF, matplotlib, one colour-blind-safe palette for the three families
and one for datasets, 9 pt fonts, no title inside the axes beyond the panel label; every
function takes the data frames (from the CSVs of `results/m4/` or the run directories) and
returns the figure, so that the tests can run them on fabricated frames:

* `persistence_ratio_bars`: E1, one panel per dataset (`ngsim`, `follownet_highd`),
  horizontal bars of the P0 ratio on a log axis with the bootstrap interval, a vertical line
  at 1, models sorted by the ratio, family colours (`results/m4/e1_ratios.csv`);
* `split_inflation_bars`: E2, one panel per dataset (`ngsim`, `openacc`), grouped bars of
  the paired inflation (subset `affected`) for `p0_rmse_a`, `p1_h10_rmse_s`,
  `p1_whole_rmse_s` with their intervals, a zero line (`results/m4/e2_paired.csv`);
* `rank_bump_chart`: E3, one panel per dataset, the rank of every model under P0, P1 10 s,
  P1 whole and the composite as connected lines (bump chart), family colours, labels at
  both ends (`results/m4/e3_rankings.csv`);
* `concave_growth_curves`: P2, for one dataset and four models (`idm`, `ddpg`, `mlp_loop`,
  `lstm`) the speed standard deviation against the vehicle index (0 = leader) for the
  in-range OpenACC profiles (thin lines, one per profile) with the recorded followers of
  the same profiles as black markers (`p2.vehicles.parquet` and `p2.profiles.parquet` of
  the runs);
* `transfer_heatmap`: E4, one panel per model of the subset, rows trained-on, columns
  tested-on, the whole-event relative degradation with a diverging colour scale centred at
  1 (log scale, capped at 10) and the value printed in every cell
  (`results/m4/e4_transfer.csv`);
* `horizon_curves`: P1, one panel per dataset, the spacing error at 10 s, 30 s and the
  whole event per model (lines, family colours, log axis).

## 5. Scripts

* `scripts/make_leaderboard.py [datasets=a,b]`: validates every run directory, builds the
  tables, the rank-stability plots and the HTML page; prints the admitted and refused
  counts.
* `scripts/make_report.py`: builds every figure from the CSVs and the run directories and
  writes `results/figures/README.md` listing the files with one line each.

## 6. Tests

`tests/test_leaderboard.py`: an entry with a complete checklist is admitted, an entry with
an empty `stability_test` in `p3.json` or a different data version in a block is refused with
the field named, the `event` split is labelled; the tables are built from a fabricated
results root with two models and have the expected columns and order; the HTML page is
written. `tests/test_figures.py`: every figure function returns a figure and writes a file
on fabricated frames (matplotlib `Agg` backend).
