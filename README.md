# TrafficLoop

A traffic-level evaluation protocol and benchmark for car-following models: one-step and
closed-loop accuracy against a persistence baseline, driver-, site- and regime-level splits,
platoon metrics, string stability, a rare-event suite and transfer between datasets.

Status: all five milestones are complete: data, splits, persistence and mechanistic
calibration (`results/m1/REPORT.md`), 25 reference entries (`results/m2/REPORT.md`,
`docs/models.md`; the certified hybrid `residual_idm_certified` added since), protocols
P1-P5 with the 1 s windows `p1s`, the human platoon reference `p2h`, the composite score and
the statistics (`docs/protocols.md`, `results/m3/REPORT.md`), experiments E1-E4 with E1b and
E3b-E3d and the sweep E5 (`docs/experiments.md`, `results/m4/REPORT.md`, `results/sweeps/`),
leaderboard and report (`docs/leaderboard.md`, `results/leaderboard/`, `results/figures/`,
`results/SUMMARY.md`).

## Conventions

| Item | Convention |
|---|---|
| Units, sampling | SI, 10 Hz |
| Relative speed | `dv = v - v_lead`, positive when closing in, everywhere |
| Gap | net gap `s = x_lead - x_follower` (leader rear bumper, follower front bumper) |
| Model interface | `acc = model(state_history)`, history `[..., H, 4]` with channels `(s, dv, v, a_prev)`, optional `predict_dist` |
| Integration | semi-implicit Euler, `v[k+1] = max(v[k] + a[k] dt, 0)`, `x[k+1] = x[k] + v[k+1] dt` |
| Closed loop | acceleration clipped to `[-8, 4]` m/s^2, collision is `s <= 0` |
| Warm-up | the first 3 s of every event are observed; all models are scored from sample 30 on |
| Data versions | `raw` and `reconstructed`; the tag is stored in every event |

## Layout

```
conf/                 hydra configuration, one file per dataset in conf/dataset/
trafficloop/data/      schema, extraction, reconstruction, regimes, splits, rare events, loaders
trafficloop/models/    baselines, mechanistic models, the learned models, registry and checkpoints
trafficloop/stability/ analytic string-stability criterion, a priori certificate of the certified hybrid
trafficloop/sim/       closed-loop rollout
trafficloop/calibration/  batched differential evolution, calibration protocol
trafficloop/protocols/ P0 and the closed-loop core of P1 (P1-P5: milestone 3)
trafficloop/train/     windows, trainer, training entry
scripts/              entry points
docs/                 data layout, licences, reconstruction, decisions, checklist
data/                 raw data, events, splits, calibration (not in git)
results/              result files with config hash and data version tag
```

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11, PyTorch 2.x (CUDA optional; every protocol runs on the CPU).

## Usage

```bash
python scripts/download_data.py --list          # sources, licences, sizes; downloads nothing
python scripts/m1.py dataset=ngsim               # events, splits, baselines, calibration
python scripts/m1.py dataset=ngsim "steps=[calibration]"   # one step; a repeated run resumes
python scripts/m1_report.py                      # results/m1/REPORT.md
python scripts/train.py dataset=ngsim model=lstm # train one model, score it on the test part
python scripts/m2.py dataset=ngsim               # all reference models, one process each
python scripts/m2_report.py                      # results/m2/REPORT.md
python scripts/evaluate.py dataset=ngsim model=lstm "protocols=[p1,p1s,p2,p2h,p3,p4]"  # protocols of one run
python scripts/reconstruct_episodes.py dataset=tgsim   # near-miss episodes for P4 (reconstructed, confirmed)
python scripts/build_human_profiles.py           # human platoon windows (NGSIM, TGSIM) of protocol p2h
python scripts/m3.py dataset=ngsim               # protocols of all trained models of a dataset
python scripts/m3_report.py                      # results/m3/REPORT.md
python scripts/make_paired_splits.py dataset=ngsim   # paired splits of E2 (clean and leaky arms)
python scripts/m4.py main                        # campaigns of milestone 4: main | seeds | folds | paired | event | transfer | ablation | protocols | recompute
python scripts/m4_report.py                      # results/m4/REPORT.md (E1-E4, E1b, E3b-E3d)
python scripts/sweep.py dataset=ngsim model=lstm n=16   # E5: hyperparameter sweep, two selection criteria
python -m pytest -q
```

Datasets scored in the benchmark: `ngsim`, `openacc`, `waymo_hu`, `follownet_highd`,
`follownet_ngsim`, `tgsim` (`docs/data_layout.md`, `docs/licences.md`). Loaders tested on
fabricated files only, pending access to the data: `ute`, `highd`, `exid`, `citysim`.

`scripts/m1.py` runs the four steps of milestone 1 for one dataset; they are also available
one by one:

| Step | Script | Output |
|---|---|---|
| events in both versions, near-miss episodes | `scripts/build_events.py` | `data/events/<dataset>/{raw,reconstructed}`, `data/events/<dataset>_nearmiss/raw` |
| splits and leakage check | `scripts/make_splits.py` | `data/splits/<dataset>_<kind>.json`, `<dataset>_report.json` |
| P0 baselines | `scripts/run_baselines.py` | `results/p0/...` |
| calibration of IDM, Gipps, Newell | `scripts/calibrate_mechanistic.py` | `data/calibration/<dataset>/<version>/`, `results/p1_whole_event/...` |

Downloads are never started implicitly: `scripts/download_data.py <source>` prints what it
would fetch and needs `--yes` to do it. highD, exiD and CitySim are available by application
only; their events must not be redistributed (`docs/licences.md`).

## Results and leaderboard

* Leaderboard: <https://gorod89.github.io/trafficloop/leaderboard/> (rebuilt by the workflow
  `.github/workflows/leaderboard.yml` on every push; sources in `results/leaderboard/`)
* `results/SUMMARY.md`: the findings in prose, with the caveats
* `results/figures/`: the figures of the report (`python scripts/make_report.py`)
* `results/m1/`, `results/m2/`, `results/m3/`, `results/m4/`: the milestone reports;
  `results/m4/` holds one CSV per table, E1b (`e1_horizons.csv`, `e1_horizon_taus.csv`) and
  E3d (`e3_human_reference.csv`, `e3_human_taus.csv`, `e3_human_top3.csv`) included
* `results/sweeps/<dataset>/<model>/`: E5 (`REPORT.md`, `sweep.csv`, `selection.json`); its
  runs are under `results/sweeps/models/` and never ranked
* `results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/`: one directory per
  run with `train.json` and the protocol summaries `p1.json` ... `p4.json`, `p1s.json` and
  `p2h.json` (JSON in the repository; the per-event tables, the checkpoints, the
  FollowNet-highD events, the near-miss episodes, the split files and the OpenACC leader
  profiles `data/scenarios/openacc_profiles.json` are release assets, see the release notes;
  the human platoon profiles `data/scenarios/human_profiles.json` join them in the next
  release)

An entry of the leaderboard is one run directory written by `scripts/train.py` and
`scripts/evaluate.py`; it is admitted only when every protocol file carries the complete
reporting checklist of `docs/checklist.md`. Entries are submitted as pull requests
(`docs/hosting.md`); the workflow lists refused entries with their problems and never ranks
them.

## Documents

* `docs/data_layout.md`: file and column layout of every source, as verified
* `docs/licences.md`: licence table
* `docs/reconstruction.md`: the physics-informed reconstruction
* `docs/loader_contract.md`: how to add a dataset
* `docs/checklist.md`: reporting checklist
* `docs/model_contract.md`, `docs/models.md`: model interface and the reference implementations
* `docs/protocols.md`: protocols P1-P5 with P1s and P2h, composite score, statistics
* `docs/experiments.md`: campaigns and analyses of E1-E5
* `docs/leaderboard.md`: entries, checklist gate, tables, figures
* `docs/decisions.md`: decisions and deviations from the specification
