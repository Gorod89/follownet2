# Reporting checklist

Every protocol result file of a run
(`results/models/<dataset>/<version>/<split>-fold<k>/<label>/seed<s>/<protocol>.json`,
written by `scripts/evaluate.py`) carries a `checklist` block. The leaderboard (milestone 5)
refuses an entry whose block has an empty field that applies to its protocol.

| Field | What to state | Allowed values / example | Needed for |
|---|---|---|---|
| `data_version` | which version of the data was scored | `raw`, `reconstructed` | all |
| `reconstruction_method` | method tag of the reconstruction, `none` for raw data | `qp-semi-implicit-v1` | all |
| `split_type` | kind of split, with fold and seed in `split` | `event`, `driver`, `site`, `regime`, `block` | all |
| `mechanistic_calibration_protocol` | objective, measure of performance, optimiser, data the parameters come from | "closed loop, NRMSE(s)+NRMSE(v), differential evolution, pooled parameters from the training part" | mechanistic models, and every table that compares against them |
| `closed_loop_horizon` | rollout horizon and warm-up | `10 s, 30 s, whole event; warm-up 3 s`; `hold 60 s then the leader profile` | P1, P2, P4, P5 |
| `stability_test` | criterion and grid | "analytic criterion (memoryless models) and step-response gain on 0.002-2 Hz, delta 0.05 m/s, v_e 5..30 m/s" | P3, composite score |
| `rare_event_suite` | version of the scenario suite and source of the empirical episodes | `{"synthetic": "p4-v1", "episodes": ["tgsim_nearmiss/reconstructed"]}` | P4 |
| `seeds` | all seeds: global, training, split (bootstrap seeds are stated in the report) | `{"global": 0, "train": 0, "split": 0}` | all |

Rules

1. One result file is one model on one dataset, one data version, one split and one fold.
   Numbers obtained on different data versions or splits are never averaged.
2. Results under the `event` split are labelled "driver leak" in every table.
3. A model that was trained or calibrated on data outside the training part of the split
   (pre-training, published parameters) says so in `extra.training_data`.
4. Every table that ranks models contains `persistence`.
5. `config_hash` identifies the protocol configuration (and `model.provenance.config_hash` the
   training configuration); the configuration itself is stored by hydra next to the run
   (`outputs/<date>/<time>/.hydra/config.yaml`).
