# Loader contract

Binding for every module in `trafficloop/data/loaders/`. Read `trafficloop/data/extraction.py`
and `trafficloop/data/schema.py` first: loaders only translate a source format into the two
intermediate representations defined there. Filtering, segmentation, acceleration estimates,
regimes, reconstruction and splits are done downstream and must not be duplicated in a loader.

## 1. Interface

```python
POSITION_SOURCE = "track"   # or "gap"

def load_pairs(raw_dir: Path, **options) -> Iterator[PairSeries]: ...
```

* `raw_dir` is read-only. A loader never downloads and never writes into `raw_dir`.
* `load_pairs` is a generator. Memory must stay bounded by one *group* (recording, run,
  15-minute period, file), never by the whole dataset.
* Missing input raises `FileNotFoundError` with the expected layout in the message.
* `options` are the keys of `conf/dataset/<name>.yaml: options` (sites, lanes, vehicle
  classes, file patterns, `max_groups` for quick runs). Every option has a default.
* Track-based sources (`POSITION_SOURCE = "track"`: NGSIM, TGSIM, highD, exiD, UTE, CitySim)
  build the standard *tracks frame* per group and yield from `tracks_to_pairs(...)`.
* Gap-based sources (`POSITION_SOURCE = "gap"`: FollowNet, OpenACC, Waymo pairs) build
  `PairSeries` directly, positions from `positions_from_gap(v_lead, s)`.

## 2. Conventions

| Item | Convention |
|---|---|
| Units | SI. Feet, mph, km/h, pixels are converted in the loader, with the factor in a named constant. |
| Time | seconds; after the loader every series is on the 10 Hz grid `t = g * 0.1` (`resample_tracks` does it for track sources; gap sources at another rate interpolate linearly themselves). |
| Longitudinal axis | increases in the driving direction. Sources with two driving directions flip the axis (and the sign of speed and acceleration) for the direction that runs towards decreasing coordinates. |
| `x_front`, `x_rear` | bumper positions of a vehicle: `x_rear = x_front - length`. |
| `x_follower`, `x_lead` | follower FRONT bumper, leader REAR bumper, so the net gap is `s = x_lead - x_follower`. A source that reports a front-to-front or centre-to-centre distance is converted with the vehicle lengths; if the lengths are unknown the fact is written to `meta["gap_definition"]`. |
| Relative speed | never stored by a loader. Downstream computes `dv = v - v_lead`. A source column with the opposite sign (FollowNet) is only used to cross-check the leader speed. |
| Speed | `v >= 0` along the driving direction, as measured. No smoothing in loaders. |
| Acceleration | column `a` only if the source provides one; otherwise absent (downstream differentiates). |
| Ids | `follower_id = "<dataset>/<site>/<raw id>"`, globally unique; `tracks_to_pairs` builds them from `dataset`, `site`, `group`, `vehicle_id`. `vehicle_id` must be unique within a group. |
| Site | the unit of the site-level split: NGSIM location, highD `locationId`, OpenACC campaign, TGSIM data set. |
| Group | the unit inside which vehicle ids and time are consistent: highD recording, TGSIM `run_index`, NGSIM 15-minute period, OpenACC file. |
| Segment key | `PairSeries.segment_key`: per-sample label whose change ends an event (ACC on/off). |
| Meta | `PairSeries.meta` is copied into every event of the follower. Required keys: `source_file` (relative to `raw_dir`), `native_rate_hz`. Use when known: `follower_length`, `leader_length` (per-sample in `extras`), `vehicle_class`, `automated` (bool), `acc_on` (bool), `ids` (`"source"` or `"pseudo"`). |

## 3. Tracks frame

One row per vehicle and time stamp, columns:

| Column | Required | Meaning |
|---|---|---|
| `group` | yes | see above |
| `vehicle_id` | yes | raw id, unique within the group |
| `t` | yes | s |
| `x_front`, `x_rear` | yes | m, along the driving direction |
| `v` | yes | m/s, >= 0 |
| `site` | no | default: the group |
| `lane` | no | integer lane id; a lane change of follower or leader ends an event |
| `a` | no | m/s^2, only if provided by the source |
| `leader` | no | raw id of the leader, missing / 0 / negative without leader; when the column is absent the leader is the next vehicle ahead in the same lane (`derive_leaders`) |
| `length` | no | m; gives `follower_length` and `leader_length` |
| others | no | named in `extras=` (follower, goes to meta) or `leader_extras=` (leader, goes to meta as `leader_<name>`) |

## 4. Tests

Every loader has tests in `tests/test_loader_<name>.py` that

1. write a *fabricated* file in the exact source format (header lines, column names, units,
   dtypes, quirks such as duplicates or id reuse) into `tmp_path`, with known kinematics;
2. run `extract_all(load_pairs(tmp_path), ExtractionConfig())` and check the expected events
   (count, ids, lengths) and known values (gap in metres, speeds, signs);
3. call `Event.validate()` on every event;
4. cover every quirk the loader handles, one test per quirk;
5. are marked `@pytest.mark.data` and skipped when the real files are absent if they read real
   data; such tests read a small part of the files only.

Tests must run in seconds on a CPU and write nothing outside `tmp_path`.
