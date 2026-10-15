# Data layout

File and column layout of every source as the loaders rely on it, and the conventions of the
events that are built from them. Checked on 2026-09-27 and 2026-09-28. Every fact carries a
mark: verified on the real file, verified in the authors' notebook, code or documentation, or
assumed. The loaders of TGSIM, UTE, highD, exiD and CitySim were written from the published
format descriptions and are tested on fabricated files only: none of these sources is on
disk (see `docs/decisions.md`, D11 and D12).

## Conventions of the events

| Item | Convention |
|---|---|
| Units, sampling | SI, 10 Hz (`DT = 0.1` s); highD and exiD (25 Hz), CitySim (30 Hz) and UTE (24 to 60 Hz) are interpolated linearly |
| Relative speed | `dv = v - v_lead`, positive when closing in. FollowNet stores the opposite sign; its channel is used as a cross-check only |
| Gap | net gap `s = x_lead - x_follower`, leader rear bumper to follower front bumper |
| Acceleration | `a[k]` at `t[k]`; in the reconstructed version the acceleration applied over `[t[k], t[k+1]]` |
| Ids | `follower_id = "<dataset>/<site>/<raw id>"`, `event_id = "<follower_id>|<raw leader id>|<k>"` |
| Data version | `meta["data_version"]` of every event: `raw` or `reconstructed` |
| On disk | `data/events/<dataset>/<version>/`: `samples.parquet` (one row per sample: `event_id`, `t`, `s`, `dv`, `v`, `a`, `v_lead`, `x_lead`, `x_follower`), `events.parquet` (one row per event: ids, `n_samples`, `duration`, `data_version`, `meta_json`), `manifest.json` (configuration, hash, source files with SHA-256, counts). Same layout as cf-stability |

Extraction rules: `trafficloop/data/extraction.py` and `docs/decisions.md` (D1, D2, D2a). The
tables of real-data checks in parts A and B were obtained while the rules still required 151
samples and treated every negative speed as invalid; the numbers of the current rules are
in `results/m1/REPORT.md`.

## Sources

| Source | Loader | Kind | On disk | Part |
|---|---|---|---|---|
| FollowNet event sets | `follownet` | gap | HighD, NGSIM I-80, Waymo | A.1 |
| OpenACC | `openacc` | gap | six campaigns | A.2 |
| Waymo pairs of Hu et al. | `waymo_hu` | gap | version 2 | A.3 |
| NGSIM | `ngsim` | track | us-101, i-80 | B.1 |
| TGSIM | `tgsim` | track | no | B.2 |
| highD | `highd` | track | no (application) | B.3 |
| exiD | `exid` | track | no (application) | B.4 |
| UTE | `ute` | track | no | B.5 |
| CitySim | `citysim` | track | no (request form) | B.6 |

## Part A. Layout of the gap-based sources: FollowNet, OpenACC, Waymo pairs

What `trafficloop/data/loaders/follownet.py`, `openacc.py` and `waymo_hu.py` rely on, checked on
2026-09-27 against the files under `../cf-stability/data/raw/` (read-only, in memory) and the
authors' documentation.

Legend of the marks:

- **[file]** verified on the real file in this session.
- **[file, range]** verified on the real file by the HTTP range reads of the reconnaissance
  (`data_sources.md`); the file is not on disk here.
- **[notebook]** verified in the authors' Colab notebook "Dataset_description_FollowNet"
  (cells as extracted in the reconnaissance); **[paper]** in Chen et al., Sci Data 10:828 (2023).
- **[notes]** verified in the OpenACC campaign notes (`<campaign>/*_notes.txt`, on disk).
- **[code]** verified in `waymo_data_process.py` of Hu et al. (on disk, read, never executed).
- **[assumed]** not verified.

### A.1 FollowNet event sets

Figshare collection 10.6084/m9.figshare.c.6777810, CC0 [file, range]. On disk: the HighD, NGSIM
and Waymo event sets [file].

#### A.1.1 Files, shapes and dtypes

| option `subset` | event set file | split files `<stem>_<split>_data.npy` | dtype | shape of the event set | rate |
|---|---|---|---|---|---|
| `highd` | `HighD_car_fol_event_set.npy` | `HighD_{train,val,test}` (8,779 / 1,881 / 1,881) | `<f8` dense | (12541, 4, 375) [file] | 25 Hz |
| `ngsim` | `NGSIM_I_80_car_fol_event_set.npy` | `NGSIM_I_80_{...}` (1,351 / 290 / 289) | `\|O` | (1930, 4) [file] | 10 Hz |
| `waymo` | `Waymo_filter_car_fol_event_set.npy` | `Waymo_{...}` (1,008 / 216 / 216) | `\|O` | (1440, 4) [file] | 10 Hz |
| `lyft` | none published | `Lyft_{...}` (19,599 / 2,254 / 2,240) | `\|O` | - | 10 Hz |
| `spmd1` | `SPMD_das1_car_fol_event_set.npy` | `SPMD1_{...}` (11,661 / 2,498 / 2,499) | `\|O` | (16658, 4) [file, range] | 10 Hz |
| `spmd2` | `SPMD_das2_car_fol_event_set.npy` | `SPMD2_{...}` (16,973 / 3,637 / 3,637) | `\|O` | (24247, 4) [file, range] | 10 Hz |

- Split shapes and the Lyft, SPMD headers: [file, range]. The 30 s HighD set exists only as split
  files (924 / 198 / 197, 4, 750) at 25 Hz [file, range]; it has no loader subset.
- `.npy` format version 1.0, 128-byte header, C order [file].
- Object arrays: element `[i, c]` is a 1-D `float64` array; the four channels of an event have the
  same length; lengths differ between events: NGSIM 150 to 914 samples, Waymo 150 to 199 [file].
  HighD events all have 375 samples [file]. No NaN in any of the three files [file].
- The pickle payload (protocol 3) of the NGSIM and Waymo files names exactly three globals:
  `numpy.core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype` [file: `pickletools`
  opcode walk, nothing executed]. The SPMD and Lyft files come from the same writer and are
  assumed to name the same globals [assumed]; any other global is refused by the loader.
- Whether an event set is the concatenation of its split files: the counts add up
  [file, range], the contents were not compared [assumed].

#### A.1.2 Channel order (axis 1)

| index | content | unit |
|---|---|---|
| 0 | spacing | m |
| 1 | follower ("subject vehicle") speed | m/s |
| 2 | relative speed `v_leader - v_follower` | m/s |
| 3 | leader speed | m/s |

[notebook] cell 6: `# 4 dimension data= [spacing, subject_vehicle_speed, relative_speed, leading_vehicle_speed]`;
the figshare collection description gives the same list (read through the figshare API by the
reconnaissance).

#### A.1.3 Sign of the relative-speed channel

Channel 2 is leader minus follower, the opposite of the benchmark's `dv = v - v_lead`:

- [notebook] cell 14, IDM docstring: `:param relSpd: lead vehicle speed - following vehicle speed [m/s].`
- [notebook] cell 14, simulator update: `relSpd_ = lvSpd[i] - svSpd_` followed by
  `spacing_ = spacing + T * (relSpd_ + relSpd) / 2`: the spacing grows when the channel is
  positive, that is when the leader is faster.
- [file] `max |ch3 - ch1 - ch2|` is exactly 0.0 in every event of the HighD (12,541), NGSIM
  (1,930) and Waymo (1,440) event sets; `ch1 - ch3 - ch2` is not zero.
- Trap: Eq. (7) of the paper writes the IDM with follower minus leader [paper, per the
  reconnaissance]; the files do not follow it.

The loader never stores channel 2. It uses it only to cross-check the leader speed,
`|ch3 - ch1 - ch2| <= 1e-6` per sample; failing events are dropped and counted by
`follownet.last_report()` (none in the three files).

#### A.1.4 Sampling interval

- HighD 0.04 s: [notebook] cell 3 sets `Ts = 0.04`, its comment says the other datasets use 0.1;
  [paper] Table 1 gives 25 Hz; [file] a least-squares fit of `diff(spacing) = dt * mean(ch2)` gives
  a median of 0.0406 s over the 12,541 events. 375 samples span 14.96 s. The loader interpolates
  linearly to `t = 0, 0.1, ..., 14.9` (150 samples), which the extraction counts as 15 s.
- NGSIM and Waymo 0.1 s: [paper] Table 1; [file] fitted median 0.0994 s (NGSIM, interquartile
  range 0.0985 to 0.0998) and 0.1000 s (Waymo). Passed through unchanged.
- Lyft, SPMD1, SPMD2 0.1 s: [paper] Table 1; single-event fits 0.0957, 0.0978, 0.1067 s
  [file, range]; the loader assumes 10 Hz [assumed].

#### A.1.5 Units and value ranges

m and m/s: [notebook] plot labels and the IDM docstring; [file] plausible ranges:

| event set | spacing [m] | follower speed [m/s] | leader speed [m/s] |
|---|---|---|---|
| HighD | 1.6 to 223.0 | 0.004 to 35.9 | 0.004 to 35.7 |
| NGSIM | 3.5 to 125.9 | -0.13 to 31.1 | -0.13 to 26.6 |
| Waymo | -26.0 to 119.2 | -0.5 to 33.8 | -0.4 to 33.3 |

- Negative speeds [file]: NGSIM 2,272 of 743,222 follower samples in 163 events; Waymo 20,563 of
  283,903 in 688 events; HighD none. The loader does not clip them; the extraction sets speeds between -0.5 m/s and 0 to
  zero and cuts at lower ones.
- Waymo spacing is not positive in 67 samples [file].
- NGSIM has 3 and Waymo 2 events of exactly 150 samples (14.9 s) [file]: they pass, 15 s being
  150 samples.

#### A.1.6 Gap definition

- `waymo`: centre to centre [file]. 1,439 of the 1,440 events are whole pairs of Hu et al. (2022)
  (both speed channels equal `filter_speed` of `all_seg_paired_cf_trj_final_with_large_vehicle.csv`
  within 1e-9, starting at the first sample of the pair; 199 pairs with the automated vehicle
  following, 267 with it leading, 973 of two humans). In all 1,439 the spacing equals
  `filter_pos_leader - filter_pos_follower` to machine precision and differs from the net gap by
  3.7 m or more. The FollowNet file has no lengths, so the loader does not convert and writes
  `meta["gap_definition"] = "centre to centre as in Hu et al. (2022), not converted: vehicle lengths are not published"`.
  The same data are loaded with net gaps and source ids by `waymo_hu` (section 3).
- `highd`, `ngsim`, `lyft`, `spmd1`, `spmd2`: not documented [assumed: used as published],
  `meta["gap_definition"] = "as published, bumper or front-to-front not documented"`. Hints, not
  proofs [file]: HighD spacing goes down to 1.59 m (3,653 samples below 4 m in 59 events, 5 %
  quantile 3.45 m below 1 m/s), impossible for a front-to-front distance; NGSIM spacing below
  1 m/s has a median of 8.4 m and a 5 % quantile of 5.55 m, typical of a front-to-front distance.
  The notebook calls the HighD spacing `space_obs_dhw` (cell 17), i.e. highD's `dhw`, which the
  highD format document defines only as "Distance Headway".

#### A.1.7 What is absent

No event id, follower or leader id, recording or location id, time stamp, lane, vehicle length,
class or acceleration [file: the files hold only the four float channels]. Consequences in the
loader: pseudo ids from the position in the file (`<dataset>/<subset>/e<index:06d>` and
`l<index:06d>`, split files `train.e000012`), `meta["ids"] = "pseudo"`; every event is its own
group with `t` starting at 0; `site = subset`; a driver-level split is impossible because the same
real driver may be behind several pseudo ids.

### A.2 OpenACC

JRC, CC BY 4.0 [file: `copyright.txt`]. On disk: the six car-following campaigns, 139 run files
[file]. The single-vehicle campaign `One_vehicle_multiple_drivers_on_road_campaign` (columns
`Time,Speed,Engine Speed,Gear`, no leader) is not on disk [file, range]; `CAMPAIGNS` skips it.

#### A.2.1 Folders

| folder | site | runs | vehicles per run | `Driver<k>` columns [file] | `ACC` flag [file] |
|---|---|---|---|---|---|
| `AstaZero` | `astazero` | 10 | 5 | all vehicles | 0 (2 runs), 1 |
| `Casale` | `casale` | 9 | 2 | `Driver2` only | 1 |
| `Cherasco` | `cherasco` | 11 | 2 or 3 | all vehicles | 1, 2 |
| `JRC low speed` | `jrc_low_speed` | 24 | 2 or 3 | 20 files, not for every vehicle | blank (9), 1, 2 |
| `Vicolungo` | `vicolungo` | 12 | 5 | all vehicles | 0, 1, 2 |
| `ZalaZone` | `zalazone` | 73 | 4 to 11 | none | 1, 0 (6 handling runs) |

Every folder also holds `*_notes.txt`, a vehicle specification CSV and `copyright.txt`; ZalaZone
adds `Long setting vehs in mixed timegap experiment handling.csv` (run, vehicles with the long
time-gap setting in runs whose `Distance_setting` is `SL`) [file]. The loader skips every CSV
without a line whose first field is `Time`.

#### A.2.2 Run file

- Metadata lines before the header: 5 in all 139 files [file], padded with empty fields in
  AstaZero, Cherasco and Vicolungo, a trailing comma after the ZalaZone names [file]. The loader
  does not rely on the count: the header is the first line whose first field is `Time`.
  - `Date,D,M,YYYY` [file].
  - `Vehicle_order,<name 1>,...`: platoon order, position 1 is the leader [notes]; the order
    changes between runs of a campaign, the spelling of a car does not [file]. Names are
    normalised to lower case with spaces as underscores: `openacc/<site>/<name>`.
  - `Number_of_vehicles,n` equals the number of names [file].
  - `ACC`: 0 manual during the whole run, 1 ACC during the whole run, 2 mixed [notes]; blank when
    no follower has Driver information [notes, JRC low speed].
  - `Distance_setting`: min, max, none (AstaZero); min (Casale, Cherasco, Vicolungo); shortest,
    second shortest (JRC low speed); S, M, L, SL or blank (ZalaZone) [file].
- Columns used [notes: definitions; file: presence]:
  - `Time` [s], common to all vehicles; on the 0.1 s grid with every step exactly 0.1 s in all
    139 files [file]; the first value differs by file, from 0.1 to 385,857.3 s, with float
    artefacts such as `55287.09999998687` [file]. Kept, snapped to the grid.
  - `Speed<k>` [m/s], raw Doppler speed [notes]. The median ratio of `|d(E, N)/dt|` to `Speed` per
    vehicle and file lies within 0.986 to 1.027 in all 139 files [file]: m/s everywhere,
    ZalaZone included.
  - `IVS<k>` [m]: spacing between vehicles k and k+1 [notes: "IVS1 denotes IVS between vehicle 1
    and vehicle 2"], computed from GNSS "after bumper to bumper correction" [notes, all six
    campaigns; differential GNSS in AstaZero]. A net gap: not converted.
  - `Driver<k>`: exactly the tokens `ACC` and `Human` (1,056,824 and 486,704 cells), never empty
    [file].
- Where the flag and the Driver columns can be compared they agree: in `ACC,1` runs the
  followers' Driver columns hold only `ACC`, in `ACC,0` runs only `Human` [file: AstaZero,
  Casale, JRC low speed, Vicolungo]. The AstaZero leader is on ACC in manual runs as well [notes,
  file]. Followers without a Driver column take the flag (0 or 1) or stay unknown (2, blank);
  that the flag holds for them is [assumed] from the notes.
- Missing values are empty fields [file], e.g. 2 % to 68 % of `IVS1` per JRC low speed file.
  Absent vehicles have no columns at all (ZalaZone) and the indices keep the `Vehicle_order`
  positions: `IVS<k>` exists only when `Speed<k>` and `Speed<k+1>` exist [file, all 73 files].
- Negative IVS [file, as listed in the notes]: 6 Cherasco files (up to 621 samples), 2 Vicolungo
  files, 16 ZalaZone handling files. Extraction cuts them.
- Not used: `Lat`, `Lon` (radians; degrees in ZalaZone), `Alt`, `E`, `N`, `U`, `VE`, `VN`, `VU`;
  ZalaZone orders them `Speed, E, N, Lon, Lat, Alt` [file].

### A.3 Waymo car-following pairs, Hu et al. (2022)

Mendeley Data 10.17632/wfn2c3437n, version 2, CC BY 4.0 [file, range]. On disk: the five files
below and `SOURCES.json` [file].

| file | bytes | content |
|---|---|---|
| `all_seg_paired_cf_trj_final_with_large_vehicle.csv` | 52,456,295 | trajectories of the pairs |
| `AV_HV_and_HV_AV_CF_pair_record_with_large_vehicle.csv` | 13,257 | pairs with the automated vehicle |
| `HV_HV_pair_record_with_large_vehicle.csv` | 11,530 | pairs of two humans |
| `all_segment_all_vehicle_object_id_and_local_id_match.csv` | 1,835,702 | Waymo object id per local id (not used) |
| `waymo_data_process.py` | 93,288 | processing code (documentation) |

#### A.3.1 Trajectory file

- Columns `segment_id, local_veh_id, length, local_time, follower_id, leader_id, filter_pos,
  filter_speed, filter_accer`; `segment_id` and `local_veh_id` are written as floats with integral
  values [file]. 626,399 rows, 1,613 pairs in 460 segments [file].
- One row per vehicle of a pair and time stamp; every pair `(segment_id, follower_id, leader_id)`
  is one contiguous block, the leader rows first [file; code: `pd.concat([out_leader_trj_final,
  out_follower_trj_final])`]. The loader streams the file in chunks and fails on a split block.
- Local ids are unique within a segment only [code; file: id match file]. Id 0 is the automated
  vehicle: [code] the comment "AV's local id is always 0" (stage 2) and the mapping of `ego` to 0
  (stage 4); its length is fixed at 5.18 m [code; file].
- `local_time` [s] from the segment start, two decimals, at most 19.85 s [file]. Steps are 0.1 s
  except 302 of 0.11 s, 282 of 0.09 s and one of 0.2 s; 4.1 % of the stamps are off the 0.1 s
  grid [file]. The loader aligns follower and leader on their common stamps (one pair, segment
  269, has 199 follower and 198 leader samples [file]) and interpolates to the grid; the 0.2 s
  hole stays a hole.
- `filter_pos` [m]: distance travelled by the centre of the vehicle box after outlier removal
  and filtering; the leader's is offset by the Euclidean centre distance at the first common time
  stamp [code: `cumulated_dis_cal`, `pair_cf_coord_cal`]. `filter_pos` is the trapezoidal
  integral of `filter_speed` in at least 90 % of 400 vehicle series checked [file]. Hence
  `filter_pos_leader - filter_pos_follower` is centre to centre.
- Net gap: `net_distance = space_hwy - 0.5 * leader length - 0.5 * follower length` [code:
  `cf_trj_info_cal`]; the loader uses the same conversion with the published `length` [m],
  constant within a vehicle of a pair, 3.32 to 19.16 m [file].
- `filter_speed` [m/s] and `filter_accer` [m/s^2] [code; file]. Negative speeds: 45,664 of 626,399
  samples (7.3 %), 1,537 below -0.1 m/s (0.245 %, the share the version 3 description says it
  fixed) [file]. Net gap not positive in 886 samples, minimum -30.4 m [file]. Both left to the
  extraction.
- The code's `speed_diff` is follower minus leader [code]; it is not in the file and not used.

#### A.3.2 Pair records and automation

- AV record: one row per segment 1 to 1,000 with `follower` (the human behind the automated
  vehicle), `leader` (the human it follows), 0 for none, and `follower_code`, `leader_code`, which
  the code calls exclusion reason codes [code]. Non-zero ids carry code 0 or 3 only [file]; 3 is
  taken to be rule 3 of the code, bus or heavy truck, matching "with_large_vehicle" [assumed].
- HV record: 1,117 pairs of two humans, no id 0 [file].
- The code extracts exactly these pairs (`cf_pair_coord_trans`) [code], and the trajectory file
  matches the records one to one: 210 pairs with the automated vehicle following (non-zero
  `leader`), 286 with a human behind it (non-zero `follower`), 1,117 human pairs, none outside
  the records [file]. The loader decides from the records which vehicle is automated.
- There is no location in these files [file]; `location` exists only in the 2.1 GB
  `segment_*_all_frame_all_object_info.csv` files [file, range], which are not read. Site `all`.

### A.4 Loaders on the real files

Bounded runs of `load_pairs` plus `extract_all` with the extraction settings of each
`conf/dataset/<name>.yaml` (2026-09-27). "Pairs" are the yielded `PairSeries`; invalid samples
count either vehicle.

| config | part read | pairs | pairs without event | invalid samples | events | followers | duration q5 / q50 / q95 [s] | gap q5 / q50 / q95 [m] | speed q5 / q50 / q95 [m/s] |
|---|---|---|---|---|---|---|---|---|---|
| `follownet_ngsim` | all 1,930 events | 1,930 | 12 | 4,263 negative speed, 64 gap >= 120 m | 2,010 | 1,918 (pseudo) | 16.6 / 32.9 / 61.2 | 9.0 / 17.2 / 40.7 | 2.5 / 7.5 / 16.0 |
| `follownet_waymo` | all 1,440 events | 1,440 | 633 | 37,027 negative speed, 67 gap <= 0 | 807 | 807 (pseudo) | 16.2 / 19.7 / 19.8 | 8.6 / 20.3 / 51.6 | 1.1 / 8.2 / 22.9 |
| `follownet_highd` | first 3,000 events | 3,000 | 0 | none | 3,000 | 3,000 (pseudo) | 14.9 / 14.9 / 14.9 | 11.6 / 25.7 / 52.9 | 17.4 / 23.0 / 25.7 |
| `openacc` | 2 files per campaign | 33 | 0 | 735 empty, 297 gap <= 0, 1,148 gap >= 120 m | 529 | 19 | 45.3 / 58.2 / 59.5 | 12.9 / 31.1 / 68.2 | 10.6 / 21.6 / 33.8 |
| `waymo_hu` | all, human followers | 1,403 | 632 | 33,732 negative speed, 886 gap <= 0, 1 hole | 771 | 771 | 16.0 / 19.7 / 19.8 | 3.6 / 13.8 / 43.5 | 1.0 / 7.8 / 23.2 |

- FollowNet channel cross-check: 0 failures in all three event sets (HighD all 12,541 events).
- `waymo_hu` with `followers: automated`: 102 events of the one automated driver; `all`: 873
  events. Of the 771 human events, 168 follow the automated vehicle.
- OpenACC per campaign (events, ACC on / off / unknown): AstaZero 192, 0 / 192 / 0 (the two files
  read are the manual runs); Casale 11, 11 / 0 / 0; Cherasco 171, 163 / 8 / 0; JRC low speed 14,
  8 / 0 / 6; Vicolungo 52, 52 / 0 / 0; ZalaZone 89, 89 / 0 / 0 (file flag).

## Part B. Layout of the track-based sources: NGSIM, TGSIM, highD, exiD, UTE, CitySim

What `trafficloop/data/loaders/ngsim.py`, `tgsim.py`, `highd.py`, `exid.py`, `ute.py` and
`citysim.py` rely on, checked on 2026-09-27/28. Only the NGSIM files are on this machine
(`../cf-stability/data/raw/ngsim`, read-only); the other sources are described from their
metadata and documentation as collected by the reconnaissance (`data_sources.md`).

Legend of the marks:

- **[file]** verified on the real file in this session.
- **[metadata]** verified in the Socrata view metadata or data-set description (NGSIM, TGSIM).
- **[header]** verified on the first bytes of the real file (HTTP range read of the reconnaissance).
- **[doc]** from the format documentation of the authors (highD and exiD format PDFs, CitySim
  wiki); **[tools]** column names confirmed by the authors' Python tools.
- **[sources.yaml]** as written by the download tool of this repository.
- **[assumed]** not verified.

Common to all six loaders: the output axis increases in the driving direction, `x_front` and
`x_rear` are bumper positions in m, `v` in m/s, the tracks frame is put on the 10 Hz grid by
`resample_tracks` (linear interpolation, `lane` and `leader` from the nearest frame), vehicle ids
are strings where a source uses id 0, and every loader takes `dataset` (the name in the ids).

### B.1 NGSIM (Socrata `8ect-6jqj`), locations us-101 and i-80

Files, in this order of precedence per location:

- `raw_dir/ngsim_<location>_raw.parquet`: cf-stability, all 25 fields, int64/double/string [file];
  the download tool writes the same name with the 13 selected fields and nullable `Int64`
  integers, without `location` [sources.yaml]. Read without the pandas metadata.
- `raw_dir/*.csv`: Socrata exports with the field names (SODA) or the display names of the UI
  export (`Vehicle_ID, Frame_ID, ..., Location`) [metadata]; names matched case-insensitively,
  rows attributed by `location`.

| column | unit | meaning | used as |
|---|---|---|---|
| `vehicle_id` | - | id, unique inside an original 15-minute file, reused across the files [metadata, file] | `vehicle_id` |
| `frame_id` | - | frame of the original file, restarts in every file [file] | file of a row: rank of `global_time - 100 * frame_id` in its location; `group = "<location>-p<k>"` |
| `global_time` | ms since 1970 | time stamp, 100 ms grid (0 stamps off the grid in both locations) [file] | `t = (global_time - t0) / 1000`, `t0` first stamp of the location |
| `local_y` | ft | FRONT centre of the vehicle along the road [metadata] | `x_front = 0.3048 * local_y` |
| `v_length` | ft | vehicle length [metadata] | `x_rear = 0.3048 * (local_y - v_length)`, `length` |
| `v_vel` | ft/s | speed [metadata] | `v = 0.3048 * v_vel` |
| `v_acc` | ft/s^2 | acceleration, clipped at ±11.2 [metadata, file] | `a = 0.3048 * v_acc` (dataset acceleration kept) |
| `lane_id` | - | lane; us-101: 1-5 main line, 6 auxiliary, 7 on-ramp, 8 off-ramp [metadata]; i-80: 1-7 [file] | `lane`; follower lanes us-101 1-5, i-80 1-6 (i-80 lane 6/7 meaning [assumed]) |
| `preceding` | - | same-lane leader of the same file, 0: none [metadata, file] | `leader` |
| `v_class` | - | 1 motorcycle, 2 auto, 3 truck [metadata] | `vehicle_class` (name); followers: autos |
| `location` | - | location name [metadata] | CSV rows only |
| `space_headway` | ft | front centre to front centre [metadata] | real-data check only |

Checked on the real files [file]:

- us-101: 4,802,933 rows, 4,098,933 distinct (`vehicle_id`, `global_time`): 704,000 exact
  duplicate rows; i-80: none. No two rows of one file share (`vehicle_id`, `global_time`).
- `global_time - 100 * frame_id` takes 3 values per location, one per original file. The files
  overlap in time: us-101 0-15.9, 14.4-31.0, 29.6-46.2 min; i-80 0-16.7, 60.5-76.8,
  73.8-93.3 min. Every (file, `vehicle_id`) has exactly `total_frames` rows and no hole (6,101
  keys in us-101, 5,678 in i-80).
- Every `preceding > 0` is found in the same file at the same time stamp (0 of 3,915,011 follower
  samples missing in us-101, 0 of 4,359,292 in i-80), and `space_headway` equals the difference
  of `local_y` to 0.005 ft; in the events of us-101 file 0 (lanes 1-2)
  `s = 0.3048 * space_headway - leader_length` holds within 0.5 m for 100 % of the samples
  (largest deviation 1.5 mm).
- In the overlaps NGSIM matched leaders within each file only: a vehicle of the other file lies
  between follower and `preceding` in 0.22 % (us-101, 26 followers) and 0.31 % (i-80, 27
  followers) of the follower samples, and 17 (us-101) and 20 (i-80) vehicles are recorded in both
  files (a vehicle of the other file within 3 ft for at least 1 s, similar length and speed).
  The loader keeps these samples as they are.

### B.2 TGSIM (ITS DataHub), freeway data sets

Files: `raw_dir/<key>.parquet` of the download tool (keys `i90_94_stationary`,
`i90_94_moving`, `i294_l1`, `i294_l2`, `i395`; `id`, `lane_kf`, `run_index` and, in I-395,
`type_most_common` as `Int64`) [sources.yaml], otherwise the Socrata CSV export named in the
data-set description (`I90_94_stationary_final.csv`, `I90_94_moving_final.csv`,
`I294_L1_final.csv`, `I294_L2_final.csv`, `I395-final.csv`) [metadata]; option `files` names
the file. The I-90/I-94 Moving table displays `id` as `ID` [metadata]. Foggy Bottom (urban
network, `speed_kf_x/_y`) is not supported.

| column | unit | meaning [metadata] | used as |
|---|---|---|---|
| `id` | - | vehicle id; reused across runs; includes 0 in I-294 | `vehicle_id` (string) |
| `time` | s | since the start of the run, 0.1 s steps | `t` |
| `xloc_kf`, `yloc_kf` | m | Kalman-filtered position in the reference image of the run (origin top left, 1 px = 0.3 m); not along the road | projected on the axis of the data set, see below |
| `lane_kf` | - | lane; I-90/I-94 Stationary codes 1-6 and 10-15 by run, I-294 signed by direction (−5..−2 southbound, 2..5 northbound, undocumented 6), I-395 −6..−1 (−1 shoulder) | `lane` |
| `speed_kf` | m/s | speed (magnitude; I-395 capped at 30) | `v` |
| `acceleration_kf` | m/s^2 | acceleration (I-395 −20..15, I-294 L2 −22..20) | `a` (kept, raw version) |
| `length_smoothed` | m | bounding-box length (proxy of the vehicle length) | `length`; `x_front/x_rear = p ± length/2` |
| `type_most_common` | - | small-/large-vehicle; I-395: 1 car, 2 truck, 3 bus, 4 automated | `vehicle_class`; I-395 `automated` = code 4 |
| `av` (I-90/I-94), `acc` (I-294) | - | yes/no, any case (I-294 L1: No, Yes, YES) | `automated` of follower and leader |
| `run_index` | - | run (not in I-395) | `group` (I-395: one group) |

Axis per data set (option `axis`): I-395 `y` (x spans 74.9-105.6 m, y 132.0-649.3 m
[metadata]); I-294 L1 and L2 `x` [assumed from the ranges: x spans 5.2 km and 1.8 km, y 0.5 km
and 0.17 km]; I-90/I-94 Stationary and Moving `pca`, the principal axis of the run's positions
[assumed: both ranges are large, the road runs diagonally]. The direction is set per run and lane
from the sum of the vehicles' displacements.

Quirks: I-395 holds 26,649 (id, time) pairs with two rows, one looking like a sentinel (speed 0,
acceleration −20, length 15, type 2) [metadata, SoQL]: both rows are dropped and counted in the
log. I-294 L1 has 3,480 null `yloc_kf` [metadata] (unused with axis `x`).

Assumed: positions are vehicle centres; a straight axis measures the chord, so gaps on curved
segments are too short by the cosine of the angle between road and axis (the centre-line files of
the descriptions would fix it; where they are hosted is unknown); lanes of opposite directions
have different ids.

### B.3 highD (levelXdata, by application)

Files per recording `XX` (01-60): `XX_tracks.csv`, `XX_tracksMeta.csv`, `XX_recordingMeta.csv`
[doc, tools].

| file: column | unit | meaning [doc] | used as |
|---|---|---|---|
| recordingMeta: `id`, `frameRate`, `locationId` | -, Hz, - | recording, frame rate, location | `group` (two digits), `t`, `site` |
| tracksMeta: `id`, `class`, `drivingDirection` | - | Car/Truck; 1 = upper lanes towards decreasing x, 2 = lower lanes towards increasing x | `vehicle_class`; direction |
| tracks: `frame` | - | frame number | `t = (frame - 1) / frameRate` |
| tracks: `x`, `width` | m | upper-left corner of the bounding box; its extent along x = vehicle length | direction 2: `x_rear = x`, `x_front = x + width`; direction 1: `x_front = -x`, `x_rear = -(x + width)`; `length = width` |
| tracks: `xVelocity`, `xAcceleration` | m/s, m/s^2 | signed along x (negative on the upper lanes) | `v`, `a`, negated for direction 1 |
| tracks: `precedingId` | - | same-lane leader, 0: none | `leader` |
| tracks: `laneId` | - | lane, from 1 | `lane` |

Assumed: `frameRate` = 25 in all recordings (the loader reads it, it is not hard-coded); frames
start at 1; the tracks `width` is the post-processed length in every frame.

### B.4 exiD (levelXdata, by application)

Files per recording `XX`: `XX_tracks.csv`, `XX_tracksMeta.csv`, `XX_recordingMeta.csv` (the
file list of the PDF says `XX_recordingsMeta.csv`; both are accepted) [doc, tools].

| file: column | unit | meaning [doc] | used as |
|---|---|---|---|
| recordingMeta: `recordingId`, `locationId`, `frameRate` | -, -, Hz | | `group`, `site`, `t = frame / frameRate` |
| tracksMeta: `trackId`, `class` | - | | `vehicle_class` |
| tracks: `trackId`, `frame` | - | track ids ascending per recording (from 0 [assumed]) | `vehicle_id` (string) |
| tracks: `length` | m | object length (the PDF calls it "height"); 0 for VRUs | `length` |
| tracks: `lonVelocity`, `lonAcceleration` | m/s, m/s^2 | along the heading | `v`, `a` |
| enrichment: `traveledDistance` | m | distance covered by the track so far | mode leadDHW: own coordinate `p` |
| enrichment: `laneletId`, `lonLaneletPos` | -, m | semicolon lists, one entry per lanelet containing the centroid; position of the centroid along the lanelet centre line from its start | mode lonLaneletPos: first entry; `lane`, `x_front/x_rear = pos ± length/2` |
| enrichment: `laneChange` | - | set in the first frame in the new lane | mode leadDHW: `lane` = number of lane changes so far |
| enrichment: `leadId` | - | same-lane leader, −1: none | `leader` |
| enrichment: `leadDHW` | m | front bumper to the rear bumper of the leader, −1: none: the net gap | mode leadDHW: `x_front = x_rear(leader) - leadDHW` (identity conversion) |

The enrichment columns exist only "if part of the purchased dataset package" [doc]; whether the
academic release has them is not verified, and without them the loader raises. Assumed:
`frameRate` = 25; `laneChange` is 0/1 or boolean.

### B.5 UTE (Ubiquitous Traffic Eyes), sample data of eight sections

Files: `raw_dir/<site>/frenet.csv`; headers [header] per site (column order differs):

| site | columns (verbatim) | frame rate [doc: README; header: first rows] |
|---|---|---|
| SQM-W-1 | `vehicleID,frame,time(s),laneID,longitudinalDistance(m),speed(km/h),acceleration(m/s^2),lateralDistance(m),vehicleLength(pixel),vehicleWidth(pixel),leader_id,follower_id,leader_distance(m),follower_distance(m)` | 24 (steps 0.041701418 s) |
| SQM-W-2 | as SQM-W-1 with `vehicleLength(m),vehicleWidth(m)` | 24 (0.041666667 s) |
| YTDJ-3, XAM-N-5 | `vehicleID,laneID,time(s),lateralDistance(m),longitudinalDistance(m),speed(km/h),acceleration(m/s^2),vehicleLength(m),vehicleWidth(m)` | 24, 30 |
| XAM-N-6 | `frame,time(s),vehicleID,laneID,longitudinalDistance(m),lateralDistance(m),speed(km/h),acceleration(m/s^2),vehicleLength(m),vehicleWidth(m)` | 60 |
| RML-7, PKDD-8, XAM-S-9 | `frame,vehicleID,laneID,time(s),lateralDistance(m),longitudinalDistance(m),speed(km/h),acceleration(m/s^2),vehicleLength(m),vehicleWidth(m)` | 30 (PKDD-8 0.0333 s) |
| SQM-N-4 | `frenet-down.xlsx`, `frenet-up.xlsx`, not read: `NotImplementedError` | 30 |

Used: `vehicleID` → `vehicle_id` (string), `time(s)` → `t`, `longitudinalDistance(m)` → position
`p`, `speed(km/h)` / 3.6 → `v`, `acceleration(m/s^2)` → `a`, `laneID` → `lane`, `vehicleLength(m)`
→ `length` and `x_front/x_rear = p ± length/2`, `leader_id` (−1: none) → `leader` in SQM-W-1/2
(the other sites derive the leader). SQM-W-1 has lengths in pixels without a scale: positions are
centres and the gap is centre to centre (`meta["gap_definition"]`).

Assumed (the per-site `data_format.pdf` was not read): `longitudinalDistance` is the centre of
the vehicle on one Frenet axis shared by all lanes of a site; its direction relative to the
traffic (the loader negates it per lane when the vehicles move towards decreasing values); lanes
of opposite directions have different `laneID`; `acceleration` is the derivative of the speed;
the meaning of `leader_distance(m)` (unused).

### B.6 CitySim (UCF SST lab, by request form), freeway sites

Files: `raw_dir/<site>/<recording>.csv`; files of a site folder without `frameNum` and `carId`
(the metadata CSV `fileName, frameSizeX, frameSizeY, recordingFrameRate, ...`, signal timing) are
skipped [doc: wiki].

| column | unit | meaning [doc: wiki] | used as |
|---|---|---|---|
| `frameNum` | - | frame at 30 fps | `t = frameNum / frame_rate` (option, 30) |
| `carId` | - | vehicle id | `vehicle_id` (string) |
| `headXft`, `headYft`, `tailXft`, `tailYft` | ft | head and tail point (spelling [assumed] from "the same points with an ft suffix") | projected on the lane axis: `x_front`, `x_rear` (× 0.3048), `length` = their difference |
| `speed` | mph | speed | `v = 0.44704 * speed` |
| `laneId` | - | lane | `lane` |

Not used: the pixel points, `carCenter*`, `boundingBox1..4*`, `*Lat/*Lon` (US sites only),
`heading`, `course`. The lane axis is the principal axis of all head points of the lane in the
recording, oriented along the mean motion of its vehicles; positions next to a lane change are
blanked before the resampling (the lanes have different origins). Leaders are derived.

Assumed: the exact column spelling; the feet coordinates are one planar frame; lanes are
straight (curved lanes shorten the gaps to chords); lanes of opposite directions have different
`laneId`; every recording has 30 fps (`recordingFrameRate` of the metadata file is not read).
