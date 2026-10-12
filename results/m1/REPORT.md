# Milestone 1: key numbers

Datasets: waymo_hu, follownet_highd, follownet_ngsim, openacc, ngsim, tgsim

## 1. Events (raw version)

| dataset | events | followers | sites | hours | median duration [s] | free / steady / congested | events per follower | events of followers with several events |
|---|---|---|---|---|---|---|---|---|
| waymo_hu | 857 | 857 | 1 | 4.60 | 19.70 | 5 / 97 / 755 | 1.00 | 0.00 |
| follownet_highd | 12512 | 12512 | 1 | 51.79 | 14.90 | 486 / 4066 / 7960 | 1.00 | 0.00 |
| follownet_ngsim | 1915 | 1915 | 1 | 20.36 | 36.00 | 38 / 240 / 1637 | 1.00 | 0.00 |
| openacc | 4819 | 25 | 6 | 70.76 | 55.60 | 133 / 1046 / 3640 | 192.76 | 1.00 |
| ngsim | 14567 | 10971 | 2 | 192.67 | 41.30 | 77 / 841 / 13649 | 1.33 | 0.46 |
| tgsim | 20901 | 18031 | 5 | 235.49 | 32.30 | 756 / 2210 / 17935 | 1.16 | 0.25 |

## 2. Reconstruction (median change per event, 99 % quantile of the extremes)

| dataset | failed | gap change [m] | speed change [m/s] | acc. change [m/s2] | q99 max abs acc.: raw | rec. | q99 max abs jerk: raw | rec.  | gap bound active |
|---|---|---|---|---|---|---|---|---|---|
| waymo_hu | 0 | 0.01 | 0.02 | 0.02 | 3.48 | 3.32 | 4.95 | 4.04 | 0.00 |
| follownet_highd | 0 | 0.04 | 0.01 | 0.02 | 2.51 | 2.56 | 1.19 | 1.46 | 0.00 |
| follownet_ngsim | 0 | 0.02 | 0.01 | 0.10 | 4.31 | 4.58 | 5.90 | 6.91 | 0.00 |
| openacc | 0 | 0.07 | 0.08 | 0.04 | 4.10 | 4.13 | 4.65 | 4.59 | 0.00 |
| ngsim | 0 | 0.13 | 0.29 | 1.21 | 3.41 | 5.02 | 68.28 | 9.22 | 0.00 |
| tgsim | 0 | 0.01 | 0.07 | 0.47 | 18.24 | 5.05 | 204.70 | 10.79 | 0.00 |

## 3. Near-miss episodes

| dataset | episodes | followers | with cut-in | trigger brake | trigger drac | trigger ttc | median of the smallest TTC [s] | trigger cut_in |
|---|---|---|---|---|---|---|---|---|
| waymo_hu | 73 | 73 | 0 | 1 | 3.00 | 73.00 | 1.70 | 0.00 |
| follownet_highd | 6 | 6 | 0 | 6 | 0.00 | 0.00 | 6.27 | 0.00 |
| follownet_ngsim | 25 | 25 | 0 | 19 | 1.00 | 8.00 | 3.48 | 0.00 |
| openacc | 80 | 18 | 0 | 25 | 22.00 | 63.00 | 1.55 | 0.00 |
| ngsim | 9348 | 6077 | 2064 | 5494 | 311.00 | 4201.00 | 1.85 | 1304.00 |
| tgsim | 5326 | 4723 | 2763 | 872 | 261.00 | 2352.00 | 1.54 | 2434.00 |

## 4. Splits (fold 0)

| dataset | split | folds | train / val / test / unused | driver-disjoint | test followers seen in training | test events led by a training follower |
|---|---|---|---|---|---|---|
| waymo_hu | block | 5 | 436 / 101 / 171 / 149 | True | 0.000 | 0.000 |
| waymo_hu | driver | 5 | 556 / 130 / 171 / 0 | True | 0.000 | 0.269 |
| waymo_hu | event | 5 | 556 / 130 / 171 / 0 | False | 0.000 | 0.287 |
| waymo_hu | regime | 3 | 83 / 19 / 755 / 0 | True | 0.000 | 0.019 |
| waymo_hu | site | skipped: a site split needs at least 2 groups, found 1 |  |  |  |  |
| follownet_highd | block | skipped: block split needs block_seconds and the recording (group) of every event |  |  |  |  |
| follownet_highd | driver | 5 | 8132 / 1877 / 2503 / 0 | True | 0.000 | 0.000 |
| follownet_highd | event | 5 | 8132 / 1877 / 2503 / 0 | False | 0.000 | 0.000 |
| follownet_highd | regime | 3 | 3698 / 854 / 7960 / 0 | True | 0.000 | 0.000 |
| follownet_highd | site | skipped: a site split needs at least 2 groups, found 1 |  |  |  |  |
| follownet_ngsim | block | skipped: block split needs block_seconds and the recording (group) of every event |  |  |  |  |
| follownet_ngsim | driver | 5 | 1238 / 294 / 383 / 0 | True | 0.000 | 0.000 |
| follownet_ngsim | event | 5 | 1238 / 294 / 383 / 0 | False | 0.000 | 0.000 |
| follownet_ngsim | regime | 3 | 223 / 55 / 1637 / 0 | True | 0.000 | 0.000 |
| follownet_ngsim | site | skipped: a site split needs at least 2 groups, found 1 |  |  |  |  |
| openacc | block | 5 | 3118 / 746 / 955 / 0 | False | 1.000 | 0.831 |
| openacc | driver | 5 | 2958 / 905 / 956 / 0 | True | 0.000 | 0.803 |
| openacc | event | 5 | 3133 / 722 / 964 / 0 | False | 1.000 | 0.791 |
| openacc | regime | 3 | 921 / 258 / 3640 / 0 | False | 0.880 | 0.802 |
| openacc | site | 5 | 1571 / 398 / 2850 / 0 | True | 0.000 | 0.000 |
| ngsim | block | 5 | 9022 / 2036 / 3249 / 260 | True | 0.000 | 0.000 |
| ngsim | driver | 5 | 9461 / 2174 / 2932 / 0 | True | 0.000 | 0.754 |
| ngsim | event | 5 | 9480 / 2174 / 2913 / 0 | False | 0.391 | 0.822 |
| ngsim | regime | 3 | 546 / 130 / 13649 / 242 | True | 0.000 | 0.036 |
| ngsim | site | 2 | 5984 / 1390 / 7193 / 0 | True | 0.000 | 0.000 |
| tgsim | block | 5 | 13782 / 3174 / 3694 / 251 | True | 0.000 | 0.000 |
| tgsim | driver | 5 | 13507 / 3205 / 4189 / 0 | True | 0.000 | 0.762 |
| tgsim | event | 5 | 13567 / 3154 / 4180 / 0 | False | 0.211 | 0.781 |
| tgsim | regime | 3 | 1975 / 461 / 17935 / 530 | True | 0.000 | 0.074 |
| tgsim | site | 5 | 10807 / 2447 / 7647 / 0 | True | 0.000 | 0.000 |

## 5. P0 baselines on the test part (fold 0)

`floor` is the score of the recorded acceleration, i.e. the inconsistency of the record.
Largest deviation from the persistence identity over all rows: 1.02e-01 m/s^2.

| dataset | version | split | n_events | RMSE_a persistence | RMSE_a constant speed | ratio | RMSE_v persistence | RMSE_v floor | RMSE_s persistence | RMSE_s floor |
|---|---|---|---|---|---|---|---|---|---|---|
| follownet_highd | raw | driver | 2503 | 0.0165 | 0.3496 | 0.0472 | 0.0027 | 0.0014 | 0.0341 | 0.0341 |
| follownet_highd | raw | event | 2503 | 0.0165 | 0.3496 | 0.0472 | 0.0027 | 0.0014 | 0.0341 | 0.0341 |
| follownet_highd | raw | regime | 7960 | 0.0188 | 0.3924 | 0.0478 | 0.0031 | 0.0015 | 0.0442 | 0.0442 |
| follownet_highd | reconstructed | driver | 2503 | 0.0188 | 0.3570 | 0.0528 | 0.0019 | 0.0000 | 0.0002 | 0.0000 |
| follownet_highd | reconstructed | event | 2503 | 0.0188 | 0.3570 | 0.0528 | 0.0019 | 0.0000 | 0.0002 | 0.0000 |
| follownet_highd | reconstructed | regime | 7960 | 0.0213 | 0.4007 | 0.0532 | 0.0021 | 0.0000 | 0.0002 | 0.0000 |
| follownet_ngsim | raw | driver | 383 | 0.1140 | 0.7439 | 0.1533 | 0.0212 | 0.0117 | 0.0067 | 0.0065 |
| follownet_ngsim | raw | event | 383 | 0.1140 | 0.7439 | 0.1533 | 0.0212 | 0.0117 | 0.0067 | 0.0065 |
| follownet_ngsim | raw | regime | 1637 | 0.1130 | 0.7597 | 0.1488 | 0.0210 | 0.0116 | 0.0065 | 0.0063 |
| follownet_ngsim | reconstructed | driver | 383 | 0.1331 | 0.7967 | 0.1671 | 0.0133 | 0.0000 | 0.0013 | 0.0000 |
| follownet_ngsim | reconstructed | event | 383 | 0.1331 | 0.7967 | 0.1671 | 0.0133 | 0.0000 | 0.0013 | 0.0000 |
| follownet_ngsim | reconstructed | regime | 1637 | 0.1318 | 0.8108 | 0.1626 | 0.0132 | 0.0000 | 0.0013 | 0.0000 |
| ngsim | raw | block | 3249 | 1.0138 | 1.4212 | 0.7133 | 0.2359 | 0.1870 | 0.0194 | 0.0209 |
| ngsim | raw | driver | 2932 | 0.9902 | 1.4038 | 0.7054 | 0.2151 | 0.1641 | 0.0163 | 0.0179 |
| ngsim | raw | event | 2913 | 1.0019 | 1.4082 | 0.7115 | 0.2254 | 0.1760 | 0.0168 | 0.0184 |
| ngsim | raw | regime | 13649 | 0.9927 | 1.4028 | 0.7077 | 0.2179 | 0.1678 | 0.0167 | 0.0182 |
| ngsim | raw | site | 7193 | 0.8233 | 1.3850 | 0.5944 | 0.1250 | 0.0712 | 0.0073 | 0.0077 |
| ngsim | reconstructed | block | 3249 | 0.0908 | 0.7579 | 0.1198 | 0.0091 | 0.0000 | 0.0009 | 0.0000 |
| ngsim | reconstructed | driver | 2932 | 0.0892 | 0.7465 | 0.1195 | 0.0089 | 0.0000 | 0.0009 | 0.0000 |
| ngsim | reconstructed | event | 2913 | 0.0902 | 0.7520 | 0.1199 | 0.0090 | 0.0000 | 0.0009 | 0.0000 |
| ngsim | reconstructed | regime | 13649 | 0.0895 | 0.7493 | 0.1194 | 0.0089 | 0.0000 | 0.0009 | 0.0000 |
| ngsim | reconstructed | site | 7193 | 0.0818 | 0.7467 | 0.1095 | 0.0082 | 0.0000 | 0.0008 | 0.0000 |
| openacc | raw | block | 955 | 0.0290 | 0.4194 | 0.0692 | 0.0140 | 0.0134 | 0.0798 | 0.0798 |
| openacc | raw | driver | 956 | 0.0340 | 0.4642 | 0.0733 | 0.0064 | 0.0042 | 0.0781 | 0.0781 |
| openacc | raw | event | 964 | 0.0310 | 0.4472 | 0.0693 | 0.0148 | 0.0141 | 0.0883 | 0.0883 |
| openacc | raw | regime | 3640 | 0.0291 | 0.4378 | 0.0664 | 0.0151 | 0.0145 | 0.0341 | 0.0341 |
| openacc | raw | site | 2850 | 0.0296 | 0.3994 | 0.0741 | 0.0055 | 0.0032 | 0.0371 | 0.0370 |
| openacc | reconstructed | block | 955 | 0.0297 | 0.4424 | 0.0672 | 0.0030 | 0.0000 | 0.0003 | 0.0000 |
| openacc | reconstructed | driver | 956 | 0.0308 | 0.4833 | 0.0638 | 0.0031 | 0.0000 | 0.0003 | 0.0000 |
| openacc | reconstructed | event | 964 | 0.0321 | 0.4727 | 0.0680 | 0.0032 | 0.0000 | 0.0003 | 0.0000 |
| openacc | reconstructed | regime | 3640 | 0.0278 | 0.4468 | 0.0623 | 0.0028 | 0.0000 | 0.0003 | 0.0000 |
| openacc | reconstructed | site | 2850 | 0.0286 | 0.4072 | 0.0703 | 0.0029 | 0.0000 | 0.0003 | 0.0000 |
| tgsim | raw | block | 3694 | 1.2539 | 1.0714 | 1.1704 | 0.1426 | 0.1083 | 0.0177 | 0.0172 |
| tgsim | raw | driver | 4189 | 1.3755 | 1.0533 | 1.3059 | 0.1451 | 0.0928 | 0.0127 | 0.0129 |
| tgsim | raw | event | 4180 | 1.3978 | 1.0705 | 1.3058 | 0.1477 | 0.0913 | 0.0162 | 0.0163 |
| tgsim | raw | regime | 17935 | 1.3871 | 1.0499 | 1.3212 | 0.1470 | 0.0916 | 0.0136 | 0.0136 |
| tgsim | raw | site | 7647 | 1.6506 | 1.0620 | 1.5543 | 0.1650 | 0.0003 | 0.0120 | 0.0123 |
| tgsim | reconstructed | block | 3694 | 0.0758 | 0.5533 | 0.1370 | 0.0076 | 0.0000 | 0.0008 | 0.0000 |
| tgsim | reconstructed | driver | 4189 | 0.0599 | 0.5435 | 0.1103 | 0.0060 | 0.0000 | 0.0006 | 0.0000 |
| tgsim | reconstructed | event | 4180 | 0.0642 | 0.5579 | 0.1151 | 0.0064 | 0.0000 | 0.0006 | 0.0000 |
| tgsim | reconstructed | regime | 17935 | 0.0597 | 0.5399 | 0.1105 | 0.0060 | 0.0000 | 0.0006 | 0.0000 |
| tgsim | reconstructed | site | 7647 | 0.0619 | 0.5717 | 0.1082 | 0.0062 | 0.0000 | 0.0006 | 0.0000 |
| waymo_hu | raw | block | 171 | 0.0540 | 0.6125 | 0.0881 | 0.0081 | 0.0029 | 0.0032 | 0.0031 |
| waymo_hu | raw | driver | 171 | 0.0608 | 0.6700 | 0.0907 | 0.0091 | 0.0033 | 0.0035 | 0.0035 |
| waymo_hu | raw | event | 171 | 0.0588 | 0.6572 | 0.0894 | 0.0088 | 0.0033 | 0.0035 | 0.0034 |
| waymo_hu | raw | regime | 755 | 0.0593 | 0.6742 | 0.0879 | 0.0089 | 0.0032 | 0.0035 | 0.0034 |
| waymo_hu | reconstructed | block | 171 | 0.0497 | 0.6109 | 0.0814 | 0.0050 | 0.0000 | 0.0005 | 0.0000 |
| waymo_hu | reconstructed | driver | 171 | 0.0557 | 0.6681 | 0.0834 | 0.0055 | 0.0000 | 0.0006 | 0.0000 |
| waymo_hu | reconstructed | event | 171 | 0.0542 | 0.6549 | 0.0827 | 0.0054 | 0.0000 | 0.0005 | 0.0000 |
| waymo_hu | reconstructed | regime | 755 | 0.0545 | 0.6721 | 0.0811 | 0.0054 | 0.0000 | 0.0005 | 0.0000 |

## 6. Calibration per event: median errors, parameter spread

| dataset | version | model | events | RMSE_s | RMSE_v | collisions | median (coefficient of variation) | share at a bound |
|---|---|---|---|---|---|---|---|---|
| waymo_hu | raw | idm | 857 | 0.351 | 0.272 | 0.000 | T 0.91 (0.69), a 1.47 (0.72), b 3.79 (0.70), s0 3.39 (0.76), v0 24.50 (0.50) | b low 0.15, v0 low 0.11 |
| waymo_hu | raw | gipps | 857 | 0.317 | 0.269 | 0.000 | V 22.41 (0.47), a 1.24 (0.67), b 2.50 (0.77), b_hat 2.39 (0.80), s0 2.32 (0.92), tau 0.83 (0.75) | s0 low 0.10, tau low 0.15 |
| waymo_hu | raw | newell | 857 | 1.062 | 0.602 | 0.000 | V 20.93 (0.44), d 3.08 (0.90), tau 1.18 (0.57) |  |
| waymo_hu | reconstructed | idm | 857 | 0.351 | 0.271 | 0.000 | T 0.91 (0.69), a 1.46 (0.71), b 3.76 (0.71), s0 3.40 (0.77), v0 24.64 (0.50) | b low 0.15, v0 low 0.11 |
| waymo_hu | reconstructed | gipps | 857 | 0.323 | 0.271 | 0.000 | V 22.36 (0.46), a 1.22 (0.66), b 2.55 (0.78), b_hat 2.45 (0.80), s0 2.32 (0.92), tau 0.84 (0.74) | s0 low 0.11, tau low 0.15 |
| waymo_hu | reconstructed | newell | 857 | 1.066 | 0.606 | 0.000 | V 21.20 (0.44), d 3.11 (0.91), tau 1.18 (0.57) |  |
| follownet_highd | raw | idm | 1000 | 0.196 | 0.135 | 0.000 | T 0.52 (0.81), a 0.75 (0.99), b 5.93 (0.48), s0 5.94 (0.73), v0 34.64 (0.32) | T low 0.20, a low 0.17, b high 0.14 |
| follownet_highd | raw | gipps | 1000 | 0.158 | 0.129 | 0.000 | V 23.71 (0.29), a 0.98 (0.77), b 5.02 (0.49), b_hat 2.71 (0.67), s0 4.37 (0.73), tau 0.95 (0.76) | a low 0.17, tau low 0.15 |
| follownet_highd | raw | newell | 1000 | 0.533 | 0.370 | 0.000 | V 23.20 (0.25), d 3.50 (0.93), tau 0.84 (0.60) |  |
| follownet_highd | reconstructed | idm | 1000 | 0.167 | 0.140 | 0.000 | T 0.53 (0.80), a 0.77 (0.99), b 5.93 (0.48), s0 6.01 (0.74), v0 34.69 (0.32) | T low 0.20, a low 0.17, b high 0.15 |
| follownet_highd | reconstructed | gipps | 1000 | 0.140 | 0.130 | 0.000 | V 23.74 (0.28), a 1.02 (0.76), b 4.95 (0.50), b_hat 2.76 (0.66), s0 4.53 (0.73), tau 0.93 (0.77) | a low 0.16, tau low 0.16 |
| follownet_highd | reconstructed | newell | 1000 | 0.530 | 0.373 | 0.000 | V 23.23 (0.26), d 3.51 (0.95), tau 0.83 (0.61) |  |
| follownet_ngsim | raw | idm | 1000 | 1.009 | 0.569 | 0.000 | T 1.15 (0.59), a 1.42 (0.67), b 2.07 (0.79), s0 8.36 (0.44), v0 26.22 (0.50) | b low 0.15 |
| follownet_ngsim | raw | gipps | 1000 | 0.958 | 0.582 | 0.000 | V 19.02 (0.53), a 0.90 (0.73), b 2.30 (0.81), b_hat 2.16 (0.84), s0 5.63 (0.67), tau 1.06 (0.64) | a low 0.10 |
| follownet_ngsim | raw | newell | 1000 | 1.917 | 0.879 | 0.000 | V 13.99 (0.54), d 6.36 (0.70), tau 1.43 (0.51) | V low 0.17 |
| follownet_ngsim | reconstructed | idm | 1000 | 1.011 | 0.565 | 0.000 | T 1.15 (0.59), a 1.44 (0.67), b 2.05 (0.80), s0 8.43 (0.43), v0 26.62 (0.50) | b low 0.14 |
| follownet_ngsim | reconstructed | gipps | 1000 | 0.952 | 0.580 | 0.000 | V 18.65 (0.53), a 0.88 (0.73), b 2.36 (0.79), b_hat 2.38 (0.83), s0 5.78 (0.65), tau 1.04 (0.65) | a low 0.11 |
| follownet_ngsim | reconstructed | newell | 1000 | 1.914 | 0.886 | 0.000 | V 13.84 (0.55), d 6.44 (0.70), tau 1.46 (0.51) | V low 0.17 |
| openacc | raw | idm | 1000 | 1.036 | 0.292 | 0.000 | T 0.75 (0.65), a 1.05 (0.74), b 5.81 (0.54), s0 9.66 (0.44), v0 40.36 (0.31) | s0 high 0.11 |
| openacc | raw | gipps | 1000 | 0.907 | 0.302 | 0.000 | V 27.80 (0.39), a 1.33 (0.64), b 3.39 (0.67), b_hat 2.33 (0.80), s0 1.97 (0.96), tau 1.59 (0.49) |  |
| openacc | raw | newell | 1000 | 1.824 | 0.517 | 0.000 | V 25.35 (0.40), d 2.24 (1.06), tau 1.28 (0.49) |  |
| openacc | reconstructed | idm | 1000 | 0.999 | 0.288 | 0.000 | T 0.75 (0.65), a 1.03 (0.74), b 5.80 (0.55), s0 9.70 (0.44), v0 40.71 (0.32) | s0 high 0.11, v0 high 0.10 |
| openacc | reconstructed | gipps | 1000 | 0.886 | 0.291 | 0.000 | V 27.76 (0.39), a 1.35 (0.64), b 3.30 (0.67), b_hat 2.29 (0.81), s0 2.07 (0.96), tau 1.57 (0.50) |  |
| openacc | reconstructed | newell | 1000 | 1.821 | 0.513 | 0.000 | V 24.98 (0.40), d 1.86 (1.09), tau 1.33 (0.49) | d low 0.11 |
| ngsim | raw | idm | 1000 | 1.231 | 0.704 | 0.000 | T 0.98 (0.62), a 1.29 (0.69), b 2.89 (0.71), s0 3.38 (0.77), v0 22.09 (0.49) |  |
| ngsim | raw | gipps | 1000 | 1.173 | 0.735 | 0.000 | V 19.05 (0.50), a 1.09 (0.71), b 2.79 (0.72), b_hat 3.48 (0.73), s0 2.38 (0.90), tau 0.74 (0.71) |  |
| ngsim | raw | newell | 1000 | 2.094 | 1.076 | 0.000 | V 14.46 (0.52), d 2.52 (0.98), tau 1.24 (0.52) |  |
| ngsim | reconstructed | idm | 1000 | 1.200 | 0.585 | 0.000 | T 0.98 (0.62), a 1.22 (0.70), b 2.54 (0.75), s0 3.41 (0.76), v0 22.21 (0.49) | b low 0.10 |
| ngsim | reconstructed | gipps | 1000 | 1.136 | 0.614 | 0.000 | V 19.62 (0.50), a 1.03 (0.71), b 2.52 (0.76), b_hat 2.63 (0.78), s0 2.00 (0.95), tau 0.85 (0.68) |  |
| ngsim | reconstructed | newell | 1000 | 2.081 | 0.904 | 0.000 | V 14.92 (0.50), d 2.45 (1.01), tau 1.25 (0.52) |  |
| tgsim | raw | idm | 1000 | 0.771 | 0.339 | 0.000 | T 0.87 (0.71), a 0.94 (0.86), b 1.59 (0.83), s0 2.97 (0.86), v0 24.02 (0.51) | T low 0.12, b low 0.18, s0 low 0.11, v0 low 0.10 |
| tgsim | raw | gipps | 1000 | 0.749 | 0.372 | 0.000 | V 22.37 (0.47), a 0.96 (0.76), b 2.29 (0.81), b_hat 2.56 (0.81), s0 1.38 (1.07), tau 0.75 (0.78) | a low 0.14, b_hat low 0.12, s0 low 0.17, tau low 0.13 |
| tgsim | raw | newell | 1000 | 1.880 | 0.742 | 0.000 | V 17.18 (0.48), d 1.99 (1.11), tau 1.17 (0.61) | d low 0.17 |
| tgsim | reconstructed | idm | 1000 | 0.777 | 0.325 | 0.000 | T 0.88 (0.71), a 0.93 (0.86), b 1.61 (0.84), s0 3.07 (0.87), v0 23.83 (0.51) | T low 0.12, b low 0.18, s0 low 0.11, v0 low 0.11 |
| tgsim | reconstructed | gipps | 1000 | 0.751 | 0.354 | 0.000 | V 22.80 (0.47), a 1.04 (0.73), b 2.19 (0.82), b_hat 2.52 (0.82), s0 1.28 (1.07), tau 0.73 (0.79) | a low 0.15, b_hat low 0.12, s0 low 0.19, tau low 0.15 |
| tgsim | reconstructed | newell | 1000 | 1.876 | 0.731 | 0.000 | V 17.49 (0.48), d 1.93 (1.10), tau 1.15 (0.61) | d low 0.15 |

## 7. Global calibration on the training part, scored on the test part (whole event)

| dataset | version | model | split | train events | train RMSE_s | test RMSE_s | test RMSE_v | test collisions | parameters | at a bound |
|---|---|---|---|---|---|---|---|---|---|---|
| waymo_hu | raw | idm | driver | 556 | 5.315 | 5.514 | 0.905 | 0.000 | T 1.51, a 1.21, b 0.51, s0 1.47, v0 39.99 | b low |
| waymo_hu | raw | idm | event | 556 | 5.341 | 5.141 | 0.875 | 0.000 | T 1.41, a 1.32, b 0.51, s0 1.92, v0 43.48 | b low |
| waymo_hu | raw | gipps | driver | 556 | 5.131 | 5.283 | 0.942 | 0.000 | V 35.22, a 1.60, b 0.63, b_hat 0.63, s0 1.44, tau 1.14 |  |
| waymo_hu | raw | gipps | event | 556 | 5.128 | 4.831 | 0.920 | 0.000 | V 31.23, a 1.88, b 0.65, b_hat 0.65, s0 1.83, tau 1.09 |  |
| waymo_hu | raw | newell | driver | 556 | 10.930 | 12.483 | 3.088 | 0.029 | V 26.99, d 15.00, tau 0.47 | d high |
| waymo_hu | raw | newell | event | 556 | 10.197 | 10.106 | 2.912 | 0.041 | V 29.93, d 12.01, tau 0.58 |  |
| waymo_hu | reconstructed | idm | driver | 556 | 5.317 | 5.519 | 0.907 | 0.000 | T 1.54, a 1.17, b 0.50, s0 1.16, v0 44.41 | b low |
| waymo_hu | reconstructed | idm | event | 556 | 5.350 | 5.056 | 0.867 | 0.000 | T 1.48, a 1.29, b 0.51, s0 1.53, v0 43.85 | b low |
| waymo_hu | reconstructed | gipps | driver | 556 | 5.137 | 5.249 | 0.939 | 0.000 | V 33.06, a 1.74, b 0.62, b_hat 0.62, s0 1.42, tau 1.12 |  |
| waymo_hu | reconstructed | gipps | event | 556 | 5.128 | 4.936 | 0.934 | 0.000 | V 30.25, a 1.70, b 0.70, b_hat 0.70, s0 1.98, tau 0.94 |  |
| waymo_hu | reconstructed | newell | driver | 556 | 11.043 | 12.932 | 3.072 | 0.029 | V 26.19, d 15.00, tau 0.45 | d high |
| waymo_hu | reconstructed | newell | event | 556 | 10.212 | 10.126 | 2.905 | 0.041 | V 29.84, d 12.16, tau 0.57 |  |
| follownet_highd | raw | idm | driver | 2000 | 3.897 | 3.974 | 0.770 | 0.000 | T 0.60, a 0.28, b 0.51, s0 2.95, v0 30.64 | b low |
| follownet_highd | raw | idm | event | 2000 | 3.897 | 3.974 | 0.770 | 0.000 | T 0.60, a 0.28, b 0.51, s0 2.95, v0 30.64 | b low |
| follownet_highd | raw | gipps | driver | 2000 | 4.259 | 4.281 | 0.807 | 0.000 | V 26.04, a 0.50, b 0.56, b_hat 0.54, s0 4.87, tau 0.68 | a low, b low, b_hat low |
| follownet_highd | raw | gipps | event | 2000 | 4.259 | 4.281 | 0.807 | 0.000 | V 26.04, a 0.50, b 0.56, b_hat 0.54, s0 4.87, tau 0.68 | a low, b low, b_hat low |
| follownet_highd | raw | newell | driver | 2000 | 10.609 | 10.642 | 2.330 | 0.004 | V 23.44, d 11.37, tau 0.31 | tau low |
| follownet_highd | raw | newell | event | 2000 | 10.609 | 10.642 | 2.330 | 0.004 | V 23.44, d 11.37, tau 0.31 | tau low |
| follownet_highd | reconstructed | idm | driver | 2000 | 3.917 | 3.990 | 0.766 | 0.000 | T 0.68, a 0.30, b 0.50, s0 2.30, v0 31.50 | b low |
| follownet_highd | reconstructed | idm | event | 2000 | 3.917 | 3.990 | 0.766 | 0.000 | T 0.68, a 0.30, b 0.50, s0 2.30, v0 31.50 | b low |
| follownet_highd | reconstructed | gipps | driver | 2000 | 4.262 | 4.283 | 0.813 | 0.000 | V 26.11, a 0.51, b 0.57, b_hat 0.55, s0 6.93, tau 0.59 | a low, b low, b_hat low |
| follownet_highd | reconstructed | gipps | event | 2000 | 4.262 | 4.283 | 0.813 | 0.000 | V 26.11, a 0.51, b 0.57, b_hat 0.55, s0 6.93, tau 0.59 | a low, b low, b_hat low |
| follownet_highd | reconstructed | newell | driver | 2000 | 10.548 | 10.588 | 2.330 | 0.004 | V 23.47, d 11.36, tau 0.30 | tau low |
| follownet_highd | reconstructed | newell | event | 2000 | 10.548 | 10.588 | 2.330 | 0.004 | V 23.47, d 11.36, tau 0.30 | tau low |
| follownet_ngsim | raw | idm | driver | 1238 | 6.412 | 5.841 | 0.843 | 0.000 | T 1.80, a 1.01, b 0.53, s0 4.63, v0 24.76 | b low |
| follownet_ngsim | raw | idm | event | 1238 | 6.412 | 5.841 | 0.843 | 0.000 | T 1.80, a 1.01, b 0.53, s0 4.63, v0 24.76 | b low |
| follownet_ngsim | raw | gipps | driver | 1238 | 6.120 | 5.611 | 0.915 | 0.000 | V 22.25, a 1.51, b 0.53, b_hat 0.52, s0 4.07, tau 1.50 | b low, b_hat low |
| follownet_ngsim | raw | gipps | event | 1238 | 6.120 | 5.611 | 0.915 | 0.000 | V 22.25, a 1.51, b 0.53, b_hat 0.52, s0 4.07, tau 1.50 | b low, b_hat low |
| follownet_ngsim | raw | newell | driver | 1238 | 8.635 | 8.486 | 2.023 | 0.000 | V 17.37, d 13.36, tau 0.73 |  |
| follownet_ngsim | raw | newell | event | 1238 | 8.635 | 8.486 | 2.023 | 0.000 | V 17.37, d 13.36, tau 0.73 |  |
| follownet_ngsim | reconstructed | idm | driver | 1238 | 6.387 | 5.820 | 0.841 | 0.000 | T 1.75, a 0.94, b 0.50, s0 4.88, v0 28.87 | b low |
| follownet_ngsim | reconstructed | idm | event | 1238 | 6.387 | 5.820 | 0.841 | 0.000 | T 1.75, a 0.94, b 0.50, s0 4.88, v0 28.87 | b low |
| follownet_ngsim | reconstructed | gipps | driver | 1238 | 6.125 | 5.597 | 0.911 | 0.000 | V 23.86, a 1.31, b 0.51, b_hat 0.51, s0 4.32, tau 1.35 | b low, b_hat low |
| follownet_ngsim | reconstructed | gipps | event | 1238 | 6.125 | 5.597 | 0.911 | 0.000 | V 23.86, a 1.31, b 0.51, b_hat 0.51, s0 4.32, tau 1.35 | b low, b_hat low |
| follownet_ngsim | reconstructed | newell | driver | 1238 | 8.653 | 8.530 | 2.022 | 0.000 | V 17.25, d 13.35, tau 0.73 |  |
| follownet_ngsim | reconstructed | newell | event | 1238 | 8.653 | 8.530 | 2.022 | 0.000 | V 17.25, d 13.35, tau 0.73 |  |
| openacc | raw | idm | driver | 2000 | 9.719 | 11.236 | 1.039 | 0.000 | T 0.88, a 1.64, b 0.51, s0 9.51, v0 44.78 | v0 high, b low |
| openacc | raw | idm | event | 2000 | 10.130 | 10.560 | 0.971 | 0.000 | T 0.85, a 1.69, b 0.51, s0 9.70, v0 44.58 | b low |
| openacc | raw | gipps | driver | 2000 | 7.958 | 9.760 | 0.971 | 0.000 | V 39.18, a 1.95, b 0.53, b_hat 0.51, s0 4.54, tau 1.44 | b low, b_hat low |
| openacc | raw | gipps | event | 2000 | 8.392 | 8.506 | 0.886 | 0.002 | V 43.70, a 3.64, b 0.52, b_hat 0.50, s0 2.86, tau 1.45 | b low, b_hat low |
| openacc | raw | newell | driver | 2000 | 10.898 | 13.051 | 2.554 | 0.004 | V 36.95, d 12.78, tau 0.76 |  |
| openacc | raw | newell | event | 2000 | 12.346 | 12.189 | 2.421 | 0.004 | V 36.44, d 14.55, tau 0.63 |  |
| openacc | reconstructed | idm | driver | 2000 | 9.670 | 11.156 | 0.999 | 0.000 | T 0.89, a 1.49, b 0.51, s0 9.70, v0 44.66 | v0 high, b low |
| openacc | reconstructed | idm | event | 2000 | 10.117 | 10.414 | 0.956 | 0.000 | T 0.88, a 1.59, b 0.50, s0 9.69, v0 44.80 | v0 high, b low |
| openacc | reconstructed | gipps | driver | 2000 | 7.882 | 9.780 | 0.945 | 0.000 | V 40.08, a 2.38, b 0.54, b_hat 0.51, s0 4.80, tau 1.41 | b low, b_hat low |
| openacc | reconstructed | gipps | event | 2000 | 8.379 | 8.385 | 0.868 | 0.000 | V 44.10, a 2.26, b 0.53, b_hat 0.51, s0 4.04, tau 1.37 | b low, b_hat low |
| openacc | reconstructed | newell | driver | 2000 | 10.931 | 13.229 | 2.460 | 0.005 | V 36.57, d 13.43, tau 0.66 |  |
| openacc | reconstructed | newell | event | 2000 | 12.393 | 12.185 | 2.526 | 0.004 | V 36.50, d 14.33, tau 0.73 |  |
| ngsim | raw | idm | driver | 2000 | 5.801 | 6.430 | 0.994 | 0.000 | T 1.43, a 1.20, b 0.66, s0 2.67, v0 24.19 |  |
| ngsim | raw | idm | event | 2000 | 5.964 | 6.372 | 0.995 | 0.000 | T 1.50, a 1.04, b 0.52, s0 1.89, v0 26.86 | b low |
| ngsim | raw | gipps | driver | 2000 | 5.595 | 6.058 | 1.078 | 0.000 | V 24.12, a 2.58, b 0.62, b_hat 0.59, s0 1.49, tau 1.34 |  |
| ngsim | raw | gipps | event | 2000 | 5.782 | 6.100 | 1.081 | 0.000 | V 30.48, a 2.22, b 0.58, b_hat 0.57, s0 1.69, tau 1.28 | b_hat low |
| ngsim | raw | newell | driver | 2000 | 7.547 | 8.026 | 1.861 | 0.004 | V 16.53, d 7.05, tau 0.96 |  |
| ngsim | raw | newell | event | 2000 | 8.551 | 8.669 | 1.795 | 0.001 | V 15.24, d 10.05, tau 0.51 |  |
| ngsim | reconstructed | idm | driver | 2000 | 5.672 | 6.099 | 0.880 | 0.000 | T 1.43, a 1.05, b 0.53, s0 2.05, v0 23.86 | b low |
| ngsim | reconstructed | idm | event | 2000 | 5.912 | 6.145 | 0.871 | 0.000 | T 1.54, a 1.02, b 0.53, s0 1.68, v0 22.90 | b low |
| ngsim | reconstructed | gipps | driver | 2000 | 5.501 | 5.959 | 0.940 | 0.000 | V 25.98, a 1.53, b 0.58, b_hat 0.56, s0 1.40, tau 1.27 | b_hat low |
| ngsim | reconstructed | gipps | event | 2000 | 5.709 | 5.978 | 0.937 | 0.000 | V 21.86, a 1.93, b 0.52, b_hat 0.51, s0 2.04, tau 1.26 | b low, b_hat low |
| ngsim | reconstructed | newell | driver | 2000 | 7.616 | 8.052 | 1.723 | 0.003 | V 16.35, d 7.76, tau 0.84 |  |
| ngsim | reconstructed | newell | event | 2000 | 8.543 | 8.649 | 1.663 | 0.001 | V 15.29, d 10.03, tau 0.44 |  |
| tgsim | raw | idm | driver | 2000 | 8.891 | 7.888 | 0.906 | 0.000 | T 1.29, a 0.88, b 0.58, s0 3.05, v0 44.29 |  |
| tgsim | raw | idm | event | 2000 | 8.396 | 8.192 | 0.905 | 0.000 | T 1.35, a 0.87, b 0.51, s0 2.71, v0 43.87 | b low |
| tgsim | raw | gipps | driver | 2000 | 8.719 | 7.687 | 0.942 | 0.000 | V 37.39, a 1.47, b 0.72, b_hat 0.70, s0 3.08, tau 1.11 |  |
| tgsim | raw | gipps | event | 2000 | 8.053 | 8.253 | 0.955 | 0.001 | V 42.99, a 2.18, b 0.59, b_hat 0.56, s0 2.91, tau 1.32 | b_hat low |
| tgsim | raw | newell | driver | 2000 | 13.777 | 13.175 | 2.845 | 0.012 | V 33.17, d 14.40, tau 0.81 |  |
| tgsim | raw | newell | event | 2000 | 12.508 | 13.088 | 2.623 | 0.017 | V 31.46, d 12.29, tau 0.71 |  |
| tgsim | reconstructed | idm | driver | 2000 | 8.899 | 7.890 | 0.894 | 0.000 | T 1.32, a 0.93, b 0.57, s0 2.93, v0 44.46 |  |
| tgsim | reconstructed | idm | event | 2000 | 8.371 | 8.168 | 0.900 | 0.000 | T 1.31, a 0.81, b 0.51, s0 2.55, v0 44.39 | b low |
| tgsim | reconstructed | gipps | driver | 2000 | 8.705 | 7.680 | 0.931 | 0.000 | V 41.15, a 1.45, b 0.74, b_hat 0.72, s0 2.68, tau 1.15 |  |
| tgsim | reconstructed | gipps | event | 2000 | 8.030 | 8.192 | 0.939 | 0.001 | V 39.12, a 1.80, b 0.60, b_hat 0.57, s0 2.39, tau 1.40 | b_hat low |
| tgsim | reconstructed | newell | driver | 2000 | 13.375 | 12.730 | 2.805 | 0.014 | V 33.25, d 13.26, tau 0.84 |  |
| tgsim | reconstructed | newell | event | 2000 | 12.384 | 12.940 | 2.599 | 0.017 | V 31.47, d 11.79, tau 0.72 |  |

## 8. Test RMSE_s under the driver split relative to the event split

| dataset | version | model | driver | event | driver / event - 1 |
|---|---|---|---|---|---|
| follownet_highd | raw | gipps | 4.281 | 4.281 | 0.000 |
| follownet_highd | raw | idm | 3.974 | 3.974 | 0.000 |
| follownet_highd | raw | newell | 10.642 | 10.642 | 0.000 |
| follownet_highd | reconstructed | gipps | 4.283 | 4.283 | 0.000 |
| follownet_highd | reconstructed | idm | 3.990 | 3.990 | 0.000 |
| follownet_highd | reconstructed | newell | 10.588 | 10.588 | 0.000 |
| follownet_ngsim | raw | gipps | 5.611 | 5.611 | 0.000 |
| follownet_ngsim | raw | idm | 5.841 | 5.841 | 0.000 |
| follownet_ngsim | raw | newell | 8.486 | 8.486 | 0.000 |
| follownet_ngsim | reconstructed | gipps | 5.597 | 5.597 | 0.000 |
| follownet_ngsim | reconstructed | idm | 5.820 | 5.820 | 0.000 |
| follownet_ngsim | reconstructed | newell | 8.530 | 8.530 | 0.000 |
| ngsim | raw | gipps | 6.058 | 6.100 | -0.007 |
| ngsim | raw | idm | 6.430 | 6.372 | 0.009 |
| ngsim | raw | newell | 8.026 | 8.669 | -0.074 |
| ngsim | reconstructed | gipps | 5.959 | 5.978 | -0.003 |
| ngsim | reconstructed | idm | 6.099 | 6.145 | -0.007 |
| ngsim | reconstructed | newell | 8.052 | 8.649 | -0.069 |
| openacc | raw | gipps | 9.760 | 8.506 | 0.147 |
| openacc | raw | idm | 11.236 | 10.560 | 0.064 |
| openacc | raw | newell | 13.051 | 12.189 | 0.071 |
| openacc | reconstructed | gipps | 9.780 | 8.385 | 0.166 |
| openacc | reconstructed | idm | 11.156 | 10.414 | 0.071 |
| openacc | reconstructed | newell | 13.229 | 12.185 | 0.086 |
| tgsim | raw | gipps | 7.687 | 8.253 | -0.069 |
| tgsim | raw | idm | 7.888 | 8.192 | -0.037 |
| tgsim | raw | newell | 13.175 | 13.088 | 0.007 |
| tgsim | reconstructed | gipps | 7.680 | 8.192 | -0.062 |
| tgsim | reconstructed | idm | 7.890 | 8.168 | -0.034 |
| tgsim | reconstructed | newell | 12.730 | 12.940 | -0.016 |
| waymo_hu | raw | gipps | 5.283 | 4.831 | 0.094 |
| waymo_hu | raw | idm | 5.514 | 5.141 | 0.072 |
| waymo_hu | raw | newell | 12.483 | 10.106 | 0.235 |
| waymo_hu | reconstructed | gipps | 5.249 | 4.936 | 0.063 |
| waymo_hu | reconstructed | idm | 5.519 | 5.056 | 0.091 |
| waymo_hu | reconstructed | newell | 12.932 | 10.126 | 0.277 |
