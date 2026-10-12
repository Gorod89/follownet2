# Milestone 2: reference models

One-step scores (P0) and closed-loop scores over whole events on the test part. `ratio` is the one-step acceleration error relative to the persistence baseline of the same data; `rank` orders the models of a block by the closed-loop spacing error.

## follownet_highd, reconstructed, split driver-0 (2503 test events)

| model | family | parameters | minutes | P0 RMSE_a | ratio | P0 RMSE_v | loop RMSE_s | loop RMSE_v | collisions | rank |
|---|---|---|---|---|---|---|---|---|---|---|
| persistence | baseline | 0 | 0.000 | 0.019 | 1.000 | 0.002 | 8.939 | 2.172 | 0.077 | 20 |
| constant_speed | baseline | 0 | 0.000 | 0.357 | 18.954 | 0.036 | 9.179 | 1.863 | 0.133 | 21 |
| idm | mechanistic | 0 | 0.053 | 0.938 | 49.786 | 0.094 | 4.005 | 0.762 | 0.000 | 12 |
| gipps | mechanistic | 0 | 0.160 | 3.089 | 164.014 | 0.309 | 4.317 | 0.819 | 0.000 | 15 |
| gipps_median | mechanistic | 0 | 0.365 | 2.987 | 158.597 | 0.299 | 7.311 | 1.277 | 0.004 | 17 |
| idm_median | mechanistic | 0 | 0.223 | 0.749 | 39.747 | 0.075 | 7.463 | 1.267 | 0.000 | 18 |
| newell | mechanistic | 0 | 0.162 | 5.609 | 297.790 | 0.561 | 10.561 | 2.316 | 0.004 | 22 |
| newell_median | mechanistic | 0 | 0.288 | 6.049 | 321.158 | 0.605 | 13.589 | 2.711 | 0.006 | 24 |
| seq2seq | network | 34881 | 1.710 | 0.120 | 6.391 | 0.012 | 3.093 | 0.658 | 0.002 | 1 |
| mlp_loop | network | 6209 | 3.237 | 0.055 | 2.919 | 0.005 | 3.214 | 0.680 | 0.004 | 2 |
| diffusion | network | 42433 | 1.460 | 0.075 | 3.991 | 0.008 | 3.361 | 0.703 | 0.006 | 3 |
| gru | network | 13313 | 0.700 | 0.089 | 4.735 | 0.009 | 3.377 | 0.722 | 0.002 | 4 |
| gru_loop | network | 13313 | 4.035 | 0.111 | 5.885 | 0.011 | 3.397 | 0.714 | 0.004 | 5 |
| lstm | network | 17729 | 0.868 | 0.098 | 5.188 | 0.010 | 3.482 | 0.739 | 0.002 | 6 |
| perl | network | 17729 | 0.743 | 0.093 | 4.922 | 0.009 | 3.516 | 0.746 | 0.000 | 7 |
| lstm_loop | network | 17729 | 4.188 | 0.119 | 6.321 | 0.012 | 3.532 | 0.741 | 0.003 | 8 |
| mdn_lstm | network | 18639 | 0.867 | 0.140 | 7.412 | 0.014 | 3.731 | 0.758 | 0.008 | 9 |
| gail | network | 13314 | 0.578 | 0.357 | 18.935 | 0.036 | 3.772 | 0.723 | 0.003 | 10 |
| stability_mlp | network | 4481 | 3.860 | 0.288 | 15.284 | 0.029 | 3.928 | 0.759 | 0.005 | 11 |
| maml_lstm | network | 17729 | 23.943 | 0.244 | 12.957 | 0.024 | 4.022 | 0.893 | 0.005 | 13 |
| knn | network | 0 | 0.028 | 0.304 | 16.142 | 0.030 | 4.090 | 0.793 | 0.008 | 14 |
| mlp | network | 6209 | 0.377 | 0.050 | 2.669 | 0.005 | 4.579 | 0.980 | 0.010 | 16 |
| pidl | network | 4486 | 0.098 | 0.352 | 18.675 | 0.035 | 8.892 | 1.800 | 0.129 | 19 |
| ddpg | network | 4481 | 2.053 | 2.466 | 130.931 | 0.247 | 11.097 | 1.730 | 0.000 | 23 |

* Kendall tau between the one-step and the closed-loop ranking, all models: 0.52 (p = 0.000, n = 24)
* Kendall tau between the one-step and the closed-loop ranking, learned models only: 0.53 (p = 0.003, n = 16)

## ngsim, reconstructed, split driver-0 (2932 test events)

| model | family | parameters | minutes | P0 RMSE_a | ratio | P0 RMSE_v | loop RMSE_s | loop RMSE_v | collisions | rank |
|---|---|---|---|---|---|---|---|---|---|---|
| constant_speed | baseline | 0 | 0.000 | 0.747 | 8.365 | 0.075 | 118.790 | 4.014 | 0.560 | 20 |
| persistence | baseline | 0 | 0.000 | 0.089 | 1.000 | 0.009 | 827.895 | 21.662 | 0.529 | 24 |
| gipps | mechanistic | 0 | 8.043 | 4.842 | 54.254 | 0.484 | 5.990 | 0.932 | 0.000 | 3 |
| idm | mechanistic | 0 | 1.882 | 2.294 | 25.702 | 0.228 | 6.107 | 0.879 | 0.000 | 4 |
| gipps_median | mechanistic | 0 | 21.502 | 5.556 | 62.259 | 0.556 | 6.719 | 0.969 | 0.000 | 6 |
| idm_median | mechanistic | 0 | 7.630 | 1.710 | 19.159 | 0.166 | 6.790 | 0.922 | 0.000 | 7 |
| newell | mechanistic | 0 | 4.955 | 6.701 | 75.093 | 0.670 | 7.992 | 1.746 | 0.004 | 8 |
| newell_median | mechanistic | 0 | 12.625 | 5.941 | 66.578 | 0.594 | 8.422 | 1.602 | 0.016 | 10 |
| mlp_loop | network | 6209 | 14.538 | 0.143 | 1.603 | 0.014 | 5.538 | 0.832 | 0.013 | 1 |
| gru_loop | network | 13313 | 10.058 | 0.173 | 1.942 | 0.017 | 5.979 | 0.911 | 0.024 | 2 |
| lstm_loop | network | 17729 | 10.357 | 0.206 | 2.309 | 0.021 | 6.521 | 0.896 | 0.015 | 5 |
| ddpg | network | 4481 | 5.747 | 2.544 | 28.505 | 0.254 | 8.128 | 1.061 | 0.000 | 9 |
| gail | network | 13314 | 1.720 | 0.643 | 7.203 | 0.064 | 8.539 | 0.989 | 0.445 | 11 |
| mlp | network | 6209 | 1.382 | 0.077 | 0.867 | 0.008 | 9.810 | 1.268 | 0.323 | 12 |
| stability_mlp | network | 4481 | 12.952 | 0.662 | 7.422 | 0.066 | 10.224 | 1.181 | 0.166 | 13 |
| knn | network | 0 | 0.070 | 0.680 | 7.614 | 0.068 | 10.993 | 1.243 | 0.237 | 14 |
| pidl | network | 4486 | 2.490 | 0.682 | 7.644 | 0.068 | 13.276 | 1.363 | 0.167 | 15 |
| perl | network | 17729 | 2.848 | 0.140 | 1.574 | 0.014 | 25.643 | 1.544 | 0.008 | 16 |
| gru | network | 13313 | 2.640 | 0.102 | 1.141 | 0.010 | 60.028 | 3.349 | 0.140 | 17 |
| mdn_lstm | network | 18639 | 3.067 | 0.156 | 1.742 | 0.016 | 68.590 | 2.828 | 0.134 | 18 |
| lstm | network | 17729 | 2.460 | 0.115 | 1.284 | 0.011 | 80.691 | 3.634 | 0.106 | 19 |
| seq2seq | network | 34881 | 5.612 | 0.270 | 3.027 | 0.027 | 159.301 | 6.148 | 0.018 | 21 |
| maml_lstm | network | 17729 | 28.577 | 0.299 | 3.348 | 0.030 | 241.820 | 10.586 | 0.284 | 22 |
| diffusion | network | 42433 | 4.273 | 0.056 | 0.632 | 0.006 | 277.361 | 12.023 | 0.241 | 23 |

* Kendall tau between the one-step and the closed-loop ranking, all models: -0.31 (p = 0.034, n = 24)
* Kendall tau between the one-step and the closed-loop ranking, learned models only: -0.13 (p = 0.506, n = 16)
