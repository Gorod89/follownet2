# Model contract

Binding for every model in `trafficloop/models/`. Read first, in this order:
`trafficloop/data/batch.py`, `trafficloop/models/base.py`, `trafficloop/models/network.py`,
`trafficloop/train/windows.py`, `trafficloop/train/trainer.py`,
`trafficloop/sim/differentiable.py`, `trafficloop/models/registry.py`,
`tests/test_training_core.py` (the class `Tiny` there is the smallest complete model).

## 1. Interface

```python
acc = model(hist)      # hist [..., rows, 4], rows >= model.history; acc [...] in m/s^2
```

| Item | Rule |
|---|---|
| History | channels `(s, dv, v, a_prev)` in SI units, rows in time order, last row = now. `dv = v - v_lead`, positive when closing in. `a_prev[k] = a[k-1]`. A model reads its last `history` rows (`history <= 30`) and ignores the rows before. |
| Output | the acceleration applied over the next 0.1 s. The protocols clip it to `[-8, 4]`; a model does not clip. |
| Batch shape | any leading dimensions; `float32` and `float64` inputs, output in the dtype of the input. |
| Determinism | in evaluation mode two calls with the same input give the same output. Stochastic models return a fixed point estimate from `forward` and draw from `sample(hist, generator)`. |
| Distribution | models with a predictive distribution implement `predict_dist(hist)`: a `torch.distributions` object (or `trafficloop.models.dist.Empirical`) over the acceleration in m/s^2, batch shape = the leading dimensions of `hist`. |
| Adaptation | a model that adapts to the event at hand sets `adaptive = True` and implements `begin(hist)`: `hist [E, 30, 4]` is the observed warm-up; afterwards `forward` gets `[E, ..., H, 4]`. Only these observed rows may be used. |
| Inputs | `use_a_prev=False` by default: the reference implementations read `(s, dv, v)` as in the literature. |
| Construction | keyword arguments with defaults, of plain types (int, float, bool, str, list); all of them are stored in `self.kwargs`, so that `registry.create(name, **kwargs)` builds the same model. `name` is the key of `registry.MODELS`. |
| Training | `fit(train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict` with at least `best_val_rmse_a`, `epochs_run`, `seconds`, `n_parameters`, `history`. Supervised models inherit it from `NetworkModel` and override `training_loss` only. `train_horizon` says how many steps after an instant the training needs. |
| Preparation | the training entry calls `before_training(context)` once before `fit` (learned models only); `context` is a `TrainingContext` with the dataset name and `calibrate(model_cls, penalty=None)`, the pooled calibration of a mechanistic model on the training part exactly as the mechanistic entries get it. The default does nothing; `residual_idm_certified` takes its IDM core and certified budget there. A `certificate` entry of the training log is written to the top level of `train.json`. |
| Seeds | everything random in `fit` comes from `cfg.seed`. |
| Side effects | none: no files, no prints, no global state. Works on CPU and GPU. |
| Configuration | `conf/model/<name>.yaml` with `name`, `kwargs`, `optim` (overrides of `conf/train.yaml: train.optim`). |

## 2. Reference implementations

Differences from the papers are deliberate where they are listed here; any further
difference goes into the docstring of the class under "Differences from the paper".

| Model | Specification |
|---|---|
| `mlp` | `history=10`; the rows `(s, dv, v)` flattened; hidden layers `[64, 64]`, tanh; one output. |
| `gru`, `lstm` | `history=10`; one recurrent layer of 64 units; linear head on the last hidden state. |
| `seq2seq` | Ma and Qu (2020): encoder LSTM over `history=10` rows that end `delay=5` samples before now (reaction delay 0.5 s; the model attribute `history` is 15); decoder LSTM that produces `out_steps=10` accelerations, fed with its previous output; loss: squared error over all outputs; `forward` returns the first output (receding horizon), `predict_sequence(hist)` all of them. `train_horizon = out_steps`. |
| `mdn_lstm` | LSTM encoder as `lstm`; head: mixture of `n_components=5` Gaussians over the acceleration; loss: negative log-likelihood; `forward`: mean of the mixture; `predict_dist`: `MixtureSameFamily`. |
| `pidl` | Mo, Shi, Di (2021): network `a = N(s, dv, v)` (`history=1`, `memoryless=True`); loss `alpha * MSE_data + (1 - alpha) * MSE_physics`, the second term between the network and the IDM at `n_collocation=4096` states drawn uniformly from the range of the training data; IDM parameters learned jointly, kept inside the calibration bounds of `IDM.bounds` by a sigmoid; `alpha=0.7`. |
| `perl` | Long et al. (2023): `a = IDM(s, dv, v; theta) + R(hist)`, `R` an LSTM over `history=10` with a zero-initialised head, so that training starts at the physics model. `fit` first estimates `theta` by least squares of the one-step acceleration on the training windows (inside `IDM.bounds`), then freezes it and trains `R`. |
| `stability_mlp` | memoryless MLP `a = f(s, dv, v)` (`history=1`, `memoryless=True`); loss: squared error plus `weight=1.0` times the stability penalty of `trafficloop/stability/analytic.py` (section 3). |
| `residual_idm_certified` | certified hybrid of cf-stability (`ResidualIDM` with the budget of its decision D86): `a = IDM(s, dv, v; theta) + r_max tanh(lipschitz g(s/20, dv/2, v/10))` on the last row (`history=1`, `memoryless=True`); `g` a tanh MLP `[64, 64]` with spectral normalisation on every layer and antisymmetric initialisation (zero residual before training); `theta` frozen, from the pooled calibration of the training part with the fallbacks of `docs/models.md`; `lipschitz = 0.9 b / r_max`, `b` the largest budget for which the a priori string-stability certificate holds at every speed (`trafficloop/stability/certificate.py`); one-step loss of the prediction clipped to `[-8, 4]`; the certificate of the trained weights must hold. |
| `ddpg` | Zhu et al. (2020): actor and critic MLPs `[64, 64]` on the last row `(s, dv, v)`; action `3 * tanh(.)` m/s^2; reward `F_ttc + F_headway + F_jerk` with `F_ttc = log(TTC / 4)` for `0 <= TTC <= 4 s` and 0 otherwise, `F_headway` the density of a log-normal distribution (`mu = 0.4226`, `sigma = 0.4365`) at the time headway `s / v`, `F_jerk = -jerk^2 / 3600`; a collision ends the episode with reward -1. The reward does not imitate the driver. Episodes: `FollowEnv` over `train_horizon=100` steps. |
| `gail` | Kuefler et al. (2017): Gaussian policy (GRU over `history=10`, state-independent log-std), discriminator MLP on (last row, action), reward `-log(1 - D)`; policy update by PPO with a value network instead of TRPO; expert pairs: the recorded (history, acceleration) of the training windows; `forward`: the mean of the policy. `train_horizon=50`. |
| `knn` | `k=20` nearest neighbours of the last row `(s, dv, v)`, scaled by the standard deviations of the training data, among `n_reference=50000` training instants drawn at random; inverse-distance weights; brute force in chunks on the device of the model. |
| `maml_lstm` | MetaFollower style: LSTM body shared by all drivers, linear head adapted per event (`adaptive=True`). Meta-training over drivers (tasks): support and query instants of one driver, `inner_steps=3` gradient steps of the head with step size `inner_lr`, first-order outer update of body and head initialisation. At test time `begin` adapts the head on the pairs (history, acceleration) inside the observed warm-up. |
| `diffusion` | conditional DDPM over the next acceleration: condition = GRU encoding of `history=10` rows; denoiser MLP on (noisy acceleration, step embedding, condition); `n_steps=50`, cosine schedule; loss: squared error of the noise. `sample(hist, generator)`: DDIM with `n_sample_steps=10`; `forward`: mean of `n_forward_samples=8` DDIM samples from a generator with a fixed seed; `predict_dist`: `Empirical` of `n_dist_samples=64` samples. |

## 3. Stability helper (`trafficloop/stability/analytic.py`)

For a memoryless model `a = f(s, dv, v)`:

* equilibrium gap `s_e(v)`: root of `f(s, 0, v) = 0` in `s` on `[0.5, 300] m` by bisection
  (no gradient); `nan` where there is no sign change;
* partial derivatives `f_s, f_dv, f_v` at `(s_e, 0, v)` by automatic differentiation
  (`create_graph=True` when used in a loss);
* string-stability margin `m = f_v^2 + 2 f_v f_dv - 2 f_s` (stable when `m >= 0`), with
  `dv = v - v_lead`; rational driving requires `f_s > 0`, `f_dv < 0`, `f_v < 0`;
* penalty: `mean(relu(margin - m)) + mean(relu(-f_s) + relu(f_dv) + relu(f_v))` over the
  speeds `v_e = 5 .. 30 m/s` (26 values) that have an equilibrium;
* `unstable_fraction(model, speeds)`: share of the equilibria with `m < 0`.

Test: for the IDM the helper reproduces the closed-form partial derivatives and the margin.

## 4. Tests

`tests/test_model_<group>.py`, CPU only, less than three minutes per file. Data: synthetic
IDM followers from `trafficloop.data.loaders.synthetic.make_events` (see the fixtures of
`tests/test_training_core.py`). For every model:

1. interface: shapes, dtypes, leading dimensions, rows before the history are ignored,
   `a_prev` is ignored unless `use_a_prev=True`;
2. determinism in evaluation mode;
3. checkpoint round trip through `registry.save_model` / `load_model` with identical outputs;
4. learning: after `fit` with a small budget the one-step error on held-out synthetic events
   is below a stated fraction of the error of a constant-speed follower, and a rollout
   (`rollout_windows`, 100 steps) is finite;
5. what is specific to the model (section 2): likelihood, physics term, penalty, reward
   values at known states, adaptation improves on a new driver, spread of the samples ...
