"""Windows, base class of the learned models, trainer, differentiable rollout, environment, registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from follownet2 import pipeline
from follownet2.data.batch import WARMUP, EventBatch
from follownet2.data.loaders.synthetic import IDM_NOMINAL, SyntheticConfig, make_events
from follownet2.models import registry
from follownet2.models.mechanistic import IDM
from follownet2.models.network import NetworkModel, mlp
from follownet2.models.persistence import PersistenceBaseline
from follownet2.sim.differentiable import FollowEnv, closed_loop_loss, rollout_windows
from follownet2.sim.rollout import rollout
from follownet2.train.run import model_dir, train_model
from follownet2.train.trainer import TrainConfig, fit_supervised
from follownet2.train.windows import WindowSet
from follownet2.utils import REPO_ROOT

F64 = dict(device="cpu", dtype=torch.float64)


class Tiny(NetworkModel):
    name = "tiny"

    def __init__(self, history: int = 1, use_a_prev: bool = False, hidden: int = 32) -> None:
        super().__init__(history, use_a_prev)
        self.kwargs["hidden"] = hidden
        self.net = mlp([history * self.n_inputs, hidden, hidden, 1])

    def core(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).squeeze(-1)


@pytest.fixture(scope="module")
def events():
    return make_events(SyntheticConfig(n_drivers=30, events_per_driver=2, theta_cv=0.0, duration=(20.0, 35.0), seed=21))


@pytest.fixture(scope="module")
def windows(events):
    return WindowSet.from_events(events, horizon=40, **F64)


# ----------------------------------------------------------------------------- windows
def test_windows_are_the_prediction_instants_of_the_protocols(events, windows):
    assert len(windows) == sum(len(ev) - 40 - WARMUP for ev in events)
    index = torch.tensor([0, 17, len(windows) - 1])
    hist = windows.history(index)
    for row, i in enumerate(index.tolist()):
        ev, k = events[int(windows.event[i])], int(windows.step[i])
        assert WARMUP <= k <= len(ev) - 1 - 40
        np.testing.assert_array_equal(hist[row, :, 0].numpy(), ev.s[k - WARMUP + 1 : k + 1])
        np.testing.assert_array_equal(hist[row, :, 1].numpy(), ev.dv[k - WARMUP + 1 : k + 1])
        np.testing.assert_array_equal(hist[row, :, 3].numpy(), ev.a[k - WARMUP : k])  # a_prev: nothing of the future
        np.testing.assert_array_equal(windows.target(index, 5)[row].numpy(), ev.a[k : k + 5])
        future = windows.future(index, 40)
        np.testing.assert_array_equal(future["s"][row].numpy(), ev.s[k + 1 : k + 41])
        np.testing.assert_array_equal(future["v_lead"][row].numpy(), ev.v_lead[k + 1 : k + 41])
        np.testing.assert_allclose(future["dx_lead"][row].numpy(), np.diff(ev.x_lead)[k : k + 40])
    with pytest.raises(ValueError, match="horizon"):
        windows.target(index, 41)


def test_batches_cover_every_instant_once_and_drivers_are_indexed(events, windows):
    gen = torch.Generator().manual_seed(0)
    seen = torch.cat(list(windows.batches(1000, gen)))
    assert sorted(seen.tolist()) == list(range(len(windows)))
    assert len(list(windows.batches(1000, gen, limit=3))) == 3
    assert len(windows.driver_names) == 30
    names = [windows.driver_names[i] for i in windows.driver_of(torch.arange(len(windows))).tolist()]
    assert names[0] == events[0].follower_id and names[-1] == events[-1].follower_id


def test_short_events_give_no_window():
    events = make_events(SyntheticConfig(n_drivers=2, events_per_driver=1, duration=(20.0, 21.0), seed=1))
    assert len(WindowSet.from_events(events, horizon=1, **F64)) == sum(len(e) - 1 - WARMUP for e in events)
    assert len(WindowSet.from_events(events, horizon=300, **F64)) == 0


# ------------------------------------------------------------------- model and trainer
def test_network_model_works_in_physical_units_and_any_batch_shape(windows):
    model = Tiny(history=3).double()
    model.set_scale(windows.statistics())
    assert torch.allclose(model.x_std, windows.statistics()["x_std"][:3])
    hist = windows.history(torch.arange(12))
    out = model(hist)
    assert out.shape == (12,) and out.dtype == hist.dtype
    stacked = model(hist.reshape(3, 4, WARMUP, 4))
    assert stacked.shape == (3, 4) and torch.allclose(stacked.reshape(-1), out)
    assert torch.allclose(model(hist.float()), out.float(), atol=1e-5)  # the dtype of the caller is kept
    changed = hist.clone()
    changed[:, :-3] += 100.0  # rows before the history of the model
    changed[..., 3] += 5.0  # a_prev is not an input of this model
    assert torch.equal(model(changed), out)
    assert Tiny(history=2, use_a_prev=True).n_inputs == 4
    with pytest.raises(ValueError, match="warm-up"):
        Tiny(history=WARMUP + 1)


def test_training_learns_the_generating_model(events, windows):
    val = WindowSet.from_events(make_events(SyntheticConfig(n_drivers=8, theta_cv=0.0, seed=22)), **F64)
    torch.manual_seed(5)  # the initial weights are drawn when the model is built
    model = Tiny(history=1).double()
    cfg = TrainConfig(epochs=30, batch_size=512, lr=3e-3, batches_per_epoch=None, patience=30, device="cpu", seed=0)
    log = fit_supervised(model, windows, val, cfg)
    index = torch.arange(len(val))
    constant = float(torch.sqrt(torch.mean(val.target(index)[:, 0] ** 2)))
    assert log["best_val_rmse_a"] < 0.15 * constant  # the data are a function of (s, dv, v)
    assert log["history"][-1]["train_loss"] < 0.2 * log["history"][0]["train_loss"]
    assert not model.training and log["n_parameters"] == sum(p.numel() for p in model.parameters())
    torch.manual_seed(5)
    again = Tiny(history=1).double()
    assert fit_supervised(again, windows, val, cfg)["best_val_rmse_a"] == log["best_val_rmse_a"]


def test_early_stopping_keeps_the_best_epoch(windows):
    val = WindowSet.from_events(make_events(SyntheticConfig(n_drivers=4, theta_cv=0.0, seed=23)), **F64)
    model = Tiny(history=1).double()
    cfg = TrainConfig(epochs=40, batch_size=256, lr=0.3, batches_per_epoch=5, patience=3, device="cpu", grad_clip=0.0)
    log = fit_supervised(model, windows, val, cfg)
    errors = [h["val_rmse_a"] for h in log["history"]]
    # the state before training competes as well: here it wins, the learning rate is far too high
    assert log["epochs_run"] < 40
    assert log["best_val_rmse_a"] == pytest.approx(min([log["initial_val_rmse_a"], *errors]))
    index = torch.arange(len(val))
    reached = float(torch.sqrt(model.validation_error(val, index) / len(val)))
    assert reached == pytest.approx(log["best_val_rmse_a"], rel=1e-6)


# ----------------------------------------------------- rollout with gradients, environment
def test_rollout_from_windows_equals_the_event_rollout(events, windows):
    model = IDM((30.0, 1.2, 3.0, 1.4, 2.0))
    first = torch.nonzero(windows.step == WARMUP).squeeze(-1)[:5]  # the instant where event rollouts start
    sim = rollout_windows(model, windows, first, 40)
    chosen = [events[int(e)] for e in windows.event[first]]
    whole = rollout(model, EventBatch.from_events(chosen, **F64))
    torch.testing.assert_close(sim["s"], whole.s[:, WARMUP + 1 : WARMUP + 41], rtol=0, atol=1e-10)
    torch.testing.assert_close(sim["a"], whole.a[:, WARMUP : WARMUP + 40], rtol=0, atol=1e-10)
    exact = rollout_windows(IDM(IDM_NOMINAL), windows, torch.arange(0, len(windows), 97), 40)
    assert float((exact["s"] - exact["s_obs"]).abs().max()) < 1e-9


def test_closed_loop_loss_has_gradients_and_training_reduces_it(events, windows):
    class Scaled(NetworkModel):
        """IDM with a learnable time gap: the closed-loop loss must find it."""

        name = "scaled"

        def __init__(self):
            super().__init__(1)
            self.gap = nn.Parameter(torch.tensor(2.4, dtype=torch.float64))

        def forward(self, hist):
            theta = torch.stack([torch.tensor(v, dtype=hist.dtype) for v in IDM_NOMINAL])
            theta = torch.cat([theta[:1], self.gap.reshape(1).to(hist.dtype), theta[2:]])
            return IDM().acc(hist, theta)

    model = Scaled()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    scale = torch.tensor(1.0, dtype=torch.float64)
    losses = []
    for step in range(60):
        index = torch.randint(0, len(windows), (64,), generator=torch.Generator().manual_seed(step))
        loss = closed_loop_loss(rollout_windows(model, windows, index, 40), scale, scale)
        optimiser.zero_grad()
        loss.backward()
        assert model.gap.grad is not None and torch.isfinite(model.gap.grad)
        optimiser.step()
        losses.append(float(loss))
    assert losses[-1] < 0.05 * losses[0]
    assert float(model.gap) == pytest.approx(IDM_NOMINAL[1], abs=0.1)


def test_environment_follows_the_integrator_and_ends_at_a_collision(windows):
    index = torch.arange(0, 600, 60)
    env = FollowEnv(windows, horizon=30)
    obs = env.reset(index)
    assert torch.equal(obs, windows.history(index))
    model = IDM(IDM_NOMINAL)
    for h in range(30):
        obs, info, done = env.step(model(obs))
        assert float((info.s - info.s_obs).abs().max()) < 1e-9 and not bool(info.collided.any())
        assert bool(done.all()) == (h == 29)
    with pytest.raises(RuntimeError):
        env.step(model(obs))

    obs = env.reset(index)
    crashed_at = torch.full((len(index),), -1)
    for h in range(30):
        obs, info, done = env.step(torch.full((len(index),), 4.0, dtype=torch.float64))
        assert float(info.a.max()) <= 4.0 + 1e-9 and float(info.jerk.abs().max()) <= 120.0 + 1e-6
        crashed_at = torch.where(info.collided, torch.full_like(crashed_at, h), crashed_at)
        frozen = (crashed_at >= 0) & (crashed_at < h)
        assert bool(done[crashed_at >= 0].all())
        if frozen.any():
            assert bool((info.s[frozen] <= 0).all()) and not bool(info.collided[frozen].any())
    assert bool((crashed_at >= 0).any())


# --------------------------------------------------------------------------- registry
def test_checkpoint_round_trip(windows, tmp_path, monkeypatch):
    monkeypatch.setitem(registry.MODELS, "tiny", "test_training_core:Tiny")
    model = Tiny(history=4, use_a_prev=True, hidden=16)
    model.set_scale(windows.statistics())
    hist = windows.history(torch.arange(50)).float()
    path = registry.save_model(model, tmp_path / "m" / "model.pt", {"dataset": "synthetic", "seed": 3})
    loaded, provenance = registry.load_model(path)
    assert provenance == {"dataset": "synthetic", "seed": 3} and not loaded.training
    assert loaded.kwargs == {"history": 4, "use_a_prev": True, "hidden": 16}
    assert torch.equal(loaded(hist), model.eval()(hist))

    idm = registry.create("idm_median", params=[30.0, 1.0, 2.0, 1.0, 2.0])
    again, _ = registry.load_model(registry.save_model(idm, tmp_path / "idm.pt"))
    assert again.name == "idm_median" and again.param_dict() == idm.param_dict()
    assert isinstance(registry.create("persistence"), PersistenceBaseline)
    with pytest.raises(KeyError, match="unknown model"):
        registry.create("transformer")


def test_every_registered_model_has_a_configuration():
    missing = [name for name in registry.MODELS if not (REPO_ROOT / "conf" / "model" / f"{name}.yaml").exists()]
    assert missing == []


# ------------------------------------------------------------------------ train script
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    overrides = [
        "dataset=synthetic", "dataset.options.n_drivers=20", "dataset.options.events_per_driver=2", "workers=1",
        "storage.reserve_mb=1", "calibration.maxiter=15", "calibration.global_sample=20",
        "calibration.per_event_sample=8", "calibration.device=cpu", "calibration.dtype=float64", "train.device=cpu",
    ]
    out = {}
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        for name in ("persistence", "idm", "idm_median"):
            cfg = compose(config_name="train", overrides=[*overrides, f"model={name}"])
            for key in ("events", "splits", "calibration", "results"):
                cfg.paths[key] = str(root / key)
            if not out:
                pipeline.build_dataset(cfg)
                pipeline.build_splits(cfg)
            out[name] = (cfg, train_model(cfg))
    return out


def test_training_run_writes_checkpoint_scores_and_provenance(trained):
    for name, (cfg, summary) in trained.items():
        folder = model_dir(cfg)
        assert folder.parts[-5:] == ("synthetic", "reconstructed", "driver-fold0", name, "seed0")
        model, provenance = registry.load_model(folder / "model.pt")
        assert model.name == name and provenance["data_version"] == "reconstructed"
        assert provenance["split"] == {"kind": "driver", "fold": 0, "seed": 0} and len(provenance["config_hash"]) == 12
        assert (folder / "test_p0.events.parquet").exists() and (folder / "test_closed_loop.events.parquet").exists()
        assert set(summary["test"]) == {"p0", "closed_loop_whole_event"}
    persistence, idm, median = (trained[n][1] for n in ("persistence", "idm", "idm_median"))
    assert persistence["family"] == "baseline" and idm["family"] == "mechanistic"
    assert persistence["test"]["p0"]["rmse_a"] < idm["test"]["p0"]["rmse_a"]  # at 0.1 s nothing beats it
    assert idm["test"]["closed_loop_whole_event"]["rmse_s"] < persistence["test"]["closed_loop_whole_event"]["rmse_s"]
    assert idm["training"]["kind"] == "pooled" and median["training"]["kind"].startswith("median")
    assert median["training"]["params"] != idm["training"]["params"]
