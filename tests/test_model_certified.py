"""Certified hybrid of cf-stability (``residual_idm_certified``, model contract sections 1 to 4): interface, choice
of the core and of the certified budget before training, learning with the certificate kept, checkpoint, and the
training entry that writes the certificate into ``train.json``."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from trafficloop import pipeline
from trafficloop.data.batch import WARMUP
from trafficloop.data.loaders.synthetic import SyntheticConfig, make_events
from trafficloop.models import registry
from trafficloop.models.base import TrainingContext
from trafficloop.models.certified import PHYSICAL_SCALE
from trafficloop.models.mechanistic import IDM
from trafficloop.sim.differentiable import rollout_windows
from trafficloop.stability import analytic, certificate
from trafficloop.train.run import model_dir, train_model
from trafficloop.train.trainer import TrainConfig
from trafficloop.train.windows import WindowSet
from trafficloop.utils import REPO_ROOT, read_json

NAME = "residual_idm_certified"
# cores of cf-stability's tests (FollowNet HighD fits): no budget at r_max 1 or 0.3; none at 1, one at 0.3; one at 1
CORE_FREE = (29.57, 0.676, 2.015, 0.351, 0.5)
CORE_M02 = (35.2, 0.98, 3.19, 1.57, 0.5)
CORE_M05 = (35.1, 0.94, 4.71, 2.74, 0.5)
THETA = (33.0, 1.5, 2.0, 2.0, 1.0)  # string stable on the whole speed grid: the drivers of the learning tests
CORE_OFF = (33.0, 1.3, 2.5, 2.0, 1.0)  # a pooled core that misses the drivers: the residual has something to learn


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


class Cores:
    """Pooled calibration of a training context that returns fixed cores and records its calls."""

    def __init__(self, free, stable) -> None:
        self.free, self.stable, self.calls = free, stable, []

    def __call__(self, model_cls, penalty=None) -> dict:
        self.calls.append((model_cls, penalty))
        core = self.free if penalty is None else self.stable
        return {"params": dict(zip(IDM.param_names, core)), "objective": 0.5, "rmse_s": 1.0, "n_events": 10}

    def context(self) -> TrainingContext:
        return TrainingContext(dataset="synthetic", calibrate=self)


def windows(seed: int, n_drivers: int, horizon: int = 1, dtype=torch.float32) -> WindowSet:
    cfg = SyntheticConfig(
        n_drivers=n_drivers, events_per_driver=2, theta=THETA, theta_cv=0.15, duration=(20.0, 35.0), seed=seed
    )
    return WindowSet.from_events(make_events(cfg), horizon=horizon, device="cpu", dtype=dtype)


def rmse(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.double() ** 2)))


def prepared(free=THETA, stable=CORE_M05, seed: int = 0, **kwargs):
    torch.manual_seed(seed)
    model = registry.create(NAME, **kwargs)
    model.before_training(Cores(free, stable).context())
    return model


# ------------------------------------------------------------------------- interface
def test_interface_and_the_untrained_model_is_its_core():
    torch.manual_seed(1)
    model = registry.create(NAME)
    model.set_core(THETA)
    model.eval()
    hist = windows(6, 4, dtype=torch.float64).history(torch.arange(40))
    out = model(hist)
    assert out.shape == (40,) and out.dtype == torch.float64
    torch.testing.assert_close(out, IDM(THETA)(hist), rtol=0, atol=1e-6)  # antisymmetric initialisation: r = 0
    with torch.no_grad():
        for p in model.g.parameters():
            p.add_(0.3 * torch.randn_like(p))  # a residual that is not zero
    out = model(hist)
    assert float((out - IDM(THETA)(hist)).detach().abs().max()) > 1e-3
    assert torch.equal(model(hist.reshape(4, 10, WARMUP, 4)).reshape(-1), out)
    single = model(hist.float())
    assert single.dtype == torch.float32
    torch.testing.assert_close(single.double(), out, rtol=1e-4, atol=1e-4)
    changed = hist.clone()
    changed[:, :-1] += 50.0  # rows before the last one
    changed[..., 3] += 5.0  # a_prev is not an input
    assert torch.equal(model(changed), out) and torch.equal(model(hist), out)
    assert model.memoryless and model.history == 1 and model.family == "network"
    assert model.x_std.tolist() == list(PHYSICAL_SCALE) and model.x_mean.tolist() == [0.0, 0.0, 0.0]
    statistics = windows(5, 4).statistics()
    model.set_scale(statistics)  # the trainer's statistics set the scale of the loss only
    assert model.x_std.tolist() == list(PHYSICAL_SCALE) and float(model.a_std) == pytest.approx(float(statistics["a_std"]))


def test_residual_is_bounded_and_its_slope_certified():
    torch.manual_seed(2)
    model = registry.create(NAME, r_max=0.5)
    model.lipschitz = 3.0
    x = torch.tensor([2.0, -5.0, 0.0]) + torch.tensor([150.0, 10.0, 40.0]) * torch.rand(4000, 3, dtype=torch.float64)
    hist = torch.cat([x, torch.zeros(4000, 1, dtype=torch.float64)], dim=-1).unsqueeze(1)
    target = 0.6 * torch.sin(x[:, 0] / 5.0) * torch.cos(x[:, 1])
    optimiser = torch.optim.Adam(model.g.parameters(), lr=0.05)
    model.train()
    for _ in range(50):  # away from the initialisation, with the power iteration of training
        loss = torch.mean((model.residual(hist) - target) ** 2)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    model.eval()
    hist = hist.clone().requires_grad_(True)
    r = model.residual(hist)
    assert float(r.detach().abs().max()) <= 0.5
    (grad,) = torch.autograd.grad(r.sum(), hist)
    slope, bound = grad[:, 0, :3].abs(), model.jacobian_bound().to(grad)
    assert (slope <= bound * (1 + 1e-5)).all() and (slope.amax(0) > 0).all()


def test_configuration_lists_every_constructor_argument():
    conf = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "conf" / "model" / f"{NAME}.yaml"))
    arguments = inspect.signature(registry.model_class(NAME)).parameters
    assert conf["name"] == NAME and set(conf["kwargs"]) == set(arguments) and conf["optim"] == {}
    assert registry.create(NAME, **conf["kwargs"]).kwargs == registry.create(NAME).kwargs == conf["kwargs"]
    assert conf["kwargs"]["safety"] == 0.9 and conf["kwargs"]["r_max"] == 0.3
    for bad in ({"safety": 1.5}, {"r_max": 0.0}, {"hidden": [64, 63]}):
        with pytest.raises(ValueError):
            registry.create(NAME, **bad)


# ------------------------------------------------------------- core and budget before training
def test_the_pooled_core_is_kept_when_it_admits_a_budget():
    cores = Cores(CORE_M05, CORE_M02)
    torch.manual_seed(0)
    model = registry.create(NAME)
    model.before_training(cores.context())
    setup = model.setup
    assert [penalty is None for _, penalty in cores.calls] == [True, False] and cores.calls[0][0] is IDM  # both cores
    assert setup["core"]["kind"] == "pooled"
    assert setup["core"]["params"] == pytest.approx(dict(zip(IDM.param_names, CORE_M05)))
    assert len(setup["attempts"]) == 2 and setup["r_max"] == model.r_max == 0.3
    assert model.lipschitz == pytest.approx(0.9 * setup["admissible"] / 0.3, rel=1e-12)
    assert model.kwargs == registry.create(NAME).kwargs  # the configuration is not changed


def test_fallbacks_core_with_a_margin_then_the_amplitude_of_d86():
    cores = Cores(CORE_FREE, CORE_M05)
    torch.manual_seed(0)
    model = registry.create(NAME)
    model.before_training(cores.context())
    assert model.setup["core"]["kind"] == "pooled with stability margin" and model.r_max == 0.3
    assert [a["admissible"] > 0 for a in model.setup["attempts"]] == [False, True]
    assert "no admissible residual budget" in model.setup["attempts"][0]["error"]
    penalty = cores.calls[1][1]  # the term of D72 with the configured margin
    candidates = torch.tensor([CORE_FREE, CORE_M02], dtype=torch.float32)
    torch.testing.assert_close(penalty(candidates), certificate.stability_shortfall(candidates, 0.2))

    cores = Cores(CORE_FREE, CORE_M02)  # the margin core needs the smaller amplitude
    torch.manual_seed(0)
    model = registry.create(NAME)
    model.before_training(cores.context())
    assert len(cores.calls) == 2  # each core is calibrated once
    attempts = model.setup["attempts"]
    assert [(a["core"], a["r_max"]) for a in attempts] == [
        ("pooled", 0.3), ("pooled with stability margin", 0.3)]
    assert model.r_max == model.setup["r_max"] == 0.3 and float(model.residual_scale[0]) == 0.3
    assert model.lipschitz == pytest.approx(0.9 * attempts[-1]["admissible"] / 0.3, rel=1e-12)
    assert model.kwargs["r_max"] == 0.3  # the configuration stays; the amplitude in force is in the state

    torch.manual_seed(0)
    model = registry.create(NAME)
    with pytest.raises(ValueError, match="no core admits a certified residual budget"):
        model.before_training(Cores(CORE_FREE, CORE_FREE).context())  # never trained without its certificate
    assert model.setup is None
    with pytest.raises(RuntimeError, match="before_training"):
        model.fit(windows(1, 2), windows(2, 2), TrainConfig(epochs=1, device="cpu"))


# ---------------------------------------------------------------------- learning
@pytest.fixture(scope="module")
def fitted():
    train, val, test = windows(1, 30), windows(2, 10), windows(3, 10, horizon=100)
    model = prepared(free=CORE_OFF, r_max=0.3)
    cfg = TrainConfig(epochs=6, batch_size=512, lr=3e-3, batches_per_epoch=None, patience=6, device="cpu", seed=0)
    return model, model.fit(train, val, cfg), test


def test_learning_keeps_the_certificate(fitted):
    model, log, test = fitted
    assert {"best_val_rmse_a", "epochs_run", "seconds", "n_parameters", "history", "certificate"} <= set(log)
    record = log["certificate"]
    assert record["holds"] and record["n_holds"] == record["n_speeds"] == len(analytic.SPEEDS)
    assert record["core"]["kind"] in ("pooled", "pooled with stability margin") and record["lipschitz"] == pytest.approx(model.lipschitz)
    assert record["guaranteed_margin_min"] > 0 and len(record["layer_norms"]) == 3
    assert math.prod(record["layer_norms"]) <= math.prod(record["layer_norms_at_enforcement"]) + 1e-6
    index = torch.arange(len(test))
    hist, target = test.history(index), test.target(index)[:, 0]
    with torch.no_grad():
        error, core_error = rmse(model(hist).clamp(-8, 4) - target), rmse(model.core_acc(hist).clamp(-8, 4) - target)
        assert error < 0.95 * core_error and error < rmse(target)
        sim = rollout_windows(model, test, torch.arange(0, len(test), 29), 100)
    assert all(bool(torch.isfinite(sim[key]).all()) for key in ("s", "v", "a"))
    # the analytic criterion of the hybrid at its equilibria is never below the certified bound
    margin = analytic.equilibrium_margin(model).double()
    certified = torch.tensor(record["guaranteed_margin"], dtype=torch.float64)
    assert torch.isfinite(margin).all() and (margin >= certified - 1e-6).all()


def test_checkpoint_round_trip(fitted, tmp_path):
    model, _, test = fitted
    hist = test.history(torch.arange(300))
    loaded, _ = registry.load_model(registry.save_model(model, tmp_path / "model.pt"))
    assert loaded.name == NAME and loaded.kwargs == model.kwargs
    state = model.state_dict()
    assert all(torch.equal(value, state[key]) for key, value in loaded.state_dict().items())
    assert torch.equal(loaded(hist), model.eval()(hist))
    assert (loaded.r_max, loaded.lipschitz, loaded.core_params()) == (model.r_max, model.lipschitz, model.core_params())
    assert loaded.layer_norms() == model.layer_norms()
    assert certificate.certificate_a_priori(loaded, n_scan=100)["holds"].all()


# ------------------------------------------------------------------ training entry
@pytest.fixture(scope="module")
def entry(tmp_path_factory):
    root = tmp_path_factory.mktemp("certified")
    overrides = [
        "dataset=synthetic", "dataset.options.n_drivers=20", "dataset.options.events_per_driver=2",
        f"+dataset.options.theta=[{','.join(str(x) for x in THETA)}]", "workers=1", "storage.reserve_mb=1",
        "calibration.maxiter=30", "calibration.global_sample=20", "calibration.device=cpu", "calibration.dtype=float64",
        "train.device=cpu", "train.optim.epochs=2", "train.optim.batches_per_epoch=5", "train.optim.batch_size=256",
    ]
    out = {}
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        for name in ("idm", NAME):
            cfg = compose(config_name="train", overrides=[*overrides, f"model={name}"])
            for key in ("events", "splits", "calibration", "results"):
                cfg.paths[key] = str(root / key)
            if not out:
                pipeline.build_dataset(cfg)
                pipeline.build_splits(cfg)
            out[name] = (cfg, train_model(cfg))
    return out


def test_training_entry_writes_the_certificate(entry):
    cfg, summary = entry[NAME]
    _, idm = entry["idm"]
    assert summary["family"] == "network" and set(summary["test"]) == {"p0", "closed_loop_whole_event"}
    record = summary["certificate"]
    assert "certificate" not in summary["training"] and record["holds"]
    first, accepted = record["attempts"][0], record["attempts"][-1]
    assert (first["core"], first["r_max"]) == ("pooled", 0.3)
    assert first["params"] == pytest.approx(idm["training"]["params"], rel=1e-12)  # the core of the idm entry first
    # the pooled core of these drivers is string unstable at 5-9 m/s within r_max = 1: the core with a margin
    assert record["admissible"] == max(a["admissible"] for a in record["attempts"]) > 0
    core = record["core"]
    assert core["kind"] in ("pooled", "pooled with stability margin")
    assert core["calibration"]["n_events"] == idm["training"]["n_events"] and (
        "fit_objective" in core["calibration"] or "objective" in core["calibration"])
    assert accepted["admissible"] == record["admissible"] > 0 and accepted["r_max"] == record["r_max"] == 0.3
    assert record["lipschitz"] == pytest.approx(0.9 * record["admissible"] / record["r_max"], rel=1e-12)
    assert read_json(model_dir(cfg) / "train.json")["certificate"] == record
    model, provenance = registry.load_model(model_dir(cfg) / "model.pt")
    assert provenance["config_hash"] == summary["config_hash"] and model.lipschitz == record["lipschitz"]
    assert model.core_params() == pytest.approx(core["params"], rel=1e-15)
