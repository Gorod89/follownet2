"""Supervised training with early stopping, optionally with a closed-loop term.

The loss of a minibatch is ``model.training_loss(windows, index)``; with
``closed_loop_weight > 0`` the squared errors of gap and speed of a rollout over
``closed_loop_horizon`` steps are added (gradients through the integrator). The best epoch
on the validation windows is kept.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import torch

from follownet2.sim.differentiable import closed_loop_loss, rollout_windows
from follownet2.train.windows import WindowSet


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 6
    batches_per_epoch: int | None = 200  # None: the whole training set
    val_batches: int | None = 50
    grad_clip: float = 1.0
    closed_loop_weight: float = 0.0
    closed_loop_horizon: int = 50
    closed_loop_batch: int = 512
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def validate(model, val: WindowSet, cfg: TrainConfig) -> float:
    """Root mean square one-step acceleration error on the validation windows [m/s^2]."""
    model.eval()
    gen = torch.Generator().manual_seed(cfg.seed)
    total, count = 0.0, 0
    for index in val.batches(cfg.batch_size, gen, limit=cfg.val_batches):
        total += float(model.validation_error(val, index))
        count += len(index)
    return (total / max(count, 1)) ** 0.5


def fit_supervised(model, train: WindowSet, val: WindowSet, cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model.to(train.device)
    statistics = train.statistics(seed=cfg.seed)
    model.set_scale(statistics)
    s_scale, v_scale = statistics["x_std"][0], statistics["x_std"][2]
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    closed = cfg.closed_loop_weight > 0.0
    if closed and train.horizon < cfg.closed_loop_horizon:
        raise ValueError(f"closed-loop training over {cfg.closed_loop_horizon} steps needs windows of that horizon")

    # the state before training is a candidate too (a hybrid model starts at its physics)
    best, best_state, stale, history = validate(model, val, cfg), copy.deepcopy(model.state_dict()), 0, []
    initial = best
    started = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        total, n_batches = 0.0, 0
        for index in train.batches(cfg.batch_size, gen, limit=cfg.batches_per_epoch):
            loss = model.training_loss(train, index)
            if closed:
                sim = rollout_windows(model, train, index[: cfg.closed_loop_batch], cfg.closed_loop_horizon)
                loss = loss + cfg.closed_loop_weight * closed_loop_loss(sim, s_scale, v_scale)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()
            total += float(loss.detach())
            n_batches += 1
        schedule.step()
        error = validate(model, val, cfg)
        history.append({"epoch": epoch, "train_loss": total / max(n_batches, 1), "val_rmse_a": error})
        if error < best - 1e-6:
            best, best_state, stale = error, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_val_rmse_a": best,
        "initial_val_rmse_a": initial,
        "epochs_run": len(history),
        "history": history,
        "seconds": round(time.time() - started, 1),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "config": asdict(cfg),
    }
