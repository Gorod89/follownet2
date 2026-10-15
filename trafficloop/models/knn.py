"""k nearest neighbours: the acceleration of the most similar training states.

A nonparametric reference in the spirit of He, Zheng and Guan (2015), "A simple nonparametric
car-following model driven by field data", Transportation Research Part B 80.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import torch

from trafficloop.models.network import NetworkModel
from trafficloop.train.windows import WindowSet

WEIGHT_OFFSET = 1e-6  # in normalised units; keeps the weights finite at zero distance
DISTANCE_FLOOR = 1e-30  # squared; keeps the gradient finite when a query equals a reference


class KNN(NetworkModel):
    """``a = sum_i w_i a_i / sum_i w_i`` over the ``k`` training instants nearest to the state.

    The state is the last row ``(s, dv, v)`` (with ``use_a_prev`` also ``a_prev``) divided by
    the standard deviations of the training data, the weights are ``1 / (distance + 1e-6)``
    in these units. :meth:`fit` stores ``n_reference`` training instants drawn at random
    (their normalised states and accelerations are buffers); nothing is trained. The search is
    brute force on the device of the model, in chunks of queries such that at most
    ``max_pairs`` distances exist at a time: the candidates are ranked by
    :func:`torch.cdist` and the weights use the exact distances of the ``k`` chosen ones.

    Differences from the paper: He, Zheng and Guan average the recorded response of the most
    similar historical states; here the response is the acceleration applied over the next
    step, the neighbours are weighted by inverse distance, the state variables are scaled by
    their standard deviations and the reference set is a random sample of the training
    instants.
    """

    name = "knn"
    memoryless = True

    def __init__(self, k: int = 20, n_reference: int = 50_000, max_pairs: int = 200_000_000, use_a_prev: bool = False) -> None:
        super().__init__(history=1, use_a_prev=use_a_prev)
        self.k, self.n_reference, self.max_pairs = int(k), int(n_reference), int(max_pairs)
        self.kwargs = {"k": self.k, "n_reference": self.n_reference, "max_pairs": self.max_pairs, "use_a_prev": bool(use_a_prev)}
        self.register_buffer("x_ref", torch.zeros(0, self.n_inputs))  # normalised states
        self.register_buffer("a_ref", torch.zeros(0))  # normalised accelerations

    def core(self, x: torch.Tensor) -> torch.Tensor:
        n_ref = int(self.x_ref.shape[0])
        if n_ref == 0:
            raise RuntimeError("knn has no reference instants: fit or restore it first")
        query, k = x[:, -1], min(self.k, n_ref)
        chunk = max(1, self.max_pairs // n_ref)
        out = []
        for q in query.split(chunk):
            with torch.no_grad():
                index = torch.cdist(q, self.x_ref).topk(k, dim=-1, largest=False).indices  # [c, k]
            distance = (q.unsqueeze(1) - self.x_ref[index]).square().sum(-1).clamp_min(DISTANCE_FLOOR).sqrt()
            weight = 1.0 / (distance + WEIGHT_OFFSET)
            out.append((weight * self.a_ref[index]).sum(-1) / weight.sum(-1))
        return torch.cat(out)

    def fit(self, train: WindowSet, val: WindowSet, cfg) -> dict:
        """Store the reference instants; the log has the validation error of the stored model."""
        from trafficloop.train.trainer import validate

        started = time.time()
        self.to(train.device)
        self.set_scale(train.statistics(seed=cfg.seed))
        gen = torch.Generator().manual_seed(cfg.seed)
        index = torch.randperm(len(train), generator=gen)[: self.n_reference].to(train.device)
        self.x_ref = self.inputs(train.history(index, rows=1))[:, -1].to(self.x_mean.dtype)
        self.a_ref = (train.target(index)[:, 0] / self.a_std).to(self.x_mean.dtype)
        error = validate(self, val, cfg)
        return {
            "best_val_rmse_a": error,
            "epochs_run": 0,
            "history": [],
            "seconds": round(time.time() - started, 1),
            "n_parameters": sum(p.numel() for p in self.parameters()),
            "n_reference": int(self.x_ref.shape[0]),
            "config": asdict(cfg),
        }

    def restore(self, state: dict) -> None:
        # the reference set of a checkpoint has the size of its training sample, not that of a new model
        self.x_ref = torch.empty(state["x_ref"].shape, dtype=self.x_ref.dtype, device=self.x_ref.device)
        self.a_ref = torch.empty(state["a_ref"].shape, dtype=self.a_ref.dtype, device=self.a_ref.device)
        self.load_state_dict(state)
