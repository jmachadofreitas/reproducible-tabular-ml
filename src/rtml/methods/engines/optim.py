from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch
from torch import nn

from rtml.methods.engines.schedulers import (
    CosineAnnealingHP,
    CosineAnnealingWarmRestartsHP,
    LinearHP,
)


def create_optimizer(
    model: nn.Module,
    *,
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Create a torch optimizer from a compact config mapping."""
    normalized_name = name.lower()
    if normalized_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    if normalized_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    if normalized_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"unknown optimizer {name!r}; expected one of: adam, adamw, sgd")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    config: dict[str, Any] | None,
    max_epochs: int,
) -> Any | None:
    """Create a torch learning-rate scheduler, or no scheduler."""
    if not config:
        return None

    scheduler_config = dict(config)
    name = str(scheduler_config.pop("name", "none")).lower()
    if name in {"", "none"}:
        return None
    if name == "linear":
        scheduler_config.setdefault("total_iters", max_epochs)
        return torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_config)
    if name == "cosine":
        scheduler_config.setdefault("T_max", max_epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    if name == "step":
        scheduler_config.setdefault("step_size", 10)
        scheduler_config.setdefault("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_config)
    raise ValueError(f"unknown lr scheduler {name!r}; expected one of: linear, cosine, step")


def create_hp_scheduler(
    hparams: MutableMapping[str, Any],
    *,
    config: dict[str, Any] | None,
    max_epochs: int,
) -> Any | None:
    """Create an arbitrary-hyperparameter scheduler, or no scheduler."""
    if not config:
        return None

    scheduler_config = dict(config)
    name = str(scheduler_config.pop("name", "none")).lower()
    if name in {"", "none"}:
        return None
    if name == "linear":
        scheduler_config.setdefault("total_iters", max_epochs)
        return LinearHP(hparams, **scheduler_config)
    if name == "cosine":
        scheduler_config.setdefault("T_max", max_epochs)
        return CosineAnnealingHP(hparams, **scheduler_config)
    if name == "cosine_warm_restarts":
        return CosineAnnealingWarmRestartsHP(hparams, **scheduler_config)
    raise ValueError(
        "unknown hyperparameter scheduler "
        f"{name!r}; expected one of: linear, cosine, cosine_warm_restarts"
    )
