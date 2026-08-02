from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.task_adapters import (
    create_loss_fn,
    create_torch_metrics,
    infer_output_dim,
)
from rtml.multi_instance.methods._torch.deep_sets.modules import DeepSets
from rtml.multi_instance.methods._torch.steps import (
    create_evaluation_step,
    create_training_step,
)
from rtml.multi_instance.tasks import MultiInstanceTask


def _dimensions(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a sequence of integers")
    dimensions = tuple(int(dimension) for dimension in value)
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError(f"{name} values must be positive")
    return dimensions


def build_deep_sets_bundle(
    *,
    task: MultiInstanceTask,
    input_dim: int,
    n_classes: int | None,
    params: Mapping[str, Any],
    fit_config: TorchFitConfig,
    device: torch.device,
) -> TorchModelBundle:
    """Build a Deep Sets model and its task-specific engine closures."""
    config = dict(params)
    encoder_dims = _dimensions(config.pop("encoder_dims", (64, 64)), "encoder_dims")
    latent_dim = int(config.pop("latent_dim", 64))
    predictor_dims = _dimensions(config.pop("predictor_dims", (64,)), "predictor_dims")
    pooling = str(config.pop("pooling", "sum"))
    dropout = float(config.pop("dropout", 0.0))
    if config:
        unknown = ", ".join(sorted(config))
        raise ValueError(f"unknown deep_sets params: {unknown}")

    output_dim = infer_output_dim(task.task_type, n_classes=n_classes)
    model = DeepSets(
        input_dim,
        output_dim,
        encoder_dims=encoder_dims,
        latent_dim=latent_dim,
        predictor_dims=predictor_dims,
        pooling=pooling,
        dropout=dropout,
    ).to(device)
    loss_fn = create_loss_fn(task.task_type)

    return TorchModelBundle(
        model=model,
        loss_fn=loss_fn,
        fit_config=fit_config,
        create_training_step=lambda optimizer: create_training_step(
            task_type=task.task_type,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        ),
        evaluation_step=create_evaluation_step(
            task_type=task.task_type,
            model=model,
            loss_fn=loss_fn,
        ),
        train_metrics_factory=lambda: create_torch_metrics(
            task.task_type, [metric.name for metric in task.metrics]
        ),
        validation_metrics_factory=lambda: create_torch_metrics(
            task.task_type, [metric.name for metric in task.metrics]
        ),
        test_metrics_factory=lambda: create_torch_metrics(
            task.task_type, [metric.name for metric in task.metrics]
        ),
        metadata={
            "model_class": model.__class__.__name__,
            "encoder_dims": list(encoder_dims),
            "latent_dim": latent_dim,
            "predictor_dims": list(predictor_dims),
            "pooling": pooling,
            "dropout": dropout,
        },
    )
