from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rtml.core.tasks import TaskType
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.task_adapters import (
    create_loss_fn,
    create_torch_metrics,
)
from rtml.multi_instance.methods._torch.attention_deep_mil.modules import (
    AttentionDeepMIL,
)
from rtml.multi_instance.methods._torch.steps import (
    create_evaluation_step,
    create_training_step,
)
from rtml.multi_instance.tasks import MultiInstanceTask


def build_attention_deep_mil_bundle(
    *,
    task: MultiInstanceTask,
    input_dim: int,
    n_classes: int | None,
    params: Mapping[str, Any],
    fit_config: TorchFitConfig,
    device: torch.device,
) -> TorchModelBundle:
    """Build AttentionDeepMIL and its binary-classification engine closures."""
    if task.task_type != TaskType.BINARY_CLASSIFICATION:
        raise ValueError("attention_deep_mil supports binary classification only")
    if n_classes != 2:
        raise ValueError("attention_deep_mil requires exactly two training classes")

    config = dict(params)
    encoder_dims = _dimensions(config.pop("encoder_dims", (128,)), "encoder_dims")
    embedding_dim = int(config.pop("embedding_dim", 128))
    attention_dim = int(config.pop("attention_dim", 64))
    attention = str(config.pop("attention", "gated"))
    n_attention_branches = int(config.pop("n_attention_branches", 1))
    dropout = float(config.pop("dropout", 0.0))
    if config:
        unknown = ", ".join(sorted(config))
        raise ValueError(f"unknown attention_deep_mil params: {unknown}")

    model = AttentionDeepMIL(
        input_dim,
        encoder_dims=encoder_dims,
        embedding_dim=embedding_dim,
        attention_dim=attention_dim,
        attention=attention,
        n_attention_branches=n_attention_branches,
        dropout=dropout,
    ).to(device)
    loss_fn = create_loss_fn(task.task_type)
    metric_names = [metric.name for metric in task.metrics]

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
        train_metrics_factory=lambda: create_torch_metrics(task.task_type, metric_names),
        validation_metrics_factory=lambda: create_torch_metrics(task.task_type, metric_names),
        test_metrics_factory=lambda: create_torch_metrics(task.task_type, metric_names),
        metadata={
            "model_class": model.__class__.__name__,
            "encoder_dims": list(encoder_dims),
            "embedding_dim": embedding_dim,
            "attention_dim": attention_dim,
            "attention": attention,
            "n_attention_branches": n_attention_branches,
            "dropout": dropout,
        },
    )


def _dimensions(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{name} must be a sequence of integers")
    dimensions = tuple(int(dimension) for dimension in value)
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError(f"{name} values must be positive")
    return dimensions
