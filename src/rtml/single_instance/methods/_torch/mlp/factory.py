from collections.abc import Mapping
from typing import Any

import torch

from rtml.core.tasks import TaskSpec
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.task_adapters import (
    create_loss_fn,
    create_torch_metrics,
    infer_output_dim,
)
from rtml.single_instance.methods._torch.mlp.modules import MLP
from rtml.single_instance.methods._torch.mlp.steps import (
    create_evaluation_step,
    create_prediction_step,
    create_training_step,
)


def _hidden_dims_from_config(params: dict[str, Any]) -> tuple[int, ...]:
    hidden_dims = params.pop("hidden_dims", (32,))
    if isinstance(hidden_dims, int):
        hidden_dims = (hidden_dims,)
    return tuple(int(dim) for dim in hidden_dims)


def build_mlp_bundle(
    *,
    task: TaskSpec,
    input_dim: int,
    n_classes: int | None,
    params: Mapping[str, Any],
    fit_config: TorchFitConfig,
    device: torch.device,
) -> TorchModelBundle:
    """Build the dense single-head MLP method from one consumed config mapping."""
    config = dict(params)
    hidden_dims = _hidden_dims_from_config(config)
    dropout = float(config.pop("dropout", 0.0))

    if config:
        unknown = ", ".join(sorted(config))
        raise ValueError(f"unknown simple_mlp params: {unknown}")

    output_dim = infer_output_dim(task.task_type, n_classes=n_classes)
    model = MLP(
        input_dim,
        [*hidden_dims, output_dim],
        dropout=dropout,
        last_dropout=False,
    ).to(device)
    loss_fn = create_loss_fn(task.task_type)

    return TorchModelBundle(
        model=model,
        loss_fn=loss_fn,
        fit_config=fit_config,
        create_training_step=lambda optimizer: create_training_step(
            task=task,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        ),
        evaluation_step=create_evaluation_step(task=task, model=model, loss_fn=loss_fn),
        prediction_step=create_prediction_step(task=task, model=model),
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
            "hidden_dims": list(hidden_dims),
            "dropout": dropout,
        },
    )
