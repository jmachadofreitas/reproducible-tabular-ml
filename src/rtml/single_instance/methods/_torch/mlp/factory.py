from collections.abc import Mapping
from typing import Any

import torch

from rtml.core.tasks import TaskSpec
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.optim import create_hp_scheduler
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
    """Build the MLP method; its optional hyperparameter scheduler controls dropout."""
    config = dict(params)
    hidden_dims = _hidden_dims_from_config(config)
    dropout = float(config.pop("dropout", 0.0))

    if config:
        unknown = ", ".join(sorted(config))
        raise ValueError(f"unknown simple_mlp params: {unknown}")

    output_dim = infer_output_dim(task.task_type, n_classes=n_classes)
    if fit_config.hp_scheduler is not None and dropout <= 0.0:
        raise ValueError("simple_mlp dropout scheduling requires model.params.dropout > 0")
    model = MLP(
        input_dim,
        [*hidden_dims, output_dim],
        dropout=dropout,
        last_dropout=False,
    ).to(device)
    dropout_layers = tuple(
        layer for layer in model.modules() if isinstance(layer, torch.nn.Dropout)
    )

    def apply_dropout(hparams: Mapping[str, Any]) -> None:
        probability = float(hparams["dropout"])
        for layer in dropout_layers:
            layer.p = probability

    hp_scheduler = create_hp_scheduler(
        {"dropout": dropout},
        config=None if fit_config.hp_scheduler is None else dict(fit_config.hp_scheduler),
        max_epochs=fit_config.max_epochs,
        apply_hparams=apply_dropout,
    )
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
        hp_scheduler=hp_scheduler,
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
