from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tabm as tabm_lib
import torch

from rtml.core.tasks import TaskSpec
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.task_adapters import (
    create_loss_fn,
    create_torch_metrics,
    infer_output_dim,
)
from rtml.single_instance.methods._torch.tabm.steps import (
    create_evaluation_step,
    create_training_step,
)

_DATA_PARAMETERS = {"n_num_features", "cat_cardinalities", "d_out", "num_embeddings"}


def build_tabm_bundle(
    *,
    task: TaskSpec,
    input_dim: int,
    n_classes: int | None,
    params: Mapping[str, Any],
    fit_config: TorchFitConfig,
    device: torch.device,
) -> TorchModelBundle:
    """Build TabM for RTML's dense, preprocessed single-instance input."""
    config = dict(params)
    unsupported = sorted(_DATA_PARAMETERS.intersection(config))
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"TabM data parameter(s) are controlled by RTML preprocessing: {names}")

    model = tabm_lib.TabM.make(
        n_num_features=input_dim,
        d_out=infer_output_dim(task.task_type, n_classes=n_classes),
        **config,
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
            "k": model.k,
            "input_representation": "preprocessed_dense",
        },
    )
