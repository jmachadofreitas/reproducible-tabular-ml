from __future__ import annotations

from collections.abc import Callable

import torch
from ignite.metrics import Accuracy, Average, MeanSquaredError
from torch import nn

from rtml.core.tasks import TaskSpec, TaskType
from rtml.methods.engines import IgniteMetric, Metrics

PreparedTarget = Callable[[torch.Tensor], torch.Tensor]
PredictionFormatter = Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]


def infer_output_dim(task: TaskSpec, *, n_classes: int | None = None) -> int:
    """Map a task definition to a neural prediction head width."""
    if task.task_type in {TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION}:
        return 1
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if n_classes is None or n_classes < 2:
            raise ValueError("multiclass tasks require n_classes >= 2")
        return n_classes
    raise ValueError(f"unsupported torch task type: {task.task_type.value}")


def create_loss_fn(task: TaskSpec) -> nn.Module:
    """Build the default torch loss for one supervised task."""
    if task.task_type == TaskType.REGRESSION:
        return nn.MSELoss()
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return nn.BCEWithLogitsLoss()
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return nn.CrossEntropyLoss()
    raise ValueError(f"unsupported torch task type: {task.task_type.value}")


def make_target_preparer(task: TaskSpec) -> PreparedTarget:
    """Create a target conversion closure once per task."""
    if task.task_type == TaskType.REGRESSION:
        return lambda y: y.float().reshape(-1, 1)
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return lambda y: y.float().reshape(-1, 1)
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return lambda y: y.long().reshape(-1)
    raise ValueError(f"unsupported torch task type: {task.task_type.value}")


def make_prediction_formatter(task: TaskSpec) -> PredictionFormatter:
    """Create a prediction formatter closure once per task."""
    if task.task_type == TaskType.REGRESSION:
        return lambda logits, y: {
            "y_pred": logits,
            "y": y,
            "mse": (logits, y),
        } # type: ignore

    if task.task_type == TaskType.BINARY_CLASSIFICATION:

        def format_binary(
            logits: torch.Tensor,
            y: torch.Tensor,
        ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
            probabilities = torch.sigmoid(logits)
            labels = (probabilities >= 0.5).long()
            return {
                "logits": logits,
                "probabilities": probabilities,
                "labels": labels,
                "y": y.long(),
                "accuracy": (labels.reshape(-1), y.reshape(-1)),
            }

        return format_binary # type: ignore

    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:

        def format_multiclass(
            logits: torch.Tensor,
            y: torch.Tensor,
        ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
            probabilities = torch.softmax(logits, dim=1)
            labels = probabilities.argmax(dim=1)
            return {
                "logits": logits,
                "probabilities": probabilities,
                "labels": labels,
                "y": y,
                "accuracy": (labels, y),
            }

        return format_multiclass # type: ignore

    raise ValueError(f"unsupported torch task type: {task.task_type.value}")


def create_torch_metrics(
    *,
    task: TaskSpec,
) -> Metrics:
    """Create engine metrics for one supervised torch task."""
    if task.task_type == TaskType.REGRESSION:
        return Metrics(
            {
                "loss": IgniteMetric(Average()),
                "mse": IgniteMetric(MeanSquaredError()),
            }
        )

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        return Metrics(
            {
                "loss": IgniteMetric(Average()),
                "accuracy": IgniteMetric(Accuracy()),
            }
        )

    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return Metrics(
            {
                "loss": IgniteMetric(Average()),
                "accuracy": IgniteMetric(Accuracy()),
            }
        )

    raise ValueError(f"unsupported torch task type: {task.task_type.value}")


def resolve_score_name(task: TaskSpec) -> str:
    if task.primary_metric is not None:
        return task.primary_metric
    if task.metrics:
        return task.metrics[0].name
    raise ValueError("task must define at least one metric to build a trainer")


def infer_score_mode(metric_name: str) -> str:
    lower_name = metric_name.lower()
    minimize_metrics = {"loss", "mse", "rmse", "mae", "mape", "log_loss", "brier"}
    return "min" if lower_name in minimize_metrics else "max"
