from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from ignite.metrics import (
    Accuracy,
    Average,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
)
from torch import nn

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.metrics import MetricRequest, metric_greater_is_better
from rtml.core.tasks import MetricSpec, TaskType
from rtml.methods.engines.metrics import IgniteMetric, Metrics

PreparedTarget = Callable[[torch.Tensor], torch.Tensor]
TrainEvaluationOutputFormatter = Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]
PredictionOutputFormatter = Callable[[torch.Tensor], dict[str, torch.Tensor]]


def require_supervised_target(case: BenchmarkCase) -> Any:
    target = case.task.target_series(case.dataset)
    if target is None:
        raise ValueError("torch method execution requires a supervised task target")
    return target


def encode_classification_target(
    y_train: Any,
    y_eval: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(np.unique(y_train.to_numpy()).tolist()))
    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    try:
        train_encoded = np.asarray([class_to_index[label] for label in y_train.to_numpy()])
        eval_encoded = np.asarray([class_to_index[label] for label in y_eval.to_numpy()])
    except KeyError as exc:
        raise ValueError(
            f"target contains class not present in training split: {exc.args[0]!r}"
        ) from exc
    return train_encoded, eval_encoded, classes


def target_tensors(
    *,
    task_type: TaskType,
    y_train: Any,
    y_eval: Any,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray | None]:
    if task_type == TaskType.REGRESSION:
        return (
            torch.as_tensor(y_train.to_numpy(dtype=np.float32)).reshape(-1, 1),
            torch.as_tensor(y_eval.to_numpy(dtype=np.float32)).reshape(-1, 1),
            None,
        )
    if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
        train_encoded, eval_encoded, classes = encode_classification_target(y_train, y_eval)
        if task_type == TaskType.BINARY_CLASSIFICATION and len(classes) != 2:
            raise ValueError(
                "binary classification requires exactly two classes in the training split"
            )
        return torch.as_tensor(train_encoded), torch.as_tensor(eval_encoded), classes
    raise ValueError(f"unsupported torch task type: {task_type.value}")


def infer_output_dim(task_type: TaskType, *, n_classes: int | None = None) -> int:
    if task_type in {TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION}:
        return 1
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if n_classes is None or n_classes < 2:
            raise ValueError("multiclass tasks require n_classes >= 2")
        return n_classes
    raise ValueError(f"unsupported torch task type: {task_type.value}")


def create_loss_fn(task_type: TaskType) -> nn.Module:
    if task_type == TaskType.REGRESSION:
        return nn.MSELoss()
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return nn.BCEWithLogitsLoss()
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return nn.CrossEntropyLoss()
    raise ValueError(f"unsupported torch task type: {task_type.value}")


def make_target_preparer(task_type: TaskType) -> PreparedTarget:
    if task_type in {TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION}:
        return lambda target: target.float().reshape(-1, 1)
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return lambda target: target.long().reshape(-1)
    raise ValueError(f"unsupported torch task type: {task_type.value}")


def make_train_evaluation_output_formatter(
    task_type: TaskType,
) -> TrainEvaluationOutputFormatter:
    if task_type == TaskType.REGRESSION:
        return lambda logits, target: {
            "y_pred": logits,
            "y": target,
            "mse": (logits, target),
            "rmse": (logits, target),
            "mae": (logits, target),
        }

    if task_type == TaskType.BINARY_CLASSIFICATION:

        def format_binary(logits: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
            probabilities = torch.sigmoid(logits)
            labels = (probabilities >= 0.5).long()
            return {
                "logits": logits,
                "probabilities": probabilities,
                "labels": labels,
                "y": target.long(),
                "accuracy": (labels.reshape(-1), target.reshape(-1)),
            }

        return format_binary

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:

        def format_multiclass(logits: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
            probabilities = torch.softmax(logits, dim=1)
            labels = probabilities.argmax(dim=1)
            return {
                "logits": logits,
                "probabilities": probabilities,
                "labels": labels,
                "y": target,
                "accuracy": (labels, target),
            }

        return format_multiclass

    raise ValueError(f"unsupported torch task type: {task_type.value}")


def make_prediction_output_formatter(task_type: TaskType) -> PredictionOutputFormatter:
    """Create targetless output formatting for a standard torch model."""
    if task_type == TaskType.REGRESSION:
        return lambda logits: {"y_pred": logits}

    if task_type == TaskType.BINARY_CLASSIFICATION:

        def format_binary(logits: torch.Tensor) -> dict[str, torch.Tensor]:
            probabilities = torch.sigmoid(logits)
            return {
                "logits": logits,
                "probabilities": probabilities,
                "labels": (probabilities >= 0.5).long(),
            }

        return format_binary

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:

        def format_multiclass(logits: torch.Tensor) -> dict[str, torch.Tensor]:
            probabilities = torch.softmax(logits, dim=1)
            return {
                "logits": logits,
                "probabilities": probabilities,
                "labels": probabilities.argmax(dim=1),
            }

        return format_multiclass

    raise ValueError(f"unsupported torch task type: {task_type.value}")


def create_torch_metrics(
    task_type: TaskType,
    metric_names: Sequence[str] = (),
) -> Metrics:
    requested = set(metric_names)
    if task_type == TaskType.REGRESSION:
        if not requested:
            requested.add("mse")
        metrics = {"loss": IgniteMetric(Average())}
        available = {
            "mse": MeanSquaredError,
            "rmse": RootMeanSquaredError,
            "mae": MeanAbsoluteError,
        }
        metrics.update(
            {name: IgniteMetric(available[name]()) for name in requested if name in available}
        )
        return Metrics(metrics)
    if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
        metrics = {"loss": IgniteMetric(Average())}
        if not requested or "accuracy" in requested:
            metrics["accuracy"] = IgniteMetric(Accuracy())
        return Metrics(metrics)
    raise ValueError(f"unsupported torch task type: {task_type.value}")


def resolve_score_name(
    primary_metric: str | None,
    metrics: Sequence[MetricSpec],
) -> str:
    if primary_metric is not None:
        return primary_metric
    if metrics:
        return metrics[0].name
    raise ValueError("task must define at least one metric to build a trainer")


def infer_score_mode(metric: MetricRequest | str) -> str:
    """Return the checkpoint optimization mode for a task metric."""
    return "max" if metric_greater_is_better(metric) else "min"
