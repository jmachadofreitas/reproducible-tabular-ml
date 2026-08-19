from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from ignite.metrics import (
    ROC_AUC,
    Accuracy,
    Average,
    EpochMetric,
    Loss,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
)
from sklearn.metrics import roc_auc_score
from torch import nn

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.methods import MethodSpec
from rtml.core.results import PredictionSet
from rtml.core.tasks import MetricSpec, TaskType
from rtml.methods.engines.core import concat_evaluator_output
from rtml.methods.engines.metrics import IgniteMetric, RunningMetrics

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
    if task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }:
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
                "roc_auc": (probabilities.reshape(-1), target.reshape(-1)),
                "log_loss": (logits, target.float()),
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
                "roc_auc": (probabilities, target),
                "log_loss": (logits, target),
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


def _multiclass_roc_auc(probabilities: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        roc_auc_score(
            target.cpu().numpy(),
            probabilities.cpu().numpy(),
            multi_class="ovr",
        )
    )


def create_torch_metrics(
    task_type: TaskType,
    metric_names: Sequence[str] = (),
) -> RunningMetrics:
    requested = set(metric_names)
    if task_type == TaskType.REGRESSION:
        metrics = {"loss": IgniteMetric(Average())}
        available = {
            "mse": MeanSquaredError,
            "rmse": RootMeanSquaredError,
            "mae": MeanAbsoluteError,
        }
        metrics.update(
            {name: IgniteMetric(available[name]()) for name in requested if name in available}
        )
        return RunningMetrics(metrics)
    if task_type in {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
    }:
        metrics = {"loss": IgniteMetric(Average())}
        if "accuracy" in requested:
            metrics["accuracy"] = IgniteMetric(Accuracy())
        if "roc_auc" in requested:
            roc_auc = (
                ROC_AUC()
                if task_type == TaskType.BINARY_CLASSIFICATION
                else EpochMetric(_multiclass_roc_auc)
            )
            metrics["roc_auc"] = IgniteMetric(roc_auc)
        if "log_loss" in requested:
            metrics["log_loss"] = IgniteMetric(Loss(create_loss_fn(task_type)))
        return RunningMetrics(metrics)
    raise ValueError(f"unsupported torch task type: {task_type.value}")


def resolve_score_metric(
    primary_metric: str | None,
    metrics: Sequence[MetricSpec],
) -> MetricSpec | None:
    """Return the requested model-selection metric, if the task defines one."""
    if not metrics:
        return None
    if primary_metric is None:
        return metrics[0]
    for metric in metrics:
        if metric.name == primary_metric:
            return metric
    raise ValueError(f"primary metric {primary_metric!r} is not present in task metrics")


def infer_score_mode(metric: MetricSpec) -> str:
    """Return the checkpoint optimization mode for a task metric."""
    return "max" if metric.greater_is_better else "min"


def build_supervised_prediction_set(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    resample_id: str,
    test_indices: np.ndarray,
    outputs: Mapping[str, list[Any]],
    classes: np.ndarray | None,
) -> PredictionSet:
    """Build sample-aligned predictions from collected Torch evaluator outputs."""
    y_true = require_supervised_target(case).iloc[test_indices].to_numpy()
    if case.task.task_type == TaskType.REGRESSION:
        return PredictionSet(
            dataset_name=case.dataset.name,
            task_name=case.task.name,
            method_name=method.name,
            resample_id=resample_id,
            sample_ids=case.dataset.sample_ids_for(test_indices),
            y_true=y_true,
            values=concat_evaluator_output(outputs, "y_pred").reshape(-1),
            metadata={"case_name": case.name},
        )

    if classes is None:
        raise ValueError("classification predictions require class labels")
    predicted_indices = concat_evaluator_output(outputs, "labels").reshape(-1).astype(int)
    probabilities = concat_evaluator_output(outputs, "probabilities")
    if case.task.task_type == TaskType.BINARY_CLASSIFICATION:
        positive_probability = probabilities.reshape(-1)
        probabilities = np.column_stack([1.0 - positive_probability, positive_probability])

    return PredictionSet(
        dataset_name=case.dataset.name,
        task_name=case.task.name,
        method_name=method.name,
        resample_id=resample_id,
        sample_ids=case.dataset.sample_ids_for(test_indices),
        y_true=y_true,
        labels=classes[predicted_indices],
        probabilities=probabilities,
        scores=concat_evaluator_output(outputs, "logits"),
        metadata={"case_name": case.name, "classes": classes.tolist()},
    )
