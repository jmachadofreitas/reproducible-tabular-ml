from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    root_mean_squared_error,
)

from rtml.core.results import PredictionSet
from rtml.core.tasks import MetricSpec

MetricFunction = Callable[[PredictionSet, Mapping[str, Any]], float]


def _require_y_true(predictions: PredictionSet) -> np.ndarray:
    if predictions.y_true is None:
        raise ValueError("metric computation requires y_true")
    return np.asarray(predictions.y_true)


def _require_labels(predictions: PredictionSet, metric_name: str) -> np.ndarray:
    if predictions.labels is None:
        raise ValueError(f"{metric_name} requires labels")
    return np.asarray(predictions.labels)


def _require_probabilities(predictions: PredictionSet, metric_name: str) -> np.ndarray:
    if predictions.probabilities is None:
        raise ValueError(f"{metric_name} requires probabilities")
    return np.asarray(predictions.probabilities)


def _require_values(predictions: PredictionSet, metric_name: str) -> np.ndarray:
    if predictions.values is None:
        raise ValueError(f"{metric_name} requires values")
    return np.asarray(predictions.values)


def _binary_or_matrix_probabilities(predictions: PredictionSet) -> np.ndarray:
    probabilities = _require_probabilities(predictions, "roc_auc")
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        return probabilities[:, 1]
    return probabilities


def _compute_accuracy(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        accuracy_score(
            _require_y_true(predictions), _require_labels(predictions, "accuracy"), **kwargs
        )
    )


def _compute_roc_auc(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    options = dict(kwargs)
    probabilities = _binary_or_matrix_probabilities(predictions)
    if probabilities.ndim == 2:
        options.setdefault("multi_class", "ovr")
    return float(roc_auc_score(_require_y_true(predictions), probabilities, **options))


def _compute_log_loss(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        log_loss(
            _require_y_true(predictions),
            _require_probabilities(predictions, "log_loss"),
            **kwargs,
        )
    )


def _compute_mse(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        mean_squared_error(
            _require_y_true(predictions),
            _require_values(predictions, "mse"),
            **kwargs,
        )
    )


def _compute_rmse(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        root_mean_squared_error(
            _require_y_true(predictions),
            _require_values(predictions, "rmse"),
            **kwargs,
        )
    )


def _compute_mae(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        mean_absolute_error(
            _require_y_true(predictions),
            _require_values(predictions, "mae"),
            **kwargs,
        )
    )


_METRIC_FUNCTIONS: dict[str, MetricFunction] = {
    "accuracy": _compute_accuracy,
    "roc_auc": _compute_roc_auc,
    "log_loss": _compute_log_loss,
    "mse": _compute_mse,
    "rmse": _compute_rmse,
    "mae": _compute_mae,
}


class EvaluationMetrics:
    """Compute the requested final metrics from complete predictions."""

    def __init__(self, metrics: Iterable[MetricSpec]) -> None:
        self.metrics = list(metrics)

    def compute(self, predictions: PredictionSet) -> dict[str, float]:
        return {metric.name: self.compute_metric(metric, predictions) for metric in self.metrics}

    @staticmethod
    def compute_metric(metric: MetricSpec, predictions: PredictionSet) -> float:
        try:
            metric_function = _METRIC_FUNCTIONS[metric.name]
        except KeyError as ex:
            known = ", ".join(sorted(_METRIC_FUNCTIONS))
            raise KeyError(
                f"unknown evaluation metric {metric.name!r}; known metrics: {known}"
            ) from ex
        return metric_function(predictions, metric.kwargs)
