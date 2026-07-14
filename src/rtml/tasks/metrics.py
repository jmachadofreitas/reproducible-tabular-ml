from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score

from rtml.results.base import PredictionSet


class MetricRequest(Protocol):
    name: str
    kwargs: dict[str, Any]


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


def compute_accuracy(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        accuracy_score(
            _require_y_true(predictions), _require_labels(predictions, "accuracy"), **kwargs
        )
    )


def compute_roc_auc(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    options = dict(kwargs)
    probabilities = _binary_or_matrix_probabilities(predictions)
    if probabilities.ndim == 2:
        options.setdefault("multi_class", "ovr")
    return float(roc_auc_score(_require_y_true(predictions), probabilities, **options))


def compute_log_loss(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        log_loss(
            _require_y_true(predictions),
            _require_probabilities(predictions, "log_loss"),
            **kwargs,
        )
    )


def compute_mse(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    residuals = _require_y_true(predictions) - _require_values(predictions, "mse")
    if kwargs:
        raise TypeError("mse does not currently support metric kwargs")
    return float(np.mean(np.square(residuals)))


def compute_rmse(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    if kwargs:
        raise TypeError("rmse does not currently support metric kwargs")
    return float(np.sqrt(compute_mse(predictions, {})))


def compute_mae(predictions: PredictionSet, kwargs: Mapping[str, Any]) -> float:
    return float(
        mean_absolute_error(
            _require_y_true(predictions),
            _require_values(predictions, "mae"),
            **kwargs,
        )
    )


METRIC_REGISTRY: dict[str, MetricFunction] = {
    "accuracy": compute_accuracy,
    "roc_auc": compute_roc_auc,
    "log_loss": compute_log_loss,
    "mse": compute_mse,
    "rmse": compute_rmse,
    "mae": compute_mae,
}


def get_metric_function(name: str) -> MetricFunction:
    try:
        return METRIC_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(METRIC_REGISTRY))
        raise KeyError(f"unknown metric {name!r}; known metrics: {known}") from exc


def compute_metric(metric: MetricRequest, predictions: PredictionSet) -> float:
    metric_fn = get_metric_function(metric.name)
    return metric_fn(predictions, dict(metric.kwargs))


def compute_metrics(
    metrics: Iterable[MetricRequest],
    predictions: PredictionSet,
) -> dict[str, float]:
    return {metric.name: compute_metric(metric, predictions) for metric in metrics}
