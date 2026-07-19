from dataclasses import dataclass, field
from typing import Any

import pytest

from rtml.core.results import PredictionSet
from rtml.core.metrics import METRIC_REGISTRY, compute_metrics, get_metric_function


@dataclass(frozen=True)
class MetricRequest:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


def test_metric_registry_exposes_backend_neutral_prediction_metrics() -> None:
    assert {"accuracy", "roc_auc", "log_loss", "mse", "rmse", "mae"}.issubset(METRIC_REGISTRY)


def test_compute_metrics_uses_prediction_set_without_backend_objects() -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="binary",
        method_name="any_backend",
        resample_id="fold_00",
        row_ids=[0, 1, 2, 3],
        y_true=[0, 1, 1, 0],
        labels=[0, 1, 0, 0],
        probabilities=[
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.7, 0.3],
        ],
    )

    metrics = compute_metrics(
        [MetricRequest("accuracy"), MetricRequest("roc_auc"), MetricRequest("log_loss")],  # type: ignore
        predictions,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] > 0.0


def test_regression_metrics_have_expected_values() -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="regression",
        method_name="any_backend",
        resample_id="fold_00",
        row_ids=[0, 1, 2],
        y_true=[1.0, 2.0, 4.0],
        values=[1.0, 4.0, 7.0],
    )

    metrics = compute_metrics(
        [MetricRequest("mse"), MetricRequest("rmse"), MetricRequest("mae")],
        predictions,
    )

    assert metrics["mse"] == pytest.approx(13.0 / 3.0)
    assert metrics["rmse"] == pytest.approx((13.0 / 3.0) ** 0.5)
    assert metrics["mae"] == pytest.approx(5.0 / 3.0)


def test_unknown_metric_reports_known_names() -> None:
    with pytest.raises(KeyError, match="accuracy"):
        get_metric_function("not_a_metric")
