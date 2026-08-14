import pytest

from rtml.core.metrics import EvaluationMetrics
from rtml.core.results import PredictionSet
from rtml.core.tasks import MetricSpec


def test_compute_metrics_uses_prediction_set_without_backend_objects() -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="binary",
        method_name="any_backend",
        resample_id="fold_00",
        sample_ids=[0, 1, 2, 3],
        y_true=[0, 1, 1, 0],
        labels=[0, 1, 0, 0],
        probabilities=[
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
            [0.7, 0.3],
        ],
    )

    metrics = EvaluationMetrics(
        [
            MetricSpec(name="accuracy", greater_is_better=True),
            MetricSpec(name="roc_auc", greater_is_better=True),
            MetricSpec(name="log_loss", greater_is_better=False),
        ]
    ).compute(predictions)

    assert metrics["accuracy"] == 0.75
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] > 0.0


def test_evaluation_metrics_only_computes_requested_metrics() -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="binary",
        method_name="method",
        resample_id="fold_00",
        sample_ids=[0, 1],
        y_true=[0, 1],
        labels=[0, 1],
    )

    assert EvaluationMetrics([MetricSpec(name="accuracy", greater_is_better=True)]).compute(
        predictions
    ) == {"accuracy": 1.0}
    assert EvaluationMetrics([]).compute(predictions) == {}


def test_binary_roc_auc_accepts_decision_scores_without_probabilities() -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="binary",
        method_name="margin_model",
        resample_id="fold_00",
        sample_ids=[0, 1, 2, 3],
        y_true=[0, 1, 1, 0],
        scores=[-2.0, 1.0, 2.0, -1.0],
    )

    metrics = EvaluationMetrics([MetricSpec(name="roc_auc", greater_is_better=True)]).compute(
        predictions
    )

    assert metrics == {"roc_auc": 1.0}


def test_regression_metrics_have_expected_values() -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="regression",
        method_name="any_backend",
        resample_id="fold_00",
        sample_ids=[0, 1, 2],
        y_true=[1.0, 2.0, 4.0],
        values=[1.0, 4.0, 7.0],
    )

    metrics = EvaluationMetrics(
        [
            MetricSpec(name="mse", greater_is_better=False),
            MetricSpec(name="rmse", greater_is_better=False),
            MetricSpec(name="mae", greater_is_better=False),
        ]
    ).compute(predictions)

    assert metrics["mse"] == pytest.approx(13.0 / 3.0)
    assert metrics["rmse"] == pytest.approx((13.0 / 3.0) ** 0.5)
    assert metrics["mae"] == pytest.approx(5.0 / 3.0)


def test_unknown_metric_reports_known_names() -> None:
    with pytest.raises(KeyError, match="accuracy"):
        EvaluationMetrics([MetricSpec(name="not_a_metric", greater_is_better=True)]).compute(
            PredictionSet(
                dataset_name="toy",
                task_name="task",
                method_name="method",
                resample_id="fold_00",
                sample_ids=[],
            )
        )


def test_metric_direction_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="greater_is_better"):
        MetricSpec(name="accuracy", greater_is_better="yes")  # type: ignore[arg-type]
