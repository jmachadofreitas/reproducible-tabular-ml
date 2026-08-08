import numpy as np
import pytest

from rtml.core.results import PredictionSet
from rtml.core.tasks import MetricSpec
from rtml.results.artifacts import (
    load_prediction_set,
    recompute_metrics_from_prediction_path,
    save_prediction_set,
)


def test_prediction_set_round_trip_preserves_arrays_and_metadata(tmp_path) -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="binary",
        method_name="logreg_linear",
        resample_id="fold_00",
        sample_ids=np.array(["r1", "r2", "r3"]),
        y_true=np.array([0, 1, 1]),
        labels=np.array([0, 1, 0]),
        probabilities=np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]]),
        subgroups={"site": np.array(["a", "a", "b"])},
        metadata={"case_name": "toy_case"},
    )

    path = save_prediction_set(predictions, tmp_path / "predictions.npz")
    loaded = load_prediction_set(path)

    assert loaded.dataset_name == predictions.dataset_name
    assert loaded.task_name == predictions.task_name
    assert loaded.method_name == predictions.method_name
    assert loaded.resample_id == predictions.resample_id
    assert loaded.metadata == {"case_name": "toy_case"}
    np.testing.assert_array_equal(loaded.sample_ids, predictions.sample_ids)
    np.testing.assert_array_equal(loaded.y_true, predictions.y_true)
    np.testing.assert_array_equal(loaded.labels, predictions.labels)
    np.testing.assert_array_equal(loaded.probabilities, predictions.probabilities)
    np.testing.assert_array_equal(loaded.subgroups["site"], predictions.subgroups["site"])
    assert loaded.values is None


def test_metrics_recompute_from_saved_prediction_set(tmp_path) -> None:
    predictions = PredictionSet(
        dataset_name="toy",
        task_name="regression",
        method_name="ridge_linear",
        resample_id="fold_00",
        sample_ids=np.array([0, 1, 2]),
        y_true=np.array([1.0, 2.0, 4.0]),
        values=np.array([1.0, 4.0, 7.0]),
    )
    path = save_prediction_set(predictions, tmp_path / "regression.npz")

    metrics = recompute_metrics_from_prediction_path(
        path,
        [
            MetricSpec(name="rmse", greater_is_better=False),
            MetricSpec(name="mae", greater_is_better=False),
        ],
    )

    assert metrics["rmse"] == np.sqrt(13.0 / 3.0)
    assert metrics["mae"] == 5.0 / 3.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("y_true", [0, 1]),
        ("labels", [0, 1]),
        ("probabilities", [[0.5, 0.5], [0.5, 0.5]]),
        ("scores", [0.1, 0.2]),
        ("values", [0.1, 0.2]),
    ],
)
def test_prediction_set_rejects_misaligned_prediction_arrays(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        PredictionSet(
            dataset_name="toy",
            task_name="task",
            method_name="method",
            resample_id="fold_00",
            sample_ids=[0, 1, 2],
            **{field: value},
        )


def test_prediction_set_rejects_misaligned_subgroups() -> None:
    with pytest.raises(ValueError, match="subgroup 'site'"):
        PredictionSet(
            dataset_name="toy",
            task_name="task",
            method_name="method",
            resample_id="fold_00",
            sample_ids=[0, 1, 2],
            subgroups={"site": ["a", "b"]},
        )
