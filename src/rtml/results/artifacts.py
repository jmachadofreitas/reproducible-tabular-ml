from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from rtml.core.results import PredictionSet
from rtml.tasks.metrics import MetricRequest, compute_metrics

_ARRAY_FIELDS = ("row_ids", "y_true", "labels", "probabilities", "scores", "values")


def _optional_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)


def save_prediction_set(predictions: PredictionSet, path: str | Path) -> Path:
    """Save one PredictionSet as a compressed local artifact."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "dataset_name": predictions.dataset_name,
        "task_name": predictions.task_name,
        "method_name": predictions.method_name,
        "resample_id": predictions.resample_id,
        "metadata": predictions.metadata,
        "present_fields": [
            field
            for field in _ARRAY_FIELDS
            if _optional_array(getattr(predictions, field)) is not None
        ],
    }
    arrays = {
        field: array
        for field in _ARRAY_FIELDS
        if (array := _optional_array(getattr(predictions, field))) is not None
    }
    np.savez_compressed(artifact_path, metadata_json=json.dumps(payload), **arrays)  # pyright: ignore[reportArgumentType]
    return artifact_path


def load_prediction_set(path: str | Path) -> PredictionSet:
    """Load a PredictionSet saved by save_prediction_set."""
    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as data:
        payload = json.loads(str(data["metadata_json"]))
        arrays = {field: data[field].copy() for field in payload["present_fields"]}

    return PredictionSet(
        dataset_name=payload["dataset_name"],
        task_name=payload["task_name"],
        method_name=payload["method_name"],
        resample_id=payload["resample_id"],
        row_ids=arrays["row_ids"],
        y_true=arrays.get("y_true"),
        labels=arrays.get("labels"),
        probabilities=arrays.get("probabilities"),
        scores=arrays.get("scores"),
        values=arrays.get("values"),
        metadata=payload.get("metadata", {}),
    )


def recompute_metrics_from_prediction_path(
    path: str | Path,
    metrics: Iterable[MetricRequest],
) -> dict[str, float]:
    """Recompute metrics from a saved PredictionSet artifact."""
    return compute_metrics(metrics, load_prediction_set(path))
