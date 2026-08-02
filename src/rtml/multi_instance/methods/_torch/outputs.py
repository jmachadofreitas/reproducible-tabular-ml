from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.methods import MethodSpec
from rtml.core.results import PredictionSet
from rtml.core.tasks import TaskType
from rtml.methods.engines.core import concat_evaluator_output
from rtml.methods.engines.task_adapters import require_supervised_target


def build_prediction_set(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    resample_id: str,
    test_indices: np.ndarray,
    outputs: Mapping[str, list[Any]],
    classes: np.ndarray | None,
) -> PredictionSet:
    """Build bag-aligned predictions from collected Torch evaluator outputs."""
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
