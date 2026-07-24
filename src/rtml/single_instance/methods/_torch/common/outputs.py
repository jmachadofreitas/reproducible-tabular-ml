from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.methods import MethodSpec
from rtml.core.results import PredictionSet
from rtml.core.tasks import TaskType
from rtml.single_instance.methods._torch.common.helpers import require_supervised_target


def concat_output(outputs: Mapping[str, list[Any]], name: str) -> np.ndarray:
    values = outputs.get(name)
    if not values:
        raise ValueError(f"evaluator did not produce {name!r} outputs")
    arrays = []
    for value in values:
        if isinstance(value, torch.Tensor):
            arrays.append(value.detach().cpu().numpy())
        else:
            arrays.append(np.asarray(value))
    return np.concatenate(arrays, axis=0)


def build_prediction_set(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    resample_id: str,
    test_indices: np.ndarray,
    outputs: Mapping[str, list[Any]],
    classes: np.ndarray | None,
) -> PredictionSet:
    y_true = require_supervised_target(case).iloc[test_indices].to_numpy()
    if case.task.task_type == TaskType.REGRESSION:
        values = concat_output(outputs, "y_pred").reshape(-1)
        return PredictionSet(
            dataset_name=case.dataset.name,
            task_name=case.task.name,
            method_name=method.name,
            resample_id=resample_id,
            row_ids=case.dataset.row_ids_for(test_indices),
            y_true=y_true,
            values=values,
            metadata={"case_name": case.name},
        )

    if classes is None:
        raise ValueError("classification predictions require class labels")

    predicted_indices = concat_output(outputs, "labels").reshape(-1).astype(int)
    predicted_labels = classes[predicted_indices]
    probabilities = concat_output(outputs, "probabilities")
    if case.task.task_type == TaskType.BINARY_CLASSIFICATION:
        positive_probability = probabilities.reshape(-1)
        probabilities = np.column_stack([1.0 - positive_probability, positive_probability])

    return PredictionSet(
        dataset_name=case.dataset.name,
        task_name=case.task.name,
        method_name=method.name,
        resample_id=resample_id,
        row_ids=case.dataset.row_ids_for(test_indices),
        y_true=y_true,
        labels=predicted_labels,
        probabilities=probabilities,
        scores=concat_output(outputs, "logits"),
        metadata={"case_name": case.name, "classes": classes.tolist()},
    )
