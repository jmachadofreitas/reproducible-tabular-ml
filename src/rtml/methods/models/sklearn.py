from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from rtml.methods.base import MethodSpec
from rtml.runtime import RuntimeSpec
from rtml.tasks.base import TaskType


def build_sklearn_estimator(
    *,
    task_type: TaskType,
    method: MethodSpec,
    seed: int,
    runtime: RuntimeSpec | None = None,
) -> Any:
    """Build a sklearn estimator from a method spec."""
    model_params = dict(method.model.params)
    model_kind = method.model.kind

    if model_kind == "logistic_regression":
        if task_type == TaskType.REGRESSION:
            raise ValueError("logistic_regression does not support regression tasks")
        model_params.setdefault("max_iter", 1000)
        model_params.setdefault("random_state", seed)
        return LogisticRegression(**model_params)

    if model_kind == "ridge":
        if task_type != TaskType.REGRESSION:
            raise ValueError("ridge only supports regression tasks")
        model_params.setdefault("random_state", seed)
        return Ridge(**model_params)

    if model_kind == "linear_regression":
        if task_type != TaskType.REGRESSION:
            raise ValueError("linear_regression only supports regression tasks")
        return LinearRegression(**model_params)

    if model_kind == "random_forest":
        model_params.setdefault("random_state", seed)
        if runtime is not None and runtime.num_threads is not None:
            model_params.setdefault("n_jobs", runtime.num_threads)
        if task_type == TaskType.REGRESSION:
            return RandomForestRegressor(**model_params)
        return RandomForestClassifier(**model_params)

    if model_kind == "boosted_trees":
        model_params.setdefault("random_state", seed)
        if task_type == TaskType.REGRESSION:
            return HistGradientBoostingRegressor(**model_params)
        return HistGradientBoostingClassifier(**model_params)

    raise NotImplementedError(f"unsupported sklearn model kind {model_kind!r}")
