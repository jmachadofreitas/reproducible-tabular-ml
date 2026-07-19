from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.methods import MethodSpec
from rtml.core.metrics import compute_metrics
from rtml.core.resampling import Resample
from rtml.core.results import PredictionSet
from rtml.core.runtime import RuntimeSpec
from rtml.core.tasks import TaskType
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.single_instance.preprocessing import build_preprocessor

SUPPORTED_SKLEARN_MODEL_KINDS = frozenset(
    {
        "boosted_trees",
        "dummy",
        "gradient_boosting",
        "linear_regression",
        "logistic_regression",
        "random_forest",
        "ridge",
    }
)


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
    if model_kind not in SUPPORTED_SKLEARN_MODEL_KINDS:
        raise NotImplementedError(f"unsupported sklearn model kind {model_kind!r}")

    if model_kind == "dummy":
        if task_type == TaskType.REGRESSION:
            return DummyRegressor(**model_params)
        return DummyClassifier(**model_params)

    if model_kind == "gradient_boosting":
        model_params.setdefault("random_state", seed)
        if task_type == TaskType.REGRESSION:
            return GradientBoostingRegressor(**model_params)
        return GradientBoostingClassifier(**model_params)

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

    raise AssertionError(f"unhandled sklearn model kind {model_kind!r}")


def default_single_instance_backends() -> tuple[MethodBackend, ...]:
    """Return built-in single-instance method backends."""
    return (SklearnBackend(),)


def _find_resample(case: BenchmarkCase, resample_id: str | None) -> Resample:
    if not case.resampling.resamples:
        raise ValueError(f"benchmark case {case.name!r} has no resamples")
    if resample_id is None:
        return case.resampling.resamples[0]
    for resample in case.resampling.resamples:
        if resample.id == resample_id:
            return resample
    raise ValueError(f"unknown resample id {resample_id!r}")


def _row_ids(case: BenchmarkCase, indices: np.ndarray) -> np.ndarray:
    if case.dataset.row_id is not None:
        return case.dataset.data.iloc[indices][case.dataset.row_id].to_numpy()
    return np.asarray(indices)


def _make_prediction_set(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    resample: Resample,
    estimator: Pipeline,
) -> PredictionSet:
    x_test = case.task.source_frame(case.dataset).iloc[resample.test_idx]
    y_test = case.task.target_series(case.dataset)
    y_true = None if y_test is None else y_test.iloc[resample.test_idx].to_numpy()

    if case.task.task_type == TaskType.REGRESSION:
        values = estimator.predict(x_test)
        return PredictionSet(
            dataset_name=case.dataset.name,
            task_name=case.task.name,
            method_name=method.name,
            resample_id=resample.id,
            row_ids=_row_ids(case, resample.test_idx),
            y_true=y_true,
            values=np.asarray(values),
            metadata={"case_name": case.name},
        )

    labels = estimator.predict(x_test)
    probabilities = estimator.predict_proba(x_test) if hasattr(estimator, "predict_proba") else None
    scores = estimator.decision_function(x_test) if hasattr(estimator, "decision_function") else None
    return PredictionSet(
        dataset_name=case.dataset.name,
        task_name=case.task.name,
        method_name=method.name,
        resample_id=resample.id,
        row_ids=_row_ids(case, resample.test_idx),
        y_true=y_true,
        labels=np.asarray(labels),
        probabilities=None if probabilities is None else np.asarray(probabilities),
        scores=None if scores is None else np.asarray(scores),
        metadata={"case_name": case.name},
    )


class SklearnBackend(MethodBackend):
    """Single-instance backend for estimators that follow the scikit-learn API."""

    name = "sklearn"
    supported_model_kinds = SUPPORTED_SKLEARN_MODEL_KINDS

    def run(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample_id: str | None = None,
        seed: int = 0,
        runtime: RuntimeSpec | None = None,
    ) -> BackendResult:
        case.task.validate_columns(case.dataset)
        resample = _find_resample(case, resample_id)

        transform_config = dict(method.transform)
        policy = transform_config.pop("policy", "linear_default")
        preprocessor = build_preprocessor(
            dataset=case.dataset,
            task=case.task,
            policy=policy,
            options=transform_config,
        )
        estimator = build_sklearn_estimator(
            task_type=case.task.task_type,
            method=method,
            seed=seed,
            runtime=runtime,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])

        x = case.task.source_frame(case.dataset)
        y = case.task.target_series(case.dataset)
        if y is None:
            raise ValueError("sklearn method execution requires a supervised task target")

        x_train = x.iloc[resample.train_idx]
        y_train = y.iloc[resample.train_idx]

        fit_start = perf_counter()
        pipeline.fit(x_train, y_train)
        fit_time = perf_counter() - fit_start

        predict_start = perf_counter()
        predictions = _make_prediction_set(
            case=case,
            method=method,
            resample=resample,
            estimator=pipeline,
        )
        predict_time = perf_counter() - predict_start

        metrics = compute_metrics(case.task.metrics, predictions)
        return BackendResult(
            predictions=predictions,
            metrics=metrics,
            fit_time=fit_time,
            predict_time=predict_time,
            metadata={"preprocessing_policy": policy},
        )
