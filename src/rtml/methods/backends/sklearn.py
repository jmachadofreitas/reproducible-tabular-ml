from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.pipeline import Pipeline

from rtml.benchmarks.base import BenchmarkCase
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.methods.base import MethodSpec
from rtml.methods.models.sklearn import build_sklearn_estimator
from rtml.preprocessing import build_preprocessor
from rtml.resampling.base import Resample
from rtml.results.base import PredictionSet
from rtml.tasks.base import TaskType
from rtml.tasks.metrics import compute_metrics


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
    scores = (
        estimator.decision_function(x_test) if hasattr(estimator, "decision_function") else None
    )
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
    """Method backend for estimators that follow the scikit-learn API."""

    name = "sklearn"

    def run(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample_id: str | None = None,
        seed: int = 0,
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
