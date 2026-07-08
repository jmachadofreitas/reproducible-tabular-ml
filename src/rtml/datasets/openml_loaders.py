"""Load datasets, tasks, and resampling definitions from OpenML into the benchmark format."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import openml
import pandas as pd

from rtml.benchmarks.base import BenchmarkSuite, BenchmarkCase
from rtml.datasets.data import Dataset, FeatureInfo, FeatureKind, FeatureSchema
from rtml.resampling.base import (
    Resample,
    ResamplingPlan,
    ResamplingSpec,
    ResamplingStrategy,
    create_openml_resample_id,
)
from rtml.tasks.base import MetricSpec, TaskSpec, TaskType

OPENML_CC18_SUITE_ID = 99
DEFAULT_OPENML_SPLIT = {"repeat": 0, "fold": 0, "sample": 0}
DEFAULT_OPENML_DATA_DIR = Path("data/openml")


def configure_openml_storage(
    root_cache_directory: str | Path = DEFAULT_OPENML_DATA_DIR,
    *,
    show_progress: bool = True,
) -> Path:
    cache_dir = Path(root_cache_directory).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(cache_dir)
    openml.config.show_progress = show_progress
    return cache_dir


def _infer_feature_kind(
    series: pd.Series,
    *,
    is_categorical: bool,
) -> FeatureKind:
    if is_categorical:
        return FeatureKind.CATEGORICAL
    if pd.api.types.is_bool_dtype(series):
        return FeatureKind.BINARY
    if pd.api.types.is_numeric_dtype(series):
        return FeatureKind.NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return FeatureKind.TIMESTAMP
    return FeatureKind.UNKNOWN


def _infer_task_type(
    openml_task: openml.tasks.OpenMLSupervisedTask,
    target: pd.Series,
) -> TaskType:
    if openml_task.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION:
        return TaskType.REGRESSION
    if openml_task.task_type_id == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION:
        return (
            TaskType.BINARY_CLASSIFICATION
            if target.nunique(dropna=True) == 2
            else TaskType.MULTICLASS_CLASSIFICATION
        )
    raise ValueError(f"unsupported OpenML task type: {openml_task.task_type}")


def _infer_target_kind(task_type: TaskType) -> FeatureKind:
    if task_type == TaskType.REGRESSION:
        return FeatureKind.NUMERIC
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return FeatureKind.BINARY
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        return FeatureKind.CATEGORICAL
    return FeatureKind.UNKNOWN


def _build_schema(
    data: pd.DataFrame,
    *,
    feature_columns: list[str],
    categorical_indicator: list[bool],
    target_name: str,
    task_type: TaskType,
) -> FeatureSchema:
    features: dict[str, FeatureInfo] = {}

    for column, is_categorical in zip(feature_columns, categorical_indicator, strict=True):
        features[column] = FeatureInfo(
            name=column,
            kind=_infer_feature_kind(data[column], is_categorical=is_categorical),
            dtype=str(data[column].dtype),
        )

    features[target_name] = FeatureInfo(
        name=target_name,
        kind=_infer_target_kind(task_type),
        dtype=str(data[target_name].dtype),
    )
    return FeatureSchema(features=features)


def _build_metric_specs(
    openml_task: openml.tasks.OpenMLSupervisedTask,
) -> tuple[list[MetricSpec], str | None]:
    measure = openml_task.evaluation_measure

    if measure is None:
        return [], None

    # Define others?
    # ...

    return [MetricSpec(name=measure)], measure


def _build_task_spec(
    dataset: Dataset,
    *,
    openml_task: openml.tasks.OpenMLSupervisedTask,
    task_type: TaskType,
    target_name: str,
) -> TaskSpec:
    metrics, primary_metric = _build_metric_specs(openml_task)
    task = TaskSpec(
        name=f"{dataset.name}_{target_name}",
        task_type=task_type,
        source=[column for column in dataset.columns if column != target_name],
        target=target_name,
        metrics=metrics,
        primary_metric=primary_metric,
        metadata={
            "source": "openml",
            "openml_task_id": openml_task.task_id,
            "openml_dataset_id": openml_task.dataset_id,
            "evaluation_measure": openml_task.evaluation_measure,
            "estimation_procedure": dict(openml_task.estimation_procedure),
        },
    )
    task.validate_columns(dataset)
    return task


def _build_resampling_spec(
    openml_task: openml.tasks.OpenMLSupervisedTask,
    *,
    split_dimensions: tuple[int, int, int],
) -> ResamplingSpec:
    repeats, folds, samples = split_dimensions
    return ResamplingSpec(
        name=f"openml_task_{openml_task.task_id}",
        strategy=ResamplingStrategy.UNKNOWN_OPENML_TASK,
        n_repeats=repeats,
        n_folds=folds,
        n_samples=samples,
        metadata={
            "source": "openml",
            "openml_task_id": openml_task.task_id,
            "estimation_procedure": dict(openml_task.estimation_procedure),
        },
    )


def _build_resampling_plan(
    openml_task: openml.tasks.OpenMLSupervisedTask,
    *,
    dataset_name: str,
    task_name: str,
) -> ResamplingPlan:
    split_dimensions = openml_task.get_split_dimensions()
    spec = _build_resampling_spec(openml_task, split_dimensions=split_dimensions)
    repeats, folds, samples = split_dimensions
    resamples: list[Resample] = []

    # Materialize every saved OpenML split so the benchmark definition is fully local and explicit.
    for repeat in range(repeats):
        for fold in range(folds):
            for sample in range(samples):
                train_idx, test_idx = openml_task.get_train_test_split_indices(
                    repeat=repeat,
                    fold=fold,
                    sample=sample,
                )
                resamples.append(
                    Resample(
                        id=create_openml_resample_id(repeat=repeat, fold=fold, sample=sample),
                        train_idx=train_idx,
                        valid_idx=None,
                        test_idx=test_idx,
                        metadata={"repeat": repeat, "fold": fold, "sample": sample},
                    )
                )

    return ResamplingPlan(
        dataset_name=dataset_name,
        task_name=task_name,
        spec=spec,
        resamples=resamples,
        metadata={
            "source": "openml",
            "openml_task_id": openml_task.task_id,
            "default_split": dict(DEFAULT_OPENML_SPLIT),
        },
    )


def get_openml_suite(suite_id: int = OPENML_CC18_SUITE_ID) -> openml.study.OpenMLBenchmarkSuite:
    return openml.study.get_suite(suite_id)


def get_openml_suite_task_ids(suite_id: int = OPENML_CC18_SUITE_ID) -> list[int]:
    suite = get_openml_suite(suite_id)
    if suite.tasks is None:
        return []
    return [int(task_id) for task_id in suite.tasks]


def get_openml_task_split_indices(
    task_id: int,
    *,
    repeat: int = 0,
    fold: int = 0,
    sample: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    task = cast(
        openml.tasks.OpenMLSupervisedTask,
        openml.tasks.get_task(task_id, download_splits=True),
    )
    return task.get_train_test_split_indices(repeat=repeat, fold=fold, sample=sample)


def load_benchmark_case(
    task_id: int,
    *,
    suite_id: int | None = None,
) -> BenchmarkCase:
    openml_task = cast(
        openml.tasks.OpenMLSupervisedTask,
        openml.tasks.get_task(task_id, download_splits=True),
    )
    openml_dataset = openml_task.get_dataset()

    x, y, categorical_indicator, attribute_names = openml_dataset.get_data(
        dataset_format="dataframe",
        target=openml_task.target_name,
    )
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=attribute_names)  # type: ignore
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("OpenML supervised tasks must expose a single target column")
        y = y.iloc[:, 0]
    elif y is None:
        raise ValueError("OpenML supervised tasks must expose a target")
    elif not isinstance(y, pd.Series):
        y = pd.Series(y, name=openml_task.target_name)

    target_name = str(openml_task.target_name)
    data = pd.concat(
        [x.reset_index(drop=True), y.rename(target_name).reset_index(drop=True)], axis=1
    )
    feature_columns = [str(column) for column in x.columns]
    task_type = _infer_task_type(openml_task, data[target_name])
    schema = _build_schema(
        data,
        feature_columns=feature_columns,
        categorical_indicator=list(categorical_indicator),
        target_name=target_name,
        task_type=task_type,
    )

    dataset = Dataset(
        name=getattr(openml_dataset, "name", f"openml_dataset_{openml_task.dataset_id}"),
        data=data,
        schema=schema,
        metadata={
            "source": "openml",
            "openml_dataset_id": openml_task.dataset_id,
            "openml_task_id": openml_task.task_id,
            "suite_id": suite_id,
        },
    )
    task = _build_task_spec(
        dataset,
        openml_task=openml_task,
        task_type=task_type,
        target_name=target_name,
    )
    resampling = _build_resampling_plan(
        openml_task,
        dataset_name=dataset.name,
        task_name=task.name,
    )

    benchmark_case = BenchmarkCase(
        name=f"openml_task_{openml_task.task_id}",
        dataset=dataset,
        task=task,
        resampling=resampling,
        metadata={
            "source": "openml",
            "openml_task_id": openml_task.task_id,
            "openml_dataset_id": openml_task.dataset_id,
            "suite_id": suite_id,
            "evaluation_measure": openml_task.evaluation_measure,
            "estimation_procedure_type": openml_task.estimation_procedure["type"],
            "split_dimensions": openml_task.get_split_dimensions(),
        },
    )
    return benchmark_case


def load_openml_task(task_id: int) -> tuple[Dataset, TaskSpec]:
    benchmark_case = load_benchmark_case(task_id)
    return benchmark_case.dataset, benchmark_case.task


def load_openml_suite(suite_id: int = OPENML_CC18_SUITE_ID) -> BenchmarkSuite:
    suite = get_openml_suite(suite_id)
    task_ids = get_openml_suite_task_ids(suite_id)
    benchmark_cases = [load_benchmark_case(task_id, suite_id=suite_id) for task_id in task_ids]
    return BenchmarkSuite(
        name=getattr(suite, "name", f"openml_suite_{suite_id}"),
        cases=benchmark_cases,
        metadata={
            "source": "openml",
            "suite_id": suite_id,
            "description": getattr(suite, "description", None),
            "task_ids": task_ids,
        },
    )


def load_openml_cc18_task(task_id: int) -> tuple[Dataset, TaskSpec]:
    suite_task_ids = get_openml_suite_task_ids(OPENML_CC18_SUITE_ID)
    if task_id not in suite_task_ids:
        raise ValueError(
            f"task_id {task_id} is not part of the OpenML-CC18 suite {OPENML_CC18_SUITE_ID}"
        )
    benchmark_case = load_benchmark_case(task_id, suite_id=OPENML_CC18_SUITE_ID)
    return benchmark_case.dataset, benchmark_case.task
