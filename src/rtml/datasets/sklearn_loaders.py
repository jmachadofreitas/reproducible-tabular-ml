from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.utils import Bunch

from rtml.benchmarks.base import BenchmarkCase, BenchmarkSuite
from rtml.datasets.data import Dataset, FeatureInfo, FeatureKind, FeatureSchema
from rtml.resampling.base import Resample, ResamplingPlan, ResamplingSpec, ResamplingStrategy
from rtml.tasks.base import MetricSpec, TaskSpec, TaskType

SklearnLoader = Callable[..., Bunch]


def _load_frame(
    loader: SklearnLoader,
    *,
    name: str,
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.Series, Bunch]:
    """Load one sklearn dataset as a frame with a named target column."""
    bunch = loader(as_frame=True)

    data = bunch.data
    target = bunch.target

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"sklearn loader {name!r} did not return a pandas DataFrame")
    if not isinstance(target, pd.Series):
        raise TypeError(f"sklearn loader {name!r} did not return a pandas Series target")

    target = target.rename(target_column)
    return data, target, bunch


def _infer_target_kind(target: pd.Series) -> FeatureKind:
    unique_values = target.dropna().nunique()

    if pd.api.types.is_bool_dtype(target):
        return FeatureKind.BINARY
    if pd.api.types.is_numeric_dtype(target) and unique_values == 2:
        return FeatureKind.BINARY
    if pd.api.types.is_numeric_dtype(target):
        return FeatureKind.NUMERIC
    return FeatureKind.CATEGORICAL


def _build_task(
    *,
    name: str,
    dataset: Dataset,
    task_type: TaskType,
    metrics: list[MetricSpec],
    primary_metric: str,
) -> TaskSpec:
    """Create the task definition for one sklearn dataset."""
    task = TaskSpec(
        name=name,
        task_type=task_type,
        source=[column for column in dataset.data.columns if column != "target"],
        target="target",
        metrics=metrics,
        primary_metric=primary_metric,
        metadata={"source": "sklearn"},
    )
    task.validate_columns(dataset)
    return task


def load_sklearn_dataset(
    loader: SklearnLoader,
    *,
    name: str,
    target_column: str = "target",
    metadata: Mapping[str, Any] | None = None,
) -> Dataset:
    """Load a complete sklearn table and infer its schema from pandas dtypes.

    This helper is intentionally narrow. It is only for sklearn datasets where
    frame dtypes are reliable enough to infer a schema directly.
    """
    data, target, bunch = _load_frame(loader, name=name, target_column=target_column)
    frame = pd.concat([data, target], axis=1)

    if frame.isna().any().any():
        raise ValueError(
            "load_sklearn_dataset only supports sklearn datasets without missing values"
        )

    schema = FeatureSchema.infer(frame, binary_columns=[target_column])
    target_kind = _infer_target_kind(target)
    schema.features[target_column] = FeatureInfo(
        name=target_column,
        kind=target_kind,
        dtype=str(target.dtype),
    )

    dataset_metadata = {
        "source": "sklearn",
        "description": getattr(bunch, "DESCR", None),
        "target_column": target_column,
        **dict(metadata or {}),
    }
    return Dataset(name=name, data=frame, schema=schema, metadata=dataset_metadata)


def _load_all_numeric_dataset(
    loader: SklearnLoader,
    *,
    name: str,
    target_kind: FeatureKind,
    metadata: Mapping[str, Any] | None = None,
) -> Dataset:
    """Load one of the standard numeric sklearn benchmark tables."""
    data, target, bunch = _load_frame(loader, name=name)
    frame = pd.concat([data, target], axis=1)

    features = {
        column: FeatureInfo(
            name=column,
            kind=FeatureKind.NUMERIC,
            dtype=str(frame[column].dtype),
        )
        for column in data.columns
    }
    features["target"] = FeatureInfo(
        name="target",
        kind=target_kind,
        dtype=str(target.dtype),
    )

    dataset_metadata = {
        "source": "sklearn",
        "description": getattr(bunch, "DESCR", None),
        **dict(metadata or {}),
    }
    return Dataset(
        name=name,
        data=frame,
        schema=FeatureSchema(features=features),
        metadata=dataset_metadata,
    )


def load_breast_cancer_dataset() -> tuple[Dataset, TaskSpec]:
    dataset = _load_all_numeric_dataset(
        load_breast_cancer,  # type: ignore as_frame=True
        name="breast_cancer",
        target_kind=FeatureKind.BINARY,
    )
    task = _build_task(
        name="breast_cancer",
        dataset=dataset,
        task_type=TaskType.BINARY_CLASSIFICATION,
        metrics=[MetricSpec("accuracy"), MetricSpec("roc_auc")],
        primary_metric="accuracy",
    )
    return dataset, task


def load_iris_dataset() -> tuple[Dataset, TaskSpec]:
    dataset = _load_all_numeric_dataset(
        load_iris,  # type: ignore as_frame=True
        name="iris",
        target_kind=FeatureKind.CATEGORICAL,
    )
    task = _build_task(
        name="iris",
        dataset=dataset,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        metrics=[MetricSpec("accuracy")],
        primary_metric="accuracy",
    )
    return dataset, task


def load_wine_dataset() -> tuple[Dataset, TaskSpec]:
    dataset = _load_all_numeric_dataset(
        load_wine,  # type: ignore as_frame=True
        name="wine",
        target_kind=FeatureKind.CATEGORICAL,
    )
    task = _build_task(
        name="wine",
        dataset=dataset,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        metrics=[MetricSpec("accuracy")],
        primary_metric="accuracy",
    )
    return dataset, task


def load_diabetes_dataset() -> tuple[Dataset, TaskSpec]:
    dataset = _load_all_numeric_dataset(
        load_diabetes,  # type: ignore as_frame=True
        name="diabetes",
        target_kind=FeatureKind.NUMERIC,
    )
    task = _build_task(
        name="diabetes",
        dataset=dataset,
        task_type=TaskType.REGRESSION,
        metrics=[MetricSpec("rmse"), MetricSpec("mae")],
        primary_metric="rmse",
    )
    return dataset, task


def build_sklearn_resampling_spec(
    *,
    name: str,
    strategy: ResamplingStrategy | str,
    n_repeats: int = 1,
    n_folds: int = 1,
    n_samples: int = 1,
    test_size: float | None = None,
    valid_size: float | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    stratify: str | None = None,
    groups: list[str] | None = None,
    timestamp: str | None = None,
    replacement: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ResamplingSpec:
    """Build a local sklearn resampling specification."""
    return ResamplingSpec(
        name=name,
        strategy=ResamplingStrategy(strategy),
        n_repeats=n_repeats,
        n_folds=n_folds,
        n_samples=n_samples,
        test_size=test_size,
        valid_size=valid_size,
        shuffle=shuffle,
        seed=seed,
        stratify=stratify,
        groups=list(groups or []),
        timestamp=timestamp,
        replacement=replacement,
        metadata={"source": "sklearn", **dict(metadata or {})},
    )


def build_sklearn_resampling_plan(
    *,
    dataset: Dataset,
    task: TaskSpec,
    spec: ResamplingSpec,
) -> ResamplingPlan:
    """Materialize a saved resampling plan with sklearn split builders."""
    task.validate_columns(dataset)

    target = task.target_series(dataset)
    if target is None:
        raise ValueError("sklearn resampling requires a supervised task target")

    row_indices = dataset.data.index.to_numpy()
    resamples: list[Resample] = []

    if spec.strategy == ResamplingStrategy.HOLDOUT:
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed,
        )
        resamples.append(
            Resample(
                id="repeat_00",
                train_idx=train_idx,
                test_idx=test_idx,
                metadata={"source": "sklearn"},
            )
        )
    elif spec.strategy == ResamplingStrategy.STRATIFIED_HOLDOUT:
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed,
            stratify=target.to_numpy(),
        )
        resamples.append(
            Resample(
                id="repeat_00",
                train_idx=train_idx,
                test_idx=test_idx,
                metadata={"source": "sklearn"},
            )
        )
    elif spec.strategy == ResamplingStrategy.KFOLD:
        splitter = KFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_pos, test_pos) in enumerate(splitter.split(dataset.data)):
            resamples.append(
                Resample(
                    id=f"fold_{fold:02d}",
                    train_idx=row_indices[train_pos],
                    test_idx=row_indices[test_pos],
                    metadata={"source": "sklearn", "fold": fold},
                )
            )
    elif spec.strategy == ResamplingStrategy.STRATIFIED_KFOLD:
        splitter = StratifiedKFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_pos, test_pos) in enumerate(splitter.split(dataset.data, target)):
            resamples.append(
                Resample(
                    id=f"fold_{fold:02d}",
                    train_idx=row_indices[train_pos],
                    test_idx=row_indices[test_pos],
                    metadata={"source": "sklearn", "fold": fold},
                )
            )
    else:
        raise NotImplementedError(
            f"sklearn resampling plan builder does not support {spec.strategy.value}"
        )

    return ResamplingPlan(
        dataset_name=dataset.name,
        task_name=task.name,
        spec=spec,
        resamples=resamples,
        metadata={"source": "sklearn"},
    )


def build_sklearn_benchmark_case(
    *,
    name: str,
    dataset: Dataset,
    task: TaskSpec,
    resampling_spec: ResamplingSpec,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkCase:
    """Create one runnable sklearn benchmark case."""
    resampling = build_sklearn_resampling_plan(dataset=dataset, task=task, spec=resampling_spec)
    return BenchmarkCase(
        name=name,
        dataset=dataset,
        task=task,
        resampling=resampling,
        metadata={"source": "sklearn", **dict(metadata or {})},
    )


def build_sklearn_benchmark_suite(
    *,
    name: str,
    cases: list[BenchmarkCase],
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkSuite:
    """Collect multiple sklearn benchmark cases into one suite."""
    return BenchmarkSuite(
        name=name,
        cases=list(cases),
        metadata={"source": "sklearn", **dict(metadata or {})},
    )


def load_sklearn_classification_suite() -> BenchmarkSuite:
    """Create a small local sklearn classification suite."""
    resampling_spec = build_sklearn_resampling_spec(
        name="sklearn_classification_stratified_kfold",
        strategy=ResamplingStrategy.STRATIFIED_KFOLD,
        n_folds=5,
        stratify="target",
        shuffle=True,
        seed=42,
    )

    cases: list[BenchmarkCase] = []
    for loader in (
        load_breast_cancer_dataset,
        load_iris_dataset,
        load_wine_dataset,
    ):
        dataset, task = loader()
        cases.append(
            build_sklearn_benchmark_case(
                name=dataset.name,
                dataset=dataset,
                task=task,
                resampling_spec=resampling_spec,
            )
        )

    return build_sklearn_benchmark_suite(name="sklearn classification", cases=cases)
