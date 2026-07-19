from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

from rtml.core.datasets import Dataset, FeatureKind, FeatureTag
from rtml.core.tasks import TaskSpec

PreprocessorBuilder = Callable[[Dataset, TaskSpec, Mapping[str, Any]], ColumnTransformer]


def _one_hot_encoder(**kwargs: Any) -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)


def _task_source_columns(
    dataset: Dataset,
    task: TaskSpec,
    *,
    kinds: list[FeatureKind],
    include_tags: list[FeatureTag | str] | None = None,
    exclude_tags: list[FeatureTag | str] | None = None,
    require_all_tags: bool = True,
) -> list[str]:
    task.validate_columns(dataset)
    selected = set(
        dataset.select(
            kinds=kinds,
            include_tags=include_tags or [],
            exclude_tags=exclude_tags or [],
            require_all_tags=require_all_tags,
        )
    )
    return [column for column in task.source if column in selected]


def _validate_supported_source_columns(dataset: Dataset, task: TaskSpec) -> None:
    supported = set(
        dataset.select(
            kinds=[
                FeatureKind.NUMERIC,
                FeatureKind.CATEGORICAL,
                FeatureKind.BINARY,
                FeatureKind.ORDINAL,
            ]
        )
    )
    unsupported = [column for column in task.source if column not in supported]
    if unsupported:
        details = ", ".join(
            f"{column}={dataset.schema.get(column).kind.value}" for column in unsupported
        )
        raise ValueError(f"unsupported source columns for preprocessing policy: {details}")


def _validate_missing_value_tags(dataset: Dataset, task: TaskSpec) -> None:
    untagged_missing = [
        column
        for column in task.source
        if dataset.data[column].isna().any()
        and FeatureTag.MISSING_VALUES not in dataset.schema.get(column).tags
    ]
    if untagged_missing:
        raise ValueError(
            "columns contain missing values but are not tagged with "
            f"{FeatureTag.MISSING_VALUES.value!r}: {untagged_missing}"
        )


def _with_optional_imputer(
    steps: list[tuple[str, Any]],
    *,
    use_imputer: bool,
    strategy: str,
) -> Pipeline:
    if use_imputer:
        steps = [("imputer", SimpleImputer(strategy=strategy)), *steps]
    return Pipeline(steps=steps)


def _standard_numeric_pipeline(options: Mapping[str, Any], *, use_imputer: bool) -> Pipeline:
    return _with_optional_imputer(
        [("scaler", StandardScaler())],
        use_imputer=use_imputer,
        strategy=options.get("numeric_impute", "median"),
    )


def _skewed_numeric_pipeline(options: Mapping[str, Any], *, use_imputer: bool) -> Pipeline:
    return _with_optional_imputer(
        [
            (
                "power",
                PowerTransformer(
                    method=options.get("skewed_numeric_transform", "yeo-johnson"),
                    standardize=True,
                ),
            )
        ],
        use_imputer=use_imputer,
        strategy=options.get("numeric_impute", "median"),
    )


def _zero_inflated_numeric_pipeline(options: Mapping[str, Any], *, use_imputer: bool) -> Pipeline:
    return _with_optional_imputer(
        [("scaler", MaxAbsScaler())],
        use_imputer=use_imputer,
        strategy=options.get("numeric_impute", "median"),
    )


def _one_hot_pipeline(options: Mapping[str, Any], *, use_imputer: bool) -> Pipeline:
    return _with_optional_imputer(
        [("encoder", _one_hot_encoder(handle_unknown="ignore"))],
        use_imputer=use_imputer,
        strategy=options.get("categorical_impute", "most_frequent"),
    )


def _infrequent_one_hot_pipeline(options: Mapping[str, Any], *, use_imputer: bool) -> Pipeline:
    return _with_optional_imputer(
        [
            (
                "encoder",
                _one_hot_encoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=options.get("high_cardinality_min_frequency"),
                    max_categories=options.get("high_cardinality_max_categories", 20),
                ),
            )
        ],
        use_imputer=use_imputer,
        strategy=options.get("categorical_impute", "most_frequent"),
    )


def _ordinal_pipeline(options: Mapping[str, Any], *, use_imputer: bool) -> Pipeline:
    return _with_optional_imputer(
        [
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    min_frequency=options.get("high_cardinality_min_frequency"),  # type: ignore
                    max_categories=options.get("high_cardinality_max_categories"),  # type: ignore
                ),
            )
        ],
        use_imputer=use_imputer,
        strategy=options.get("categorical_impute", "most_frequent"),
    )


def _add_pipeline(
    transformers: list[tuple[str, Pipeline, list[str]]],
    name: str,
    columns: list[str],
    pipeline: Pipeline,
) -> None:
    if columns:
        transformers.append((name, pipeline, columns))


def build_linear_default_policy(
    dataset: Dataset,
    task: TaskSpec,
    options: Mapping[str, Any],
) -> ColumnTransformer:
    _validate_supported_source_columns(dataset, task)
    _validate_missing_value_tags(dataset, task)
    regular_numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        exclude_tags=[
            FeatureTag.SKEWED,
            FeatureTag.ZERO_INFLATED,
            FeatureTag.MISSING_VALUES,
        ],
    )
    regular_missing_numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        include_tags=[FeatureTag.MISSING_VALUES],
        exclude_tags=[FeatureTag.SKEWED, FeatureTag.ZERO_INFLATED],
    )
    skewed_numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        include_tags=[FeatureTag.SKEWED],
        exclude_tags=[FeatureTag.MISSING_VALUES],
    )
    skewed_missing_numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        include_tags=[FeatureTag.SKEWED, FeatureTag.MISSING_VALUES],
    )
    zero_inflated_numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        include_tags=[FeatureTag.ZERO_INFLATED],
        exclude_tags=[FeatureTag.SKEWED, FeatureTag.MISSING_VALUES],
    )
    zero_inflated_missing_numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        include_tags=[FeatureTag.ZERO_INFLATED, FeatureTag.MISSING_VALUES],
        exclude_tags=[FeatureTag.SKEWED],
    )
    regular_categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        exclude_tags=[FeatureTag.HIGH_CARDINALITY, FeatureTag.MISSING_VALUES],
    )
    regular_missing_categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        include_tags=[FeatureTag.MISSING_VALUES],
        exclude_tags=[FeatureTag.HIGH_CARDINALITY],
    )
    high_cardinality_categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        include_tags=[FeatureTag.HIGH_CARDINALITY],
        exclude_tags=[FeatureTag.MISSING_VALUES],
    )
    high_cardinality_missing_categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        include_tags=[FeatureTag.HIGH_CARDINALITY, FeatureTag.MISSING_VALUES],
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    _add_pipeline(
        transformers,
        "numeric",
        regular_numeric_columns,
        _standard_numeric_pipeline(options, use_imputer=False),
    )
    _add_pipeline(
        transformers,
        "numeric_missing",
        regular_missing_numeric_columns,
        _standard_numeric_pipeline(options, use_imputer=True),
    )
    _add_pipeline(
        transformers,
        "numeric_skewed",
        skewed_numeric_columns,
        _skewed_numeric_pipeline(options, use_imputer=False),
    )
    _add_pipeline(
        transformers,
        "numeric_skewed_missing",
        skewed_missing_numeric_columns,
        _skewed_numeric_pipeline(options, use_imputer=True),
    )
    _add_pipeline(
        transformers,
        "numeric_zero_inflated",
        zero_inflated_numeric_columns,
        _zero_inflated_numeric_pipeline(options, use_imputer=False),
    )
    _add_pipeline(
        transformers,
        "numeric_zero_inflated_missing",
        zero_inflated_missing_numeric_columns,
        _zero_inflated_numeric_pipeline(options, use_imputer=True),
    )
    _add_pipeline(
        transformers,
        "categorical",
        regular_categorical_columns,
        _one_hot_pipeline(options, use_imputer=False),
    )
    _add_pipeline(
        transformers,
        "categorical_missing",
        regular_missing_categorical_columns,
        _one_hot_pipeline(options, use_imputer=True),
    )
    _add_pipeline(
        transformers,
        "categorical_high_cardinality",
        high_cardinality_categorical_columns,
        _infrequent_one_hot_pipeline(options, use_imputer=False),
    )
    _add_pipeline(
        transformers,
        "categorical_high_cardinality_missing",
        high_cardinality_missing_categorical_columns,
        _infrequent_one_hot_pipeline(options, use_imputer=True),
    )

    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def build_tree_default_policy(
    dataset: Dataset,
    task: TaskSpec,
    options: Mapping[str, Any],
) -> ColumnTransformer:
    _validate_supported_source_columns(dataset, task)
    _validate_missing_value_tags(dataset, task)
    numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        exclude_tags=[FeatureTag.MISSING_VALUES],
    )
    missing_numeric_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.NUMERIC],
        include_tags=[FeatureTag.MISSING_VALUES],
    )
    categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        exclude_tags=[FeatureTag.HIGH_CARDINALITY, FeatureTag.MISSING_VALUES],
    )
    missing_categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        include_tags=[FeatureTag.MISSING_VALUES],
        exclude_tags=[FeatureTag.HIGH_CARDINALITY],
    )
    high_cardinality_categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        include_tags=[FeatureTag.HIGH_CARDINALITY],
        exclude_tags=[FeatureTag.MISSING_VALUES],
    )
    high_cardinality_missing_categorical_columns = _task_source_columns(
        dataset,
        task,
        kinds=[FeatureKind.CATEGORICAL, FeatureKind.BINARY, FeatureKind.ORDINAL],
        include_tags=[FeatureTag.HIGH_CARDINALITY, FeatureTag.MISSING_VALUES],
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    _add_pipeline(
        transformers,
        "numeric",
        numeric_columns,
        Pipeline(steps=[("identity", FunctionTransformer(feature_names_out="one-to-one"))]),
    )
    _add_pipeline(
        transformers,
        "numeric_missing",
        missing_numeric_columns,
        _with_optional_imputer(
            [],
            use_imputer=True,
            strategy=options.get("numeric_impute", "median"),
        ),
    )
    for name, columns in (
        ("categorical", categorical_columns),
        ("categorical_missing", missing_categorical_columns),
        ("categorical_high_cardinality", high_cardinality_categorical_columns),
        ("categorical_high_cardinality_missing", high_cardinality_missing_categorical_columns),
    ):
        if not columns:
            continue
        use_imputer = name.endswith("_missing")
        transformers.append(
            (
                name,
                _ordinal_pipeline(
                    options,
                    use_imputer=use_imputer,
                ),
                columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )


def build_neural_default_policy(
    dataset: Dataset,
    task: TaskSpec,
    options: Mapping[str, Any],
) -> ColumnTransformer:
    return build_linear_default_policy(dataset, task, options)


PREPROCESSING_POLICIES: dict[str, PreprocessorBuilder] = {
    "linear_default": build_linear_default_policy,
    "tree_default": build_tree_default_policy,
    "neural_default": build_neural_default_policy,
}


def get_preprocessing_policy(name: str) -> PreprocessorBuilder:
    try:
        return PREPROCESSING_POLICIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(PREPROCESSING_POLICIES))
        raise KeyError(f"unknown preprocessing policy {name!r}; known policies: {known}") from exc


def build_preprocessor(
    *,
    dataset: Dataset,
    task: TaskSpec,
    policy: str,
    options: Mapping[str, Any] | None = None,
) -> ColumnTransformer:
    builder = get_preprocessing_policy(policy)
    return builder(dataset, task, dict(options or {}))
