"""Deterministic bag-feature aggregation for fixed-width sklearn methods."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from rtml.core.datasets import Dataset, FeatureInfo, FeatureKind, FeatureSchema, FeatureTag
from rtml.core.tasks import TaskSpec
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.tasks import MultiInstanceTask

DEFAULT_STATISTICS = ("mean", "std", "min", "max")


def build_bag_feature_dataset(
    dataset: MultiInstanceDataset,
    task: MultiInstanceTask,
    *,
    policy: str = "summary_default",
    options: Mapping[str, Any] | None = None,
) -> tuple[Dataset, TaskSpec]:
    """Convert a multi-instance dataset into one fixed feature row per bag."""
    if policy != "summary_default":
        raise ValueError(f"unsupported bag aggregation policy {policy!r}")
    config = dict(options or {})
    statistics = _statistics(config.pop("statistics", DEFAULT_STATISTICS))
    include_size = bool(config.pop("include_size", True))
    if config:
        unknown = ", ".join(sorted(config))
        raise ValueError(f"unknown summary_default aggregation options: {unknown}")

    task.validate_columns(dataset)
    unsupported = [
        column
        for column in task.instance_source
        if dataset.instance_schema.get(column).kind != FeatureKind.NUMERIC
    ]
    if unsupported:
        raise ValueError(
            f"summary_default currently supports numeric instance features only: {unsupported}"
        )

    summary = _aggregate_numeric_bags(
        dataset,
        task.instance_source,
        statistics=statistics,
        include_size=include_size,
    )
    overlap = sorted(set(summary).intersection(dataset.bag_columns))
    if overlap:
        raise ValueError(f"aggregated feature names overlap bag columns: {overlap}")
    frame = pd.concat([dataset.bag_table.reset_index(drop=True), summary], axis=1)
    schema = FeatureSchema(
        {
            **dataset.bag_schema.features,
            **{
                column: FeatureInfo(
                    name=column,
                    kind=FeatureKind.NUMERIC,
                    dtype=str(frame[column].dtype),
                    tags={FeatureTag.MISSING_VALUES} if frame[column].isna().any() else set(),
                )
                for column in summary.columns
            },
        }
    )
    source = [*task.bag_source, *summary.columns]
    return (
        Dataset(
            name=dataset.name,
            data=frame,
            schema=schema,
            row_id=dataset.bag_id_column,
            metadata={
                **dataset.metadata,
                "paradigm": "single_instance",
                "source_paradigm": "multi_instance",
                "aggregation_policy": policy,
            },
        ),
        TaskSpec(
            name=task.name,
            task_type=task.task_type,
            source=source,
            target=task.target,
            groups=task.groups,
            metrics=task.metrics,
            primary_metric=task.primary_metric,
            metadata={
                **task.metadata,
                "source_paradigm": "multi_instance",
                "aggregation_policy": policy,
            },
        ),
    )


def _aggregate_numeric_bags(
    dataset: MultiInstanceDataset,
    columns: Sequence[str],
    *,
    statistics: Mapping[str, Callable[[pd.DataFrame], pd.Series]],
    include_size: bool,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for bag_position in range(dataset.n_bags):
        instances = dataset.bag_instances(bag_position).loc[:, columns]
        row = {
            f"{column}__{name}": float(value)
            for name, statistic in statistics.items()
            for column, value in statistic(instances).items()
        }
        if include_size:
            row["bag_size"] = float(len(instances))
        rows.append(row)
    return pd.DataFrame(rows)


def _statistics(
    names: Any,
) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    if not isinstance(names, Sequence) or isinstance(names, str | bytes):
        raise TypeError("aggregation statistics must be a sequence of names")
    available: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "mean": lambda frame: frame.mean(axis=0),
        "std": lambda frame: frame.std(axis=0, ddof=0),
        "min": lambda frame: frame.min(axis=0),
        "max": lambda frame: frame.max(axis=0),
    }
    selected = {}
    for name in names:
        key = str(name)
        try:
            selected[key] = available[key]
        except KeyError as exc:
            known = ", ".join(available)
            raise ValueError(
                f"unknown bag aggregation statistic {key!r}; known statistics: {known}"
            ) from exc
    if not selected:
        raise ValueError("bag aggregation requires at least one statistic")
    return selected


__all__ = ["DEFAULT_STATISTICS", "build_bag_feature_dataset"]
