from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rtml.datasets.data import Dataset


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNSUPERVISED = "unsupervised"


@dataclass
class MetricSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must be non-empty")
        self.kwargs = dict(self.kwargs or {})


@dataclass
class TaskSpec:
    name: str
    task_type: TaskType

    source: list[str]
    target: str | None = None

    # Auxiliary
    sample_weight: str | None = None
    groups: list[str] = field(default_factory=list)
    timestamp: str | None = None
    sensitive_attributes: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    # Metrics
    metrics: list[MetricSpec] = field(default_factory=list)
    primary_metric: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must be non-empty")

        self.task_type = TaskType(self.task_type)
        self.source = list(self.source)
        self.metrics = list(self.metrics)
        self.groups = list(self.groups)
        self.sensitive_attributes = list(self.sensitive_attributes)
        self.context = list(self.context)
        self.metadata = dict(self.metadata or {})

        if not self.source and self.task_type != TaskType.UNSUPERVISED:
            raise ValueError("supervised tasks must define at least one input column")
        if self.task_type == TaskType.UNSUPERVISED and self.target is not None:
            raise ValueError("unsupervised tasks must not define a target")
        if self.task_type != TaskType.UNSUPERVISED and self.target is None:
            raise ValueError("supervised tasks must define a target")

        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError(f"metric names must be unique: {metric_names}")
        if self.primary_metric is not None and self.primary_metric not in metric_names:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} is not present in metrics {metric_names}"
            )

        reserved = [
            column
            for column in (
                self.target,
                self.sample_weight,
                self.timestamp,
                *self.groups,
                *self.sensitive_attributes,
            )
            if column is not None
        ]
        overlapping_inputs = sorted(set(self.source).intersection(reserved))
        if overlapping_inputs:
            raise ValueError(
                f"input columns cannot be reused for target/control roles: {overlapping_inputs}"
            )

    @property
    def required_columns(self) -> list[str]:
        columns: list[str] = [*self.source]
        if self.target is not None:
            columns.append(self.target)
        if self.sample_weight is not None:
            columns.append(self.sample_weight)
        if self.timestamp is not None:
            columns.append(self.timestamp)
        columns.extend(self.groups)
        columns.extend(self.sensitive_attributes)
        columns.extend(self.context)
        return columns

    def validate_columns(self, dataset: Dataset) -> None:
        dataset.require_columns(self.required_columns)

    def source_frame(self, dataset: Dataset):
        self.validate_columns(dataset)
        return dataset.data.loc[:, self.source]

    def target_series(self, dataset: Dataset):
        self.validate_columns(dataset)
        if self.target is None:
            return None
        return dataset.data.loc[:, self.target]
