from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from rtml.core.tasks import MetricSpec, TaskType
from rtml.multi_instance.datasets.base import MultiInstanceDataset


@dataclass
class MultiInstanceTask:
    """Bag-level supervised task over instance-level input features."""

    name: str
    task_type: TaskType
    instance_source: list[str]
    target: str | None = None
    bag_source: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    sensitive_attributes: list[str] = field(default_factory=list)
    metrics: list[MetricSpec] = field(default_factory=list)
    primary_metric: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("task name must be non-empty")

        self.task_type = TaskType(self.task_type)
        self.instance_source = list(self.instance_source)
        self.bag_source = list(self.bag_source)
        self.groups = list(self.groups)
        self.sensitive_attributes = list(self.sensitive_attributes)
        self.metrics = list(self.metrics)
        self.metadata = dict(self.metadata or {})

        if not self.instance_source and self.task_type != TaskType.UNSUPERVISED:
            raise ValueError("supervised MIL tasks must define instance input columns")
        if self.task_type == TaskType.UNSUPERVISED and self.target is not None:
            raise ValueError("unsupervised tasks must not define a target")
        if self.task_type != TaskType.UNSUPERVISED and self.target is None:
            raise ValueError("supervised tasks must define a bag-level target")

        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError(f"metric names must be unique: {metric_names}")
        if self.primary_metric is not None and self.primary_metric not in metric_names:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} is not present in metrics {metric_names}"
            )

        reserved = [
            column
            for column in (self.target, *self.groups, *self.sensitive_attributes)
            if column is not None
        ]
        overlapping_bag_inputs = sorted(set(self.bag_source).intersection(reserved))
        if overlapping_bag_inputs:
            raise ValueError(
                f"bag input columns cannot be reused for target/control roles: "
                f"{overlapping_bag_inputs}"
            )

    @property
    def required_bag_columns(self) -> list[str]:
        columns: list[str] = [
            *self.bag_source,
            *self.groups,
            *self.sensitive_attributes,
        ]
        if self.target is not None:
            columns.append(self.target)
        return columns

    def validate_columns(self, dataset: MultiInstanceDataset) -> None:
        dataset.require_bag_columns(self.required_bag_columns)
        dataset.require_instance_columns(self.instance_source)

    def bag_frame(self, dataset: MultiInstanceDataset) -> pd.DataFrame:
        self.validate_columns(dataset)
        return dataset.bag_table.loc[:, self.bag_source]

    def target_series(self, dataset: MultiInstanceDataset) -> pd.Series | None:
        self.validate_columns(dataset)
        if self.target is None:
            return None
        return dataset.bag_table.loc[:, self.target]
