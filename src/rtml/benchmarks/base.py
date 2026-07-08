from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rtml.datasets.data import Dataset
from rtml.resampling.base import ResamplingPlan
from rtml.tasks.base import TaskSpec


@dataclass
class BenchmarkCase:
    """Defines a benchmark case

    Combines the dataset, task specification, and resampling plan
    required to run a method.
    """

    name: str
    dataset: Dataset
    task: TaskSpec
    resampling: ResamplingPlan
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark task name must be non-empty")
        self.metadata = dict(self.metadata or {})


@dataclass
class BenchmarkSuite:
    """A collection of benchmark tasks."""

    name: str
    cases: list[BenchmarkCase]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark suite name must be non-empty")
        self.cases = list(self.cases)
        self.metadata = dict(self.metadata or {})
