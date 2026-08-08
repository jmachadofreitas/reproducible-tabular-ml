from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from rtml.core.resampling import ResamplingPlan

DatasetT = TypeVar("DatasetT")
TaskT = TypeVar("TaskT")


@dataclass
class BenchmarkCase(Generic[DatasetT, TaskT]):
    """Defines a benchmark case

    Combines the dataset, task specification, and resampling plan
    required to run a method.
    """

    name: str
    dataset: DatasetT
    task: TaskT
    resampling: ResamplingPlan
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark task name must be non-empty")
        self.metadata = dict(self.metadata or {})


@dataclass
class BenchmarkSuite(Generic[DatasetT, TaskT]):
    """A collection of benchmark tasks."""

    name: str
    cases: list[BenchmarkCase[DatasetT, TaskT]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark suite name must be non-empty")
        self.cases = list(self.cases)
        if not self.cases:
            raise ValueError("benchmark suite must contain at least one case")
        case_names = [case.name for case in self.cases]
        if len(case_names) != len(set(case_names)):
            raise ValueError(f"benchmark case names must be unique: {case_names}")
        self.metadata = dict(self.metadata or {})
