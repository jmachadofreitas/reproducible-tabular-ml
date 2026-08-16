from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rtml.core.benchmarks import BenchmarkSuite
from rtml.core.methods import MethodSpec


class StudyKind(StrEnum):
    """Intent of a method-comparison study."""

    COMPARISON = "comparison"
    ABLATION = "ablation"
    SENSITIVITY = "sensitivity"
    FACTORIAL = "factorial"


@dataclass
class Study:
    """A methodological comparison over a benchmark suite."""

    name: str
    suite: BenchmarkSuite
    methods: list[MethodSpec]
    kind: StudyKind = StudyKind.COMPARISON
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("study name must be non-empty")
        self.methods = list(self.methods)
        if not self.methods:
            raise ValueError("study must define at least one method")
        method_names = [method.name for method in self.methods]
        duplicate_names = sorted({name for name in method_names if method_names.count(name) > 1})
        if duplicate_names:
            raise ValueError(f"study method names must be unique: {duplicate_names}")
        self.kind = StudyKind(self.kind)
        self.metadata = dict(self.metadata or {})
