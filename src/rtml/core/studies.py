from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Sequence
from typing import Any

from rtml.core.benchmarks import BenchmarkCase, BenchmarkSuite
from rtml.methods.base import MethodSpec


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

    @classmethod
    def from_suite(
        cls,
        *,
        name: str | None = None,
        suite: BenchmarkSuite,
        methods: Sequence[MethodSpec],
        kind: StudyKind = StudyKind.COMPARISON,
        metadata: dict[str, Any] | None = None,
    ) -> Study:
        """Create a study from a benchmark suite and complete methods."""
        return cls(
            name=name or suite.name,
            suite=suite,
            methods=list(methods),
            kind=kind,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_case(
        cls,
        *,
        name: str | None = None,
        case: BenchmarkCase,
        methods: Sequence[MethodSpec],
        kind: StudyKind = StudyKind.COMPARISON,
        metadata: dict[str, Any] | None = None,
    ) -> Study:
        """Create a study from one benchmark case."""
        suite = BenchmarkSuite(name=case.name, cases=[case])
        return cls.from_suite(
            name=name or case.name,
            suite=suite,
            methods=methods,
            kind=kind,
            metadata=metadata,
        )

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
