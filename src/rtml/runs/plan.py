"""Run planning across studies and benchmark suites.

This module turns a `Study` or `BenchmarkSuite` into concrete `RunSpec` objects. The
actual execution APIs live in `rtml.runs.execution`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rtml.benchmarks.base import BenchmarkCase, BenchmarkSuite
from rtml.methods.base import MethodSpec
from rtml.runtime import RuntimeSpec
from rtml.studies.base import Study


@dataclass(frozen=True)
class ExecutionResources:
    """Scheduler-facing resource hints for one `RunSpec`.

    The fields mirror Ray task options where possible.
    They reserve resources for execution but do not tell a method backend which
    device, precision, or threading mode to use.
    """

    num_cpus: float | None = None
    num_gpus: float | None = None
    memory: int | None = None
    custom: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    """Planned input for one case/method/resample/seed execution."""

    case: BenchmarkCase
    method: MethodSpec
    resample_id: str
    seed: int = 0
    runtime: RuntimeSpec | None = None
    scheduler_resources: ExecutionResources = field(default_factory=ExecutionResources)


@dataclass(frozen=True)
class ExecutionPlan:
    """Materialized collection of `RunSpec` objects."""

    name: str
    runs: tuple[RunSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_suite(
        cls,
        *,
        name: str,
        suite: BenchmarkSuite,
        methods: Sequence[MethodSpec],
        seeds: Sequence[int] = (0,),
        runtime_specs: Mapping[str, RuntimeSpec] | None = None,
        scheduler_resources: Mapping[str, ExecutionResources] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Expand a suite and method list into concrete `RunSpec` objects.

        Runtime specs and scheduler resources are selected by exact method name.
        Missing entries mean the method/backend uses its library defaults.
        """
        run_specs: list[RunSpec] = []
        runtime_map = runtime_specs or {}
        resource_map = scheduler_resources or {}
        _validate_method_name_keys(methods=methods, mapping=runtime_map, label="runtime spec")
        _validate_method_name_keys(
            methods=methods,
            mapping=resource_map,
            label="scheduler resource",
        )
        for case in suite.cases:
            for method in methods:
                method_runtime = runtime_map.get(method.name)
                method_resources = resource_map.get(method.name, ExecutionResources())
                for resample in case.resampling.resamples:
                    for seed in seeds:
                        run_specs.append(
                            RunSpec(
                                case=case,
                                method=method,
                                resample_id=resample.id,
                                seed=seed,
                                runtime=method_runtime,
                                scheduler_resources=method_resources,
                            )
                        )
        return cls(name=name, runs=tuple(run_specs), metadata=dict(metadata or {}))

    @classmethod
    def from_study(
        cls,
        *,
        study: Study,
        seeds: Sequence[int] = (0,),
        runtime_specs: Mapping[str, RuntimeSpec] | None = None,
        scheduler_resources: Mapping[str, ExecutionResources] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionPlan:
        """Expand a study into concrete `RunSpec` objects with study metadata."""
        study_metadata = {
            **study.metadata,
            **dict(metadata or {}),
            "study_name": study.name,
            "study_kind": study.kind.value,
        }
        return cls.from_suite(
            name=study.name,
            suite=study.suite,
            methods=study.methods,
            seeds=seeds,
            runtime_specs=runtime_specs,
            scheduler_resources=scheduler_resources,
            metadata=study_metadata,
        )

def _validate_method_name_keys(
    *,
    methods: Sequence[MethodSpec],
    mapping: Mapping[str, object],
    label: str,
) -> None:
    valid_keys = {method.name for method in methods}
    unknown_keys = sorted(set(mapping) - valid_keys)
    if unknown_keys:
        valid = ", ".join(sorted(valid_keys)) or "<none>"
        raise ValueError(f"unknown {label} keys {unknown_keys}; valid method names: {valid}")
