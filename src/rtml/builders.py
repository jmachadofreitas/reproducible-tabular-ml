"""Small builders for constructing RTML specs from plain Python mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rtml.benchmarks.base import BenchmarkSuite
from rtml.loggers.base import RunLogger
from rtml.loggers.mlflow import MLflowLogger
from rtml.methods.base import MethodSpec, ModelSpec
from rtml.runs import RayExecutor, ExecutionResources, RunExecutor, SequentialExecutor
from rtml.runtime import RuntimeSpec
from rtml.studies import Study, StudyKind


def build_methods(config: Sequence[Mapping[str, Any]]) -> list[MethodSpec]:
    """Build complete method specs from Hydra/notebook-friendly dictionaries."""
    methods = []
    for method in config:
        model = dict(method["model"])
        methods.append(
            MethodSpec(
                name=str(method["name"]),
                transform=dict(method.get("transform", {})),
                model=ModelSpec(
                    kind=str(model["kind"]),
                    backend=str(model["backend"]),
                    params=dict(model.get("params", {})),
                ),
                training=dict(method.get("training", {})),
                metadata=dict(method.get("metadata", {})),
            )
        )
    if not methods:
        raise ValueError("config must define at least one method")
    return methods


def build_study(
    config: Mapping[str, Any],
    *,
    suite: BenchmarkSuite,
    methods: Sequence[MethodSpec],
    default_name: str,
) -> Study:
    """Build a study around an already constructed benchmark suite."""
    return Study(
        name=str(config.get("name", default_name)),
        suite=suite,
        methods=list(methods),
        kind=StudyKind(str(config.get("kind", StudyKind.COMPARISON.value))),
        metadata=dict(config.get("metadata", {})),
    )


def build_scheduler_resources(
    config: Mapping[str, Mapping[str, Any]],
) -> dict[str, ExecutionResources]:
    """Build scheduler resource hints keyed by exact method name."""
    return {
        name: ExecutionResources(
            num_cpus=resource.get("num_cpus"),
            num_gpus=resource.get("num_gpus"),
            memory=resource.get("memory"),
            custom=dict(resource.get("custom", {})),
        )
        for name, resource in config.items()
    }


def build_runtime_specs(config: Mapping[str, Mapping[str, Any]]) -> dict[str, RuntimeSpec]:
    """Build runtime hints keyed by exact method name."""
    return {
        name: RuntimeSpec(
            python_version=runtime.get("python_version"),
            package_versions=dict(runtime.get("package_versions", {})),
            device=runtime.get("device"),
            accelerator=runtime.get("accelerator"),
            precision=runtime.get("precision"),
            deterministic=runtime.get("deterministic"),
            num_threads=runtime.get("num_threads"),
            code_version=runtime.get("code_version"),
        )
        for name, runtime in config.items()
    }


def build_executor(config: Mapping[str, Any]) -> RunExecutor:
    """Build a run executor from a small execution mapping."""
    name = str(config.get("executor", "sequential"))
    if name == "sequential":
        return SequentialExecutor()
    if name == "ray":
        ray = dict(config.get("ray", {}))
        return RayExecutor(
            address=ray.get("address"),
            init=bool(ray.get("init", True)),
            init_kwargs=dict(ray.get("init_kwargs", {})),
            propagate_uv_runtime_env=bool(ray.get("propagate_uv_runtime_env", False)),
        )
    raise ValueError(f"unsupported executor {name!r}")


def build_logger(config: Mapping[str, Any]) -> RunLogger | None:
    """Build an optional run logger from a small logger mapping."""
    backend = config.get("backend", "none")
    if backend in {None, "none"}:
        return None
    if backend == "mlflow":
        return MLflowLogger(
            experiment_name=config.get("experiment_name"),
            tracking_uri=config.get("tracking_uri"),
            artifact_subdir=str(config.get("artifact_subdir", "artifacts")),
        )
    raise ValueError(f"unsupported logger backend {backend!r}")
