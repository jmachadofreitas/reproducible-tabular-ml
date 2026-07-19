"""Small builders for constructing RTML specs from plain Python mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rtml.core.benchmarks import BenchmarkSuite
from rtml.loggers.base import RunLogger
from rtml.loggers.mlflow import MLflowLogger
from rtml.core.methods import MethodSpec, ModelSpec
from rtml.core.runs import ExecutionResources
from rtml.runs import RayExecutor, RunExecutor, SequentialExecutor
from rtml.core.runtime import RuntimeSpec
from rtml.core.studies import Study, StudyKind


def build_methods(config: Sequence[Mapping[str, Any]] | None) -> list[MethodSpec]:
    """Build complete method specs from Hydra/notebook-friendly dictionaries."""
    methods = []
    for method in config or ():
        model = dict(method["model"])
        methods.append(
            MethodSpec(
                name=str(method["name"]),
                transform=dict(method.get("transform") or {}),
                model=ModelSpec(
                    kind=str(model["kind"]),
                    backend=str(model["backend"]),
                    params=dict(model.get("params") or {}),
                ),
                training=dict(method.get("training") or {}),
                metadata=dict(method.get("metadata") or {}),
            )
        )
    if not methods:
        raise ValueError("config must define at least one method")
    return methods


def build_study(
    config: Mapping[str, Any] | None,
    *,
    suite: BenchmarkSuite,
    methods: Sequence[MethodSpec],
    default_name: str,
) -> Study:
    """Build a study around an already constructed benchmark suite."""
    config = config or {}
    return Study(
        name=str(config.get("name") or default_name),
        suite=suite,
        methods=list(methods),
        kind=StudyKind(str(config.get("kind") or StudyKind.COMPARISON.value)),
        metadata=dict(config.get("metadata") or {}),
    )


def _defaulted_method_mapping(
    config: Mapping[str, Any] | None,
    *,
    nested_fields: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    if config is None:
        return {}
    defaults = dict(config.get("defaults") or {})
    methods = dict(config.get("methods", config) or {})
    methods.pop("defaults", None)

    merged = {}
    for name, values in methods.items():
        method_values = dict(values or {})
        merged_values = {**defaults, **method_values}
        for field in nested_fields:
            if field in defaults or field in method_values:
                merged_values[field] = {
                    **dict(defaults.get(field) or {}),
                    **dict(method_values.get(field) or {}),
                }
        merged[str(name)] = merged_values
    return merged


def build_scheduler_resources(config: Mapping[str, Any] | None) -> dict[str, ExecutionResources]:
    """Build scheduler resource hints keyed by exact method name."""
    return {
        name: ExecutionResources(
            num_cpus=resource.get("num_cpus"),
            num_gpus=resource.get("num_gpus"),
            memory=resource.get("memory"),
            custom=dict(resource.get("custom") or {}),
        )
        for name, resource in _defaulted_method_mapping(config, nested_fields=("custom",)).items()
    }


def build_runtime_specs(config: Mapping[str, Any] | None) -> dict[str, RuntimeSpec]:
    """Build runtime hints keyed by exact method name."""
    return {
        name: RuntimeSpec(
            python_version=runtime.get("python_version"),
            package_versions=dict(runtime.get("package_versions") or {}),
            device=runtime.get("device"),
            accelerator=runtime.get("accelerator"),
            precision=runtime.get("precision"),
            deterministic=runtime.get("deterministic"),
            num_threads=runtime.get("num_threads"),
            code_version=runtime.get("code_version"),
        )
        for name, runtime in _defaulted_method_mapping(
            config,
            nested_fields=("package_versions",),
        ).items()
    }


def build_executor(config: Mapping[str, Any] | None) -> RunExecutor:
    """Build a run executor from a small execution mapping."""
    config = config or {}
    name = str(config.get("executor", "sequential"))
    if name == "sequential":
        return SequentialExecutor()
    if name == "ray":
        ray = dict(config.get("ray") or {})
        return RayExecutor(
            address=ray.get("address"),
            init=bool(ray.get("init", True)),
            init_kwargs=dict(ray.get("init_kwargs", {})),
            propagate_uv_runtime_env=bool(ray.get("propagate_uv_runtime_env", False)),
        )
    raise ValueError(f"unsupported executor {name!r}")


def build_logger(config: Mapping[str, Any] | None) -> RunLogger | None:
    """Build an optional run logger from a small logger mapping."""
    config = config or {}
    backend = config.get("backend", "none")
    if backend in {None, "none"}:
        return None
    if backend == "mlflow":
        return MLflowLogger(
            experiment_name=config.get("experiment_name"),
            tracking_uri=config.get("tracking_uri"),
            artifact_subdir=config.get("artifact_subdir"),
        )
    raise ValueError(f"unsupported logger backend {backend!r}")
