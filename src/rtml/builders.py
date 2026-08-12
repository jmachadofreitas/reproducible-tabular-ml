"""Small builders for constructing RTML specs from plain Python mappings."""

from collections.abc import Mapping, Sequence
from typing import Any

from rtml.core.benchmarks import BenchmarkSuite
from rtml.core.methods import MethodSpec, ModelSpec
from rtml.core.runs import ExecutionResources
from rtml.core.runtime import RuntimeSpec
from rtml.core.studies import Study, StudyKind
from rtml.runs import RayExecutor, RunExecutor, SequentialExecutor


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
                fit=dict(method.get("fit") or {}),
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


def build_scheduler_resources(config: Mapping[str, Any] | None) -> dict[str, ExecutionResources]:
    """Build scheduler resource hints keyed by exact method name."""
    config = config or {}
    defaults = dict(config.get("defaults") or {})
    methods = dict(config.get("methods", config) or {})
    methods.pop("defaults", None)
    resources = {}
    for name, values in methods.items():
        method = dict(values or {})
        resources[str(name)] = ExecutionResources(
            num_cpus=method.get("num_cpus", defaults.get("num_cpus")),
            num_gpus=method.get("num_gpus", defaults.get("num_gpus")),
            memory=method.get("memory", defaults.get("memory")),
            custom={
                **dict(defaults.get("custom") or {}),
                **dict(method.get("custom") or {}),
            },
        )
    return resources


def build_runtime_specs(config: Mapping[str, Any] | None) -> dict[str, RuntimeSpec]:
    """Build runtime hints keyed by exact method name."""
    config = config or {}
    defaults = dict(config.get("defaults") or {})
    methods = dict(config.get("methods", config) or {})
    methods.pop("defaults", None)
    runtimes = {}
    for name, values in methods.items():
        method = {**defaults, **dict(values or {})}
        runtimes[str(name)] = RuntimeSpec(
            device=method.get("device"),
            deterministic=method.get("deterministic"),
            num_threads=method.get("num_threads"),
        )
    return runtimes


def build_executor(
    config: Mapping[str, Any] | None,
    *,
    logger_config: Mapping[str, Any] | None = None,
) -> RunExecutor:
    """Build a run executor from a small execution mapping."""
    config = config or {}
    name = str(config.get("executor", "sequential"))
    if name == "sequential":
        return SequentialExecutor()
    if name == "ray":
        ray = dict(config.get("ray") or {})
        worker_logging = bool(ray.get("worker_logging", False))
        return RayExecutor(
            address=ray.get("address"),
            init=bool(ray.get("init", True)),
            init_kwargs=dict(ray.get("init_kwargs", {})),
            propagate_uv_runtime_env=bool(ray.get("propagate_uv_runtime_env", False)),
            worker_logger_config=dict(logger_config or {}) if worker_logging else None,
        )
    raise ValueError(f"unsupported executor {name!r}")
