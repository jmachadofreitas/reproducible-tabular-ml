"""Run execution APIs for methods, plans, suites, and studies."""

import os
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from rtml.core.benchmarks import BenchmarkCase, BenchmarkSuite
from rtml.core.methods import MethodSpec
from rtml.core.runs import ExecutionPlan, ExecutionResources, RunRecord, RunResult, RunSpec
from rtml.core.runtime import RuntimeSpec, capture_environment
from rtml.core.studies import Study
from rtml.loggers import Logger
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.runs.artifacts import (
    prepare_execution_artifacts,
    prepare_run_artifacts,
    save_run_artifacts,
)


def _backend_by_name(backends: Sequence[MethodBackend]) -> dict[str, MethodBackend]:
    backend_by_name = {backend.name: backend for backend in backends}
    if len(backend_by_name) != len(backends):
        backend_names = [backend.name for backend in backends]
        raise ValueError(f"method backend names must be unique: {backend_names}")
    return backend_by_name


def _validate_method_backend(method: MethodSpec, backend: MethodBackend) -> None:
    if backend.name != method.model.backend:
        raise ValueError(
            f"method {method.name!r} requires backend {method.model.backend!r}, "
            f"received {backend.name!r}"
        )
    backend.validate_method(method)


def _logger_run_context(
    logger: Logger | None,
    *,
    case_name: str,
    method_name: str,
    resample_id: str,
) -> Any:
    if logger is None:
        return nullcontext()
    return logger.start_run(run_name=f"{case_name}/{method_name}/{resample_id}")


def _with_metadata(result: RunResult, metadata: Mapping[str, Any] | None) -> RunResult:
    if not metadata:
        return result
    merged = {**result.record.metadata, **dict(metadata)}
    return replace(result, record=replace(result.record, metadata=merged))


def _subgroup_columns(case: BenchmarkCase, configured_columns: Sequence[str] | None) -> list[str]:
    if configured_columns is None:
        return []
    columns: list[str] = []
    for column in configured_columns:
        if column not in columns:
            columns.append(column)
    return columns


def _subgroup_values(
    *,
    case: BenchmarkCase,
    resample_id: str,
    columns: Sequence[str] | None,
) -> dict[str, Any]:
    selected_columns = _subgroup_columns(case, columns)
    if not selected_columns:
        return {}
    test_idx = case.resampling.get_resample(resample_id).test_idx
    return case.dataset.subgroup_values(selected_columns, test_idx)


def build_run_id(
    *,
    case_name: str,
    resample_id: str,
    method_name: str,
    seed: int,
) -> str:
    """Build a readable id for one execution-plan unit."""
    return f"{case_name}:{method_name}:{resample_id}:seed-{seed}"


def _primary_metric_direction(case: BenchmarkCase) -> bool | None:
    primary_metric = case.task.primary_metric
    if primary_metric is None:
        return None
    for metric in case.task.metrics:
        if metric.name == primary_metric:
            return metric.greater_is_better
    raise ValueError(f"primary metric {primary_metric!r} is not present in task metrics")


def build_run_record(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backend_result: BackendResult,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
) -> RunRecord:

    resample_id = backend_result.predictions.resample_id
    return RunRecord(
        run_id=build_run_id(
            case_name=case.name,
            resample_id=resample_id,
            method_name=method.name,
            seed=seed,
        ),
        case_name=case.name,
        dataset_name=case.dataset.name,
        task_name=case.task.name,
        task_type=case.task.task_type,
        primary_metric=case.task.primary_metric,
        resample_id=resample_id,
        method=method,
        seed=seed,
        runtime=runtime,
        environment=capture_environment(),
        status="success",
        primary_metric_greater_is_better=_primary_metric_direction(case),
        metrics=backend_result.metrics,
        fit_time=backend_result.fit_time,
        predict_time=backend_result.predict_time,
        metadata=dict(backend_result.metadata),
    )


def build_failed_run_record(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    resample_id: str,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    error: Exception,
) -> RunRecord:
    error_message = str(error) or repr(error)
    return RunRecord(
        run_id=build_run_id(
            case_name=case.name,
            resample_id=resample_id,
            method_name=method.name,
            seed=seed,
        ),
        case_name=case.name,
        dataset_name=case.dataset.name,
        task_name=case.task.name,
        task_type=case.task.task_type,
        primary_metric=case.task.primary_metric,
        resample_id=resample_id,
        method=method,
        seed=seed,
        runtime=runtime,
        environment=capture_environment(),
        status="failed",
        primary_metric_greater_is_better=_primary_metric_direction(case),
        error=error_message,
        metadata={"error_type": type(error).__name__},
    )


def _run_method_in_context(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backend: MethodBackend,
    resample_id: str | None = None,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    artifact_dir: str | Path | None = None,
    logger: Logger | None = None,
    metadata: Mapping[str, Any] | None = None,
    subgroup_columns: Sequence[str] | None = None,
) -> RunResult:
    _validate_method_backend(method, backend)

    # Backend execution owns fitting, predicting, and backend-level metrics.
    # Run execution owns run IDs, evidence, artifact paths, and final logging.
    backend_result = backend.run(
        case=case,
        method=method,
        resample_id=resample_id,
        seed=seed,
        runtime=runtime,
        logger=logger,
    )
    record = build_run_record(
        case=case,
        method=method,
        backend_result=backend_result,
        seed=seed,
        runtime=runtime,
    )
    result = _with_metadata(
        RunResult(predictions=backend_result.predictions, record=record),
        metadata,
    )
    enriched_predictions = replace(
        backend_result.predictions,
        subgroups={
            **dict(backend_result.predictions.subgroups or {}),
            **_subgroup_values(
                case=case,
                resample_id=result.record.resample_id,
                columns=subgroup_columns,
            ),
        },
    )
    result = save_run_artifacts(
        case=case,
        result=replace(result, predictions=enriched_predictions),
        artifact_dir=artifact_dir,
    )
    if logger is not None:
        logger.log_run(result.record)
    return result


def run_method(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backend: MethodBackend,
    resample_id: str | None = None,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    artifact_dir: str | Path | None = None,
    logger: Logger | None = None,
    metadata: Mapping[str, Any] | None = None,
    subgroup_columns: Sequence[str] | None = None,
) -> RunResult:
    """Execute one complete method on one benchmark case/resample."""
    planned_resample_id = case.resampling.get_resample(resample_id).id
    prepare_run_artifacts(
        RunSpec(
            case=case,
            method=method,
            resample_id=planned_resample_id,
            seed=seed,
            runtime=runtime,
        ),
        artifact_dir,
    )
    with _logger_run_context(
        logger,
        case_name=case.name,
        method_name=method.name,
        resample_id=planned_resample_id,
    ):
        return _run_method_in_context(
            case=case,
            method=method,
            backend=backend,
            resample_id=resample_id,
            seed=seed,
            runtime=runtime,
            artifact_dir=artifact_dir,
            logger=logger,
            metadata={
                **case.metadata,
                **method.metadata,
                **dict(metadata or {}),
            },
            subgroup_columns=subgroup_columns,
        )


class RunExecutor(Protocol):
    """Protocol implemented by execution-plan executors."""

    name: str

    def run(
        self,
        plan: ExecutionPlan,
        *,
        backends: Sequence[MethodBackend],
        artifact_dir: str | Path | None = None,
        logger: Logger | None = None,
        continue_on_error: bool = False,
        show_progress: bool = False,
        subgroup_columns: Sequence[str] | None = None,
    ) -> list[RunResult]:
        """Run every `RunSpec` in the plan and return RTML-native results."""
        ...


def _log_results(results: Sequence[RunResult], logger: Logger | None) -> None:
    if logger is None:
        return
    for result in results:
        with _logger_run_context(
            logger,
            case_name=result.record.case_name,
            method_name=result.record.method.name,
            resample_id=result.record.resample_id,
        ):
            logger.log_run(result.record)


def _execute_run_spec(
    run_spec: RunSpec,
    artifact_dir: str | Path | None,
    *,
    backend: MethodBackend | None,
    continue_on_error: bool,
    logger: Logger | None = None,
    metadata: Mapping[str, Any] | None = None,
    subgroup_columns: Sequence[str] | None = None,
) -> RunResult:
    with _logger_run_context(
        logger,
        case_name=run_spec.case.name,
        method_name=run_spec.method.name,
        resample_id=run_spec.resample_id,
    ):
        try:
            if backend is None:
                raise ValueError(
                    f"no method backend named {run_spec.method.model.backend!r} "
                    f"for method {run_spec.method.name!r}"
                )
            return _run_method_in_context(
                case=run_spec.case,
                method=run_spec.method,
                backend=backend,
                resample_id=run_spec.resample_id,
                seed=run_spec.seed,
                runtime=run_spec.runtime,
                artifact_dir=artifact_dir,
                logger=logger,
                metadata=metadata,
                subgroup_columns=subgroup_columns,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            # Failed specs still produce records so summaries can show the
            # missing cells in a study instead of discarding completed runs.
            result = _with_metadata(
                RunResult(
                    predictions=None,
                    record=build_failed_run_record(
                        case=run_spec.case,
                        method=run_spec.method,
                        resample_id=run_spec.resample_id,
                        seed=run_spec.seed,
                        runtime=run_spec.runtime,
                        error=exc,
                    ),
                ),
                metadata,
            )
            result = save_run_artifacts(
                case=run_spec.case,
                result=result,
                artifact_dir=artifact_dir,
            )
            if logger is not None:
                logger.log_run(result.record)
            return result


class SequentialExecutor:
    """Execute an execution plan in-process."""

    name = "sequential"

    def run(
        self,
        plan: ExecutionPlan,
        *,
        backends: Sequence[MethodBackend],
        artifact_dir: str | Path | None = None,
        logger: Logger | None = None,
        continue_on_error: bool = False,
        show_progress: bool = False,
        subgroup_columns: Sequence[str] | None = None,
    ) -> list[RunResult]:
        prepare_execution_artifacts(plan, artifact_dir)
        backend_by_name = _backend_by_name(backends)
        results = []
        for run_spec in tqdm(
            plan.runs,
            total=len(plan.runs),
            desc=f"{plan.name} ({self.name})",
            unit="run",
            disable=not show_progress,
        ):
            results.append(
                _execute_run_spec(
                    run_spec,
                    artifact_dir,
                    backend=backend_by_name.get(run_spec.method.model.backend),
                    continue_on_error=continue_on_error,
                    logger=logger,
                    metadata={
                        **run_spec.case.metadata,
                        **run_spec.method.metadata,
                        **plan.metadata,
                    },
                    subgroup_columns=subgroup_columns,
                )
            )
        return results


class RayExecutor:
    """Execute an execution plan with Ray using each `RunSpec`'s resource hints."""

    name = "ray"

    def __init__(
        self,
        *,
        address: str | None = None,
        init: bool = True,
        init_kwargs: Mapping[str, Any] | None = None,
        propagate_uv_runtime_env: bool = False,
        worker_logger_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.address = address
        self.init = init
        self.init_kwargs = dict(init_kwargs or {})
        self.propagate_uv_runtime_env = propagate_uv_runtime_env
        self.worker_logger_config = self._active_worker_logger_config(worker_logger_config)

    def run(
        self,
        plan: ExecutionPlan,
        *,
        backends: Sequence[MethodBackend],
        artifact_dir: str | Path | None = None,
        logger: Logger | None = None,
        continue_on_error: bool = False,
        show_progress: bool = False,
        subgroup_columns: Sequence[str] | None = None,
    ) -> list[RunResult]:
        prepare_execution_artifacts(plan, artifact_dir)
        try:
            import ray
        except ImportError as exc:
            raise ImportError("RayExecutor requires the optional 'ray' dependency") from exc

        self._configure_uv_runtime_env(
            ray,
            propagate_uv_runtime_env=self.propagate_uv_runtime_env,
        )
        if self.init and not ray.is_initialized():
            ray.init(address=self.address, **self.init_kwargs)

        backend_by_name = _backend_by_name(backends)
        # Cases can carry full data frames. Put each shared case once and pass
        # object refs to per-resample/per-seed tasks.
        case_refs = self._put_cases(ray, plan.runs)
        remote_run = ray.remote(self._execute_run_spec_with_case)
        refs = []
        for run_spec in plan.runs:
            # Logger instances can hold process-local run context, for example
            # MLflow's active run. Pass plain logger config so workers can
            # build one logger per RunSpec when worker logging is enabled.
            refs.append(
                remote_run.options(**self._ray_options(run_spec.scheduler_resources)).remote(
                    case_refs[id(run_spec.case)],
                    run_spec.method,
                    run_spec.resample_id,
                    run_spec.seed,
                    run_spec.runtime,
                    backend_by_name.get(run_spec.method.model.backend),
                    artifact_dir,
                    continue_on_error,
                    self.worker_logger_config,
                    {
                        **run_spec.case.metadata,
                        **run_spec.method.metadata,
                        **plan.metadata,
                    },
                    subgroup_columns,
                )
            )

        results = self._get_results(
            ray,
            refs,
            show_progress=show_progress,
            label=f"{plan.name} ({self.name})",
        )
        if not self.worker_logger_config:
            _log_results(results, logger)
        return results

    @staticmethod
    def _configure_uv_runtime_env(ray: Any, *, propagate_uv_runtime_env: bool) -> None:
        enabled = "1" if propagate_uv_runtime_env else "0"
        os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = enabled
        try:
            ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = propagate_uv_runtime_env
        except AttributeError:
            pass

    @staticmethod
    def _ray_options(resources: ExecutionResources) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if resources.num_cpus is not None:
            options["num_cpus"] = resources.num_cpus
        if resources.num_gpus is not None:
            options["num_gpus"] = resources.num_gpus
        if resources.memory is not None:
            options["memory"] = resources.memory
        if resources.custom:
            options["resources"] = resources.custom
        return options

    @staticmethod
    def _put_cases(ray: Any, run_specs: Sequence[RunSpec]) -> dict[int, Any]:
        case_refs = {}
        for run_spec in run_specs:
            case_key = id(run_spec.case)
            if case_key not in case_refs:
                case_refs[case_key] = ray.put(run_spec.case)
        return case_refs

    @staticmethod
    def _active_worker_logger_config(
        config: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        config = dict(config or {})
        if config.get("backend", "none") in {None, "none"}:
            return {}
        if config.get("backend") == "mlflow":
            config["tracking_uri"] = RayExecutor._absolute_mlflow_tracking_uri(
                config.get("tracking_uri")
            )
        return config

    @staticmethod
    def _absolute_mlflow_tracking_uri(tracking_uri: Any) -> str:
        from rtml.loggers.mlflow import DEFAULT_MLFLOW_TRACKING_URI

        uri = str(tracking_uri or DEFAULT_MLFLOW_TRACKING_URI)
        if not uri.startswith("sqlite:///"):
            return uri
        db_path = uri.removeprefix("sqlite:///")
        if not db_path or db_path == ":memory:":
            return uri
        path = Path(db_path).expanduser()
        if path.is_absolute():
            return uri
        return f"sqlite:///{path.resolve()}"

    @staticmethod
    def _execute_run_spec_with_case(
        case: BenchmarkCase,
        method: MethodSpec,
        resample_id: str,
        seed: int,
        runtime: RuntimeSpec | None,
        backend: MethodBackend | None,
        artifact_dir: str | Path | None,
        continue_on_error: bool,
        worker_logger_config: Mapping[str, Any],
        metadata: Mapping[str, Any],
        subgroup_columns: Sequence[str] | None,
    ) -> RunResult:
        worker_logger = RayExecutor._build_worker_logger(worker_logger_config)
        return _execute_run_spec(
            RunSpec(
                case=case,
                method=method,
                resample_id=resample_id,
                seed=seed,
                runtime=runtime,
            ),
            artifact_dir,
            backend=backend,
            continue_on_error=continue_on_error,
            logger=worker_logger,
            metadata=metadata,
            subgroup_columns=subgroup_columns,
        )

    @staticmethod
    def _build_worker_logger(config: Mapping[str, Any]) -> Logger | None:
        if not config:
            return None
        from rtml.loggers import build_logger

        return build_logger(config)

    @staticmethod
    def _get_results(
        ray: Any,
        refs: Sequence[Any],
        *,
        show_progress: bool,
        label: str,
    ) -> list[RunResult]:
        if not show_progress:
            return list(ray.get(refs))
        if not refs:
            return []
        if not hasattr(ray, "wait"):
            results = []
            with tqdm(total=len(refs), desc=label, unit="run") as progress:
                for ref in refs:
                    results.append(ray.get(ref))
                    progress.update()
            return results

        results_by_position: list[RunResult | None] = [None] * len(refs)
        pending = list(refs)
        with tqdm(total=len(refs), desc=label, unit="run") as progress:
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                ready_results = ray.get(ready)
                for ref, result in zip(ready, ready_results, strict=True):
                    results_by_position[refs.index(ref)] = result
                progress.update(len(ready))

        if any(result is None for result in results_by_position):
            raise RuntimeError("Ray completed without returning every run result")
        return [result for result in results_by_position if result is not None]


def run_suite(
    *,
    suite: BenchmarkSuite,
    methods: Sequence[MethodSpec],
    backends: Sequence[MethodBackend],
    seeds: Sequence[int] = (0,),
    executor: RunExecutor | None = None,
    runtime_specs: Mapping[str, RuntimeSpec] | None = None,
    scheduler_resources: Mapping[str, ExecutionResources] | None = None,
    artifact_dir: str | Path | None = None,
    logger: Logger | None = None,
    plan_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    continue_on_error: bool = False,
    show_progress: bool = False,
    subgroup_columns: Sequence[str] | None = None,
) -> list[RunResult]:
    """Execute a suite by wrapping it in a default comparison study."""
    study = Study.from_suite(
        name=plan_name or suite.name,
        suite=suite,
        methods=list(methods),
    )
    return run_study(
        study=study,
        backends=backends,
        seeds=seeds,
        executor=executor,
        runtime_specs=runtime_specs,
        scheduler_resources=scheduler_resources,
        artifact_dir=artifact_dir,
        logger=logger,
        metadata=metadata,
        continue_on_error=continue_on_error,
        show_progress=show_progress,
        subgroup_columns=subgroup_columns,
    )


def run_study(
    *,
    study: Study,
    backends: Sequence[MethodBackend],
    seeds: Sequence[int] = (0,),
    executor: RunExecutor | None = None,
    runtime_specs: Mapping[str, RuntimeSpec] | None = None,
    scheduler_resources: Mapping[str, ExecutionResources] | None = None,
    artifact_dir: str | Path | None = None,
    logger: Logger | None = None,
    metadata: Mapping[str, Any] | None = None,
    continue_on_error: bool = False,
    show_progress: bool = False,
    subgroup_columns: Sequence[str] | None = None,
) -> list[RunResult]:
    """Expand a study into an execution plan and execute it."""
    plan = ExecutionPlan.from_study(
        study=study,
        seeds=seeds,
        runtime_specs=runtime_specs,
        scheduler_resources=scheduler_resources,
        metadata=metadata,
    )
    return (executor or SequentialExecutor()).run(
        plan,
        backends=backends,
        artifact_dir=artifact_dir,
        logger=logger,
        continue_on_error=continue_on_error,
        show_progress=show_progress,
        subgroup_columns=subgroup_columns,
    )
