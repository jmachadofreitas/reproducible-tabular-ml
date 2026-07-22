"""Run execution APIs for methods, plans, suites, and studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
import os
from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from rtml.core.benchmarks import BenchmarkCase, BenchmarkSuite
from rtml.core.fingerprints import (
    fingerprint_dataset,
    fingerprint_method,
    fingerprint_runtime,
    fingerprint_task,
    stable_fingerprint,
)
from rtml.loggers import Logger
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.core.methods import MethodSpec
from rtml.core.results import PredictionSet
from rtml.results.artifacts import save_prediction_set
from rtml.core.runs import ExecutionPlan, ExecutionResources, RunRecord, RunResult, RunSpec
from rtml.core.runtime import RuntimeSpec, capture_runtime
from rtml.core.studies import Study


def _backend_by_name(backends: Sequence[MethodBackend]) -> dict[str, MethodBackend]:
    backend_by_name = {backend.name: backend for backend in backends}
    if len(backend_by_name) != len(backends):
        backend_names = [backend.name for backend in backends]
        raise ValueError(f"method backend names must be unique: {backend_names}")
    return backend_by_name


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


def _save_predictions(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    record: RunRecord,
    predictions: PredictionSet,
    prediction_dir: str | Path | None,
    seed: int,
) -> str | None:
    if prediction_dir is None:
        return None
    run_digest = record.run_id.rsplit(":", maxsplit=1)[-1]
    # Include the run digest so distinct method specs with the same display name
    # cannot overwrite each other.
    return str(
        save_prediction_set(
            _with_prediction_evidence(predictions, record),
            Path(prediction_dir)
            / case.dataset.name
            / case.task.name
            / method.name
            / record.resample_id
            / f"seed_{seed}_{run_digest}.npz",
        )
    )


def _subgroup_columns(case: BenchmarkCase, configured_columns: Sequence[str] | None) -> list[str]:
    if configured_columns is None:
        return []
    columns: list[str] = []
    for column in (*case.task.groups, *case.task.sensitive_attributes, *configured_columns):
        if column not in columns:
            columns.append(column)
    case.dataset.require_columns(columns)
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
    data = case.dataset.data.iloc[test_idx]
    return {
        column: data[column].astype("string").fillna("<NA>").to_numpy(dtype=str)
        for column in selected_columns
    }


def _with_prediction_evidence(predictions: PredictionSet, record: RunRecord) -> PredictionSet:
    return replace(
        predictions,
        subgroups={**dict(predictions.subgroups or {})},
        metadata={
            **dict(predictions.metadata or {}),
            **prediction_evidence_metadata(record),
        },
    )


def prediction_evidence_metadata(record: RunRecord) -> dict[str, Any]:
    """Return run evidence metadata that should travel with saved predictions."""
    return {
        "run_id": record.run_id,
        "case_name": record.case_name,
        "dataset_fingerprint": record.dataset_fingerprint,
        "task_fingerprint": record.task_fingerprint,
        "resampling_plan_fingerprint": record.resampling_plan_fingerprint,
        "method_fingerprint": record.method_fingerprint,
        "runtime_fingerprint": record.runtime_fingerprint,
        "seed": record.seed,
    }


def build_run_id(
    *,
    case_name: str,
    dataset_fingerprint: str,
    task_fingerprint: str,
    resampling_plan_fingerprint: str,
    resample_id: str,
    method_name: str,
    method_fingerprint: str,
    seed: int,
    runtime_fingerprint: str,
) -> str:
    """Build a stable run id from planned inputs."""
    payload = {
        "case_name": case_name,
        "dataset_fingerprint": dataset_fingerprint,
        "task_fingerprint": task_fingerprint,
        "resampling_plan_fingerprint": resampling_plan_fingerprint,
        "resample_id": resample_id,
        "method_name": method_name,
        "method_fingerprint": method_fingerprint,
        "seed": seed,
        "runtime_fingerprint": runtime_fingerprint,
    }
    digest = stable_fingerprint(payload).removeprefix("sha256:")[:16]
    return f"{case_name}:{method_name}:{resample_id}:{seed}:sha256:{digest}"


def build_run_record(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backend_result: BackendResult,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    prediction_path: str | None = None,
) -> RunRecord:

    observed_runtime = runtime or capture_runtime()
    resample_id = backend_result.predictions.resample_id
    dataset_fingerprint = fingerprint_dataset(case.dataset)
    task_fingerprint = fingerprint_task(case.task)
    resampling_fingerprint = case.resampling.fingerprint or ""
    method_fingerprint = fingerprint_method(method)
    runtime_fingerprint = fingerprint_runtime(observed_runtime)
    run_id_runtime_fingerprint = (
        fingerprint_runtime(runtime) if runtime is not None else stable_fingerprint(None)
    )
    return RunRecord(
        run_id=build_run_id(
            case_name=case.name,
            dataset_fingerprint=dataset_fingerprint,
            task_fingerprint=task_fingerprint,
            resampling_plan_fingerprint=resampling_fingerprint,
            resample_id=resample_id,
            method_name=method.name,
            method_fingerprint=method_fingerprint,
            seed=seed,
            runtime_fingerprint=run_id_runtime_fingerprint,
        ),
        case_name=case.name,
        dataset_name=case.dataset.name,
        dataset_fingerprint=dataset_fingerprint,
        task_name=case.task.name,
        task_type=case.task.task_type,
        task_fingerprint=task_fingerprint,
        primary_metric=case.task.primary_metric,
        resampling_plan_fingerprint=resampling_fingerprint,
        resample_id=resample_id,
        method=method,
        method_fingerprint=method_fingerprint,
        seed=seed,
        runtime=observed_runtime,
        runtime_fingerprint=runtime_fingerprint,
        status="success",
        metrics=backend_result.metrics,
        fit_time=backend_result.fit_time,
        predict_time=backend_result.predict_time,
        prediction_path=prediction_path,
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
    observed_runtime = runtime or capture_runtime()
    dataset_fingerprint = fingerprint_dataset(case.dataset)
    task_fingerprint = fingerprint_task(case.task)
    resampling_fingerprint = case.resampling.fingerprint or ""
    method_fingerprint = fingerprint_method(method)
    runtime_fingerprint = fingerprint_runtime(observed_runtime)
    run_id_runtime_fingerprint = (
        fingerprint_runtime(runtime) if runtime is not None else stable_fingerprint(None)
    )
    error_message = str(error) or repr(error)
    return RunRecord(
        run_id=build_run_id(
            case_name=case.name,
            dataset_fingerprint=dataset_fingerprint,
            task_fingerprint=task_fingerprint,
            resampling_plan_fingerprint=resampling_fingerprint,
            resample_id=resample_id,
            method_name=method.name,
            method_fingerprint=method_fingerprint,
            seed=seed,
            runtime_fingerprint=run_id_runtime_fingerprint,
        ),
        case_name=case.name,
        dataset_name=case.dataset.name,
        dataset_fingerprint=dataset_fingerprint,
        task_name=case.task.name,
        task_type=case.task.task_type,
        task_fingerprint=task_fingerprint,
        primary_metric=case.task.primary_metric,
        resampling_plan_fingerprint=resampling_fingerprint,
        resample_id=resample_id,
        method=method,
        method_fingerprint=method_fingerprint,
        seed=seed,
        runtime=observed_runtime,
        runtime_fingerprint=runtime_fingerprint,
        status="failed",
        error=error_message,
        metadata={"error_type": type(error).__name__},
    )


def _run_method_in_context(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backends: Sequence[MethodBackend],
    resample_id: str | None = None,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    prediction_dir: str | Path | None = None,
    logger: Logger | None = None,
    metadata: Mapping[str, Any] | None = None,
    subgroup_columns: Sequence[str] | None = None,
) -> RunResult:
    requested_backend = method.model.backend
    backend_by_name = _backend_by_name(backends)
    selected_backend = backend_by_name.get(requested_backend)
    if selected_backend is None:
        available_backends = ", ".join(backend_by_name) or "<none>"
        raise ValueError(
            f"no method backend named {requested_backend!r} "
            f"for method {method.name!r}; available backends: {available_backends}"
        )
    selected_backend.validate_method(method)

    # Backend execution owns fitting, predicting, and backend-level metrics.
    # Run execution owns stable IDs, artifact paths, and final logging.
    backend_result = selected_backend.run(
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
        _with_prediction_evidence(backend_result.predictions, result.record),
        subgroups={
            **dict(backend_result.predictions.subgroups or {}),
            **_subgroup_values(
                case=case,
                resample_id=result.record.resample_id,
                columns=subgroup_columns,
            ),
        },
    )
    result = replace(result, predictions=enriched_predictions)
    prediction_path = _save_predictions(
        case=case,
        method=method,
        record=result.record,
        predictions=enriched_predictions,
        prediction_dir=prediction_dir,
        seed=seed,
    )
    if prediction_path is not None:
        result = replace(
            result,
            record=replace(result.record, prediction_path=prediction_path),
        )
    if logger is not None:
        logger.log_run(result.record)
    return result


def run_method(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backends: Sequence[MethodBackend],
    resample_id: str | None = None,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    prediction_dir: str | Path | None = None,
    logger: Logger | None = None,
    metadata: Mapping[str, Any] | None = None,
    subgroup_columns: Sequence[str] | None = None,
) -> RunResult:
    """Execute one complete method on one benchmark case/resample."""
    planned_resample_id = case.resampling.get_resample(resample_id).id
    with _logger_run_context(
        logger,
        case_name=case.name,
        method_name=method.name,
        resample_id=planned_resample_id,
    ):
        return _run_method_in_context(
            case=case,
            method=method,
            backends=backends,
            resample_id=resample_id,
            seed=seed,
            runtime=runtime,
            prediction_dir=prediction_dir,
            logger=logger,
            metadata=metadata,
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
        prediction_dir: str | Path | None = None,
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


def _attach_plan_metadata(
    results: Sequence[RunResult],
    plan_metadata: Mapping[str, Any],
) -> list[RunResult]:
    updated_results = []
    for result in results:
        # Plan metadata gives study/experiment context, method metadata gives
        # reporting factors, and backend metadata gives observed execution info.
        metadata = {
            **result.record.metadata,
            **result.record.method.metadata,
            **dict(plan_metadata),
        }
        if metadata == result.record.metadata:
            updated_results.append(result)
            continue
        updated_results.append(
            replace(
                result,
                record=replace(result.record, metadata=metadata),
            )
        )
    return updated_results


def _execute_run_spec(
    run_spec: RunSpec,
    prediction_dir: str | Path | None,
    *,
    backends: Sequence[MethodBackend],
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
            return _run_method_in_context(
                case=run_spec.case,
                method=run_spec.method,
                backends=backends,
                resample_id=run_spec.resample_id,
                seed=run_spec.seed,
                runtime=run_spec.runtime,
                prediction_dir=prediction_dir,
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
            if logger is not None:
                logger.log_run(result.record)
            return result


def _execution_metadata(
    *,
    method: MethodSpec,
    plan_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {**method.metadata, **dict(plan_metadata)}


class SequentialExecutor:
    """Execute an execution plan in-process."""

    name = "sequential"

    def run(
        self,
        plan: ExecutionPlan,
        *,
        backends: Sequence[MethodBackend],
        prediction_dir: str | Path | None = None,
        logger: Logger | None = None,
        continue_on_error: bool = False,
        show_progress: bool = False,
        subgroup_columns: Sequence[str] | None = None,
    ) -> list[RunResult]:
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
                    prediction_dir,
                    backends=backends,
                    continue_on_error=continue_on_error,
                    logger=logger,
                    metadata=_execution_metadata(
                        method=run_spec.method,
                        plan_metadata=plan.metadata,
                    ),
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
        prediction_dir: str | Path | None = None,
        logger: Logger | None = None,
        continue_on_error: bool = False,
        show_progress: bool = False,
        subgroup_columns: Sequence[str] | None = None,
    ) -> list[RunResult]:
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
                    backends,
                    prediction_dir,
                    continue_on_error,
                    self.worker_logger_config,
                    _execution_metadata(
                        method=run_spec.method,
                        plan_metadata=plan.metadata,
                    ),
                    subgroup_columns,
                )
            )

        raw_results = self._get_results(
            ray,
            refs,
            show_progress=show_progress,
            label=f"{plan.name} ({self.name})",
        )
        results = _attach_plan_metadata(raw_results, plan.metadata)
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
        backends: Sequence[MethodBackend],
        prediction_dir: str | Path | None,
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
            prediction_dir,
            backends=backends,
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
    prediction_dir: str | Path | None = None,
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
        prediction_dir=prediction_dir,
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
    prediction_dir: str | Path | None = None,
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
        prediction_dir=prediction_dir,
        logger=logger,
        continue_on_error=continue_on_error,
        show_progress=show_progress,
        subgroup_columns=subgroup_columns,
    )


def run_execution_plan_sequential(
    plan: ExecutionPlan,
    *,
    backends: Sequence[MethodBackend],
    prediction_dir: str | Path | None = None,
    logger: Logger | None = None,
    continue_on_error: bool = False,
    show_progress: bool = False,
    subgroup_columns: Sequence[str] | None = None,
) -> list[RunResult]:
    """Execute an execution plan in-process."""
    return SequentialExecutor().run(
        plan,
        backends=backends,
        prediction_dir=prediction_dir,
        logger=logger,
        continue_on_error=continue_on_error,
        show_progress=show_progress,
        subgroup_columns=subgroup_columns,
    )


def run_execution_plan_ray(
    plan: ExecutionPlan,
    *,
    backends: Sequence[MethodBackend],
    prediction_dir: str | Path | None = None,
    logger: Logger | None = None,
    continue_on_error: bool = False,
    show_progress: bool = False,
    subgroup_columns: Sequence[str] | None = None,
) -> list[RunResult]:
    """Execute an execution plan with Ray, using each `RunSpec`'s resource hints."""
    return RayExecutor().run(
        plan,
        backends=backends,
        prediction_dir=prediction_dir,
        logger=logger,
        continue_on_error=continue_on_error,
        show_progress=show_progress,
        subgroup_columns=subgroup_columns,
    )
