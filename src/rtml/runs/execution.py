"""Run execution APIs for methods, plans, suites, and studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from rtml.benchmarks.base import BenchmarkCase, BenchmarkSuite
from rtml.loggers.base import RunLogger
from rtml.methods.backends.base import BackendResult, MethodBackend
from rtml.methods.backends.registry import default_method_backends
from rtml.methods.base import MethodSpec
from rtml.results.artifacts import save_prediction_set
from rtml.runs.base import RunRecord, RunResult, RuntimeSpec
from rtml.runs.plan import ExecutionResources, RunSpec, ExecutionPlan
from rtml.studies.base import Study
from rtml.tasks.base import TaskSpec


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _default_backend_by_name() -> dict[str, MethodBackend]:
    backends = default_method_backends()
    backend_by_name = {backend.name: backend for backend in backends}
    if len(backend_by_name) != len(backends):
        backend_names = [backend.name for backend in backends]
        raise ValueError(f"method backend names must be unique: {backend_names}")
    return backend_by_name


def _save_predictions(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backend_result: BackendResult,
    prediction_dir: str | Path | None,
    seed: int,
    run_id: str,
) -> str | None:
    if prediction_dir is None:
        return None
    run_digest = run_id.rsplit(":", maxsplit=1)[-1]
    # Include the run digest so distinct method specs with the same display name
    # cannot overwrite each other.
    return str(
        save_prediction_set(
            backend_result.predictions,
            Path(prediction_dir)
            / case.dataset.name
            / case.task.name
            / method.name
            / backend_result.predictions.resample_id
            / f"seed_{seed}_{run_digest}.npz",
        )
    )


def build_run_id(
    *,
    case_name: str,
    dataset_fingerprint: str,
    task: TaskSpec,
    resampling_plan_fingerprint: str,
    resample_id: str,
    method: MethodSpec,
    seed: int,
    runtime: RuntimeSpec | None = None,
) -> str:
    """Build a stable run id from planned inputs."""
    payload = {
        "case_name": case_name,
        "dataset_fingerprint": dataset_fingerprint,
        "task": task,
        "resampling_plan_fingerprint": resampling_plan_fingerprint,
        "resample_id": resample_id,
        "method": {
            "name": method.name,
            "transform": method.transform,
            "model": method.model,
            "training": method.training,
        },
        "seed": seed,
        "runtime": runtime,
    }
    digest = hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{case_name}:{method.name}:{resample_id}:{seed}:sha256:{digest}"


def build_run_record(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    backend_result: BackendResult,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    prediction_path: str | None = None,
) -> RunRecord:

    resample_id = backend_result.predictions.resample_id
    dataset_fingerprint = str(case.dataset.metadata.get("fingerprint", case.dataset.name))
    resampling_fingerprint = case.resampling.fingerprint or ""
    return RunRecord(
        run_id=build_run_id(
            case_name=case.name,
            dataset_fingerprint=dataset_fingerprint,
            task=case.task,
            resampling_plan_fingerprint=resampling_fingerprint,
            resample_id=resample_id,
            method=method,
            seed=seed,
            runtime=runtime,
        ),
        case_name=case.name,
        dataset_name=case.dataset.name,
        dataset_fingerprint=dataset_fingerprint,
        task_name=case.task.name,
        task_type=case.task.task_type,
        primary_metric=case.task.primary_metric,
        resampling_plan_fingerprint=resampling_fingerprint,
        resample_id=resample_id,
        method=method,
        seed=seed,
        runtime=runtime or RuntimeSpec(),
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
    dataset_fingerprint = str(case.dataset.metadata.get("fingerprint", case.dataset.name))
    resampling_fingerprint = case.resampling.fingerprint or ""
    error_message = str(error) or repr(error)
    return RunRecord(
        run_id=build_run_id(
            case_name=case.name,
            dataset_fingerprint=dataset_fingerprint,
            task=case.task,
            resampling_plan_fingerprint=resampling_fingerprint,
            resample_id=resample_id,
            method=method,
            seed=seed,
            runtime=runtime,
        ),
        case_name=case.name,
        dataset_name=case.dataset.name,
        dataset_fingerprint=dataset_fingerprint,
        task_name=case.task.name,
        task_type=case.task.task_type,
        primary_metric=case.task.primary_metric,
        resampling_plan_fingerprint=resampling_fingerprint,
        resample_id=resample_id,
        method=method,
        seed=seed,
        runtime=runtime or RuntimeSpec(),
        status="failed",
        error=error_message,
        metadata={"error_type": type(error).__name__},
    )


def run_method(
    *,
    case: BenchmarkCase,
    method: MethodSpec,
    resample_id: str | None = None,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    prediction_dir: str | Path | None = None,
    logger: RunLogger | None = None,
) -> RunResult:
    """Execute one complete method on one benchmark case/resample."""
    requested_backend = method.model.backend
    backend_by_name = _default_backend_by_name()
    selected_backend = backend_by_name.get(requested_backend)
    if selected_backend is None:
        available_backends = ", ".join(backend_by_name) or "<none>"
        raise ValueError(
            f"no method backend named {requested_backend!r} "
            f"for method {method.name!r}; available backends: {available_backends}"
        )

    # Backend execution owns fitting, predicting, and backend-level metrics.
    # Run execution owns stable IDs, artifact paths, and final logging.
    backend_result = selected_backend.run(
        case=case,
        method=method,
        resample_id=resample_id,
        seed=seed,
        runtime=runtime,
    )
    record = build_run_record(
        case=case,
        method=method,
        backend_result=backend_result,
        seed=seed,
        runtime=runtime,
    )
    prediction_path = _save_predictions(
        case=case,
        method=method,
        backend_result=backend_result,
        prediction_dir=prediction_dir,
        seed=seed,
        run_id=record.run_id,
    )
    if prediction_path is not None:
        record = replace(record, prediction_path=prediction_path)
    if logger is not None:
        logger.log_run(record)
    return RunResult(predictions=backend_result.predictions, record=record)


class RunExecutor(Protocol):
    """Protocol implemented by execution-plan executors."""

    name: str

    def run(
        self,
        plan: ExecutionPlan,
        *,
        prediction_dir: str | Path | None = None,
        logger: RunLogger | None = None,
        continue_on_error: bool = False,
    ) -> list[RunResult]:
        """Run every `RunSpec` in the plan and return RTML-native results."""
        ...


def _log_results(results: Sequence[RunResult], logger: RunLogger | None) -> None:
    if logger is None:
        return
    for result in results:
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
    continue_on_error: bool,
) -> RunResult:
    try:
        return run_method(
            case=run_spec.case,
            method=run_spec.method,
            resample_id=run_spec.resample_id,
            seed=run_spec.seed,
            runtime=run_spec.runtime,
            prediction_dir=prediction_dir,
        )
    except Exception as exc:
        if not continue_on_error:
            raise
        # Failed specs still produce records so summaries can show the missing
        # cells in a study instead of discarding all completed runs.
        return RunResult(
            predictions=None,
            record=build_failed_run_record(
                case=run_spec.case,
                method=run_spec.method,
                resample_id=run_spec.resample_id,
                seed=run_spec.seed,
                runtime=run_spec.runtime,
                error=exc,
            ),
        )


class SequentialExecutor:
    """Execute an execution plan in-process."""

    name = "sequential"

    def run(
        self,
        plan: ExecutionPlan,
        *,
        prediction_dir: str | Path | None = None,
        logger: RunLogger | None = None,
        continue_on_error: bool = False,
    ) -> list[RunResult]:
        results = [
            _execute_run_spec(
                run_spec,
                prediction_dir,
                continue_on_error=continue_on_error,
            )
            for run_spec in plan.runs
        ]
        results = _attach_plan_metadata(results, plan.metadata)
        _log_results(results, logger)
        return results


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


def _put_cases(ray: Any, run_specs: Sequence[RunSpec]) -> dict[int, Any]:
    case_refs = {}
    for run_spec in run_specs:
        case_key = id(run_spec.case)
        if case_key not in case_refs:
            case_refs[case_key] = ray.put(run_spec.case)
    return case_refs


def _execute_run_spec_with_case(
    case: BenchmarkCase,
    method: MethodSpec,
    resample_id: str,
    seed: int,
    runtime: RuntimeSpec | None,
    prediction_dir: str | Path | None,
    continue_on_error: bool,
) -> RunResult:
    return _execute_run_spec(
        RunSpec(
            case=case,
            method=method,
            resample_id=resample_id,
            seed=seed,
            runtime=runtime,
        ),
        prediction_dir,
        continue_on_error=continue_on_error,
    )


class RayExecutor:
    """Execute an execution plan with Ray using each `RunSpec`'s resource hints."""

    name = "ray"

    def __init__(
        self,
        *,
        address: str | None = None,
        init: bool = True,
        init_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.address = address
        self.init = init
        self.init_kwargs = dict(init_kwargs or {})

    def run(
        self,
        plan: ExecutionPlan,
        *,
        prediction_dir: str | Path | None = None,
        logger: RunLogger | None = None,
        continue_on_error: bool = False,
    ) -> list[RunResult]:
        try:
            import ray
        except ImportError as exc:
            raise ImportError("RayExecutor requires the optional 'ray' dependency") from exc

        if self.init and not ray.is_initialized():
            ray.init(address=self.address, **self.init_kwargs)

        # Cases can carry full data frames. Put each shared case once and pass
        # object refs to per-resample/per-seed tasks.
        case_refs = _put_cases(ray, plan.runs)
        remote_run = ray.remote(_execute_run_spec_with_case)
        refs = []
        for run_spec in plan.runs:
            refs.append(
                remote_run.options(**_ray_options(run_spec.scheduler_resources)).remote(
                    case_refs[id(run_spec.case)],
                    run_spec.method,
                    run_spec.resample_id,
                    run_spec.seed,
                    run_spec.runtime,
                    prediction_dir,
                    continue_on_error,
                )
            )

        results = _attach_plan_metadata(list(ray.get(refs)), plan.metadata)
        _log_results(results, logger)
        return results


def run_suite(
    *,
    suite: BenchmarkSuite,
    methods: Sequence[MethodSpec],
    seeds: Sequence[int] = (0,),
    executor: RunExecutor | None = None,
    runtime_specs: Mapping[str, RuntimeSpec] | None = None,
    scheduler_resources: Mapping[str, ExecutionResources] | None = None,
    prediction_dir: str | Path | None = None,
    logger: RunLogger | None = None,
    plan_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    continue_on_error: bool = False,
) -> list[RunResult]:
    """Execute a suite by wrapping it in a default comparison study."""
    study = Study.from_suite(
        name=plan_name or suite.name,
        suite=suite,
        methods=list(methods),
    )
    return run_study(
        study=study,
        seeds=seeds,
        executor=executor,
        runtime_specs=runtime_specs,
        scheduler_resources=scheduler_resources,
        prediction_dir=prediction_dir,
        logger=logger,
        metadata=metadata,
        continue_on_error=continue_on_error,
    )


def run_study(
    *,
    study: Study,
    seeds: Sequence[int] = (0,),
    executor: RunExecutor | None = None,
    runtime_specs: Mapping[str, RuntimeSpec] | None = None,
    scheduler_resources: Mapping[str, ExecutionResources] | None = None,
    prediction_dir: str | Path | None = None,
    logger: RunLogger | None = None,
    metadata: Mapping[str, Any] | None = None,
    continue_on_error: bool = False,
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
        prediction_dir=prediction_dir,
        logger=logger,
        continue_on_error=continue_on_error,
    )


def run_execution_plan_sequential(
    plan: ExecutionPlan,
    *,
    prediction_dir: str | Path | None = None,
    logger: RunLogger | None = None,
    continue_on_error: bool = False,
) -> list[RunResult]:
    """Execute an execution plan in-process."""
    return SequentialExecutor().run(
        plan,
        prediction_dir=prediction_dir,
        logger=logger,
        continue_on_error=continue_on_error,
    )


def run_execution_plan_ray(
    plan: ExecutionPlan,
    *,
    prediction_dir: str | Path | None = None,
    logger: RunLogger | None = None,
    continue_on_error: bool = False,
) -> list[RunResult]:
    """Execute an execution plan with Ray, using each `RunSpec`'s resource hints."""
    return RayExecutor().run(
        plan,
        prediction_dir=prediction_dir,
        logger=logger,
        continue_on_error=continue_on_error,
    )
