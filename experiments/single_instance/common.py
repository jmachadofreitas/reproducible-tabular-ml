from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rtml.single_instance.benchmarks.builders import build_benchmark_suite
from rtml.builders import (
    build_executor,
    build_logger,
    build_methods,
    build_runtime_specs,
    build_scheduler_resources,
    build_study,
)
from rtml.results.reports import save_aggregate_summary, save_run_summary
from rtml.runs import run_study
from rtml.core.runs import RunResult
from rtml.single_instance.methods.sklearn import default_single_instance_backends

PARADIGM = "single_instance"


def run_config(config: Mapping[str, Any], *, experiment_name: str) -> list[RunResult]:
    suite = build_benchmark_suite(config.get("benchmark", {}))
    methods = build_methods(config.get("methods", []))
    study = build_study(
        config.get("study", {}),
        suite=suite,
        methods=methods,
        default_name=experiment_name,
    )
    runtime_specs = build_runtime_specs(config.get("runtime_specs", {}))
    scheduler_resources = build_scheduler_resources(config.get("scheduler_resources", {}))
    logger = build_logger(config.get("logger", {}))
    executor = build_executor(config.get("execution", {}))
    execution = dict(config.get("execution") or {})

    results = run_study(
        study=study,
        backends=default_single_instance_backends(),
        seeds=list(config.get("seeds", [0])),
        executor=executor,
        runtime_specs=runtime_specs,
        scheduler_resources=scheduler_resources,
        prediction_dir=execution.get("prediction_dir"),
        logger=logger,
        metadata={"experiment": experiment_name, "paradigm": PARADIGM},
        continue_on_error=bool(execution.get("continue_on_error", False)),
        show_progress=bool(execution.get("show_progress", True)),
    )
    rows = save_run_summary(
        results,
        csv_path=execution.get("summary_csv"),
        json_path=execution.get("summary_json"),
    )
    save_aggregate_summary(
        rows,
        csv_path=execution.get("aggregate_csv"),
        json_path=execution.get("aggregate_json"),
    )
    return results
