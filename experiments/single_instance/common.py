from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
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
from rtml.results.subgroups import save_subgroup_summary
from rtml.runs import run_study
from rtml.core.runs import RunResult
from rtml.single_instance.methods import default_single_instance_backends

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
    logger_config = config.get("logger", {})
    logger = build_logger(logger_config)
    executor = build_executor(config.get("execution", {}), logger_config=logger_config)
    execution = dict(config.get("execution") or {})
    subgroup_report = dict(config.get("subgroup_report") or {})
    subgroup_enabled = bool(subgroup_report.get("enabled", False))
    if subgroup_enabled and not execution.get("prediction_dir"):
        raise ValueError("subgroup_report requires execution.prediction_dir")

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
        subgroup_columns=list(subgroup_report.get("columns") or []) if subgroup_enabled else None,
    )
    report_paths = []
    rows = save_run_summary(
        results,
        csv_path=execution.get("summary_csv"),
        json_path=execution.get("summary_json"),
        markdown_path=execution.get("summary_markdown"),
    )
    report_paths.extend(
        _existing_paths(
            execution.get("summary_csv"),
            execution.get("summary_json"),
            execution.get("summary_markdown"),
        )
    )
    save_aggregate_summary(
        rows,
        csv_path=execution.get("aggregate_csv"),
        json_path=execution.get("aggregate_json"),
        markdown_path=execution.get("aggregate_markdown"),
    )
    report_paths.extend(
        _existing_paths(
            execution.get("aggregate_csv"),
            execution.get("aggregate_json"),
            execution.get("aggregate_markdown"),
        )
    )
    if subgroup_enabled:
        save_subgroup_summary(
            rows,
            metrics_by_task={case.task.name: case.task.metrics for case in suite.cases},
            csv_path=subgroup_report.get("csv"),
            json_path=subgroup_report.get("json"),
            markdown_path=subgroup_report.get("markdown"),
            min_count=int(subgroup_report.get("min_count", 1)),
        )
        report_paths.extend(
            _existing_paths(
                subgroup_report.get("csv"),
                subgroup_report.get("json"),
                subgroup_report.get("markdown"),
            )
        )
    _log_report_artifacts(logger, experiment_name=experiment_name, paths=report_paths)
    return results


def _existing_paths(*paths: Any) -> list[Path]:
    existing = []
    for path in paths:
        if path is None:
            continue
        report_path = Path(str(path))
        if report_path.exists():
            existing.append(report_path)
    return existing


def _log_report_artifacts(
    logger: Any | None,
    *,
    experiment_name: str,
    paths: list[Path],
) -> None:
    if logger is None or not paths:
        return
    with logger.start_run(run_name=f"{experiment_name}/reports"):
        for path in paths:
            logger.log_artifact(path, artifact_path="reports")
