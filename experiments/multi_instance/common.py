from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rtml.builders import (
    build_executor,
    build_logger,
    build_methods,
    build_runtime_specs,
    build_scheduler_resources,
    build_study,
)
from rtml.core.benchmarks import BenchmarkSuite
from rtml.core.runs import RunResult
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.datasets.classic import load_classic_mil_suite
from rtml.multi_instance.datasets.classic.constants import (
    CLASSIC_MIL_DATASETS,
    DEFAULT_CLASSIC_MIL_DATA_DIR,
)
from rtml.multi_instance.datasets.popstats import load_popstats_suite
from rtml.multi_instance.methods import default_multi_instance_backends
from rtml.multi_instance.tasks import MultiInstanceTask
from rtml.results.reports import save_aggregate_summary, save_run_summary
from rtml.runs import run_study


PARADIGM = "multi_instance"


def build_benchmark_suite(
    config: Mapping[str, Any] | None,
) -> BenchmarkSuite[MultiInstanceDataset, MultiInstanceTask]:
    config = config or {}
    source = str(config.get("source") or "popstats").lower()
    if source == "popstats":
        return load_popstats_suite(
            task_ids=config.get("task_ids"),
            n_bags=int(config.get("n_bags", 512)),
            instances_per_bag=int(config.get("instances_per_bag", 512)),
            seed=int(config.get("seed", 3)),
            n_folds=int(config.get("n_folds", 5)),
            valid_size=config.get("valid_size"),
        )
    if source == "classic_mil":
        return load_classic_mil_suite(
            dataset_names=config.get("datasets") or CLASSIC_MIL_DATASETS,
            root=config.get("root") or DEFAULT_CLASSIC_MIL_DATA_DIR,
            seed=int(config.get("seed", 0)),
            n_folds=int(config.get("n_folds", 5)),
            valid_size=config.get("valid_size"),
        )
    raise ValueError(f"unsupported multi-instance benchmark source {source!r}")


def run_config(config: Mapping[str, Any], *, experiment_name: str) -> list[RunResult]:
    suite = build_benchmark_suite(config.get("benchmark", {}))
    study = build_study(
        config.get("study", {}),
        suite=suite,
        methods=build_methods(config.get("methods", [])),
        default_name=experiment_name,
    )
    logger_config = config.get("logger", {})
    logger = build_logger(logger_config)
    execution = dict(config.get("execution") or {})
    results = run_study(
        study=study,
        backends=default_multi_instance_backends(),
        seeds=list(config.get("seeds", [0])),
        executor=build_executor(execution, logger_config=logger_config),
        runtime_specs=build_runtime_specs(config.get("runtime_specs", {})),
        scheduler_resources=build_scheduler_resources(config.get("scheduler_resources", {})),
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
        markdown_path=execution.get("summary_markdown"),
    )
    save_aggregate_summary(
        rows,
        csv_path=execution.get("aggregate_csv"),
        json_path=execution.get("aggregate_json"),
        markdown_path=execution.get("aggregate_markdown"),
    )
    _log_report_artifacts(
        logger,
        experiment_name=experiment_name,
        paths=_existing_paths(
            execution.get("summary_csv"),
            execution.get("summary_json"),
            execution.get("summary_markdown"),
            execution.get("aggregate_csv"),
            execution.get("aggregate_json"),
            execution.get("aggregate_markdown"),
        ),
    )
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
