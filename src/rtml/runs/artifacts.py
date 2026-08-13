"""Readable local artifacts for benchmark cases and executed runs."""

import os
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.datasets import Dataset, dataset_source
from rtml.core.results import PredictionSet
from rtml.core.runs import ExecutionPlan, RunResult, RunSpec
from rtml.core.serialization import JSONEncoder
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.results.artifacts import save_prediction_set


def prepare_execution_artifacts(
    plan: ExecutionPlan,
    artifact_dir: str | Path | None,
) -> None:
    """Write shared case evidence and reject existing run destinations."""
    if artifact_dir is None:
        return

    root = Path(artifact_dir)
    json.dumps(plan.metadata, cls=JSONEncoder)
    cases: dict[str, BenchmarkCase] = {}
    for run_spec in plan.runs:
        case = cases.setdefault(run_spec.case.name, run_spec.case)
        if case is not run_spec.case and json.dumps(
            _case_payload(case), cls=JSONEncoder, sort_keys=True
        ) != json.dumps(_case_payload(run_spec.case), cls=JSONEncoder, sort_keys=True):
            raise ValueError(f"execution plan contains conflicting cases named {case.name!r}")

    case_artifacts = [_case_artifact(root, case) for case in cases.values()]
    for path, text in case_artifacts:
        _check_case_artifact(path, text)
    for run_spec in plan.runs:
        _prepare_run_destination(run_spec, root)
    for path, text in case_artifacts:
        if not path.exists():
            _write_text_atomic(path, text)


def prepare_run_artifacts(
    run_spec: RunSpec,
    artifact_dir: str | Path | None,
) -> None:
    """Prepare artifacts for the direct execution of one run spec."""
    if artifact_dir is None:
        return
    root = Path(artifact_dir)
    _prepare_run_destination(run_spec, root)
    path, text = _case_artifact(root, run_spec.case)
    _check_case_artifact(path, text)
    if not path.exists():
        _write_text_atomic(path, text)


def save_run_artifacts(
    *,
    case: BenchmarkCase,
    result: RunResult,
    artifact_dir: str | Path | None,
) -> RunResult:
    """Save one run record and optional predictions beside its case evidence."""
    if artifact_dir is None:
        return replace(
            result,
            predictions=_with_run_identity(result),
        )

    root = Path(artifact_dir)
    record = result.record
    run_dir = _run_directory(
        root,
        case_name=record.case_name,
        method_name=record.method.name,
        resample_id=record.resample_id,
        seed=record.seed,
    )
    case_path = _case_path(root, case.name)
    run_path = run_dir / "run.json"
    prediction_path = run_dir / "predictions.npz" if result.predictions is not None else None
    record = replace(
        record,
        case_path=str(case_path),
        run_path=str(run_path),
        prediction_path=None if prediction_path is None else str(prediction_path),
    )
    result = replace(result, record=record)
    predictions = _with_run_identity(result)
    result = replace(result, predictions=predictions)

    # Serialize first so unsupported metadata cannot leave partial artifacts.
    run_text = (
        json.dumps(
            _run_payload(result, run_dir=run_dir, case_path=case_path),
            cls=JSONEncoder,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        if predictions is not None and prediction_path is not None:
            save_prediction_set(predictions, temporary_dir / prediction_path.name)
        (temporary_dir / run_path.name).write_text(run_text, encoding="utf-8")
        temporary_dir.rename(run_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return result


def _prepare_run_destination(run_spec: RunSpec, root: Path) -> None:
    # This also validates that planned method/runtime metadata is JSON evidence.
    json.dumps(
        {
            "method": asdict(run_spec.method),
            "runtime": run_spec.runtime,
        },
        cls=JSONEncoder,
    )
    run_dir = _run_directory(
        root,
        case_name=run_spec.case.name,
        method_name=run_spec.method.name,
        resample_id=run_spec.resample_id,
        seed=run_spec.seed,
    )
    if run_dir.exists():
        raise FileExistsError(f"run artifact destination already exists: {run_dir}")


def _case_payload(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "case": {"name": case.name, "metadata": case.metadata},
        "dataset": _dataset_payload(case.dataset),
        "task": asdict(case.task),
        "resampling": {
            "spec": asdict(case.resampling.spec),
            "resamples": [
                {
                    "id": resample.id,
                    "train_idx": resample.train_idx,
                    "valid_idx": resample.valid_idx,
                    "test_idx": resample.test_idx,
                    "metadata": resample.metadata,
                }
                for resample in case.resampling.resamples
            ],
            "metadata": case.resampling.metadata,
        },
    }


def _dataset_payload(dataset: Any) -> dict[str, Any]:
    source = dataset_source(dataset.metadata)
    if isinstance(dataset, Dataset):
        return {
            "name": dataset.name,
            "source": source,
            "rows": len(dataset),
            "columns": [str(column) for column in dataset.data.columns],
            "schema": asdict(dataset.schema),
            "row_id": dataset.row_id,
        }
    if isinstance(dataset, MultiInstanceDataset):
        return {
            "name": dataset.name,
            "source": source,
            "bags": dataset.n_bags,
            "instances": dataset.n_instances,
            "bag_columns": [str(column) for column in dataset.bag_table.columns],
            "instance_columns": [str(column) for column in dataset.instance_table.columns],
            "bag_schema": asdict(dataset.bag_schema),
            "instance_schema": asdict(dataset.instance_schema),
            "bag_id_column": dataset.bag_id_column,
            "instance_id_column": dataset.instance_id_column,
        }
    raise TypeError(f"unsupported benchmark dataset type {type(dataset).__name__}")


def _run_payload(result: RunResult, *, run_dir: Path, case_path: Path) -> dict[str, Any]:
    record = result.record
    return {
        "run": {
            "id": record.run_id,
            "case_name": record.case_name,
            "selected_resample_id": record.resample_id,
            "seed": record.seed,
            "status": record.status,
            "runtime": record.runtime,
            "environment": record.environment,
            "metadata": record.metadata,
        },
        "case_path": os.path.relpath(case_path, run_dir),
        "method": asdict(record.method),
        "result": {
            "primary_metric": record.primary_metric,
            "primary_metric_greater_is_better": record.primary_metric_greater_is_better,
            "metrics": record.metrics,
            "fit_time": record.fit_time,
            "predict_time": record.predict_time,
            "prediction_path": None if result.predictions is None else "predictions.npz",
            "error": record.error,
        },
    }


def _with_run_identity(result: RunResult) -> PredictionSet | None:
    predictions = result.predictions
    if predictions is None:
        return None
    record = result.record
    return replace(
        predictions,
        metadata={
            **dict(predictions.metadata or {}),
            "run_id": record.run_id,
            "case_name": record.case_name,
            "seed": record.seed,
        },
    )


def _case_path(root: Path, case_name: str) -> Path:
    return root / _path_part(case_name, "case name") / "case.json"


def _case_artifact(root: Path, case: BenchmarkCase) -> tuple[Path, str]:
    text = json.dumps(_case_payload(case), cls=JSONEncoder, indent=2, sort_keys=True) + "\n"
    return _case_path(root, case.name), text


def _check_case_artifact(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"case evidence already exists with different content: {path}")


def _run_directory(
    root: Path,
    *,
    case_name: str,
    method_name: str,
    resample_id: str,
    seed: int,
) -> Path:
    return (
        root
        / _path_part(case_name, "case name")
        / _path_part(method_name, "method name")
        / _path_part(resample_id, "resample id")
        / f"seed_{seed}"
    )


def _path_part(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a single path component, got {value!r}")
    return value


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as file:
        file.write(text)
        temporary_path = Path(file.name)
    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
