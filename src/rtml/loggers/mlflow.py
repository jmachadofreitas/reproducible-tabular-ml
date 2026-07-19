from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rtml.runs.base import RunRecord

DEFAULT_MLFLOW_EXPERIMENT_NAME = "rtml"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///.runs/mlflow/mlflow.db"
DEFAULT_MLFLOW_ARTIFACT_SUBDIR = "artifacts"


def _ensure_sqlite_parent(tracking_uri: str) -> None:
    if not tracking_uri.startswith("sqlite:///"):
        return
    db_path = tracking_uri.removeprefix("sqlite:///")
    if db_path and db_path != ":memory:":
        Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


class MLflowLogger:
    """MLflow adapter for RTML run records and artifacts."""

    def __init__(
        self,
        *,
        experiment_name: str | None = DEFAULT_MLFLOW_EXPERIMENT_NAME,
        tracking_uri: str | None = DEFAULT_MLFLOW_TRACKING_URI,
        artifact_subdir: str | None = DEFAULT_MLFLOW_ARTIFACT_SUBDIR,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        self.experiment_name = experiment_name or DEFAULT_MLFLOW_EXPERIMENT_NAME
        self.tracking_uri = tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
        self.artifact_subdir = artifact_subdir or DEFAULT_MLFLOW_ARTIFACT_SUBDIR
        _ensure_sqlite_parent(self.tracking_uri)
        self._mlflow.set_tracking_uri(self.tracking_uri)
        experiment = self._mlflow.set_experiment(self.experiment_name)
        self._experiment_id = experiment.experiment_id

    def log_running_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        self._mlflow.log_metrics(dict(metrics), step=step)

    def start_run(self, *, run_name: str | None = None):
        """Open an MLflow run for streaming metrics before final RTML logging."""
        return self._mlflow.start_run(run_name=run_name, experiment_id=self._experiment_id)

    def log_run(
        self,
        record: RunRecord,
        *,
        artifact_paths: Sequence[str | Path] = (),
    ) -> str | None:
        run_name = f"{record.case_name}/{record.method.name}/{record.resample_id}"
        active_run = self._mlflow.active_run()
        if active_run is not None:
            self._log_run_info(record)
            self._log_metrics(record)
            self._log_artifacts(record, artifact_paths)
            return active_run.info.run_id

        with self.start_run(run_name=run_name) as active_run:
            self._log_run_info(record)
            self._log_metrics(record)
            self._log_artifacts(record, artifact_paths)
            return active_run.info.run_id

    def _log_run_info(self, record: RunRecord) -> None:
        params: dict[str, str | int | float | bool] = {
            "case_name": record.case_name,
            "dataset_name": record.dataset_name,
            "dataset_fingerprint": record.dataset_fingerprint,
            "task_name": record.task_name,
            "task_type": record.task_type.value,
            "primary_metric": record.primary_metric or "",
            "resampling_plan_fingerprint": record.resampling_plan_fingerprint,
            "resample_id": record.resample_id,
            "method_name": record.method.name,
            "seed": record.seed,
        }
        params.update(self._flatten_mapping("transform", record.method.transform))
        params.update(self._flatten_mapping("model", asdict(record.method.model)))
        params.update(self._flatten_mapping("training", record.method.training))
        params.update(self._flatten_mapping("runtime", asdict(record.runtime)))
        params.update(self._flatten_mapping("metadata", record.metadata))
        self._mlflow.log_params(params)

        tags = {
            "rtml.run_id": record.run_id,
            "rtml.status": record.status,
            "rtml.dataset": record.dataset_name,
            "rtml.task": record.task_name,
            "rtml.method": record.method.name,
        }
        if "paradigm" in record.metadata:
            tags["rtml.paradigm"] = str(record.metadata["paradigm"])
        if record.error is not None:
            tags["rtml.error"] = record.error
        self._mlflow.set_tags(tags)

    def _log_metrics(self, record: RunRecord) -> None:
        metrics = dict(record.metrics)
        if record.fit_time is not None:
            metrics["fit_time"] = record.fit_time
        if record.predict_time is not None:
            metrics["predict_time"] = record.predict_time
        if metrics:
            self._mlflow.log_metrics(metrics)

    def _log_artifacts(
        self,
        record: RunRecord,
        artifact_paths: Sequence[str | Path],
    ) -> None:
        paths: list[str | Path] = list(artifact_paths)
        if record.prediction_path is not None:
            paths.append(record.prediction_path)
        for path in paths:
            artifact_path = Path(path)
            if artifact_path.is_file():
                self._mlflow.log_artifact(str(artifact_path), artifact_path=self.artifact_subdir)
            elif artifact_path.is_dir():
                self._mlflow.log_artifacts(str(artifact_path), artifact_path=self.artifact_subdir)

    @staticmethod
    def _clean_param_value(value: Any) -> str | int | float | bool:
        if value is None:
            return ""
        if isinstance(value, str | int | float | bool):
            return value
        return str(value)

    @staticmethod
    def _flatten_mapping(
        prefix: str,
        values: Mapping[str, Any],
    ) -> dict[str, str | int | float | bool]:
        flattened: dict[str, str | int | float | bool] = {}
        for key, value in values.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                flattened.update(MLflowLogger._flatten_mapping(name, value))
            else:
                flattened[name] = MLflowLogger._clean_param_value(value)
        return flattened
