from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rtml.runs.base import RunRecord, RunResult

DEFAULT_AGGREGATE_GROUP_BY = (
    "metadata.study_name",
    "case_name",
    "dataset_name",
    "task_name",
    "method_name",
)
DEFAULT_TIMING_FIELDS = ("fit_time", "predict_time")


def run_record_row(record: RunRecord) -> dict[str, Any]:
    """Flatten one run record into a summary-table row."""
    row: dict[str, Any] = {
        "run_id": record.run_id,
        "case_name": record.case_name,
        "dataset_name": record.dataset_name,
        "task_name": record.task_name,
        "task_type": record.task_type.value,
        "method_name": record.method.name,
        "model_kind": record.method.model.kind,
        "model_backend": record.method.model.backend,
        "resample_id": record.resample_id,
        "seed": record.seed,
        "status": record.status,
        "primary_metric": record.primary_metric or "",
        "fit_time": record.fit_time,
        "predict_time": record.predict_time,
        "prediction_path": record.prediction_path or "",
        "error": record.error or "",
    }
    for name, value in sorted(record.metrics.items()):
        row[f"metric.{name}"] = value
    for name, value in sorted(record.metadata.items()):
        row[f"metadata.{name}"] = value
    return row


def run_results_table(results: list[RunResult]) -> list[dict[str, Any]]:
    """Convert run results into rows suitable for CSV/JSON reporting."""
    return [run_record_row(result.record) for result in results]


def save_run_summary(
    results: list[RunResult],
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Save a lightweight run summary table and return the generated rows."""
    rows = run_results_table(results)
    if csv_path is not None:
        _write_csv(rows, Path(csv_path))
    if json_path is not None:
        _write_json(rows, Path(json_path))
    return rows


def load_run_summary(path: str | Path) -> list[dict[str, Any]]:
    """Load a saved run summary from CSV or JSON."""
    path = Path(path)
    if path.suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"expected a list of rows in {path}")
        return [dict(row) for row in loaded]
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def aggregate_run_summary(
    rows: list[dict[str, Any]],
    *,
    group_by: tuple[str, ...] = DEFAULT_AGGREGATE_GROUP_BY,
    timing_fields: tuple[str, ...] = DEFAULT_TIMING_FIELDS,
) -> list[dict[str, Any]]:
    """Aggregate per-run summary rows into comparison-ready grouped rows."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_by)
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group_rows in sorted(groups.items()):
        aggregate_row = dict(zip(group_by, key, strict=True))
        aggregate_row["n_runs"] = len(group_rows)
        aggregate_row["n_success"] = sum(row.get("status") == "success" for row in group_rows)
        aggregate_row["n_failed"] = sum(row.get("status") != "success" for row in group_rows)
        _copy_unique_field(aggregate_row, group_rows, "task_type")
        _copy_unique_metadata_fields(aggregate_row, group_rows)
        _copy_primary_metric_summary(aggregate_row, group_rows)

        for field in _aggregate_value_fields(group_rows, timing_fields=timing_fields):
            values = _numeric_values(group_rows, field)
            if not values:
                continue
            for stat_name, value in _summary_stats(values).items():
                aggregate_row[f"{field}.{stat_name}"] = value

        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def save_aggregate_summary(
    rows: list[dict[str, Any]],
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    group_by: tuple[str, ...] = DEFAULT_AGGREGATE_GROUP_BY,
    timing_fields: tuple[str, ...] = DEFAULT_TIMING_FIELDS,
) -> list[dict[str, Any]]:
    """Save grouped aggregate reports from a run summary table."""
    aggregate_rows = aggregate_run_summary(
        rows,
        group_by=group_by,
        timing_fields=timing_fields,
    )
    if csv_path is not None:
        _write_csv(aggregate_rows, Path(csv_path))
    if json_path is not None:
        _write_json(aggregate_rows, Path(json_path))
    return aggregate_rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def _aggregate_value_fields(
    rows: list[dict[str, Any]],
    *,
    timing_fields: tuple[str, ...],
) -> list[str]:
    fields = {
        field
        for row in rows
        for field in row
        if field.startswith("metric.") or field in timing_fields
    }
    return sorted(fields)


def _copy_unique_field(
    aggregate_row: dict[str, Any],
    rows: list[dict[str, Any]],
    field: str,
) -> None:
    values = sorted({row.get(field) for row in rows if row.get(field) not in {None, ""}})
    if len(values) == 1:
        aggregate_row[field] = values[0]


def _copy_unique_metadata_fields(
    aggregate_row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    fields = sorted({field for row in rows for field in row if field.startswith("metadata.")})
    for field in fields:
        _copy_unique_field(aggregate_row, rows, field)


def _copy_primary_metric_summary(
    aggregate_row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    values = sorted({row.get("primary_metric") for row in rows if row.get("primary_metric")})
    if len(values) != 1:
        return

    metric_name = str(values[0])
    metric_field = f"metric.{metric_name}"
    metric_values = _numeric_values(rows, metric_field)
    aggregate_row["primary_metric_name"] = metric_name
    if not metric_values:
        return

    stats = _summary_stats(metric_values)
    aggregate_row["primary_metric_count"] = stats["count"]
    aggregate_row["primary_metric_mean"] = stats["mean"]
    aggregate_row["primary_metric_std"] = stats["std"]
    aggregate_row["primary_metric_iqr"] = stats["iqr"]


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in {None, ""}:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def _summary_stats(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    q25 = _quantile(ordered, 0.25)
    q75 = _quantile(ordered, 0.75)
    mean = sum(ordered) / len(ordered)
    return {
        "count": len(ordered),
        "mean": mean,
        "std": _sample_std(ordered, mean),
        "min": ordered[0],
        "q25": q25,
        "median": _quantile(ordered, 0.5),
        "q75": q75,
        "max": ordered[-1],
        "iqr": q75 - q25,
    }


def _sample_std(values: list[float], mean: float) -> float | None:
    if len(values) < 2:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _quantile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)
