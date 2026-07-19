from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from rtml.core.runs import RunRecord, RunResult

Row: TypeAlias = dict[str, Any]

DEFAULT_AGGREGATE_GROUP_BY = (
    "metadata.study_name",
    "case_name",
    "dataset_name",
    "task_name",
    "method_name",
)
DEFAULT_TIMING_FIELDS = ("fit_time", "predict_time")


def run_record_row(record: RunRecord) -> Row:
    """Flatten one run record into a summary-table row."""
    row: Row = {
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


def run_results_table(results: list[RunResult]) -> list[Row]:
    """Convert run results into rows suitable for CSV/JSON reporting."""
    return [run_record_row(result.record) for result in results]


def save_run_summary(
    results: list[RunResult],
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> list[Row]:
    """Save a lightweight run summary table and return the generated rows."""
    rows = run_results_table(results)
    if csv_path is not None:
        _write_csv(rows, Path(csv_path))
    if json_path is not None:
        _write_json(rows, Path(json_path))
    return rows


def load_run_summary(path: str | Path) -> list[Row]:
    """Load a saved run summary from CSV or JSON."""
    path = Path(path)
    if path.suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"expected a list of rows in {path}")
        rows = []
        for row in loaded:
            if not isinstance(row, Mapping):
                raise ValueError(f"expected row objects in {path}")
            rows.append(dict(row))
        return rows
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def aggregate_run_summary(
    rows: list[Row],
    *,
    group_by: tuple[str, ...] = DEFAULT_AGGREGATE_GROUP_BY,
    timing_fields: tuple[str, ...] = DEFAULT_TIMING_FIELDS,
) -> list[Row]:
    """Aggregate per-run summary rows into comparison-ready grouped rows."""
    groups: dict[tuple[Any, ...], list[Row]] = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in group_by)
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group_rows in groups.items():
        aggregate_row: Row = dict(zip(group_by, key, strict=True))
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
    rows: list[Row],
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    group_by: tuple[str, ...] = DEFAULT_AGGREGATE_GROUP_BY,
    timing_fields: tuple[str, ...] = DEFAULT_TIMING_FIELDS,
) -> list[Row]:
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


def _write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _row_fields(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _aggregate_value_fields(
    rows: list[Row],
    *,
    timing_fields: tuple[str, ...],
) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if (field.startswith("metric.") or field in timing_fields) and field not in fields:
                fields.append(field)
    return fields


def _row_fields(rows: list[Row]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def _copy_unique_field(
    aggregate_row: Row,
    rows: list[Row],
    field: str,
) -> None:
    value = _consistent_non_empty_value(rows, field)
    if value is not None:
        aggregate_row[field] = value


def _copy_unique_metadata_fields(
    aggregate_row: Row,
    rows: list[Row],
) -> None:
    for field in _metadata_fields(rows):
        _copy_unique_field(aggregate_row, rows, field)


def _copy_primary_metric_summary(
    aggregate_row: Row,
    rows: list[Row],
) -> None:
    primary_metric = _consistent_non_empty_value(rows, "primary_metric")
    if primary_metric is None:
        return

    metric_name = str(primary_metric)
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


def _metadata_fields(rows: list[Row]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field.startswith("metadata.") and field not in fields:
                fields.append(field)
    return fields


def _consistent_non_empty_value(rows: list[Row], field: str) -> Any | None:
    sentinel = object()
    first_value: Any = sentinel
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        if first_value is sentinel:
            first_value = value
            continue
        if value != first_value:
            return None
    if first_value is sentinel:
        return None
    return first_value


def _numeric_values(rows: list[Row], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
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
