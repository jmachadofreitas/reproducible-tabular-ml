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
DEFAULT_RANK_GROUP_BY = ("metadata.study_name", "case_name", "dataset_name", "task_name")
DEFAULT_OVERALL_RANK_GROUP_BY = ("metadata.study_name",)
DEFAULT_METHOD_FIELD = "method_name"
DEFAULT_TIMING_FIELDS = ("fit_time", "predict_time")
LOWER_IS_BETTER_METRICS = frozenset({"log_loss", "mse", "rmse", "mae"})


def run_record_row(record: RunRecord) -> Row:
    """Flatten one run record into a summary-table row."""
    row: Row = {
        "run_id": record.run_id,
        "case_name": record.case_name,
        "dataset_name": record.dataset_name,
        "task_name": record.task_name,
        "task_type": record.task_type.value,
        "dataset_fingerprint": record.dataset_fingerprint,
        "task_fingerprint": record.task_fingerprint,
        "method_name": record.method.name,
        "method_fingerprint": record.method_fingerprint,
        "model_kind": record.method.model.kind,
        "model_backend": record.method.model.backend,
        "resample_id": record.resample_id,
        "resampling_plan_fingerprint": record.resampling_plan_fingerprint,
        "seed": record.seed,
        "runtime_fingerprint": record.runtime_fingerprint,
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
    markdown_path: str | Path | None = None,
) -> list[Row]:
    """Save a lightweight run summary table and return the generated rows."""
    rows = run_results_table(results)
    save_rows(rows, csv_path=csv_path, json_path=json_path, markdown_path=markdown_path)
    return rows


def save_rows(
    rows: list[Row],
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    """Save already prepared rows as CSV, JSON, and/or Markdown."""
    if csv_path is not None:
        _write_csv(rows, Path(csv_path))
    if json_path is not None:
        _write_json(rows, Path(json_path))
    if markdown_path is not None:
        _write_markdown(rows, Path(markdown_path))


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
    include_factor_grouping: bool = False,
    add_ranks: bool = True,
    rank_group_by: tuple[str, ...] = DEFAULT_RANK_GROUP_BY,
    overall_rank_group_by: tuple[str, ...] = DEFAULT_OVERALL_RANK_GROUP_BY,
    method_field: str = DEFAULT_METHOD_FIELD,
) -> list[Row]:
    """Aggregate per-run summary rows into comparison-ready grouped rows."""
    group_by = _with_factor_fields(rows, group_by) if include_factor_grouping else group_by
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
    if add_ranks:
        _add_primary_metric_ranks(
            aggregate_rows,
            rank_group_by=rank_group_by,
            overall_rank_group_by=overall_rank_group_by,
            method_field=method_field,
        )
    return aggregate_rows


def save_aggregate_summary(
    rows: list[Row],
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    group_by: tuple[str, ...] = DEFAULT_AGGREGATE_GROUP_BY,
    timing_fields: tuple[str, ...] = DEFAULT_TIMING_FIELDS,
    include_factor_grouping: bool = False,
    add_ranks: bool = True,
    rank_group_by: tuple[str, ...] = DEFAULT_RANK_GROUP_BY,
    overall_rank_group_by: tuple[str, ...] = DEFAULT_OVERALL_RANK_GROUP_BY,
    method_field: str = DEFAULT_METHOD_FIELD,
) -> list[Row]:
    """Save grouped aggregate reports from a run summary table."""
    aggregate_rows = aggregate_run_summary(
        rows,
        group_by=group_by,
        timing_fields=timing_fields,
        include_factor_grouping=include_factor_grouping,
        add_ranks=add_ranks,
        rank_group_by=rank_group_by,
        overall_rank_group_by=overall_rank_group_by,
        method_field=method_field,
    )
    save_rows(
        aggregate_rows,
        csv_path=csv_path,
        json_path=json_path,
        markdown_path=markdown_path,
    )
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


def _write_markdown(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _row_fields(rows)
    if not fields:
        path.write_text("", encoding="utf-8")
        return

    lines = [
        "| " + " | ".join(_markdown_cell(field) for field in fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(_format_markdown_value(row.get(field))) for field in fields)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


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


def _with_factor_fields(rows: list[Row], group_by: tuple[str, ...]) -> tuple[str, ...]:
    fields = list(group_by)
    for field in _factor_fields(rows):
        if field not in fields:
            fields.append(field)
    return tuple(fields)


def _factor_fields(rows: list[Row]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field.startswith("metadata.factor.") and field not in fields:
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


def _add_primary_metric_ranks(
    rows: list[Row],
    *,
    rank_group_by: tuple[str, ...],
    overall_rank_group_by: tuple[str, ...],
    method_field: str,
) -> None:
    rank_groups: dict[tuple[Any, ...], list[Row]] = {}
    for row in rows:
        if row.get("primary_metric_name") and row.get("primary_metric_mean") is not None:
            key = tuple(row.get(field, "") for field in rank_group_by)
            rank_groups.setdefault(key, []).append(row)

    for group_rows in rank_groups.values():
        metric_name = _consistent_non_empty_value(group_rows, "primary_metric_name")
        if metric_name is None:
            continue
        reverse = not _lower_is_better(str(metric_name))
        ranked_rows = [
            row for row in group_rows if _finite_number(row.get("primary_metric_mean")) is not None
        ]
        _assign_competition_ranks(
            ranked_rows,
            value_field="primary_metric_mean",
            rank_field="primary_metric_rank_by_dataset",
            reverse=reverse,
        )

    overall_groups: dict[tuple[Any, ...], list[Row]] = {}
    for row in rows:
        if row.get("primary_metric_rank_by_dataset") is not None:
            key = tuple(row.get(field, "") for field in overall_rank_group_by)
            overall_groups.setdefault(key, []).append(row)

    for group_rows in overall_groups.values():
        rank_by_method: dict[Any, list[float]] = {}
        for row in group_rows:
            method = row.get(method_field)
            rank = _finite_number(row.get("primary_metric_rank_by_dataset"))
            if method is not None and rank is not None:
                rank_by_method.setdefault(method, []).append(rank)

        method_rows: list[Row] = []
        for method, ranks in rank_by_method.items():
            mean_rank = sum(ranks) / len(ranks)
            for row in group_rows:
                if row.get(method_field) == method:
                    row["primary_metric_mean_rank"] = mean_rank
            method_rows.append({method_field: method, "primary_metric_mean_rank": mean_rank})

        _assign_competition_ranks(
            method_rows,
            value_field="primary_metric_mean_rank",
            rank_field="primary_metric_overall_rank",
            reverse=False,
        )
        overall_rank_by_method = {
            row[method_field]: row["primary_metric_overall_rank"] for row in method_rows
        }
        for row in group_rows:
            method = row.get(method_field)
            if method in overall_rank_by_method:
                row["primary_metric_overall_rank"] = overall_rank_by_method[method]


def _assign_competition_ranks(
    rows: list[Row],
    *,
    value_field: str,
    rank_field: str,
    reverse: bool,
) -> None:
    rows.sort(key=lambda row: _finite_number(row.get(value_field)) or 0.0, reverse=reverse)
    previous_value: float | None = None
    previous_rank: int | None = None
    for index, row in enumerate(rows, start=1):
        value = _finite_number(row.get(value_field))
        if value is None:
            continue
        rank = previous_rank if previous_value == value and previous_rank is not None else index
        row[rank_field] = rank
        previous_value = value
        previous_rank = rank


def _lower_is_better(metric_name: str) -> bool:
    return metric_name in LOWER_IS_BETTER_METRICS


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
        numeric = _finite_number(row.get(field))
        if numeric is not None:
            values.append(numeric)
    return values


def _finite_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric):
        return numeric
    return None


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
