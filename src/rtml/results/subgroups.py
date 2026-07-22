from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from rtml.core.metrics import MetricRequest, compute_metric
from rtml.core.results import PredictionSet
from rtml.results.artifacts import load_prediction_set
from rtml.results.reports import LOWER_IS_BETTER_METRICS, Row, save_rows


def subgroup_metric_rows(
    rows: list[Row],
    *,
    metrics_by_task: Mapping[str, Iterable[MetricRequest]],
    min_count: int = 1,
) -> list[Row]:
    """Compute subgroup metric rows from saved prediction artifacts."""
    subgroup_rows: list[Row] = []
    for row in rows:
        prediction_path = row.get("prediction_path")
        if not prediction_path:
            continue
        predictions = load_prediction_set(Path(str(prediction_path)))
        if not predictions.subgroups:
            continue
        metrics = list(metrics_by_task.get(str(row.get("task_name", "")), ()))
        if not metrics:
            continue
        for subgroup_column, values in predictions.subgroups.items():
            subgroup_rows.extend(
                _subgroup_metric_rows(
                    row=row,
                    predictions=predictions,
                    metrics=metrics,
                    subgroup_column=subgroup_column,
                    values=np.asarray(values),
                    min_count=min_count,
                )
            )
    _mark_worst_primary_metric_subgroups(subgroup_rows)
    return subgroup_rows


def save_subgroup_summary(
    rows: list[Row],
    *,
    metrics_by_task: Mapping[str, Iterable[MetricRequest]],
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    min_count: int = 1,
) -> list[Row]:
    """Save optional subgroup metrics derived from prediction artifacts."""
    subgroup_rows = subgroup_metric_rows(
        rows,
        metrics_by_task=metrics_by_task,
        min_count=min_count,
    )
    save_rows(subgroup_rows, csv_path=csv_path, json_path=json_path, markdown_path=markdown_path)
    return subgroup_rows


def _subgroup_metric_rows(
    *,
    row: Row,
    predictions: PredictionSet,
    metrics: list[MetricRequest],
    subgroup_column: str,
    values: np.ndarray,
    min_count: int,
) -> list[Row]:
    output_rows: list[Row] = []
    overall_metrics = _compute_available_metrics(predictions, metrics)
    for subgroup_value in _ordered_unique(values):
        mask = values == subgroup_value
        count = int(np.sum(mask))
        if count < min_count:
            continue
        subgroup_predictions = _subset_predictions(predictions, mask)
        output_row = _base_subgroup_row(
            row=row,
            subgroup_column=subgroup_column,
            subgroup_value=subgroup_value,
            count=count,
            total=len(values),
        )
        for metric in metrics:
            metric_name = metric.name
            try:
                value = _compute_metric_quietly(metric, subgroup_predictions)
            except Exception as exc:  # noqa: BLE001 - report per-subgroup metric failures.
                output_row[f"metric.{metric_name}.error"] = str(exc) or type(exc).__name__
                continue
            if not np.isfinite(value):
                output_row[f"metric.{metric_name}.error"] = "metric returned non-finite value"
                continue
            output_row[f"metric.{metric_name}"] = value
            if metric_name in overall_metrics:
                output_row[f"metric.{metric_name}.delta_from_overall"] = (
                    value - overall_metrics[metric_name]
                )
        output_rows.append(output_row)
    return output_rows


def _base_subgroup_row(
    *,
    row: Row,
    subgroup_column: str,
    subgroup_value: object,
    count: int,
    total: int,
) -> Row:
    output_row: Row = {
        "run_id": row.get("run_id", ""),
        "case_name": row.get("case_name", ""),
        "dataset_name": row.get("dataset_name", ""),
        "task_name": row.get("task_name", ""),
        "method_name": row.get("method_name", ""),
        "resample_id": row.get("resample_id", ""),
        "seed": row.get("seed", ""),
        "primary_metric": row.get("primary_metric", ""),
        "subgroup_column": subgroup_column,
        "subgroup_value": subgroup_value,
        "subgroup_count": count,
        "subgroup_fraction": count / total if total else None,
    }
    for name, value in row.items():
        if name.startswith("metadata."):
            output_row[name] = value
    return output_row


def _compute_available_metrics(
    predictions: PredictionSet,
    metrics: list[MetricRequest],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in metrics:
        try:
            value = _compute_metric_quietly(metric, predictions)
        except Exception:  # noqa: BLE001 - invalid overall metric is reflected by missing deltas.
            continue
        if np.isfinite(value):
            values[metric.name] = value
    return values


def _compute_metric_quietly(metric: MetricRequest, predictions: PredictionSet) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_metric(metric, predictions)


def _subset_predictions(predictions: PredictionSet, mask: np.ndarray) -> PredictionSet:
    return replace(
        predictions,
        row_ids=predictions.row_ids[mask],
        y_true=None if predictions.y_true is None else predictions.y_true[mask],
        labels=None if predictions.labels is None else predictions.labels[mask],
        probabilities=None
        if predictions.probabilities is None
        else predictions.probabilities[mask],
        scores=None if predictions.scores is None else predictions.scores[mask],
        values=None if predictions.values is None else predictions.values[mask],
        subgroups={name: values[mask] for name, values in predictions.subgroups.items()},
    )


def _ordered_unique(values: np.ndarray) -> list[Any]:
    seen: list[Any] = []
    for value in values.tolist():
        if value not in seen:
            seen.append(value)
    return seen


def _mark_worst_primary_metric_subgroups(rows: list[Row]) -> None:
    groups: dict[tuple[Any, Any, Any], list[Row]] = {}
    for row in rows:
        metric_name = row.get("primary_metric")
        if not metric_name:
            continue
        metric_value = _finite_number(row.get(f"metric.{metric_name}"))
        if metric_value is None:
            continue
        key = (row.get("run_id"), row.get("subgroup_column"), metric_name)
        groups.setdefault(key, []).append(row)

    for (_, _, metric_name), group_rows in groups.items():
        reverse = metric_name in LOWER_IS_BETTER_METRICS
        worst = sorted(
            group_rows,
            key=lambda row: _finite_number(row.get(f"metric.{metric_name}")) or 0.0,
            reverse=reverse,
        )[0]
        worst["primary_metric_worst_subgroup"] = True


def _finite_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(number):
        return number
    return None
