import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

from rtml.core.serialization import JSONEncoder

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


def run_record_row(record: RunRecord) -> Row:
    """Flatten one run record into a report row."""
    row: Row = {
        "run_key": record.run_key,
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
        "primary_metric_greater_is_better": record.primary_metric_greater_is_better,
        "fit_time": record.fit_time,
        "predict_time": record.predict_time,
        "prediction_path": record.prediction_path or "",
        "run_path": record.run_path or "",
        "case_path": record.case_path or "",
        "error": record.error or "",
    }
    row.update({f"metric.{name}": value for name, value in sorted(record.metrics.items())})
    row.update({f"metadata.{name}": value for name, value in sorted(record.metadata.items())})
    return row


def run_results_table(results: list[RunResult]) -> pd.DataFrame:
    """Build the per-run report table."""
    return pd.DataFrame.from_records(run_record_row(result.record) for result in results)


def save_run_summary(
    results: list[RunResult],
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build, save, and return the per-run report table."""
    table = run_results_table(results)
    save_report(
        table,
        csv_path=csv_path,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    return table


def save_report(
    table: pd.DataFrame,
    *,
    csv_path: str | Path | None = None,
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    """Save a report table as CSV, JSON, and/or Markdown."""
    if csv_path is not None:
        path = _output_path(csv_path)
        table.to_csv(path, index=False)
    if json_path is not None:
        path = _output_path(json_path)
        records = table.astype(object).where(table.notna(), None).to_dict(orient="records")
        path.write_text(
            json.dumps(records, cls=JSONEncoder, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if markdown_path is not None:
        path = _output_path(markdown_path)
        if table.empty:
            path.write_text("", encoding="utf-8")
        else:
            display = table.astype(object).where(table.notna(), "").map(_markdown_value)
            path.write_text(
                display.to_markdown(index=False, floatfmt=".6g") + "\n",
                encoding="utf-8",
            )


def load_run_summary(path: str | Path) -> pd.DataFrame:
    """Load a saved run report from CSV or JSON."""
    path = Path(path)
    try:
        if path.suffix == ".json":
            return pd.read_json(path, orient="records")
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def aggregate_run_summary(
    table: pd.DataFrame,
    *,
    group_by: tuple[str, ...] = DEFAULT_AGGREGATE_GROUP_BY,
    timing_fields: tuple[str, ...] = DEFAULT_TIMING_FIELDS,
    include_factor_grouping: bool = False,
    add_ranks: bool = True,
    rank_group_by: tuple[str, ...] = DEFAULT_RANK_GROUP_BY,
    overall_rank_group_by: tuple[str, ...] = DEFAULT_OVERALL_RANK_GROUP_BY,
    method_field: str = DEFAULT_METHOD_FIELD,
) -> pd.DataFrame:
    """Aggregate per-run metrics, failures, timings, and method ranks."""
    if table.empty:
        return pd.DataFrame()

    runs = table.copy()
    groups = list(group_by)
    if include_factor_grouping:
        groups.extend(
            field
            for field in runs.columns
            if field.startswith("metadata.factor.") and field not in groups
        )
    for field in groups:
        if field not in runs:
            runs[field] = ""

    value_fields = [
        field for field in runs.columns if field.startswith("metric.") or field in timing_fields
    ]
    if value_fields:
        numeric = runs[value_fields].apply(pd.to_numeric, errors="coerce")
        runs[value_fields] = numeric.where(np.isfinite(numeric))

    runs["_success"] = runs["status"].eq("success").astype(int)
    aggregations: dict[str, tuple[str, Any]] = {
        "n_runs": ("_success", "size"),
        "n_success": ("_success", "sum"),
    }
    if "task_type" in runs:
        aggregations["task_type"] = ("task_type", _one_non_empty_value)
    if "primary_metric" in runs:
        aggregations["primary_metric_name"] = (
            "primary_metric",
            _one_non_empty_value,
        )
    if "primary_metric_greater_is_better" in runs:
        aggregations["primary_metric_greater_is_better"] = (
            "primary_metric_greater_is_better",
            _one_non_empty_value,
        )
    metadata_fields = [
        field for field in runs.columns if field.startswith("metadata.") and field not in groups
    ]
    for field in metadata_fields:
        aggregations[field] = (field, _one_non_empty_value)
    for field in value_fields:
        aggregations.update(_summary_aggregations(field))

    summary = runs.groupby(groups, dropna=False, sort=False).agg(**aggregations).reset_index()
    summary = summary.drop(
        columns=[field for field in metadata_fields if summary[field].isna().all()]
    )
    summary.insert(
        summary.columns.get_loc("n_success") + 1, "n_failed", summary.n_runs - summary.n_success
    )

    for field in value_fields:
        summary[f"{field}.iqr"] = summary[f"{field}.q75"] - summary[f"{field}.q25"]
    _copy_primary_metric_stats(summary)

    if add_ranks:
        summary = _add_primary_metric_ranks(
            summary,
            rank_group_by=rank_group_by,
            overall_rank_group_by=overall_rank_group_by,
            method_field=method_field,
        )
    return summary


def save_aggregate_summary(
    table: pd.DataFrame,
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
) -> pd.DataFrame:
    """Build, save, and return the aggregate report table."""
    aggregate = aggregate_run_summary(
        table,
        group_by=group_by,
        timing_fields=timing_fields,
        include_factor_grouping=include_factor_grouping,
        add_ranks=add_ranks,
        rank_group_by=rank_group_by,
        overall_rank_group_by=overall_rank_group_by,
        method_field=method_field,
    )
    save_report(
        aggregate,
        csv_path=csv_path,
        json_path=json_path,
        markdown_path=markdown_path,
    )
    return aggregate


def _output_path(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _markdown_value(value: object) -> object:
    if isinstance(value, str):
        return value.replace("|", "\\|").replace("\n", " ")
    return value


def _one_non_empty_value(values: pd.Series) -> Any:
    present = [value for value in values.dropna() if not isinstance(value, str) or value]
    if not present:
        return pd.NA
    encoded = {json.dumps(value, cls=JSONEncoder, sort_keys=True) for value in present}
    return present[0] if len(encoded) == 1 else pd.NA


def _summary_aggregations(field: str) -> dict[str, tuple[str, Any]]:
    return {
        f"{field}.count": (field, "count"),
        f"{field}.mean": (field, "mean"),
        f"{field}.std": (field, "std"),
        f"{field}.min": (field, "min"),
        f"{field}.q25": (field, lambda values: values.quantile(0.25)),
        f"{field}.median": (field, "median"),
        f"{field}.q75": (field, lambda values: values.quantile(0.75)),
        f"{field}.max": (field, "max"),
    }


def _copy_primary_metric_stats(summary: pd.DataFrame) -> None:
    if "primary_metric_name" not in summary:
        return
    for metric_name in summary.primary_metric_name.dropna().unique():
        selected = summary.primary_metric_name.eq(metric_name)
        for statistic in ("count", "mean", "std", "iqr"):
            source = f"metric.{metric_name}.{statistic}"
            if source in summary:
                summary.loc[selected, f"primary_metric_{statistic}"] = summary.loc[selected, source]


def _add_primary_metric_ranks(
    summary: pd.DataFrame,
    *,
    rank_group_by: tuple[str, ...],
    overall_rank_group_by: tuple[str, ...],
    method_field: str,
) -> pd.DataFrame:
    required = {
        "primary_metric_mean",
        "primary_metric_count",
        "primary_metric_greater_is_better",
        "n_runs",
        "n_success",
        "n_failed",
        method_field,
    }
    if not required.issubset(summary.columns):
        return summary

    ranked = summary.copy()
    for field in (*rank_group_by, *overall_rank_group_by):
        if field not in ranked:
            ranked[field] = ""

    score = pd.to_numeric(ranked.primary_metric_mean, errors="coerce")
    count = pd.to_numeric(ranked.primary_metric_count, errors="coerce")
    n_runs = pd.to_numeric(ranked.n_runs, errors="coerce")
    n_success = pd.to_numeric(ranked.n_success, errors="coerce")
    n_failed = pd.to_numeric(ranked.n_failed, errors="coerce")
    direction = _boolean_series(ranked.primary_metric_greater_is_better)
    complete = (
        n_runs.gt(0)
        & n_success.eq(n_runs)
        & n_failed.eq(0)
        & count.eq(n_success)
        & score.notna()
        & direction.notna()
    )

    rank_field = "primary_metric_rank_by_dataset"
    for greater_is_better in (True, False):
        selected = complete & direction.eq(greater_is_better)
        ranked.loc[selected, rank_field] = (
            ranked.loc[selected]
            .groupby(list(rank_group_by), dropna=False, sort=False)["primary_metric_mean"]
            .rank(method="min", ascending=not greater_is_better)
        )
    if rank_field not in ranked:
        return ranked

    overall_groups = list(overall_rank_group_by)
    method_groups = [*overall_groups, method_field]
    comparison_fields = list(dict.fromkeys((*overall_groups, *rank_group_by)))
    comparison_counts = (
        ranked.drop_duplicates(comparison_fields)
        .groupby(overall_groups, dropna=False, sort=False)
        .size()
        .rename("_comparison_count")
        .reset_index()
    )
    method_ranks = (
        ranked.loc[ranked[rank_field].notna()]
        .groupby(method_groups, dropna=False, sort=False)[rank_field]
        .agg(primary_metric_rank_count="count", primary_metric_mean_rank="mean")
        .reset_index()
        .merge(comparison_counts, on=overall_groups, how="left", sort=False)
    )
    method_ranks["primary_metric_rank_coverage"] = (
        method_ranks.primary_metric_rank_count / method_ranks._comparison_count
    )
    has_full_coverage = method_ranks.primary_metric_rank_count.eq(method_ranks._comparison_count)
    method_ranks.loc[has_full_coverage, "primary_metric_overall_rank"] = (
        method_ranks.loc[has_full_coverage]
        .groupby(overall_groups, dropna=False, sort=False)["primary_metric_mean_rank"]
        .rank(method="min")
    )
    return ranked.merge(
        method_ranks.drop(columns="_comparison_count"),
        on=method_groups,
        how="left",
        sort=False,
    )


def _boolean_series(values: pd.Series) -> pd.Series:
    return values.astype("string").str.lower().map({"true": True, "false": False})
