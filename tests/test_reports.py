import csv
import json
from typing import Any

import pandas as pd
import pytest

from rtml.core.methods import MethodSpec, ModelSpec
from rtml.core.runs import RunRecord, RunResult
from rtml.core.runtime import RuntimeSpec
from rtml.core.tasks import TaskType
from rtml.results.reports import (
    aggregate_run_summary,
    load_run_summary,
    save_aggregate_summary,
    save_run_summary,
)


def make_result(
    *,
    accuracy: float = 0.9,
    metric_name: str = "accuracy",
    dataset_name: str = "dataset",
    method_name: str = "logreg_linear",
    resample_id: str = "fold_00",
    seed: int = 0,
    status: str = "success",
    metadata: dict[str, Any] | None = None,
    greater_is_better: bool = True,
) -> RunResult:
    method = MethodSpec(
        name=method_name,
        transform={"policy": "linear_default"},
        model=ModelSpec(kind="logistic_regression", backend="sklearn"),
    )
    record = RunRecord(
        run_key=f"{dataset_name}:{method_name}:{resample_id}:{seed}",
        case_name=f"{dataset_name}_case",
        dataset_name=dataset_name,
        task_name="task",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric=metric_name,
        resample_id=resample_id,
        method=method,
        seed=seed,
        runtime=RuntimeSpec(num_threads=2),
        environment={"python_version": "3.14", "platform": "test", "packages": {}},
        status=status,  # type: ignore
        primary_metric_greater_is_better=greater_is_better,
        metrics={metric_name: accuracy},
        fit_time=0.1,
        predict_time=0.01,
        prediction_path="predictions.npz",
        run_path="run.json",
        case_path="case.json",
        metadata={
            "factor.family": "linear",
            "preprocessing_policy": "linear_default",
            **dict(metadata or {}),
        },
    )
    return RunResult(predictions=None, record=record)


def test_save_run_summary_writes_csv_and_json(tmp_path) -> None:
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"

    table = save_run_summary([make_result()], csv_path=csv_path, json_path=json_path)
    row = table.iloc[0]

    assert row["metric.accuracy"] == 0.9
    assert row["run_path"] == "run.json"
    assert row["case_path"] == "case.json"
    assert row["model_kind"] == "logistic_regression"
    assert row["model_backend"] == "sklearn"
    with csv_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))

    assert csv_rows[0]["method_name"] == "logreg_linear"
    assert csv_rows[0]["metric.accuracy"] == "0.9"
    assert json_rows[0]["metadata.preprocessing_policy"] == "linear_default"


def test_save_aggregate_summary_groups_runs_with_dispersion(tmp_path) -> None:
    table = save_run_summary(
        [
            make_result(accuracy=0.8, resample_id="fold_00", seed=0),
            make_result(accuracy=0.9, resample_id="fold_01", seed=1),
            make_result(accuracy=1.0, resample_id="fold_02", seed=2),
        ]
    )

    aggregate = save_aggregate_summary(
        table,
        csv_path=tmp_path / "aggregate.csv",
        json_path=tmp_path / "aggregate.json",
    )

    assert len(aggregate) == 1
    row = aggregate.iloc[0]
    assert row["n_runs"] == 3
    assert row["n_success"] == 3
    assert row["n_failed"] == 0
    assert row["metadata.factor.family"] == "linear"
    assert row["metadata.preprocessing_policy"] == "linear_default"
    assert row["primary_metric_name"] == "accuracy"
    assert row["primary_metric_mean"] == pytest.approx(0.9)
    assert row["primary_metric_std"] == pytest.approx(0.1)
    assert row["primary_metric_iqr"] == pytest.approx(0.1)
    assert row["metric.accuracy.count"] == 3
    assert row["metric.accuracy.q25"] == pytest.approx(0.85)
    assert row["metric.accuracy.q75"] == pytest.approx(0.95)
    assert row["metric.accuracy.iqr"] == pytest.approx(0.1)
    assert (tmp_path / "aggregate.csv").exists()
    assert (tmp_path / "aggregate.json").exists()


def test_aggregate_summary_adds_case_and_overall_ranks() -> None:
    table = save_run_summary(
        [
            make_result(dataset_name="dataset_a", method_name="method_a", accuracy=0.9),
            make_result(dataset_name="dataset_a", method_name="method_b", accuracy=0.8),
            make_result(dataset_name="dataset_b", method_name="method_a", accuracy=0.7),
            make_result(dataset_name="dataset_b", method_name="method_b", accuracy=0.9),
            make_result(dataset_name="dataset_c", method_name="method_a", accuracy=0.75),
            make_result(dataset_name="dataset_c", method_name="method_b", accuracy=0.8),
        ]
    )
    by_dataset_method = aggregate_run_summary(table).set_index(["dataset_name", "method_name"])

    assert by_dataset_method.loc[("dataset_a", "method_a"), "primary_metric_rank_by_dataset"] == 1
    assert by_dataset_method.loc[("dataset_a", "method_b"), "primary_metric_rank_by_dataset"] == 2
    assert by_dataset_method.loc[("dataset_b", "method_b"), "primary_metric_rank_by_dataset"] == 1
    assert by_dataset_method.loc[("dataset_c", "method_b"), "primary_metric_rank_by_dataset"] == 1
    assert by_dataset_method.loc[("dataset_a", "method_a"), "primary_metric_overall_rank"] == 2
    assert by_dataset_method.loc[("dataset_a", "method_b"), "primary_metric_overall_rank"] == 1


def test_aggregate_summary_ranks_lower_error_metrics_as_better() -> None:
    table = save_run_summary(
        [
            make_result(
                metric_name="rmse",
                dataset_name="dataset",
                method_name="large",
                accuracy=2.0,
                greater_is_better=False,
            ),
            make_result(
                metric_name="rmse",
                dataset_name="dataset",
                method_name="small",
                accuracy=1.0,
                greater_is_better=False,
            ),
        ]
    )
    table["primary_metric_greater_is_better"] = "false"

    by_method = aggregate_run_summary(table).set_index("method_name")

    assert by_method.loc["small", "primary_metric_rank_by_dataset"] == 1
    assert by_method.loc["large", "primary_metric_rank_by_dataset"] == 2


def test_aggregate_summary_uses_declared_direction_for_custom_metrics() -> None:
    table = save_run_summary(
        [
            make_result(
                metric_name="custom_cost",
                method_name="large",
                accuracy=2.0,
                greater_is_better=False,
            ),
            make_result(
                metric_name="custom_cost",
                method_name="small",
                accuracy=1.0,
                greater_is_better=False,
            ),
        ]
    )

    by_method = aggregate_run_summary(table).set_index("method_name")

    assert by_method.loc["small", "primary_metric_rank_by_dataset"] == 1
    assert by_method.loc["large", "primary_metric_rank_by_dataset"] == 2


def test_aggregate_summary_does_not_rank_incomplete_method_coverage() -> None:
    table = save_run_summary(
        [
            make_result(dataset_name="dataset_a", method_name="complete", accuracy=0.8),
            make_result(dataset_name="dataset_b", method_name="complete", accuracy=0.7),
            make_result(dataset_name="dataset_a", method_name="incomplete", accuracy=0.9),
            make_result(
                dataset_name="dataset_b",
                method_name="incomplete",
                accuracy=0.9,
                status="failed",
            ),
        ]
    )

    by_dataset_method = aggregate_run_summary(table).set_index(["dataset_name", "method_name"])

    assert by_dataset_method.loc[("dataset_a", "complete"), "primary_metric_overall_rank"] == 1
    assert by_dataset_method.loc[("dataset_a", "incomplete"), "primary_metric_rank_coverage"] == 0.5
    assert pd.isna(
        by_dataset_method.loc[("dataset_a", "incomplete"), "primary_metric_overall_rank"]
    )


def test_aggregate_summary_can_group_by_factor_metadata() -> None:
    table = save_run_summary(
        [
            make_result(accuracy=0.7, metadata={"factor.dropout": "0.1"}),
            make_result(accuracy=0.9, metadata={"factor.dropout": "0.2"}),
        ]
    )

    default_table = aggregate_run_summary(table, add_ranks=False)
    factor_table = aggregate_run_summary(table, include_factor_grouping=True, add_ranks=False)

    assert len(default_table) == 1
    assert "metadata.factor.dropout" not in default_table
    assert len(factor_table) == 2
    assert set(factor_table["metadata.factor.dropout"]) == {"0.1", "0.2"}


def test_save_summaries_can_write_markdown(tmp_path) -> None:
    table = save_run_summary(
        [make_result(metadata={"factor.family": "linear|baseline"})],
        markdown_path=tmp_path / "summary.md",
    )
    aggregate = save_aggregate_summary(
        table,
        markdown_path=tmp_path / "aggregate.md",
    )

    summary_markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    aggregate_markdown = (tmp_path / "aggregate.md").read_text(encoding="utf-8")

    assert "run_key" in summary_markdown and "case_name" in summary_markdown
    assert "linear\\|baseline" in summary_markdown
    assert all(field in aggregate_markdown for field in ("n_runs", "n_success", "n_failed"))
    assert aggregate.iloc[0]["n_runs"] == 1


def test_aggregate_summary_can_be_recomputed_from_saved_summary(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    table = save_run_summary(
        [
            make_result(accuracy=0.7, resample_id="fold_00", seed=0),
            make_result(accuracy=0.9, resample_id="fold_01", seed=1, status="failed"),
        ],
        csv_path=summary_path,
        json_path=tmp_path / "summary.json",
    )

    loaded_csv = load_run_summary(summary_path)
    loaded_json = load_run_summary(tmp_path / "summary.json")
    aggregate = save_aggregate_summary(loaded_csv)

    assert loaded_csv.iloc[0]["metric.accuracy"] == 0.7
    pd.testing.assert_frame_equal(loaded_json, table, check_dtype=False)
    assert aggregate.iloc[0]["n_runs"] == 2
    assert aggregate.iloc[0]["n_success"] == 1
    assert aggregate.iloc[0]["n_failed"] == 1
    assert aggregate.iloc[0]["metric.accuracy.mean"] == pytest.approx(0.8)


def test_aggregate_summary_keeps_studies_separate_by_default() -> None:
    table = save_run_summary(
        [
            make_result(accuracy=0.7, metadata={"study_name": "baseline"}),
            make_result(accuracy=0.9, metadata={"study_name": "ablation"}),
        ]
    )

    aggregate = save_aggregate_summary(table)

    assert len(aggregate) == 2
    assert set(aggregate["metadata.study_name"]) == {"baseline", "ablation"}


def test_aggregate_summary_does_not_copy_conflicting_metadata() -> None:
    table = save_run_summary(
        [
            make_result(accuracy=0.7, metadata={"study_name": "same", "factor.C": "0.1"}),
            make_result(accuracy=0.9, metadata={"study_name": "same", "factor.C": "1.0"}),
        ]
    )

    aggregate = aggregate_run_summary(table)

    assert "metadata.factor.C" not in aggregate
    assert aggregate.iloc[0]["metadata.study_name"] == "same"


def test_aggregate_summary_keeps_consistent_structured_metadata() -> None:
    table = save_run_summary(
        [
            make_result(accuracy=0.7, metadata={"classes": ["no", "yes"]}),
            make_result(accuracy=0.9, metadata={"classes": ["no", "yes"]}),
        ]
    )

    aggregate = aggregate_run_summary(table)

    assert aggregate.iloc[0]["metadata.classes"] == ["no", "yes"]


def test_csv_summary_preserves_first_seen_column_order(tmp_path) -> None:
    path = tmp_path / "summary.csv"
    table = save_run_summary([make_result()], csv_path=path)

    with path.open(newline="", encoding="utf-8") as file:
        header = file.readline().strip().split(",")

    assert header[:5] == ["run_key", "case_name", "dataset_name", "task_name", "task_type"]
    assert header == list(table.columns)
