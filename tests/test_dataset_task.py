import pandas as pd
import pytest

from rtml.datasets import Dataset, FeatureInfo, FeatureKind, FeatureSchema
from rtml.datasets.sklearn_loaders import (
    load_breast_cancer_dataset,
    load_diabetes_dataset,
    load_iris_dataset,
    load_sklearn_dataset,
)
from rtml.tasks import MetricSpec, TaskSpec, TaskType


def make_dataset() -> Dataset:
    data = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3"],
            "age": [32, 45, 28],
            "segment": ["a", "b", "a"],
            "income": [0, 1, 0],
            "weight": [1.0, 0.5, 1.2],
        }
    )
    schema = FeatureSchema.infer(
        data,
        id_columns=["row_id"],
        binary_columns=["income"],
        weight_columns=["weight"],
    )
    return Dataset(name="toy", data=data, schema=schema, row_id="row_id")


def make_tagged_dataset() -> Dataset:
    data = pd.DataFrame(
        {
            "x_num": [None, 1.0, 10.0],
            "x_sparse": [0.0, 0.0, 5.0],
            "x_cat": ["a", "b", "a"],
            "target": [1, 0, 1],
        }
    )
    schema = FeatureSchema(
        features={
            "x_num": FeatureInfo(
                name="x_num",
                kind=FeatureKind.NUMERIC,
                tags={"skewed", "missing_values"},
            ),
            "x_sparse": FeatureInfo(
                name="x_sparse",
                kind=FeatureKind.NUMERIC,
                tags={"zero_inflated", "many_zeros"},
            ),
            "x_cat": FeatureInfo(
                name="x_cat",
                kind=FeatureKind.CATEGORICAL,
                tags={"high_cardinality"},
            ),
            "target": FeatureInfo(name="target", kind=FeatureKind.BINARY),
        }
    )
    return Dataset(name="tagged", data=data, schema=schema)


def test_schema_infer_and_dataset_validation() -> None:
    dataset = make_dataset()

    assert len(dataset) == 3
    assert dataset.schema.get("row_id").kind == FeatureKind.ID
    assert dataset.schema.get("age").kind == FeatureKind.NUMERIC
    assert dataset.schema.get("segment").kind == FeatureKind.CATEGORICAL
    assert dataset.schema.by_kind(FeatureKind.NUMERIC) == ["age"]


def test_feature_schema_select_supports_kind_and_tag_queries() -> None:
    dataset = make_tagged_dataset()

    assert dataset.schema.select(kinds=[FeatureKind.NUMERIC]) == ["x_num", "x_sparse"]
    assert dataset.schema.select(kinds=["numeric"], include_tags=["skewed"]) == ["x_num"]
    assert dataset.schema.select(include_tags=["zero_inflated", "many_zeros"]) == ["x_sparse"]
    assert dataset.schema.select(
        include_tags=["skewed", "zero_inflated"],
        require_all_tags=False,
    ) == ["x_num", "x_sparse"]
    assert dataset.schema.select(
        kinds=[FeatureKind.NUMERIC, FeatureKind.CATEGORICAL],
        exclude_tags=["high_cardinality"],
    ) == ["x_num", "x_sparse"]


def test_dataset_select_returns_columns_or_feature_info() -> None:
    dataset = make_tagged_dataset()

    assert dataset.select(kinds=[FeatureKind.NUMERIC], include_tags=["skewed"]) == ["x_num"]
    selected = dataset.select(include_tags=["zero_inflated"], return_features=True)
    assert list(selected) == ["x_sparse"]
    assert selected["x_sparse"].kind == FeatureKind.NUMERIC


def test_dataset_rejects_schema_column_mismatch() -> None:
    data = pd.DataFrame({"x": [1, 2]})
    schema = FeatureSchema({"y": FeatureInfo(name="y", kind=FeatureKind.NUMERIC)})

    with pytest.raises(ValueError, match="columns and schema"):
        Dataset(name="bad", data=data, schema=schema)


def test_dataset_select_rows_keeps_schema_and_metadata() -> None:
    dataset = make_dataset()
    selected = dataset.select_rows([2, 0])

    assert selected.data["row_id"].tolist() == ["r3", "r1"]
    assert selected.schema is dataset.schema


def test_task_spec_validates_roles_against_dataset() -> None:
    dataset = make_dataset()
    task = TaskSpec(
        name="income_prediction",
        task_type=TaskType.BINARY_CLASSIFICATION,
        source=["age", "segment"],
        target="income",
        sample_weight="weight",
        metrics=[MetricSpec("accuracy")],
        primary_metric="accuracy",
    )

    task.validate_columns(dataset)
    assert task.source_frame(dataset).columns.tolist() == ["age", "segment"]
    assert task.target_series(dataset).tolist() == [0, 1, 0]  # type: ignore


def test_task_spec_rejects_resampling_like_split_state() -> None:
    fields = TaskSpec.__dataclass_fields__

    assert "split" not in fields
    assert "resampling" not in fields


def test_task_spec_requires_primary_metric_to_be_configured() -> None:
    with pytest.raises(ValueError, match="primary_metric"):
        TaskSpec(
            name="bad_metric",
            task_type=TaskType.REGRESSION,
            source=["x"],
            target="y",
            metrics=[MetricSpec("mae")],
            primary_metric="rmse",
        )


def test_sklearn_loader_returns_passive_dataset_and_task() -> None:
    dataset, task = load_breast_cancer_dataset()

    assert dataset.name == "breast_cancer"
    assert "target" in dataset.columns
    assert dataset.schema.get("target").kind == FeatureKind.BINARY
    assert task.target == "target"
    assert "target" not in task.source
    task.validate_columns(dataset)


def test_iris_loader_returns_multiclass_task() -> None:
    dataset, task = load_iris_dataset()

    assert dataset.name == "iris"
    assert dataset.schema.get("sepal length (cm)").kind == FeatureKind.NUMERIC
    assert dataset.schema.get("target").kind == FeatureKind.CATEGORICAL
    assert task.task_type == TaskType.MULTICLASS_CLASSIFICATION
    assert task.primary_metric == "accuracy"
    task.validate_columns(dataset)


def test_diabetes_loader_returns_regression_task() -> None:
    dataset, task = load_diabetes_dataset()

    assert dataset.name == "diabetes"
    assert dataset.schema.get("target").kind == FeatureKind.NUMERIC
    assert task.task_type == TaskType.REGRESSION
    assert task.primary_metric == "rmse"
    task.validate_columns(dataset)


def test_generic_sklearn_loader_rejects_missing_values_during_schema_inference() -> None:
    def loader_with_missing_values(*, as_frame: bool = True):
        assert as_frame is True
        return type(
            "FakeBunch",
            (),
            {
                "data": pd.DataFrame({"x": [1.0, None], "z": [0.5, 1.5]}),
                "target": pd.Series([0, 1], name="target"),
                "target_names": ["no", "yes"],
                "feature_names": ["x", "z"],
                "DESCR": "synthetic dataset",
            },
        )()

    with pytest.raises(ValueError, match="without missing values"):
        load_sklearn_dataset(loader_with_missing_values, name="synthetic")  # type: ignore
