import pandas as pd
import pytest

from rtml.core.datasets import Dataset, FeatureInfo, FeatureKind, FeatureSchema, FeatureTag
from rtml.single_instance.preprocessing import build_preprocessor
from rtml.core.tasks import MetricSpec, TaskSpec, TaskType


def make_mixed_dataset() -> tuple[Dataset, TaskSpec]:
    data = pd.DataFrame(
        {
            "num": [1.0, None, 3.0, 100.0],
            "cat": ["a", "b", "a", "unseen"],
            "target": [0, 1, 0, 1],
        }
    )
    schema = FeatureSchema(
        features={
            "num": FeatureInfo(
                name="num",
                kind=FeatureKind.NUMERIC,
                tags={FeatureTag.MISSING_VALUES},
            ),
            "cat": FeatureInfo(name="cat", kind=FeatureKind.CATEGORICAL),
            "target": FeatureInfo(name="target", kind=FeatureKind.BINARY),
        }
    )
    dataset = Dataset(name="mixed", data=data, schema=schema)
    task = TaskSpec(
        name="mixed_binary",
        task_type=TaskType.BINARY_CLASSIFICATION,
        source=["num", "cat"],
        target="target",
        metrics=[MetricSpec("accuracy")],
        primary_metric="accuracy",
    )
    return dataset, task


def make_tagged_policy_dataset() -> tuple[Dataset, TaskSpec]:
    data = pd.DataFrame(
        {
            "regular_num": [1.0, 2.0, 3.0, 4.0],
            "missing_num": [1.0, None, 3.0, 4.0],
            "skewed_num": [0.0, 1.0, 9.0, 99.0],
            "zero_num": [0.0, 0.0, 5.0, 0.0],
            "regular_cat": ["a", "b", "a", "c"],
            "wide_cat": ["x1", "x2", "x3", "x4"],
            "target": [0, 1, 0, 1],
        }
    )
    schema = FeatureSchema(
        features={
            "regular_num": FeatureInfo(name="regular_num", kind=FeatureKind.NUMERIC),
            "missing_num": FeatureInfo(
                name="missing_num",
                kind=FeatureKind.NUMERIC,
                tags={FeatureTag.MISSING_VALUES},
            ),
            "skewed_num": FeatureInfo(
                name="skewed_num",
                kind=FeatureKind.NUMERIC,
                tags={FeatureTag.SKEWED},
            ),
            "zero_num": FeatureInfo(
                name="zero_num",
                kind=FeatureKind.NUMERIC,
                tags={FeatureTag.ZERO_INFLATED},
            ),
            "regular_cat": FeatureInfo(name="regular_cat", kind=FeatureKind.CATEGORICAL),
            "wide_cat": FeatureInfo(
                name="wide_cat",
                kind=FeatureKind.CATEGORICAL,
                tags={FeatureTag.HIGH_CARDINALITY},
            ),
            "target": FeatureInfo(name="target", kind=FeatureKind.BINARY),
        }
    )
    dataset = Dataset(name="tagged_policy", data=data, schema=schema)
    task = TaskSpec(
        name="tagged_policy_binary",
        task_type=TaskType.BINARY_CLASSIFICATION,
        source=[
            "regular_num",
            "missing_num",
            "skewed_num",
            "zero_num",
            "regular_cat",
            "wide_cat",
        ],
        target="target",
        metrics=[MetricSpec("accuracy")],
        primary_metric="accuracy",
    )
    return dataset, task


def test_linear_policy_fits_categories_only_from_training_rows() -> None:
    dataset, task = make_mixed_dataset()
    preprocessor = build_preprocessor(dataset=dataset, task=task, policy="linear_default")

    preprocessor.fit(task.source_frame(dataset).iloc[[0, 1, 2]])
    transformed = preprocessor.transform(task.source_frame(dataset).iloc[[3]])

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["encoder"]

    assert encoder.categories_[0].tolist() == ["a", "b"]
    assert transformed.shape == (1, 3)


def test_linear_policy_uses_feature_tags_for_transform_branches() -> None:
    dataset, task = make_tagged_policy_dataset()
    preprocessor = build_preprocessor(dataset=dataset, task=task, policy="linear_default")

    preprocessor.fit(task.source_frame(dataset))

    assert preprocessor.transformers_[0][0] == "numeric"
    assert preprocessor.transformers_[0][2] == ["regular_num"]
    assert "imputer" not in preprocessor.named_transformers_["numeric"].named_steps
    assert preprocessor.transformers_[1][0] == "numeric_missing"
    assert preprocessor.transformers_[1][2] == ["missing_num"]
    assert "imputer" in preprocessor.named_transformers_["numeric_missing"].named_steps
    assert preprocessor.transformers_[2][0] == "numeric_skewed"
    assert preprocessor.transformers_[2][2] == ["skewed_num"]
    assert (
        preprocessor.named_transformers_["numeric_skewed"].named_steps["power"].__class__.__name__
        == "PowerTransformer"
    )
    assert preprocessor.transformers_[3][0] == "numeric_zero_inflated"
    assert preprocessor.transformers_[3][2] == ["zero_num"]
    assert (
        preprocessor.named_transformers_["numeric_zero_inflated"]
        .named_steps["scaler"]
        .__class__.__name__
        == "MaxAbsScaler"
    )
    assert preprocessor.transformers_[4][0] == "categorical"
    assert preprocessor.transformers_[4][2] == ["regular_cat"]
    assert preprocessor.transformers_[5][0] == "categorical_high_cardinality"
    assert preprocessor.transformers_[5][2] == ["wide_cat"]
    assert (
        preprocessor.named_transformers_["categorical_high_cardinality"]
        .named_steps["encoder"]
        .__class__.__name__
        == "OneHotEncoder"
    )
    assert (
        preprocessor.named_transformers_["categorical_high_cardinality"]
        .named_steps["encoder"]
        .handle_unknown
        == "infrequent_if_exist"
    )


def test_tree_policy_keeps_high_cardinality_categorical_branch_separate() -> None:
    dataset, task = make_tagged_policy_dataset()
    preprocessor = build_preprocessor(dataset=dataset, task=task, policy="tree_default")

    preprocessor.fit(task.source_frame(dataset))

    assert preprocessor.transformers_[0][0] == "numeric"
    assert "imputer" not in preprocessor.named_transformers_["numeric"].named_steps
    assert preprocessor.transformers_[1][0] == "numeric_missing"
    assert preprocessor.transformers_[1][2] == ["missing_num"]
    assert "imputer" in preprocessor.named_transformers_["numeric_missing"].named_steps
    assert preprocessor.transformers_[2][0] == "categorical"
    assert preprocessor.transformers_[2][2] == ["regular_cat"]
    assert preprocessor.transformers_[3][0] == "categorical_high_cardinality"
    assert preprocessor.transformers_[3][2] == ["wide_cat"]


def test_policy_rejects_untagged_missing_values() -> None:
    dataset, task = make_mixed_dataset()
    dataset.schema.get("num").tags.clear()

    with pytest.raises(ValueError, match="missing_values"):
        build_preprocessor(dataset=dataset, task=task, policy="linear_default")


def test_unknown_preprocessing_policy_reports_known_names() -> None:
    dataset, task = make_mixed_dataset()

    with pytest.raises(KeyError, match="linear_default"):
        build_preprocessor(dataset=dataset, task=task, policy="does_not_exist")
