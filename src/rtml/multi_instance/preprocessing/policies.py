from collections.abc import Mapping
from typing import Any

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rtml.core.datasets import FeatureKind, FeatureTag
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.tasks import MultiInstanceTask


def build_preprocessor(
    dataset: MultiInstanceDataset,
    task: MultiInstanceTask,
    *,
    policy: str = "neural_default",
    options: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build a numeric instance transform for a multi-instance method."""
    if policy not in {"linear_default", "neural_default"}:
        raise ValueError(f"unsupported multi-instance preprocessing policy {policy!r}")
    if options:
        unknown = ", ".join(sorted(options))
        raise ValueError(f"unknown {policy} preprocessing options: {unknown}")

    task.validate_columns(dataset)
    unsupported = [
        column
        for column in task.instance_source
        if dataset.instance_schema.get(column).kind != FeatureKind.NUMERIC
    ]
    if unsupported:
        raise ValueError(
            f"{policy} currently supports numeric MIL instance features only: {unsupported}"
        )

    missing_columns = [
        column for column in task.instance_source if dataset.instance_table[column].isna().any()
    ]
    untagged = [
        column
        for column in missing_columns
        if FeatureTag.MISSING_VALUES not in dataset.instance_schema.get(column).tags
    ]
    if untagged:
        raise ValueError(
            "MIL instance columns contain missing values but are not tagged with "
            f"{FeatureTag.MISSING_VALUES.value!r}: {untagged}"
        )

    steps: list[tuple[str, Any]] = []
    if any(
        FeatureTag.MISSING_VALUES in dataset.instance_schema.get(column).tags
        for column in task.instance_source
    ):
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
