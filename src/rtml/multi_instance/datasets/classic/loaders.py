"""Adapters from parsed classic MIL tables to RTML domain objects."""

from pathlib import Path

import pandas as pd

from rtml.core.datasets import FeatureInfo, FeatureKind, FeatureSchema
from rtml.core.tasks import MetricSpec, TaskType
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.datasets.classic.constants import (
    CLASSIC_MIL_ARCHIVE_SHA256,
    CLASSIC_MIL_RELEASE,
    DEFAULT_CLASSIC_MIL_DATA_DIR,
)
from rtml.multi_instance.datasets.classic.downloader import classic_mil_arff_path
from rtml.multi_instance.datasets.classic.parser import (
    ParsedRelationalArff,
    parse_weka_relational_arff,
)
from rtml.multi_instance.tasks import MultiInstanceTask


def parse_classic_mil_arff(arff_path: str | Path) -> tuple[MultiInstanceDataset, MultiInstanceTask]:
    """Parse one WEKA relational ARFF file into RTML multi-instance objects."""
    return _build_dataset(parse_weka_relational_arff(arff_path))


def load_classic_mil_dataset(
    dataset_name: str,
    root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR,
) -> tuple[MultiInstanceDataset, MultiInstanceTask]:
    """Load one classic WEKA MIL dataset from the local cache."""
    parsed = parse_weka_relational_arff(classic_mil_arff_path(dataset_name, root))
    return _build_dataset(
        parsed,
        source_identity={
            "source": "classic_mil",
            "release": CLASSIC_MIL_RELEASE,
            "archive_sha256": CLASSIC_MIL_ARCHIVE_SHA256,
            "dataset": parsed.relation,
        },
    )


def _build_dataset(
    parsed: ParsedRelationalArff,
    *,
    source_identity: dict[str, object] | None = None,
) -> tuple[MultiInstanceDataset, MultiInstanceTask]:
    bag_schema = _build_bag_schema(
        parsed.bag_table,
        outer_attributes=parsed.outer_attributes,
        bag_id_column=parsed.bag_id_column,
        target_column=parsed.target_column,
    )
    instance_schema = _build_instance_schema(
        parsed.instance_table,
        instance_attributes=parsed.instance_attributes,
    )
    metadata = {
        "source": "classic_mil",
        "paradigm": "multi_instance",
        "relation": parsed.relation,
    }
    if source_identity is not None:
        metadata["source_identity"] = source_identity

    dataset = MultiInstanceDataset(
        name=parsed.relation,
        bag_table=parsed.bag_table,
        instance_table=parsed.instance_table,
        bag_schema=bag_schema,
        instance_schema=instance_schema,
        bag_offsets=parsed.bag_offsets,
        bag_id_column=parsed.bag_id_column,
        metadata=metadata,
    )
    task = MultiInstanceTask(
        name=parsed.relation,
        task_type=TaskType.BINARY_CLASSIFICATION,
        instance_source=dataset.select_instance_features(kinds=[FeatureKind.NUMERIC]),
        target=parsed.target_column,
        metrics=[MetricSpec(name="accuracy", greater_is_better=True)],
        primary_metric="accuracy",
        metadata={
            "source": "classic_mil",
            "paradigm": "multi_instance",
            "target_level": "bag",
        },
    )
    task.validate_columns(dataset)
    return dataset, task


def _build_bag_schema(
    bag_table: pd.DataFrame,
    *,
    outer_attributes: list[tuple[str, str]],
    bag_id_column: str,
    target_column: str,
) -> FeatureSchema:
    type_by_name = {name: type_spec for name, type_spec in outer_attributes}
    features = {}
    for column in bag_table.columns:
        kind = _feature_kind(type_by_name[column], values=bag_table[column])
        if column == bag_id_column:
            kind = FeatureKind.ID
        elif column == target_column and kind == FeatureKind.CATEGORICAL:
            kind = FeatureKind.BINARY if bag_table[column].nunique(dropna=True) == 2 else kind
        features[column] = FeatureInfo(
            name=column,
            kind=kind,
            dtype=str(bag_table[column].dtype),
            metadata={"arff_type": type_by_name[column]},
        )
    return FeatureSchema(features)


def _build_instance_schema(
    instance_table: pd.DataFrame,
    *,
    instance_attributes: list[tuple[str, str]],
) -> FeatureSchema:
    type_by_name = dict(instance_attributes)
    return FeatureSchema(
        {
            column: FeatureInfo(
                name=column,
                kind=_feature_kind(type_by_name[column], values=instance_table[column]),
                dtype=str(instance_table[column].dtype),
                metadata={"arff_type": type_by_name[column]},
            )
            for column in instance_table.columns
        }
    )


def _feature_kind(type_spec: str, *, values: pd.Series) -> FeatureKind:
    normalized_type = type_spec.strip().lower()
    if normalized_type in {"numeric", "real", "integer"}:
        return FeatureKind.NUMERIC
    if normalized_type.startswith("{") and normalized_type.endswith("}"):
        return FeatureKind.BINARY if values.nunique(dropna=True) == 2 else FeatureKind.CATEGORICAL
    if normalized_type == "string":
        return FeatureKind.TEXT
    return FeatureKind.UNKNOWN
