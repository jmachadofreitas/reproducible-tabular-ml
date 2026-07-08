from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, overload

import pandas as pd


class FeatureKind(str, Enum):
    ID = "id"
    GROUP = "group"
    TIMESTAMP = "timestamp"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    TEXT = "text"
    EMBEDDING = "embedding"
    WEIGHT = "weight"
    MASK = "mask"
    UNKNOWN = "unknown"


@dataclass
class FeatureInfo:
    name: str
    kind: FeatureKind
    dtype: str | None = None
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature name must be non-empty")

        self.kind = FeatureKind(self.kind)
        self.tags = set(self.tags or ())
        self.metadata = dict(self.metadata or {})


@dataclass
class FeatureSchema:
    features: dict[str, FeatureInfo]

    def __post_init__(self) -> None:
        features = dict(self.features)
        for name, info in features.items():
            if name != info.name:
                raise ValueError(f"schema key {name!r} does not match feature name {info.name!r}")
        self.features = features

    def __contains__(self, name: str) -> bool:
        return name in self.features

    def __iter__(self):
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    @property
    def names(self) -> list[str]:
        return list(self.features)

    def get(self, name: str) -> FeatureInfo:
        try:
            return self.features[name]
        except KeyError as exc:
            raise KeyError(f"unknown feature {name!r}") from exc

    def require(self, columns: Iterable[str], *, role: str = "columns") -> None:
        missing = [column for column in columns if column not in self.features]
        if missing:
            raise ValueError(f"{role} not present in schema: {missing}")

    def by_kind(self, *kinds: FeatureKind | str) -> list[str]:
        wanted = {FeatureKind(kind) for kind in kinds}
        return [name for name, info in self.features.items() if info.kind in wanted]

    def tagged(self, tag: str) -> list[str]:
        return [name for name, info in self.features.items() if tag in info.tags]

    def select(
        self,
        *,
        kinds: Iterable[FeatureKind | str] | None = None,
        include_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
        require_all_tags: bool = True,
    ) -> list[str]:
        selected = list(self.features.items())
        if kinds is not None:
            wanted_kinds = {FeatureKind(kind) for kind in kinds}
            selected = [(name, info) for name, info in selected if info.kind in wanted_kinds]

        include_tags_set = set(include_tags)
        exclude_tags_set = set(exclude_tags)

        # This supports preprocessing policies such as "numeric + skewed" or
        # "categorical without high_cardinality".
        if include_tags_set:
            if require_all_tags:
                selected = [
                    (name, info) for name, info in selected if include_tags_set.issubset(info.tags)
                ]
            else:
                selected = [
                    (name, info)
                    for name, info in selected
                    if info.tags.intersection(include_tags_set)
                ]

        if exclude_tags_set:
            selected = [
                (name, info)
                for name, info in selected
                if not info.tags.intersection(exclude_tags_set)
            ]

        return [name for name, _ in selected]

    @classmethod
    def infer(
        cls,
        data: pd.DataFrame,
        *,
        id_columns: Iterable[str] = (),
        categorical_columns: Iterable[str] = (),
        binary_columns: Iterable[str] = (),
        group_columns: Iterable[str] = (),
        weight_columns: Iterable[str] = (),
        unknown_columns: Iterable[str] = (),
    ) -> FeatureSchema:
        id_set = set(id_columns)
        categorical_set = set(categorical_columns)
        binary_set = set(binary_columns)
        group_set = set(group_columns)
        weight_set = set(weight_columns)
        unknown_set = set(unknown_columns)
        features: dict[str, FeatureInfo] = {}

        for raw_column in data.columns:
            column = str(raw_column)
            series = data[raw_column]
            if column in id_set:
                kind = FeatureKind.ID
            elif column in group_set:
                kind = FeatureKind.GROUP
            elif column in weight_set:
                kind = FeatureKind.WEIGHT
            elif column in binary_set:
                kind = FeatureKind.BINARY
            elif column in categorical_set:
                kind = FeatureKind.CATEGORICAL
            elif column in unknown_set:
                kind = FeatureKind.UNKNOWN
            elif pd.api.types.is_bool_dtype(series):
                kind = FeatureKind.BINARY
            elif pd.api.types.is_numeric_dtype(series):
                kind = FeatureKind.NUMERIC
            elif pd.api.types.is_datetime64_any_dtype(series):
                kind = FeatureKind.TIMESTAMP
            else:
                kind = FeatureKind.CATEGORICAL

            features[column] = FeatureInfo(name=column, kind=kind, dtype=str(series.dtype))

        return cls(features=features)


@dataclass
class Dataset:
    name: str
    data: pd.DataFrame
    schema: FeatureSchema
    row_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _column_set: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        self.metadata = dict(self.metadata or {})

        if not self.data.columns.is_unique:
            duplicates = self.data.columns[self.data.columns.duplicated()].tolist()
            raise ValueError(f"data contains duplicate columns: {duplicates}")

        data_columns = [str(column) for column in self.data.columns]
        schema_columns = self.schema.names
        missing_from_schema = [column for column in data_columns if column not in self.schema]
        extra_in_schema = [column for column in schema_columns if column not in data_columns]
        if missing_from_schema or extra_in_schema:
            raise ValueError(
                "dataset columns and schema features must match "
                f"(missing_from_schema={missing_from_schema}, extra_in_schema={extra_in_schema})"
            )

        if self.row_id is not None:
            if self.row_id not in data_columns:
                raise ValueError(f"row_id column {self.row_id!r} is not present in data")
            row_feature = self.schema.get(self.row_id)
            if row_feature.kind != FeatureKind.ID:
                raise ValueError(f"row_id column {self.row_id!r} must have FeatureKind.ID")
            if self.data[self.row_id].duplicated().any():
                raise ValueError(f"row_id column {self.row_id!r} contains duplicate values")

        self._column_set = {str(column) for column in self.data.columns}

    def __len__(self) -> int:
        return len(self.data)

    @property
    def columns(self) -> set[str]:
        return self._column_set

    def require_columns(self, columns: Iterable[str]) -> None:
        """Ensure that the requested columns exist in the dataset."""
        missing = [column for column in columns if column not in self._column_set]
        if missing:
            raise ValueError(f"columns not present in dataset {self.name!r}: {missing}")

    def select_rows(self, rows: Sequence[int] | slice) -> Dataset:
        if isinstance(rows, slice):
            selected = self.data.iloc[rows]
        else:
            selected = self.data.iloc[list(rows)]
        return Dataset(
            name=self.name,
            data=selected,
            schema=self.schema,
            row_id=self.row_id,
            metadata=self.metadata,
        )

    @overload
    def select(
        self,
        *,
        kinds: Iterable[FeatureKind | str] | None = None,
        include_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
        require_all_tags: bool = True,
        return_features: Literal[False] = False,
    ) -> list[str]: ...

    @overload
    def select(
        self,
        *,
        kinds: Iterable[FeatureKind | str] | None = None,
        include_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
        require_all_tags: bool = True,
        return_features: Literal[True],
    ) -> dict[str, FeatureInfo]: ...

    def select(
        self,
        *,
        kinds: Iterable[FeatureKind | str] | None = None,
        include_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
        require_all_tags: bool = True,
        return_features: bool = False,
    ) -> list[str] | dict[str, FeatureInfo]:
        """Select dataset features by semantic kind and/or tags.

        By default, returns column names suitable for preprocessing pipelines or method input.

        If ``return_features=True``, returns the matching ``FeatureInfo`` objects
        keyed by column name, allowing preprocessing policies to inspect dtype,
        tags, and metadata.
        """
        columns = self.schema.select(
            kinds=kinds,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            require_all_tags=require_all_tags,
        )

        if return_features:
            return {column: self.schema.get(column) for column in columns}

        return columns
