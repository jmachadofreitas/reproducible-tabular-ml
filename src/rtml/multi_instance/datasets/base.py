from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from rtml.core.datasets import FeatureInfo, FeatureKind, FeatureSchema, FeatureTagLike

from rtml.core.fingerprints import fingerprint_frame

IndexArray: TypeAlias = npt.NDArray[np.integer[Any]]


@dataclass
class MultiInstanceDataset:
    """In-memory multiple-instance dataset backed by two aligned tables.

    Attributes:
        name: Dataset identifier.
        bag_table: One row per bag, including bag-level targets and metadata.
        instance_table: Flattened instance rows stored in contiguous bag order.
        bag_schema: Schema for bag-level columns.
        instance_schema: Schema for instance-level columns.
        bag_offsets: Row offsets into ``instance_table``. The slice
            ``bag_offsets[i]:bag_offsets[i + 1]`` belongs to bag position ``i``.
        bag_id_column: Optional unique bag identifier column in ``bag_table``.
        instance_id_column: Optional unique instance identifier column in
            ``instance_table``.
        metadata: Free-form dataset metadata.
    """

    name: str
    bag_table: pd.DataFrame
    instance_table: pd.DataFrame
    bag_schema: FeatureSchema
    instance_schema: FeatureSchema
    bag_offsets: np.ndarray
    bag_id_column: str | None = None
    instance_id_column: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _bag_columns: set[str] = field(init=False, repr=False)
    _instance_columns: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        if not isinstance(self.bag_table, pd.DataFrame):
            raise TypeError("bag_table must be a pandas DataFrame")
        if not isinstance(self.instance_table, pd.DataFrame):
            raise TypeError("instance_table must be a pandas DataFrame")

        self.bag_offsets = np.asarray(self.bag_offsets, dtype=int)
        self.metadata = dict(self.metadata or {})

        self._validate_frame_schema(
            frame=self.bag_table,
            schema=self.bag_schema,
            frame_name="bag_table",
        )
        self._validate_frame_schema(
            frame=self.instance_table,
            schema=self.instance_schema,
            frame_name="instance_table",
        )
        self._validate_bag_offsets()

        self._bag_columns = {str(column) for column in self.bag_table.columns}
        self._instance_columns = {str(column) for column in self.instance_table.columns}
        self._validate_ids()

    def __len__(self) -> int:
        return len(self.bag_table)

    @property
    def n_bags(self) -> int:
        return len(self.bag_table)

    @property
    def n_instances(self) -> int:
        return len(self.instance_table)

    @property
    def bag_columns(self) -> set[str]:
        return self._bag_columns

    @property
    def instance_columns(self) -> set[str]:
        return self._instance_columns

    def bag_size(self, bag_position: int) -> int:
        self._require_bag_position(bag_position)
        return int(self.bag_offsets[bag_position + 1] - self.bag_offsets[bag_position])

    def bag_instances(self, bag_position: int) -> pd.DataFrame:
        self._require_bag_position(bag_position)
        start = self.bag_offsets[bag_position]
        stop = self.bag_offsets[bag_position + 1]
        return self.instance_table.iloc[start:stop]

    def sample_ids_for(self, indices: Sequence[int] | IndexArray) -> np.ndarray:
        """Return stable bag ids for selected positional bag indices."""
        positions = list(indices)
        if self.bag_id_column is not None:
            return self.bag_table.iloc[positions][self.bag_id_column].to_numpy()
        return np.asarray(indices)

    def subgroup_values(
        self,
        columns: Iterable[str],
        indices: Sequence[int] | IndexArray,
    ) -> dict[str, np.ndarray]:
        """Return bag-level subgroup columns aligned with selected bags."""
        selected_columns = list(columns)
        self.require_bag_columns(selected_columns)
        data = self.bag_table.iloc[list(indices)]
        return {
            column: data[column].astype("string").fillna("<NA>").to_numpy(dtype=str)
            for column in selected_columns
        }

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the dataset structure and provenance used for its fingerprint."""
        source_identity = self.metadata.get("source_identity")
        payload = {
            "name": self.name,
            "bag_table": {
                "shape": tuple(self.bag_table.shape),
                "columns": [str(column) for column in self.bag_table.columns],
                "schema": self.bag_schema,
                "id_column": self.bag_id_column,
            },
            "instance_table": {
                "shape": tuple(self.instance_table.shape),
                "columns": [str(column) for column in self.instance_table.columns],
                "schema": self.instance_schema,
                "id_column": self.instance_id_column,
            },
            "bag_offsets": self.bag_offsets,
        }
        if source_identity is not None:
            payload["source_identity"] = source_identity
        else:
            payload["bag_content"] = fingerprint_frame(self.bag_table)
            payload["instance_content"] = fingerprint_frame(self.instance_table)
            payload["metadata"] = {
                key: value for key, value in self.metadata.items() if key != "fingerprint"
            }
        return payload

    def require_bag_columns(self, columns: Iterable[str]) -> None:
        missing = [column for column in columns if column not in self._bag_columns]
        if missing:
            raise ValueError(f"bag columns not present in dataset {self.name!r}: {missing}")

    def require_instance_columns(self, columns: Iterable[str]) -> None:
        missing = [column for column in columns if column not in self._instance_columns]
        if missing:
            raise ValueError(f"instance columns not present in dataset {self.name!r}: {missing}")

    def select_bags(
        self, bag_positions: Sequence[int] | slice | IndexArray
    ) -> MultiInstanceDataset:
        if isinstance(bag_positions, slice):
            positions = list(range(self.n_bags))[bag_positions]
        else:
            positions = [int(position) for position in bag_positions]

        instance_chunks = [self.bag_instances(position) for position in positions]
        selected_instances = (
            pd.concat(instance_chunks, axis=0, ignore_index=True)
            if instance_chunks
            else self.instance_table.iloc[0:0].copy()
        )
        selected_bags = self.bag_table.iloc[positions].reset_index(drop=True)
        bag_offsets = np.zeros(len(positions) + 1, dtype=int)
        if positions:
            bag_offsets[1:] = np.cumsum([len(chunk) for chunk in instance_chunks])

        return MultiInstanceDataset(
            name=self.name,
            bag_table=selected_bags,
            instance_table=selected_instances,
            bag_schema=self.bag_schema,
            instance_schema=self.instance_schema,
            bag_offsets=bag_offsets,
            bag_id_column=self.bag_id_column,
            instance_id_column=self.instance_id_column,
            metadata=self.metadata,
        )

    @overload
    def select_instance_features(
        self,
        *,
        kinds: Iterable[FeatureKind | str] | None = None,
        include_tags: Iterable[FeatureTagLike] = (),
        exclude_tags: Iterable[FeatureTagLike] = (),
        require_all_tags: bool = True,
        return_features: Literal[False] = False,
    ) -> list[str]: ...

    @overload
    def select_instance_features(
        self,
        *,
        kinds: Iterable[FeatureKind | str] | None = None,
        include_tags: Iterable[FeatureTagLike] = (),
        exclude_tags: Iterable[FeatureTagLike] = (),
        require_all_tags: bool = True,
        return_features: Literal[True],
    ) -> dict[str, FeatureInfo]: ...

    def select_instance_features(
        self,
        *,
        kinds: Iterable[FeatureKind | str] | None = None,
        include_tags: Iterable[FeatureTagLike] = (),
        exclude_tags: Iterable[FeatureTagLike] = (),
        require_all_tags: bool = True,
        return_features: bool = False,
    ) -> list[str] | dict[str, FeatureInfo]:
        columns = self.instance_schema.select(
            kinds=kinds,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            require_all_tags=require_all_tags,
        )
        if return_features:
            return {column: self.instance_schema.get(column) for column in columns}
        return columns

    def _validate_frame_schema(
        self,
        *,
        frame: pd.DataFrame,
        schema: FeatureSchema,
        frame_name: str,
    ) -> None:
        if not frame.columns.is_unique:
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"{frame_name} contains duplicate columns: {duplicates}")

        frame_columns = [str(column) for column in frame.columns]
        schema_columns = schema.names
        missing_from_schema = [
            column for column in frame_columns if column not in schema
        ]
        extra_in_schema = [
            column for column in schema_columns if column not in frame_columns
        ]
        if missing_from_schema or extra_in_schema:
            raise ValueError(
                f"{frame_name} columns and schema features must match "
                f"(missing_from_schema={missing_from_schema}, extra_in_schema={extra_in_schema})"
            )

    def _validate_bag_offsets(self) -> None:
        if self.bag_offsets.ndim != 1:
            raise ValueError("bag_offsets must be a one-dimensional array")
        if len(self.bag_offsets) != len(self.bag_table) + 1:
            raise ValueError("bag_offsets length must equal number of bags plus one")
        if len(self.bag_offsets) == 0 or self.bag_offsets[0] != 0:
            raise ValueError("bag_offsets must start at 0")
        if self.bag_offsets[-1] != len(self.instance_table):
            raise ValueError("last bag offset must equal number of instances")
        if np.any(np.diff(self.bag_offsets) < 0):
            raise ValueError("bag_offsets must be monotonically increasing")

    def _validate_ids(self) -> None:
        if self.bag_id_column is not None:
            self.require_bag_columns([self.bag_id_column])
            if self.bag_schema.get(self.bag_id_column).kind != FeatureKind.ID:
                raise ValueError(
                    f"bag_id_column {self.bag_id_column!r} must have FeatureKind.ID"
                )
            if self.bag_table[self.bag_id_column].duplicated().any():
                raise ValueError(
                    f"bag_id_column {self.bag_id_column!r} contains duplicate values"
                )

        if self.instance_id_column is not None:
            self.require_instance_columns([self.instance_id_column])
            if self.instance_schema.get(self.instance_id_column).kind != FeatureKind.ID:
                raise ValueError(
                    f"instance_id_column {self.instance_id_column!r} must have FeatureKind.ID"
                )
            if self.instance_table[self.instance_id_column].duplicated().any():
                raise ValueError(
                    f"instance_id_column {self.instance_id_column!r} contains duplicate values"
                )

    def _require_bag_position(self, bag_position: int) -> None:
        if not 0 <= bag_position < self.n_bags:
            raise IndexError(f"bag position {bag_position} out of range")
