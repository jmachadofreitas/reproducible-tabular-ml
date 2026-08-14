from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class ResamplingStrategy(str, Enum):
    HOLDOUT = "holdout"
    STRATIFIED_HOLDOUT = "stratified_holdout"
    REPEATED_HOLDOUT = "repeated_holdout"
    REPEATED_STRATIFIED_HOLDOUT = "repeated_stratified_holdout"
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    GROUP_KFOLD = "group_kfold"
    BOOTSTRAP = "bootstrap"
    # OpenML exposes the saved split indices, but the strategy may not be there.
    UNKNOWN_OPENML_TASK = "unknown_openml_task"


@dataclass
class Resample:
    """Materialized train, optional validation, and test positions for one run.

    Methods may use ``valid_idx`` for training control. Methods without a
    validation phase should fit on both ``train_idx`` and ``valid_idx`` while
    keeping ``test_idx`` untouched.
    """

    id: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    valid_idx: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.train_idx = _index_array(self.train_idx, "train_idx")
        self.test_idx = _index_array(self.test_idx, "test_idx")
        self.valid_idx = (
            None if self.valid_idx is None else _index_array(self.valid_idx, "valid_idx")
        )
        partitions = {"train_idx": self.train_idx, "test_idx": self.test_idx}
        if self.valid_idx is not None:
            partitions["valid_idx"] = self.valid_idx
        names = list(partitions)
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                if np.intersect1d(partitions[left_name], partitions[right_name]).size:
                    raise ValueError(f"{left_name} and {right_name} must not overlap")
        self.metadata = dict(self.metadata or {})


@dataclass
class ResamplingSpec:
    name: str
    strategy: ResamplingStrategy
    n_repeats: int = 1
    n_folds: int = 1
    n_samples: int = 1
    test_size: float | None = None
    valid_size: float | None = None
    shuffle: bool = False
    seed: int | None = None
    stratify: str | None = None
    groups: list[str] = field(default_factory=list)
    replacement: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("resampling name must be non-empty")
        self.strategy = ResamplingStrategy(self.strategy)
        if self.strategy == ResamplingStrategy.STRATIFIED_HOLDOUT:
            # sklearn stratified holdout requires shuffling. Keep the recorded
            # specification aligned with the split that will be materialized.
            self.shuffle = True
        self.groups = list(self.groups)
        self.metadata = dict(self.metadata or {})
        self._validate()

    def _validate(self) -> None:
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
        if self.n_folds < 1:
            raise ValueError("n_folds must be at least 1")
        if self.n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        for field_name, value in (("test_size", self.test_size), ("valid_size", self.valid_size)):
            if value is not None and not 0.0 < value < 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")

        if (
            self.strategy
            in {
                ResamplingStrategy.HOLDOUT,
                ResamplingStrategy.STRATIFIED_HOLDOUT,
                ResamplingStrategy.REPEATED_HOLDOUT,
                ResamplingStrategy.REPEATED_STRATIFIED_HOLDOUT,
            }
            and self.test_size is None
        ):
            raise ValueError(f"{self.strategy.value} requires test_size")

        if (
            self.strategy
            in {
                ResamplingStrategy.KFOLD,
                ResamplingStrategy.STRATIFIED_KFOLD,
                ResamplingStrategy.GROUP_KFOLD,
            }
            and self.n_folds < 2
        ):
            raise ValueError(f"{self.strategy.value} requires n_folds >= 2")

        if self.strategy == ResamplingStrategy.BOOTSTRAP and not self.replacement:
            raise ValueError("bootstrap requires sampling with replacement")
        if self.strategy == ResamplingStrategy.BOOTSTRAP and self.valid_size is not None:
            raise ValueError("bootstrap does not support a validation split")

        if (
            self.strategy
            in {
                ResamplingStrategy.STRATIFIED_HOLDOUT,
                ResamplingStrategy.REPEATED_STRATIFIED_HOLDOUT,
                ResamplingStrategy.STRATIFIED_KFOLD,
            }
            and self.stratify is None
        ):
            raise ValueError(f"{self.strategy.value} requires stratify")

        if self.strategy == ResamplingStrategy.GROUP_KFOLD and not self.groups:
            raise ValueError("group_kfold requires groups")


@dataclass
class ResamplingPlan:
    dataset_name: str
    task_name: str
    spec: ResamplingSpec
    resamples: list[Resample]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resamples = list(self.resamples)
        self.metadata = dict(self.metadata or {})

        resample_ids = [resample.id for resample in self.resamples]
        if len(resample_ids) != len(set(resample_ids)):
            raise ValueError(f"resample ids must be unique: {resample_ids}")

    def get_resample(self, resample_id: str | None = None) -> Resample:
        """Return one materialized resample by id, or the first one by default."""
        if not self.resamples:
            raise ValueError(
                f"resampling plan for {self.dataset_name!r}/{self.task_name!r} has no resamples"
            )
        if resample_id is None:
            return self.resamples[0]
        for resample in self.resamples:
            if resample.id == resample_id:
                return resample
        raise ValueError(f"unknown resample id {resample_id!r}")


def _index_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size and array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer positions")
    return array.astype(int, copy=False)
