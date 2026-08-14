from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)


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


def materialize_resamples(
    *,
    n_items: int,
    spec: ResamplingSpec,
    unit: str,
    metadata: dict[str, Any],
    stratify_values: np.ndarray | None = None,
    group_values: np.ndarray | None = None,
) -> list[Resample]:
    """Materialize positional splits after a paradigm resolves auxiliary columns."""
    positions = np.arange(n_items)
    resamples: list[Resample] = []

    if spec.strategy == ResamplingStrategy.HOLDOUT:
        train_idx, test_idx = train_test_split(
            positions,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        resamples.append(_resample("repeat_00", train_idx, test_idx, metadata=metadata))
    elif spec.strategy == ResamplingStrategy.STRATIFIED_HOLDOUT:
        if stratify_values is None:
            raise ValueError("stratified resampling requires stratify values")
        train_idx, test_idx = train_test_split(
            positions,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed,
            stratify=stratify_values,
        )
        resamples.append(_resample("repeat_00", train_idx, test_idx, metadata=metadata))
    elif spec.strategy == ResamplingStrategy.KFOLD:
        splitter = KFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(positions)):
            resamples.append(
                _resample(
                    f"fold_{fold:02d}",
                    train_idx,
                    test_idx,
                    metadata=metadata,
                    fold=fold,
                )
            )
    elif spec.strategy == ResamplingStrategy.STRATIFIED_KFOLD:
        if stratify_values is None:
            raise ValueError("stratified resampling requires stratify values")
        splitter = StratifiedKFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(positions, stratify_values)):
            resamples.append(
                _resample(
                    f"fold_{fold:02d}",
                    train_idx,
                    test_idx,
                    metadata=metadata,
                    fold=fold,
                )
            )
    elif spec.strategy == ResamplingStrategy.GROUP_KFOLD:
        if group_values is None:
            raise ValueError("grouped resampling requires group values")
        splitter = GroupKFold(n_splits=spec.n_folds)
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(positions, groups=group_values)
        ):
            resamples.append(
                _resample(
                    f"fold_{fold:02d}",
                    train_idx,
                    test_idx,
                    metadata=metadata,
                    fold=fold,
                )
            )
    elif spec.strategy == ResamplingStrategy.BOOTSTRAP:
        if n_items < 2:
            raise ValueError(f"bootstrap requires at least two {unit}s")
        rng = np.random.default_rng(spec.seed)
        for sample in range(spec.n_samples):
            for _ in range(100):
                train_idx = rng.choice(positions, size=n_items, replace=True)
                test_idx = np.setdiff1d(positions, np.unique(train_idx))
                if len(test_idx) > 0:
                    break
            else:
                raise RuntimeError(f"could not draw a bootstrap sample with out-of-bag {unit}s")
            resamples.append(
                _resample(
                    f"sample_{sample:02d}",
                    train_idx,
                    test_idx,
                    metadata=metadata,
                    sample=sample,
                )
            )
    else:
        raise NotImplementedError(f"resampling does not support {spec.strategy.value}")

    if spec.valid_size is not None:
        resamples = [
            _with_validation_split(
                resample,
                spec=spec,
                stratify_values=stratify_values,
                group_values=group_values,
            )
            for resample in resamples
        ]
    return resamples


def _with_validation_split(
    resample: Resample,
    *,
    spec: ResamplingSpec,
    stratify_values: np.ndarray | None,
    group_values: np.ndarray | None,
) -> Resample:
    if spec.strategy == ResamplingStrategy.GROUP_KFOLD:
        if group_values is None:
            raise ValueError("grouped validation requires group values")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=spec.valid_size,
            random_state=spec.seed,
        )
        train_pos, valid_pos = next(
            splitter.split(resample.train_idx, groups=group_values[resample.train_idx])
        )
        train_idx = resample.train_idx[train_pos]
        valid_idx = resample.train_idx[valid_pos]
    else:
        train_stratify = None if stratify_values is None else stratify_values[resample.train_idx]
        shuffle = spec.shuffle or train_stratify is not None
        train_idx, valid_idx = train_test_split(
            resample.train_idx,
            test_size=spec.valid_size,
            shuffle=shuffle,
            random_state=spec.seed if shuffle else None,
            stratify=train_stratify,
        )

    return Resample(
        id=resample.id,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=resample.test_idx,
        metadata={**resample.metadata, "valid_size": spec.valid_size},
    )


def _resample(
    id: str,
    train_idx: Any,
    test_idx: Any,
    *,
    metadata: dict[str, Any],
    fold: int | None = None,
    sample: int | None = None,
) -> Resample:
    resample_metadata = dict(metadata)
    if fold is not None:
        resample_metadata["fold"] = fold
    if sample is not None:
        resample_metadata["sample"] = sample
    return Resample(
        id=id,
        train_idx=train_idx,
        test_idx=test_idx,
        metadata=resample_metadata,
    )


def _index_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size and array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer positions")
    return array.astype(int, copy=False)
