from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)

from rtml.core.datasets import Dataset
from rtml.core.resampling import Resample, ResamplingPlan, ResamplingSpec, ResamplingStrategy
from rtml.core.tasks import TaskSpec


def build_single_instance_resampling_plan(
    *,
    dataset: Dataset,
    task: TaskSpec,
    spec: ResamplingSpec,
) -> ResamplingPlan:
    """Materialize row-position splits and optional saved validation rows."""
    task.validate_columns(dataset)
    row_indices = np.arange(len(dataset))
    resamples: list[Resample] = []

    if spec.strategy == ResamplingStrategy.HOLDOUT:
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        resamples.append(_resample("repeat_00", train_idx, test_idx))
    elif spec.strategy == ResamplingStrategy.STRATIFIED_HOLDOUT:
        stratify = _column_values(dataset, spec.stratify, role="stratify")
        train_idx, test_idx = train_test_split(
            row_indices,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
            stratify=stratify,
        )
        resamples.append(_resample("repeat_00", train_idx, test_idx))
    elif spec.strategy == ResamplingStrategy.KFOLD:
        splitter = KFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset.data)):
            resamples.append(_resample(f"fold_{fold:02d}", train_idx, test_idx, fold=fold))
    elif spec.strategy == ResamplingStrategy.STRATIFIED_KFOLD:
        stratify = _column_values(dataset, spec.stratify, role="stratify")
        splitter = StratifiedKFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset.data, stratify)):
            resamples.append(_resample(f"fold_{fold:02d}", train_idx, test_idx, fold=fold))
    elif spec.strategy == ResamplingStrategy.GROUP_KFOLD:
        groups = _group_values(dataset, spec.groups)
        splitter = GroupKFold(n_splits=spec.n_folds)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset.data, groups=groups)):
            resamples.append(_resample(f"fold_{fold:02d}", train_idx, test_idx, fold=fold))
    elif spec.strategy == ResamplingStrategy.BOOTSTRAP:
        if len(row_indices) < 2:
            raise ValueError("bootstrap requires at least two dataset rows")
        rng = np.random.default_rng(spec.seed)
        for sample in range(spec.n_samples):
            for _ in range(100):
                train_idx = rng.choice(row_indices, size=len(row_indices), replace=True)
                test_idx = np.setdiff1d(row_indices, np.unique(train_idx))
                if len(test_idx) > 0:
                    break
            else:
                raise RuntimeError("could not draw a bootstrap sample with out-of-bag rows")
            resamples.append(_resample(f"sample_{sample:02d}", train_idx, test_idx, sample=sample))
    else:
        raise NotImplementedError(
            f"single-instance resampling does not support {spec.strategy.value}"
        )

    if spec.valid_size is not None:
        resamples = [
            _with_validation_split(resample, dataset=dataset, spec=spec) for resample in resamples
        ]

    return ResamplingPlan(
        dataset_name=dataset.name,
        task_name=task.name,
        spec=spec,
        resamples=resamples,
        metadata={"paradigm": "single_instance"},
    )


def _column_values(dataset: Dataset, column: str | None, *, role: str) -> np.ndarray:
    if column is None:
        raise ValueError(f"{role} column must be configured")
    dataset.require_columns([column])
    return dataset.data[column].to_numpy()


def _group_values(dataset: Dataset, columns: list[str]) -> np.ndarray:
    dataset.require_columns(columns)
    if len(columns) == 1:
        return dataset.data[columns[0]].to_numpy()

    values = np.empty(len(dataset), dtype=object)
    values[:] = list(dataset.data.loc[:, columns].itertuples(index=False, name=None))
    return values


def _with_validation_split(
    resample: Resample,
    *,
    dataset: Dataset,
    spec: ResamplingSpec,
) -> Resample:
    if spec.strategy == ResamplingStrategy.GROUP_KFOLD:
        groups = _group_values(dataset, spec.groups)
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=spec.valid_size,
            random_state=spec.seed,
        )
        train_pos, valid_pos = next(
            splitter.split(resample.train_idx, groups=groups[resample.train_idx])
        )
        train_idx = resample.train_idx[train_pos]
        valid_idx = resample.train_idx[valid_pos]
    else:
        stratify = (
            None
            if spec.stratify is None
            else _column_values(dataset, spec.stratify, role="stratify")[resample.train_idx]
        )
        shuffle = spec.shuffle or stratify is not None
        train_idx, valid_idx = train_test_split(
            resample.train_idx,
            test_size=spec.valid_size,
            shuffle=shuffle,
            random_state=spec.seed if shuffle else None,
            stratify=stratify,
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
    fold: int | None = None,
    sample: int | None = None,
) -> Resample:
    metadata: dict[str, Any] = {"paradigm": "single_instance", "unit": "row"}
    if fold is not None:
        metadata["fold"] = fold
    if sample is not None:
        metadata["sample"] = sample
    return Resample(id=id, train_idx=train_idx, test_idx=test_idx, metadata=metadata)
