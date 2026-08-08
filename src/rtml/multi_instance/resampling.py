from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
)

from rtml.core.resampling import Resample, ResamplingPlan, ResamplingSpec, ResamplingStrategy
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.tasks import MultiInstanceTask


def build_multi_instance_resampling_plan(
    *,
    dataset: MultiInstanceDataset,
    task: MultiInstanceTask,
    spec: ResamplingSpec,
) -> ResamplingPlan:
    """Materialize bag-position splits, including optional saved validation bags.

    ``valid_size`` is taken from each outer training partition. Grouped
    resampling keeps groups isolated across training, validation, and test.
    """
    task.validate_columns(dataset)
    bag_indices = np.arange(dataset.n_bags)
    resamples: list[Resample] = []

    if spec.strategy == ResamplingStrategy.HOLDOUT:
        train_idx, test_idx = train_test_split(
            bag_indices,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed,
        )
        resamples.append(_resample("repeat_00", train_idx, test_idx))
    elif spec.strategy == ResamplingStrategy.STRATIFIED_HOLDOUT:
        stratify = _bag_column(dataset, spec.stratify, role="stratify")
        train_idx, test_idx = train_test_split(
            bag_indices,
            test_size=spec.test_size,
            shuffle=spec.shuffle,
            random_state=spec.seed,
            stratify=stratify,
        )
        resamples.append(_resample("repeat_00", train_idx, test_idx))
    elif spec.strategy == ResamplingStrategy.KFOLD:
        splitter = KFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_pos, test_pos) in enumerate(splitter.split(dataset.bag_table)):
            resamples.append(
                _resample(
                    f"fold_{fold:02d}",
                    bag_indices[train_pos],
                    bag_indices[test_pos],
                    fold=fold,
                )
            )
    elif spec.strategy == ResamplingStrategy.STRATIFIED_KFOLD:
        stratify = _bag_column(dataset, spec.stratify, role="stratify")
        splitter = StratifiedKFold(
            n_splits=spec.n_folds,
            shuffle=spec.shuffle,
            random_state=spec.seed if spec.shuffle else None,
        )
        for fold, (train_pos, test_pos) in enumerate(splitter.split(dataset.bag_table, stratify)):
            resamples.append(
                _resample(
                    f"fold_{fold:02d}",
                    bag_indices[train_pos],
                    bag_indices[test_pos],
                    fold=fold,
                )
            )
    elif spec.strategy == ResamplingStrategy.GROUP_KFOLD:
        groups = _group_values(dataset, spec.groups)
        splitter = GroupKFold(n_splits=spec.n_folds)
        for fold, (train_pos, test_pos) in enumerate(
            splitter.split(dataset.bag_table, groups=groups)
        ):
            resamples.append(
                _resample(
                    f"fold_{fold:02d}",
                    bag_indices[train_pos],
                    bag_indices[test_pos],
                    fold=fold,
                )
            )
    elif spec.strategy == ResamplingStrategy.BOOTSTRAP:
        if dataset.n_bags < 2:
            raise ValueError("bootstrap requires at least two bags")
        rng = np.random.default_rng(spec.seed)
        for sample in range(spec.n_samples):
            for _ in range(100):
                train_idx = rng.choice(
                    bag_indices,
                    size=dataset.n_bags,
                    replace=True,
                )
                test_idx = np.setdiff1d(bag_indices, np.unique(train_idx))
                if len(test_idx) > 0:
                    break
            else:
                raise RuntimeError("could not draw a bootstrap sample with out-of-bag bags")
            resamples.append(
                _resample(
                    f"sample_{sample:02d}",
                    train_idx,
                    test_idx,
                    sample=sample,
                )
            )
    else:
        raise NotImplementedError(
            f"multi-instance resampling does not support {spec.strategy.value}"
        )

    if spec.valid_size is not None:
        resamples = [
            _with_validation_split(
                resample,
                dataset=dataset,
                spec=spec,
            )
            for resample in resamples
        ]

    return ResamplingPlan(
        dataset_name=dataset.name,
        task_name=task.name,
        spec=spec,
        resamples=resamples,
        metadata={"paradigm": "multi_instance"},
    )


def _bag_column(
    dataset: MultiInstanceDataset,
    column: str | None,
    *,
    role: str,
) -> np.ndarray:
    if column is None:
        raise ValueError(f"{role} column must be configured")
    dataset.require_bag_columns([column])
    return dataset.bag_table[column].to_numpy()


def _group_values(dataset: MultiInstanceDataset, columns: list[str]) -> np.ndarray:
    dataset.require_bag_columns(columns)
    if len(columns) == 1:
        return dataset.bag_table[columns[0]].to_numpy()

    values = np.empty(dataset.n_bags, dtype=object)
    values[:] = list(dataset.bag_table.loc[:, columns].itertuples(index=False, name=None))
    return values


def _with_validation_split(
    resample: Resample,
    *,
    dataset: MultiInstanceDataset,
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
            splitter.split(
                resample.train_idx,
                groups=groups[resample.train_idx],
            )
        )
        train_idx = resample.train_idx[train_pos]
        valid_idx = resample.train_idx[valid_pos]
    else:
        stratify = (
            None
            if spec.stratify is None
            else _bag_column(dataset, spec.stratify, role="stratify")[resample.train_idx]
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
    train_idx,
    test_idx,
    *,
    fold: int | None = None,
    sample: int | None = None,
) -> Resample:
    metadata: dict[str, Any] = {"paradigm": "multi_instance", "unit": "bag"}
    if fold is not None:
        metadata["fold"] = fold
    if sample is not None:
        metadata["sample"] = sample
    return Resample(id=id, train_idx=train_idx, test_idx=test_idx, metadata=metadata)
