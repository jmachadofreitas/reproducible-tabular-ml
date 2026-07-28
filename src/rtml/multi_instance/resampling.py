from __future__ import annotations

from typing import Any

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split
import numpy as np

from rtml.core.resampling import Resample, ResamplingPlan, ResamplingSpec, ResamplingStrategy
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.tasks import MultiInstanceTask


def build_multi_instance_resampling_plan(
    *,
    dataset: MultiInstanceDataset,
    task: MultiInstanceTask,
    spec: ResamplingSpec,
) -> ResamplingPlan:
    """Materialize a resampling plan whose indices refer to bag positions."""
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
    else:
        raise NotImplementedError(
            f"multi-instance resampling does not support {spec.strategy.value}"
        )

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


def _resample(id: str, train_idx, test_idx, *, fold: int | None = None) -> Resample:
    metadata: dict[str, Any] = {"paradigm": "multi_instance", "unit": "bag"}
    if fold is not None:
        metadata["fold"] = fold
    return Resample(id=id, train_idx=train_idx, test_idx=test_idx, metadata=metadata)
