import numpy as np

from rtml.core.resampling import (
    ResamplingPlan,
    ResamplingSpec,
    ResamplingStrategy,
    materialize_resamples,
)
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.tasks import MultiInstanceTask


def build_multi_instance_resampling_plan(
    *,
    dataset: MultiInstanceDataset,
    task: MultiInstanceTask,
    spec: ResamplingSpec,
) -> ResamplingPlan:
    """Materialize bag-position splits and optional saved validation bags."""
    task.validate_columns(dataset)
    _validate_group_columns(task.groups, spec.groups)
    stratify_values = None if spec.stratify is None else _bag_column(dataset, spec.stratify)
    group_values = (
        _group_values(dataset, spec.groups)
        if spec.strategy == ResamplingStrategy.GROUP_KFOLD
        else None
    )
    return ResamplingPlan(
        dataset_name=dataset.name,
        task_name=task.name,
        spec=spec,
        resamples=materialize_resamples(
            n_items=dataset.n_bags,
            spec=spec,
            unit="bag",
            metadata={"paradigm": "multi_instance", "unit": "bag"},
            stratify_values=stratify_values,
            group_values=group_values,
        ),
        metadata={"paradigm": "multi_instance"},
    )


def _validate_group_columns(declared: list[str], selected: list[str]) -> None:
    undeclared = [column for column in selected if column not in declared]
    if undeclared:
        raise ValueError(f"resampling groups are not declared by the task: {undeclared}")


def _bag_column(dataset: MultiInstanceDataset, column: str) -> np.ndarray:
    dataset.require_bag_columns([column])
    return dataset.bag_table[column].to_numpy()


def _group_values(dataset: MultiInstanceDataset, columns: list[str]) -> np.ndarray:
    dataset.require_bag_columns(columns)
    if len(columns) == 1:
        return dataset.bag_table[columns[0]].to_numpy()

    values = np.empty(dataset.n_bags, dtype=object)
    values[:] = list(dataset.bag_table.loc[:, columns].itertuples(index=False, name=None))
    return values
