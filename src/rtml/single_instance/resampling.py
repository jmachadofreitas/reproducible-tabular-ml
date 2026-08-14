import numpy as np

from rtml.core.datasets import Dataset
from rtml.core.resampling import (
    ResamplingPlan,
    ResamplingSpec,
    ResamplingStrategy,
    materialize_resamples,
)
from rtml.core.tasks import TaskSpec


def build_single_instance_resampling_plan(
    *,
    dataset: Dataset,
    task: TaskSpec,
    spec: ResamplingSpec,
) -> ResamplingPlan:
    """Materialize row-position splits and optional saved validation rows."""
    task.validate_columns(dataset)
    _validate_group_columns(task.groups, spec.groups)
    stratify_values = None if spec.stratify is None else _column_values(dataset, spec.stratify)
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
            n_items=len(dataset),
            spec=spec,
            unit="row",
            metadata={"paradigm": "single_instance", "unit": "row"},
            stratify_values=stratify_values,
            group_values=group_values,
        ),
        metadata={"paradigm": "single_instance"},
    )


def _validate_group_columns(declared: list[str], selected: list[str]) -> None:
    undeclared = [column for column in selected if column not in declared]
    if undeclared:
        raise ValueError(f"resampling groups are not declared by the task: {undeclared}")


def _column_values(dataset: Dataset, column: str) -> np.ndarray:
    dataset.require_columns([column])
    return dataset.data[column].to_numpy()


def _group_values(dataset: Dataset, columns: list[str]) -> np.ndarray:
    dataset.require_columns(columns)
    if len(columns) == 1:
        return dataset.data[columns[0]].to_numpy()

    values = np.empty(len(dataset), dtype=object)
    values[:] = list(dataset.data.loc[:, columns].itertuples(index=False, name=None))
    return values
