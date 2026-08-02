"""Population-statistics tasks adapted from the Deep Sets paper.

Reference:
    https://github.com/manzilzaheer/DeepSets/tree/master/PopStats/generator
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

from rtml.core.benchmarks import BenchmarkCase, BenchmarkSuite
from rtml.core.datasets import FeatureInfo, FeatureKind, FeatureSchema
from rtml.core.resampling import ResamplingSpec, ResamplingStrategy
from rtml.core.tasks import MetricSpec, TaskType
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.resampling import build_multi_instance_resampling_plan
from rtml.multi_instance.tasks import MultiInstanceTask

DEFAULT_N_BAGS = 512
DEFAULT_INSTANCES_PER_BAG = 512
DEFAULT_SEED = 3

BAG_ID_COLUMN = "bag_id"
TARGET_COLUMN = "target"

PopstatsTask = Literal["entropy_2d", "correlation_mi", "rank1_mi", "random_covariance_mi"]
PopstatsGenerator = Callable[
    [int, int, int],
    tuple[list[np.ndarray], np.ndarray, str, np.ndarray],
]

POPSTATS_TASKS: dict[int, PopstatsTask] = {
    1: "entropy_2d",
    2: "correlation_mi",
    3: "rank1_mi",
    4: "random_covariance_mi",
}


def generate_entropy_2d_bags(
    n_bags: int,
    instances_per_bag: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray, str, np.ndarray]:
    """Generate 2D Gaussian bags with entropy of the first coordinate as target."""
    rng = np.random.default_rng(seed)
    base_transform = rng.random((2, 2))
    angles = np.linspace(0, np.pi, n_bags)

    bags = []
    targets = np.zeros(n_bags)
    for bag_index, angle in enumerate(angles):
        rotation = _rotation_matrix_2d(angle) @ base_transform
        bag = (rotation @ rng.standard_normal((2, instances_per_bag))).T
        covariance = rotation @ rotation.T
        variance = covariance[0, 0]

        bags.append(bag)
        targets[bag_index] = 0.5 * np.log(2 * np.pi * np.e * variance)

    return bags, targets, "angle", angles


def generate_correlation_mi_bags(
    n_bags: int,
    instances_per_bag: int,
    seed: int,
    *,
    dim: int = 16,
) -> tuple[list[np.ndarray], np.ndarray, str, np.ndarray]:
    """Generate Gaussian bags with varying normalized block mutual information."""
    if n_bags % 4 != 0:
        raise ValueError("correlation_mi requires n_bags divisible by 4")
    if dim % 2 != 0:
        raise ValueError("correlation_mi requires an even dimension")

    rng = np.random.default_rng(seed)
    low_corr = np.linspace(0.01, 0.99, n_bags // 4)
    nonlinear_corr = np.sqrt(1 - np.exp(-4 * low_corr))
    correlations = np.sort(np.concatenate([-low_corr, -nonlinear_corr, low_corr, nonlinear_corr]))

    half_dim = dim // 2
    seed_matrix = rng.random((half_dim, half_dim))
    base_covariance = (seed_matrix @ seed_matrix.T + np.eye(half_dim)) / np.sum(seed_matrix**2)

    bags = []
    targets = np.zeros(n_bags)
    for bag_index, correlation in enumerate(correlations):
        covariance = np.block(
            [
                [base_covariance, correlation * base_covariance],
                [correlation * base_covariance, base_covariance],
            ]
        )
        cholesky = np.linalg.cholesky(covariance)
        bag = (cholesky @ rng.standard_normal((dim, instances_per_bag))).T

        bags.append(bag)
        targets[bag_index] = -0.5 * np.log(1 - correlation**2)

    return bags, targets, "correlation", correlations


def generate_rank1_mi_bags(
    n_bags: int,
    instances_per_bag: int,
    seed: int,
    *,
    dim: int = 32,
) -> tuple[list[np.ndarray], np.ndarray, str, np.ndarray]:
    """Generate Gaussian bags with total correlation from rank-1 covariance updates."""
    rng = np.random.default_rng(seed)
    direction = rng.random((dim, 1))
    direction = direction / np.linalg.norm(direction)
    strengths = np.linspace(0, 1, n_bags)

    bags = []
    targets = np.zeros(n_bags)
    identity = np.eye(dim)
    for bag_index, strength in enumerate(strengths):
        update = np.sqrt(strength) * direction
        cholesky = np.linalg.cholesky(identity + update @ update.T)
        bag = (cholesky @ rng.standard_normal((dim, instances_per_bag))).T

        bags.append(bag)
        targets[bag_index] = _gaussian_total_correlation_from_cholesky(cholesky)

    return bags, targets, "rank1_strength", strengths


def generate_random_covariance_mi_bags(
    n_bags: int,
    instances_per_bag: int,
    seed: int,
    *,
    dim: int = 32,
) -> tuple[list[np.ndarray], np.ndarray, str, np.ndarray]:
    """Generate Gaussian bags from random covariances and sort by total correlation."""
    rng = np.random.default_rng(seed)

    bags = []
    targets = np.zeros(n_bags)
    for bag_index in range(n_bags):
        seed_matrix = rng.random((dim, dim))
        covariance = (seed_matrix @ seed_matrix.T + np.eye(dim)) / np.sum(seed_matrix**2)
        cholesky = np.linalg.cholesky(covariance)
        bag = (cholesky @ rng.standard_normal((dim, instances_per_bag))).T

        bags.append(bag)
        targets[bag_index] = _gaussian_total_correlation_from_cholesky(cholesky)

    order = np.argsort(targets)
    sorted_bags = [bags[index] for index in order]
    sorted_targets = targets[order]
    return sorted_bags, sorted_targets, "rank", np.linspace(0, 1, n_bags)


def load_popstats_dataset(
    task_id: int,
    *,
    n_bags: int = DEFAULT_N_BAGS,
    instances_per_bag: int = DEFAULT_INSTANCES_PER_BAG,
    seed: int = DEFAULT_SEED,
) -> tuple[MultiInstanceDataset, MultiInstanceTask]:
    """Load one synthetic PopStats multiple-instance regression task."""
    generator = _popstats_generator(task_id)
    bags, target, parameter_name, parameter_values = generator(n_bags, instances_per_bag, seed)
    dataset_name = f"popstats_{POPSTATS_TASKS[task_id]}"

    bag_frame = pd.DataFrame(
        {
            BAG_ID_COLUMN: np.arange(len(bags)),
            TARGET_COLUMN: target,
            parameter_name: parameter_values,
        }
    )
    instance_frame, offsets = _stack_instances(bags)

    bag_schema = FeatureSchema(
        {
            BAG_ID_COLUMN: FeatureInfo(
                BAG_ID_COLUMN, FeatureKind.ID, dtype=str(bag_frame[BAG_ID_COLUMN].dtype)
            ),
            TARGET_COLUMN: FeatureInfo(
                TARGET_COLUMN, FeatureKind.NUMERIC, dtype=str(bag_frame[TARGET_COLUMN].dtype)
            ),
            parameter_name: FeatureInfo(
                parameter_name,
                FeatureKind.NUMERIC,
                dtype=str(bag_frame[parameter_name].dtype),
                metadata={"role": "generator_parameter"},
            ),
        }
    )
    instance_schema = FeatureSchema(
        {
            column: FeatureInfo(
                column, FeatureKind.NUMERIC, dtype=str(instance_frame[column].dtype)
            )
            for column in instance_frame.columns
        }
    )

    dataset = MultiInstanceDataset(
        name=dataset_name,
        bag_table=bag_frame,
        instance_table=instance_frame,
        bag_schema=bag_schema,
        instance_schema=instance_schema,
        bag_offsets=offsets,
        bag_id_column=BAG_ID_COLUMN,
        metadata={
            "source": "popstats",
            "paradigm": "multi_instance",
            "task_id": task_id,
            "task": POPSTATS_TASKS[task_id],
            "n_bags": n_bags,
            "instances_per_bag": instances_per_bag,
            "seed": seed,
        },
    )
    task = MultiInstanceTask(
        name=dataset_name,
        task_type=TaskType.REGRESSION,
        instance_source=dataset.select_instance_features(kinds=[FeatureKind.NUMERIC]),
        target=TARGET_COLUMN,
        metrics=[MetricSpec("rmse"), MetricSpec("mae")],
        primary_metric="rmse",
        metadata={
            "source": "popstats",
            "paradigm": "multi_instance",
            "target_level": "bag",
        },
    )
    task.validate_columns(dataset)
    return dataset, task


def load_popstats_benchmark_case(
    task_id: int,
    *,
    n_bags: int = DEFAULT_N_BAGS,
    instances_per_bag: int = DEFAULT_INSTANCES_PER_BAG,
    seed: int = DEFAULT_SEED,
    n_folds: int = 5,
    valid_size: float | None = None,
) -> BenchmarkCase[MultiInstanceDataset, MultiInstanceTask]:
    """Create one PopStats benchmark case with bag-level K-fold resampling."""
    dataset, task = load_popstats_dataset(
        task_id,
        n_bags=n_bags,
        instances_per_bag=instances_per_bag,
        seed=seed,
    )
    spec = ResamplingSpec(
        name=f"{dataset.name}_bag_kfold",
        strategy=ResamplingStrategy.KFOLD,
        n_folds=n_folds,
        valid_size=valid_size,
        shuffle=True,
        seed=seed,
        metadata={"source": "popstats", "paradigm": "multi_instance"},
    )
    return BenchmarkCase(
        name=dataset.name,
        dataset=dataset,
        task=task,
        resampling=build_multi_instance_resampling_plan(dataset=dataset, task=task, spec=spec),
        metadata={"source": "popstats", "paradigm": "multi_instance"},
    )


def load_popstats_suite(
    *,
    task_ids: list[int] | None = None,
    n_bags: int = DEFAULT_N_BAGS,
    instances_per_bag: int = DEFAULT_INSTANCES_PER_BAG,
    seed: int = DEFAULT_SEED,
    n_folds: int = 5,
    valid_size: float | None = None,
) -> BenchmarkSuite[MultiInstanceDataset, MultiInstanceTask]:
    """Create a small PopStats multiple-instance regression suite."""
    ids = list(task_ids or POPSTATS_TASKS)
    cases = [
        load_popstats_benchmark_case(
            task_id,
            n_bags=n_bags,
            instances_per_bag=instances_per_bag,
            seed=seed,
            n_folds=n_folds,
            valid_size=valid_size,
        )
        for task_id in ids
    ]
    return BenchmarkSuite(
        name="popstats",
        cases=cases,
        metadata={"source": "popstats", "paradigm": "multi_instance"},
    )


def _popstats_generator(task_id: int) -> PopstatsGenerator:
    try:
        task = POPSTATS_TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"unknown PopStats task_id {task_id!r}") from exc

    if task == "entropy_2d":
        return generate_entropy_2d_bags
    if task == "correlation_mi":
        return generate_correlation_mi_bags
    if task == "rank1_mi":
        return generate_rank1_mi_bags
    if task == "random_covariance_mi":
        return generate_random_covariance_mi_bags
    raise AssertionError(f"unhandled PopStats task {task!r}")


def _stack_instances(bags: list[np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    if not bags:
        raise ValueError("PopStats dataset requires at least one bag")

    n_features = bags[0].shape[1]
    for index, bag in enumerate(bags):
        if bag.ndim != 2:
            raise ValueError(f"bag {index} must be a two-dimensional array")
        if bag.shape[1] != n_features:
            raise ValueError("all bags must have the same number of instance features")

    offsets = np.zeros(len(bags) + 1, dtype=int)
    offsets[1:] = np.cumsum([len(bag) for bag in bags])
    values = np.concatenate(bags, axis=0)
    columns = [f"x_{index:02d}" for index in range(n_features)]
    return pd.DataFrame(values, columns=columns), offsets


def _rotation_matrix_2d(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]])


def _gaussian_total_correlation_from_cholesky(cholesky: np.ndarray) -> float:
    return float(
        -np.sum(np.log(np.diag(cholesky))) + 0.5 * np.sum(np.log(np.sum(cholesky**2, axis=1)))
    )
