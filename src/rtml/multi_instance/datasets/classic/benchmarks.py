from collections.abc import Sequence
from pathlib import Path

from rtml.core.benchmarks import BenchmarkCase, BenchmarkSuite
from rtml.core.resampling import ResamplingSpec, ResamplingStrategy
from rtml.multi_instance.datasets.base import MultiInstanceDataset
from rtml.multi_instance.datasets.classic.constants import (
    CLASSIC_MIL_DATASETS,
    DEFAULT_CLASSIC_MIL_DATA_DIR,
)
from rtml.multi_instance.datasets.classic.loaders import load_classic_mil_dataset
from rtml.multi_instance.resampling import build_multi_instance_resampling_plan
from rtml.multi_instance.tasks import MultiInstanceTask


def load_classic_mil_case(
    dataset_name: str,
    *,
    root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR,
    n_folds: int = 5,
    seed: int = 0,
    valid_size: float | None = None,
) -> BenchmarkCase[MultiInstanceDataset, MultiInstanceTask]:
    """Load one classic MIL dataset with bag-level stratified folds."""
    dataset, task = load_classic_mil_dataset(dataset_name, root)
    spec = ResamplingSpec(
        name=f"{dataset.name}_stratified_kfold",
        strategy=ResamplingStrategy.STRATIFIED_KFOLD,
        n_folds=n_folds,
        valid_size=valid_size,
        shuffle=True,
        seed=seed,
        stratify=task.target,
        metadata={"source": "classic_mil", "paradigm": "multi_instance"},
    )
    return BenchmarkCase(
        name=dataset.name,
        dataset=dataset,
        task=task,
        resampling=build_multi_instance_resampling_plan(
            dataset=dataset,
            task=task,
            spec=spec,
        ),
        metadata={"source": "classic_mil", "paradigm": "multi_instance"},
    )


def load_classic_mil_suite(
    *,
    dataset_names: Sequence[str] = CLASSIC_MIL_DATASETS,
    root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR,
    n_folds: int = 5,
    seed: int = 0,
    valid_size: float | None = None,
) -> BenchmarkSuite[MultiInstanceDataset, MultiInstanceTask]:
    """Load a benchmark suite from selected classic MIL datasets."""
    names = list(dataset_names)
    if not names:
        raise ValueError("classic MIL suite requires at least one dataset")
    return BenchmarkSuite(
        name="classic_mil",
        cases=[
            load_classic_mil_case(
                name,
                root=root,
                n_folds=n_folds,
                seed=seed,
                valid_size=valid_size,
            )
            for name in names
        ],
        metadata={"source": "classic_mil", "paradigm": "multi_instance"},
    )
