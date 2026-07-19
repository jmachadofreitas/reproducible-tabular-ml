from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import rtml.single_instance.datasets.openml_loaders as openml_loaders
from rtml.core.benchmarks import BenchmarkSuite
from rtml.core.datasets import FeatureKind
from rtml.single_instance.datasets.openml_loaders import DEFAULT_OPENML_SPLIT
from rtml.core.resampling import ResamplingStrategy
from rtml.core.tasks import TaskType


class FakeOpenMLDataset:
    name = "adult"

    def get_data(self, *, dataset_format: str, target: str):
        assert dataset_format == "dataframe"
        assert target == "income"
        x = pd.DataFrame(
            {
                "age": [39.0, None],
                "workclass": ["state-gov", None],
            }
        )
        y = pd.Series(["<=50K", ">50K"], name="income")
        categorical_indicator = [False, True]
        attribute_names = list(x.columns)
        return x, y, categorical_indicator, attribute_names


class FakeOpenMLTask:
    task_id = 1590
    dataset_id = 1590
    target_name = "income"
    task_type_id = openml_loaders.openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
    task_type = "Supervised Classification"
    evaluation_measure = "area_under_roc_curve"
    estimation_procedure = {"type": "crossvalidation", "parameters": {}, "data_splits_url": "mock"}

    def get_dataset(self) -> FakeOpenMLDataset:
        return FakeOpenMLDataset()

    def get_split_dimensions(self) -> tuple[int, int, int]:
        return (1, 2, 1)

    def get_train_test_split_indices(
        self,
        *,
        repeat: int,
        fold: int,
        sample: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert repeat == 0
        assert sample == 0
        if fold == 0:
            return np.array([0]), np.array([1])
        return np.array([1]), np.array([0])


def make_fake_task(task_id: int) -> FakeOpenMLTask:
    task = FakeOpenMLTask()
    task.task_id = task_id
    task.dataset_id = task_id
    return task


def test_load_openml_benchmark_case_builds_dataset_task_and_resampling(monkeypatch) -> None:
    fake_task = make_fake_task(1590)
    monkeypatch.setattr(openml_loaders.openml.tasks, "get_task", lambda *args, **kwargs: fake_task)

    benchmark_case = openml_loaders.load_openml_benchmark_case(1590, suite_id=99)

    assert benchmark_case.dataset.name == "adult"
    assert benchmark_case.dataset.schema.get("age").kind == FeatureKind.NUMERIC
    assert benchmark_case.dataset.schema.get("workclass").kind == FeatureKind.CATEGORICAL
    assert benchmark_case.dataset.schema.get("income").kind == FeatureKind.BINARY
    assert benchmark_case.task.task_type == TaskType.BINARY_CLASSIFICATION
    assert benchmark_case.task.primary_metric == "area_under_roc_curve"
    assert benchmark_case.resampling.spec.strategy == ResamplingStrategy.UNKNOWN_OPENML_TASK
    assert benchmark_case.resampling.spec.n_folds == 2
    assert benchmark_case.resampling.metadata["default_split"] == DEFAULT_OPENML_SPLIT
    assert len(benchmark_case.resampling.resamples) == 2
    assert benchmark_case.metadata["split_dimensions"] == (1, 2, 1)


def test_load_openml_suite_returns_benchmark_suite(monkeypatch) -> None:
    fake_suite = SimpleNamespace(name="OpenML-CC18", description="benchmark", tasks=[1590, 31])
    monkeypatch.setattr(openml_loaders.openml.study, "get_suite", lambda suite_id: fake_suite)
    monkeypatch.setattr(
        openml_loaders.openml.tasks,
        "get_task",
        lambda task_id, **kwargs: make_fake_task(task_id),
    )

    suite = openml_loaders.load_openml_suite(99)

    assert isinstance(suite, BenchmarkSuite)
    assert suite.name == "OpenML-CC18"
    assert [task.metadata["openml_task_id"] for task in suite.cases] == [1590, 31]
    assert suite.metadata["suite_id"] == 99


def test_load_openml_cc18_task_rejects_unknown_task_id(monkeypatch) -> None:
    fake_suite = SimpleNamespace(tasks=[1590, 31])
    monkeypatch.setattr(openml_loaders.openml.study, "get_suite", lambda suite_id: fake_suite)

    with pytest.raises(ValueError, match="OpenML-CC18 suite"):
        openml_loaders.load_openml_cc18_task(999999)


def test_get_openml_task_split_indices_uses_fixed_openml_task_splits(monkeypatch) -> None:
    fake_task = make_fake_task(1590)
    monkeypatch.setattr(openml_loaders.openml.tasks, "get_task", lambda *args, **kwargs: fake_task)

    train_idx, test_idx = openml_loaders.get_openml_task_split_indices(1590)

    assert train_idx.tolist() == [0]
    assert test_idx.tolist() == [1]
