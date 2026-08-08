import numpy as np
import pandas as pd
import pytest

from rtml.core.benchmarks import BenchmarkCase, BenchmarkSuite
from rtml.core.datasets import FeatureInfo, FeatureKind, FeatureSchema
from rtml.core.fingerprints import fingerprint_dataset
from rtml.core.resampling import ResamplingSpec, ResamplingStrategy
from rtml.core.tasks import MetricSpec, TaskType
from rtml.multi_instance import MultiInstanceDataset, MultiInstanceTask
from rtml.multi_instance.datasets.popstats import (
    generate_correlation_mi_bags,
    generate_entropy_2d_bags,
    generate_random_covariance_mi_bags,
    generate_rank1_mi_bags,
    load_popstats_dataset,
    load_popstats_suite,
)
from rtml.multi_instance.resampling import build_multi_instance_resampling_plan


def make_mil_dataset() -> MultiInstanceDataset:
    bag_table = pd.DataFrame({"bag_id": [10, 20, 30], "target": [0.1, 0.2, 0.3]})
    instance_table = pd.DataFrame(
        {
            "x_00": [1.0, 1.2, 2.0, 2.2, 2.4, 3.0],
            "x_01": [0.0, 0.2, 1.0, 1.2, 1.4, 2.0],
        }
    )
    return MultiInstanceDataset(
        name="toy_mil",
        bag_table=bag_table,
        instance_table=instance_table,
        bag_schema=FeatureSchema(
            {
                "bag_id": FeatureInfo("bag_id", FeatureKind.ID),
                "target": FeatureInfo("target", FeatureKind.NUMERIC),
            }
        ),
        instance_schema=FeatureSchema(
            {
                "x_00": FeatureInfo("x_00", FeatureKind.NUMERIC),
                "x_01": FeatureInfo("x_01", FeatureKind.NUMERIC),
            }
        ),
        bag_offsets=np.array([0, 2, 5, 6]),
        bag_id_column="bag_id",
    )


def make_mil_task() -> MultiInstanceTask:
    return MultiInstanceTask(
        name="toy_mil",
        task_type=TaskType.REGRESSION,
        instance_source=["x_00", "x_01"],
        target="target",
        metrics=[
            MetricSpec(name="rmse", greater_is_better=False),
            MetricSpec(name="mae", greater_is_better=False),
        ],
        primary_metric="rmse",
    )


def make_grouped_mil_dataset() -> MultiInstanceDataset:
    bag_table = pd.DataFrame(
        {
            "bag_id": np.arange(8),
            "target": np.linspace(0.0, 1.0, 8),
            "site": ["a", "b"] * 4,
            "subject": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
        }
    )
    instance_table = pd.DataFrame({"x": np.arange(16, dtype=float)})
    return MultiInstanceDataset(
        name="grouped_mil",
        bag_table=bag_table,
        instance_table=instance_table,
        bag_schema=FeatureSchema.infer(
            bag_table,
            id_columns=["bag_id"],
            categorical_columns=["site", "subject"],
        ),
        instance_schema=FeatureSchema.infer(instance_table),
        bag_offsets=np.arange(0, 17, 2),
        bag_id_column="bag_id",
    )


def test_multi_instance_dataset_uses_offsets_for_bag_storage() -> None:
    dataset = make_mil_dataset()

    assert dataset.n_bags == 3
    assert dataset.n_instances == 6
    assert dataset.bag_size(1) == 3
    assert dataset.bag_instances(1)["x_00"].tolist() == [2.0, 2.2, 2.4]
    assert dataset.select_instance_features(kinds=[FeatureKind.NUMERIC]) == [
        "x_00",
        "x_01",
    ]
    assert dataset.sample_ids_for([2, 0]).tolist() == [30, 10]


def test_multi_instance_dataset_supports_evidence_and_bag_subgroups() -> None:
    dataset = make_grouped_mil_dataset()

    assert fingerprint_dataset(dataset).startswith("sha256:")
    assert dataset.subgroup_values(["site"], [1, 4])["site"].tolist() == ["b", "a"]


def test_local_multi_instance_fingerprint_covers_tables_and_offsets() -> None:
    original = make_mil_dataset()
    changed_bag = make_mil_dataset()
    changed_instance = make_mil_dataset()
    changed_offsets = make_mil_dataset()
    changed_bag.bag_table.loc[0, "target"] = 0.9
    changed_instance.instance_table.loc[0, "x_00"] = 9.0
    changed_offsets.bag_offsets = np.array([0, 1, 5, 6])

    original_fingerprint = fingerprint_dataset(original)
    assert fingerprint_dataset(changed_bag) != original_fingerprint
    assert fingerprint_dataset(changed_instance) != original_fingerprint
    assert fingerprint_dataset(changed_offsets) != original_fingerprint


def test_multi_instance_dataset_rejects_bad_offsets() -> None:
    dataset = make_mil_dataset()

    with pytest.raises(ValueError, match="last bag offset"):
        MultiInstanceDataset(
            name="bad",
            bag_table=dataset.bag_table,
            instance_table=dataset.instance_table,
            bag_schema=dataset.bag_schema,
            instance_schema=dataset.instance_schema,
            bag_offsets=np.array([0, 2, 5, 7]),
        )


def test_multi_instance_select_bags_builds_contiguous_offsets() -> None:
    dataset = make_mil_dataset()

    instances, offsets = dataset.select_bags([2, 0])

    assert offsets.tolist() == [0, 1, 3]
    assert instances["x_00"].tolist() == [3.0, 1.0, 1.2]


def test_multi_instance_getitem_selects_bags() -> None:
    dataset = make_mil_dataset()

    instances, offsets = dataset[1]

    assert offsets.tolist() == [0, 3]
    assert instances["x_00"].tolist() == [2.0, 2.2, 2.4]


def test_multi_instance_select_bags_allows_repeated_positions() -> None:
    dataset = make_mil_dataset()

    instances, offsets = dataset.select_bags([1, 2, 1])

    assert offsets.tolist() == [0, 3, 4, 7]
    assert instances["x_00"].tolist() == [2.0, 2.2, 2.4, 3.0, 2.0, 2.2, 2.4]


def test_multi_instance_task_validates_bag_target_and_instance_inputs() -> None:
    dataset = make_mil_dataset()
    task = make_mil_task()

    task.validate_columns(dataset)
    assert task.target_series(dataset).tolist() == [0.1, 0.2, 0.3]

    bad_task = MultiInstanceTask(
        name="bad",
        task_type=TaskType.REGRESSION,
        instance_source=["missing"],
        target="target",
    )
    with pytest.raises(ValueError, match="instance columns"):
        bad_task.validate_columns(dataset)


def test_multi_instance_resampling_indices_are_bag_positions() -> None:
    dataset = make_mil_dataset()
    dataset.bag_table.index = [10, 20, 30]
    task = make_mil_task()
    spec = ResamplingSpec(
        name="toy_kfold",
        strategy=ResamplingStrategy.KFOLD,
        n_folds=3,
        shuffle=False,
        metadata={"paradigm": "multi_instance"},
    )

    plan = build_multi_instance_resampling_plan(dataset=dataset, task=task, spec=spec)

    assert len(plan.resamples) == 3
    assert set(plan.resamples[0].train_idx).union(plan.resamples[0].test_idx) == {0, 1, 2}
    assert plan.metadata["paradigm"] == "multi_instance"
    assert plan.resamples[0].metadata["unit"] == "bag"


def test_multi_instance_bootstrap_repeats_bags_and_uses_out_of_bag_test_set() -> None:
    dataset = make_grouped_mil_dataset()
    task = MultiInstanceTask(
        name="grouped_mil",
        task_type=TaskType.REGRESSION,
        instance_source=["x"],
        target="target",
    )
    spec = ResamplingSpec(
        name="bag_bootstrap",
        strategy=ResamplingStrategy.BOOTSTRAP,
        n_samples=3,
        seed=7,
    )

    plan = build_multi_instance_resampling_plan(dataset=dataset, task=task, spec=spec)

    assert len(plan.resamples) == 3
    for resample in plan.resamples:
        assert len(resample.train_idx) == dataset.n_bags
        assert len(np.unique(resample.train_idx)) < dataset.n_bags
        assert set(resample.train_idx).isdisjoint(resample.test_idx)


def test_multi_instance_stratification_uses_configured_bag_column() -> None:
    dataset = make_grouped_mil_dataset()
    task = MultiInstanceTask(
        name="grouped_mil",
        task_type=TaskType.REGRESSION,
        instance_source=["x"],
        target="target",
    )
    spec = ResamplingSpec(
        name="site_stratified",
        strategy=ResamplingStrategy.STRATIFIED_KFOLD,
        n_folds=2,
        valid_size=0.5,
        shuffle=True,
        seed=3,
        stratify="site",
    )

    plan = build_multi_instance_resampling_plan(dataset=dataset, task=task, spec=spec)

    for resample in plan.resamples:
        assert resample.valid_idx is not None
        train_sites = set(dataset.bag_table.iloc[resample.train_idx]["site"])
        valid_sites = set(dataset.bag_table.iloc[resample.valid_idx]["site"])
        test_sites = set(dataset.bag_table.iloc[resample.test_idx]["site"])
        assert train_sites == {"a", "b"}
        assert valid_sites == {"a", "b"}
        assert test_sites == {"a", "b"}
        assert set(resample.train_idx).isdisjoint(resample.valid_idx)
        assert set(resample.valid_idx).isdisjoint(resample.test_idx)
        assert set(resample.train_idx).union(resample.valid_idx, resample.test_idx) == set(
            range(dataset.n_bags)
        )


def test_multi_instance_group_kfold_keeps_bag_groups_together() -> None:
    dataset = make_grouped_mil_dataset()
    task = MultiInstanceTask(
        name="grouped_mil",
        task_type=TaskType.REGRESSION,
        instance_source=["x"],
        target="target",
        groups=["subject"],
    )
    spec = ResamplingSpec(
        name="subject_group_kfold",
        strategy=ResamplingStrategy.GROUP_KFOLD,
        n_folds=4,
        valid_size=1 / 3,
        groups=["subject"],
        seed=3,
    )

    plan = build_multi_instance_resampling_plan(dataset=dataset, task=task, spec=spec)

    assert len(plan.resamples) == 4
    for resample in plan.resamples:
        assert resample.valid_idx is not None
        train_groups = set(dataset.bag_table.iloc[resample.train_idx]["subject"])
        valid_groups = set(dataset.bag_table.iloc[resample.valid_idx]["subject"])
        test_groups = set(dataset.bag_table.iloc[resample.test_idx]["subject"])
        assert train_groups.isdisjoint(test_groups)
        assert train_groups.isdisjoint(valid_groups)
        assert valid_groups.isdisjoint(test_groups)

    repeated = build_multi_instance_resampling_plan(dataset=dataset, task=task, spec=spec)
    assert repeated.fingerprint == plan.fingerprint
    for expected, actual in zip(plan.resamples, repeated.resamples, strict=True):
        assert np.array_equal(expected.train_idx, actual.train_idx)
        assert np.array_equal(expected.valid_idx, actual.valid_idx)
        assert np.array_equal(expected.test_idx, actual.test_idx)


def test_popstats_loader_returns_bag_level_regression_task() -> None:
    dataset, task = load_popstats_dataset(1, n_bags=8, instances_per_bag=4, seed=13)
    dataset_again, _ = load_popstats_dataset(1, n_bags=8, instances_per_bag=4, seed=13)

    assert dataset.name == "popstats_entropy_2d"
    assert dataset.bag_table.shape[0] == 8
    assert dataset.instance_table.shape == (32, 2)
    assert dataset.bag_offsets.tolist() == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert task.target == "target"
    assert task.task_type == TaskType.REGRESSION
    assert np.allclose(dataset.bag_table["target"], dataset_again.bag_table["target"])
    assert np.allclose(dataset.instance_table, dataset_again.instance_table)
    assert "source_identity" not in dataset.metadata
    assert fingerprint_dataset(dataset) == fingerprint_dataset(dataset_again)

    dataset_again.instance_table.loc[0, "x_00"] += 1.0
    assert fingerprint_dataset(dataset) != fingerprint_dataset(dataset_again)


def test_popstats_targets_match_their_gaussian_statistics() -> None:
    seed = 13
    _, entropy, _, angles = generate_entropy_2d_bags(8, 4, seed)
    base_transform = np.random.default_rng(seed).random((2, 2))
    expected_entropy = []
    for angle in angles:
        cosine = np.cos(angle)
        sine = np.sin(angle)
        transform = np.array([[cosine, -sine], [sine, cosine]]) @ base_transform
        variance = (transform @ transform.T)[0, 0]
        expected_entropy.append(0.5 * np.log(2 * np.pi * np.e * variance))
    assert np.allclose(entropy, expected_entropy)

    _, normalized_mi, _, correlations = generate_correlation_mi_bags(8, 4, seed)
    assert np.allclose(normalized_mi, -0.5 * np.log(1 - correlations**2))

    _, total_correlation, _, strengths = generate_rank1_mi_bags(8, 4, seed)
    direction = np.random.default_rng(seed).random((32, 1))
    direction /= np.linalg.norm(direction)
    expected_total_correlation = []
    for strength in strengths:
        covariance = np.eye(32) + strength * direction @ direction.T
        _, log_determinant = np.linalg.slogdet(covariance)
        expected_total_correlation.append(
            0.5 * np.log(np.diag(covariance)).sum() - 0.5 * log_determinant
        )
    assert np.allclose(total_correlation, expected_total_correlation)


def test_random_covariance_popstats_uses_normalized_sorted_rank() -> None:
    _, targets, parameter_name, ranks = generate_random_covariance_mi_bags(8, 4, seed=13)

    assert parameter_name == "rank"
    assert np.all(np.diff(targets) >= 0)
    assert np.allclose(ranks, np.linspace(0, 1, 8))


def test_popstats_suite_builds_multiple_instance_cases() -> None:
    suite = load_popstats_suite(task_ids=[1, 2], n_bags=8, instances_per_bag=3, n_folds=4)

    assert suite.name == "popstats"
    assert isinstance(suite, BenchmarkSuite)
    assert all(isinstance(case, BenchmarkCase) for case in suite.cases)
    assert suite.metadata["paradigm"] == "multi_instance"
    assert [case.name for case in suite.cases] == [
        "popstats_entropy_2d",
        "popstats_correlation_mi",
    ]
    assert all(len(case.resampling.resamples) == 4 for case in suite.cases)
