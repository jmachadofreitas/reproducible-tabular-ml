import pytest

from rtml.core.resampling import Resample, ResamplingSpec, ResamplingStrategy


def test_resample_normalizes_indices_and_metadata() -> None:
    resample = Resample(
        id="fold_00",
        train_idx=[0, 2, 4],
        test_idx=[1, 3],
        metadata={"fold": 0},
    )

    assert resample.train_idx.tolist() == [0, 2, 4]
    assert resample.test_idx.tolist() == [1, 3]
    assert resample.metadata == {"fold": 0}


def test_kfold_resampling_spec_requires_multiple_folds() -> None:
    with pytest.raises(ValueError, match="n_folds >= 2"):
        ResamplingSpec(name="bad_kfold", strategy=ResamplingStrategy.KFOLD, n_folds=1)


def test_stratified_holdout_requires_test_size_and_stratify() -> None:
    with pytest.raises(ValueError, match="requires test_size"):
        ResamplingSpec(name="bad_holdout", strategy=ResamplingStrategy.STRATIFIED_HOLDOUT)

    with pytest.raises(ValueError, match="requires stratify"):
        ResamplingSpec(
            name="bad_holdout",
            strategy=ResamplingStrategy.STRATIFIED_HOLDOUT,
            test_size=0.2,
        )


def test_group_kfold_requires_group_columns() -> None:
    with pytest.raises(ValueError, match="requires groups"):
        ResamplingSpec(name="bad_group_kfold", strategy=ResamplingStrategy.GROUP_KFOLD, n_folds=5)


def test_openml_task_resampling_spec_stays_valid() -> None:
    spec = ResamplingSpec(
        name="openml_task_1590",
        strategy=ResamplingStrategy.UNKNOWN_OPENML_TASK,
        n_repeats=1,
        n_folds=10,
        n_samples=1,
    )

    assert spec.strategy == ResamplingStrategy.UNKNOWN_OPENML_TASK
    assert spec.n_folds == 10
