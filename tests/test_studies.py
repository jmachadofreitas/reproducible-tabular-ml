from dataclasses import replace

import pytest

from rtml.core.benchmarks import BenchmarkCase, BenchmarkSuite
from rtml.core.methods import MethodSpec, ModelSpec
from rtml.core.resampling import ResamplingSpec, ResamplingStrategy
from rtml.core.studies import Study, StudyKind
from rtml.single_instance.datasets.sklearn_loaders import (
    build_sklearn_benchmark_case,
    load_breast_cancer_dataset,
)


def make_suite() -> BenchmarkSuite:
    dataset, task = load_breast_cancer_dataset()
    spec = ResamplingSpec(
        name="breast_cancer_holdout",
        strategy=ResamplingStrategy.STRATIFIED_HOLDOUT,
        test_size=0.25,
        shuffle=True,
        stratify="target",
        seed=42,
    )
    case = build_sklearn_benchmark_case(
        name="breast_cancer_case",
        dataset=dataset,
        task=task,
        resampling_spec=spec,
    )
    return BenchmarkSuite(name="sklearn_suite", cases=[case])


def make_method(name: str = "logreg_linear") -> MethodSpec:
    return MethodSpec(
        name=name,
        transform={"policy": "linear_default"},
        model=ModelSpec(kind="logistic_regression", backend="sklearn"),
        metadata={"factor.family": "linear"},
    )


def test_study_normalizes_kind_methods_and_metadata() -> None:
    study = Study(
        name="linear_comparison",
        suite=make_suite(),
        methods=[make_method()],
        kind="comparison",
        metadata={"question": "baseline"},
    )

    assert study.kind is StudyKind.COMPARISON
    assert study.methods[0].metadata["factor.family"] == "linear"
    assert study.metadata == {"question": "baseline"}


def test_model_spec_requires_backend() -> None:
    with pytest.raises(ValueError, match="backend must be non-empty"):
        ModelSpec(kind="logistic_regression", backend="")


def test_method_spec_requires_model_spec() -> None:
    with pytest.raises(TypeError, match="must be a ModelSpec"):
        MethodSpec(
            name="missing_backend",
            transform={"policy": "linear_default"},
            model={"kind": "logistic_regression", "backend": "sklearn"},  # type: ignore[arg-type]
        )


def test_model_spec_keeps_backend_selection_without_validating_implementation() -> None:
    spec = ModelSpec(kind="logistic_regression", backend="torch")

    assert spec.kind == "logistic_regression"
    assert spec.backend == "torch"


def test_study_requires_at_least_one_method() -> None:
    with pytest.raises(ValueError, match="at least one method"):
        Study(name="empty", suite=make_suite(), methods=[])


def test_study_rejects_duplicate_method_names() -> None:
    with pytest.raises(ValueError, match="method names must be unique"):
        Study(
            name="duplicate_methods",
            suite=make_suite(),
            methods=[make_method("same"), make_method("same")],
        )


def test_benchmark_suite_requires_at_least_one_case() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        BenchmarkSuite(name="empty", cases=[])


def test_benchmark_suite_rejects_duplicate_case_names() -> None:
    case = make_suite().cases[0]

    with pytest.raises(ValueError, match="case names must be unique"):
        BenchmarkSuite(name="duplicates", cases=[case, case])


@pytest.mark.parametrize("field", ["dataset_name", "task_name"])
def test_benchmark_case_requires_matching_resampling_names(field) -> None:
    case = make_suite().cases[0]
    resampling = replace(case.resampling, **{field: "different"})

    with pytest.raises(ValueError, match=f"{field.removesuffix('_name')} name must match"):
        BenchmarkCase(
            name=case.name,
            dataset=case.dataset,
            task=case.task,
            resampling=resampling,
        )
