import pytest

from rtml.core.benchmarks import BenchmarkSuite
from rtml.single_instance.datasets.sklearn_loaders import (
    build_sklearn_benchmark_case,
    build_sklearn_resampling_spec,
    load_breast_cancer_dataset,
)
from rtml.core.methods import MethodSpec, ModelSpec
from rtml.core.resampling import ResamplingStrategy
from rtml.core.studies import Study, StudyKind


def make_suite() -> BenchmarkSuite:
    dataset, task = load_breast_cancer_dataset()
    spec = build_sklearn_resampling_spec(
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


def test_study_from_suite_uses_suite_name_by_default() -> None:
    suite = make_suite()
    method = make_method()

    study = Study.from_suite(suite=suite, methods=[method])

    assert study.name == suite.name
    assert study.suite == suite
    assert study.methods == [method]
    assert study.kind is StudyKind.COMPARISON


def test_study_from_case_wraps_one_case_suite() -> None:
    case = make_suite().cases[0]
    method = make_method()

    study = Study.from_case(case=case, methods=[method], name="single_case")

    assert study.name == "single_case"
    assert study.suite.name == case.name
    assert study.suite.cases == [case]
    assert study.methods == [method]


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
