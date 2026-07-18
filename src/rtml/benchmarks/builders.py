"""Build benchmark suites from configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rtml.benchmarks.base import BenchmarkSuite
from rtml.datasets.openml_loaders import OPENML_CC18_SUITE_ID, load_openml_suite
from rtml.datasets.sklearn_loaders import (
    load_sklearn_classification_suite,
    load_sklearn_regression_suite,
)

OPENML_SUITE_ALIASES = {
    "cc18": OPENML_CC18_SUITE_ID,
    "openml-cc18": OPENML_CC18_SUITE_ID,
}


def build_benchmark_suite(config: Mapping[str, Any] | None) -> BenchmarkSuite:
    """Build a BenchmarkSuite from a benchmark source config."""
    config = config or {}
    source = str(config.get("source") or "sklearn").lower()

    if source == "sklearn":
        return build_sklearn_benchmark_suite(config)
    if source == "openml":
        return build_openml_benchmark_suite(config)
    if source in {"huggingface", "hf"}:
        raise NotImplementedError("huggingface benchmark suites are not implemented yet")
    if source == "local":
        raise NotImplementedError("local benchmark suites are not implemented yet")

    raise ValueError(f"unsupported benchmark source {source!r}")


def build_sklearn_benchmark_suite(config: Mapping[str, Any]) -> BenchmarkSuite:
    suite_name = str(config.get("suite") or "classification").lower()
    if suite_name == "classification":
        suite = load_sklearn_classification_suite()
    elif suite_name == "regression":
        suite = load_sklearn_regression_suite()
    else:
        raise ValueError(
            "sklearn benchmark configs currently support suite='classification' "
            "or suite='regression'"
        )

    return BenchmarkSuite(
        name=str(config.get("name") or suite.name),
        cases=suite.cases,
        metadata=suite.metadata,
    )


def build_openml_benchmark_suite(config: Mapping[str, Any]) -> BenchmarkSuite:
    suite_id = openml_suite_id(config)
    suite = load_openml_suite(suite_id)
    return BenchmarkSuite(
        name=str(config.get("name") or suite.name),
        cases=suite.cases,
        metadata=suite.metadata,
    )


def openml_suite_id(config: Mapping[str, Any]) -> int:
    suite_id = config.get("suite_id")
    if suite_id is not None:
        return int(suite_id)

    suite_name = config.get("suite")
    if suite_name is not None:
        alias = str(suite_name).lower()
        if alias in OPENML_SUITE_ALIASES:
            return OPENML_SUITE_ALIASES[alias]

    raise ValueError("OpenML benchmark configs must define suite_id or a known suite alias")
