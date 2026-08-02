"""Classic WEKA multiple-instance dataset source."""

from rtml.multi_instance.datasets.classic.benchmarks import (
    load_classic_mil_case,
    load_classic_mil_suite,
)
from rtml.multi_instance.datasets.classic.loaders import (
    load_classic_mil_dataset,
    parse_classic_mil_arff,
)

__all__ = [
    "load_classic_mil_case",
    "load_classic_mil_dataset",
    "load_classic_mil_suite",
    "parse_classic_mil_arff",
]
