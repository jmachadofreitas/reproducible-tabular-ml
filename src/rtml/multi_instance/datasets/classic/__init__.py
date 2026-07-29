"""Classic WEKA multiple-instance dataset source."""

from rtml.multi_instance.datasets.classic.loaders import (
    load_classic_mil_dataset,
    parse_classic_mil_arff,
)

__all__ = [
    "load_classic_mil_dataset",
    "parse_classic_mil_arff",
]
