"""Instance-pooling baseline for multiple-instance classification."""

from collections.abc import Callable
from typing import Any

import numpy as np


class InstancePoolingClassifier:
    """Fit an instance classifier with bag labels and pool predictions by bag.

    Every training instance receives its bag label as a weak label. At
    prediction time, class probabilities are pooled into one probability
    distribution per bag.
    """

    def __init__(self, estimator: Any, *, pooling: str = "max") -> None:
        self.estimator = estimator
        self.pooling = pooling
        self._pool = _resolve_pooling(pooling)

    def fit(
        self,
        instances: Any,
        bag_offsets: np.ndarray,
        bag_targets: Any,
    ) -> InstancePoolingClassifier:
        targets = np.asarray(bag_targets)
        if targets.ndim != 1:
            raise ValueError("bag targets must be one-dimensional")
        bag_sizes = _validate_bags(
            bag_offsets,
            n_instances=len(instances),
            n_bags=len(targets),
        )
        self.estimator.fit(instances, np.repeat(targets, bag_sizes))
        self.classes_ = np.asarray(self.estimator.classes_)
        if len(self.classes_) < 2:
            raise ValueError("instance pooling requires at least two training classes")
        return self

    def predict_proba(self, instances: Any, bag_offsets: np.ndarray) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("instance-pooling classifier must be fitted before prediction")
        _validate_bags(
            bag_offsets,
            n_instances=len(instances),
        )
        instance_probabilities = self.estimator.predict_proba(instances)
        offsets = np.asarray(bag_offsets, dtype=int)
        pooled = np.asarray(
            [
                self._pool(instance_probabilities[start:stop])
                for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
            ]
        )
        return pooled / pooled.sum(axis=1, keepdims=True)

    def predict(self, instances: Any, bag_offsets: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(instances, bag_offsets)
        return self.classes_[probabilities.argmax(axis=1)]


def _resolve_pooling(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name == "max":
        return lambda values: np.max(values, axis=0)
    if name == "mean":
        return lambda values: np.mean(values, axis=0)
    raise ValueError(f"instance pooling must be 'max' or 'mean', got {name!r}")


def _validate_bags(
    bag_offsets: np.ndarray,
    *,
    n_instances: int,
    n_bags: int | None = None,
) -> np.ndarray:
    offsets = np.asarray(bag_offsets, dtype=int)
    if offsets.ndim != 1:
        raise ValueError("bag offsets must be one-dimensional")
    if len(offsets) == 0 or offsets[0] != 0 or offsets[-1] != n_instances:
        raise ValueError("bag offsets must span all instance rows")
    if n_bags is not None and len(offsets) != n_bags + 1:
        raise ValueError("bag offsets length must equal target count plus one")
    bag_sizes = np.diff(offsets)
    if np.any(bag_sizes <= 0):
        raise ValueError("instance pooling does not support empty bags")
    return bag_sizes


__all__ = ["InstancePoolingClassifier"]
