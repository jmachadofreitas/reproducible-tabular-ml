"""Binary MI-SVM heuristic for multiple-instance classification."""

from typing import Any

import numpy as np
from sklearn.svm import SVC


class MISVMClassifier:
    """Alternate SVM fitting with positive-bag witness selection.

    Positive bags are initially represented by their centroids. Each iteration
    fits an sklearn ``SVC`` using every negative instance and one witness from
    each positive bag, then synchronously replaces each witness with that bag's
    highest-scoring instance. Training stops at a fixed point, a repeated
    selection, or ``max_iterations``. Prediction uses the maximum instance
    decision score as the bag score.

    This is the practical standard-SVM heuristic, not the custom bag-slack QP
    solver used by some MI-SVM implementations.

    References:
        Andrews, Tsochantaridis, and Hofmann. "Support Vector Machines for
        Multiple-Instance Learning." NeurIPS, 2002.
        https://research.google/pubs/support-vector-machines-for-multiple-instance-learning/

        Doran and Ray. ``garydoranjr/misvm`` reference implementation.
        https://github.com/garydoranjr/misvm

        The ``mildsvm`` R implementation of the standard-SVM heuristic.
        https://cran.r-project.org/package=mildsvm
    """

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        random_state: int | None = None,
        **svc_params: Any,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.svc_params = svc_params

    def fit(
        self,
        instances: Any,
        bag_offsets: np.ndarray,
        bag_targets: Any,
    ) -> MISVMClassifier:
        features = np.asarray(instances)
        targets = np.asarray(bag_targets)
        offsets = _validate_training_data(features, bag_offsets, targets)

        self.classes_ = np.unique(targets)
        if len(self.classes_) != 2:
            raise ValueError(f"MI-SVM requires exactly two classes, got {len(self.classes_)}")

        negative_bags = np.flatnonzero(targets == self.classes_[0])
        positive_bags = np.flatnonzero(targets == self.classes_[1])
        if len(negative_bags) == 0 or len(positive_bags) == 0:
            raise ValueError("MI-SVM training requires positive and negative bags")

        negative_indices = _instance_indices(negative_bags, offsets)
        positive_centroids = _bag_centroids(features, positive_bags, offsets)
        initial_features = np.vstack((features[negative_indices], positive_centroids))
        initial_labels = np.concatenate(
            (
                -np.ones(len(negative_indices), dtype=int),
                np.ones(len(positive_centroids), dtype=int),
            )
        )
        initial_estimator = self._new_estimator().fit(
            initial_features,
            initial_labels,
        )
        witness_indices = _select_witnesses(
            initial_estimator.decision_function(features),
            positive_bags,
            offsets,
        )
        seen_selections = {tuple(witness_indices)}
        termination = "max_iterations"

        for iteration in range(1, self.max_iterations + 1):
            estimator = self._fit_estimator(
                features,
                negative_indices,
                witness_indices,
            )
            updated_witnesses = _select_witnesses(
                estimator.decision_function(features),
                positive_bags,
                offsets,
            )
            if np.array_equal(updated_witnesses, witness_indices):
                termination = "fixed_point"
                break
            selection = tuple(updated_witnesses)
            if selection in seen_selections:
                termination = "cycle"
                break
            seen_selections.add(selection)
            witness_indices = updated_witnesses

        if termination == "max_iterations":
            estimator = self._fit_estimator(
                features,
                negative_indices,
                witness_indices,
            )
        self.estimator_ = estimator
        self.witness_indices_ = witness_indices
        self.n_iterations_ = iteration
        self.termination_ = termination
        self.converged_ = termination == "fixed_point"
        return self

    def decision_function(
        self,
        instances: Any,
        bag_offsets: np.ndarray,
    ) -> np.ndarray:
        estimator = self._fitted_estimator()
        features = np.asarray(instances)
        offsets = _validate_prediction_data(features, bag_offsets)
        instance_scores = estimator.decision_function(features)
        return _max_by_bag(instance_scores, offsets)

    def predict(self, instances: Any, bag_offsets: np.ndarray) -> np.ndarray:
        bag_scores = self.decision_function(instances, bag_offsets)
        return self.classes_[(bag_scores >= 0).astype(int)]

    def _fit_estimator(
        self,
        features: np.ndarray,
        negative_indices: np.ndarray,
        witness_indices: np.ndarray,
    ) -> SVC:
        selected_features, selected_labels = _selected_training_data(
            features,
            negative_indices,
            witness_indices,
        )
        return self._new_estimator().fit(selected_features, selected_labels)

    def _new_estimator(self) -> SVC:
        return SVC(
            **self.svc_params,
            random_state=self.random_state,
        )

    def _fitted_estimator(self) -> SVC:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("MI-SVM must be fitted before prediction")
        return self.estimator_


def _validate_training_data(
    features: np.ndarray,
    bag_offsets: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("instance features must be a two-dimensional array")
    if targets.ndim != 1:
        raise ValueError("bag targets must be one-dimensional")
    return _validate_bag_offsets(
        bag_offsets,
        n_instances=len(features),
        n_bags=len(targets),
    )


def _validate_prediction_data(
    features: np.ndarray,
    bag_offsets: np.ndarray,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("instance features must be a two-dimensional array")
    return _validate_bag_offsets(bag_offsets, n_instances=len(features))


def _validate_bag_offsets(
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
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("MI-SVM does not support empty bags")
    return offsets


def _instance_indices(bag_positions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(offsets[bag], offsets[bag + 1]) for bag in bag_positions])


def _selected_training_data(
    features: np.ndarray,
    negative_indices: np.ndarray,
    witness_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.concatenate((negative_indices, witness_indices))
    labels = np.concatenate(
        (
            -np.ones(len(negative_indices), dtype=int),
            np.ones(len(witness_indices), dtype=int),
        )
    )
    return features[selected], labels


def _bag_centroids(
    features: np.ndarray,
    bag_positions: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    return np.vstack(
        [features[offsets[bag] : offsets[bag + 1]].mean(axis=0) for bag in bag_positions]
    )


def _select_witnesses(
    instance_scores: np.ndarray,
    positive_bags: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            offsets[bag] + np.argmax(instance_scores[offsets[bag] : offsets[bag + 1]])
            for bag in positive_bags
        ],
        dtype=int,
    )


def _max_by_bag(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.max(values[start:stop]) for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]
    )


__all__ = ["MISVMClassifier"]
