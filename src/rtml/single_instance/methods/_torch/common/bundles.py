from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rtml.methods.engines import EvaluationStep, Evaluator, Metrics, TorchFitConfig, TrainingStep

CreateTrainingStep = Callable[[torch.optim.Optimizer], TrainingStep]
MetricsFactory = Callable[[], Metrics]


class TensorDatasetBundle(NamedTuple):
    """Tensor datasets produced from one benchmark split."""

    train_dataset: TensorDataset
    test_dataset: TensorDataset
    y_train_tensor: torch.Tensor
    classes: np.ndarray | None
    input_dim: int


class DataLoaderBundle(NamedTuple):
    """Dataloaders and optional validation evaluator for one torch run."""

    train: DataLoader
    validation: DataLoader | None
    test: DataLoader
    validation_evaluator: Evaluator | None
    test_evaluator: Evaluator | None


class TorchModelBundle:
    """Torch-side model objects needed to train and evaluate one MethodSpec.

    The bundle is built after preprocessing determines the input shape. It
    keeps model-specific code out of TorchBackend while preprocessing remains
    owned by the single-instance backend.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        fit_config: TorchFitConfig,
        create_training_step: CreateTrainingStep,
        evaluation_step: EvaluationStep,
        train_metrics_factory: MetricsFactory | None = None,
        validation_metrics_factory: MetricsFactory | None = None,
        test_metrics_factory: MetricsFactory | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.fit_config = fit_config
        self.metadata = dict(metadata or {})
        self._create_training_step = create_training_step
        self._evaluation_step = evaluation_step
        self._train_metrics_factory = train_metrics_factory
        self._validation_metrics_factory = validation_metrics_factory
        self._test_metrics_factory = test_metrics_factory

    def create_training_step(self, optimizer: torch.optim.Optimizer) -> TrainingStep:
        """Create the training step after the backend has created the optimizer."""
        return self._create_training_step(optimizer)

    def make_evaluation_step(self) -> EvaluationStep:
        """Return the model-specific evaluation step."""
        return self._evaluation_step

    def make_train_metrics(self) -> Metrics:
        """Create fresh training metrics for one engine run."""
        if self._train_metrics_factory is None:
            return Metrics()
        return self._train_metrics_factory()

    def make_validation_metrics(self) -> Metrics:
        """Create fresh validation metrics for one engine run."""
        if self._validation_metrics_factory is None:
            return Metrics()
        return self._validation_metrics_factory()

    def make_test_metrics(self) -> Metrics:
        """Create fresh test metrics for one engine run."""
        if self._test_metrics_factory is None:
            return Metrics()
        return self._test_metrics_factory()
