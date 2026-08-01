from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.core import EvaluationStep, TrainingStep
from rtml.methods.engines.metrics import Metrics

CreateTrainingStep = Callable[[torch.optim.Optimizer], TrainingStep]
MetricsFactory = Callable[[], Metrics]


class TorchModelBundle:
    """Model-owned objects required by the generic Torch training engine."""

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
        self.evaluation_step = evaluation_step
        self.metadata = dict(metadata or {})
        self._create_training_step = create_training_step
        self._train_metrics_factory = train_metrics_factory
        self._validation_metrics_factory = validation_metrics_factory
        self._test_metrics_factory = test_metrics_factory

    def create_training_step(self, optimizer: torch.optim.Optimizer) -> TrainingStep:
        return self._create_training_step(optimizer)

    def make_evaluation_step(self) -> EvaluationStep:
        """Return the model-specific evaluation callable."""
        return self.evaluation_step

    def make_train_metrics(self) -> Metrics:
        return Metrics() if self._train_metrics_factory is None else self._train_metrics_factory()

    def make_validation_metrics(self) -> Metrics:
        if self._validation_metrics_factory is None:
            return Metrics()
        return self._validation_metrics_factory()

    def make_test_metrics(self) -> Metrics:
        return Metrics() if self._test_metrics_factory is None else self._test_metrics_factory()
