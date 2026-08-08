from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.core import EvaluationStep, TrainingStep
from rtml.methods.engines.metrics import RunningMetrics

CreateTrainingStep = Callable[[torch.optim.Optimizer], TrainingStep]
RunningMetricsFactory = Callable[[], RunningMetrics]
PredictionStep = Callable[[Any], Mapping[str, torch.Tensor]]


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
        prediction_step: PredictionStep | None = None,
        train_metrics_factory: RunningMetricsFactory | None = None,
        validation_metrics_factory: RunningMetricsFactory | None = None,
        test_metrics_factory: RunningMetricsFactory | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.fit_config = fit_config
        self.evaluation_step = evaluation_step
        self.prediction_step = prediction_step
        self.metadata = dict(metadata or {})
        self._create_training_step = create_training_step
        self._train_metrics_factory = train_metrics_factory
        self._validation_metrics_factory = validation_metrics_factory
        self._test_metrics_factory = test_metrics_factory

    def create_training_step(self, optimizer: torch.optim.Optimizer) -> TrainingStep:
        return self._create_training_step(optimizer)

    def predict_batch(self, inputs: Any) -> Mapping[str, torch.Tensor]:
        """Run model-owned targetless prediction for one prepared batch."""
        if self.prediction_step is None:
            raise NotImplementedError("this torch model bundle has no targetless prediction step")
        return self.prediction_step(inputs)

    def make_train_metrics(self) -> RunningMetrics:
        return (
            RunningMetrics()
            if self._train_metrics_factory is None
            else self._train_metrics_factory()
        )

    def make_validation_metrics(self) -> RunningMetrics:
        if self._validation_metrics_factory is None:
            return RunningMetrics()
        return self._validation_metrics_factory()

    def make_test_metrics(self) -> RunningMetrics:
        return (
            RunningMetrics() if self._test_metrics_factory is None else self._test_metrics_factory()
        )
