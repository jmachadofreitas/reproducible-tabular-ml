"""Reusable Torch/Ignite training and evaluation machinery."""

from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.checkpointing import CheckpointManager
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.core import (
    EvaluationStep,
    Evaluator,
    Trainer,
    TrainingStep,
)
from rtml.methods.engines.fitting import fit_model_bundle
from rtml.methods.engines.metrics import IgniteMetric, Metric, Metrics
from rtml.methods.engines.optim import (
    create_hp_scheduler,
    create_lr_scheduler,
    create_optimizer,
)

__all__ = [
    "CheckpointManager",
    "EvaluationStep",
    "Evaluator",
    "IgniteMetric",
    "Metric",
    "Metrics",
    "TorchFitConfig",
    "TorchModelBundle",
    "TrainingStep",
    "Trainer",
    "create_hp_scheduler",
    "create_lr_scheduler",
    "create_optimizer",
    "fit_model_bundle",
]
