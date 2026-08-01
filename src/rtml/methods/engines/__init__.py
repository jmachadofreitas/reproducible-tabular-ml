"""Reusable Torch/Ignite training and evaluation machinery."""

from rtml.methods.engines.checkpointing import (
    CheckpointManager,
    build_checkpoint_manager,
    checkpoint_directory,
    load_checkpoint,
)
from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.config import TorchFitConfig
from rtml.methods.engines.core import (
    EvaluationStep,
    Evaluator,
    TrainingStep,
    Trainer,
    concat_evaluator_output,
    default_prepare_batch,
    send_to_device,
)
from rtml.methods.engines.fitting import fit_model_bundle
from rtml.methods.engines.metrics import IgniteMetric, Metric, Metrics
from rtml.methods.engines.optim import (
    create_hp_scheduler,
    create_lr_scheduler,
    create_optimizer,
)
from rtml.methods.engines.runtime import resolve_device, seed_torch
from rtml.methods.engines.task_adapters import (
    create_loss_fn,
    create_torch_metrics,
    infer_output_dim,
    infer_score_mode,
    make_prediction_formatter,
    make_target_preparer,
    require_supervised_target,
    resolve_score_name,
    target_tensors,
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
    "build_checkpoint_manager",
    "checkpoint_directory",
    "concat_evaluator_output",
    "create_hp_scheduler",
    "create_loss_fn",
    "create_lr_scheduler",
    "create_optimizer",
    "create_torch_metrics",
    "default_prepare_batch",
    "fit_model_bundle",
    "infer_output_dim",
    "infer_score_mode",
    "load_checkpoint",
    "make_prediction_formatter",
    "make_target_preparer",
    "require_supervised_target",
    "resolve_device",
    "resolve_score_name",
    "seed_torch",
    "send_to_device",
    "target_tensors",
]
