from typing import Any

import torch
from torch import nn

from rtml.core.tasks import TaskSpec
from rtml.methods.engines.core import EvaluationStep, TrainingStep
from rtml.methods.engines.task_adapters import (
    make_prediction_output_formatter,
    make_target_preparer,
    make_train_evaluation_output_formatter,
)


def create_prediction_step(*, task: TaskSpec, model: nn.Module):
    """Compose targetless MLP prediction once for fitted-method inference."""
    format_predictions = make_prediction_output_formatter(task.task_type)

    def prediction_step(features: torch.Tensor) -> dict[str, torch.Tensor]:
        model.eval()
        with torch.inference_mode():
            return format_predictions(model(features.float()))

    return prediction_step


def create_training_step(
    *,
    task: TaskSpec,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> TrainingStep:
    """Compose the MLP training closure once, outside the batch loop."""
    prepare_target = make_target_preparer(task.task_type)
    format_predictions = make_train_evaluation_output_formatter(task.task_type)

    def training_step(batch: Any) -> dict[str, Any]:
        model.train()
        optimizer.zero_grad()

        x, y = batch
        y_target = prepare_target(y)
        logits = model(x.float())
        loss = loss_fn(logits, y_target)

        loss.backward()
        optimizer.step()

        output = format_predictions(logits.detach(), y_target.detach())
        output["loss"] = loss.item()
        return output

    return training_step


def create_evaluation_step(
    *,
    task: TaskSpec,
    model: nn.Module,
    loss_fn: nn.Module,
) -> EvaluationStep:
    """Compose the MLP evaluation closure once, outside the batch loop."""
    prepare_target = make_target_preparer(task.task_type)
    format_predictions = make_train_evaluation_output_formatter(task.task_type)

    def evaluation_step(batch: Any) -> dict[str, Any]:
        model.eval()
        with torch.inference_mode():
            x, y = batch
            y_target = prepare_target(y)
            logits = model(x.float())
            output = format_predictions(logits, y_target)
            output["loss"] = loss_fn(logits, y_target)
            return output

    return evaluation_step
