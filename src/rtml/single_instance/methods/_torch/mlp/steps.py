from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rtml.core.tasks import TaskSpec
from rtml.methods.engines import EvaluationStep, TrainingStep
from rtml.single_instance.methods._torch.common.task_adapters import (
    make_prediction_formatter,
    make_target_preparer,
)


def create_training_step(
    *,
    task: TaskSpec,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> TrainingStep:
    """Compose the MLP training closure once, outside the batch loop."""
    prepare_target = make_target_preparer(task)
    format_predictions = make_prediction_formatter(task)

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
    prepare_target = make_target_preparer(task)
    format_predictions = make_prediction_formatter(task)

    def evaluation_step(batch: Any) -> dict[str, torch.Tensor]:
        model.eval()
        with torch.inference_mode():
            x, y = batch
            y_target = prepare_target(y)
            logits = model(x.float())
            output = format_predictions(logits, y_target)
            output["loss"] = loss_fn(logits, y_target)
            return output

    return evaluation_step
