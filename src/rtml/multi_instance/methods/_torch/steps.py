from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rtml.core.tasks import TaskType
from rtml.methods.engines.core import EvaluationStep, TrainingStep
from rtml.methods.engines.task_adapters import (
    make_target_preparer,
    make_train_evaluation_output_formatter,
)


def create_training_step(
    *,
    task_type: TaskType,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> TrainingStep:
    """Compose a bag-level training closure once for one supervised task."""
    prepare_target = make_target_preparer(task_type)
    format_predictions = make_train_evaluation_output_formatter(task_type)

    def training_step(batch: Any) -> dict[str, Any]:
        model.train()
        optimizer.zero_grad()

        instances, mask, target = batch
        prepared_target = prepare_target(target)
        logits = model(instances.float(), mask)
        loss = loss_fn(logits, prepared_target)
        loss.backward()
        optimizer.step()

        output = format_predictions(logits.detach(), prepared_target.detach())
        output["loss"] = loss.item()
        return output

    return training_step


def create_evaluation_step(
    *,
    task_type: TaskType,
    model: nn.Module,
    loss_fn: nn.Module,
) -> EvaluationStep:
    """Compose a bag-level evaluation closure once for one supervised task."""
    prepare_target = make_target_preparer(task_type)
    format_predictions = make_train_evaluation_output_formatter(task_type)

    def evaluation_step(batch: Any) -> dict[str, Any]:
        model.eval()
        with torch.inference_mode():
            instances, mask, target = batch
            prepared_target = prepare_target(target)
            logits = model(instances.float(), mask)
            output = format_predictions(logits, prepared_target)
            output["loss"] = loss_fn(logits, prepared_target)
            return output

    return evaluation_step
