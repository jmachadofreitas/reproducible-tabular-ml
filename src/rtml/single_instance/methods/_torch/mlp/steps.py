from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from rtml.core.tasks import TaskSpec
from rtml.methods.engines import EvaluationStep, TrainingStep
from rtml.single_instance.methods._torch.common.task_adapters import (
    make_prediction_formatter,
    make_target_preparer,
)

Batch = Any


def unpack_batch(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Support tuple batches and simple mapping batches in the same model code."""
    if isinstance(batch, Mapping):
        x = batch.get("x")
        y = batch.get("y")
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return x, y
        raise TypeError("mapping batches must provide tensor values for 'x' and 'y'")

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return x, y

    raise TypeError("dense tabular MLP batches must be (x, y) or {'x': x, 'y': y}")


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

    def training_step(batch: Batch) -> dict[str, Any]:
        model.train()
        optimizer.zero_grad()

        x, y = unpack_batch(batch)
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

    def evaluation_step(batch: Batch) -> dict[str, torch.Tensor]:
        model.eval()
        with torch.inference_mode():
            x, y = unpack_batch(batch)
            y_target = prepare_target(y)
            logits = model(x.float())
            output = format_predictions(logits, y_target)
            output["loss"] = loss_fn(logits, y_target)
            return output

    return evaluation_step
