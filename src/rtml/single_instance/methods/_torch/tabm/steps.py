from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from rtml.core.tasks import TaskSpec, TaskType
from rtml.methods.engines.core import EvaluationStep, TrainingStep
from rtml.methods.engines.task_adapters import make_target_preparer

MemberLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
PredictionFormatter = Callable[[torch.Tensor, torch.Tensor], dict[str, Any]]


def make_ensemble_member_loss(task: TaskSpec, loss_fn: nn.Module) -> MemberLoss:
    """Compose the loss applied independently to every TabM member."""
    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:

        def multiclass_loss(member_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return loss_fn(
                member_logits.flatten(0, 1), target.repeat_interleave(member_logits.shape[1])
            )

        return multiclass_loss

    if task.task_type in {TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION}:

        def value_loss(member_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return loss_fn(member_logits, target.unsqueeze(1).expand_as(member_logits))

        return value_loss

    raise ValueError(f"unsupported TabM task type: {task.task_type.value}")


def make_ensemble_prediction_formatter(task: TaskSpec) -> PredictionFormatter:
    """Compose one task-specific formatter for averaged TabM predictions."""
    if task.task_type == TaskType.REGRESSION:

        def format_regression(member_logits: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
            predictions = member_logits.mean(dim=1)
            return {"y_pred": predictions, "y": target, "mse": (predictions, target)}

        return format_regression

    if task.task_type == TaskType.BINARY_CLASSIFICATION:

        def format_binary(member_logits: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
            probabilities = torch.sigmoid(member_logits).mean(dim=1)
            labels = (probabilities >= 0.5).long()
            scores = torch.logit(probabilities, eps=torch.finfo(probabilities.dtype).eps)
            return {
                "logits": scores,
                "probabilities": probabilities,
                "labels": labels,
                "y": target.long(),
                "accuracy": (labels.reshape(-1), target.reshape(-1)),
            }

        return format_binary

    if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:

        def format_multiclass(member_logits: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
            probabilities = torch.softmax(member_logits, dim=-1).mean(dim=1)
            labels = probabilities.argmax(dim=1)
            scores = probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log()
            return {
                "logits": scores,
                "probabilities": probabilities,
                "labels": labels,
                "y": target,
                "accuracy": (labels, target),
            }

        return format_multiclass

    raise ValueError(f"unsupported TabM task type: {task.task_type.value}")


def create_training_step(
    *,
    task: TaskSpec,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> TrainingStep:
    """Compose a training closure that optimizes each TabM member independently."""
    prepare_target = make_target_preparer(task.task_type)
    member_loss = make_ensemble_member_loss(task, loss_fn)
    format_predictions = make_ensemble_prediction_formatter(task)

    def training_step(batch: Any) -> dict[str, Any]:
        model.train()
        optimizer.zero_grad()

        x, y = batch
        target = prepare_target(y)
        member_logits = model(x.float())
        loss = member_loss(member_logits, target)
        loss.backward()
        optimizer.step()

        output = format_predictions(member_logits.detach(), target.detach())
        output["loss"] = loss.item()
        return output

    return training_step


def create_evaluation_step(
    *,
    task: TaskSpec,
    model: nn.Module,
    loss_fn: nn.Module,
) -> EvaluationStep:
    """Compose an evaluation closure that averages TabM members for reporting."""
    prepare_target = make_target_preparer(task.task_type)
    member_loss = make_ensemble_member_loss(task, loss_fn)
    format_predictions = make_ensemble_prediction_formatter(task)

    def evaluation_step(batch: Any) -> dict[str, Any]:
        model.eval()
        with torch.inference_mode():
            x, y = batch
            target = prepare_target(y)
            member_logits = model(x.float())
            output = format_predictions(member_logits, target)
            output["loss"] = member_loss(member_logits, target)
            return output

    return evaluation_step
