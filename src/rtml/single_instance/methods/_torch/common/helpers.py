from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rtml.core.benchmarks import BenchmarkCase
from rtml.core.runtime import RuntimeSpec
from rtml.core.tasks import TaskSpec, TaskType


def seed_torch(seed: int, *, deterministic: bool | None = None) -> torch.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic is not None:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(runtime: RuntimeSpec | None) -> torch.device:
    if runtime is not None and runtime.device:
        return torch.device(runtime.device)
    if runtime is not None and runtime.accelerator in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def as_float32_array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


def require_supervised_target(case: BenchmarkCase) -> Any:
    y = case.task.target_series(case.dataset)
    if y is None:
        raise ValueError("torch method execution requires a supervised task target")
    return y


def encode_classification_target(
    y_train: Any, y_eval: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(np.unique(y_train.to_numpy()).tolist()))
    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    try:
        train_encoded = np.asarray([class_to_index[label] for label in y_train.to_numpy()])
        eval_encoded = np.asarray([class_to_index[label] for label in y_eval.to_numpy()])
    except KeyError as exc:
        raise ValueError(
            f"target contains class not present in training split: {exc.args[0]!r}"
        ) from exc
    return train_encoded, eval_encoded, classes


def target_tensors(
    *,
    task: TaskSpec,
    y_train: Any,
    y_eval: Any,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray | None]:
    if task.task_type == TaskType.REGRESSION:
        return (
            torch.as_tensor(y_train.to_numpy(dtype=np.float32)).reshape(-1, 1),
            torch.as_tensor(y_eval.to_numpy(dtype=np.float32)).reshape(-1, 1),
            None,
        )
    if task.task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
        train_encoded, eval_encoded, classes = encode_classification_target(y_train, y_eval)
        return torch.as_tensor(train_encoded), torch.as_tensor(eval_encoded), classes
    raise ValueError(f"unsupported torch task type: {task.task_type.value}")
