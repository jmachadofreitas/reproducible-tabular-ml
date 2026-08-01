from __future__ import annotations

import numpy as np
import torch

from rtml.core.runtime import RuntimeSpec


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
