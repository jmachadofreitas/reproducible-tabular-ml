from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionSet:
    """Predictions for evaluation samples from one method on one resample."""

    dataset_name: str
    task_name: str
    method_name: str
    resample_id: str

    sample_ids: np.ndarray
    y_true: np.ndarray | None = None

    labels: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    scores: np.ndarray | None = None
    values: np.ndarray | None = None

    subgroups: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
