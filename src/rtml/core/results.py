from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
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

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = np.asarray(self.sample_ids)
        if sample_ids.ndim != 1:
            raise ValueError("sample_ids must be one-dimensional")
        self.sample_ids = sample_ids

        sample_count = len(sample_ids)
        for name in ("y_true", "labels", "probabilities", "scores", "values"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim == 0 or len(array) != sample_count:
                raise ValueError(
                    f"{name} must have first-axis length {sample_count}, got shape {array.shape}"
                )
            setattr(self, name, array)

        self.metadata = dict(self.metadata or {})
