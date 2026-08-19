from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from rtml.methods.engines.bundles import TorchModelBundle
from rtml.methods.engines.core import as_float32_array


class TorchFittedMethod:
    """A fitted single-instance method: preprocessing, model, and target mapping."""

    def __init__(
        self,
        *,
        preprocessor: Any,
        model_bundle: TorchModelBundle,
        source_columns: list[str],
        classes: np.ndarray | None,
        device: torch.device,
    ) -> None:
        self.preprocessor = preprocessor
        self.model_bundle = model_bundle
        self.source_columns = list(source_columns)
        self.classes = classes
        self.device = device

    @property
    def model(self) -> torch.nn.Module:
        return self.model_bundle.model

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        outputs = self._predict_outputs(data)
        if "y_pred" in outputs:
            return outputs["y_pred"].reshape(-1)
        labels = outputs["labels"].reshape(-1).astype(int)
        return labels if self.classes is None else self.classes[labels]

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        outputs = self._predict_outputs(data)
        if "probabilities" not in outputs:
            raise AttributeError("regression methods do not provide predict_proba")
        probabilities = outputs["probabilities"]
        if probabilities.ndim == 1 or probabilities.shape[1] == 1:
            positive = probabilities.reshape(-1)
            return np.column_stack([1.0 - positive, positive])
        return probabilities

    def _predict_outputs(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        if data.empty:
            raise ValueError("prediction data must contain at least one row")
        missing = [column for column in self.source_columns if column not in data.columns]
        if missing:
            raise ValueError(f"prediction data is missing required columns: {missing}")
        transformed = as_float32_array(
            self.preprocessor.transform(data.loc[:, self.source_columns])
        )
        batches = DataLoader(
            TensorDataset(torch.as_tensor(transformed)),
            batch_size=self.model_bundle.fit_config.batch_size,
            shuffle=False,
        )
        collected: dict[str, list[np.ndarray]] = {}
        for (features,) in batches:
            outputs = self.model_bundle.predict_batch(features.to(self.device))
            for name, value in outputs.items():
                collected.setdefault(name, []).append(value.detach().cpu().numpy())
        return {name: np.concatenate(values, axis=0) for name, values in collected.items()}
