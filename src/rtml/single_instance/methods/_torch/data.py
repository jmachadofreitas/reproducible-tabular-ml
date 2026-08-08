from typing import Any, NamedTuple

import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def as_float32_array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32)


class TensorDatasetBundle(NamedTuple):
    """Tensor datasets produced from one benchmark split."""

    train: TensorDataset
    validation: TensorDataset | None
    test: TensorDataset
    classes: np.ndarray | None
    input_dim: int


class DataLoaderBundle(NamedTuple):
    """Dataloaders produced for one single-instance Torch run."""

    train: DataLoader
    validation: DataLoader | None
    test: DataLoader
