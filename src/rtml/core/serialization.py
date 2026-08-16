"""JSON encoding for values commonly stored in RTML artifacts."""

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


class JSONEncoder(json.JSONEncoder):
    """Encode the scientific Python values used in RTML metadata and artifacts."""

    def default(self, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.ndim == 0 else tensor.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return value.to_list()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="list")
        if isinstance(value, ListConfig | DictConfig):
            return OmegaConf.to_object(value)
        if isinstance(value, set | frozenset):
            return sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        return super().default(value)
