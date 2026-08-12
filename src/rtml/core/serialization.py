"""JSON serialization shared by durable RTML artifacts."""

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def json_text(value: Any) -> str:
    """Serialize RTML values as readable, deterministic JSON text."""
    return json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n"


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
