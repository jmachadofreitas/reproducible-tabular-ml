"""Stable fingerprints for RTML experiment evidence."""

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


def stable_jsonable(value: Any) -> Any:
    """Normalize common RTML values into deterministic JSON-compatible data."""
    if is_dataclass(value) and not isinstance(value, type):
        return stable_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): stable_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [stable_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [stable_jsonable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, np.ndarray):
        return stable_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return stable_jsonable(value.item())
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def stable_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for normalized JSON-compatible data."""
    payload = json.dumps(
        stable_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def fingerprint_dataset(dataset: Any) -> str:
    """Fingerprint dataset identity, schema, shape, and provenance metadata.

    This intentionally avoids hashing all dataframe values. Full content hashes
    can be added later as explicit dataset metadata when the cost is acceptable.
    """
    existing = getattr(dataset, "metadata", {}).get("fingerprint")
    if existing:
        return str(existing)
    return stable_fingerprint(dataset.fingerprint_payload())


def fingerprint_task(task: Any) -> str:
    """Fingerprint the task definition."""
    return stable_fingerprint(task)


def fingerprint_method(method: Any) -> str:
    """Fingerprint the complete method definition."""
    return stable_fingerprint(
        {
            "name": method.name,
            "transform": method.transform,
            "model": method.model,
            "fit": method.fit,
        }
    )


def fingerprint_runtime(runtime: Any) -> str:
    """Fingerprint the runtime context recorded for a run."""
    return stable_fingerprint(runtime)
