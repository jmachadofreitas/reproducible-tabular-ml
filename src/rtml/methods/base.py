from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


COMPATIBLE_MODEL_BACKENDS: dict[str, set[str]] = {
    "dummy": {"sklearn"},
    "gradient_boosting": {"sklearn"},
    "linear_regression": {"sklearn"},
    "logistic_regression": {"sklearn"},
    "ridge": {"sklearn"},
    "random_forest": {"sklearn"},
    "boosted_trees": {"catboost", "sklearn", "xgboost"},
}


@dataclass
class ModelSpec:
    """Generic model intent plus the backend implementation that runs it."""

    kind: str
    backend: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("model kind must be non-empty")
        if not self.backend:
            raise ValueError("model backend must be non-empty")
        self.params = dict(self.params or {})
        supported_backends = COMPATIBLE_MODEL_BACKENDS.get(self.kind)
        if supported_backends is None:
            raise ValueError(f"unsupported model kind {self.kind!r}")
        if self.backend not in supported_backends:
            supported = ", ".join(sorted(supported_backends))
            raise ValueError(
                f"backend {self.backend!r} does not support model kind {self.kind!r}; "
                f"supported backends: {supported}"
            )


@dataclass
class MethodSpec:
    """Declarative transform, model, and training configuration for one method."""

    name: str
    transform: dict[str, Any]
    model: ModelSpec
    training: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("method name must be non-empty")
        self.transform = dict(self.transform or {})
        self.training = dict(self.training or {})
        self.metadata = dict(self.metadata or {})
        if not isinstance(self.model, ModelSpec):
            raise TypeError("method model must be a ModelSpec")
