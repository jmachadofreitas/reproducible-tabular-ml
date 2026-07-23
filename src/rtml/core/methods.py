from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    """Declarative model intent and selected implementation backend."""

    kind: str
    backend: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("model kind must be non-empty")
        if not self.backend:
            raise ValueError("model backend must be non-empty")
        self.params = dict(self.params or {})


@dataclass
class MethodSpec:
    """Declarative transform, model, fit, and metadata for one method."""

    name: str
    transform: dict[str, Any]
    model: ModelSpec
    fit: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("method name must be non-empty")
        self.transform = dict(self.transform or {})
        self.fit = dict(self.fit or {})
        self.metadata = dict(self.metadata or {})
        if not isinstance(self.model, ModelSpec):
            raise TypeError("method model must be a ModelSpec")
