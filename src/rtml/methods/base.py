from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MethodSpec:
    """Declarative transform, model, and training configuration for one method."""

    name: str
    transform: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("method name must be non-empty")
        self.transform = dict(self.transform or {})
        self.model = dict(self.model or {})
        self.training = dict(self.training or {})
        self.metadata = dict(self.metadata or {})
