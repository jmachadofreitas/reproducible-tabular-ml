from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class TorchFitConfig:
    """Torch training controls shared by torch method backends."""

    def __init__(
        self,
        *,
        batch_size: int = 32,
        max_epochs: int = 10,
        validation_fraction: float = 0.0,
        tracking: Mapping[str, Any] | None = None,
        optimizer: Mapping[str, Any] | None = None,
        lr_scheduler: Mapping[str, Any] | None = None,
        hp_scheduler: Mapping[str, Any] | None = None,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.validation_fraction = float(validation_fraction or 0.0)
        self.tracking = dict(tracking or {})
        self.every_n_epochs = int(self.tracking.get("every_n_epochs", 1))
        self.early_stopping_patience = self.tracking.get("early_stopping_patience")
        self.optimizer = dict(optimizer or {})
        self.lr_scheduler = None if lr_scheduler is None else dict(lr_scheduler)
        self.hp_scheduler = None if hp_scheduler is None else dict(hp_scheduler)
        self.checkpoint = dict(checkpoint or {})
        self._validate()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> TorchFitConfig:
        """Build torch fit controls from a plain config mapping."""
        values = dict(config or {})
        scheduler = values.pop("scheduler", None)
        fit = cls(
            batch_size=values.pop("batch_size", 32),
            max_epochs=values.pop("max_epochs", 10),
            validation_fraction=values.pop("validation_fraction", 0.0),
            tracking=values.pop("tracking", None),
            optimizer=values.pop("optimizer", None),
            lr_scheduler=values.pop("lr_scheduler", scheduler),
            hp_scheduler=values.pop("hp_scheduler", None),
            checkpoint=values.pop("checkpoint", None),
        )
        if values:
            unknown = ", ".join(sorted(values))
            raise ValueError(f"unknown torch fit config: {unknown}")
        return fit

    def _validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        if self.every_n_epochs < 1:
            raise ValueError("tracking.every_n_epochs must be >= 1")
        if self.validation_fraction and not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.early_stopping_patience is not None and not self.validation_fraction:
            raise ValueError("early_stopping_patience requires validation_fraction")

