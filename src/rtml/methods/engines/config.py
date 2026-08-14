from collections.abc import Mapping
from typing import Any


class TorchFitConfig:
    """Torch training controls shared by torch method backends."""

    def __init__(
        self,
        *,
        batch_size: int = 32,
        max_epochs: int = 10,
        validation_every_n_epochs: int = 1,
        early_stopping_patience: int | None = None,
        tracking: Mapping[str, Any] | None = None,
        optimizer: Mapping[str, Any] | None = None,
        lr_scheduler: Mapping[str, Any] | None = None,
        hp_scheduler: Mapping[str, Any] | None = None,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.validation_every_n_epochs = int(validation_every_n_epochs)
        self.early_stopping_patience = (
            None if early_stopping_patience is None else int(early_stopping_patience)
        )
        self.tracking = dict(tracking or {})
        self.optimizer = dict(optimizer or {})
        self.lr_scheduler = None if lr_scheduler is None else dict(lr_scheduler)
        self.hp_scheduler = None if hp_scheduler is None else dict(hp_scheduler)
        self.checkpoint = dict(checkpoint or {})
        self._validate()

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> TorchFitConfig:
        """Build torch fit controls from a plain config mapping."""
        values = dict(config or {})
        fit = cls(
            batch_size=values.pop("batch_size", 32),
            max_epochs=values.pop("max_epochs", 10),
            validation_every_n_epochs=values.pop("validation_every_n_epochs", 1),
            early_stopping_patience=values.pop("early_stopping_patience", None),
            tracking=values.pop("tracking", None),
            optimizer=values.pop("optimizer", None),
            lr_scheduler=values.pop("lr_scheduler", None),
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
        if self.validation_every_n_epochs < 1:
            raise ValueError("validation_every_n_epochs must be >= 1")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1")
        unknown_tracking = sorted(set(self.tracking) - {"log_test_metrics"})
        if unknown_tracking:
            raise ValueError(f"unknown torch tracking config: {', '.join(unknown_tracking)}")
