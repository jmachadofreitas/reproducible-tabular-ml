from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rtml.core.runs import RunRecord


class RunLogger(Protocol):
    """Adapter interface for external run logging backends."""

    def log_running_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        """Log metrics while a run is still executing."""
        ...

    def log_run(
        self,
        record: RunRecord,
        *,
        artifact_paths: Sequence[str | Path] = (),
    ) -> str | None:
        """Log final run information and return the backend run id when available."""
        ...
