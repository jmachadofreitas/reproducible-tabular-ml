from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from rtml.methods.base import MethodSpec
from rtml.results.base import PredictionSet

if TYPE_CHECKING:
    from rtml.benchmarks.base import BenchmarkCase


@dataclass(frozen=True)
class BackendResult:
    """Backend-native output before run persistence and logging are applied."""

    predictions: PredictionSet
    metrics: dict[str, float] = field(default_factory=dict)
    fit_time: float | None = None
    predict_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MethodBackend(Protocol):
    """Execution backend for one family of method implementations."""

    name: str

    def can_run(self, method: MethodSpec) -> bool:
        """Return whether this backend can execute the method specification."""
        ...

    def run(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample_id: str | None = None,
        seed: int = 0,
    ) -> BackendResult:
        """Execute one method on one benchmark case/resample."""
        ...
