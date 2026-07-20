from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from rtml.core.results import PredictionSet
from rtml.core.methods import MethodSpec
from rtml.core.runtime import RuntimeSpec

if TYPE_CHECKING:
    from rtml.core.benchmarks import BenchmarkCase


@dataclass(frozen=True)
class BackendResult:
    """Backend-native output before run persistence and logging are applied."""

    predictions: PredictionSet
    metrics: dict[str, float] = field(default_factory=dict)
    fit_time: float | None = None
    predict_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MethodBackend(Protocol):
    name: str

    def validate_method(self, method: MethodSpec) -> None:
        """Reject methods this backend cannot execute."""
        ...

    def run(
        self,
        *,
        case: BenchmarkCase,
        method: MethodSpec,
        resample_id: str | None = None,
        seed: int = 0,
        runtime: RuntimeSpec | None = None,
        logger: Any | None = None,
    ) -> BackendResult:
        """Execute one method on one benchmark case/resample."""
        ...
