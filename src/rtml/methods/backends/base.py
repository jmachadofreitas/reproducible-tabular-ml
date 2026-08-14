from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rtml.core.methods import MethodSpec
from rtml.core.results import PredictionSet
from rtml.core.runtime import RuntimeSpec

if TYPE_CHECKING:
    from rtml.core.benchmarks import BenchmarkCase


@dataclass
class BackendResult:
    """Backend-native output before run persistence and logging are applied."""

    predictions: PredictionSet
    metrics: dict[str, float] = field(default_factory=dict)
    fit_time: float | None = None
    predict_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendRefitResult:
    """Fitted method and native files produced by one backend refit."""

    fitted_method: Any
    artifact_paths: dict[str, Path]
    artifact_formats: dict[str, str]
    training_size: int
    input_schema: dict[str, Any]
    fit_time: float
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


class RefitBackend(Protocol):
    """Backend capability required by final method refitting."""

    name: str

    def refit(
        self,
        *,
        dataset: Any,
        task: Any,
        method: MethodSpec,
        artifact_dir: Path,
        seed: int = 0,
        runtime: RuntimeSpec | None = None,
        logger: Any | None = None,
    ) -> BackendRefitResult: ...

    def load_refit(
        self,
        *,
        artifact_dir: Path,
        manifest: Mapping[str, Any],
        runtime: RuntimeSpec | None = None,
    ) -> Any: ...
