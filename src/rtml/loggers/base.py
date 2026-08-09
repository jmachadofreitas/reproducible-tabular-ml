import sys
import warnings
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from rtml.core.runs import RunRecord


class LogWriter(Protocol):
    """Destination adapter used by `Logger`.

    Implement this protocol to send RTML run information to a concrete system
    such as MLflow, TensorBoard, CSV, or JSON.
    """

    def start_run(self, *, run_name: str | None = None) -> Any:
        """Open one destination run context."""
        ...

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        """Log already-computed metrics."""
        ...

    def log_run(
        self,
        record: RunRecord,
        *,
        artifact_paths: Sequence[str | Path] = (),
    ) -> str | None:
        """Log final run information and return the destination run id when available."""
        ...

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        """Log one artifact while a run is still executing."""
        ...


class Logger:
    """Project-facing logger that fans out to one or more writers.

    `Logger` owns no metric computation and no destination-specific logic. Runs
    open the per-run context and log final records; engines may log intermediate
    training metrics and artifacts while that context is active. Use
    `rtml.loggers.build_logger` at config boundaries such as Hydra entrypoints
    or Ray workers; keep this constructor for explicit writer composition.
    """

    def __init__(self, writers: LogWriter | Sequence[LogWriter]) -> None:
        if isinstance(writers, Sequence):
            self.writers = list(writers)
        else:
            self.writers = [writers]
        if not self.writers:
            raise ValueError("Logger requires at least one log writer")

    @contextmanager
    def start_run(self, *, run_name: str | None = None) -> Any:
        """Open one logical run across every writer."""
        with ExitStack() as stack:
            active_runs = [
                stack.enter_context(_writer_run(writer, run_name=run_name))
                for writer in self.writers
            ]
            yield active_runs

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: int | None = None,
    ) -> None:
        for writer in self.writers:
            try:
                writer.log_metrics(metrics, step=step)
            except Exception as exc:
                _warn_writer_failure(writer, "log metrics", exc)

    def log_run(
        self,
        record: RunRecord,
        *,
        artifact_paths: Sequence[str | Path] = (),
    ) -> list[str | None]:
        run_ids = []
        for writer in self.writers:
            try:
                run_ids.append(writer.log_run(record, artifact_paths=artifact_paths))
            except Exception as exc:
                _warn_writer_failure(writer, "log final run", exc)
                run_ids.append(None)
        return run_ids

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        for writer in self.writers:
            try:
                writer.log_artifact(path, artifact_path=artifact_path)
            except Exception as exc:
                _warn_writer_failure(writer, "log artifact", exc)


@contextmanager
def _writer_run(writer: LogWriter, *, run_name: str | None) -> Any:
    try:
        context = writer.start_run(run_name=run_name)
        active_run = context.__enter__()
    except Exception as exc:
        _warn_writer_failure(writer, "start run", exc)
        yield None
        return

    try:
        yield active_run
    except BaseException:
        try:
            context.__exit__(*sys.exc_info())
        except Exception as exc:
            _warn_writer_failure(writer, "end run", exc)
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except Exception as exc:
            _warn_writer_failure(writer, "end run", exc)


def _warn_writer_failure(writer: LogWriter, operation: str, error: Exception) -> None:
    warnings.warn(
        f"{type(writer).__name__} failed to {operation}: {error}",
        RuntimeWarning,
        stacklevel=3,
    )
