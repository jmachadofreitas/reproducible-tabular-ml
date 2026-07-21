from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rtml.loggers.base import Logger
from rtml.loggers.mlflow import MLflowWriter


def build_logger(config: Mapping[str, Any] | None) -> Logger | None:
    """Build a logger from plain config at Hydra/Ray boundaries.

    `Logger.__init__` composes already-built writers; this helper keeps config
    parsing and concrete writer selection out of the runtime logger object.
    """
    config = config or {}
    backend = config.get("backend", "none")
    if backend in {None, "none"}:
        return None
    if backend == "mlflow":
        return Logger(
            MLflowWriter(
                experiment_name=config.get("experiment_name"),
                tracking_uri=config.get("tracking_uri"),
                artifact_subdir=config.get("artifact_subdir"),
            )
        )
    raise ValueError(f"unsupported logger backend {backend!r}")
