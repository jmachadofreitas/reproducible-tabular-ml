"""Run-level records returned by RTML execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from rtml.methods.base import MethodSpec
from rtml.core.results import PredictionSet
from rtml.runtime import RuntimeSpec
from rtml.core.tasks import TaskType


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    case_name: str
    dataset_name: str
    dataset_fingerprint: str
    task_name: str
    task_type: TaskType
    primary_metric: str | None
    resampling_plan_fingerprint: str
    resample_id: str
    method: MethodSpec
    seed: int
    runtime: RuntimeSpec
    status: Literal["success", "failed"]
    metrics: dict[str, float] = field(default_factory=dict)
    fit_time: float | None = None
    predict_time: float | None = None
    prediction_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    predictions: PredictionSet | None
    record: RunRecord
