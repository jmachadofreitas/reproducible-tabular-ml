"""Public run execution API.

Use `run_method(...)` for one method on one benchmark case, `run_study(...)` for
method-comparison studies, and `run_suite(...)` when a caller only has a suite
plus methods and does not need to name a study explicitly.
"""

from rtml.runs.base import RunRecord, RunResult, RuntimeSpec
from rtml.runs.execution import (
    RayExecutor,
    RunExecutor,
    SequentialExecutor,
    run_method,
    run_execution_plan_ray,
    run_execution_plan_sequential,
    run_study,
    run_suite,
)
from rtml.runs.plan import ExecutionResources, RunSpec, ExecutionPlan

__all__ = [
    "RayExecutor",
    "ExecutionResources",
    "RunSpec",
    "RunExecutor",
    "ExecutionPlan",
    "RunRecord",
    "RunResult",
    "SequentialExecutor",
    "RuntimeSpec",
    "run_method",
    "run_execution_plan_ray",
    "run_execution_plan_sequential",
    "run_study",
    "run_suite",
]
