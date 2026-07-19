"""Public run execution API.

Use `run_method(...)` for one method on one benchmark case, `run_study(...)` for
method-comparison studies, and `run_suite(...)` when a caller only has a suite
plus methods and does not need to name a study explicitly.
"""

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

__all__ = [
    "RayExecutor",
    "RunExecutor",
    "SequentialExecutor",
    "run_method",
    "run_execution_plan_ray",
    "run_execution_plan_sequential",
    "run_study",
    "run_suite",
]
