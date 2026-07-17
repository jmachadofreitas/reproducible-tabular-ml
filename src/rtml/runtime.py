from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeSpec:
    """Backend-facing runtime settings and observed environment context.

    Runtime settings may affect how a method executes, for example device,
    precision, determinism, or thread count.

    Scheduler reservations live on `RunSpec.scheduler_resources` instead.
    """

    python_version: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    device: str | None = None
    accelerator: str | None = None
    precision: str | None = None
    deterministic: bool | None = None
    num_threads: int | None = None
    code_version: str | None = None
