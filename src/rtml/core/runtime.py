import platform as platform_module
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

DEFAULT_ENVIRONMENT_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "pytorch-ignite",
    "tabm",
    "openml",
    "ray",
    "mlflow",
    "rtml",
)


@dataclass(frozen=True)
class RuntimeSpec:
    """Backend-facing hints that may change how a method executes.

    Scheduler reservations and observed environment information are separate
    concerns: they live on `RunSpec.scheduler_resources` and the resulting
    `RunRecord.environment`, respectively.
    """

    device: str | None = None
    deterministic: bool | None = None
    num_threads: int | None = None


def capture_environment(
    *,
    packages: tuple[str, ...] = DEFAULT_ENVIRONMENT_PACKAGES,
) -> dict[str, Any]:
    """Capture observed software and platform information for a result."""
    package_versions: dict[str, str] = {}
    for package in packages:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return {
        "python_version": sys.version.split()[0],
        "platform": platform_module.platform(),
        "packages": package_versions,
    }
