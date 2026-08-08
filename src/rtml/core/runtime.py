import platform as platform_module
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version

DEFAULT_RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "xgboost",
    "catboost",
    "lightgbm",
    "ray",
    "mlflow",
    "rtml",
)


@dataclass(frozen=True)
class RuntimeSpec:
    """Backend-facing runtime settings and observed environment context.

    Runtime settings may affect how a method executes, for example device,
    precision, determinism, or thread count.

    Scheduler reservations live on `RunSpec.scheduler_resources` instead.
    """

    python_version: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    platform: str | None = None
    device: str | None = None
    accelerator: str | None = None
    precision: str | None = None
    deterministic: bool | None = None
    num_threads: int | None = None
    code_version: str | None = None


def capture_runtime(
    *,
    packages: tuple[str, ...] = DEFAULT_RUNTIME_PACKAGES,
    code_version: str | None = None,
    hints: RuntimeSpec | None = None,
) -> RuntimeSpec:
    """Capture environment evidence and retain backend-facing runtime hints."""
    recorded_code_version = code_version
    if recorded_code_version is None and hints is not None:
        recorded_code_version = hints.code_version
    package_versions: dict[str, str] = {}
    for package in packages:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return RuntimeSpec(
        python_version=sys.version.split()[0],
        package_versions=package_versions,
        platform=platform_module.platform(),
        device=None if hints is None else hints.device,
        accelerator=None if hints is None else hints.accelerator,
        precision=None if hints is None else hints.precision,
        deterministic=None if hints is None else hints.deterministic,
        num_threads=None if hints is None else hints.num_threads,
        code_version=recorded_code_version,
    )
