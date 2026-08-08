"""Backend-neutral final fitting and artifact persistence."""

import hashlib
import json
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rtml.core.fingerprints import (
    fingerprint_dataset,
    fingerprint_method,
    fingerprint_task,
    stable_fingerprint,
    stable_jsonable,
)
from rtml.core.methods import MethodSpec
from rtml.core.runtime import RuntimeSpec, capture_runtime
from rtml.loggers import Logger
from rtml.methods.backends.base import BackendRefitResult, RefitBackend


@dataclass(frozen=True)
class RefitRecord:
    """Observed lineage and artifact details for one final method fit."""

    refit_id: str
    dataset_name: str
    task: Any
    method: MethodSpec
    seed: int
    fingerprints: dict[str, str]
    runtime: RuntimeSpec
    training_size: int
    input_schema: dict[str, Any]
    fit_time: float
    artifact_dir: str
    artifacts: dict[str, dict[str, Any]]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


def refit_method(
    *,
    dataset: Any,
    task: Any,
    method: MethodSpec,
    backend: RefitBackend,
    output_dir: str | Path,
    seed: int = 0,
    runtime: RuntimeSpec | None = None,
    logger: Logger | None = None,
) -> tuple[Any, RefitRecord]:
    """Fit a complete method and persist its backend-native artifacts."""
    _require_backend(method, backend)

    observed_runtime = capture_runtime(hints=runtime)
    fingerprints = {
        "dataset": fingerprint_dataset(dataset),
        "task": fingerprint_task(task),
        "method": fingerprint_method(method),
    }
    refit_digest = stable_fingerprint(
        {
            "fingerprints": fingerprints,
            "seed": seed,
        }
    ).removeprefix("sha256:")[:16]
    refit_id = f"{dataset.name}:{method.name}:sha256:{refit_digest}"

    root = Path(output_dir)
    artifact_dir = root / refit_digest
    if artifact_dir.exists():
        raise FileExistsError(f"refit artifact already exists: {artifact_dir}")
    root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{refit_digest}-", dir=root))

    run_context = (
        nullcontext()
        if logger is None
        else logger.start_run(run_name=f"refit/{dataset.name}/{method.name}")
    )
    try:
        with run_context:
            backend_result = backend.refit(
                dataset=dataset,
                task=task,
                method=method,
                artifact_dir=temporary_dir,
                seed=seed,
                runtime=runtime,
                logger=logger,
            )
            artifacts = _artifact_manifest(temporary_dir, backend_result)
            record = RefitRecord(
                refit_id=refit_id,
                dataset_name=dataset.name,
                task=task,
                method=method,
                seed=seed,
                fingerprints=fingerprints,
                runtime=observed_runtime,
                training_size=backend_result.training_size,
                input_schema=backend_result.input_schema,
                fit_time=backend_result.fit_time,
                artifact_dir=str(artifact_dir),
                artifacts=artifacts,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "backend": backend.name,
                    **backend_result.metadata,
                },
            )
            manifest_path = temporary_dir / "manifest.json"
            manifest = asdict(record)
            manifest.pop("artifact_dir")
            manifest_path.write_text(
                json.dumps(stable_jsonable(manifest), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if logger is not None:
                logger.log_metrics({"refit_time": record.fit_time})
                for path in (*backend_result.artifact_paths.values(), manifest_path):
                    logger.log_artifact(path, artifact_path="refit")
            temporary_dir.rename(artifact_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return backend_result.fitted_method, record


def load_refit(
    path: str | Path,
    *,
    backend: RefitBackend,
    runtime: RuntimeSpec | None = None,
) -> Any:
    """Verify and load a trusted refit artifact with its declared backend."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    method_backend = manifest["method"]["model"]["backend"]
    if method_backend != backend.name:
        raise ValueError(f"refit requires backend {method_backend!r}, received {backend.name!r}")

    artifact_dir = manifest_path.parent
    for name, artifact in manifest["artifacts"].items():
        artifact_path = _artifact_path(artifact_dir, artifact["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing refit artifact {name!r}: {artifact_path}")
        if artifact_path.stat().st_size != artifact["size"]:
            raise ValueError(f"refit artifact {name!r} has an unexpected size")
        if _file_hash(artifact_path) != artifact["sha256"]:
            raise ValueError(f"refit artifact {name!r} failed checksum verification")

    return backend.load_refit(
        artifact_dir=artifact_dir,
        manifest=manifest,
        runtime=runtime,
    )


def _require_backend(method: MethodSpec, backend: RefitBackend) -> None:
    if method.model.backend != backend.name:
        raise ValueError(
            f"method {method.name!r} requires backend {method.model.backend!r}, "
            f"received {backend.name!r}"
        )


def _artifact_manifest(
    artifact_dir: Path,
    result: BackendRefitResult,
) -> dict[str, dict[str, Any]]:
    if set(result.artifact_paths) != set(result.artifact_formats):
        raise ValueError("backend refit artifact paths and formats must have matching names")
    artifacts: dict[str, dict[str, Any]] = {}
    for name, path in result.artifact_paths.items():
        artifact_path = Path(path).resolve()
        relative_path = artifact_path.relative_to(artifact_dir.resolve())
        if not artifact_path.is_file():
            raise FileNotFoundError(f"backend did not produce refit artifact {name!r}")
        artifacts[name] = {
            "path": str(relative_path),
            "format": result.artifact_formats[name],
            "sha256": _file_hash(artifact_path),
            "size": artifact_path.stat().st_size,
        }
    if not artifacts:
        raise ValueError("backend refit produced no artifacts")
    return artifacts


def _artifact_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    path.relative_to(root.resolve())
    return path


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
