"""Download and cache helpers for the classic WEKA MIL archive."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import shutil
import zipfile

from rtml.multi_instance.datasets.classic.constants import (
    CLASSIC_MIL_ARCHIVE_NAME,
    CLASSIC_MIL_ARCHIVE_ROOT,
    CLASSIC_MIL_ARCHIVE_SHA256,
    CLASSIC_MIL_ARCHIVE_SIZE_BYTES,
    CLASSIC_MIL_ARCHIVE_URLS,
    CLASSIC_MIL_DATASETS,
    DEFAULT_CLASSIC_MIL_DATA_DIR,
)


def classic_mil_cache_dir(root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR) -> Path:
    """Return the local cache directory for the classic MIL archive."""
    return Path(root).expanduser().resolve()


def download_classic_mil_archive(
    root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR,
    *,
    force: bool = False,
    urls: Sequence[str] = CLASSIC_MIL_ARCHIVE_URLS,
) -> Path:
    """Download the classic MIL archive into the local cache."""
    cache_dir = classic_mil_cache_dir(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / CLASSIC_MIL_ARCHIVE_NAME

    if archive_path.exists() and not force:
        validate_classic_mil_archive(archive_path)
        return archive_path

    temporary_path = archive_path.with_suffix(".zip.part")
    errors: list[str] = []
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": "rtml/0.1"})
            with urlopen(request, timeout=120) as response, temporary_path.open("wb") as file:
                shutil.copyfileobj(response, file)

            validate_classic_mil_archive(temporary_path)
            temporary_path.replace(archive_path)
            return archive_path
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")

    temporary_path.unlink(missing_ok=True)
    raise RuntimeError("failed to download classic MIL archive:\n" + "\n".join(errors))


def validate_classic_mil_archive(archive_path: str | Path) -> None:
    """Validate archive integrity and expected dataset entries."""
    path = Path(archive_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size != CLASSIC_MIL_ARCHIVE_SIZE_BYTES:
        raise ValueError(f"classic MIL archive has unexpected size: {path.stat().st_size} bytes")
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != CLASSIC_MIL_ARCHIVE_SHA256:
        raise ValueError(f"classic MIL archive has unexpected sha256: {digest}")

    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())

    expected = {
        f"{CLASSIC_MIL_ARCHIVE_ROOT}/{name}_relational.arff" for name in CLASSIC_MIL_DATASETS
    }
    missing = sorted(expected - members)
    if missing:
        raise ValueError(f"classic MIL archive is missing expected files: {missing}")


def extract_classic_mil_archive(
    root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR,
    *,
    archive_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Extract the classic MIL archive and return the extracted dataset directory."""
    cache_dir = classic_mil_cache_dir(root)
    archive = (
        Path(archive_path).expanduser().resolve()
        if archive_path is not None
        else download_classic_mil_archive(cache_dir)
    )
    validate_classic_mil_archive(archive)

    dataset_dir = cache_dir / CLASSIC_MIL_ARCHIVE_ROOT
    if dataset_dir.exists() and not force:
        _validate_extracted_files(dataset_dir)
        return dataset_dir

    staging_dir = cache_dir / f".{CLASSIC_MIL_ARCHIVE_ROOT}.extracting"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zip_file:
        _safe_extract(zip_file, staging_dir)

    staged_dataset_dir = staging_dir / CLASSIC_MIL_ARCHIVE_ROOT
    _validate_extracted_files(staged_dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    shutil.move(str(staged_dataset_dir), str(dataset_dir))
    shutil.rmtree(staging_dir)
    _validate_extracted_files(dataset_dir)
    return dataset_dir


def ensure_classic_mil_data(
    root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR,
    *,
    force_download: bool = False,
    force_extract: bool = False,
) -> Path:
    """Download and extract the classic MIL archive when needed."""
    cache_dir = classic_mil_cache_dir(root)
    dataset_dir = cache_dir / CLASSIC_MIL_ARCHIVE_ROOT
    if dataset_dir.exists() and not force_download and not force_extract:
        _validate_extracted_files(dataset_dir)
        return dataset_dir

    archive_path = download_classic_mil_archive(root, force=force_download)
    return extract_classic_mil_archive(
        root,
        archive_path=archive_path,
        force=force_extract,
    )


def classic_mil_arff_path(
    dataset_name: str,
    root: str | Path = DEFAULT_CLASSIC_MIL_DATA_DIR,
) -> Path:
    """Return the extracted ARFF path for one classic MIL dataset."""
    normalized_name = dataset_name.lower().replace("-", "_")
    if normalized_name not in CLASSIC_MIL_DATASETS:
        valid = ", ".join(CLASSIC_MIL_DATASETS)
        raise ValueError(f"unknown classic MIL dataset {dataset_name!r}; valid: {valid}")

    dataset_dir = ensure_classic_mil_data(root)
    filename = f"{normalized_name}_relational.arff"
    path = dataset_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _validate_extracted_files(dataset_dir: Path) -> None:
    missing = [
        f"{name}_relational.arff"
        for name in CLASSIC_MIL_DATASETS
        if not (dataset_dir / f"{name}_relational.arff").exists()
    ]
    if missing:
        raise ValueError(f"classic MIL dataset directory is missing files: {missing}")


def _safe_extract(zip_file: zipfile.ZipFile, target_dir: Path) -> None:
    root = target_dir.resolve()
    for member in zip_file.infolist():
        destination = (target_dir / member.filename).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"archive member escapes target directory: {member.filename}")
    zip_file.extractall(target_dir)
