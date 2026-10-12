"""Dataset ingestion, validation, and content-integrity utilities.

Responsibilities:

* read and write JSONL manifests,
* validate manifests as a whole (not just record-by-record),
* compute SHA-256 digests by streaming, so multi-gigabyte video never lands in memory,
* verify that a sample's file still matches its recorded digest,
* hash a manifest so a run record can pin exactly which manifest was used.

Integrity model
---------------
``manifest_hash`` is order-independent: it is computed over canonical JSON of the
*sorted* record set. Re-ordering lines in a JSONL file therefore does not change
the hash, while changing any recorded value does. This is deliberate: the hash
answers "did the data definition change?" rather than "did the file's bytes
change?".

Path resolution
---------------
``filepath`` in a manifest is interpreted relative to an explicit base directory,
never to the process working directory, so the same manifest resolves identically
from any location.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.schema import (
    MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    MediaSample,
    as_json,
    utc_timestamp,
    validate_samples,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "ManifestSummary",
    "ManifestValidationReport",
    "calculate_file_sha256",
    "calculate_manifest_hash",
    "count_labels",
    "count_modalities",
    "describe_manifest",
    "iter_manifest_records",
    "load_manifest",
    "resolve_sample_path",
    "validate_manifest_file",
    "validate_sample_file",
    "write_manifest",
]

# 1 MiB. Large enough that syscall overhead is negligible, small enough that
# memory stays flat regardless of sample size.
DEFAULT_CHUNK_SIZE = 1024 * 1024


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def calculate_file_sha256(
    file_path: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    algorithm: str = "sha256",
) -> str:
    """Compute a hex digest of a file by streaming it in fixed-size chunks.

    The file is never loaded into memory, so this is safe for large media.

    Args:
        file_path: File to hash.
        chunk_size: Read size in bytes.
        algorithm: Any :mod:`hashlib` algorithm name.

    Returns:
        Lowercase hex digest.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If ``algorithm`` is unsupported or ``chunk_size`` is not positive.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    try:
        digest = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"unsupported hash algorithm: {algorithm!r}") from exc

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot hash missing file: {path}")

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_manifest_hash(samples: Iterable[MediaSample]) -> str:
    """Compute a deterministic, order-independent hash of a manifest.

    Each sample is canonicalized via :meth:`MediaSample.to_dict`, the records are
    sorted by their canonical JSON, and the concatenation is hashed. Result format
    is ``"sha256:<hex>"`` so the algorithm is self-describing in a run record.

    Args:
        samples: Manifest records. Order does not affect the result.

    Returns:
        A ``sha256:<hex>`` digest string.
    """
    canonical = sorted(as_json(sample.to_dict()) for sample in samples)
    hasher = hashlib.sha256()
    for record in canonical:
        hasher.update(record.encode("utf-8"))
        hasher.update(b"\n")
    return f"sha256:{hasher.hexdigest()}"


# ---------------------------------------------------------------------------
# Path resolution and verification
# ---------------------------------------------------------------------------


def resolve_sample_path(sample: MediaSample, base_dir: Path | str) -> Path:
    """Resolve a sample's ``filepath`` against a base directory.

    Relative paths resolve under ``base_dir``. Absolute paths are honoured as-is,
    which supports manifests that reference mounted or shared storage.
    """
    candidate = Path(sample.filepath)
    if candidate.is_absolute():
        return candidate
    return (Path(base_dir) / candidate).resolve()


def validate_sample_file(
    sample: MediaSample,
    base_dir: Path | str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[bool, str | None]:
    """Verify that a sample's file exists and matches its recorded SHA-256.

    Args:
        sample: The manifest record.
        base_dir: Directory that relative ``filepath`` values resolve against.
        chunk_size: Read size for hashing.

    Returns:
        ``(True, None)`` on success, otherwise ``(False, reason)``. Returns a
        reason string instead of raising so a caller can report every bad sample in
        one pass over a manifest.
    """
    resolved = resolve_sample_path(sample, base_dir)
    if not resolved.is_file():
        return False, f"missing file: {resolved}"
    try:
        actual = calculate_file_sha256(resolved, chunk_size=chunk_size)
    except OSError as exc:
        return False, f"unreadable file {resolved}: {exc}"
    if actual != sample.sha256:
        return False, f"sha256 mismatch for {resolved}: expected {sample.sha256}, got {actual}"
    return True, None


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def iter_manifest_records(path: Path | str) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream ``(line_number, record)`` pairs from a JSONL manifest.

    Blank lines are skipped. A JSON syntax error raises :class:`ManifestValidationError`
    naming the line, since a half-read manifest is worse than a failed one.

    Raises:
        ManifestValidationError: On malformed JSON, or a non-object record.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestValidationError(f"manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestValidationError(
                    f"malformed JSON at {manifest_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ManifestValidationError(
                    f"{manifest_path}:{line_number} must be a JSON object, "
                    f"got {type(record).__name__}"
                )
            yield line_number, record


def load_manifest(
    path: Path | str,
    *,
    strict: bool = True,
) -> list[MediaSample]:
    """Load a JSONL manifest into validated :class:`MediaSample` objects.

    Args:
        path: JSONL file, one sample object per line.
        strict: When ``True`` (default), any invalid record aborts the load with a
            :class:`ManifestValidationError` that lists every problem. When
            ``False``, invalid records are skipped and returned as a side channel.

    Returns:
        The samples, in file order. Empty when ``strict=False`` skips everything.

    Raises:
        ManifestValidationError: If ``strict`` and any record is invalid.
    """
    samples: list[MediaSample] = []
    errors: list[str] = []

    for line_number, record in iter_manifest_records(path):
        try:
            samples.append(MediaSample.from_dict(record))
        except ManifestValidationError as exc:
            errors.append(f"line {line_number}: {exc}")

    if strict and errors:
        raise ManifestValidationError(
            f"manifest {path} has {len(errors)} invalid record(s)", errors=errors
        )
    return samples


def write_manifest(
    path: Path | str,
    samples: Iterable[MediaSample],
    *,
    overwrite: bool = False,
) -> Path:
    """Write samples to a JSONL manifest with deterministic ordering and formatting.

    Records are sorted by ``(dataset_name, sample_id)`` so that regenerating a
    manifest from the same inputs produces a byte-identical file. That property is
    what makes the file reviewable in git.

    Args:
        path: Destination JSONL file. Parent directories are created.
        samples: Samples to write.
        overwrite: Permit replacing an existing manifest.

    Returns:
        The written path.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is ``False``.
        ManifestValidationError: If any sample is invalid.
    """
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {destination} (use overwrite=True)")

    ordered = sorted(samples, key=lambda s: (s.dataset_name, s.sample_id))
    # Validate before opening the destination, so a bad batch cannot destroy good data.
    validate_samples(ordered)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in ordered:
            handle.write(as_json(sample.to_dict()) + "\n")
    return destination


# ---------------------------------------------------------------------------
# Validation and reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManifestValidationReport:
    """Outcome of validating a manifest file.

    Attributes:
        path: The manifest that was checked.
        total_records: Records read.
        valid_samples: Successfully parsed samples.
        errors: Human-readable problems; empty means valid.
        duplicate_sample_ids: Duplicate ``dataset_name/sample_id`` pairs.
    """

    path: Path
    total_records: int
    valid_samples: int
    errors: list[str]
    duplicate_sample_ids: list[str]

    @property
    def is_valid(self) -> bool:
        """``True`` when no errors and no duplicates were found."""
        return not self.errors and not self.duplicate_sample_ids

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"{status}: {self.path}", f"records={self.total_records}"]
        if self.valid_samples:
            parts.append(f"parsed={self.valid_samples}")
        if self.duplicate_sample_ids:
            parts.append(f"duplicates={len(self.duplicate_sample_ids)}")
        if self.errors:
            parts.append(f"errors={len(self.errors)}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "total_records": self.total_records,
            "valid_samples": self.valid_samples,
            "errors": list(self.errors),
            "duplicate_sample_ids": list(self.duplicate_sample_ids),
            "is_valid": self.is_valid,
        }


def validate_manifest_file(
    path: Path | str,
    *,
    check_files: bool = False,
    base_dir: Path | str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> ManifestValidationReport:
    """Validate a manifest file and report every problem found.

    This never raises for content problems: it returns a report. That matters
    because the useful behaviour is "tell me everything wrong with this file",
    which a fail-fast exception cannot express.

    Args:
        path: JSONL manifest.
        check_files: Also verify each sample's file exists and matches its digest.
            Off by default: it requires the media to be present locally.
        base_dir: Base for resolving relative ``filepath`` values. Required when
            ``check_files`` is set.
        chunk_size: Read size for hashing.

    Returns:
        A :class:`ManifestValidationReport`.
    """
    manifest_path = Path(path)
    errors: list[str] = []
    samples: list[MediaSample] = []
    total = 0

    try:
        for line_number, record in iter_manifest_records(manifest_path):
            total += 1
            try:
                samples.append(MediaSample.from_dict(record))
            except ManifestValidationError as exc:
                errors.append(f"line {line_number}: {exc}")
    except ManifestValidationError as exc:
        errors.append(str(exc))

    duplicates: list[str] = []
    if samples:
        try:
            validate_samples(samples)
        except ManifestValidationError as exc:
            duplicates = exc.errors

    if check_files:
        if base_dir is None:
            errors.append("check_files=True requires base_dir to resolve relative paths")
        else:
            for sample in samples:
                ok, reason = validate_sample_file(sample, base_dir, chunk_size=chunk_size)
                if not ok:
                    errors.append(f"{sample.sample_id}: {reason}")

    return ManifestValidationReport(
        path=manifest_path,
        total_records=total,
        valid_samples=len(samples),
        errors=errors,
        duplicate_sample_ids=duplicates,
    )


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    """Composition summary of a manifest, for dataset documentation."""

    total_samples: int
    dataset_names: tuple[str, ...]
    modalities: dict[str, int]
    labels: dict[str, int]
    manipulation_types: dict[str, int]
    licenses: tuple[str, ...]
    manifest_hash: str
    schema_version: int = MANIFEST_SCHEMA_VERSION
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, suitable for ``dataset_info.json``."""
        return {
            "total_samples": self.total_samples,
            "dataset_names": list(self.dataset_names),
            "modalities": dict(self.modalities),
            "labels": dict(self.labels),
            "manipulation_types": dict(self.manipulation_types),
            "licenses": list(self.licenses),
            "manifest_hash": self.manifest_hash,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at or utc_timestamp(),
        }


def count_labels(samples: Iterable[MediaSample]) -> dict[str, int]:
    """Count samples per label."""
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.label.value] = counts.get(sample.label.value, 0) + 1
    return counts


def count_modalities(samples: Iterable[MediaSample]) -> dict[str, int]:
    """Count samples per modality."""
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.modality.value] = counts.get(sample.modality.value, 0) + 1
    return counts


def describe_manifest(
    samples: Iterable[MediaSample],
    *,
    generated_at: str | None = None,
) -> ManifestSummary:
    """Summarize manifest composition, including its content hash."""
    materialized = list(samples)
    manipulation: dict[str, int] = {}
    for sample in materialized:
        key = sample.manipulation_type.value
        manipulation[key] = manipulation.get(key, 0) + 1

    return ManifestSummary(
        total_samples=len(materialized),
        dataset_names=tuple(sorted({s.dataset_name for s in materialized})),
        modalities=count_modalities(materialized),
        labels=count_labels(materialized),
        manipulation_types=dict(sorted(manipulation.items())),
        licenses=tuple(sorted({s.license for s in materialized})),
        manifest_hash=calculate_manifest_hash(materialized),
        generated_at=generated_at or utc_timestamp(),
    )


def default_cache_dir() -> Path | None:
    """Return ``TEMP``-based cache directory, or ``None`` if unavailable.

    Provided for callers that need scratch space; the project does not itself
    download or cache any datasets.
    """
    temp = os.environ.get("TEMP") or os.environ.get("TMP")
    return Path(temp) if temp else None
