"""Dataset ingestion: external directory in, standardized JSONL manifest out.

The pipeline
------------
::

    External Dataset
        -> Dataset Adapter            (datasets/adapters/)
        -> Normalize Sample Info      (datasets/normalization.py)
        -> Calculate SHA-256          (datasets/manifest.py)
        -> Validate Labels / Modality (datasets/schema.py)
        -> Generate Manifest          (datasets/manifest.py)
        -> Validate Manifest          (datasets/manifest.py)
        -> Register Dataset           (manifest + provenance record)

Usage::

    # see what an adapter would produce, changing nothing
    python -m datasets.ingest --dataset-id demo --adapter template --dry-run

    # actually write datasets/manifests/<dataset_id>.jsonl and its provenance record
    python -m datasets.ingest --dataset-id demo --adapter template \\
        --adapter-config configs/adapters/demo_corpus.yaml

Design commitments
------------------
**Uncertainty survives.** If the dataset does not state a manipulation type, the
manifest says ``unknown``. If it does not state a device or a creation time, no such
field appears. Nothing is invented to make a record look complete, because a manifest
that is silently wrong is more damaging than one that admits a gap: it survives
review.

**Source media is read-only.** The pipeline opens files to hash them and never writes,
moves, or deletes inside the dataset directory. Ingestion is an observation.

**Failure is loud and complete.** Every problem found in a run is collected and
reported together rather than aborting on the first, and a manifest is written only
when the whole sample set is valid. A partially correct manifest is not written at all,
because there is no way to tell from the file which records are trustworthy.

**Determinism.** Given the same dataset directory, adapter, and configuration, the
manifest bytes are identical. Ordering is canonicalized before writing and no
timestamp enters the manifest. The provenance record carries the ingestion timestamp
and nothing else time-dependent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from datasets.adapters import DatasetAdapter, adapter_names, resolve_adapter
from datasets.adapters.base import AdapterError, RawSample, validate_dataset_id
from datasets.adapters.template import TemplateAdapter
from datasets.manifest import (
    calculate_file_sha256,
    calculate_manifest_hash,
    validate_manifest_file,
    validate_sample_file,
    write_manifest,
)
from datasets.normalization import (
    NormalizationError,
    resolve_manipulation,
    resolve_modality,
)
from datasets.provenance import (
    DatasetProvenance,
    provenance_path_for,
    write_provenance,
)
from datasets.schema import (
    MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    ManipulationType,
    MediaSample,
    Modality,
    SampleLabel,
    utc_timestamp,
    validate_samples,
)

__all__ = [
    "DEFAULT_MANIFEST_DIRNAME",
    "INGESTION_SCRIPT_VERSION",
    "HashVerificationStatus",
    "IngestionError",
    "IngestionRequest",
    "IngestionSummary",
    "build_media_sample",
    "ingest_dataset",
    "main",
]

#: Bump when discovery, normalization, or manifest content changes in a way that
#: would alter the manifest hash of an unchanged dataset. Recorded in provenance so a
#: changed hash can be explained.
INGESTION_SCRIPT_VERSION = "1"

#: Manifest directory, relative to the repository root, used when ``--output`` is absent.
DEFAULT_MANIFEST_DIRNAME = "datasets/manifests"

#: Raw directory convention: ``datasets/raw/<dataset_id>/``.
DEFAULT_RAW_DIRNAME = "datasets/raw"

#: Optional fields whose absence is counted in the summary. These are exactly the
#: places where a dataset usually tells us nothing, so counting them is how the summary
#: stays honest about coverage.
_COUNTED_METADATA_FIELDS = ("source", "manipulation_type", "metadata")


class IngestionError(RuntimeError):
    """Raised when an ingestion cannot be attempted at all.

    Reserved for problems with the *request* — an unusable dataset id, a missing
    source directory, an unknown adapter, a misconfigured adapter. Problems with
    individual samples are collected into the summary instead, so one bad file does not
    hide the other ten thousand good ones.
    """


class HashVerificationStatus(str, Enum):
    """Outcome of re-verifying recorded digests against files on disk."""

    NOT_REQUESTED = "not_requested"
    VERIFIED = "verified"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    """Everything one ingestion run needs.

    Attributes:
        dataset_id: Identifier, used as the manifest file stem and the raw directory
            name. Must be a single safe path segment.
        adapter: The adapter describing this dataset.
        source_dir: Root of the dataset on disk. Read-only.
        output_dir: Directory receiving ``<dataset_id>.jsonl`` and its provenance
            record. ``None`` uses the project default.
        dry_run: Discover, normalize, hash, and report without writing anything.
        verify_hashes: Re-verify each recorded digest against the file after writing.
        overwrite: Permit replacing an existing manifest.
    """

    dataset_id: str
    adapter: DatasetAdapter
    source_dir: Path
    output_dir: Path | None = None
    dry_run: bool = False
    verify_hashes: bool = True
    overwrite: bool = False

    @property
    def manifest_path(self) -> Path:
        """Where the manifest would be, or would be written."""
        base = Path(self.output_dir) if self.output_dir is not None else _default_output_dir()
        return base / f"{self.dataset_id}.jsonl"

    @property
    def provenance_path(self) -> Path:
        """Provenance record paired with :attr:`manifest_path`."""
        return provenance_path_for(self.manifest_path)


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """What an ingestion run found, and whether it succeeded.

    Reported by both dry and real runs. A dry run produces the same counts as a real one
    — it simply leaves ``manifest_path`` and ``provenance_path`` unset.

    Attributes:
        dataset_id: Dataset this run describes.
        dataset_version: Version declared by the adapter.
        adapter_name: Adapter used.
        adapter_version: Adapter revision used.
        ingestion_script_version: Revision of this pipeline.
        dry_run: Whether anything was written.
        source_dir: Directory that was read.
        manifest_path: Written manifest, or ``None`` in a dry run or on failure.
        provenance_path: Written provenance record, or ``None``.
        total_samples: Samples successfully ingested.
        real_samples: Samples labelled ``real``.
        fake_samples: Samples labelled ``fake``.
        modality_counts: Sample count per canonical modality.
        manipulation_counts: Sample count per canonical manipulation type, including
            ``unknown`` — the size of the ``unknown`` bucket is a property of the
            dataset worth reporting.
        missing_metadata_counts: How many samples lack each optional field.
        missing_metadata_keys: How many samples lack each metadata key, so a dataset
            that never records a device says so once instead of per sample.
        hash_verification: Outcome of digest verification.
        manifest_hash: ``sha256:<hex>`` of the manifest content.
        ingestion_timestamp: RFC 3339 UTC time the run started.
        errors: Every problem found, in discovery order. Empty means success.
    """

    dataset_id: str
    dataset_version: str
    adapter_name: str
    adapter_version: str
    ingestion_script_version: str
    dry_run: bool
    source_dir: Path
    manifest_path: Path | None
    provenance_path: Path | None
    total_samples: int
    real_samples: int
    fake_samples: int
    modality_counts: dict[str, int]
    manipulation_counts: dict[str, int]
    missing_metadata_counts: dict[str, int]
    missing_metadata_keys: dict[str, int]
    hash_verification: HashVerificationStatus
    manifest_hash: str
    ingestion_timestamp: str
    errors: list[str] = field(default_factory=list)
    manifest_schema_version: int = MANIFEST_SCHEMA_VERSION

    @property
    def is_successful(self) -> bool:
        """``True`` when no errors were found."""
        return not self.errors

    @property
    def manifest_written(self) -> bool:
        """``True`` when a manifest file was produced by this run."""
        return self.manifest_path is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, suitable for logging or a run record."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "ingestion_script_version": self.ingestion_script_version,
            "manifest_schema_version": self.manifest_schema_version,
            "dry_run": self.dry_run,
            "source_dir": str(self.source_dir),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "provenance_path": str(self.provenance_path) if self.provenance_path else None,
            "total_samples": self.total_samples,
            "real_samples": self.real_samples,
            "fake_samples": self.fake_samples,
            "modality_counts": dict(sorted(self.modality_counts.items())),
            "manipulation_counts": dict(sorted(self.manipulation_counts.items())),
            "missing_metadata_counts": dict(sorted(self.missing_metadata_counts.items())),
            "missing_metadata_keys": dict(sorted(self.missing_metadata_keys.items())),
            "hash_verification": self.hash_verification.value,
            "manifest_hash": self.manifest_hash,
            "ingestion_timestamp": self.ingestion_timestamp,
            "is_successful": self.is_successful,
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        """One-line canonical JSON."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def render(self) -> str:
        """Human-readable report, as printed by the CLI."""
        lines = [
            f"dataset            : {self.dataset_id}",
            f"adapter            : {self.adapter_name} v{self.adapter_version}",
            f"dataset version    : {self.dataset_version}",
            f"script version     : {self.ingestion_script_version}",
            f"mode               : {'dry-run (nothing written)' if self.dry_run else 'ingest'}",
            f"source             : {self.source_dir}",
            f"samples            : {self.total_samples}",
            f"  real             : {self.real_samples}",
            f"  fake             : {self.fake_samples}",
            f"modalities         : {_render_counts(self.modality_counts)}",
            f"manipulations      : {_render_counts(self.manipulation_counts)}",
            f"missing metadata   : {_render_counts(self.missing_metadata_counts)}",
        ]
        if self.missing_metadata_keys:
            lines.append(f"missing metadata keys: {_render_counts(self.missing_metadata_keys)}")
        lines.append(f"hash verification  : {self.hash_verification.value}")
        lines.append(f"manifest hash      : {self.manifest_hash}")
        lines.append(f"ingestion time     : {self.ingestion_timestamp}")
        if self.manifest_path:
            lines.append(f"manifest           : {self.manifest_path}")
            lines.append(f"provenance         : {self.provenance_path}")
        if self.errors:
            lines.append(f"errors             : {len(self.errors)}")
            lines.extend(f"  - {error}" for error in self.errors[:_MAX_REPORTED_ERRORS])
            if len(self.errors) > _MAX_REPORTED_ERRORS:
                lines.append(f"  ... and {len(self.errors) - _MAX_REPORTED_ERRORS} more")
        return "\n".join(lines)


#: Cap on errors echoed to the terminal. A dataset with a systematic mistake can
#: produce hundreds of thousands, and a wall of text hides the pattern.
_MAX_REPORTED_ERRORS = 25


# ---------------------------------------------------------------------------
# Normalization: one RawSample -> one MediaSample
# ---------------------------------------------------------------------------


def build_media_sample(
    raw: RawSample,
    *,
    adapter: DatasetAdapter,
    dataset_name: str,
    source_root: Path,
    license_id: str | None = None,
) -> MediaSample:
    """Normalize, hash, and validate one discovered sample.

    The order matters: attribution is resolved *before* the file is hashed, so a sample
    whose label or modality cannot be established costs nothing beyond an error
    message. Hashing last also means the recorded digest always belongs to the exact
    bytes the rest of the record describes.

    Args:
        raw: The adapter's output for one file.
        adapter: Adapter supplying the declared vocabularies.
        dataset_name: Dataset name recorded on the sample.
        source_root: Directory that ``filepath`` is made relative to.
        license_id: Licence to record. Defaults to the adapter's.

    Returns:
        A validated :class:`~datasets.schema.MediaSample`.

    Raises:
        NormalizationError: If the label, modality, or manipulation type cannot be
            resolved from declared information.
        ManifestValidationError: If the file is missing, unreadable, outside the source
            root, or the resulting record is invalid.
    """
    label = adapter.label_mapping().resolve(raw.label_token)
    modality = resolve_modality(
        authoritative=raw.modality,
        token=raw.modality_token,
        mapping=adapter.modality_mapping(),
        subject=raw.sample_id,
    )
    manipulation = resolve_manipulation(
        token=raw.manipulation_token, mapping=adapter.manipulation_mapping()
    )

    path = Path(raw.path)
    if not path.is_file():
        raise ManifestValidationError(f"{raw.sample_id}: media file not found: {path}")
    try:
        relative = path.resolve().relative_to(Path(source_root).resolve())
    except ValueError as exc:
        raise ManifestValidationError(
            f"{raw.sample_id}: {path} is outside the dataset root {source_root}. A manifest "
            "may only reference files inside the dataset it describes."
        ) from exc

    metadata = dict(raw.metadata)
    # Record the dataset's own wording when it differs from ours. This is the dataset's
    # vocabulary, copied verbatim — not an interpretation — and it is what makes a
    # future re-ingestion auditable.
    if raw.label_token and raw.label_token.strip().lower() != label.value:
        metadata.setdefault("source_label", raw.label_token.strip())
    if raw.manipulation_token and manipulation in {
        ManipulationType.UNKNOWN,
        ManipulationType.OTHER,
    }:
        metadata.setdefault("source_manipulation_label", raw.manipulation_token.strip())

    return MediaSample(
        sample_id=raw.sample_id,
        dataset_name=dataset_name,
        filepath=relative.as_posix(),
        modality=modality,
        label=label,
        manipulation_type=manipulation,
        source=raw.source_token,
        # split stays None: authoritative assignments live in the frozen split
        # definition under datasets/splits/, not in the manifest.
        split=None,
        sha256=calculate_file_sha256(path),
        license=license_id or adapter.license_id,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def ingest_dataset(request: IngestionRequest) -> IngestionSummary:
    """Run one ingestion end to end.

    Args:
        request: What to ingest, with which adapter, into what.

    Returns:
        An :class:`IngestionSummary`. Sample-level problems are reported in
        ``summary.errors`` rather than raised, so one run surfaces every problem at
        once. Request-level problems raise :class:`IngestionError`.

    Raises:
        IngestionError: If the dataset id is unusable, the source directory is missing,
            or the adapter is misconfigured or unregistered.
    """
    dataset_id = validate_dataset_id(request.dataset_id)
    source_dir = Path(request.source_dir)
    if not source_dir.is_dir():
        raise IngestionError(
            f"dataset source directory not found: {source_dir}. Place the dataset under "
            f"{DEFAULT_RAW_DIRNAME}/<dataset_id>/ or pass --source explicitly."
        )

    adapter = request.adapter
    adapter.validate_configuration()
    started_at = utc_timestamp()

    samples, errors = _collect_samples(adapter, dataset_id, source_dir)
    _check_duplicates(samples, errors)
    manifest_hash = calculate_manifest_hash(samples)

    manifest_path: Path | None = None
    provenance_path: Path | None = None
    verification = HashVerificationStatus.NOT_APPLICABLE

    if errors:
        # No manifest is written from a sample set with known problems: a reader
        # cannot tell from the file which records are trustworthy.
        pass
    elif request.dry_run:
        verification = (
            _verify(request, samples)
            if request.verify_hashes
            else HashVerificationStatus.NOT_REQUESTED
        )
    else:
        manifest_path = request.manifest_path
        provenance_path = request.provenance_path
        _write_outputs(
            request,
            dataset_id=dataset_id,
            samples=samples,
            manifest_hash=manifest_hash,
            manifest_path=manifest_path,
            provenance_path=provenance_path,
            ingestion_timestamp=started_at,
        )
        verification = _verify(request, samples)

    return IngestionSummary(
        dataset_id=dataset_id,
        dataset_version=adapter.dataset_version,
        adapter_name=adapter.name,
        adapter_version=adapter.adapter_version,
        ingestion_script_version=INGESTION_SCRIPT_VERSION,
        dry_run=request.dry_run,
        source_dir=source_dir,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        total_samples=len(samples),
        real_samples=_count_label(samples, SampleLabel.REAL),
        fake_samples=_count_label(samples, SampleLabel.FAKE),
        modality_counts=_count_modalities(samples),
        manipulation_counts=_count_manipulations(samples),
        missing_metadata_counts=_count_missing_fields(samples),
        missing_metadata_keys=_count_missing_metadata_keys(samples),
        hash_verification=verification,
        manifest_hash=manifest_hash,
        ingestion_timestamp=started_at,
        errors=errors,
    )


def _collect_samples(
    adapter: DatasetAdapter, dataset_id: str, source_dir: Path
) -> tuple[list[MediaSample], list[str]]:
    """Normalize every discovered sample, collecting problems instead of raising.

    ``adapter.dataset_name`` is used verbatim when set, because a dataset's own name is
    more informative than our identifier for it; ``dataset_id`` is the fallback.
    """
    dataset_name = adapter.dataset_name or dataset_id
    samples: list[MediaSample] = []
    errors: list[str] = []
    for raw in adapter.discover(source_dir):
        try:
            samples.append(
                build_media_sample(
                    raw, adapter=adapter, dataset_name=dataset_name, source_root=source_dir
                )
            )
        except (NormalizationError, ManifestValidationError) as exc:
            errors.append(f"{_display(raw)}: {exc}")
        except OSError as exc:
            errors.append(f"{_display(raw)}: could not read media file: {exc}")
    return samples, errors


def _check_duplicates(samples: Sequence[MediaSample], errors: list[str]) -> None:
    """Append duplicate-identifier problems to ``errors``.

    Checked in dry runs too: a collision is a property of the dataset and adapter, not
    of the writing step, and discovering it during a preview is the whole point of one.

    Duplicate ids would also make a split ambiguous, since an identifier could no
    longer resolve to exactly one file. They are reported rather than disambiguated
    automatically, because a collision is a defect worth seeing rather than papering
    over with a counter suffix.
    """
    try:
        validate_samples(list(samples))
    except ManifestValidationError as exc:
        errors.extend(f"duplicate sample_id: {entry}" for entry in exc.errors)


def _write_outputs(
    request: IngestionRequest,
    *,
    dataset_id: str,
    samples: Sequence[MediaSample],
    manifest_hash: str,
    manifest_path: Path,
    provenance_path: Path,
    ingestion_timestamp: str,
) -> None:
    """Write the manifest, re-validate it, then write the provenance record.

    ``manifest_hash`` is computed over the sorted record *set*, while
    :func:`~datasets.manifest.write_manifest` emits records sorted by
    ``(dataset_name, sample_id)``. The two orderings differ but the hash does not
    depend on order, so the recorded digest describes exactly the file just written.
    """
    write_manifest(manifest_path, samples, overwrite=request.overwrite)

    report = validate_manifest_file(manifest_path)
    if not report.is_valid:
        raise IngestionError(
            f"generated manifest failed validation ({manifest_path}): "
            + "; ".join(report.errors or report.duplicate_sample_ids)
        )

    adapter = request.adapter
    provenance = DatasetProvenance(
        dataset_id=dataset_id,
        dataset_name=adapter.dataset_name,
        version=adapter.dataset_version,
        license=adapter.license_id,
        source=None,
        source_reference=adapter.source_reference,
        access_date=adapter.access_date,
        ingestion_script_version=INGESTION_SCRIPT_VERSION,
        manifest_hash=manifest_hash,
        ingestion_timestamp=ingestion_timestamp,
        adapter_name=adapter.name,
        adapter_version=adapter.adapter_version,
        notes=(
            "Produced by datasets.ingest. source and access_date are recorded only when "
            "the adapter supplied them; absence means unrecorded, not 'not applicable'."
        ),
    )
    write_provenance(provenance_path, provenance)


def _verify(request: IngestionRequest, samples: Sequence[MediaSample]) -> HashVerificationStatus:
    """Re-hash every sample file and confirm it matches the manifest.

    Cheap insurance against a file changing between the first hash and the write, and
    the operational form of ``data.verify_hashes_on_load``. Returns a status instead of
    raising so a failure is reported alongside the rest of the run.
    """
    if not request.verify_hashes:
        return HashVerificationStatus.NOT_REQUESTED
    root = Path(request.source_dir)
    for sample in samples:
        ok, _reason = validate_sample_file(sample, root)
        if not ok:
            return HashVerificationStatus.FAILED
    return HashVerificationStatus.VERIFIED


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def _count_label(samples: Iterable[MediaSample], label: SampleLabel) -> int:
    return sum(1 for sample in samples if sample.label is label)


def _count_modalities(samples: Iterable[MediaSample]) -> dict[str, int]:
    counts = dict.fromkeys(Modality.values(), 0)
    for sample in samples:
        counts[sample.modality.value] += 1
    return {key: value for key, value in counts.items() if value}


def _count_manipulations(samples: Iterable[MediaSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = sample.manipulation_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_missing_fields(samples: Sequence[MediaSample]) -> dict[str, int]:
    """How many samples lack each optional field."""
    counts = dict.fromkeys(_COUNTED_METADATA_FIELDS, 0)
    for sample in samples:
        populated = set(sample.which_fields_are_populated())
        for name in _COUNTED_METADATA_FIELDS:
            if name not in populated:
                counts[name] += 1
    return {name: count for name, count in counts.items() if count}


def _count_missing_metadata_keys(samples: Sequence[MediaSample]) -> dict[str, int]:
    """How many samples lack each metadata key, across the whole dataset.

    Only keys that some sample omits are reported. A key every sample carries is a
    property of this dataset's shape, not a gap, and listing it as ``key: 0`` would
    bury the real ones.
    """
    totals: dict[str, int] = {}
    for sample in samples:
        for key in sample.metadata:
            totals[key] = totals.get(key, 0) + 1
    gaps = {key: len(samples) - seen for key, seen in totals.items() if seen < len(samples)}
    return dict(sorted(gaps.items()))


def _display(raw: RawSample) -> str:
    """A short identifier for a sample that failed to normalize."""
    return raw.sample_id or str(raw.path)


def _render_counts(counts: dict[str, int]) -> str:
    """Render a count mapping compactly, e.g. ``image=2 audio=2 video=2``."""
    if not counts:
        return "(none)"
    return " ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def _default_output_dir() -> Path:
    """Default manifest directory, read from ``configs/project.yaml`` when possible.

    Configuration is the single source of truth for paths, so it is consulted first; the
    literal fallback exists only so the module still works outside a checkout.
    """
    if _DEFAULT_OUTPUT_DIR is not None:
        return _DEFAULT_OUTPUT_DIR
    try:
        from ml.reproducibility.config import load_project_config

        resolved = load_project_config().get_path("paths.source.datasets_manifests")
    except Exception:  # pragma: no cover - exercised only outside a valid checkout
        return Path(DEFAULT_MANIFEST_DIRNAME)
    globals()["_DEFAULT_OUTPUT_DIR"] = resolved
    return resolved


#: Resolved once per process. ``ingest_dataset`` asks for the output directory twice per
#: run (manifest and provenance); re-parsing YAML for each would be wasteful.
_DEFAULT_OUTPUT_DIR: Path | None = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m datasets.ingest",
        description=(
            "Convert an external dataset into a standardized JSONL manifest. Reads the "
            "source media without modifying it."
        ),
        epilog=(
            "Research datasets are selected separately; this command only standardizes "
            "what a chosen dataset already contains. Licences must be confirmed before "
            "ingestion via the adapter's 'license' field."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset identifier; also the manifest file stem and raw directory name.",
    )
    parser.add_argument(
        "--adapter",
        default=TemplateAdapter.name,
        help="Registered adapter name. Use --list-adapters to see the options.",
    )
    parser.add_argument(
        "--adapter-config",
        type=Path,
        default=None,
        help="JSON or YAML file configuring a declarative adapter (its licence, "
        "vocabulary, layout).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"Dataset root. Defaults to {DEFAULT_RAW_DIRNAME}/<dataset_id>/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            f"Directory for the manifest and provenance record. "
            f"Defaults to {DEFAULT_MANIFEST_DIRNAME}/."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover, validate, and count without writing or modifying anything.",
    )
    parser.add_argument(
        "--verify-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-hash each sample after writing and confirm it matches the manifest.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing manifest instead of refusing.",
    )
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="Print the registered adapter names and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of text.",
    )
    return parser


def _build_adapter(adapter_name: str, config_path: Path | None, dataset_id: str) -> DatasetAdapter:
    """Instantiate the requested adapter, honouring ``--adapter-config``.

    Adapters supply their vocabulary either in code (they override
    :meth:`~datasets.adapters.base.DatasetAdapter.from_dataset_id`) or in a JSON or YAML
    configuration file (``from_config_file``). Neither path accepts a licence or label
    vocabulary as a command-line argument: those are exactly the values that must not
    be typed in on a whim.
    """
    adapter_class = resolve_adapter(adapter_name)

    if config_path is not None:
        adapter = adapter_class.from_config_file(config_path, dataset_id=dataset_id)
        if adapter is None:
            raise IngestionError(
                f"adapter {adapter_name!r} does not support --adapter-config; it takes its "
                "configuration in code."
            )
        return adapter

    adapter = adapter_class.from_dataset_id(dataset_id)
    if adapter is None:
        raise IngestionError(
            f"adapter {adapter_name!r} needs configuration. Pass --adapter-config "
            "<file.yaml> declaring at least 'license' and 'labels'; ingestion will not "
            "assume a licence or a label vocabulary."
        )
    return adapter


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Returns:
        ``0`` on success, ``1`` when any sample failed validation, ``2`` for a bad
        request (unknown adapter, missing source directory, malformed config).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_adapters:
        print("registered adapters:")
        for name in adapter_names():
            print(f"  {name}")
        return 0

    try:
        adapter = _build_adapter(args.adapter, args.adapter_config, args.dataset_id)
        source_dir = args.source or (Path(DEFAULT_RAW_DIRNAME) / args.dataset_id)
        request = IngestionRequest(
            dataset_id=args.dataset_id,
            adapter=adapter,
            source_dir=source_dir,
            output_dir=args.output,
            dry_run=args.dry_run,
            verify_hashes=args.verify_hashes,
            overwrite=args.overwrite,
        )
        summary = ingest_dataset(request)
    except (IngestionError, AdapterError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(summary.to_json() if args.json else summary.render())
    return 0 if summary.is_successful else 1


if __name__ == "__main__":
    sys.exit(main())
