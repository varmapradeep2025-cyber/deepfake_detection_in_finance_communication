"""Dataset-level provenance: where a dataset came from and who says so.

Why this is separate from the manifest
--------------------------------------
A manifest answers "what is in each file". Provenance answers "what is this dataset,
where did it come from, under what licence, and which ingestion produced this view of
it". The second question is asked once per dataset and must be answerable without
reading a million manifest lines — and it is the question a reader asks first when
deciding whether to trust a result.

Recorded fields
---------------
=========================  ===========================================================
Field                      Why it exists
=========================  ===========================================================
``dataset_id``             The key used in paths, manifests, and run records.
``dataset_name``           Human-readable name.
``version``                Datasets change. Results without a version are not
                           reproducible.
``source``                 How the data was obtained (vendor download, internal
                           collection, …). Free-form; may be ``None``.
``license``                Whether the data may lawfully be used. Required.
``access_date``            When it was obtained. ``None`` when not recorded.
``source_reference``       Where it is published. ``None`` rather than invented.
``ingestion_script_version`` Which ingestion produced this manifest.
``manifest_hash``          Which exact sample set the record describes.
``ingestion_timestamp``    When the ingestion ran. The one field allowed to vary
                           between two otherwise identical runs.
=========================  ===========================================================

Fabrication rules
-----------------
``source_reference``, ``access_date``, ``source``, and every sample-level provenance
field are ``None`` when unknown. No URL is invented to fill a field, because a
plausible-looking link that does not resolve is worse than an explicit gap: it
survives review and misleads later readers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.schema import as_json

__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "DatasetProvenance",
    "ProvenanceError",
    "load_provenance",
    "provenance_path_for",
    "write_provenance",
]

PROVENANCE_SCHEMA_VERSION = 1

#: Fields excluded from :meth:`DatasetProvenance.content_hash`.
#:
#: The timestamp is the only field that legitimately differs between two runs over
#: identical data, so excluding it is what lets "same data, same provenance" be
#: asserted mechanically.
_TIMESTAMP_FIELDS = frozenset({"ingestion_timestamp"})


class ProvenanceError(ValueError):
    """Raised when a provenance record is invalid.

    A :class:`ValueError` because it is almost always raised while parsing a file
    somebody else wrote.
    """


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """One dataset's acquisition and ingestion record.

    Attributes:
        dataset_id: Identifier used everywhere else in the project.
        dataset_name: Human-readable name.
        version: Dataset release identifier.
        license: SPDX identifier or dataset-specific licence string. Required: a
            dataset whose licence is unknown cannot lawfully be used.
        source: How the data was obtained, or ``None`` when not recorded.
        source_reference: Where the dataset is published, or ``None``. Never invented.
        access_date: When the data was accessed (RFC 3339 or ``YYYY-MM-DD``), or
            ``None``.
        ingestion_script_version: Version of the ingestion code that produced the
            manifest.
        manifest_hash: ``sha256:<hex>`` of the manifest this record describes.
        ingestion_timestamp: RFC 3339 UTC timestamp of the ingestion run.
        adapter_name: Adapter that performed the ingestion.
        adapter_version: Revision of that adapter.
        notes: Free-form remarks. Kept for the things that do not fit a field.
        provenance_schema_version: Schema version, for forward compatibility.
    """

    dataset_id: str
    dataset_name: str
    version: str
    license: str
    ingestion_script_version: str
    manifest_hash: str
    ingestion_timestamp: str
    source: str | None = None
    source_reference: str | None = None
    access_date: str | None = None
    adapter_name: str = "unknown"
    adapter_version: str = "unknown"
    notes: str = ""
    provenance_schema_version: int = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize and validate, so an invalid record cannot exist in memory."""
        for field_name in (
            "dataset_id",
            "dataset_name",
            "version",
            "license",
            "ingestion_script_version",
            "manifest_hash",
            "ingestion_timestamp",
            "adapter_name",
            "adapter_version",
        ):
            object.__setattr__(
                self, field_name, _require_text(getattr(self, field_name), field_name)
            )
        for field_name in ("source", "source_reference", "access_date"):
            object.__setattr__(
                self, field_name, _optional_text(getattr(self, field_name), field_name)
            )
        object.__setattr__(self, "notes", str(self.notes))

        if not self.manifest_hash.lower().startswith("sha256:"):
            raise ProvenanceError(
                "manifest_hash must be prefixed with 'sha256:'; "
                f"got {self.manifest_hash!r}. Compute it with "
                "datasets.manifest.calculate_manifest_hash()."
            )
        digest = self.manifest_hash.split(":", 1)[1].strip()
        if len(digest) != 64 or not set(digest) <= set("0123456789abcdef"):
            raise ProvenanceError(
                f"manifest_hash must carry 64 lowercase hex characters, got {digest!r}"
            )
        object.__setattr__(self, "manifest_hash", f"sha256:{digest}")

    # --- derived ------------------------------------------------------------

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        """Names of the optional fields this record does not know.

        Stated explicitly so a reader can see at a glance which provenance is missing
        rather than having to diff against the schema. An empty tuple means the record
        is complete.
        """
        candidates = ("source", "source_reference", "access_date")
        return tuple(name for name in candidates if getattr(self, name) is None)

    def content_hash(self) -> str:
        """Deterministic ``sha256:<hex>`` over everything except the timestamp.

        Two ingestions of the same dataset with the same adapter produce the same value,
        which is the machine-checkable form of "reproducible". The timestamp is excluded
        because it is the one field that is expected to differ.
        """
        payload = {
            key: value
            for key, value in sorted(self.to_dict().items())
            if key not in _TIMESTAMP_FIELDS
        }
        return f"sha256:{hashlib.sha256(as_json(payload).encode('utf-8')).hexdigest()}"

    # --- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict form, with keys in a stable order."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "license": self.license,
            "source": self.source,
            "source_reference": self.source_reference,
            "access_date": self.access_date,
            "ingestion_script_version": self.ingestion_script_version,
            "manifest_hash": self.manifest_hash,
            "ingestion_timestamp": self.ingestion_timestamp,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "notes": self.notes,
            "provenance_schema_version": self.provenance_schema_version,
        }

    def to_json(self) -> str:
        """Pretty, sorted, newline-terminated JSON — the on-disk form."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetProvenance:
        """Rebuild a record from parsed JSON.

        Raises:
            ProvenanceError: If a required field is missing or invalid.
        """
        if not isinstance(raw, dict):
            raise ProvenanceError(
                f"provenance record must be a JSON object, got {type(raw).__name__}"
            )
        known = set(cls.__dataclass_fields__)
        missing = sorted(
            name
            for name in (
                "dataset_id",
                "dataset_name",
                "version",
                "license",
                "ingestion_script_version",
                "manifest_hash",
                "ingestion_timestamp",
            )
            if name not in raw
        )
        if missing:
            raise ProvenanceError(
                f"provenance record missing required field(s): {', '.join(missing)}",
            )
        return cls(**{key: raw[key] for key in known if key in raw})


def provenance_path_for(manifest_path: Path | str) -> Path:
    """Derive the provenance file that belongs beside a manifest.

    ``datasets/manifests/<dataset_id>.jsonl`` pairs with
    ``datasets/manifests/<dataset_id>.provenance.json``. Keeping the two adjacent means
    a reviewer sees the record and the data together, and the documented directory
    layout needs no extra folder.
    """
    manifest = Path(manifest_path)
    return manifest.with_name(f"{manifest.stem}.provenance.json")


def write_provenance(path: Path | str, provenance: DatasetProvenance) -> Path:
    """Write a provenance record as pretty JSON.

    Args:
        path: Destination file. Parent directories are created.
        provenance: The validated record.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(provenance.to_json(), encoding="utf-8", newline="\n")
    return destination


def load_provenance(path: Path | str) -> DatasetProvenance:
    """Load and validate a provenance record.

    Raises:
        ProvenanceError: If the file is missing, is not JSON, or is invalid.
    """
    source = Path(path)
    if not source.is_file():
        raise ProvenanceError(f"provenance record not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"malformed JSON at {source}: {exc.msg}") from exc
    return DatasetProvenance.from_dict(raw)


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProvenanceError(f"{field_name} must be a string or null, got {type(value).__name__}")
    return value.strip() or None
