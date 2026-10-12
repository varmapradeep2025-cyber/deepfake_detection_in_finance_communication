"""Persisted analysis records and the file store that holds them.

A :class:`AnalysisRecord` is the durable form of one analysis: an indexed header
(identifier, time, modality, status, source name) plus the full serialized report.
Keeping the whole report — not a summary — is deliberate: the multimodal report builder
and any later audit need the evidence and warnings, and a stored summary would silently
drop them.

The store is a directory of JSON files, one per record, written atomically via a
temporary file and a rename. It performs no database work and needs no server, which
matches the project's current stage; it is a real, tested store, not a stub.

Raw media is never copied into the store. A record holds the artifact's *path*, digest
and reported properties, so storing an analysis of a large video does not duplicate the
video. This keeps the store small, avoids a second copy of potentially sensitive media,
and keeps the JSON portable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.inference.report import AnalysisReport

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisRecord",
    "AnalysisStore",
]

#: Version of the on-disk record envelope. Bumped only for an incompatible change.
SCHEMA_VERSION = 1

#: Subdirectory of the store root that holds record files.
_RECORDS_DIRNAME = "records"

#: Characters a report identifier may not contain, because it becomes a file name.
_FORBIDDEN_ID_CHARS = frozenset('\\/:*?"<>|')


def _safe_stem(report_id: str) -> str:
    """Validate a report identifier for use as a file name.

    Raises:
        ValueError: If the identifier is empty or contains path separators or other
            characters that would let it escape the store directory.
    """
    if not report_id or report_id in {".", ".."}:
        raise ValueError("report identifier must be non-empty and not a path dot")
    if any(char in _FORBIDDEN_ID_CHARS for char in report_id):
        raise ValueError(
            f"report identifier {report_id!r} contains a character not allowed in a file name"
        )
    return report_id


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    """One stored analysis: an index header plus the full report.

    Attributes:
        report_id: The report's request identifier; also the file stem on disk.
        created_at: RFC 3339 UTC time the record was stored (the report's completion).
        modality: The analysis modality value, duplicated from the report so a listing
            needs only the header.
        status: The analysis status value, duplicated for the same reason.
        source_name: Base name of the analyzed artifact, for display in a listing.
        report: The full serialized :class:`~ml.inference.report.AnalysisReport`.
    """

    report_id: str
    created_at: str
    modality: str
    status: str
    source_name: str
    report: Mapping[str, Any]

    @classmethod
    def from_report(
        cls,
        report: AnalysisReport,
        *,
        source_name: str | None = None,
        created_at: str | None = None,
    ) -> AnalysisRecord:
        """Build a record from an in-memory report.

        Args:
            report: The report to store.
            source_name: Display name for the artifact. Defaults to the report path's
                base name.
            created_at: Override for the stored time; defaults to the report's
                completion time, falling back to its start.
        """
        name = source_name or Path(report.path).name or report.path
        return cls(
            report_id=report.request_id,
            created_at=created_at or report.completed_at or report.started_at,
            modality=report.modality.value,
            status=report.status.value,
            source_name=name,
            report=report.to_dict(),
        )

    @property
    def has_score(self) -> bool:
        """Whether the stored report carries a model score."""
        output = self.report.get("output")
        return isinstance(output, Mapping) and output.get("raw_score") is not None

    @property
    def risk_level(self) -> str | None:
        """The stored risk level, or ``None`` when no risk was assessed."""
        risk = self.report.get("risk")
        if isinstance(risk, Mapping):
            level = risk.get("level")
            if isinstance(level, str):
                return level
        return None

    @property
    def errors(self) -> tuple[str, ...]:
        """The stored errors, if any."""
        raw = self.report.get("errors")
        if isinstance(raw, list):
            return tuple(str(item) for item in raw)
        return ()

    @property
    def warnings(self) -> tuple[str, ...]:
        """The stored warnings, if any."""
        raw = self.report.get("warnings")
        if isinstance(raw, list):
            return tuple(str(item) for item in raw)
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "report_id": self.report_id,
            "created_at": self.created_at,
            "modality": self.modality,
            "status": self.status,
            "source_name": self.source_name,
            "report": dict(self.report),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnalysisRecord:
        """Rebuild a record from its on-disk mapping.

        Raises:
            ValueError: If a required field is missing or mistyped, or the schema
                version is not one this code understands.
        """
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported record schema_version {version!r}; this build writes "
                f"version {SCHEMA_VERSION}"
            )
        report = data.get("report")
        if not isinstance(report, Mapping):
            raise TypeError("record is missing its 'report' object")
        header: dict[str, str] = {}
        for field_name in ("report_id", "created_at", "modality", "status", "source_name"):
            value = data.get(field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"record field {field_name!r} must be a non-empty string")
            header[field_name] = value
        return cls(report=dict(report), **header)


class AnalysisStore:
    """A file-backed store of :class:`AnalysisRecord` objects.

    Args:
        root: Directory that holds the store. Created on demand. Records live under
            ``<root>/records/<report_id>.json``.

    Writes are atomic: the record is written to a sibling temporary file and renamed
    over the target, so a crash mid-write leaves either the old file or the new one,
    never a truncated one.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._records_dir = self._root / _RECORDS_DIRNAME

    @property
    def root(self) -> Path:
        """The store root directory."""
        return self._root

    @property
    def records_dir(self) -> Path:
        """Directory holding the record files."""
        return self._records_dir

    def path_for(self, report_id: str) -> Path:
        """Return the file that would hold ``report_id``, without checking existence."""
        return self._records_dir / f"{_safe_stem(report_id)}.json"

    def save(self, record: AnalysisRecord, *, overwrite: bool = False) -> Path:
        """Write a record atomically and return its path.

        Args:
            record: The record to store.
            overwrite: Whether to replace an existing record with the same identifier.
                Defaults to ``False`` so a second analysis cannot silently clobber the
                first; pass ``True`` only when replacement is intended.

        Raises:
            FileExistsError: If a record with this identifier exists and ``overwrite``
                is ``False``.
        """
        self._records_dir.mkdir(parents=True, exist_ok=True)
        target = self._records_dir / f"{_safe_stem(record.report_id)}.json"
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"record {record.report_id!r} already exists at {target}; pass "
                "overwrite=True to replace it"
            )
        payload = json.dumps(record.to_dict(), indent=2, sort_keys=False)
        temp = target.with_suffix(".json.tmp")
        temp.write_text(payload + "\n", encoding="utf-8")
        temp.replace(target)
        return target

    def load(self, report_id: str) -> AnalysisRecord:
        """Load one record by identifier.

        Raises:
            FileNotFoundError: If no such record exists.
            ValueError: If the file is not a valid record.
        """
        target = self._records_dir / f"{_safe_stem(report_id)}.json"
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise TypeError(f"record file {target} does not contain a JSON object")
        return AnalysisRecord.from_dict(data)

    def exists(self, report_id: str) -> bool:
        """Whether a record with this identifier exists."""
        return (self._records_dir / f"{_safe_stem(report_id)}.json").is_file()

    def iter_records(self) -> Iterator[AnalysisRecord]:
        """Yield every stored record, ordered by ``(created_at, report_id)``.

        A malformed record file raises ``ValueError`` rather than being skipped: a
        record that cannot be read is a problem to surface, not to hide.
        """
        if not self._records_dir.is_dir():
            return
        paths = sorted(self._records_dir.glob("*.json"))
        records = [self._load_path(path) for path in paths]
        records.sort(key=lambda record: (record.created_at, record.report_id))
        yield from records

    def list_records(self) -> list[AnalysisRecord]:
        """Return every stored record as a list."""
        return list(self.iter_records())

    def count(self) -> int:
        """Number of stored records."""
        if not self._records_dir.is_dir():
            return 0
        return sum(1 for _ in self._records_dir.glob("*.json"))

    @staticmethod
    def _load_path(path: Path) -> AnalysisRecord:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise TypeError(f"record file {path} does not contain a JSON object")
        return AnalysisRecord.from_dict(data)
