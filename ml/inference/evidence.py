"""Evidence items: every statement a report makes is tagged with how it is known.

The repository's forensic rule is that no claim is a bare assertion. This module
gives that rule a type. An :class:`EvidenceItem` records what was found, where it
came from, how it was obtained, and — crucially — whether it was

* **observed** — read directly from the artifact (its bytes, size, decoded
  dimensions). Reproducible by re-reading the file.
* **derived** — computed from observed data by a deterministic transform (a
  preprocessing digest, a normalized tensor shape). Reproducible given the same
  inputs and code.
* **inferred** — produced by a model or a heuristic, and therefore only as good as
  that model. Carries ``confidence`` only when an honest number exists; it is
  ``None`` rather than a made-up value when it does not.

The field names and enums mirror ``frontend/src/types/evidence.ts`` so a future API
can serialize these records for the existing UI without a translation layer. The
serialized form is snake_case, matching every other schema in this repository; the
correspondence to the frontend's camelCase names is documented in
``docs/inference.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "ClaimKind",
    "EvidenceItem",
    "EvidenceType",
    "make_evidence_item",
]


class ClaimKind(str, Enum):
    """How a piece of evidence is known. Mirrors the frontend ``ClaimKind``."""

    #: Read directly from the artifact by re-reading it.
    OBSERVED = "observed"
    #: Computed from observed data by a deterministic transform.
    DERIVED = "derived"
    #: Produced by a model or heuristic; only as reliable as that method.
    INFERRED = "inferred"

    @classmethod
    def parse(cls, value: Any) -> ClaimKind:
        """Resolve a member from a member or its string value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(f"{value!r} is not a valid claim kind")


class EvidenceType(str, Enum):
    """Category of an evidence item. Mirrors the frontend ``EvidenceType``."""

    #: A cryptographic digest of a file or buffer.
    HASH = "hash"
    #: A property read from the media (dimensions, channel layout, size).
    METADATA = "metadata"
    #: Where the artifact came from and what was done to it.
    PROVENANCE = "provenance"
    #: A signal produced by analysis, such as a detector score.
    SIGNAL = "signal"
    #: Anything that came from outside the artifact itself.
    EXTERNAL = "external"

    @classmethod
    def parse(cls, value: Any) -> EvidenceType:
        """Resolve a member from a member or its string value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(f"{value!r} is not a valid evidence type")


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One tagged statement in a report.

    Attributes:
        id: Stable identifier, unique within a report.
        type: Category of the item.
        label: Short human-readable name, e.g. ``"File SHA-256"``.
        value: The finding, rendered as text. Kept a string so any value type can be
            recorded without a schema change per type.
        source: Where the value came from, e.g. the file path or ``"model:<id>"``.
        method: How it was obtained, e.g. ``"opencv.imread"`` or
            ``"sha256(file bytes)"``. Names the tool so the claim can be re-derived.
        timestamp: RFC 3339 UTC time the item was recorded.
        confidence: ``0..1`` when an honest confidence exists, else ``None``. An
            uncalibrated model output has no confidence and must say so rather than
            reuse its raw score as one.
        claim_kind: Whether the item is observed, derived, or inferred.
    """

    id: str
    type: EvidenceType
    label: str
    value: str
    source: str
    method: str
    timestamp: str
    confidence: float | None = None
    claim_kind: ClaimKind = ClaimKind.OBSERVED

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"evidence confidence must be within [0, 1] or None, got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "value": self.value,
            "source": self.source,
            "method": self.method,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "claim_kind": self.claim_kind.value,
        }


def make_evidence_item(
    counter: Any,
    *,
    evidence_type: EvidenceType,
    label: str,
    value: str,
    source: str,
    method: str,
    claim_kind: ClaimKind,
    timestamp: str,
    confidence: float | None = None,
) -> EvidenceItem:
    """Build an :class:`EvidenceItem` with a sequential ``ev-NNN`` identifier.

    Shared by the modality analyzers so their evidence is numbered and shaped
    identically; ``counter`` is an ``itertools.count`` owned by the caller, which keeps
    the numbering per-report.
    """
    return EvidenceItem(
        id=f"ev-{next(counter):03d}",
        type=evidence_type,
        label=label,
        value=value,
        source=source,
        method=method,
        timestamp=timestamp,
        confidence=confidence,
        claim_kind=claim_kind,
    )
