"""An analysis request and the small helpers every request needs.

Kept separate from the report so the analyzer and the service can share the request
type without importing each other.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ml.inference.modality import AnalysisModality

__all__ = [
    "AnalysisRequest",
    "new_request_id",
    "utc_now",
]


def utc_now() -> str:
    """Current time as an RFC 3339 UTC string, seconds precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_request_id() -> str:
    """A fresh, collision-resistant request identifier."""
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """One request to analyze one artifact.

    Attributes:
        path: The artifact to analyze.
        modality: The analysis workflow. Defaults to ``IMAGE`` because image is the
            first modality with a real analyzer.
        model_id: A specific model to use, or ``None`` to select the first registered
            model for the modality. Naming a missing model is reported as
            ``model_unavailable`` rather than silently falling back.
        actor: Who requested the analysis, for the audit trail. Optional.
        note: Free-text note carried into the report's provenance. Optional.
    """

    path: str
    modality: AnalysisModality = AnalysisModality.IMAGE
    model_id: str | None = None
    actor: str | None = None
    note: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> AnalysisRequest:
        """Build a request from a loosely typed mapping, e.g. parsed JSON.

        Raises:
            ValueError: If ``path`` is missing.
        """
        raw_path = payload.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("analysis request requires a non-empty 'path'")
        raw_modality = payload.get("modality", AnalysisModality.IMAGE)
        model_id = payload.get("model_id")
        actor = payload.get("actor")
        note = payload.get("note")
        return cls(
            path=raw_path.strip(),
            modality=AnalysisModality.parse(raw_modality),
            model_id=model_id if isinstance(model_id, str) else None,
            actor=actor if isinstance(actor, str) else None,
            note=note if isinstance(note, str) else None,
        )
