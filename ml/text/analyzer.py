"""Text analysis: honest probe of a text message with no fabricated detection."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from ml.inference.evidence import (
    ClaimKind,
    EvidenceItem,
    EvidenceType,
    make_evidence_item,
)
from ml.inference.modality import AnalysisModality
from ml.inference.models import ModelRegistry
from ml.inference.report import DEFAULT_RECOMMENDATION, AnalysisReport
from ml.inference.request import AnalysisRequest, new_request_id, utc_now
from ml.inference.status import AnalysisStatus
from ml.text.preprocess import inspect_text

__all__ = [
    "DEFAULT_MAX_TEXT_BYTES",
    "TextAnalyzer",
]

DEFAULT_MAX_TEXT_BYTES = 100_000


class TextAnalyzer:
    """Analyze a text message and report only supportable observations."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        max_chars: int = DEFAULT_MAX_TEXT_BYTES,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {max_chars}")
        self._registry = registry
        self._max_chars = max_chars
        self._clock = clock

    @classmethod
    def from_project_config(cls, config: Any | None = None) -> TextAnalyzer:
        if config is None:
            from ml.reproducibility.config import load_project_config
            config = load_project_config()
        from pathlib import Path
        artifacts_root = config.get_path("paths.generated.artifacts")
        registry = ModelRegistry.discover(Path(artifacts_root) / "models")
        return cls(registry=registry)

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        request_id = new_request_id()
        started = self._clock()
        path = request.path
        evidence: list[EvidenceItem] = []
        warnings: list[str] = list(self._registry.problems)
        errors: list[str] = []
        counter = itertools.count(1)
        ts = self._clock()

        text_content: str | None = None
        text_source: str = str(path)

        if getattr(request, "note", None) and isinstance(request.note, str):
            text_content = request.note
            text_source = "request.note"
        else:
            from pathlib import Path
            try:
                p = Path(str(path))
                if p.exists() and p.is_file():
                    raw = p.read_bytes()
                    try:
                        text_content = raw.decode("utf-8")
                    except UnicodeError:
                        text_content = raw.decode("latin-1", errors="replace")
                    text_source = str(p)
            except Exception:
                text_content = None

        if text_content is None:
            if isinstance(path, str) and path:
                text_content = path
                text_source = "request.path (string)"
            else:
                errors.append("text content is missing or empty")
                return AnalysisReport(
                    request_id=request_id,
                    status=AnalysisStatus.INVALID_INPUT,
                    modality=AnalysisModality("text"),
                    path=str(path),
                    model=None,
                    output=None,
                    risk=None,
                    evidence=tuple(evidence),
                    warnings=tuple(warnings),
                    errors=tuple(errors),
                    recommendation=DEFAULT_RECOMMENDATION,
                    started_at=started,
                    completed_at=self._clock(),
                    preprocessing=None,
                    result=None,
                )

        if len(text_content) > self._max_chars:
            errors.append(
                f"text exceeds maximum length ({len(text_content)} > {self._max_chars})"
            )
            return AnalysisReport(
                request_id=request_id,
                status=AnalysisStatus.INVALID_INPUT,
                modality=AnalysisModality("text"),
                path=str(path),
                model=None,
                output=None,
                risk=None,
                evidence=tuple(evidence),
                warnings=tuple(warnings),
                errors=tuple(errors),
                recommendation=DEFAULT_RECOMMENDATION,
                started_at=started,
                completed_at=self._clock(),
                preprocessing=None,
                result=None,
            )

        inspection = inspect_text(text_content)
        if inspection.is_empty:
            errors.append("text is empty after stripping whitespace")
            return AnalysisReport(
                request_id=request_id,
                status=AnalysisStatus.INVALID_INPUT,
                modality=AnalysisModality("text"),
                path=str(path),
                model=None,
                output=None,
                risk=None,
                evidence=tuple(evidence),
                warnings=tuple(warnings),
                errors=tuple(errors),
                recommendation=DEFAULT_RECOMMENDATION,
                started_at=started,
                completed_at=self._clock(),
                preprocessing=None,
                result=None,
            )

        evidence.append(
            make_evidence_item(
                counter,
                evidence_type=EvidenceType.METADATA,
                label="Text source",
                value=text_source,
                source=text_source,
                method="text inspection",
                claim_kind=ClaimKind.OBSERVED,
                timestamp=ts,
            )
        )
        evidence.append(
            make_evidence_item(
                counter,
                evidence_type=EvidenceType.METADATA,
                label="Character count",
                value=str(inspection.character_count),
                source=text_source,
                method="text inspection",
                claim_kind=ClaimKind.OBSERVED,
                timestamp=ts,
            )
        )
        evidence.append(
            make_evidence_item(
                counter,
                evidence_type=EvidenceType.METADATA,
                label="Non-whitespace characters",
                value=str(inspection.non_whitespace_count),
                source=text_source,
                method="text inspection",
                claim_kind=ClaimKind.OBSERVED,
                timestamp=ts,
            )
        )
        evidence.append(
            make_evidence_item(
                counter,
                evidence_type=EvidenceType.METADATA,
                label="Word count",
                value=str(inspection.word_count),
                source=text_source,
                method="text inspection",
                claim_kind=ClaimKind.OBSERVED,
                timestamp=ts,
            )
        )
        evidence.append(
            make_evidence_item(
                counter,
                evidence_type=EvidenceType.METADATA,
                label="Line count",
                value=str(inspection.line_count),
                source=text_source,
                method="text inspection",
                claim_kind=ClaimKind.OBSERVED,
                timestamp=ts,
            )
        )
        if inspection.has_leading_trailing_whitespace:
            evidence.append(
                make_evidence_item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label="Leading/trailing whitespace",
                    value="present",
                    source=text_source,
                    method="text inspection",
                    claim_kind=ClaimKind.OBSERVED,
                    timestamp=ts,
                )
            )

        warnings.extend(self._registry.problems)
        result: dict[str, Any] = {
            "text_source": text_source,
            "character_count": inspection.character_count,
            "non_whitespace_count": inspection.non_whitespace_count,
            "word_count": inspection.word_count,
            "line_count": inspection.line_count,
            "has_leading_trailing_whitespace": inspection.has_leading_trailing_whitespace,
            "is_empty": inspection.is_empty,
        }

        errors_list = list(errors)
        errors_list.append(
            "No text detector is registered or available in this build; "
            "only supportable text characteristics were recorded."
        )

        return AnalysisReport(
            request_id=request_id,
            status=AnalysisStatus.MODEL_UNAVAILABLE,
            modality=AnalysisModality("text"),
            path=str(path),
            model=None,
            output=None,
            risk=None,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            errors=tuple(errors_list),
            recommendation=DEFAULT_RECOMMENDATION,
            started_at=started,
            completed_at=self._clock(),
            preprocessing=None,
            result=result,
        )

