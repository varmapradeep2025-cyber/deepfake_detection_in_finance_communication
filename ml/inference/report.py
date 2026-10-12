"""The analysis report: everything one request produced, including what it could not.

A report is the unit of output this project stands behind, so it records not only the
model's score but the status that says whether a score exists at all, the evidence
behind every statement, and — when the answer is "no model was available" — a clear
statement of that fact and what would change it. The fields mirror the required
outputs of an analysis: status, modality, model identity and version, the model's
actual output, the score's semantics and limitations, a risk assessment only when one
is supported, preprocessing warnings, processing errors, and a standing recommendation
to verify independently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ml.inference.evidence import EvidenceItem
from ml.inference.modality import AnalysisModality
from ml.inference.models import ModelDescriptor, ModelOutput
from ml.inference.status import AnalysisStatus
from ml.risk.assess import RiskAssessment

__all__ = [
    "DEFAULT_RECOMMENDATION",
    "AnalysisReport",
]

#: Attached to every report. Deepfake detection produces a signal, not a verdict,
#: and a financial communication carries consequences that a single model score
#: cannot justify acting on alone.
DEFAULT_RECOMMENDATION = (
    "Treat this output as a supporting signal, not proof. Before acting on a financial "
    "communication, verify the source through an independent channel and have a human "
    "review the original media."
)


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """The complete, self-describing result of one analysis request.

    Attributes:
        request_id: Identifier for this request, echoed into every record.
        status: How the request ended; the authority on whether a score exists.
        modality: The media type or communication analyzed.
        path: The artifact that was analyzed, as given.
        model: The model card used, or ``None`` when no model was involved.
        output: The model's output, or ``None``. Present only when a model produced
            one; some analyzers (QR decoding) complete without a model and so leave
            this ``None`` while ``status`` is
            :attr:`~ml.inference.status.AnalysisStatus.ANALYSIS_COMPLETE`.
        risk: The risk assessment, or ``None``. Present only when a score supported
            one; an unavailable model yields no risk assessment rather than a default.
        evidence: Observed, derived, and inferred items behind the report.
        warnings: Non-fatal observations (a missing decode field, an uncalibrated
            score, and so on).
        errors: What prevented completion. Empty on success.
        recommendation: The standing advice to verify independently.
        started_at, completed_at: RFC 3339 UTC timestamps.
        preprocessing: Serialized preprocessing summary, or ``None`` when the input
            never reached preprocessing.
        result: Structured output that is not a model output, such as the decoded QR
            payload and inspected destination URL, or the probed video stream
            properties. ``None`` when the analyzer produced no such record.
    """

    request_id: str
    status: AnalysisStatus
    modality: AnalysisModality
    path: str
    model: ModelDescriptor | None = None
    output: ModelOutput | None = None
    risk: RiskAssessment | None = None
    evidence: tuple[EvidenceItem, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    recommendation: str = DEFAULT_RECOMMENDATION
    started_at: str = ""
    completed_at: str = ""
    preprocessing: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        """Whether the analysis completed.

        This says the workflow ran to completion, not that a model produced a score:
        a QR analysis completes without a model. Use :attr:`has_score` to ask whether
        a detector score exists.
        """
        return self.status is AnalysisStatus.ANALYSIS_COMPLETE

    @property
    def has_score(self) -> bool:
        """Whether a model produced an output for this request."""
        return self.output is not None and self.output.raw_score is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form.

        Field names are snake_case, consistent with the rest of the repository's
        schemas. The correspondence to the frontend's camelCase types is documented in
        ``docs/inference.md``.
        """
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "modality": self.modality.value,
            "path": self.path,
            "model": self.model.to_dict() if self.model is not None else None,
            "output": self.output.to_dict() if self.output is not None else None,
            "risk": self.risk.to_dict() if self.risk is not None else None,
            "evidence": [item.to_dict() for item in self.evidence],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "recommendation": self.recommendation,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "preprocessing": dict(self.preprocessing) if self.preprocessing else None,
            "result": dict(self.result) if self.result else None,
        }
