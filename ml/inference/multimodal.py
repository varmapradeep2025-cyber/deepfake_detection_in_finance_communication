"""Assemble a single report from several analyses — without fusing their scores.

Why there is no combined score
------------------------------
An image detector explains a manipulation score, a video probe explains stream
properties, an audio probe explains recording properties, and a QR read explains a decode
and some URL heuristics. These are different measurements on different scales with
different meanings. Averaging them, or summing "risk", would manufacture a number no
component supports and present it as the finding. This module therefore **never**
produces a combined score or a fused risk level: it presents the analyses side by side,
summarizes their coverage and status, and collects the notable per-analysis findings.

What it does produce
--------------------
A :class:`MultimodalReport` that says, honestly, which modalities were included, how each
one ended, which produced a score and which did not, and — as a clearly labelled list of
per-record observations, not a single verdict — what stands out. Reading it should never
require knowing that a component was missing or unscored; those facts are in the report.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ml.inference.records import AnalysisRecord
from ml.inference.request import new_request_id, utc_now
from ml.inference.status import AnalysisStatus
from ml.risk.assess import RiskLevel

__all__ = [
    "DEFAULT_MULTIMODAL_DISCLAIMER",
    "DEFAULT_MULTIMODAL_TITLE",
    "MultimodalReport",
    "MultimodalReportBuilder",
    "ReportSection",
]

#: Default heading for an assembled report.
DEFAULT_MULTIMODAL_TITLE = "Multimodal analysis report"

#: Stamped on every assembled report. States plainly that nothing was fused.
DEFAULT_MULTIMODAL_DISCLAIMER = (
    "This report collects independent per-modality analyses. Their scores are not "
    "combined, averaged, or summed, because each modality measures a different thing on "
    "a different scale; a higher count of findings is not a higher probability. Read each "
    "section on its own terms, and treat the whole as supporting signals, not proof."
)

#: Risk levels that deserve a top-level mention.
_ELEVATED_RISK_LEVELS = frozenset({RiskLevel.ELEVATED, RiskLevel.HIGH})

#: URL finding severities that deserve a top-level mention.
_NOTABLE_URL_SEVERITIES = frozenset({"high"})


@dataclass(frozen=True, slots=True)
class ReportSection:
    """One record's contribution to a multimodal report.

    Attributes:
        report_id: Identifier of the source analysis.
        source_name: Display name of the analyzed artifact.
        modality: The analysis modality value.
        status: The analysis status value.
        has_score: Whether a model score is present.
        risk_level: The risk level value, or ``None`` when none was assessed.
        headline: A one-line, human-readable summary of the outcome.
        evidence_count: Number of evidence items behind the analysis.
        warning_count: Number of warnings the analysis raised.
        error_count: Number of errors the analysis raised.
        notable: Per-record observations worth surfacing (a high risk, a high-severity
            URL finding, an error). Empty when nothing stood out.
    """

    report_id: str
    source_name: str
    modality: str
    status: str
    has_score: bool
    risk_level: str | None
    headline: str
    evidence_count: int
    warning_count: int
    error_count: int
    notable: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "source_name": self.source_name,
            "modality": self.modality,
            "status": self.status,
            "has_score": self.has_score,
            "risk_level": self.risk_level,
            "headline": self.headline,
            "evidence_count": self.evidence_count,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "notable": list(self.notable),
        }


@dataclass(frozen=True, slots=True)
class MultimodalReport:
    """An assembled, non-fusing report over several analyses.

    Attributes:
        report_id: Identifier for this assembled report.
        title: Heading.
        generated_at: RFC 3339 UTC time of assembly.
        sections: One section per included analysis, in the order supplied.
        modality_counts: How many analyses of each modality were included.
        status_counts: How many analyses ended with each status value.
        scored_count: How many analyses produced a model score.
        unscored_count: How many did not.
        notable_findings: Collected notable observations across all sections.
        disclaimer: The statement that scores were not fused.
    """

    report_id: str
    title: str
    generated_at: str
    sections: tuple[ReportSection, ...]
    modality_counts: Mapping[str, int]
    status_counts: Mapping[str, int]
    scored_count: int
    unscored_count: int
    notable_findings: tuple[str, ...]
    disclaimer: str

    @property
    def record_count(self) -> int:
        """Number of analyses included."""
        return len(self.sections)

    def section_for(self, report_id: str) -> ReportSection | None:
        """Return the section for a given source analysis, or ``None``."""
        for section in self.sections:
            if section.report_id == report_id:
                return section
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "title": self.title,
            "generated_at": self.generated_at,
            "sections": [section.to_dict() for section in self.sections],
            "modality_counts": dict(self.modality_counts),
            "status_counts": dict(self.status_counts),
            "scored_count": self.scored_count,
            "unscored_count": self.unscored_count,
            "notable_findings": list(self.notable_findings),
            "disclaimer": self.disclaimer,
        }


class MultimodalReportBuilder:
    """Build a :class:`MultimodalReport` from stored analysis records.

    Args:
        clock: Clock returning RFC 3339 UTC strings; injected for deterministic tests.
        title: Heading for the assembled report.
        disclaimer: Fusing disclaimer text.

    The builder reads only the records it is given, so a caller decides exactly which
    analyses are included — including none of a modality whose result is not wanted.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], str] = utc_now,
        title: str = DEFAULT_MULTIMODAL_TITLE,
        disclaimer: str = DEFAULT_MULTIMODAL_DISCLAIMER,
    ) -> None:
        self._clock = clock
        self._title = title
        self._disclaimer = disclaimer

    def build(self, records: Sequence[AnalysisRecord]) -> MultimodalReport:
        """Assemble a report from ``records``.

        Raises:
            ValueError: If ``records`` is empty. An assembled report over nothing has no
                sections to present and would be a misleading artifact.
        """
        if not records:
            raise ValueError("a multimodal report requires at least one analysis record")

        sections = tuple(self._section(record) for record in records)
        modality_counts = Counter(section.modality for section in sections)
        status_counts = Counter(section.status for section in sections)
        scored = sum(1 for section in sections if section.has_score)
        notable = tuple(
            f"{section.source_name} ({section.modality}): {item}"
            for section in sections
            for item in section.notable
        )

        return MultimodalReport(
            report_id=new_request_id(),
            title=self._title,
            generated_at=self._clock(),
            sections=sections,
            modality_counts=dict(sorted(modality_counts.items())),
            status_counts=dict(sorted(status_counts.items())),
            scored_count=scored,
            unscored_count=len(sections) - scored,
            notable_findings=notable,
            disclaimer=self._disclaimer,
        )

    # --- internals --------------------------------------------------------

    def _section(self, record: AnalysisRecord) -> ReportSection:
        risk_level = record.risk_level
        notable = self._notable(record, risk_level)
        return ReportSection(
            report_id=record.report_id,
            source_name=record.source_name,
            modality=record.modality,
            status=record.status,
            has_score=record.has_score,
            risk_level=risk_level,
            headline=self._headline(record.status, record.has_score, risk_level),
            evidence_count=self._count(record, "evidence"),
            warning_count=len(record.warnings),
            error_count=len(record.errors),
            notable=notable,
        )

    @staticmethod
    def _headline(status: str, has_score: bool, risk_level: str | None) -> str:
        if status == AnalysisStatus.ANALYSIS_COMPLETE.value:
            if has_score:
                if risk_level is not None and risk_level != RiskLevel.UNKNOWN.value:
                    return f"Analyzed; detector score produced, risk assessed as {risk_level}."
                return "Analyzed; detector score produced (no risk level assessed)."
            return (
                "Completed without a detector score, e.g. a QR decode; "
                "completion and scoring are separate."
            )
        if status == AnalysisStatus.MODEL_UNAVAILABLE.value:
            return "No model was available for this modality, so nothing was scored."
        if status == AnalysisStatus.CAPABILITY_UNAVAILABLE.value:
            return "A required capability is missing in this environment; analysis did not run."
        if status == AnalysisStatus.UNSUPPORTED_MODALITY.value:
            return "No analyzer is implemented for this modality."
        if status == AnalysisStatus.INVALID_INPUT.value:
            return "The input could not be read or decoded."
        if status == AnalysisStatus.ANALYSIS_FAILED.value:
            return "The analysis backend failed partway through."
        return f"Completed with status {status!r}."

    def _notable(self, record: AnalysisRecord, risk_level: str | None) -> tuple[str, ...]:
        notable: list[str] = []
        if risk_level is not None:
            try:
                level = RiskLevel.parse(risk_level)
            except ValueError:
                level = None
            if level is not None and level in _ELEVATED_RISK_LEVELS:
                notable.append(f"risk assessed as {level.value}")
        notable.extend(self._url_notables(record))
        for error in record.errors:
            notable.append(f"error: {error}")
        return tuple(notable)

    @staticmethod
    def _url_notables(record: AnalysisRecord) -> list[str]:
        """High-severity URL findings among any decoded QR codes."""
        result = record.report.get("result")
        if not isinstance(result, Mapping):
            return []
        codes = result.get("codes")
        if not isinstance(codes, list):
            return []
        notables: list[str] = []
        for index, code in enumerate(codes, start=1):
            if not isinstance(code, Mapping):
                continue
            url = code.get("url")
            if not isinstance(url, Mapping):
                continue
            findings = url.get("findings")
            if not isinstance(findings, list):
                continue
            for finding in findings:
                if not isinstance(finding, Mapping):
                    continue
                severity = finding.get("severity")
                code_name = finding.get("code")
                if isinstance(severity, str) and severity in _NOTABLE_URL_SEVERITIES:
                    notables.append(
                        f"QR payload #{index}: {severity}-severity URL finding {code_name!r}"
                    )
        return notables

    @staticmethod
    def _count(record: AnalysisRecord, key: str) -> int:
        value = record.report.get(key)
        return len(value) if isinstance(value, list) else 0
