"""QR-code analysis: decode the payload, then grade it if it is a link.

A QR image is analyzed in two genuinely separable steps:

1. **Decode** the payload(s) from the image. This is observed content — the image either
   contains a code or it does not.
2. **Analyze** each payload. A payload that is an ``http(s)`` URL is run through
   :func:`ml.url.analyze.analyze_url`, whose heuristic findings (IP-literal host,
    punycode, embedded userinfo, suspicious TLD, phishing keywords) are recorded as
    *inferred* evidence. Any other payload (plain text, a ``WIFI:`` or ``mailto:``
    string) is recorded as-is and marked as not-a-URL.

A QR analysis therefore completes **without a model**: decoding is the analysis. The
report's status is ``analysis_complete`` even though it carries no detector score, and
``has_score`` is ``False``. This is the one modality where completion and scoring are
genuinely independent, which is why the report separates them.

No network request is ever made: the URL is graded on its text only. A benign-looking
result means no known pattern matched, not that the link is safe.

Status mapping:

* **invalid_input** — the file is missing, empty, over the limit, or not an image.
* **capability_unavailable** — no QR decoder is installed.
* **analysis_complete** — decoding ran; ``result`` lists every payload found (possibly
  none).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets.preprocessing.inspection import InspectionStatus, MediaInspector
from datasets.schema import Modality
from ml.inference.evidence import (
    ClaimKind,
    EvidenceItem,
    EvidenceType,
    make_evidence_item,
)
from ml.inference.modality import AnalysisModality
from ml.inference.report import DEFAULT_RECOMMENDATION, AnalysisReport
from ml.inference.request import AnalysisRequest, new_request_id, utc_now
from ml.inference.status import AnalysisStatus
from ml.qr.decode import QrDecoder
from ml.url.analyze import analyze_url, looks_like_url

__all__ = [
    "DEFAULT_MAX_QR_BYTES",
    "QrAnalyzer",
]

#: Default input ceiling for a QR image, 10 MiB.
DEFAULT_MAX_QR_BYTES = 10 * 1024 * 1024

#: Maximum number of QR codes to record and analyze.
MAX_QR_CODES = 20

#: Maximum length of payload string stored in evidence/value representation.
MAX_PAYLOAD_LENGTH = 2000


class QrAnalyzer:
    """Decode QR payloads from an image and grade any URLs among them.

    Args:
        inspector: Media inspector used for file identity (hash, size) and image
            dimensions. Its image decoder is not used for decoding the QR itself.
        decoder: QR decoder, or ``None`` when none is available; a missing decoder is
            reported as a capability gap.
        max_bytes: Input ceiling in bytes.
        clock: Clock returning RFC 3339 UTC strings; injected for deterministic tests.
    """

    def __init__(
        self,
        *,
        inspector: MediaInspector,
        decoder: QrDecoder | None,
        max_bytes: int = DEFAULT_MAX_QR_BYTES,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._inspector = inspector
        self._decoder = decoder
        self._max_bytes = max_bytes
        self._clock = clock

    @classmethod
    def from_project_config(cls, config: Any | None = None) -> QrAnalyzer:
        """Build the analyzer from ``configs/project.yaml``."""
        if config is None:
            from ml.reproducibility.config import load_project_config

            config = load_project_config()

        from ml.qr.decode import default_qr_decoder

        return cls(
            inspector=MediaInspector.from_project_config(config),
            decoder=default_qr_decoder(),
        )

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Decode the QR image named by ``request`` and return a report."""
        request_id = new_request_id()
        started = self._clock()
        path = request.path
        evidence: list[EvidenceItem] = []
        warnings: list[str] = []
        errors: list[str] = []
        counter = itertools.count(1)

        inspection = self._inspector.inspect(path, modality=Modality.IMAGE)
        if inspection.sha256 is not None:
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.HASH,
                    label="File SHA-256",
                    value=inspection.sha256,
                    source=path,
                    method="sha256(file bytes)",
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
        if inspection.file_size is not None:
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label="File size",
                    value=f"{inspection.file_size} bytes",
                    source=path,
                    method="stat",
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
        warnings.extend(inspection.warnings)

        if not inspection.is_usable:
            status = (
                AnalysisStatus.CAPABILITY_UNAVAILABLE
                if inspection.status is InspectionStatus.NO_DECODER
                else AnalysisStatus.INVALID_INPUT
            )
            errors.append(
                inspection.error or f"inspection failed: {inspection.status.value}"
            )
            return self._report(
                request_id, status, path, started, evidence, warnings, errors
            )

        if inspection.file_size is not None and inspection.file_size > self._max_bytes:
            errors.append(
                f"file is {inspection.file_size} bytes, over the {self._max_bytes}-byte limit"
            )
            return self._report(
                request_id, AnalysisStatus.INVALID_INPUT, path, started, evidence,
                warnings, errors,
            )

        if inspection.resolution is not None:
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label="Image dimensions",
                    value=inspection.resolution,
                    source=path,
                    method=f"decoder:{self._inspector.image_decoder_name}",
                    claim_kind=ClaimKind.OBSERVED,
                )
            )

        if self._decoder is None:
            errors.append(
                "no QR decoder is installed; install opencv-python-headless or inject a "
                "QrDecoder to decode this image"
            )
            return self._report(
                request_id, AnalysisStatus.CAPABILITY_UNAVAILABLE, path, started,
                evidence, warnings, errors,
            )

        try:
            payloads = self._decoder.decode(Path(path))
        except Exception as exc:
            errors.append(f"QR decoder {self._decoder.name!r} failed on {path}: {exc}")
            return self._report(
                request_id, AnalysisStatus.ANALYSIS_FAILED, path, started, evidence,
                warnings, errors,
            )
        codes = self._record_codes(payloads, path, counter, evidence, warnings)

        if not codes:
            warnings.append(
                "no QR code was detected in the image; this is a complete result, not a failure"
            )

        result = {
            "code_count": len(codes),
            "url_count": sum(1 for code in codes if code["kind"] == "url"),
            "codes": codes,
        }
        return self._report(
            request_id, AnalysisStatus.ANALYSIS_COMPLETE, path, started, evidence,
            warnings, errors, result=result,
        )

    # --- internals --------------------------------------------------------

    def _record_codes(
        self,
        payloads: tuple[str, ...],
        path: str,
        counter: itertools.count[int],
        evidence: list[EvidenceItem],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """Turn decoded payloads into result entries, recording evidence as it goes."""
        codes: list[dict[str, Any]] = []
        method = f"decoder:{self._decoder.name if self._decoder else 'none'}"
        seen: set[str] = set()
        unique_payloads: list[str] = []
        for p in payloads:
            if p not in seen:
                seen.add(p)
                unique_payloads.append(p)
        payloads_list = unique_payloads
        if len(payloads_list) > MAX_QR_CODES:
            warnings.append(
                f"excessive QR code count detected ({len(payloads_list)}); truncating to {MAX_QR_CODES}"
            )
            payloads_list = payloads_list[:MAX_QR_CODES]
        for index, payload in enumerate(payloads_list, start=1):
            value = payload
            if len(value) > MAX_PAYLOAD_LENGTH:
                value = value[:MAX_PAYLOAD_LENGTH] + "... (truncated)"
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label=f"QR payload #{index}",
                    value=value,
                    source=path,
                    method=method,
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
            if not looks_like_url(payload):
                warnings.append(
                    f"QR payload #{index} is not an http(s) URL; recorded as text and not graded"
                )
                codes.append({"payload": payload, "kind": "text", "url": None})
                continue
            try:
                analysis = analyze_url(payload)
            except ValueError as exc:
                warnings.append(
                    f"QR payload #{index} looked like a URL but could not be parsed: {exc}"
                )
                codes.append({"payload": payload, "kind": "text", "url": None})
                continue
            for finding in analysis.findings:
                evidence.append(
                    self._item(
                        counter,
                        evidence_type=EvidenceType.SIGNAL,
                        label=f"URL signal: {finding.code}",
                        value=finding.message,
                        source=payload,
                        method="ml.url.analyze_url",
                        claim_kind=ClaimKind.INFERRED,
                    )
                )
            codes.append(
                {"payload": payload, "kind": "url", "url": analysis.to_dict()}
            )
        return codes


    def _item(
        self,
        counter: itertools.count[int],
        *,
        evidence_type: EvidenceType,
        label: str,
        value: str,
        source: str,
        method: str,
        claim_kind: ClaimKind,
        confidence: float | None = None,
    ) -> EvidenceItem:
        return make_evidence_item(
            counter,
            evidence_type=evidence_type,
            label=label,
            value=value,
            source=source,
            method=method,
            claim_kind=claim_kind,
            timestamp=self._clock(),
            confidence=confidence,
        )

    def _report(
        self,
        request_id: str,
        status: AnalysisStatus,
        path: str,
        started: str,
        evidence: list[EvidenceItem],
        warnings: list[str],
        errors: list[str],
        *,
        result: dict[str, Any] | None = None,
    ) -> AnalysisReport:
        return AnalysisReport(
            request_id=request_id,
            status=status,
            modality=AnalysisModality.QR,
            path=path,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            errors=tuple(errors),
            recommendation=DEFAULT_RECOMMENDATION,
            started_at=started,
            completed_at=self._clock(),
            result=result,
        )
