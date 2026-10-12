"""Video analysis: real stream metadata, and an honest gap where detection would be.

What this analyzer does, and does not, do
-----------------------------------------
It reads what the container actually reports — format, codec, dimensions, duration,
declared frame count, frame rate — and records each as *observed* evidence, alongside
the file's digest and size. It does **not** decode frames for a detector, sample
timestamps, or claim any frame is suspicious. Those require a video detector and a
frame-preprocessing transform, neither of which exists yet, so the report says so
rather than inventing them.

The failure taxonomy mirrors the image analyzer:

* **invalid_input** — the file is missing, empty, over the limit, or not openable as
  video.
* **capability_unavailable** — the file was read but no video decoder is installed, or
  a video model is registered but frame preprocessing is not implemented.
* **model_unavailable** — the file was read and no video detector is registered.
* **analysis_complete** — reserved for when a model actually scores the clip; not
  reachable in the foundation-stage checkout, which ships no weights.

Observed metadata is captured before any decision, so a request answered with
"no model" still records the clip's digest, size, and stream properties.
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
from ml.inference.models import ModelRegistry
from ml.inference.report import DEFAULT_RECOMMENDATION, AnalysisReport
from ml.inference.request import AnalysisRequest, new_request_id, utc_now
from ml.inference.status import AnalysisStatus

__all__ = [
    "DEFAULT_MAX_VIDEO_BYTES",
    "VideoAnalyzer",
]

#: Default input ceiling for a video clip, 250 MiB. Matches the frontend's video limit.
DEFAULT_MAX_VIDEO_BYTES = 250 * 1024 * 1024


class VideoAnalyzer:
    """Read a clip's stream properties and report detection availability honestly.

    Args:
        inspector: Media inspector with a video decoder for the stream probe. With no
            decoder it still hashes and sizes the file and reports ``no_decoder``.
        registry: Model registry used to report whether a video detector exists.
        max_bytes: Input ceiling in bytes.
        clock: Clock returning RFC 3339 UTC strings; injected for deterministic tests.
    """

    def __init__(
        self,
        *,
        inspector: MediaInspector,
        registry: ModelRegistry,
        max_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._inspector = inspector
        self._registry = registry
        self._max_bytes = max_bytes
        self._clock = clock

    @classmethod
    def from_project_config(cls, config: Any | None = None) -> VideoAnalyzer:
        """Build the analyzer from ``configs/project.yaml``."""
        if config is None:
            from ml.reproducibility.config import load_project_config

            config = load_project_config()

        artifacts_root = config.get_path("paths.generated.artifacts")
        registry = ModelRegistry.discover(Path(artifacts_root) / "models")
        return cls(inspector=MediaInspector.from_project_config(config), registry=registry)

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Analyze the clip named by ``request`` and return a report."""
        request_id = new_request_id()
        started = self._clock()
        path = request.path
        evidence: list[EvidenceItem] = []
        warnings: list[str] = list(self._registry.problems)
        errors: list[str] = []
        counter = itertools.count(1)

        inspection = self._inspector.inspect(path, modality=Modality.VIDEO)
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

        stream = self._stream_evidence(inspection, path, counter, evidence)
        if inspection.unavailable_fields:
            warnings.append(
                "stream properties not reported by the decoder: "
                + ", ".join(inspection.unavailable_fields)
            )

        candidates = self._registry.for_modality(Modality.VIDEO)
        if not candidates:
            errors.append(
                "no video model is registered; add a model card at "
                f"{self._registry.models_dir / '<model_id>' / 'model.json'}"
            )
            return self._report(
                request_id, AnalysisStatus.MODEL_UNAVAILABLE, path, started, evidence,
                warnings, errors, result=stream,
            )

        warnings.append(
            "a video model is registered but frame preprocessing for video detection is "
            "not implemented; only stream metadata is reported and no score is produced"
        )
        return self._report(
            request_id, AnalysisStatus.CAPABILITY_UNAVAILABLE, path, started, evidence,
            warnings, errors, result=stream,
        )

    # --- internals --------------------------------------------------------

    def _stream_evidence(
        self,
        inspection: Any,
        path: str,
        counter: itertools.count[int],
        evidence: list[EvidenceItem],
    ) -> dict[str, Any]:
        """Append observed stream-property evidence and return the structured summary."""
        method = f"decoder:{self._inspector.decoder_name}"
        observed = (
            ("Container format", inspection.container_format),
            ("Codec", inspection.codec),
            ("Dimensions", inspection.resolution),
            ("Duration", self._format_duration(inspection.duration_seconds)),
            ("Declared frame count", self._format_count(inspection.declared_frame_count)),
            ("Frame rate", self._format_rate(inspection.frame_rate)),
        )
        for label, value in observed:
            if value is None:
                continue
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label=label,
                    value=str(value),
                    source=path,
                    method=method,
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
        return {
            "container_format": inspection.container_format,
            "codec": inspection.codec,
            "width": inspection.width,
            "height": inspection.height,
            "duration_seconds": inspection.duration_seconds,
            "declared_frame_count": inspection.declared_frame_count,
            "frame_rate": inspection.frame_rate,
            "unavailable_fields": list(inspection.unavailable_fields),
        }

    @staticmethod
    def _format_duration(seconds: float | None) -> str | None:
        return None if seconds is None else f"{seconds:.3f} s"

    @staticmethod
    def _format_count(count: int | None) -> str | None:
        return None if count is None else f"{count} frame(s)"

    @staticmethod
    def _format_rate(rate: float | None) -> str | None:
        return None if rate is None else f"{rate:.3f} fps"

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
            modality=AnalysisModality.VIDEO,
            path=path,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            errors=tuple(errors),
            recommendation=DEFAULT_RECOMMENDATION,
            started_at=started,
            completed_at=self._clock(),
            result=result,
        )
