"""Audio analysis: real recording properties, and an honest detection gap.

The foundation-stage checkout ships no audio detector and no general audio decoder:
the only decoder wired in is the standard library's WAV reader, so a WAV can be
inspected (duration, sample rate, channels, sample width) while compressed formats
such as MP3 or FLAC cannot be opened at all. Both facts are reported rather than
papered over. No waveform, spectrogram, or speaker claim is produced.

Status mapping:

* **invalid_input** — the file is missing, empty, unreadable, or over the limit.
* **capability_unavailable** — no audio decoder is installed, or the file needs a
  decoder the environment does not have (the stdlib reader handles WAV only), or an
  audio model is registered but feature extraction is not implemented.
* **model_unavailable** — the file was read and no audio detector is registered.
* **analysis_complete** — reserved for when a model scores the audio; unreachable here.
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
    "DEFAULT_MAX_AUDIO_BYTES",
    "AudioAnalyzer",
]

#: Default input ceiling for an audio recording, 100 MiB.
DEFAULT_MAX_AUDIO_BYTES = 100 * 1024 * 1024

#: Inspection states that mean "we cannot decode this container", not "the file is
#: broken". A valid MP3 the stdlib reader cannot open lands here.
_CAPABILITY_STATES = frozenset(
    {InspectionStatus.NO_DECODER, InspectionStatus.DECODE_FAILED}
)


class AudioAnalyzer:
    """Read an audio file's properties and report detection availability honestly.

    Args:
        inspector: Media inspector with an audio decoder for the property probe. The
            default stdlib decoder handles WAV only.
        registry: Model registry used to report whether an audio detector exists.
        max_bytes: Input ceiling in bytes.
        clock: Clock returning RFC 3339 UTC strings; injected for deterministic tests.
    """

    def __init__(
        self,
        *,
        inspector: MediaInspector,
        registry: ModelRegistry,
        max_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._inspector = inspector
        self._registry = registry
        self._max_bytes = max_bytes
        self._clock = clock

    @classmethod
    def from_project_config(cls, config: Any | None = None) -> AudioAnalyzer:
        """Build the analyzer from ``configs/project.yaml``."""
        if config is None:
            from ml.reproducibility.config import load_project_config

            config = load_project_config()

        artifacts_root = config.get_path("paths.generated.artifacts")
        registry = ModelRegistry.discover(Path(artifacts_root) / "models")
        return cls(inspector=MediaInspector.from_project_config(config), registry=registry)

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Analyze the recording named by ``request`` and return a report."""
        request_id = new_request_id()
        started = self._clock()
        path = request.path
        evidence: list[EvidenceItem] = []
        warnings: list[str] = list(self._registry.problems)
        errors: list[str] = []
        counter = itertools.count(1)

        inspection = self._inspector.inspect(path, modality=Modality.AUDIO)
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
                if inspection.status in _CAPABILITY_STATES
                else AnalysisStatus.INVALID_INPUT
            )
            errors.append(
                inspection.error or f"inspection failed: {inspection.status.value}"
            )
            if status is AnalysisStatus.CAPABILITY_UNAVAILABLE:
                warnings.append(
                    "only uncompressed WAV is decodable in this environment; "
                    "compressed audio formats need a decoder that is not installed"
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

        properties = self._property_evidence(inspection, path, counter, evidence)
        if inspection.unavailable_fields:
            warnings.append(
                "audio properties not reported by the decoder: "
                + ", ".join(inspection.unavailable_fields)
            )

        candidates = self._registry.for_modality(Modality.AUDIO)
        if not candidates:
            errors.append(
                "no audio model is registered; add a model card at "
                f"{self._registry.models_dir / '<model_id>' / 'model.json'}"
            )
            return self._report(
                request_id, AnalysisStatus.MODEL_UNAVAILABLE, path, started, evidence,
                warnings, errors, result=properties,
            )

        warnings.append(
            "an audio model is registered but acoustic feature extraction is not "
            "implemented; only file properties are reported and no score is produced"
        )
        return self._report(
            request_id, AnalysisStatus.CAPABILITY_UNAVAILABLE, path, started, evidence,
            warnings, errors, result=properties,
        )

    # --- internals --------------------------------------------------------

    def _property_evidence(
        self,
        inspection: Any,
        path: str,
        counter: itertools.count[int],
        evidence: list[EvidenceItem],
    ) -> dict[str, Any]:
        """Append observed audio-property evidence and return the structured summary."""
        method = f"decoder:{self._inspector.decoder_name}"
        observed = (
            ("Container format", inspection.container_format),
            ("Codec", inspection.codec),
            ("Duration", self._format_duration(inspection.duration_seconds)),
            ("Sample rate", self._format_rate(inspection.sample_rate)),
            ("Channels", self._format_channels(inspection.channels)),
            ("Sample width", self._format_width(inspection.sample_width_bits)),
            ("Sample count", self._format_count(inspection.sample_count)),
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
            "duration_seconds": inspection.duration_seconds,
            "sample_rate_hz": inspection.sample_rate,
            "channels": inspection.channels,
            "sample_width_bits": inspection.sample_width_bits,
            "sample_count": inspection.sample_count,
            "unavailable_fields": list(inspection.unavailable_fields),
        }

    @staticmethod
    def _format_duration(seconds: float | None) -> str | None:
        return None if seconds is None else f"{seconds:.3f} s"

    @staticmethod
    def _format_rate(rate: int | None) -> str | None:
        return None if rate is None else f"{rate} Hz"

    @staticmethod
    def _format_count(count: int | None) -> str | None:
        return None if count is None else f"{count} sample(s)"

    @staticmethod
    def _format_channels(channels: int | None) -> str | None:
        return None if channels is None else f"{channels} channel(s)"

    @staticmethod
    def _format_width(bits: int | None) -> str | None:
        return None if bits is None else f"{bits}-bit"

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
            modality=AnalysisModality.AUDIO,
            path=path,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            errors=tuple(errors),
            recommendation=DEFAULT_RECOMMENDATION,
            started_at=started,
            completed_at=self._clock(),
            result=result,
        )
