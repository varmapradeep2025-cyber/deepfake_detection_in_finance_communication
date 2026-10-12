"""The analysis service: the single entry point that routes by modality.

Each analysis modality — image, video, audio, and QR — has its own analyzer with its
own honest capabilities. This service holds one analyzer per modality and dispatches
to it, so a caller does not need to know which workflow handles which media type.

When no analyzer is configured for a modality the service returns an
``unsupported_modality`` report rather than raising, so a missing workflow is a normal,
reported state. An unrecognised modality *string* still raises ``ValueError``: that is
a programming error, not a media problem, and letting it through as a report would hide
a typo in a caller.

The analyzers are imported here rather than in ``ml.inference.__init__`` because they
pull in modality packages; ``MediaAnalysisService`` is exposed lazily (PEP 562) so the
core schema can be imported without them.
"""

from __future__ import annotations

from typing import Any

from ml.audio.analyzer import AudioAnalyzer
from ml.image.analyzer import ImageAnalyzer
from ml.inference.modality import AnalysisModality
from ml.inference.report import DEFAULT_RECOMMENDATION, AnalysisReport
from ml.inference.request import AnalysisRequest, new_request_id, utc_now
from ml.inference.status import AnalysisStatus
from ml.qr.analyzer import QrAnalyzer
from ml.video.analyzer import VideoAnalyzer

try:
    from ml.text.analyzer import TextAnalyzer
except Exception:  # pragma: no cover
    TextAnalyzer = None  # type: ignore[assignment,misc]

__all__ = ["MediaAnalysisService"]


class MediaAnalysisService:
    """Routes analysis requests to the analyzer for their modality.

    Args:
        image_analyzer: Analyzer for still images.
        video_analyzer: Analyzer for video clips.
        audio_analyzer: Analyzer for audio recordings.
        qr_analyzer: Analyzer for QR-code images.
        text_analyzer: Analyzer for text messages.

    Every analyzer is optional. A modality without an analyzer yields an explicit
    ``unsupported_modality`` report naming what is missing, never an exception.
    """

    def __init__(
        self,
        *,
        image_analyzer: ImageAnalyzer | None = None,
        video_analyzer: VideoAnalyzer | None = None,
        audio_analyzer: AudioAnalyzer | None = None,
        qr_analyzer: QrAnalyzer | None = None,
        text_analyzer: Any | None = None,
    ) -> None:
        self._analyzers: dict[AnalysisModality, Any] = {
            AnalysisModality.IMAGE: image_analyzer,
            AnalysisModality.VIDEO: video_analyzer,
            AnalysisModality.AUDIO: audio_analyzer,
            AnalysisModality.QR: qr_analyzer,
        }
        if text_analyzer is not None:
            self._analyzers[AnalysisModality.TEXT] = text_analyzer

    @classmethod
    def from_project_config(cls, config: Any | None = None) -> MediaAnalysisService:
        """Build the service with each modality's analyzer wired from project config.

        Image analysis needs an image pixel decoder (OpenCV); when one is absent the
        image analyzer is omitted and image requests report ``unsupported_modality``.
        Video, audio, and QR analyzers construct without a codec and report their own
        capability gaps per request, so they are always built.
        """
        if config is None:
            from ml.reproducibility.config import load_project_config

            config = load_project_config()

        image_analyzer: ImageAnalyzer | None
        try:
            image_analyzer = ImageAnalyzer.from_project_config(config)
        except RuntimeError:
            image_analyzer = None

        text_analyzer: Any | None = None
        if TextAnalyzer is not None:
            try:
                text_analyzer = TextAnalyzer.from_project_config(config)
            except Exception:
                text_analyzer = None

        return cls(
            image_analyzer=image_analyzer,
            video_analyzer=VideoAnalyzer.from_project_config(config),
            audio_analyzer=AudioAnalyzer.from_project_config(config),
            qr_analyzer=QrAnalyzer.from_project_config(config),
            text_analyzer=text_analyzer,
        )

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Analyze the artifact named by ``request`` and return a report.

        Raises:
            ValueError: If ``request.modality`` is not a known analysis modality.
        """
        modality = AnalysisModality.parse(request.modality)
        analyzer = self._analyzers.get(modality)
        if analyzer is None:
            return self._unsupported(
                request,
                modality,
                reason=(
                    f"no analyzer is configured for modality {modality.value!r}; "
                    "construct the service with one, or use "
                    "MediaAnalysisService.from_project_config()"
                ),
            )
        report: AnalysisReport = analyzer.analyze(request)
        return report

    def _unsupported(
        self,
        request: AnalysisRequest,
        modality: AnalysisModality,
        *,
        reason: str,
    ) -> AnalysisReport:
        started = utc_now()
        return AnalysisReport(
            request_id=new_request_id(),
            status=AnalysisStatus.UNSUPPORTED_MODALITY,
            modality=modality,
            path=request.path,
            errors=(reason,),
            recommendation=DEFAULT_RECOMMENDATION,
            started_at=started,
            completed_at=utc_now(),
        )
