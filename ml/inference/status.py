"""Terminal status of one analysis request.

The status is separate from the report's fields on purpose, mirroring
:class:`datasets.preprocessing.inspection.InspectionStatus`. ``score=None`` could
mean the model was not installed, the input was not decodable, or the modality is
not implemented yet; those are different problems with different fixes, so the
status says which one occurred instead of leaving a consumer to infer it from
missing values.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = ["AnalysisStatus"]


class AnalysisStatus(str, Enum):
    """How an analysis request ended."""

    #: A model produced output that passed validation and was recorded.
    ANALYSIS_COMPLETE = "analysis_complete"
    #: No usable model was available (not installed, missing weights, or no entry
    #: point), so no score exists. This is the expected status in a foundation-stage
    #: checkout that ships no model weights.
    MODEL_UNAVAILABLE = "model_unavailable"
    #: The input could not be read or decoded, so there was nothing to analyze.
    INVALID_INPUT = "invalid_input"
    #: An analyzer's backend ran but failed partway (a model raised, or a decoder that
    #: had opened the artifact failed mid-step). Distinct from ``invalid_input``: the
    #: input may be fine and the failure is in the backend.
    ANALYSIS_FAILED = "analysis_failed"
    #: The modality is recognised and an analyzer exists, but a local capability it
    #: needs is missing (for example a QR decoder when OpenCV is not installed, or
    #: frame preprocessing for video detection). Distinct from
    #: ``unsupported_modality``: the analyzer is implemented, the environment is not
    #: equipped to run it. Distinct from ``model_unavailable``: the gap is a decoder
    #: or transform, not a detector.
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    #: The modality is recognised but no analyzer is implemented for it yet.
    UNSUPPORTED_MODALITY = "unsupported_modality"

    @classmethod
    def parse(cls, value: Any) -> AnalysisStatus:
        """Resolve a member from a member or its string value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(f"{value!r} is not a valid analysis status")

    @property
    def completed(self) -> bool:
        """Whether the analysis ran to completion.

        This says the workflow finished, **not** that a model produced a score: a QR
        analysis completes by decoding without any model. Whether a score exists is a
        property of the report (:attr:`~ml.inference.report.AnalysisReport.has_score`),
        not of the status.
        """
        return self is AnalysisStatus.ANALYSIS_COMPLETE
