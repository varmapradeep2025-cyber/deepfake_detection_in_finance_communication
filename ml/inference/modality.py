"""The analysis modality: the unit of routing for the analysis workflows.

Why this is not :class:`datasets.schema.Modality`
-------------------------------------------------
``datasets.schema.Modality`` describes the media type of a *stored sample* — image,
audio, video — and is pinned to those three values because it is written into
manifests that must stay stable. The analysis workflows are a different space: they
add QR decoding and URL inspection, which are analyses of a communication rather than
media types a dataset can store. Widening the dataset enum to fit a workflow would
let a QR code be written into a manifest as if it were a dataset modality, so the two
are kept apart.

The overlap is deliberate and safe: :class:`AnalysisModality` reuses the exact string
values ``"image"``, ``"audio"``, and ``"video"``, so a value from either enum compares
and serializes identically, and :meth:`AnalysisModality.parse` accepts a
:class:`~datasets.schema.Modality` without a translation table.

QR and URL
----------
``QR`` is a single analysis modality. Decoding a QR code and inspecting the URL it
carries are two steps of one workflow, and the URL analysis is reported as evidence
inside the QR analysis rather than as a peer modality. Decoding a QR code is not, by
itself, a statement that the communication is malicious; the URL findings are kept
separate and labelled as derived heuristics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = ["AnalysisModality"]


class AnalysisModality(str, Enum):
    """Media type or communication analyzed by one workflow."""

    #: A still image, analyzed for manipulation and artifact signals.
    IMAGE = "image"
    #: A video clip, analyzed for temporal consistency.
    VIDEO = "video"
    #: An audio recording, analyzed for speech and vocoder artifacts.
    AUDIO = "audio"
    #: A QR-code image: the code is decoded, and any destination URL is inspected.
    QR = "qr"
    #: A text message (financial communication). Probe-level inspection only.
    TEXT = "text"

    @classmethod
    def parse(cls, value: Any) -> AnalysisModality:
        """Resolve a member from a member or its string value.

        Accepts a :class:`~datasets.schema.Modality` as well, because the three
        dataset media types are also analysis modalities and share their values.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, str):
            candidate = value.strip().lower()
            for member in cls:
                if member.value == candidate:
                    return member
        raise ValueError(
            f"invalid analysis modality {value!r}; expected one of "
            f"{', '.join(member.value for member in cls)}"
        )

    @property
    def is_deepfake_detection(self) -> bool:
        """Whether this modality is an attempt to detect synthesized media.

        ``QR`` is excluded: decoding a QR code and inspecting its URL is a
        communication-safety analysis, not a deepfake detection. ``TEXT`` is
        probe-level only (no deepfake detection implemented).
        """
        return self in (AnalysisModality.IMAGE, AnalysisModality.VIDEO, AnalysisModality.AUDIO)
