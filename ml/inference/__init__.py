"""Model-agnostic inference: request, evidence, model registry, and report.

This package holds the parts of an end-to-end analysis that are the same for every
modality. Modality-specific code lives in ``ml/image`` (still images), ``ml/video``
(stream metadata), ``ml/audio`` (recording properties), and ``ml/qr`` (QR decoding plus
URL grading).

``MediaAnalysisService`` — the single entry point that routes by modality — is
exposed lazily (PEP 562) so that importing the core schema here never pulls in a
modality package, which would be a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ml.inference.evidence import ClaimKind, EvidenceItem, EvidenceType
from ml.inference.modality import AnalysisModality
from ml.inference.models import (
    ModelBackend,
    ModelDescriptor,
    ModelDescriptorError,
    ModelOutput,
    ModelRegistry,
    ModelUnavailableError,
    load_model_backend,
)
from ml.inference.multimodal import (
    DEFAULT_MULTIMODAL_DISCLAIMER,
    DEFAULT_MULTIMODAL_TITLE,
    MultimodalReport,
    MultimodalReportBuilder,
    ReportSection,
)
from ml.inference.records import SCHEMA_VERSION, AnalysisRecord, AnalysisStore
from ml.inference.report import DEFAULT_RECOMMENDATION, AnalysisReport
from ml.inference.request import AnalysisRequest, new_request_id, utc_now
from ml.inference.status import AnalysisStatus

if TYPE_CHECKING:
    from ml.inference.service import MediaAnalysisService

__all__ = [
    "DEFAULT_MULTIMODAL_DISCLAIMER",
    "DEFAULT_MULTIMODAL_TITLE",
    "DEFAULT_RECOMMENDATION",
    "SCHEMA_VERSION",
    "AnalysisModality",
    "AnalysisRecord",
    "AnalysisReport",
    "AnalysisRequest",
    "AnalysisStatus",
    "AnalysisStore",
    "ClaimKind",
    "EvidenceItem",
    "EvidenceType",
    "MediaAnalysisService",
    "ModelBackend",
    "ModelDescriptor",
    "ModelDescriptorError",
    "ModelOutput",
    "ModelRegistry",
    "ModelUnavailableError",
    "MultimodalReport",
    "MultimodalReportBuilder",
    "ReportSection",
    "load_model_backend",
    "new_request_id",
    "utc_now",
]

_LAZY = {"MediaAnalysisService"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from ml.inference import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
