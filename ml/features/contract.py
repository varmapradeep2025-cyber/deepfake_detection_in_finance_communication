"""Typed feature-extraction contract.

Represents feature extraction separately from scoring/prediction.
All fields are explicit about availability and representation kind.
No dummy/fabricated vectors are generated.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ml.inference.modality import AnalysisModality

__all__ = [
    "FeatureAvailability",
    "FeatureExtractorBackend",
    "FeatureExtractorUnavailableError",
    "FeatureRepresentation",
    "FeatureRepresentationKind",
    "make_unavailable_features",
]


class FeatureRepresentationKind(str, Enum):
    """What kind of representation this is.

    Distinguishes learned embeddings from hand-crafted features and raw
    observations. This prevents mislabelling probe/metadata as learned.
    """

    #: Learned embedding (e.g. neural encoder) - only present if genuinely produced.
    LEARNED_EMBEDDING = "learned_embedding"
    #: Hand-crafted features derived from input via explicit rules.
    HANDCRAFTED = "handcrafted"
    #: Raw observation/measurement (e.g. counts, lengths) - not a learned embedding.
    RAW_OBSERVATION = "raw_observation"
    #: Unavailable / not produced.
    UNAVAILABLE = "unavailable"


class FeatureAvailability(str, Enum):
    """Explicit availability of extracted features."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE_INPUT = "incompatible_input"
    MISSING_EXTRACTOR = "missing_extractor"
    MISSING_WEIGHTS = "missing_weights"
    EXTRACTION_FAILED = "extraction_failed"
    NOT_SUPPORTED = "not_supported"


class FeatureExtractorUnavailableError(RuntimeError):
    """Raised when feature extraction cannot be performed honestly."""

    def __init__(self, message: str, *, reason: str, model_id: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.model_id = model_id


@dataclass(frozen=True, slots=True)
class FeatureRepresentation:
    """A typed representation of extracted features.

    Never contains fabricated data. If unavailable, availability != AVAILABLE
    and vector/array data must be empty/None. Cross-modal compatibility is
    explicitly stated; dimensions alone do not imply alignment.
    """

    availability: FeatureAvailability
    representation_kind: FeatureRepresentationKind
    modality: AnalysisModality
    model_id: str | None = None
    model_version: str | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    # Shapes/dims as explicit metadata (not implying cross-modal compat)
    feature_dim: int | None = None
    feature_shape: tuple[int, ...] | None = None
    # Feature values: None if unavailable. Use immutable-friendly structures.
    feature_vector: tuple[float, ...] | None = None
    feature_array_shape: tuple[int, ...] | None = None
    # Explicit compatibility/alignment metadata
    modality_space_id: str | None = None
    cross_modal_compatible: bool = False
    alignment_metadata: Mapping[str, Any] = field(default_factory=dict)
    # Provenance/semantics
    description: str = ""
    limitations: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.availability is FeatureAvailability.AVAILABLE:
            if self.representation_kind is FeatureRepresentationKind.UNAVAILABLE:
                raise ValueError("available features must not have UNAVAILABLE kind")
            # dims/shape optional; interface-only
        else:
            if self.representation_kind is FeatureRepresentationKind.LEARNED_EMBEDDING:
                raise ValueError("unavailable features cannot be claimed as learned embeddings")
            if self.feature_vector is not None and len(self.feature_vector) > 0:
                raise ValueError("unavailable features must not contain fabricated vectors")
            if self.cross_modal_compatible:
                object.__setattr__(self, "cross_modal_compatible", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability.value,
            "representation_kind": self.representation_kind.value,
            "modality": self.modality.value,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "feature_dim": self.feature_dim,
            "feature_shape": self.feature_shape,
            "feature_vector": (
                list(self.feature_vector) if self.feature_vector is not None else None
            ),
            "feature_array_shape": self.feature_array_shape,
            "modality_space_id": self.modality_space_id,
            "cross_modal_compatible": self.cross_modal_compatible,
            "alignment_metadata": dict(self.alignment_metadata),
            "description": self.description,
            "limitations": list(self.limitations),
            "extra": dict(self.extra),
            "failure_reason": self.failure_reason,
        }


def make_unavailable_features(
    *,
    modality: AnalysisModality,
    reason: str,
    availability: FeatureAvailability = FeatureAvailability.UNAVAILABLE,
    representation_kind: FeatureRepresentationKind = FeatureRepresentationKind.UNAVAILABLE,
    extractor_name: str | None = None,
    model_id: str | None = None,
    limitations: tuple[str, ...] = (),
    extra: Mapping[str, Any] | None = None,
) -> FeatureRepresentation:
    """Create an explicit unavailable feature representation (no fabrication)."""
    return FeatureRepresentation(
        availability=availability,
        representation_kind=representation_kind,
        modality=modality,
        model_id=model_id,
        extractor_name=extractor_name,
        feature_dim=None,
        feature_shape=None,
        feature_vector=None,
        feature_array_shape=None,
        modality_space_id=None,
        cross_modal_compatible=False,
        alignment_metadata={},
        description="Features not extracted (unavailable).",
        limitations=limitations,
        extra=dict(extra) if extra is not None else {},
        failure_reason=reason,
    )


@runtime_checkable
class FeatureExtractorBackend(Protocol):
    """Typed backend for feature extraction (separate from scoring)."""

    extractor_name: str
    extractor_version: str | None

    def extract_features(self, sample: Any) -> FeatureRepresentation:
        """Extract features from preprocessed sample.

        Must not fabricate vectors on failure; return make_unavailable_features
        or raise FeatureExtractorUnavailableError. Never return fabricated data.
        """
        ...

