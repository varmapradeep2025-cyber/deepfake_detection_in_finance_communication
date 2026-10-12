"""Feature extraction interfaces (typed, optional, separate from scoring).

This module defines a minimal, honest contract for representing extracted
features. It is interface-only: no trained models, no fabricated vectors.
Key points:
- Feature extraction is separate from prediction/scoring (has_score unchanged).
- Availability/failure is explicit (unavailable vs present).
- Cross-modal alignment is not assumed; compatibility must be explicit.
- Only supportable claims are allowed; learned embeddings must not be
  fabricated from observations/metadata.
"""
from __future__ import annotations

from ml.features.contract import (
    FeatureAvailability,
    FeatureExtractorBackend,
    FeatureExtractorUnavailableError,
    FeatureRepresentation,
    FeatureRepresentationKind,
    make_unavailable_features,
)

__all__ = [
    "FeatureAvailability",
    "FeatureExtractorBackend",
    "FeatureExtractorUnavailableError",
    "FeatureRepresentation",
    "FeatureRepresentationKind",
    "make_unavailable_features",
]


# re-export for convenience

