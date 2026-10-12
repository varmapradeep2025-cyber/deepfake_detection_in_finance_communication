"""Tests for the feature extraction contract (Milestone 2)."""
from __future__ import annotations

from ml.features import (
    FeatureAvailability,
    FeatureRepresentation,
    FeatureRepresentationKind,
    make_unavailable_features,
)
from ml.inference.modality import AnalysisModality
from ml.inference.request import AnalysisRequest
from ml.inference.service import MediaAnalysisService


def test_valid_feature_representation_handcrafted():
    f = FeatureRepresentation(
        availability=FeatureAvailability.AVAILABLE,
        representation_kind=FeatureRepresentationKind.HANDCRAFTED,
        modality=AnalysisModality.TEXT,
        feature_dim=4,
        feature_vector=(0.1, 0.2, 0.3, 0.4),
        cross_modal_compatible=False,
    )
    assert f.availability is FeatureAvailability.AVAILABLE
    assert f.feature_dim == 4
    assert f.feature_vector is not None
    d = f.to_dict()
    assert d["availability"] == "available"
    assert d["feature_vector"] == [0.1, 0.2, 0.3, 0.4]


def test_valid_feature_representation_learned():
    f = FeatureRepresentation(
        availability=FeatureAvailability.AVAILABLE,
        representation_kind=FeatureRepresentationKind.LEARNED_EMBEDDING,
        modality=AnalysisModality.IMAGE,
        extractor_name="test-encoder",
        extractor_version="1.0.0",
        model_id="enc-1",
        model_version="1.0.0",
        feature_dim=128,
        feature_shape=(1, 128),
        feature_vector=tuple(0.0 for _ in range(128)),
        modality_space_id="image-enc-v1",
        cross_modal_compatible=True,
        alignment_metadata={"norm": "l2"},
    )
    assert f.cross_modal_compatible is True
    assert f.modality_space_id == "image-enc-v1"


def test_unavailable_extraction_explicit():
    f = make_unavailable_features(
        modality=AnalysisModality.TEXT,
        reason="no extractor registered",
        availability=FeatureAvailability.MISSING_EXTRACTOR,
        extractor_name=None,
    )
    assert f.availability is FeatureAvailability.MISSING_EXTRACTOR
    assert f.representation_kind is FeatureRepresentationKind.UNAVAILABLE
    assert f.feature_vector is None
    assert f.cross_modal_compatible is False
    assert f.failure_reason == "no extractor registered"


def test_invalid_unavailable_with_vector_raises():
    try:
        FeatureRepresentation(
            availability=FeatureAvailability.UNAVAILABLE,
            representation_kind=FeatureRepresentationKind.UNAVAILABLE,
            modality=AnalysisModality.TEXT,
            feature_vector=(1.0, 2.0),
        )
    except ValueError:
        assert True
        return
    assert False


def test_invalid_unavailable_claimed_learned_raises():
    try:
        FeatureRepresentation(
            availability=FeatureAvailability.UNAVAILABLE,
            representation_kind=FeatureRepresentationKind.LEARNED_EMBEDDING,
            modality=AnalysisModality.TEXT,
        )
    except ValueError:
        assert True
        return
    assert False


def test_invalid_available_but_kind_unavailable_raises():
    try:
        FeatureRepresentation(
            availability=FeatureAvailability.AVAILABLE,
            representation_kind=FeatureRepresentationKind.UNAVAILABLE,
            modality=AnalysisModality.TEXT,
        )
    except ValueError:
        assert True
        return
    assert False


def test_cross_modal_incompatibility_not_assumed():
    f1 = FeatureRepresentation(
        availability=FeatureAvailability.AVAILABLE,
        representation_kind=FeatureRepresentationKind.LEARNED_EMBEDDING,
        modality=AnalysisModality.IMAGE,
        feature_dim=64,
        modality_space_id="image-space-a",
        cross_modal_compatible=False,
    )
    f2 = FeatureRepresentation(
        availability=FeatureAvailability.AVAILABLE,
        representation_kind=FeatureRepresentationKind.LEARNED_EMBEDDING,
        modality=AnalysisModality.TEXT,
        feature_dim=64,
        modality_space_id="text-space-b",
        cross_modal_compatible=False,
    )
    assert f1.feature_dim == f2.feature_dim
    assert f1.cross_modal_compatible is False
    assert f2.cross_modal_compatible is False
    assert f1.modality_space_id != f2.modality_space_id


def test_missing_extractor_model():
    f = make_unavailable_features(
        modality=AnalysisModality.AUDIO,
        reason="missing extractor",
        availability=FeatureAvailability.MISSING_EXTRACTOR,
        model_id=None,
    )
    assert f.model_id is None
    assert f.availability is FeatureAvailability.MISSING_EXTRACTOR


def test_backwards_compatibility_existing_prediction_paths():
    svc = MediaAnalysisService.from_project_config()
    r_img = svc.analyze(AnalysisRequest(path="x.png", modality="image"))
    r_vid = svc.analyze(AnalysisRequest(path="x.mp4", modality="video"))
    r_aud = svc.analyze(AnalysisRequest(path="x.wav", modality="audio"))
    r_qr = svc.analyze(AnalysisRequest(path="x.png", modality="qr"))
    r_txt = svc.analyze(AnalysisRequest(path="hello", modality="text"))
    assert r_img.status.value in ("invalid_input", "model_unavailable")
    assert r_vid.status.value in ("invalid_input", "capability_unavailable", "model_unavailable")
    assert r_aud.status.value in ("invalid_input", "capability_unavailable", "model_unavailable")
    assert r_qr.status.value in ("invalid_input", "analysis_complete", "capability_unavailable")
    assert r_txt.status.value == "model_unavailable"
    assert r_txt.has_score is False
    assert r_txt.risk is None

