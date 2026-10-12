"""Tests for the classical image detector backend and its JSON artifact.

The tests build a real artifact on disk (weights + model card), load it through the
genuine :func:`ml.inference.models.load_model_backend` path, and score a synthetic
preprocessed image. This proves the adapter and the loader agree; it says nothing
about detection quality.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.manifest import calculate_file_sha256
from ml.image.detectors.classical import (
    ClassicalImageBackend,
    ClassicalWeights,
    ClassicalWeightsError,
    build,
    load_classical_weights,
    write_classical_weights,
)
from ml.image.features import IMAGE_FEATURE_NAMES, ImageFeatureConfig
from ml.image.preprocess import ImagePreprocessConfig, ImagePreprocessor
from ml.image.training import FittedClassifier, train_classifier
from ml.inference.models import ModelRegistry, ModelUnavailableError, load_model_backend
from tests.synthetic_inference import (
    FIXED_TIMESTAMP,
    FakeImagePixelDecoder,
    make_image_array,
    write_model_card,
)

_DIM = len(IMAGE_FEATURE_NAMES)
_ENTRYPOINT = "ml.image.detectors.classical:build"


def _fit(seed: int = 0) -> FittedClassifier:
    rng = np.random.default_rng(seed)
    real = rng.normal(0.0, 1.0, (30, _DIM))
    fake = rng.normal(2.5, 1.0, (30, _DIM))
    matrix = np.vstack([real, fake])
    labels = ["real"] * 30 + ["fake"] * 30
    return train_classifier(
        matrix,
        labels,
        feature_names=IMAGE_FEATURE_NAMES,
        seed=seed,
        trained_at=FIXED_TIMESTAMP,
    )


def _weights(threshold: float = 0.0) -> ClassicalWeights:
    return ClassicalWeights.from_fitted(
        _fit(),
        feature_config=ImageFeatureConfig(),
        threshold=threshold,
        training={"dataset_id": "synthetic"},
        created_at=FIXED_TIMESTAMP,
    )


def _write_artifact(models_dir, *, model_id: str = "classical", threshold: float = 0.0):
    weights = _weights(threshold)
    weights_path = write_classical_weights(weights, models_dir / model_id / "weights.json")
    digest = calculate_file_sha256(weights_path)
    write_model_card(
        models_dir,
        model_id,
        framework="numpy",
        entrypoint=_ENTRYPOINT,
        weights_file="weights.json",
        weights_sha256=digest,
        labels=list(weights.labels),
        decision_threshold=weights.threshold,
        score_kind="uncalibrated_score",
        score_name="manipulation_score",
        score_semantics="Higher values indicate a greater likelihood of manipulation.",
        input_size=[224, 224],
        color_space="RGB",
    )
    return weights


def _preprocessed(size: int = 32):
    array = make_image_array(size, size, 3)
    return ImagePreprocessor(
        FakeImagePixelDecoder(array, "BGR"),
        ImagePreprocessConfig(target_size=(size, size)),
    ).preprocess("synthetic.png")


class TestWeights:
    def test_round_trip_through_disk(self, tmp_path):
        weights = _weights(0.25)
        path = write_classical_weights(weights, tmp_path / "weights.json")
        rebuilt = load_classical_weights(path)
        assert rebuilt == weights
        assert rebuilt.to_dict() == weights.to_dict()
        assert rebuilt.threshold == 0.25

    def test_score_is_a_float(self):
        weights = _weights()
        score = weights.score(np.asarray(weights.scaler_mean, dtype=np.float64))
        assert isinstance(score, float)

    def test_feature_name_mismatch_rejected(self):
        payload = _weights().to_dict()
        payload["feature_names"] = ["wrong"] * _DIM
        with pytest.raises(ClassicalWeightsError):
            ClassicalWeights.from_dict(payload)

    def test_unknown_schema_version_rejected(self):
        payload = _weights().to_dict()
        payload["weights_schema_version"] = 999
        with pytest.raises(ClassicalWeightsError):
            ClassicalWeights.from_dict(payload)

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ClassicalWeightsError):
            load_classical_weights(tmp_path / "absent.json")


class TestBuild:
    def test_build_returns_a_backend(self, tmp_path):
        _write_artifact(tmp_path / "models")
        descriptor = ModelRegistry.discover(tmp_path / "models").get("classical")
        backend = build(
            descriptor,
            model_dir=tmp_path / "models" / "classical",
            weights_path=tmp_path / "models" / "classical" / "weights.json",
        )
        assert isinstance(backend, ClassicalImageBackend)
        assert backend.descriptor.model_id == "classical"

    def test_no_weights_path_is_refused(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(models, "classical", framework="numpy", entrypoint=_ENTRYPOINT)
        descriptor = ModelRegistry.discover(models).get("classical")
        with pytest.raises(ClassicalWeightsError):
            build(descriptor, model_dir=models / "classical", weights_path=None)


class TestPredict:
    def test_predict_returns_honest_model_output(self, tmp_path):
        _write_artifact(tmp_path / "models")
        models = tmp_path / "models"
        descriptor = ModelRegistry.discover(models).get("classical")
        backend = load_model_backend(descriptor, models_dir=models)

        output = backend.predict(_preprocessed())
        assert output.model_id == "classical"
        assert output.raw_score is not None and np.isfinite(output.raw_score)
        assert output.calibrated is False
        assert output.confidence is None
        assert output.score_kind == "uncalibrated_score"
        assert output.labels == ("real", "fake")
        assert output.score_semantics
        assert output.limitations

    def test_score_matches_manual_weights_math(self, tmp_path):
        weights = _write_artifact(tmp_path / "models")
        models = tmp_path / "models"
        descriptor = ModelRegistry.discover(models).get("classical")
        backend = load_model_backend(descriptor, models_dir=models)

        sample = _preprocessed()
        output = backend.predict(sample)
        manual = weights.score(np.asarray(output.extra["feature_values"], dtype=np.float64))
        assert output.raw_score == pytest.approx(manual)

    def test_bad_sample_is_raised_not_scored(self, tmp_path):
        _write_artifact(tmp_path / "models")
        models = tmp_path / "models"
        descriptor = ModelRegistry.discover(models).get("classical")
        backend = load_model_backend(descriptor, models_dir=models)
        with pytest.raises(ClassicalWeightsError):
            backend.predict(np.zeros((8, 8), dtype=np.float64))


class TestLoaderGates:
    def test_missing_weights_file_is_model_unavailable(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(
            models,
            "classical",
            framework="numpy",
            entrypoint=_ENTRYPOINT,
            weights_file="weights.json",
        )
        descriptor = ModelRegistry.discover(models).get("classical")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "weights_missing"

    def test_digest_mismatch_is_model_unavailable(self, tmp_path):
        models = tmp_path / "models"
        _write_artifact(models)
        weights_file = models / "classical" / "weights.json"
        weights_file.write_text(weights_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        descriptor = ModelRegistry.discover(models).get("classical")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "weights_digest_mismatch"

    def test_declared_no_weights_becomes_load_failed(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(models, "classical", framework="numpy", entrypoint=_ENTRYPOINT)
        descriptor = ModelRegistry.discover(models).get("classical")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "load_failed"

    def test_label_mismatch_between_card_and_weights_is_load_failed(self, tmp_path):
        models = tmp_path / "models"
        _write_artifact(models)
        write_model_card(
            models,
            "classical",
            framework="numpy",
            entrypoint=_ENTRYPOINT,
            weights_file="weights.json",
            weights_sha256=calculate_file_sha256(models / "classical" / "weights.json"),
            labels=["authentic", "manipulated"],
            decision_threshold=0.0,
        )
        descriptor = ModelRegistry.discover(models).get("classical")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "load_failed"
