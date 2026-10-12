"""Tests for the model registry, descriptors, outputs, and the loader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from datasets.schema import Modality
from ml.inference.models import (
    ModelDescriptor,
    ModelDescriptorError,
    ModelOutput,
    ModelRegistry,
    ModelUnavailableError,
    load_model_backend,
)
from tests.synthetic_inference import (
    ENTRYPOINT_MODULE,
    write_model_card,
)


def minimal_card(**overrides):
    payload = {
        "model_id": "demo",
        "version": "1.2.0",
        "modality": "image",
        "framework": "numpy",
    }
    payload.update(overrides)
    return payload


class TestModelDescriptor:
    def test_from_dict_minimal(self):
        descriptor = ModelDescriptor.from_dict(minimal_card())
        assert descriptor.model_id == "demo"
        assert descriptor.version == "1.2.0"
        assert descriptor.modality is Modality.IMAGE
        assert descriptor.qualified_id == "demo:1.2.0"
        assert descriptor.score_kind == "uncalibrated_score"

    def test_from_dict_parses_input_size_and_labels(self):
        descriptor = ModelDescriptor.from_dict(
            minimal_card(input_size=[224, 224], labels=["real", "fake"], calibrated=True)
        )
        assert descriptor.input_size == (224, 224)
        assert descriptor.labels == ("real", "fake")
        assert descriptor.calibrated is True

    def test_from_dict_rejects_missing_required_field(self):
        with pytest.raises(ModelDescriptorError):
            ModelDescriptor.from_dict({"version": "1", "modality": "image", "framework": "numpy"})

    def test_from_dict_rejects_bad_modality(self):
        with pytest.raises(ModelDescriptorError):
            ModelDescriptor.from_dict(minimal_card(modality="hologram"))

    def test_from_dict_rejects_bad_input_size(self):
        with pytest.raises(ModelDescriptorError):
            ModelDescriptor.from_dict(minimal_card(input_size=[1, 2, 3]))

    def test_from_dict_rejects_bool_in_input_size(self):
        with pytest.raises(ModelDescriptorError):
            ModelDescriptor.from_dict(minimal_card(input_size=[1, True]))

    def test_from_dict_rejects_non_object_root(self):
        with pytest.raises(ModelDescriptorError):
            ModelDescriptor.from_dict(["not", "a", "mapping"])

    def test_to_dict_roundtrips_through_from_dict(self):
        descriptor = ModelDescriptor.from_dict(minimal_card(input_size=[32, 32]))
        rebuilt = ModelDescriptor.from_dict(descriptor.to_dict())
        assert rebuilt == descriptor

    def test_weights_path_is_none_without_weights_file(self, tmp_path):
        descriptor = ModelDescriptor.from_dict(minimal_card())
        assert descriptor.weights_path(tmp_path) is None

    def test_weights_path_resolves_under_model_dir(self, tmp_path):
        descriptor = ModelDescriptor.from_dict(minimal_card(weights_file="w.bin"))
        assert descriptor.weights_path(tmp_path) == tmp_path / "demo" / "w.bin"


class TestModelOutput:
    def test_confidence_defaults_to_none(self):
        output = ModelOutput(
            model_id="demo", version="1", score_name="s", raw_score=0.5, score_kind="score"
        )
        assert output.confidence is None

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            ModelOutput(
                model_id="demo",
                version="1",
                score_name="s",
                raw_score=0.5,
                score_kind="score",
                confidence=1.5,
            )

    def test_to_dict_preserves_score_kind(self):
        output = ModelOutput(
            model_id="demo",
            version="1",
            score_name="s",
            raw_score=0.5,
            score_kind="uncalibrated_score",
            limitations=("demo",),
        )
        payload = output.to_dict()
        assert payload["score_kind"] == "uncalibrated_score"
        assert payload["limitations"] == ["demo"]


class TestModelRegistry:
    def test_empty_when_directory_missing(self, tmp_path):
        registry = ModelRegistry.discover(tmp_path / "nope")
        assert registry.is_empty
        assert len(registry) == 0
        assert registry.descriptors == ()
        assert registry.problems == ()

    def test_empty_when_directory_has_no_cards(self, tmp_path):
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "stuff").mkdir()
        registry = ModelRegistry.discover(tmp_path / "models")
        assert registry.is_empty

    def test_discovers_and_sorts_by_model_id(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(models, "beta")
        write_model_card(models, "alpha")
        registry = ModelRegistry.discover(models)
        assert [d.model_id for d in registry.descriptors] == ["alpha", "beta"]

    def test_malformed_card_is_reported_not_fatal(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(models, "good")
        bad = models / "bad" / "model.json"
        bad.parent.mkdir(parents=True)
        bad.write_text("{ not json", encoding="utf-8")
        registry = ModelRegistry.discover(models)
        assert [d.model_id for d in registry.descriptors] == ["good"]
        assert len(registry.problems) == 1
        assert "bad" in registry.problems[0]

    def test_for_modality_filters(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(models, "img", modality="image")
        write_model_card(models, "aud", modality="audio")
        registry = ModelRegistry.discover(models)
        assert [d.model_id for d in registry.for_modality(Modality.IMAGE)] == ["img"]
        assert [d.model_id for d in registry.for_modality("audio")] == ["aud"]
        assert registry.for_modality(Modality.VIDEO) == ()

    def test_get_by_id(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(models, "img")
        registry = ModelRegistry.discover(models)
        assert registry.get("img") is not None
        assert registry.get("missing") is None


class TestLoadModelBackend:
    def _descriptor(self, models_dir: Path, **overrides) -> ModelDescriptor:
        write_model_card(models_dir, "demo", **overrides)
        descriptor = ModelRegistry.discover(models_dir).get("demo")
        assert descriptor is not None
        return descriptor

    def test_weights_missing(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(models, weights_file="w.bin")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "weights_missing"

    def test_weights_digest_mismatch(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(models, weights_file="w.bin", weights_sha256="0" * 64)
        (models / "demo" / "w.bin").write_bytes(b"weights")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "weights_digest_mismatch"

    def test_weights_digest_match_proceeds(self, tmp_path):
        models = tmp_path / "models"
        payload = b"weights"
        digest = hashlib.sha256(payload).hexdigest()
        descriptor = self._descriptor(
            models,
            weights_file="w.bin",
            weights_sha256=digest,
            entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend",
        )
        (models / "demo" / "w.bin").write_bytes(payload)
        backend = load_model_backend(descriptor, models_dir=models)
        assert backend.descriptor.model_id == "demo"

    def test_dependency_missing(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(models, framework="torch_absent_xyz")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "dependency_missing"

    def test_no_entrypoint(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(models)
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "no_entrypoint"

    def test_entrypoint_malformed(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(models, entrypoint="no_colon")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "entrypoint_malformed"

    def test_entrypoint_module_missing(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(models, entrypoint="absent_module_xyz:factory")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "entrypoint_missing"

    def test_entrypoint_attribute_missing(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(models, entrypoint=f"{ENTRYPOINT_MODULE}:absent_attr")
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "entrypoint_attribute_missing"

    def test_entrypoint_raises_is_load_failed(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(
            models, entrypoint=f"{ENTRYPOINT_MODULE}:build_raising_entrypoint"
        )
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "load_failed"

    def test_entrypoint_wrong_object_is_load_failed(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(
            models, entrypoint=f"{ENTRYPOINT_MODULE}:build_wrong_object"
        )
        with pytest.raises(ModelUnavailableError) as excinfo:
            load_model_backend(descriptor, models_dir=models)
        assert excinfo.value.reason == "load_failed"

    def test_successful_load_returns_working_backend(self, tmp_path):
        models = tmp_path / "models"
        descriptor = self._descriptor(
            models, entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend"
        )
        backend = load_model_backend(descriptor, models_dir=models)
        output = backend.predict(None)
        assert output.raw_score == 0.93
        assert output.calibrated is False
