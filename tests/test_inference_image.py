"""Tests for image preprocessing, the image analyzer, and the service routing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from datasets.schema import Modality
from ml.image.preprocess import (
    ImagePreprocessConfig,
    ImagePreprocessError,
    ImagePreprocessor,
)
from ml.inference.evidence import ClaimKind
from ml.inference.request import AnalysisRequest
from ml.inference.service import MediaAnalysisService
from ml.inference.status import AnalysisStatus
from tests.synthetic_inference import (
    ENTRYPOINT_MODULE,
    FakeImagePixelDecoder,
    make_image_analyzer,
    make_image_array,
    write_model_card,
)
from tests.synthetic_preprocessing import requires_opencv, write_real_image


def _preprocessor(array, color_space="BGR", target_size=(4, 4)) -> ImagePreprocessor:
    return ImagePreprocessor(
        FakeImagePixelDecoder(array, color_space),
        ImagePreprocessConfig(target_size=target_size),
    )


def _media_file(tmp_path, content: bytes = b"not-a-real-png-but-a-real-file"):
    target = tmp_path / "media.png"
    target.write_bytes(content)
    return target


class TestPreprocessing:
    def test_bgr_is_reordered_to_rgb(self):
        array = np.array([[[10, 20, 30]]], dtype=np.uint8)
        result = _preprocessor(array, color_space="BGR", target_size=(1, 1)).preprocess("x.png")
        assert result.color_space == "RGB"
        assert result.array[0, 0, 0] == pytest.approx(30 / 255)
        assert result.array[0, 0, 2] == pytest.approx(10 / 255)

    def test_rgb_order_is_kept(self):
        array = np.array([[[10, 20, 30]]], dtype=np.uint8)
        result = _preprocessor(array, color_space="RGB", target_size=(1, 1)).preprocess("x.png")
        assert result.array[0, 0, 0] == pytest.approx(10 / 255)

    def test_grayscale_is_replicated_to_three_channels(self):
        array = np.array([[100, 150]], dtype=np.uint8)
        result = _preprocessor(array, color_space="L", target_size=(1, 2)).preprocess("x.png")
        assert result.array.shape == (1, 2, 3)
        assert result.array[0, 0, 0] == pytest.approx(100 / 255)
        assert result.array[0, 1, 0] == pytest.approx(150 / 255)

    def test_bgra_alpha_is_dropped(self):
        array = np.zeros((1, 1, 4), dtype=np.uint8)
        array[0, 0] = [10, 20, 30, 255]
        result = _preprocessor(array, color_space="BGRA", target_size=(1, 1)).preprocess("x.png")
        assert result.channels == 3
        assert result.array.shape == (1, 1, 3)
        assert result.array[0, 0, 0] == pytest.approx(30 / 255)

    def test_resizes_to_target_size(self):
        result = _preprocessor(make_image_array(8, 6, 3), target_size=(4, 4)).preprocess("x.png")
        assert (result.height, result.width) == (4, 4)
        assert result.array.shape == (4, 4, 3)

    def test_output_is_float32_in_unit_range(self):
        result = _preprocessor(make_image_array(8, 6, 3)).preprocess("x.png")
        assert result.array.dtype == np.float32
        low, high = result.value_range()
        assert 0.0 <= low <= high <= 1.0

    def test_preprocessing_is_deterministic(self):
        array = make_image_array(8, 6, 3)
        first = _preprocessor(array).preprocess("x.png")
        second = _preprocessor(array).preprocess("x.png")
        assert first.digest == second.digest

    def test_unknown_channel_ordering_raises(self):
        with pytest.raises(ImagePreprocessError):
            _preprocessor(make_image_array(2, 2, 3), color_space="XN").preprocess("x.png")

    def test_zero_sized_image_raises(self):
        with pytest.raises(ImagePreprocessError):
            _preprocessor(np.zeros((0, 0, 3), dtype=np.uint8)).preprocess("x.png")

    def test_summary_is_json_serializable_and_omits_pixels(self):
        result = _preprocessor(make_image_array(8, 6, 3)).preprocess("x.png")
        summary = result.to_summary()
        assert "array" not in summary
        assert summary["dtype"] == "float32"
        json.dumps(summary)


class TestImageAnalyzerInput:
    def test_missing_file_is_invalid_input(self, tmp_path):
        analyzer = make_image_analyzer(models_dir=tmp_path / "models", array=make_image_array())
        report = analyzer.analyze(AnalysisRequest(path=str(tmp_path / "absent.png")))
        assert report.status is AnalysisStatus.INVALID_INPUT
        assert report.errors
        assert report.output is None

    def test_empty_file_is_invalid_input(self, tmp_path):
        target = _media_file(tmp_path, content=b"")
        analyzer = make_image_analyzer(models_dir=tmp_path / "models", array=make_image_array())
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.INVALID_INPUT

    def test_oversize_file_is_invalid_input(self, tmp_path):
        target = _media_file(tmp_path, content=b"x" * 50)
        analyzer = make_image_analyzer(
            models_dir=tmp_path / "models", array=make_image_array(), max_bytes=10
        )
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.INVALID_INPUT
        assert any("limit" in error for error in report.errors)

    def test_undecodable_pixels_are_invalid_input(self, tmp_path):
        target = _media_file(tmp_path)

        class BoomDecoder(FakeImagePixelDecoder):
            def decode(self, path):
                raise RuntimeError("no pixels here")

        base = make_image_analyzer(models_dir=tmp_path / "models", array=make_image_array())
        base._preprocessor._decoder = BoomDecoder(make_image_array())
        report = base.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.INVALID_INPUT
        assert any("decode pixels" in error for error in report.errors)


class TestImageAnalyzerModelSelection:
    def test_no_registered_model_is_model_unavailable(self, tmp_path):
        target = _media_file(tmp_path)
        analyzer = make_image_analyzer(models_dir=tmp_path / "models", array=make_image_array())
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.MODEL_UNAVAILABLE
        assert report.model is None
        # observed + derived evidence survives even with no model
        kinds = {item.claim_kind for item in report.evidence}
        assert ClaimKind.OBSERVED in kinds
        assert ClaimKind.DERIVED in kinds
        assert report.preprocessing is not None

    def test_unknown_requested_model_is_model_unavailable(self, tmp_path):
        target = _media_file(tmp_path)
        models = tmp_path / "models"
        write_model_card(models, "demo", entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend")
        analyzer = make_image_analyzer(models_dir=models, array=make_image_array())
        report = analyzer.analyze(AnalysisRequest(path=str(target), model_id="ghost"))
        assert report.status is AnalysisStatus.MODEL_UNAVAILABLE
        assert any("ghost" in error for error in report.errors)

    def test_dependency_missing_is_model_unavailable_with_model_recorded(self, tmp_path):
        target = _media_file(tmp_path)
        models = tmp_path / "models"
        write_model_card(
            models,
            "demo",
            framework="torch_absent_xyz",
            entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend",
        )
        analyzer = make_image_analyzer(models_dir=models, array=make_image_array())
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.MODEL_UNAVAILABLE
        assert report.model is not None
        assert any("dependency_missing" in warning for warning in report.warnings)

    def test_backend_exception_is_analysis_failed(self, tmp_path):
        target = _media_file(tmp_path)
        models = tmp_path / "models"
        write_model_card(models, "demo", entrypoint=f"{ENTRYPOINT_MODULE}:build_crashing_backend")
        analyzer = make_image_analyzer(models_dir=models, array=make_image_array())
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.ANALYSIS_FAILED
        assert report.model is not None
        assert report.output is None


class TestImageAnalyzerComplete:
    def _analyzer(self, tmp_path):
        models = tmp_path / "models"
        write_model_card(models, "demo", entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend")
        return make_image_analyzer(models_dir=models, array=make_image_array(8, 6, 3))

    def test_complete_report_records_score_risk_and_evidence(self, tmp_path):
        target = _media_file(tmp_path)
        report = self._analyzer(tmp_path).analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
        assert report.is_success is True
        assert report.output is not None
        assert report.output.raw_score == 0.93
        assert report.output.calibrated is False
        assert report.risk is not None
        assert report.risk.level.value == "high"
        assert report.risk.calibrated is False
        kinds = {item.claim_kind for item in report.evidence}
        assert {ClaimKind.OBSERVED, ClaimKind.DERIVED, ClaimKind.INFERRED} <= kinds

    def test_uncalibrated_score_warns_and_has_no_confidence(self, tmp_path):
        target = _media_file(tmp_path)
        report = self._analyzer(tmp_path).analyze(AnalysisRequest(path=str(target)))
        assert any("uncalibrated" in warning for warning in report.warnings)
        signal = next(
            item for item in report.evidence if item.claim_kind is ClaimKind.INFERRED
        )
        assert signal.confidence is None

    def test_report_is_json_serializable(self, tmp_path):
        target = _media_file(tmp_path)
        report = self._analyzer(tmp_path).analyze(AnalysisRequest(path=str(target)))
        payload = report.to_dict()
        assert payload["status"] == "analysis_complete"
        assert payload["model"]["model_id"] == "demo"
        json.dumps(payload)

    def test_calibrated_backend_reports_probability_without_warning(self, tmp_path):
        target = _media_file(tmp_path)
        models = tmp_path / "models"
        write_model_card(models, "cal", entrypoint=f"{ENTRYPOINT_MODULE}:build_calibrated_backend")
        analyzer = make_image_analyzer(models_dir=models, array=make_image_array(8, 6, 3))
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
        assert report.output.calibrated is True
        assert report.risk.calibrated is True
        assert not any("uncalibrated" in warning for warning in report.warnings)

    def test_alpha_channel_is_reported_as_a_warning(self, tmp_path):
        target = _media_file(tmp_path)
        models = tmp_path / "models"
        write_model_card(models, "demo", entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend")
        analyzer = make_image_analyzer(
            models_dir=models, array=make_image_array(8, 6, 4), color_space="BGRA"
        )
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
        assert any("alpha" in warning for warning in report.warnings)


class TestServiceRouting:
    def test_audio_is_unsupported_modality(self, tmp_path):
        service = MediaAnalysisService()
        request = AnalysisRequest(path=str(tmp_path / "x.wav"), modality=Modality.AUDIO)
        report = service.analyze(request)
        assert report.status is AnalysisStatus.UNSUPPORTED_MODALITY
        assert report.errors

    def test_image_without_analyzer_is_unsupported(self, tmp_path):
        service = MediaAnalysisService()
        report = service.analyze(AnalysisRequest(path=str(tmp_path / "x.png")))
        assert report.status is AnalysisStatus.UNSUPPORTED_MODALITY

    def test_image_routes_to_analyzer(self, tmp_path):
        target = _media_file(tmp_path)
        models = tmp_path / "models"
        write_model_card(models, "demo", entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend")
        analyzer = make_image_analyzer(models_dir=models, array=make_image_array(8, 6, 3))
        service = MediaAnalysisService(image_analyzer=analyzer)
        report = service.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.ANALYSIS_COMPLETE

    def test_unknown_modality_string_raises(self, tmp_path):
        service = MediaAnalysisService()
        with pytest.raises(ValueError):
            service.analyze(
                AnalysisRequest(path=str(tmp_path / "x.bin"), modality="hologram")
            )


@requires_opencv
class TestRealImageIntegration:
    def test_end_to_end_with_real_pixels(self, tmp_path):
        from datasets.preprocessing.inspection import MediaInspector
        from ml.image.analyzer import ImageAnalyzer
        from ml.image.preprocess import (
            ImagePreprocessConfig,
            ImagePreprocessor,
            default_image_pixel_decoder,
        )
        from ml.inference.models import ModelRegistry

        target = write_real_image(tmp_path / "real.png", width=64, height=48, channels=3)
        models = tmp_path / "models"
        write_model_card(models, "demo", entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend")

        analyzer = ImageAnalyzer(
            inspector=MediaInspector(
                None,
                image_decoder=_real_image_decoder(),
                compute_hashes=True,
            ),
            preprocessor=ImagePreprocessor(
                default_image_pixel_decoder(), ImagePreprocessConfig(target_size=(8, 8))
            ),
            registry=ModelRegistry.discover(models),
        )
        report = analyzer.analyze(AnalysisRequest(path=str(target)))
        assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
        assert report.preprocessing is not None
        assert report.preprocessing["source_width"] == 64
        assert report.preprocessing["source_height"] == 48
        assert report.preprocessing["width"] == 8


def _real_image_decoder():
    from datasets.preprocessing.inspection import default_image_decoder

    decoder = default_image_decoder()
    if decoder is None:  # pragma: no cover - guarded by requires_opencv
        pytest.skip("opencv image decoder unavailable")
    return decoder
