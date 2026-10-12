"""Tests for the audio analyzer: real recording properties, honest detection gap."""

from __future__ import annotations

from pathlib import Path

from datasets.preprocessing.inspection import MediaInspector
from ml.audio.analyzer import AudioAnalyzer
from ml.inference.models import ModelRegistry
from ml.inference.request import AnalysisRequest
from ml.inference.status import AnalysisStatus
from tests.synthetic_inference import write_model_card
from tests.synthetic_preprocessing import FakeAudio, FakeAudioDecoder


def _clock() -> str:
    return "2024-01-01T00:00:00+00:00"


def _media_file(tmp_path: Path, content: bytes = b"a-real-file-on-disk") -> Path:
    target = tmp_path / "voice.wav"
    target.write_bytes(content)
    return target


def _analyzer(tmp_path: Path, *, registered: bool = True, decoder=None, max_bytes=None):
    models = tmp_path / "models"
    if registered:
        write_model_card(models, "aud", modality="audio")
    if decoder is None:
        decoder = FakeAudioDecoder()
    inspector = MediaInspector(None, audio_decoder=decoder, compute_hashes=True)
    kwargs = {}
    if max_bytes is not None:
        kwargs["max_bytes"] = max_bytes
    return AudioAnalyzer(
        inspector=inspector,
        registry=ModelRegistry.discover(models),
        clock=_clock,
        **kwargs,
    )


def test_missing_file_is_invalid_input(tmp_path):
    report = _analyzer(tmp_path).analyze(
        AnalysisRequest(path=str(tmp_path / "absent.wav"))
    )
    assert report.status is AnalysisStatus.INVALID_INPUT
    assert report.errors


def test_empty_file_is_invalid_input(tmp_path):
    target = _media_file(tmp_path, content=b"")
    report = _analyzer(tmp_path).analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.INVALID_INPUT


def test_no_decoder_is_capability_unavailable(tmp_path):
    target = _media_file(tmp_path)
    inspector = MediaInspector(None, audio_decoder=None, compute_hashes=True)
    analyzer = AudioAnalyzer(
        inspector=inspector,
        registry=ModelRegistry.discover(tmp_path / "models"),
        clock=_clock,
    )
    report = analyzer.analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.CAPABILITY_UNAVAILABLE


def test_undecodable_audio_is_capability_unavailable(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeAudioDecoder()
    decoder.add(target, FakeAudio(undecodable=True))
    report = _analyzer(tmp_path, decoder=decoder).analyze(
        AnalysisRequest(path=str(target))
    )
    assert report.status is AnalysisStatus.CAPABILITY_UNAVAILABLE
    assert any("WAV" in warning for warning in report.warnings)


def test_no_audio_model_is_model_unavailable(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeAudioDecoder()
    decoder.add(target, FakeAudio(sample_rate=16000, channels=1, sample_count=32000))
    report = _analyzer(tmp_path, registered=False, decoder=decoder).analyze(
        AnalysisRequest(path=str(target))
    )
    assert report.status is AnalysisStatus.MODEL_UNAVAILABLE
    assert report.result is not None
    assert report.result["sample_rate_hz"] == 16000
    assert report.result["sample_count"] == 32000


def test_properties_become_observed_evidence(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeAudioDecoder()
    decoder.add(target, FakeAudio())
    report = _analyzer(tmp_path, registered=False, decoder=decoder).analyze(
        AnalysisRequest(path=str(target))
    )
    labels = {item.label for item in report.evidence}
    assert "File SHA-256" in labels
    assert "Sample rate" in labels
    assert "Channels" in labels


def test_registered_audio_model_is_capability_unavailable(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeAudioDecoder()
    decoder.add(target, FakeAudio())
    report = _analyzer(tmp_path, registered=True, decoder=decoder).analyze(
        AnalysisRequest(path=str(target))
    )
    assert report.status is AnalysisStatus.CAPABILITY_UNAVAILABLE
    assert report.output is None
