"""Tests for the video analyzer: real stream metadata, honest detection gap."""

from __future__ import annotations

from pathlib import Path

from datasets.preprocessing.inspection import MediaInspector
from ml.inference.models import ModelRegistry
from ml.inference.request import AnalysisRequest
from ml.inference.status import AnalysisStatus
from ml.video.analyzer import VideoAnalyzer
from tests.synthetic_inference import write_model_card
from tests.synthetic_preprocessing import FakeVideo, FakeVideoDecoder


def _clock() -> str:
    return "2024-01-01T00:00:00+00:00"


def _media_file(tmp_path: Path, content: bytes = b"a-real-file-on-disk") -> Path:
    target = tmp_path / "clip.mp4"
    target.write_bytes(content)
    return target


def _analyzer(tmp_path: Path, *, registered: bool = True, decoder=None, max_bytes=None):
    models = tmp_path / "models"
    if registered:
        write_model_card(models, "vid", modality="video")
    inspector = MediaInspector(
        decoder if decoder is not None else FakeVideoDecoder(), compute_hashes=True
    )
    kwargs = {}
    if max_bytes is not None:
        kwargs["max_bytes"] = max_bytes
    return VideoAnalyzer(
        inspector=inspector,
        registry=ModelRegistry.discover(models),
        clock=_clock,
        **kwargs,
    )


def test_missing_file_is_invalid_input(tmp_path):
    analyzer = _analyzer(tmp_path)
    report = analyzer.analyze(AnalysisRequest(path=str(tmp_path / "absent.mp4")))
    assert report.status is AnalysisStatus.INVALID_INPUT
    assert report.errors


def test_empty_file_is_invalid_input(tmp_path):
    target = _media_file(tmp_path, content=b"")
    report = _analyzer(tmp_path).analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.INVALID_INPUT


def test_oversize_file_is_invalid_input(tmp_path):
    target = _media_file(tmp_path, content=b"x" * 64)
    report = _analyzer(tmp_path, max_bytes=8).analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.INVALID_INPUT


def test_no_decoder_is_capability_unavailable(tmp_path):
    target = _media_file(tmp_path)
    inspector = MediaInspector(None, compute_hashes=True)
    analyzer = VideoAnalyzer(
        inspector=inspector,
        registry=ModelRegistry.discover(tmp_path / "models"),
        clock=_clock,
    )
    report = analyzer.analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.CAPABILITY_UNAVAILABLE
    assert report.errors


def test_undecodable_file_is_invalid_input(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeVideoDecoder()
    decoder.add(target, FakeVideo(undecodable=True))
    report = _analyzer(tmp_path, decoder=decoder).analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.INVALID_INPUT


def test_no_video_model_is_model_unavailable(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeVideoDecoder()
    decoder.add(target, FakeVideo(frame_count=30, width=64, height=48, frame_rate=25.0))
    analyzer = _analyzer(tmp_path, registered=False, decoder=decoder)
    report = analyzer.analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.MODEL_UNAVAILABLE
    assert report.output is None
    assert report.result is not None
    assert report.result["width"] == 64
    assert report.result["declared_frame_count"] == 30
    assert report.result["frame_rate"] == 25.0


def test_stream_metadata_becomes_evidence(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeVideoDecoder()
    decoder.add(target, FakeVideo(frame_count=12, width=32, height=24, frame_rate=24.0))
    report = _analyzer(tmp_path, registered=False, decoder=decoder).analyze(
        AnalysisRequest(path=str(target))
    )
    labels = {item.label for item in report.evidence}
    assert "File SHA-256" in labels
    assert "Dimensions" in labels
    assert "Frame rate" in labels
    assert all(item.claim_kind.value == "observed" for item in report.evidence)


def test_registered_video_model_is_capability_unavailable(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeVideoDecoder()
    decoder.add(target, FakeVideo())
    analyzer = _analyzer(tmp_path, registered=True, decoder=decoder)
    report = analyzer.analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.CAPABILITY_UNAVAILABLE
    assert report.output is None
    assert any("frame preprocessing" in warning for warning in report.warnings)


def test_analysis_is_deterministic(tmp_path):
    target = _media_file(tmp_path)
    decoder = FakeVideoDecoder()
    decoder.add(target, FakeVideo())
    analyzer = _analyzer(tmp_path, registered=False, decoder=decoder)
    first = analyzer.analyze(AnalysisRequest(path=str(target))).to_dict()
    second = analyzer.analyze(AnalysisRequest(path=str(target))).to_dict()
    first.pop("request_id")
    second.pop("request_id")
    assert first == second
