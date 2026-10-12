"""Tests for the QR analyzer and decoder: decode, then grade any URL honestly."""

from __future__ import annotations

from pathlib import Path

from datasets.preprocessing.inspection import MediaInspector, default_image_decoder
from ml.inference.request import AnalysisRequest
from ml.inference.status import AnalysisStatus
from ml.qr.analyzer import QrAnalyzer
from ml.qr.decode import OpenCVQrDecoder
from tests.synthetic_preprocessing import (
    FakeImage,
    FakeImageDecoder,
    requires_opencv,
)


def _clock() -> str:
    return "2024-01-01T00:00:00+00:00"


def _media_file(tmp_path: Path, name: str = "code.png", content: bytes = b"img-bytes") -> Path:
    target = tmp_path / name
    target.write_bytes(content)
    return target


class FakeQrDecoder:
    """A QR decoder that returns fixed payloads, or raises when told to."""

    name = "fake-qr"

    def __init__(self, payloads: tuple[str, ...] = (), *, raises: bool = False) -> None:
        self.payloads = payloads
        self.raises = raises
        self.calls: list[Path] = []

    def decode(self, path: Path) -> tuple[str, ...]:
        self.calls.append(path)
        if self.raises:
            raise RuntimeError("synthetic decoder failure")
        return self.payloads


def _analyzer(tmp_path: Path, decoder, *, with_image: bool = True):
    image_decoder = FakeImageDecoder()
    if with_image:
        image_decoder.add(_media_file(tmp_path), FakeImage(width=21, height=21, channels=1))
    inspector = MediaInspector(None, image_decoder=image_decoder, compute_hashes=True)
    return QrAnalyzer(inspector=inspector, decoder=decoder, clock=_clock)


def test_missing_file_is_invalid_input(tmp_path):
    report = _analyzer(tmp_path, FakeQrDecoder()).analyze(
        AnalysisRequest(path=str(tmp_path / "absent.png"))
    )
    assert report.status is AnalysisStatus.INVALID_INPUT
    assert report.errors


def test_no_decoder_is_capability_unavailable(tmp_path):
    report = _analyzer(tmp_path, None).analyze(
        AnalysisRequest(path=str(_media_file(tmp_path)))
    )
    assert report.status is AnalysisStatus.CAPABILITY_UNAVAILABLE
    assert report.errors


def test_no_codes_is_complete_with_warning(tmp_path):
    report = _analyzer(tmp_path, FakeQrDecoder()).analyze(
        AnalysisRequest(path=str(_media_file(tmp_path)))
    )
    assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
    assert report.result is not None
    assert report.result["code_count"] == 0
    assert any("no QR code" in warning for warning in report.warnings)
    assert report.has_score is False


def test_text_payload_is_recorded_and_not_graded(tmp_path):
    decoder = FakeQrDecoder(("WIFI:S:home;T:WPA;P:secret;;",))
    report = _analyzer(tmp_path, decoder).analyze(
        AnalysisRequest(path=str(_media_file(tmp_path)))
    )
    assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
    assert report.result is not None
    assert report.result["codes"][0]["kind"] == "text"
    assert report.result["codes"][0]["url"] is None
    assert any("not an http" in warning for warning in report.warnings)


def test_url_payload_is_graded(tmp_path):
    decoder = FakeQrDecoder(("http://203.0.113.7/login",))
    report = _analyzer(tmp_path, decoder).analyze(
        AnalysisRequest(path=str(_media_file(tmp_path)))
    )
    assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
    codes = report.result["codes"]
    assert codes[0]["kind"] == "url"
    assert codes[0]["url"]["is_ip_literal"] is True
    codes_found = {finding["code"] for finding in codes[0]["url"]["findings"]}
    assert "ip_literal_host" in codes_found
    assert "insecure_transport" in codes_found
    assert report.result["url_count"] == 1


def test_decoder_failure_is_analysis_failed(tmp_path):
    report = _analyzer(tmp_path, FakeQrDecoder(raises=True)).analyze(
        AnalysisRequest(path=str(_media_file(tmp_path)))
    )
    assert report.status is AnalysisStatus.ANALYSIS_FAILED
    assert report.errors


def test_evidence_includes_hash(tmp_path):
    decoder = FakeQrDecoder(("https://example.com/",))
    report = _analyzer(tmp_path, decoder).analyze(
        AnalysisRequest(path=str(_media_file(tmp_path)))
    )
    labels = {item.label for item in report.evidence}
    assert "File SHA-256" in labels


def _write_qr_image(path: Path, text: str) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    factory = getattr(cv2, "QRCodeEncoder_create", None) or cv2.QRCodeEncoder.create
    qr = np.asarray(factory().encode(text)).astype(np.uint8)
    scale = 8
    qr = cv2.resize(
        qr, (qr.shape[1] * scale, qr.shape[0] * scale), interpolation=cv2.INTER_NEAREST
    )
    qr = cv2.copyMakeBorder(qr, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
    if qr.ndim == 2:
        qr = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), qr)
    return path


@requires_opencv
def test_real_qr_decode_roundtrip(tmp_path):
    target = _write_qr_image(tmp_path / "real.png", "https://example.com/login")
    payloads = OpenCVQrDecoder().decode(target)
    assert any("example.com" in payload for payload in payloads)


@requires_opencv
def test_real_qr_end_to_end(tmp_path):
    target = _write_qr_image(tmp_path / "real.png", "http://198.51.100.9/verify")
    inspector = MediaInspector(None, image_decoder=default_image_decoder(), compute_hashes=True)
    analyzer = QrAnalyzer(inspector=inspector, decoder=OpenCVQrDecoder(), clock=_clock)
    report = analyzer.analyze(AnalysisRequest(path=str(target)))
    assert report.status is AnalysisStatus.ANALYSIS_COMPLETE
    assert report.result is not None
    assert report.result["url_count"] == 1
    codes_found = {
        finding["code"] for finding in report.result["codes"][0]["url"]["findings"]
    }
    assert "ip_literal_host" in codes_found
