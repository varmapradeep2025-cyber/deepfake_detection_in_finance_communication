from __future__ import annotations

from pathlib import Path
from ml.inference.request import AnalysisRequest
from ml.qr.analyzer import QrAnalyzer
from ml.url.analyze import Severity, analyze_url
from tests.synthetic_inference import FakeImageProbeDecoder

try:
    from datasets.preprocessing.inspection import MediaInspector
except Exception:
    MediaInspector = None  # type: ignore


def test_url_long_hostname_and_deep_subdomains() -> None:
    analysis = analyze_url('https://a.b.c.d.example.com/path')
    codes = {f.code for f in analysis.findings}
    assert 'excessive_subdomains' in codes
    long_host = 'x' * 50 + '.com'
    analysis2 = analyze_url(f'https://{long_host}/')
    codes2 = {f.code for f in analysis2.findings}
    assert 'unusually_long' in codes2


def test_qr_deduplicates_and_truncates_payloads(tmp_path: Path) -> None:
    if MediaInspector is None:
        return

    class DummyDecoder:
        name = 'dummy'
        def decode(self, path: Path) -> tuple[str, ...]:
            longp = 'https://example.com/' + 'x' * 5000
            return ('https://example.com/', 'https://example.com/', longp)

    inspector = MediaInspector(None, image_decoder=FakeImageProbeDecoder(), compute_hashes=True)
    target = tmp_path / 'img.png'
    target.write_bytes(b'fake')
    analyzer = QrAnalyzer(inspector=inspector, decoder=DummyDecoder(), max_bytes=10**7)
    report = analyzer.analyze(AnalysisRequest(path=str(target)))
    d = report.to_dict()
    evs = [e for e in d['evidence'] if e['label'].startswith('QR payload')]
    assert len(evs) == 2
    assert d['result']['code_count'] == 2


def test_qr_caps_excessive_codes(tmp_path: Path) -> None:
    if MediaInspector is None:
        return

    class DummyDecoder:
        name = 'dummy'
        def decode(self, path: Path) -> tuple[str, ...]:
            return tuple(f'https://example{i}.com/' for i in range(25))

    inspector = MediaInspector(None, image_decoder=FakeImageProbeDecoder(), compute_hashes=True)
    target = tmp_path / 'img.png'
    target.write_bytes(b'fake')
    analyzer = QrAnalyzer(inspector=inspector, decoder=DummyDecoder(), max_bytes=10**7)
    report = analyzer.analyze(AnalysisRequest(path=str(target)))
    d = report.to_dict()
    assert d['result']['code_count'] == 20
    assert any('truncating' in w for w in d['warnings'])
