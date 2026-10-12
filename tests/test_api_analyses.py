"""End-to-end tests for the standard-library inference API.

These drive a real ``ThreadingHTTPServer`` on an ephemeral port and speak to it with
``urllib``, so the multipart parsing, routing, status codes, and the storage round
trip are all exercised over the wire rather than by calling the handlers directly.

The analyzer is the synthetic one from :mod:`tests.synthetic_inference`: it needs no
real image, so the tests stay codec-independent. The one place a genuine model is
needed — asserting a scored ``analysis_complete`` response — uses a synthetic model
card whose entry point is a real import target, so the production ``load_model_backend``
path runs.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from backend.api import BackendApplication, build_handler, parse_multipart
from ml.inference.records import AnalysisStore
from ml.inference.service import MediaAnalysisService
from tests.synthetic_inference import (
    ENTRYPOINT_MODULE,
    make_image_analyzer,
    make_image_array,
    write_model_card,
)

_BOUNDARY = "----deepfakeFinanceTestBoundary"

_SAMPLE_IMAGE_BYTES = b"\x89PNG\r\n\x1a\nnot-a-real-png-but-real-bytes"


def _build_multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes]],
) -> tuple[bytes, str]:
    """Encode a multipart/form-data body, returning ``(body, content_type)``."""
    body = bytearray()
    for name, value in fields.items():
        body += f"--{_BOUNDARY}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += value.encode("utf-8")
        body += b"\r\n"
    for name, (filename, content) in files.items():
        body += f"--{_BOUNDARY}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        ).encode()
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += content
        body += b"\r\n"
    body += f"--{_BOUNDARY}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={_BOUNDARY}"


def _call(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, object, dict[str, str]]:
    """Perform one request and return ``(status, json_payload, headers)``."""
    headers: dict[str, str] = {"Accept": "application/json"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload, dict(response.headers)
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8")
        try:
            payload: object = json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - responses are always JSON
            payload = raw
        return error.code, payload, dict(error.headers)


@pytest.fixture
def run_server():
    """Start a server per call and tear every one down at test end."""
    running: list[tuple[ThreadingHTTPServer, threading.Thread]] = []

    def _run(app: BackendApplication) -> str:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(app))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        running.append((httpd, thread))
        host, port = httpd.server_address[:2]
        return f"http://{host}:{port}"

    yield _run
    for httpd, thread in running:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _make_app(tmp_path, *, with_model: bool = False, max_upload_bytes: int | None = None):
    models_dir = tmp_path / "models"
    if with_model:
        write_model_card(
            models_dir,
            "demo",
            entrypoint=f"{ENTRYPOINT_MODULE}:build_scoring_backend",
        )
    analyzer = make_image_analyzer(models_dir=models_dir, array=make_image_array())
    kwargs: dict[str, int] = {}
    if max_upload_bytes is not None:
        kwargs["max_upload_bytes"] = max_upload_bytes
    return BackendApplication(
        service=MediaAnalysisService(image_analyzer=analyzer),
        store=AnalysisStore(tmp_path / "store"),
        **kwargs,
    )


def _post_image(url: str, *, modality: str | None = "image", filename: str = "sample.png"):
    fields = {} if modality is None else {"modality": modality}
    body, content_type = _build_multipart(fields, {"file": (filename, _SAMPLE_IMAGE_BYTES)})
    return _call(f"{url}/analyses", method="POST", body=body, content_type=content_type)


class TestMultipartParsing:
    def test_parses_fields_and_file(self):
        body, content_type = _build_multipart(
            {"modality": "image"}, {"file": ("photo.png", b"payload")}
        )
        parsed = parse_multipart(content_type, body)
        assert parsed is not None
        fields, files = parsed
        assert fields == {"modality": "image"}
        filename, content = files["file"]
        assert filename == "photo.png"
        assert content == b"payload"

    def test_rejects_non_multipart(self):
        assert parse_multipart("application/json", b"{}") is None


class TestHealth:
    def test_health_reports_ok(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, payload, _ = _call(f"{url}/health")
        assert status == 200
        assert payload["status"] == "ok"

    def test_health_sets_cors_header(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        _, _, headers = _call(f"{url}/health")
        assert headers.get("Access-Control-Allow-Origin") == "*"


class TestAnalyzeUpload:
    def test_image_without_model_is_model_unavailable(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, payload, _ = _post_image(url)
        assert status == 200
        assert payload["status"] == "model_unavailable"
        assert payload["modality"] == "image"
        assert payload["source_name"] == "sample.png"
        assert payload["report"]["path"] == "sample.png"
        assert payload["report"]["output"] is None

    def test_image_with_model_is_analysis_complete(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path, with_model=True))
        status, payload, _ = _post_image(url)
        assert status == 200
        assert payload["status"] == "analysis_complete"
        assert payload["report"]["output"]["raw_score"] == pytest.approx(0.93)

    def test_unimplemented_modality_is_reported_not_error(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, payload, _ = _post_image(url, modality="video", filename="clip.mp4")
        assert status == 200
        assert payload["status"] == "unsupported_modality"

    def test_missing_modality_is_400(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, payload, _ = _post_image(url, modality=None)
        assert status == 400
        assert "modality" in payload["error"]

    def test_missing_file_is_400(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        body, content_type = _build_multipart({"modality": "image"}, {})
        status, payload, _ = _call(
            f"{url}/analyses", method="POST", body=body, content_type=content_type
        )
        assert status == 400
        assert "file" in payload["error"]

    def test_unknown_modality_is_400(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, payload, _ = _post_image(url, modality="telepathy")
        assert status == 400
        assert "telepathy" in payload["error"]

    def test_empty_file_is_422(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        body, content_type = _build_multipart(
            {"modality": "image"}, {"file": ("empty.png", b"")}
        )
        status, payload, _ = _call(
            f"{url}/analyses", method="POST", body=body, content_type=content_type
        )
        assert status == 422
        assert "empty" in payload["error"]

    def test_oversize_upload_is_413(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path, max_upload_bytes=16))
        body, content_type = _build_multipart(
            {"modality": "image"}, {"file": ("big.png", b"x" * 128)}
        )
        status, _, _ = _call(
            f"{url}/analyses", method="POST", body=body, content_type=content_type
        )
        assert status == 413

    def test_non_multipart_body_is_415(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, _, _ = _call(
            f"{url}/analyses",
            method="POST",
            body=b"{}",
            content_type="application/json",
        )
        assert status == 415

    def test_filename_path_traversal_is_neutralized(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, payload, _ = _post_image(url, filename="../../evil.png")
        assert status == 200
        assert payload["source_name"] == "evil.png"
        assert payload["report"]["path"] == "evil.png"


class TestRecords:
    def test_post_then_list_and_get_roundtrip(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path, with_model=True))
        _, created, _ = _post_image(url)
        report_id = created["report_id"]

        status, listing, _ = _call(f"{url}/analyses")
        assert status == 200
        assert isinstance(listing, list)
        assert [record["report_id"] for record in listing] == [report_id]

        status, fetched, _ = _call(f"{url}/analyses/{report_id}")
        assert status == 200
        assert fetched["report_id"] == report_id
        assert fetched["status"] == "analysis_complete"

    def test_unknown_record_is_404(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, _, _ = _call(f"{url}/analyses/does-not-exist")
        assert status == 404

    def test_unknown_route_is_404(self, tmp_path, run_server):
        url = run_server(_make_app(tmp_path))
        status, _, _ = _call(f"{url}/nope")
        assert status == 404
