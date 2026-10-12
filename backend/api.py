"""A small, standard-library HTTP server over the inference service.

Why no framework
----------------
The project's ``requirements.txt`` keeps FastAPI commented out on purpose
(``scripts/verify_structure.py`` asserts it stays that way), and this checkout ships
no web dependency. The frontend already speaks a plain contract — ``POST /analyses``
as ``multipart/form-data`` with ``modality`` and ``file`` fields, plus ``GET`` routes
for the stored records and a ``/health`` probe — so the server implements exactly
that over :mod:`http.server`. It adds no capability the library does not have and
invents no result: an uploaded image with no registered model is answered with the
library's ``model_unavailable`` report, not a fabricated score.

Request lifecycle
-----------------
1. The multipart body is bounded by ``Content-Length`` before it is read, so an
   oversize upload is refused with ``413`` rather than buffered into memory.
2. The ``file`` part is written to a temporary file inside a per-request temporary
   directory. The library analyzes a *server-side path*, never the socket, so this is
   the bridge from the uploaded bytes to the analyzer.
3. :class:`~ml.inference.service.MediaAnalysisService` runs and returns an
   :class:`~ml.inference.report.AnalysisReport`.
4. The report is stored as an :class:`~ml.inference.records.AnalysisRecord` and
   returned as JSON. The recorded artifact path is the client's file name, not the
   now-deleted temporary path, since the temporary path is meaningless after the
   request and the uploaded bytes were the named file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from email.message import Message
from email.parser import BytesParser
from email.policy import default as default_email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

from ml.image.analyzer import DEFAULT_MAX_IMAGE_BYTES, ImageAnalyzer
from ml.inference.modality import AnalysisModality
from ml.inference.records import AnalysisRecord, AnalysisStore
from ml.inference.request import AnalysisRequest
from ml.inference.service import MediaAnalysisService

__all__ = [
    "BackendApplication",
    "build_application",
    "build_handler",
    "serve",
]

LOGGER = logging.getLogger("backend.api")

#: Slack allowed on top of the file-size ceiling to cover multipart boundaries and
#: the small text fields. Chosen so a single request can never exceed the limit by
#: more than one sanely-sized header block.
_MULTIPART_OVERHEAD = 1024 * 1024

#: Version reported by the health probe. Kept in step with ``project.version``.
SERVICE_VERSION = "0.1.0"


def _display_name(filename: str) -> str:
    """A safe display name derived from a client-supplied file name.

    Only the final path component is kept, so a crafted ``../../etc/passwd`` cannot
    land outside the temporary directory; an empty result falls back to ``"upload"``.
    """
    candidate = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return candidate or "upload"


def _part_bytes(part: Message) -> bytes:
    """Return a MIME part's decoded payload as bytes, defensively."""
    payload = part.get_payload(decode=True)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    return b""


def _part_text(part: Message) -> str:
    """Return a MIME part's payload as UTF-8 text."""
    return _part_bytes(part).decode("utf-8", "replace")


def parse_multipart(
    content_type: str,
    body: bytes,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]] | None:
    """Parse a ``multipart/form-data`` body into text fields and file parts.

    Args:
        content_type: The request's ``Content-Type`` header, carrying the boundary.
        body: The raw request body.

    Returns:
        ``(fields, files)`` where ``fields`` maps a text field name to its value and
        ``files`` maps a file field name to ``(filename, content)``. ``None`` when the
        content type is not multipart or the body cannot be parsed.
    """
    lowered = content_type.lower()
    if "multipart/form-data" not in lowered or "boundary=" not in lowered:
        return None

    prelude = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    try:
        message = BytesParser(policy=default_email_policy).parsebytes(prelude + body)
    except Exception:
        return None

    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in message.walk():
        if part.get_content_maintype() == "multipart":
            continue
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            continue
        filename = part.get_filename()
        if filename is not None:
            files[name] = (filename, _part_bytes(part))
        else:
            fields[name] = _part_text(part)
    return fields, files


@dataclass(slots=True)
class BackendApplication:
    """The request handlers, independent of any HTTP transport.

    Args:
        service: The analysis service that routes by modality.
        store: Where completed analyses are persisted.
        max_upload_bytes: Ceiling on the uploaded file's size. Defaults to the image
            analyzer's own limit so both layers agree.
        version: Reported by ``/health``.
    """

    service: MediaAnalysisService
    store: AnalysisStore
    max_upload_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    version: str = SERVICE_VERSION

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0:
            raise ValueError(f"max_upload_bytes must be positive, got {self.max_upload_bytes}")

    @property
    def max_request_bytes(self) -> int:
        """Upper bound on the whole request body, file plus multipart framing."""
        return self.max_upload_bytes + _MULTIPART_OVERHEAD

    def health(self) -> dict[str, Any]:
        """The health payload."""
        return {
            "status": "ok",
            "service": "deepfake-finance-inference",
            "version": self.version,
        }

    def list_records(self) -> list[dict[str, Any]]:
        """Every stored record, as serialized JSON objects."""
        return [record.to_dict() for record in self.store.list_records()]

    def get_record(self, report_id: str) -> tuple[int, dict[str, Any]]:
        """Return one stored record as ``(status, payload)``."""
        try:
            record = self.store.load(report_id)
        except FileNotFoundError:
            return 404, {"error": f"no analysis record {report_id!r}"}
        except (ValueError, TypeError) as exc:
            return 500, {"error": f"stored record {report_id!r} is unreadable: {exc}"}
        return 200, record.to_dict()

    def analyze(
        self,
        modality_raw: str | None,
        filename: str | None,
        data: bytes | None,
    ) -> tuple[int, dict[str, Any]]:
        """Analyze one uploaded file and return ``(http_status, payload)``.

        Transport-level problems are HTTP status codes; once the bytes reach the
        library, the outcome is an honest ``AnalysisReport`` with its own status,
        returned with ``200``. A missing model is therefore a successful request that
        reports ``model_unavailable``, exactly as the library defines it.
        """
        if modality_raw is None or not modality_raw.strip():
            return 400, {"error": "the 'modality' form field is required"}
        if filename is None or not filename.strip():
            return 400, {"error": "the 'file' form field is required"}
        if not data:
            return 422, {"error": "the uploaded file is empty"}

        try:
            modality = AnalysisModality.parse(modality_raw)
        except ValueError as exc:
            return 400, {"error": str(exc)}

        if len(data) > self.max_upload_bytes:
            return 413, {
                "error": (
                    f"upload is {len(data)} bytes, over the {self.max_upload_bytes}-byte "
                    f"{modality.value} limit"
                )
            }

        display_name = _display_name(filename)
        try:
            with TemporaryDirectory(prefix="dfr-analysis-") as temporary:
                upload_path = Path(temporary) / "upload"
                upload_path.write_bytes(data)
                request = AnalysisRequest(path=str(upload_path), modality=modality)
                report = self.service.analyze(request)
        except OSError as exc:
            logging.getLogger("backend.api").exception("could not persist the upload")
            return 500, {"error": f"could not persist the uploaded file: {exc}"}

        stored_report = replace(report, path=display_name)
        record = AnalysisRecord.from_report(stored_report, source_name=display_name)
        try:
            self.store.save(record)
        except (OSError, FileExistsError) as exc:  # pragma: no cover - disk failures
            logging.getLogger("backend.api").warning(
                "could not store record %s: %s", record.report_id, exc
            )
        return 200, record.to_dict()

    def list_routes(self) -> dict[str, Any]:
        """A tiny service description, for a browser hitting the root by hand."""
        return {
            "service": "deepfake-finance-inference",
            "version": self.version,
            "endpoints": [
                "GET /health",
                "GET /analyses",
                "GET /analyses/{report_id}",
                "POST /analyses",
            ],
        }


def build_application(
    *,
    config: Any | None = None,
    models_dir: Path | str | None = None,
    store_dir: Path | str | None = None,
) -> BackendApplication:
    """Wire a :class:`BackendApplication` from project config.

    Only the image analyzer is constructed: it is the modality with a real, exercised
    workflow. A request for any other modality is answered by the service with
    ``unsupported_modality`` rather than an error. When OpenCV is unavailable the
    image analyzer cannot be built, and the application says so by reporting
    ``unsupported_modality`` for image requests instead of failing to start.

    Args:
        config: A loaded project config, or ``None`` to load the default.
        models_dir: Where to scan for model cards. Defaults to the configured
            ``<artifacts>/models``.
        store_dir: Where to persist analysis records. Defaults to
            ``<artifacts>/records``.
    """
    from ml.reproducibility.config import load_project_config

    if config is None:
        config = load_project_config()

    image_analyzer: ImageAnalyzer | None
    try:
        image_analyzer = ImageAnalyzer.from_project_config(config, models_dir=models_dir)
    except RuntimeError as exc:
        logging.getLogger(__name__).warning("image analysis disabled: %s", exc)
        image_analyzer = None

    if store_dir is None:
        store_dir = Path(config.get_path("paths.generated.artifacts")) / "records"

    return BackendApplication(
        service=MediaAnalysisService(image_analyzer=image_analyzer),
        store=AnalysisStore(store_dir),
    )


def build_handler(app: BackendApplication) -> type[BaseHTTPRequestHandler]:
    """Return a request handler class bound to ``app``."""

    class _AnalysisRequestHandler(BaseHTTPRequestHandler):
        server_version = "DeepfakeFinanceInference/0.1"
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:
            path = _normalized_path(self.path)
            if path == "/health":
                self._send_json(200, app.health())
                return
            if path == "/":
                self._send_json(200, app.list_routes())
                return
            if path == "/analyses":
                self._send_json(200, app.list_records())
                return
            prefix = "/analyses/"
            if path.startswith(prefix):
                report_id = path[len(prefix) :]
                status, payload = app.get_record(report_id)
                self._send_json(status, payload)
                return
            self._send_json(404, {"error": f"no route for GET {path}"})

        def do_POST(self) -> None:
            path = _normalized_path(self.path)
            if path != "/analyses":
                self._send_json(404, {"error": f"no route for POST {path}"})
                return

            content_type = self.headers.get("Content-Type", "")
            length_header = self.headers.get("Content-Length")
            if length_header is None:
                self._send_json(411, {"error": "a Content-Length header is required"})
                return
            try:
                length = int(length_header)
            except ValueError:
                self._send_json(400, {"error": "Content-Length must be an integer"})
                return
            if length < 0:
                self._send_json(400, {"error": "Content-Length must not be negative"})
                return
            if length > app.max_request_bytes:
                # Drain the declared body so closing does not reset a client that is
                # still writing, then refuse it without ever holding it in memory.
                self._drain(length)
                self.close_connection = True
                self._send_json(
                    413,
                    {"error": "request body exceeds the maximum upload size"},
                )
                return

            body = self.rfile.read(length) if length else b""
            parsed = parse_multipart(content_type, body)
            if parsed is None:
                self._send_json(
                    415,
                    {"error": "expected a multipart/form-data request body"},
                )
                return

            fields, files = parsed
            filename, file_bytes = files.get("file", (None, None))
            status, payload = app.analyze(fields.get("modality"), filename, file_bytes)
            self._send_json(status, payload)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_common_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            LOGGER.debug("%s - %s", self.address_string(), fmt % args)

        # --- helpers ------------------------------------------------------

        def _send_common_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")

        def _drain(self, length: int) -> None:
            """Read and discard ``length`` bytes in bounded chunks."""
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._send_common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _AnalysisRequestHandler


def _normalized_path(raw_path: str) -> str:
    """The route path, without query or a trailing slash."""
    path = urlsplit(raw_path).path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def serve(app: BackendApplication, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Serve ``app`` until interrupted.

    Args:
        app: The application to expose.
        host: Interface to bind. Defaults to loopback; the service is local-only.
        port: TCP port to bind. Use ``0`` to pick a free port.
    """
    handler = build_handler(app)
    httpd = ThreadingHTTPServer((host, port), handler)
    bound_host, bound_port = httpd.server_address[:2]
    LOGGER.info("inference API listening on http://%s:%s", bound_host, bound_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown
        LOGGER.info("shutdown requested")
    finally:
        httpd.server_close()
