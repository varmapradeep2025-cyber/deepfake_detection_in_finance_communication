"""QR decoding, with the decoder injected so the logic is codec-independent.

Decoding a QR image is a genuine, verifiable operation: the payload comes out of the
image, or the image holds no QR code. This module wraps that operation behind a tiny
:class:`QrDecoder` interface so the analyzer is testable with a fake decoder and can run
with no OpenCV installed (reporting a capability gap rather than pretending).

The bundled :class:`OpenCVQrDecoder` uses ``cv2.QRCodeDetector``, which reads one or more
codes from an image. It reports *what the image encodes*; it says nothing about whether
that payload is trustworthy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "OpenCVQrDecoder",
    "QrDecoder",
    "QrDecoderUnavailableError",
    "default_qr_decoder",
]


class QrDecoderUnavailableError(RuntimeError):
    """Raised when a QR decoder is required but not installed."""


@runtime_checkable
class QrDecoder(Protocol):
    """What the analyzer needs from a QR decoder.

    Stateless: :meth:`decode` takes a path and returns every payload the image contains,
    so no handle is held between calls.
    """

    #: Short stable identifier recorded in evidence.
    name: str

    def decode(self, path: Path) -> tuple[str, ...]:
        """Return every decoded payload in the image, or an empty tuple if none.

        Raises:
            Exception: If the file cannot be opened as an image.
        """
        ...


class OpenCVQrDecoder:
    """A :class:`QrDecoder` backed by headless OpenCV.

    Uses ``detectAndDecodeMulti`` so an image carrying several codes yields all of them,
    falling back to single detection on builds where multi-detection is unavailable. An
    image with no detectable code returns an empty tuple — that is a legitimate result,
    distinct from a file that is not an image at all, which raises.
    """

    name = "opencv-qrcode"

    def __init__(self) -> None:
        self._cv2: Any | None = None

    def _module(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise QrDecoderUnavailableError(
                    "opencv-python-headless is not installed. Install it with "
                    "'pip install -r requirements.txt', or inject another QrDecoder."
                ) from exc
            self._cv2 = cv2
        return self._cv2

    def decode(self, path: Path) -> tuple[str, ...]:
        """Return every payload decoded from the image.

        Raises:
            QrDecoderUnavailableError: If OpenCV is not installed.
            RuntimeError: If the file cannot be opened as an image.
        """
        cv2 = self._module()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"OpenCV could not open {path} as an image")

        detector = cv2.QRCodeDetector()
        if hasattr(detector, "detectAndDecodeMulti"):
            ok, decoded_info, _points, _straight = detector.detectAndDecodeMulti(image)
            if ok:
                payloads = tuple(text for text in decoded_info if text)
                if payloads:
                    return payloads

        data, _points, _straight = detector.detectAndDecode(image)
        return (data,) if data else ()


def default_qr_decoder() -> QrDecoder | None:
    """Return the OpenCV QR decoder, or ``None`` when OpenCV is not installed.

    Returning ``None`` mirrors the media inspector: QR analysis is still useful without a
    decoder (the file is hashed and sized and a capability gap is reported), so a missing
    optional dependency is a reportable state, not a crash.
    """
    try:
        import cv2  # noqa: F401  (probe only)
    except ImportError:
        return None
    return OpenCVQrDecoder()
