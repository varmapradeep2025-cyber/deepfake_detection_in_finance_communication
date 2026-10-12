"""QR-code analysis: decode the payload, then grade any URL it contains.

Decoding is a genuine, verifiable operation; the phishing grading that follows is
heuristic and never a verdict. No network access is performed.
"""

from __future__ import annotations

from ml.qr.analyzer import DEFAULT_MAX_QR_BYTES, QrAnalyzer
from ml.qr.decode import (
    OpenCVQrDecoder,
    QrDecoder,
    QrDecoderUnavailableError,
    default_qr_decoder,
)

__all__ = [
    "DEFAULT_MAX_QR_BYTES",
    "OpenCVQrDecoder",
    "QrAnalyzer",
    "QrDecoder",
    "QrDecoderUnavailableError",
    "default_qr_decoder",
]
