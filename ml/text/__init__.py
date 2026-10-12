"""Text analysis: probe-level inspection for financial communication messages.

This package provides a minimal, honest text analysis workflow. It records
supportable observations (e.g. empty input, length, basic characteristics)
and returns explicit honest statuses when no real text detector is available.
No model is trained or fabricated.
"""
from __future__ import annotations

from ml.text.analyzer import DEFAULT_MAX_TEXT_BYTES, TextAnalyzer

__all__ = [
    "DEFAULT_MAX_TEXT_BYTES",
    "TextAnalyzer",
]
