"""Text preprocessing and inspection helpers (probe-level).

This module provides minimal, deterministic helpers to inspect text inputs
without inferring intent, fabricating features, or training models. All
observations are supportable from the input string itself.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "TextInspection",
    "inspect_text",
]


@dataclass(frozen=True, slots=True)
class TextInspection:
    """Basic, supportable observations about a text input.

    Attributes:
        is_empty: Whether the text contains no non-whitespace characters.
        character_count: Total character count (including whitespace).
        non_whitespace_count: Number of non-whitespace characters.
        line_count: Number of lines.
        word_count: Number of whitespace-separated tokens.
        has_leading_trailing_whitespace: Whether leading/trailing whitespace exists.
    """

    is_empty: bool
    character_count: int
    non_whitespace_count: int
    line_count: int
    word_count: int
    has_leading_trailing_whitespace: bool


def inspect_text(text: str) -> TextInspection:
    """Return basic characteristics of the text without interpretation."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    stripped = text.strip()
    is_empty = len(stripped) == 0
    character_count = len(text)
    non_ws = sum(1 for c in text if not c.isspace())
    lines = text.splitlines() if text else []
    if text and lines:
        line_count = len(lines)
    elif text:
        line_count = 1
    else:
        line_count = 0
    word_count = len(stripped.split()) if stripped else 0
    has_leading_trailing_whitespace = text != stripped
    return TextInspection(
        is_empty=is_empty,
        character_count=character_count,
        non_whitespace_count=non_ws,
        line_count=line_count,
        word_count=word_count,
        has_leading_trailing_whitespace=has_leading_trailing_whitespace,
    )

