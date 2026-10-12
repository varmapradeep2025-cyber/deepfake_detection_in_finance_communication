"""Video deepfake detection: stream inspection and the honest analyzer.

No video detector is trained, so this package reads and reports real stream metadata
and states plainly that frame-level detection is not implemented. It never samples or
invents frames.
"""

from __future__ import annotations

from ml.video.analyzer import DEFAULT_MAX_VIDEO_BYTES, VideoAnalyzer

__all__ = [
    "DEFAULT_MAX_VIDEO_BYTES",
    "VideoAnalyzer",
]
