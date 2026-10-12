"""Audio deepfake detection: recording inspection and the honest analyzer.

No audio detector is trained and no general audio decoder is installed, so this package
reads real properties from decodable (WAV) files and reports the detection gap plainly.
It never fabricates a waveform or a score.
"""

from __future__ import annotations

from ml.audio.analyzer import DEFAULT_MAX_AUDIO_BYTES, AudioAnalyzer

__all__ = [
    "DEFAULT_MAX_AUDIO_BYTES",
    "AudioAnalyzer",
]
