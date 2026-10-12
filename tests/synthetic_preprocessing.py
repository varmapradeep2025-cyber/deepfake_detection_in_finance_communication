"""Synthetic fixtures for preprocessing tests.

Two tiers, on purpose
---------------------
**Fake decoder** — a :class:`~datasets.preprocessing.inspection.VideoDecoder` built from
plain Python, returning frames with known statistics. The entire pipeline runs on this with
no codec installed, so the selection, quality, grouping, and split logic are testable
anywhere. Because the statistics are constructed rather than measured, a test can assert
that a specific frame is rejected for a specific reason.

**Real tiny videos** — genuinely encoded MP4s, written and read by OpenCV, skipped when
OpenCV is absent. These prove the decoder actually works on real bytes, which a fake
decoder cannot show.

Existing FF++ fixtures are deliberately *not* used for extraction tests: they are
placeholder text files that mirror the directory layout, so they are ideal for testing the
adapter and useless for testing decoding.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from datasets.preprocessing.inspection import (
    AudioStreamInfo,
    DecodedFrame,
    FrameStatistics,
    ImageStreamInfo,
    VideoStreamInfo,
)

# ---------------------------------------------------------------------------
# Fake decoder
# ---------------------------------------------------------------------------


def path_key(path: Path) -> str:
    """Normalised, case-insensitive lookup key for a media path.

    Windows and POSIX spellings of the same file have to resolve to one registration,
    otherwise a test that builds paths by string concatenation registers
    ``original/000/clip`` and then looks it up as ``original\\000\\clip`` and finds nothing.
    """
    return str(Path(path)).replace("\\", "/").strip("/").lower()


@dataclass
class FakeVideo:
    """A synthetic video: a frame count, and what each frame looks like."""

    frame_count: int = 30
    width: int = 64
    height: int = 48
    frame_rate: float = 25.0
    container_format: str = "mp4"
    #: Frames whose statistics override the defaults, by index.
    overrides: dict[int, FrameStatistics] = None  # type: ignore[assignment]
    #: When True, probe() raises, simulating an undecodable file.
    undecodable: bool = False
    #: Frame counts to report as declared, overriding ``frame_count``.
    declared_frame_count: int | None = None

    def __post_init__(self) -> None:
        if self.overrides is None:
            self.overrides = {}

    @property
    def statistics(self) -> FrameStatistics:
        """Default per-frame statistics: mid-grey, textured, not flat."""
        return FrameStatistics(
            width=self.width,
            height=self.height,
            mean_intensity=120.0,
            std_intensity=45.0,
            min_intensity=40,
            max_intensity=210,
            is_uniform=False,
        )


class FakeVideoDecoder:
    """A decoder that serves :class:`FakeVideo` instances by path stem.

    Implements the full :class:`~datasets.preprocessing.inspection.VideoDecoder` protocol,
    including ``name``. Every method raises rather than guessing when a file is not
    registered, which mirrors the real decoder's behaviour on an unknown path and stops a
    typo in a test from silently producing an empty result.
    """

    name = "fake"

    def __init__(self, videos: dict[str, FakeVideo] | None = None) -> None:
        self.videos: dict[str, FakeVideo] = dict(videos or {})
        self.read_calls: list[str] = []
        self.keep_images_seen: list[bool] = []

    def add(self, path: Path | str, video: FakeVideo) -> FakeVideo:
        """Register a video for a path, and return it.

        Registered under both the full path and the bare stem. The path is what makes two
        same-named clips in different directories distinguishable, which is the whole
        point of a dataset laid out as `<manipulation>/<recording>/clip.mp4`; the stem
        keeps short registrations ergonomic.
        """
        candidate = Path(path)
        self.videos[path_key(candidate)] = video
        self.videos[candidate.stem] = video
        return video

    def _lookup(self, path: Path) -> FakeVideo:
        for key in (path_key(path), path.stem):
            if key in self.videos:
                return self.videos[key]
        raise RuntimeError(f"FakeVideoDecoder has no video registered for {path}")

    def probe(self, path: Path) -> VideoStreamInfo:
        """Return stream properties, or raise for an undecodable video."""
        video = self._lookup(Path(path))
        if video.undecodable:
            raise RuntimeError(f"cannot decode {path}")
        declared = (
            video.declared_frame_count
            if video.declared_frame_count is not None
            else video.frame_count
        )
        return VideoStreamInfo(
            container_format=video.container_format,
            codec="fake_codec",
            width=video.width,
            height=video.height,
            duration_seconds=video.frame_count / video.frame_rate,
            declared_frame_count=declared,
            frame_rate=video.frame_rate,
        )

    def read_frames(self, path: Path, *, keep_images: bool = False) -> Iterator[DecodedFrame]:
        """Yield every frame in order, honouring per-frame statistics overrides."""
        target = Path(path)
        video = self._lookup(target)
        if video.undecodable:
            raise RuntimeError(f"cannot decode {target}")
        self.read_calls.append(target.stem)
        self.keep_images_seen.append(keep_images)
        for index in range(video.frame_count):
            yield DecodedFrame(
                index=index,
                statistics=video.overrides.get(index, video.statistics),
                image=b"pixels" if keep_images else None,
            )


# ---------------------------------------------------------------------------
# Fake image and audio decoders
# ---------------------------------------------------------------------------


@dataclass
class FakeImage:
    """A synthetic still image: its pixel dimensions and channel layout."""

    width: int = 64
    height: int = 48
    channels: int = 3
    color_mode: str = "BGR"
    format: str = "png"
    #: When True, probe() raises, simulating an undecodable file.
    undecodable: bool = False


class FakeImageDecoder:
    """An :class:`~datasets.preprocessing.inspection.ImageDecoder` serving :class:`FakeImage`.

    Mirrors :class:`FakeVideoDecoder`: registered by path and stem, raising on an unknown
    path so a typo cannot quietly yield an empty probe.
    """

    name = "fake-image"

    def __init__(self, images: dict[str, FakeImage] | None = None) -> None:
        self.images: dict[str, FakeImage] = dict(images or {})

    def add(self, path: Path | str, image: FakeImage) -> FakeImage:
        candidate = Path(path)
        self.images[path_key(candidate)] = image
        self.images[candidate.stem] = image
        return image

    def _lookup(self, path: Path) -> FakeImage:
        for key in (path_key(path), path.stem):
            if key in self.images:
                return self.images[key]
        raise RuntimeError(f"FakeImageDecoder has no image registered for {path}")

    def probe(self, path: Path) -> ImageStreamInfo:
        image = self._lookup(Path(path))
        if image.undecodable:
            raise RuntimeError(f"cannot decode {path}")
        return ImageStreamInfo(
            width=image.width,
            height=image.height,
            channels=image.channels,
            color_mode=image.color_mode,
            format=image.format,
        )


@dataclass
class FakeAudio:
    """A synthetic audio stream."""

    sample_rate: int = 16000
    channels: int = 1
    sample_count: int = 32000
    sample_width_bits: int = 16
    codec: str = "pcm"
    #: When True, probe() raises, simulating an undecodable file.
    undecodable: bool = False

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate


class FakeAudioDecoder:
    """An :class:`~datasets.preprocessing.inspection.AudioDecoder` serving :class:`FakeAudio`."""

    name = "fake-audio"

    def __init__(self, clips: dict[str, FakeAudio] | None = None) -> None:
        self.clips: dict[str, FakeAudio] = dict(clips or {})

    def add(self, path: Path | str, clip: FakeAudio) -> FakeAudio:
        candidate = Path(path)
        self.clips[path_key(candidate)] = clip
        self.clips[candidate.stem] = clip
        return clip

    def _lookup(self, path: Path) -> FakeAudio:
        for key in (path_key(path), path.stem):
            if key in self.clips:
                return self.clips[key]
        raise RuntimeError(f"FakeAudioDecoder has no clip registered for {path}")

    def probe(self, path: Path) -> AudioStreamInfo:
        clip = self._lookup(Path(path))
        if clip.undecodable:
            raise RuntimeError(f"cannot decode {path}")
        return AudioStreamInfo(
            sample_rate=clip.sample_rate,
            channels=clip.channels,
            duration_seconds=clip.duration_seconds,
            sample_count=clip.sample_count,
            sample_width_bits=clip.sample_width_bits,
            codec=clip.codec,
        )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def textured_statistics(
    *, mean: float = 120.0, width: int = 64, height: int = 48
) -> FrameStatistics:
    """Statistics for a healthy frame: mid-tone, varied, not flat.

    The default-rejection tests need a frame that is ordinary, which
    :func:uniform_statistics is not - it is flat by construction.
    """
    return FrameStatistics(
        width=width,
        height=height,
        mean_intensity=mean,
        std_intensity=45.0,
        min_intensity=40,
        max_intensity=210,
        is_uniform=False,
    )


def uniform_statistics(value: int = 7, *, width: int = 64, height: int = 48) -> FrameStatistics:
    """Statistics for a completely flat frame."""
    return FrameStatistics(
        width=width,
        height=height,
        mean_intensity=float(value),
        std_intensity=0.0,
        min_intensity=value,
        max_intensity=value,
        is_uniform=True,
    )


def dark_statistics(mean: float = 3.0, *, width: int = 64, height: int = 48) -> FrameStatistics:
    """Statistics for a well-formed but very dark frame: not flat, low intensity."""
    return FrameStatistics(
        width=width,
        height=height,
        mean_intensity=mean,
        std_intensity=12.0,
        min_intensity=0,
        max_intensity=int(mean) + 30,
        is_uniform=False,
    )


def small_statistics(width: int = 8, height: int = 8) -> FrameStatistics:
    """Statistics for a frame below the default minimum resolution."""
    return FrameStatistics(
        width=width,
        height=height,
        mean_intensity=120.0,
        std_intensity=40.0,
        min_intensity=30,
        max_intensity=220,
        is_uniform=False,
    )


# ---------------------------------------------------------------------------
# Real videos
# ---------------------------------------------------------------------------


def opencv_available() -> bool:
    """Whether OpenCV can be imported."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return False
    return True


requires_opencv = pytest.mark.skipif(
    not opencv_available(), reason="opencv-python-headless is not installed"
)


def write_real_video(
    path: Path,
    *,
    frame_count: int = 24,
    width: int = 64,
    height: int = 48,
    frame_rate: float = 25.0,
) -> Path:
    """Encode a real MP4 whose frames differ from one another.

    A gradient that advances with the frame index, so a decode that returns the wrong
    frame is detectable rather than looking identical to the right one.

    Raises:
        AssertionError: If OpenCV cannot write the file, which would make any test using
            it meaningless.
    """
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height)
    )
    if not writer.isOpened():
        raise AssertionError(f"OpenCV could not open a writer for {path}")
    try:
        for index in range(frame_count):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[:, :, 0] = (index * 7) % 256
            image[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
            writer.write(image)
    finally:
        writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"OpenCV wrote no bytes to {path}")
    return path


def write_real_image(
    path: Path,
    *,
    width: int = 64,
    height: int = 48,
    channels: int = 3,
) -> Path:
    """Encode a real still image with the requested channel count.

    ``channels`` is 1 for grayscale, 3 for BGR colour, or 4 for BGRA.

    Raises:
        AssertionError: If OpenCV cannot write the file, which would make any test using
            it meaningless.
    """
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        image = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, axis=0)
    else:
        image = np.zeros((height, width, channels), dtype=np.uint8)
        image[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        image[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        if channels == 4:
            image[:, :, 3] = 255
    if not cv2.imwrite(str(path), image):
        raise AssertionError(f"OpenCV could not write an image to {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"OpenCV wrote no bytes to {path}")
    return path


def write_real_wav(
    path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_count: int = 1600,
    sample_width: int = 2,
) -> Path:
    """Encode a real PCM WAV file using the standard-library writer.

    ``sample_width`` is the byte width per sample (2 for 16-bit). A quiet tone is written
    so the file is not just a header, and an all-zero body would still be a valid probe
    target.

    Raises:
        AssertionError: If the file cannot be written or ends up empty.
    """
    import array
    import math
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = array.array("h")
    for index in range(sample_count):
        value = int(3000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend([value] * channels)
    try:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(sample_width)
            handle.setframerate(sample_rate)
            handle.writeframes(frames.tobytes())
    except wave.Error as exc:  # pragma: no cover - defensive
        raise AssertionError(f"could not write a WAV file to {path}: {exc}") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"the WAV writer wrote no bytes to {path}")
    return path


def real_frame_statistics(path: Path) -> list[int]:
    """Read the mean blue channel of every frame, for identity assertions.

    Comparing these across two extractions of the same file is the direct test that
    selection is reproducible.
    """
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"OpenCV could not open {path}")
    means: list[int] = []
    try:
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            means.append(int(np.asarray(image)[:, :, 0].mean()))
    finally:
        capture.release()
    return means


def make_dummy_file(path: Path, *, size: int = 32) -> Path:
    """Create a small non-video file with the right extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)
    return path


def numpy_frame(value: int = 128, *, width: int = 8, height: int = 8) -> Any:
    """A small solid-colour array, for testing :func:`compute_frame_statistics`."""
    import numpy as np

    return np.full((height, width, 3), value, dtype=np.uint8)
