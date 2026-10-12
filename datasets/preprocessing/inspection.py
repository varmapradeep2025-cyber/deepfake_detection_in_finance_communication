"""Media inspection: what a media file actually is, and whether it can be decoded.

Covers all three modalities the project ingests — video, image, and audio — behind one
record type. A video probe reads container, codec, dimensions, frame rate and duration; an
image probe reads pixel dimensions and channel layout; an audio probe reads sample rate,
channel count and duration.

The rule this module enforces
-----------------------------
**A field that could not be read is recorded as unavailable, never as a plausible
default.** A decoder that cannot open a file does not get to produce a record saying
``width=1920, height=1080, duration=0.0`` — a fabricated inspection is worse than a
missing one, because it looks like evidence. So every field is ``None`` unless it came
back from a real probe, and every failure carries a status and a message naming the
file.

The status is separate from the fields on purpose. ``width=None`` could mean the file is
a text file, or that the codec is missing, or that probing was skipped.
:class:`InspectionStatus` says which, so a report can distinguish "this is not media" from
"this is media and we could not read it", which are different problems with different
fixes.

Decoders are injected
---------------------
Nothing here imports a video, image, or audio library. The inspector takes a
:class:`VideoDecoder`, an :class:`ImageDecoder`, and an :class:`AudioDecoder`, so:

* the decoding strategy is a decision recorded in one place,
* the logic here is testable with no codec installed at all,
* and swapping :class:`~datasets.preprocessing.inspection.OpenCVDecoder` or
  :class:`~datasets.preprocessing.inspection.OpenCVImageDecoder` for a different one does
  not touch a line of the pipeline.

The bundled audio decoder is the standard-library :mod:`wave` reader, which needs no
dependency and is exact for the ``.wav`` files the project's tooling can produce; other
audio containers require an injected :class:`AudioDecoder`.

Hashing is not delegated. :func:`~datasets.manifest.calculate_file_sha256` is the single
implementation of file identity in this project, so it is used directly rather than
reimplemented.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from datasets.manifest import DEFAULT_CHUNK_SIZE, calculate_file_sha256
from datasets.schema import Modality

__all__ = [
    "AudioDecoder",
    "AudioStreamInfo",
    "DecodedFrame",
    "FrameStatistics",
    "ImageDecoder",
    "ImageStreamInfo",
    "InspectionStatus",
    "MediaInspection",
    "OpenCVDecoder",
    "OpenCVImageDecoder",
    "VideoDecoder",
    "VideoDecoderUnavailableError",
    "VideoStreamInfo",
    "WaveAudioDecoder",
    "compute_frame_statistics",
    "default_audio_decoder",
    "default_decoder",
    "default_image_decoder",
    "infer_modality",
    "is_audio_candidate",
    "is_image_candidate",
    "is_video_candidate",
]

#: Extensions treated as video when a caller does not supply its own list. Used for
#: *reporting* and grouping only — never to reject a file without saying so.
DEFAULT_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
)

#: Extensions treated as still images. Same reporting-only contract as the video set.
DEFAULT_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
)

#: Extensions treated as audio. Same reporting-only contract as the video set.
DEFAULT_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".mp3", ".ogg", ".oga", ".m4a", ".aac", ".aif", ".aiff", ".wma"}
)


class InspectionStatus(str, Enum):
    """Outcome of inspecting one file."""

    #: The file was read and its stream properties were probed.
    OK = "ok"
    #: The path is missing, is not a file, or could not be read or hashed.
    FILE_UNREADABLE = "file_unreadable"
    #: The file exists but is zero bytes, so it cannot contain any media.
    EMPTY = "empty"
    #: The file exists and hashed, but no decoder could open it as media.
    DECODE_FAILED = "decode_failed"
    #: No decoder is installed, so media properties cannot be read at all.
    NO_DECODER = "no_decoder"
    #: The file's type could not be matched to any known modality.
    UNSUPPORTED = "unsupported"

    @classmethod
    def parse(cls, value: Any) -> InspectionStatus:
        """Resolve a member from a member or its string value."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(f"{value!r} is not a valid inspection status")


class VideoDecoderUnavailableError(RuntimeError):
    """Raised when a decoder is required but not installed."""


@dataclass(frozen=True, slots=True)
class FrameStatistics:
    """Summary statistics of one decoded frame.

    Computed once at decode time and stored, so quality filtering is a pure comparison
    against a policy rather than a second pass over the pixels. That keeps
    :mod:`datasets.preprocessing.quality` independent of any decoding library.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        mean_intensity: Mean of all channels over all pixels, 0-255.
        std_intensity: Standard deviation, 0-255. Near zero means a flat image.
        min_intensity: Darkest pixel value.
        max_intensity: Brightest pixel value.
        is_uniform: Whether every pixel is identical. Computed, not inferred from
            ``std_intensity``, so an exactly flat frame is distinguishable from a merely
            low-contrast one.
    """

    width: int
    height: int
    mean_intensity: float
    std_intensity: float
    min_intensity: int
    max_intensity: int
    is_uniform: bool = False

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def dynamic_range(self) -> int:
        """``max - min``. Zero for a flat frame."""
        return self.max_intensity - self.min_intensity

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "mean_intensity": round(self.mean_intensity, 4),
            "std_intensity": round(self.std_intensity, 4),
            "min_intensity": self.min_intensity,
            "max_intensity": self.max_intensity,
            "is_uniform": self.is_uniform,
        }


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One frame, with the index it occupied in the source video.

    Attributes:
        index: Zero-based position in the video. This is the *observed* index counted as
            frames were read, not a seek target, so it does not depend on a container's
            declared frame count being accurate.
        statistics: Pixel summary for quality filtering.
        image: The decoded pixels, or ``None`` when the caller asked not to keep them.
            OpenCV returns BGR-ordered ``uint8`` arrays.
    """

    index: int
    statistics: FrameStatistics
    image: Any = None


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    """Stream properties as reported by a decoder's probe.

    Every field is optional. A decoder that cannot determine a codec says so with
    ``None`` rather than guessing a container's default.
    """

    container_format: str | None = None
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    #: Frame count as *declared* by the container. Kept distinct from the observed count
    #: because the two disagree often enough to matter, and silently trusting the declared
    #: value is how two machines end up extracting different frames from one file.
    declared_frame_count: int | None = None
    frame_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "container_format": self.container_format,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "declared_frame_count": self.declared_frame_count,
            "frame_rate": self.frame_rate,
        }


@runtime_checkable
class VideoDecoder(Protocol):
    """What the pipeline needs from a video decoder.

    Deliberately stateless. Every method takes a path, so a decoder holds no open file
    handle between calls — which means an interrupted run cannot leave a capture open,
    and a decoder can be shared without locking.

    ``read_frames`` yields frames **in order, starting at zero**, and the caller is
    expected to consume it sequentially. Implementations must not seek: seeking is
    approximate in most implementations, and an approximate seek makes extracted frame
    indices depend on the decoder rather than on the file.
    """

    #: Short stable identifier recorded in inspection records.
    name: str

    def probe(self, path: Path) -> VideoStreamInfo:
        """Return stream properties, or raise if the file cannot be opened as media."""
        ...

    def read_frames(self, path: Path, *, keep_images: bool = False) -> Iterator[DecodedFrame]:
        """Yield every decodable frame in order.

        Args:
            path: Video to read.
            keep_images: Whether to retain pixel data on each frame. ``False`` still
                yields frames, with :attr:`DecodedFrame.image` set to ``None``, which is
                enough for counting and quality filtering without holding a whole video
                in memory.
        """
        ...


@dataclass(frozen=True, slots=True)
class ImageStreamInfo:
    """Stream properties as reported by an image decoder's probe.

    Every field is optional for the same reason as :class:`VideoStreamInfo`: a decoder
    that cannot determine a value reports ``None`` rather than a container default.
    """

    width: int | None = None
    height: int | None = None
    #: Number of stored channels: 1 for grayscale, 3 for colour, 4 when an alpha channel
    #: is present, and so on.
    channels: int | None = None
    #: Human-readable channel layout as reported by the decoder, such as ``"RGB"``,
    #: ``"RGBA"``, ``"BGR"`` or ``"L"``. Recorded verbatim because the ordering is a
    #: property of the decoder, and a consumer that assumes RGB when the decoder produced
    #: BGR corrupts every downstream pixel.
    color_mode: str | None = None
    #: Container/encoding inferred from the file extension, when known.
    format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "color_mode": self.color_mode,
            "format": self.format,
        }


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """Stream properties as reported by an audio decoder's probe.

    Every field is optional, on the same principle as the other modalities.
    """

    sample_rate: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    #: Total number of sample frames (one per channel), as declared by the container.
    sample_count: int | None = None
    #: Bit depth of one sample per channel, e.g. 16 for 16-bit PCM.
    sample_width_bits: int | None = None
    #: Encoding name when the decoder exposes it, such as ``"pcm"``.
    codec: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "sample_width_bits": self.sample_width_bits,
            "codec": self.codec,
        }


@runtime_checkable
class ImageDecoder(Protocol):
    """What the inspector needs from an image decoder.

    Stateless and probe-only: inspecting a still image does not require decoding pixels to
    a buffer, and keeping the pixel data out of the interface is what lets the same
    inspector run with no image library installed.
    """

    #: Short stable identifier recorded in inspection records.
    name: str

    def probe(self, path: Path) -> ImageStreamInfo:
        """Return image dimensions and channel layout.

        Raises:
            Exception: If the file cannot be opened as an image.
        """
        ...


@runtime_checkable
class AudioDecoder(Protocol):
    """What the inspector needs from an audio decoder.

    Stateless and probe-only, mirroring :class:`ImageDecoder`.
    """

    #: Short stable identifier recorded in inspection records.
    name: str

    def probe(self, path: Path) -> AudioStreamInfo:
        """Return audio sample rate, channel count and duration.

        Raises:
            Exception: If the file cannot be opened as audio.
        """
        ...


# ---------------------------------------------------------------------------
# Inspection record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MediaInspection:
    """What is known about one media file after inspection.

    Attributes:
        path: Path as given, not resolved, so records stay portable.
        modality: Which media type was inspected, or ``None`` when it could not be
            established (unreadable or unsupported paths).
        file_size: Size in bytes, or ``None`` if unreadable.
        sha256: Bare lowercase hex digest, or ``None`` if unreadable. Computed
            independently of decoding, so a file that cannot be decoded still has a
            verifiable identity. Same format as :attr:`datasets.schema.MediaSample.sha256`,
            so the two can be compared directly.
        container_format, codec, width, height: From the decoder's probe, else ``None``.
        duration_seconds: From the probe, else ``None``.
        frame_count: **Observed** count, filled in only by a counting pass.
        declared_frame_count: As claimed by the container, else ``None``.
        frame_rate: From the probe, else ``None``.
        channels, color_mode: Image channel count and layout, else ``None``.
        sample_rate, sample_count, sample_width_bits: Audio stream properties, else
            ``None``.
        status: Outcome of the inspection.
        error: Failure message, or ``None``. Names the file and the cause.
        warnings: Non-fatal observations, such as a declared frame count disagreeing with
            the observed one.
    """

    path: str
    #: Which of the three modalities this record describes. ``None`` only for a record
    #: built before a modality was established, such as an unreadable or unsupported path.
    modality: Modality | None = None
    file_size: int | None = None
    sha256: str | None = None
    container_format: str | None = None
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    frame_count: int | None = None
    declared_frame_count: int | None = None
    frame_rate: float | None = None
    #: Image channel count (1 grayscale, 3 colour, 4 with alpha). ``None`` for non-images.
    channels: int | None = None
    #: Image channel layout as reported by the decoder (``"RGB"``, ``"BGR"``, ``"L"``...).
    color_mode: str | None = None
    #: Audio sample rate in Hz. ``None`` for non-audio.
    sample_rate: int | None = None
    #: Audio sample frames (one per channel), as declared by the container.
    sample_count: int | None = None
    #: Audio bit depth per sample per channel, e.g. 16.
    sample_width_bits: int | None = None
    status: InspectionStatus = InspectionStatus.OK
    error: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    # --- derived ----------------------------------------------------------

    @property
    def is_usable(self) -> bool:
        """Whether the file was read and opened as media."""
        return self.status is InspectionStatus.OK

    @property
    def is_decodable(self) -> bool:
        """Whether frames were counted, which is stronger than being openable."""
        return self.frame_count is not None

    @property
    def resolution(self) -> str | None:
        """``"WxH"``, or ``None`` when either dimension is unavailable."""
        if self.width is None or self.height is None:
            return None
        return f"{self.width}x{self.height}"

    @property
    def unavailable_fields(self) -> tuple[str, ...]:
        """Names of the media fields that could not be determined.

        Reported rather than hidden, because "the file decoded but has no codec string"
        and "the file did not decode" need different responses and a record showing
        zeros for both does not distinguish them.

        The candidate list is modality-specific: a video has no sample rate and an image
        has no frame count, so listing those as "unavailable" for the wrong modality would
        be noise masquerading as a finding.
        """
        common: tuple[str, ...] = ("file_size", "sha256")
        if self.modality is Modality.IMAGE:
            specific: tuple[str, ...] = (
                "container_format",
                "width",
                "height",
                "channels",
                "color_mode",
            )
        elif self.modality is Modality.AUDIO:
            specific = (
                "codec",
                "duration_seconds",
                "sample_rate",
                "channels",
                "sample_count",
                "sample_width_bits",
            )
        else:
            specific = (
                "container_format",
                "codec",
                "width",
                "height",
                "duration_seconds",
                "frame_count",
                "declared_frame_count",
                "frame_rate",
            )
        return tuple(name for name in (*common, *specific) if getattr(self, name) is None)

    @property
    def frame_count_disagrees_with_declared(self) -> bool:
        """Whether the counted and declared frame counts differ.

        Common for variable-frame-rate files, and load-bearing: uniform sampling chooses
        its indices from this number, so a disagreement means the extracted frames are
        not the frames another machine would extract.
        """
        if self.frame_count is None or self.declared_frame_count is None:
            return False
        return self.frame_count != self.declared_frame_count

    def require_usable(self) -> MediaInspection:
        """Return self, or raise if inspection did not succeed.

        Raises:
            MediaInspectionError: With the recorded status and error message.
        """
        if self.is_usable:
            return self
        raise MediaInspectionError(
            f"cannot use {self.path}: {self.status.value}"
            + (f" - {self.error}" if self.error else "")
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "path": self.path,
            "status": self.status.value,
            "error": self.error,
            "modality": self.modality.value if self.modality is not None else None,
            "file_size": self.file_size,
            "sha256": self.sha256,
            "container_format": self.container_format,
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "declared_frame_count": self.declared_frame_count,
            "frame_rate": self.frame_rate,
            "channels": self.channels,
            "color_mode": self.color_mode,
            "sample_rate": self.sample_rate,
            "sample_count": self.sample_count,
            "sample_width_bits": self.sample_width_bits,
            "resolution": self.resolution,
            "unavailable_fields": list(self.unavailable_fields),
            "warnings": list(self.warnings),
        }


class MediaInspectionError(RuntimeError):
    """Raised when an inspection result is used despite having failed."""


class MediaInspector:
    """Inspects video, image, and audio files with injected decoders.

    Args:
        decoder: Decoder used to probe **video**. Never defaulted silently —
            :func:`default_decoder` is the explicit way to get the OpenCV one. ``None``
            is accepted so a caller can inspect only file identity and size.
        image_decoder: Decoder used to probe **images**. Keyword-only, default ``None``;
            :func:`default_image_decoder` returns the OpenCV one.
        audio_decoder: Decoder used to probe **audio**. Keyword-only, default ``None``;
            :func:`default_audio_decoder` returns the standard-library ``wave`` reader.
        compute_hashes: Whether to hash each file. Hashing is separate from decoding on
            purpose: identity is knowable for a file that will not open, and that is
            exactly the file you most want a stable identifier for.

    The same object inspects all three modalities, dispatching on the requested
    :class:`~datasets.schema.Modality` rather than on the file extension, so a caller that
    knows the modality gets an authoritative answer and a caller that does not can use
    :meth:`inspect_auto`.
    """

    def __init__(
        self,
        decoder: VideoDecoder | None,
        *,
        image_decoder: ImageDecoder | None = None,
        audio_decoder: AudioDecoder | None = None,
        compute_hashes: bool = True,
        hash_chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._decoder = decoder
        self._image_decoder = image_decoder
        self._audio_decoder = audio_decoder
        self._compute_hashes = compute_hashes
        if hash_chunk_size <= 0:
            raise ValueError(f"hash_chunk_size must be positive, got {hash_chunk_size}")
        self._hash_chunk_size = hash_chunk_size

    @property
    def decoder_name(self) -> str:
        """Name of the injected video decoder, or ``"none"``."""
        return self._decoder.name if self._decoder is not None else "none"

    @property
    def image_decoder_name(self) -> str:
        """Name of the injected image decoder, or ``"none"``."""
        return self._image_decoder.name if self._image_decoder is not None else "none"

    @property
    def audio_decoder_name(self) -> str:
        """Name of the injected audio decoder, or ``"none"``."""
        return self._audio_decoder.name if self._audio_decoder is not None else "none"

    def inspect(
        self,
        path: Path | str,
        *,
        modality: Modality | str = Modality.VIDEO,
        count_frames: bool = False,
    ) -> MediaInspection:
        """Inspect one file as the given modality.

        Args:
            path: File to inspect.
            modality: Which decoder to probe with. Defaults to ``VIDEO`` so existing
                callers keep their behaviour; pass ``IMAGE`` or ``AUDIO`` for the others.
            count_frames: Whether to decode every video frame to get an observed count.
                Costs a full pass, so it is opt-in: the declared count is usually enough to
                plan sampling, and the observed count is what you need when reproducibility
                is at stake. Ignored for images and audio.

        Returns:
            A record with either the properties that could be read, or a status and error
            explaining what failed. Never raises for a bad file.
        """
        return self._inspect(Path(path), Modality.parse(modality), count_frames)

    def inspect_image(self, path: Path | str) -> MediaInspection:
        """Inspect one file as a still image."""
        return self._inspect(Path(path), Modality.IMAGE, False)

    def inspect_audio(self, path: Path | str) -> MediaInspection:
        """Inspect one file as audio."""
        return self._inspect(Path(path), Modality.AUDIO, False)

    def inspect_auto(self, path: Path | str, *, count_frames: bool = False) -> MediaInspection:
        """Inspect one file, choosing the modality from its extension.

        Used when the caller has no manifest entry asserting a modality. A path whose
        extension matches none of :data:`DEFAULT_VIDEO_EXTENSIONS`,
        :data:`DEFAULT_IMAGE_EXTENSIONS` or :data:`DEFAULT_AUDIO_EXTENSIONS` is recorded as
        :attr:`InspectionStatus.UNSUPPORTED` — after its size and hash are still taken, so
        an unidentified file is not an unidentifiable one.
        """
        return self._inspect(Path(path), infer_modality(path), count_frames)

    @classmethod
    def from_project_config(
        cls,
        config: Any | None = None,
        *,
        compute_hashes: bool = True,
    ) -> MediaInspector:
        """Build an inspector with the bundled decoders and the project's hash settings.

        Reads ``data.hashing.chunk_size_bytes`` from ``configs/project.yaml`` so the chunk
        size used to digest media is the one recorded in the single source of truth rather
        than a value baked into this module. The config is imported lazily, matching
        :mod:`datasets.ingest`, so importing this module never pulls in ``ml`` or ``yaml``.

        Args:
            config: A loaded :class:`~ml.reproducibility.config.ProjectConfig`. When
                ``None``, ``configs/project.yaml`` is loaded on demand.
            compute_hashes: Whether to hash files at all.

        Raises:
            ValueError: If the configured chunk size is not a positive integer.
        """
        if config is None:
            from ml.reproducibility.config import load_project_config

            config = load_project_config()

        raw_chunk = config.get("data.hashing.chunk_size_bytes", DEFAULT_CHUNK_SIZE)
        if not isinstance(raw_chunk, int) or isinstance(raw_chunk, bool) or raw_chunk <= 0:
            raise ValueError(
                "data.hashing.chunk_size_bytes must be a positive integer, "
                f"got {raw_chunk!r}"
            )

        return cls(
            default_decoder(),
            image_decoder=default_image_decoder(),
            audio_decoder=default_audio_decoder(),
            compute_hashes=compute_hashes,
            hash_chunk_size=raw_chunk,
        )

    # --- internals --------------------------------------------------------

    def _inspect(
        self,
        target: Path,
        modality: Modality | None,
        count_frames: bool,
    ) -> MediaInspection:
        warnings: list[str] = []

        if not target.is_file():
            return MediaInspection(
                path=str(target),
                modality=modality,
                status=InspectionStatus.FILE_UNREADABLE,
                error=f"not a file: {target}",
            )

        try:
            file_size = target.stat().st_size
        except OSError as exc:
            return MediaInspection(
                path=str(target),
                modality=modality,
                status=InspectionStatus.FILE_UNREADABLE,
                error=f"could not stat {target}: {exc}",
            )

        sha256 = self._hash(target, warnings)

        if file_size == 0:
            # Short-circuited before decoding: no decoder can extract media from zero
            # bytes, and a distinct status keeps "empty" from looking like "corrupt".
            return MediaInspection(
                path=str(target),
                modality=modality,
                file_size=0,
                sha256=sha256,
                status=InspectionStatus.EMPTY,
                error=f"empty file: {target}",
                warnings=tuple(warnings),
            )

        if modality is None:
            return MediaInspection(
                path=str(target),
                file_size=file_size,
                sha256=sha256,
                status=InspectionStatus.UNSUPPORTED,
                error=(
                    f"could not determine the media type of {target} from its extension; "
                    "inspect it with an explicit modality"
                ),
                warnings=tuple(warnings),
            )

        if modality is Modality.IMAGE:
            return self._inspect_image(target, file_size, sha256, warnings)
        if modality is Modality.AUDIO:
            return self._inspect_audio(target, file_size, sha256, warnings)
        return self._inspect_video(target, file_size, sha256, warnings, count_frames)

    def _hash(self, target: Path, warnings: list[str]) -> str | None:
        if not self._compute_hashes:
            return None
        try:
            return calculate_file_sha256(target, chunk_size=self._hash_chunk_size)
        except OSError as exc:
            # Identity lost, but the file is still readable as media. Say which of the
            # two happened rather than reporting a decode failure.
            warnings.append(f"could not hash {target}: {exc}")
            return None

    def _inspect_video(
        self,
        target: Path,
        file_size: int,
        sha256: str | None,
        warnings: list[str],
        count_frames: bool,
    ) -> MediaInspection:
        decoder = self._decoder
        if decoder is None:
            return MediaInspection(
                path=str(target),
                modality=Modality.VIDEO,
                file_size=file_size,
                sha256=sha256,
                status=InspectionStatus.NO_DECODER,
                error=(
                    "no video decoder is configured, so media properties are unavailable. "
                    "Pass a VideoDecoder, or install opencv-python-headless."
                ),
                warnings=tuple(warnings),
            )

        try:
            info = decoder.probe(target)
        except Exception as exc:
            # Any decoder failure is a decode failure. Catching broadly is correct here:
            # a probe that raises an unexpected exception type must not be reported as a
            # successful inspection with empty fields.
            return MediaInspection(
                path=str(target),
                modality=Modality.VIDEO,
                file_size=file_size,
                sha256=sha256,
                status=InspectionStatus.DECODE_FAILED,
                error=f"decoder {decoder.name!r} could not open {target}: {exc}",
                warnings=tuple(warnings),
            )

        frame_count: int | None = None
        if count_frames:
            try:
                frame_count = sum(1 for _ in decoder.read_frames(target))
            except Exception as exc:
                warnings.append(
                    f"frame counting failed for {target}: {exc}; frame_count is "
                    "unavailable rather than guessed"
                )

        record = MediaInspection(
            path=str(target),
            modality=Modality.VIDEO,
            file_size=file_size,
            sha256=sha256,
            container_format=info.container_format,
            codec=info.codec,
            width=info.width,
            height=info.height,
            duration_seconds=info.duration_seconds,
            frame_count=frame_count,
            declared_frame_count=info.declared_frame_count,
            frame_rate=info.frame_rate,
            status=InspectionStatus.OK,
            warnings=tuple(warnings),
        )
        if record.frame_count_disagrees_with_declared:
            warnings_text = (
                f"counted {record.frame_count} frame(s) but the container declares "
                f"{record.declared_frame_count}. Uniform sampling picks indices from this "
                "number, so another decoder may extract different frames from the same "
                "file; pin the decoder version in requirements.txt to keep them identical."
            )
            # replace(), not **to_dict(): to_dict() carries derived keys such as
            # "resolution" that are not constructor parameters.
            return replace(record, warnings=(*warnings, warnings_text))
        return record

    def _inspect_image(
        self,
        target: Path,
        file_size: int,
        sha256: str | None,
        warnings: list[str],
    ) -> MediaInspection:
        decoder = self._image_decoder
        if decoder is None:
            return MediaInspection(
                path=str(target),
                modality=Modality.IMAGE,
                file_size=file_size,
                sha256=sha256,
                status=InspectionStatus.NO_DECODER,
                error=(
                    "no image decoder is configured, so image properties are unavailable. "
                    "Pass an ImageDecoder, or install opencv-python-headless."
                ),
                warnings=tuple(warnings),
            )

        try:
            info = decoder.probe(target)
        except Exception as exc:
            return MediaInspection(
                path=str(target),
                modality=Modality.IMAGE,
                file_size=file_size,
                sha256=sha256,
                status=InspectionStatus.DECODE_FAILED,
                error=f"decoder {decoder.name!r} could not open {target}: {exc}",
                warnings=tuple(warnings),
            )

        return MediaInspection(
            path=str(target),
            modality=Modality.IMAGE,
            file_size=file_size,
            sha256=sha256,
            container_format=info.format,
            width=info.width,
            height=info.height,
            channels=info.channels,
            color_mode=info.color_mode,
            status=InspectionStatus.OK,
            warnings=tuple(warnings),
        )

    def _inspect_audio(
        self,
        target: Path,
        file_size: int,
        sha256: str | None,
        warnings: list[str],
    ) -> MediaInspection:
        decoder = self._audio_decoder
        if decoder is None:
            return MediaInspection(
                path=str(target),
                modality=Modality.AUDIO,
                file_size=file_size,
                sha256=sha256,
                status=InspectionStatus.NO_DECODER,
                error=(
                    "no audio decoder is configured, so audio properties are unavailable. "
                    "Pass an AudioDecoder, or use the standard-library wave decoder."
                ),
                warnings=tuple(warnings),
            )

        try:
            info = decoder.probe(target)
        except Exception as exc:
            return MediaInspection(
                path=str(target),
                modality=Modality.AUDIO,
                file_size=file_size,
                sha256=sha256,
                status=InspectionStatus.DECODE_FAILED,
                error=f"decoder {decoder.name!r} could not open {target}: {exc}",
                warnings=tuple(warnings),
            )

        return MediaInspection(
            path=str(target),
            modality=Modality.AUDIO,
            file_size=file_size,
            sha256=sha256,
            codec=info.codec,
            duration_seconds=info.duration_seconds,
            channels=info.channels,
            sample_rate=info.sample_rate,
            sample_count=info.sample_count,
            sample_width_bits=info.sample_width_bits,
            status=InspectionStatus.OK,
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# OpenCV decoder
# ---------------------------------------------------------------------------


class OpenCVDecoder:
    """A :class:`VideoDecoder` backed by headless OpenCV.

    Chosen over alternatives for three reasons, in order of weight for this project:

    1. **Frame-accurate, seek-free extraction.** Frames are read sequentially and
       counted as they arrive, so ``DecodedFrame.index`` is a fact about the file rather
       than a seek target. Implementations that jump to ``CAP_PROP_POS_FRAMES`` land on
       an approximate keyframe-derived position, which would make extracted frame indices
       a property of the decoder.
    2. **No GUI dependency.** ``opencv-python-headless`` ships no Qt, which removes the
       well-known Windows conflict where a desktop Qt application and OpenCV load
       different Qt versions in one process. This pipeline is a batch job and never
       displays a frame.
    3. **One pinned wheel.** Reproducibility here means a version pin, and this is one
       pinned artifact rather than a decoder plus its own bundled FFmpeg ABI.

    The honest limitation: ``CAP_PROP_FRAME_COUNT`` comes from container metadata and is
    wrong often enough to matter, particularly for variable-frame-rate files. That is
    handled rather than ignored — the declared count and the observed count are recorded
    separately, their disagreement is reported as a warning, and
    :meth:`MediaInspector.inspect` can be asked to count frames so the observed value is
    available before sampling. :mod:`datasets.preprocessing.extraction` samples from the
    observed count whenever one was measured.

    For a project that later needs exact presentation timestamps or codecs OpenCV's
    bundled FFmpeg omits, swapping in PyAV means implementing :class:`VideoDecoder` and
    changing one construction site; nothing else in the pipeline is decoder-aware.
    """

    name = "opencv"

    def __init__(self) -> None:
        self._cv2: Any | None = None

    def _module(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise VideoDecoderUnavailableError(
                    "opencv-python-headless is not installed. Install it with "
                    "'pip install -r requirements.txt', or inject another VideoDecoder."
                ) from exc
            self._cv2 = cv2
        return self._cv2

    def probe(self, path: Path) -> VideoStreamInfo:
        """Probe stream properties.

        Raises:
            VideoDecoderUnavailableError: If OpenCV is not installed.
            RuntimeError: If the file cannot be opened as video.
        """
        cv2 = self._module()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"OpenCV could not open {path}")
            width = _as_positive_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = _as_positive_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = _as_positive_float(capture.get(cv2.CAP_PROP_FPS))
            declared_frame_count = _as_positive_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = _fourcc_to_text(_as_int(capture.get(cv2.CAP_PROP_FOURCC)))
            return VideoStreamInfo(
                # OpenCV reports no container, only the stream's FourCC. Recording the
                # FourCC as `container_format` would claim a codec name is a file format
                # and invite comparisons against "mp4" or "mov". It is filed under codec,
                # which is what it actually is, and the container is left unknown unless
                # the caller can supply it from the file extension.
                container_format=path.suffix.lstrip(".").lower() or None,
                codec=fourcc,
                width=width,
                height=height,
                # Duration is frame count over rate, not either value alone. OpenCV also
                # exposes CAP_PROP_POS_MSEC, but for variable-frame-rate files it reports
                # a seek position rather than a trustworthy total, so the ratio of two
                # properties this probe already reads is the defensible choice.
                duration_seconds=_duration_seconds(declared_frame_count, frame_rate),
                declared_frame_count=declared_frame_count,
                frame_rate=frame_rate,
            )
        finally:
            capture.release()

    def read_frames(self, path: Path, *, keep_images: bool = False) -> Iterator[DecodedFrame]:
        """Yield every decodable frame in order, counting as it goes.

        Raises:
            VideoDecoderUnavailableError: If OpenCV is not installed.
            RuntimeError: If the file cannot be opened as video.
        """
        cv2 = self._module()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not open {path}")
        try:
            index = 0
            while True:
                ok, image = capture.read()
                if not ok or image is None:
                    return
                statistics = compute_frame_statistics(image)
                yield DecodedFrame(
                    index=index,
                    statistics=statistics,
                    image=image if keep_images else None,
                )
                index += 1
        finally:
            # Closed on generator close as well as exhaustion, so an early ``break`` in
            # the consumer does not leave the file handle open on Windows.
            capture.release()


def default_decoder() -> VideoDecoder | None:
    """Return the OpenCV decoder, or ``None`` when OpenCV is not installed.

    Returning ``None`` rather than raising is deliberate: inspection is useful without a
    decoder (size, hash, and a status explaining why media properties are missing), and a
    pipeline that hard-fails on a missing optional dependency is harder to inspect than
    one that reports what it could not determine.
    """
    try:
        import cv2  # noqa: F401  (probe only)
    except ImportError:
        return None
    return OpenCVDecoder()


# ---------------------------------------------------------------------------
# OpenCV image decoder
# ---------------------------------------------------------------------------

#: Maps an OpenCV channel count to the layout name OpenCV itself uses for that ordering.
_OPENCV_COLOR_MODES: dict[int, str] = {1: "L", 2: "LA", 3: "BGR", 4: "BGRA"}


class OpenCVImageDecoder:
    """An :class:`ImageDecoder` backed by headless OpenCV.

    ``imread`` with ``IMREAD_UNCHANGED`` is used so the returned array keeps its true
    channel count: a grayscale PNG yields one channel, an RGBA PNG yields four, and a
    palette image is expanded to its stored form rather than silently flattened. That
    matters because downscaling or color-converting an image before a detector sees it is
    a preprocessing decision, not something an inspection probe should make.

    The recorded :attr:`ImageStreamInfo.color_mode` is the decoder's native ordering, so
    it reads ``"BGR"``/``"BGRA"`` for color images. Reporting ``"RGB"`` because most
    humans assume it would be exactly the kind of plausible-but-wrong default this module
    exists to prevent.

    The honest limitation: OpenCV decodes the whole image to read its dimensions. There is
    no header-only probe in the library, so inspecting a directory of very large images
    costs their full decode; a decoder that reads only headers can be injected instead.
    """

    name = "opencv"

    def __init__(self) -> None:
        self._cv2: Any | None = None

    def _module(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise VideoDecoderUnavailableError(
                    "opencv-python-headless is not installed. Install it with "
                    "'pip install -r requirements.txt', or inject another ImageDecoder."
                ) from exc
            self._cv2 = cv2
        return self._cv2

    def probe(self, path: Path) -> ImageStreamInfo:
        """Probe image dimensions and channel layout.

        Raises:
            VideoDecoderUnavailableError: If OpenCV is not installed.
            RuntimeError: If the file cannot be opened as an image.
        """
        cv2 = self._module()
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise RuntimeError(f"OpenCV could not open {path} as an image")
        shape = array.shape
        height = int(shape[0])
        width = int(shape[1])
        channels = 1 if array.ndim == 2 else int(shape[2])
        if height <= 0 or width <= 0:
            raise RuntimeError(f"OpenCV reported a zero-sized image for {path}")
        return ImageStreamInfo(
            width=width,
            height=height,
            channels=channels,
            color_mode=_OPENCV_COLOR_MODES.get(channels, f"{channels}-channel"),
            format=path.suffix.lstrip(".").lower() or None,
        )


def default_image_decoder() -> ImageDecoder | None:
    """Return the OpenCV image decoder, or ``None`` when OpenCV is not installed.

    Same contract as :func:`default_decoder`: ``None`` is a first-class answer that lets
    file identity still be recorded.
    """
    try:
        import cv2  # noqa: F401  (probe only)
    except ImportError:
        return None
    return OpenCVImageDecoder()


# ---------------------------------------------------------------------------
# Standard-library audio decoder
# ---------------------------------------------------------------------------


class WaveAudioDecoder:
    """An :class:`AudioDecoder` backed by the standard-library :mod:`wave` reader.

    Chosen as the default because it needs no dependency at all and is exact for PCM WAV,
    which is the interchange format the project can produce deterministically. It is
    deliberately *not* a general audio decoder: an MP3, FLAC or OGG fed to it raises, and
    the inspector records that as :attr:`InspectionStatus.DECODE_FAILED` rather than
    guessing. A caller who needs those containers injects a decoder such as one backed by
    ``soundfile`` or ``librosa``; nothing else in the module changes.
    """

    name = "wave"

    def probe(self, path: Path) -> AudioStreamInfo:
        """Probe sample rate, channel count and duration.

        Raises:
            RuntimeError: If the file is not a readable RIFF/WAV stream.
        """
        import wave

        try:
            with wave.open(str(path), "rb") as handle:
                channels = int(handle.getnchannels())
                sample_rate = int(handle.getframerate())
                frame_count = int(handle.getnframes())
                sample_width = int(handle.getsampwidth())
        except (wave.Error, EOFError) as exc:
            raise RuntimeError(f"not a readable WAV file: {exc}") from exc

        return AudioStreamInfo(
            sample_rate=sample_rate if sample_rate > 0 else None,
            channels=channels if channels > 0 else None,
            duration_seconds=(
                frame_count / sample_rate if sample_rate > 0 and frame_count >= 0 else None
            ),
            sample_count=frame_count if frame_count >= 0 else None,
            sample_width_bits=sample_width * 8 if sample_width > 0 else None,
            codec="pcm",
        )


def default_audio_decoder() -> AudioDecoder:
    """Return the standard-library WAV decoder.

    Unlike the video and image factories this never returns ``None``: :mod:`wave` ships
    with Python, so there is always a decoder available for at least one audio container.
    """
    return WaveAudioDecoder()


def is_video_candidate(path: Path | str, *, extensions: frozenset[str] | None = None) -> bool:
    """Whether a path has a recognised video extension.

    A cheap pre-filter for reporting. It is never used to reject a file silently: the
    caller records anything it skipped, so an unexpected format surfaces instead of
    vanishing.
    """
    allowed = DEFAULT_VIDEO_EXTENSIONS if extensions is None else extensions
    return Path(path).suffix.lower() in allowed


def is_image_candidate(path: Path | str, *, extensions: frozenset[str] | None = None) -> bool:
    """Whether a path has a recognised still-image extension. See :func:`is_video_candidate`."""
    allowed = DEFAULT_IMAGE_EXTENSIONS if extensions is None else extensions
    return Path(path).suffix.lower() in allowed


def is_audio_candidate(path: Path | str, *, extensions: frozenset[str] | None = None) -> bool:
    """Whether a path has a recognised audio extension. See :func:`is_video_candidate`."""
    allowed = DEFAULT_AUDIO_EXTENSIONS if extensions is None else extensions
    return Path(path).suffix.lower() in allowed


def infer_modality(path: Path | str) -> Modality | None:
    """Best-effort modality from a file extension, or ``None`` when unrecognised.

    Extension-based and therefore a *guess*, which is why it is only used by
    :meth:`MediaInspector.inspect_auto` and never to override a modality a manifest
    already asserts. Video is checked first so a container extension such as ``.ogg``
    (both audio and video in the wild) resolves the way the project's video corpora use it.
    """
    suffix = Path(path).suffix.lower()
    if suffix in DEFAULT_VIDEO_EXTENSIONS:
        return Modality.VIDEO
    if suffix in DEFAULT_IMAGE_EXTENSIONS:
        return Modality.IMAGE
    if suffix in DEFAULT_AUDIO_EXTENSIONS:
        return Modality.AUDIO
    return None


def compute_frame_statistics(image: Any) -> FrameStatistics:
    """Summarize a decoded frame.

    Works with any array-like of shape ``(H, W)`` or ``(H, W, C)``, so it does not
    require a particular decoding library. The uniformity test compares the extremes
    rather than relying on the standard deviation, which can be exactly zero for reasons
    a floating-point round-off would otherwise hide.

    Args:
        image: Pixel data.

    Raises:
        ValueError: If the array is empty or not at least two-dimensional.
    """
    array = _as_2d_array(image)
    height, width = array.shape[0], array.shape[1]
    if height == 0 or width == 0:
        raise ValueError("cannot compute statistics for a zero-sized frame")

    low = int(array.min())
    high = int(array.max())
    flat = array.reshape(-1).astype("float64")
    return FrameStatistics(
        width=int(width),
        height=int(height),
        mean_intensity=float(flat.mean()),
        std_intensity=float(flat.std()),
        min_intensity=low,
        max_intensity=high,
        is_uniform=low == high,
    )


# ---------------------------------------------------------------------------
# Small conversion helpers
# ---------------------------------------------------------------------------


def _as_int(value: Any) -> int | None:
    try:
        # round() with a single argument already returns an int.
        number = round(float(value))
    except (TypeError, ValueError):
        return None
    return number


def _as_positive_int(value: Any) -> int | None:
    number = _as_int(value)
    if number is None or number <= 0:
        return None
    return number


def _as_positive_float(value: Any) -> float | None:
    """A positive finite float, or ``None``.

    Anything that is not a finite number greater than zero becomes ``None``. Decoders
    report unknown properties as ``0.0`` and ``NaN``, and treating either as a real value
    would put fabricated zeros and NaNs into an inspection record that is supposed to be
    the record of what is actually known.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _duration_seconds(declared_frame_count: int | None, frame_rate: float | None) -> float | None:
    """Duration from a frame count and a rate, or ``None`` if either is unknown.

    Returns ``None`` rather than a partial answer when the frame rate is missing, because
    a duration without a rate is not a weaker fact but a different and wrong one.
    """
    if declared_frame_count is None or frame_rate is None:
        return None
    return declared_frame_count / frame_rate


def _fourcc_to_text(fourcc: int | None) -> str | None:
    """Decode a packed FOURCC integer, or ``None`` when it carries no information."""
    if not fourcc or fourcc <= 0:
        return None
    try:
        text = "".join(chr((fourcc >> (8 * offset)) & 0xFF) for offset in range(4))
    except ValueError:
        return None
    stripped = text.strip("\x00").strip()
    # OpenCV reports 0xFFFFFFFF and similar sentinels for "unknown"; a field of
    # non-printable characters is not a codec name and must not be recorded as one.
    if not stripped or not all(32 <= ord(char) < 127 for char in stripped):
        return None
    return stripped


def _as_2d_array(image: Any) -> Any:
    """Coerce decoded pixels to a 2-D-or-more numeric array."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is required by the project
        raise ValueError("numpy is required to compute frame statistics") from exc

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError(f"expected an image with at least 2 dimensions, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("expected a non-empty image")
    return array
