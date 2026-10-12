"""Deterministic frame selection and extraction.

The reproducibility contract
----------------------------
Given the same video file, the same frame count, and the same configuration, this module
must select the *same frame indices* every time, on every machine, in every Python build.
That is the property the rest of the pipeline is built on: if indices move, frame ids
move, and every frozen split and every published result pinned to it silently stops
describing the data.

Two decisions follow from that:

**Count frames, do not seek.** Indices are positions in a sequential decode, so
:class:`~datasets.preprocessing.inspection.DecodedFrame` numbering is a fact about the
decoded sequence. An extractor that seeks to ``CAP_PROP_POS_FRAMES`` gets a keyframe-derived
approximation, which would make the index a property of the decoder rather than the file.

**Prefer the observed frame count.** Uniform indices are chosen from a frame count, and
OpenCV's container-derived count is unreliable for variable-frame-rate files. So the
counted count wins whenever one was measured, and which count was used is recorded in the
:class:`SamplingPlan` rather than assumed.

Output format
-------------
Frames are written as **PNG by default**. Not for size, but because the task is detecting
low-level synthesis artifacts, and lossy JPEG re-encoding injects artifacts of its own.
A lossy step between decode and training would add a confound to exactly the signal the
benchmark exists to measure. Set :attr:`ExtractionConfig.image_format` to ``"jpeg"`` only
with that on the record.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from datasets.preprocessing.inspection import (
    DecodedFrame,
    FrameStatistics,
    MediaInspection,
    VideoDecoder,
)
from datasets.preprocessing.quality import (
    FrameQualityFilter,
    QualityPolicy,
    QualityVerdict,
)

__all__ = [
    "DEFAULT_EXTRACTION_CONFIG",
    "ExtractionConfig",
    "ExtractionError",
    "FrameCountSource",
    "FrameExtractionResult",
    "FrameIndexSelector",
    "FrameWriter",
    "NullFrameWriter",
    "OpenCVFrameWriter",
    "SamplingPlan",
    "SamplingStrategy",
    "VideoFrameExtractor",
    "available_strategies",
    "default_frame_writer",
    "every_n_indices",
    "plan_many",
    "plan_uniform_sampling",
    "register_strategy",
    "resolve_frame_count",
    "uniform_frame_indices",
]


class ExtractionError(RuntimeError):
    """Raised when extraction cannot proceed."""


class SamplingStrategy(str, Enum):
    """How frames are chosen from a video.

    Members exist before they are implemented. A declared-but-unimplemented strategy
    raises at *plan* time naming itself, rather than silently falling back to uniform
    sampling — a silent fallback would mean believing you sampled by timestamp when you
    sampled by index.
    """

    UNIFORM = "uniform"
    EVERY_N = "every_n"
    FPS_BASED = "fps_based"
    TIMESTAMP_BASED = "timestamp_based"

    @classmethod
    def parse(cls, value: Any) -> SamplingStrategy:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ExtractionError(
            f"{value!r} is not a valid sampling strategy; available: "
            f"{', '.join(s.value for s in available_strategies())}"
        )


class FrameCountSource(str, Enum):
    """Which frame count the sampling indices were derived from.

    Recorded in every :class:`SamplingPlan` because it is a reproducibility-relevant
    fact: a plan built from a declared count and one built from a counted count can
    select different indices for the same file.
    """

    #: Frames were actually decoded and counted. Reproducible.
    OBSERVED = "observed"
    #: The container's declared count, used unverified. Trustworthy only for
    #: constant-frame-rate files.
    DECLARED = "declared"
    #: Neither count was available, so no indices were selected.
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Deterministic index selection
# ---------------------------------------------------------------------------


#: A callable that turns a frame count into ascending, unique, in-range indices.
#:
#: The contract, which the type system cannot express, is: accept ``frame_count`` as the
#: first positional argument, accept ``frames_per_video`` (and any strategy-specific
#: keywords) as keywords, and return ascending unique indices all less than
#: ``frame_count``. A ``Protocol`` with ``__call__(..., **kwargs)`` cannot be satisfied by
#: a function with named parameters, so the alias is documented rather than enforced, and
#: :func:`plan_uniform_sampling` verifies the return value's shape at runtime.
FrameIndexSelector = Callable[..., "tuple[int, ...]"]


def uniform_frame_indices(frame_count: int, frames_per_video: int) -> tuple[int, ...]:
    """Choose evenly spaced indices spanning the whole video, endpoints included.

    The formula is ``round(i * (frame_count - 1) / (frames_per_video - 1))`` for
    ``i`` in ``[0, frames_per_video)``, so the first and last frames are always
    included. Covering the endpoints matters: a detector that only ever sees the middle
    of a clip never sees the transitions that distinguish many forgeries.

    Using ``round`` rather than a stride keeps the result strictly increasing whenever
    ``frames_per_video <= frame_count``, because consecutive unrounded values differ by
    ``(frame_count - 1) / (frames_per_video - 1) >= 1``.

    Args:
        frame_count: Number of frames available.
        frames_per_video: Number wanted.

    Returns:
        Ascending unique indices. Empty when ``frame_count`` is zero. Shorter than
        ``frames_per_video`` when the video has fewer frames — the caller is told via
        :attr:`SamplingPlan.is_short`, because quietly returning fewer frames than asked
        for is exactly the kind of thing that skews a corpus.

    Raises:
        ExtractionError: If either argument is negative or ``frames_per_video`` is zero.
    """
    if frame_count < 0:
        raise ExtractionError(f"frame_count must be non-negative, got {frame_count}")
    if frames_per_video < 1:
        raise ExtractionError(f"frames_per_video must be at least 1, got {frames_per_video}")
    if frame_count == 0:
        return ()
    if frames_per_video >= frame_count:
        return tuple(range(frame_count))
    if frames_per_video == 1:
        # Only one index can be chosen. The first frame is chosen over a midpoint
        # because a midpoint index would depend on frame_count in a way that makes the
        # result harder to reason about when a video is re-encoded one frame longer.
        return (0,)

    span = frame_count - 1
    steps = frames_per_video - 1
    return tuple(round(index * span / steps) for index in range(frames_per_video))


def every_n_indices(frame_count: int, interval: int = 10) -> tuple[int, ...]:
    """Choose every ``interval``-th frame, starting at the first.

    Provided mainly to exercise the strategy registry: unlike uniform sampling it
    covers only the head of the video when ``interval`` is large, so it is unsuitable
    for whole-video coverage and is not the default.

    Raises:
        ExtractionError: On a negative count or a non-positive interval.
    """
    if frame_count < 0:
        raise ExtractionError(f"frame_count must be non-negative, got {frame_count}")
    if interval < 1:
        raise ExtractionError(f"interval must be at least 1, got {interval}")
    return tuple(range(0, frame_count, interval))


_STRATEGIES: dict[SamplingStrategy, FrameIndexSelector] = {
    SamplingStrategy.UNIFORM: uniform_frame_indices,
    SamplingStrategy.EVERY_N: every_n_indices,
}


def register_strategy(strategy: SamplingStrategy, selector: FrameIndexSelector) -> None:
    """Register an index selector for a strategy.

    The extension point for a dataset that needs its own selection rule. Re-registering
    a strategy is allowed and replaces the previous selector, so a caller that needs
    different uniform behaviour can say so explicitly rather than monkey-patching a
    private table.
    """
    if not callable(selector):
        raise ExtractionError(f"selector for {strategy.value} must be callable")
    _STRATEGIES[strategy] = selector


def available_strategies() -> tuple[SamplingStrategy, ...]:
    """Strategies that are implemented, in declaration order."""
    return tuple(_STRATEGIES)


def resolve_frame_count(inspection: MediaInspection) -> tuple[int | None, FrameCountSource]:
    """Pick the frame count to sample from, preferring the measured one.

    Returns:
        ``(frame_count, source)``. ``(None, UNAVAILABLE)`` when neither count is known.
    """
    if inspection.frame_count is not None:
        return inspection.frame_count, FrameCountSource.OBSERVED
    if inspection.declared_frame_count is not None:
        return inspection.declared_frame_count, FrameCountSource.DECLARED
    return None, FrameCountSource.UNAVAILABLE


def _validate_indices(
    indices: tuple[int, ...], frame_count: int, strategy: SamplingStrategy
) -> None:
    """Assert a selector returned a usable set of frame indices.

    Checks ascending order, no duplicates, and that every index is within the video. Each
    would otherwise fail quietly: duplicates collapse into one frame id, and an
    out-of-range index never matches a decoded frame so the frame is simply missing.

    Raises:
        ExtractionError: Naming the strategy and the offending value.
    """
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ExtractionError(
                f"selector for {strategy.value!r} returned {index!r}, which is not a frame "
                "index; indices must be integers"
            )
    for previous, current in itertools.pairwise(indices):
        if current <= previous:
            raise ExtractionError(
                f"selector for {strategy.value!r} returned indices that are not strictly "
                f"ascending: ...{previous}, {current}..."
            )
    for index in indices:
        if not 0 <= index < frame_count:
            raise ExtractionError(
                f"selector for {strategy.value!r} returned index {index}, which is outside "
                f"the video's {frame_count} frame(s)"
            )


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """Which frames will be taken from one video, and why.

    Attributes:
        source_path: Video the plan applies to.
        strategy: Selection rule used.
        frame_count: The count indices were derived from.
        frame_count_source: Whether that count was measured or declared.
        requested: Number of frames asked for.
        indices: Selected indices, ascending.
        notes: Deviations the caller must know about — a video shorter than requested, or
            sampling from an unverified declared count.
    """

    source_path: str
    strategy: SamplingStrategy
    frame_count: int
    frame_count_source: FrameCountSource
    requested: int
    indices: tuple[int, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def selected_count(self) -> int:
        return len(self.indices)

    @property
    def is_short(self) -> bool:
        """Whether fewer frames were available than requested."""
        return self.selected_count < self.requested

    @property
    def is_empty(self) -> bool:
        return not self.indices

    @property
    def covers_whole_video(self) -> bool:
        """Whether the selection reaches the first and last frame.

        Meaningless when the video has a single frame, hence the explicit zero check.
        """
        if not self.indices:
            return False
        return self.indices[0] == 0 and self.indices[-1] == self.frame_count - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "strategy": self.strategy.value,
            "frame_count": self.frame_count,
            "frame_count_source": self.frame_count_source.value,
            "requested": self.requested,
            "selected_count": self.selected_count,
            "indices": list(self.indices),
            "is_short": self.is_short,
            "covers_whole_video": self.covers_whole_video,
            "notes": list(self.notes),
        }


def plan_uniform_sampling(
    inspection: MediaInspection,
    frames_per_video: int,
    *,
    strategy: SamplingStrategy | str = SamplingStrategy.UNIFORM,
    **strategy_kwargs: int,
) -> SamplingPlan:
    """Build a :class:`SamplingPlan` from an inspection.

    Sampling from an unverified declared count emits a note. It is allowed — for
    constant-frame-rate video it is usually right, and refusing it would mean no frames
    at all until someone runs a full counting pass — but it is never silent.

    Raises:
        ExtractionError: If the strategy is unimplemented, or no frame count is available.
    """
    resolved = SamplingStrategy.parse(strategy)
    selector = _STRATEGIES.get(resolved)
    if selector is None:
        raise ExtractionError(
            f"sampling strategy {resolved.value!r} is declared but not implemented; "
            f"available: {', '.join(s.value for s in available_strategies())}"
        )

    frame_count, source = resolve_frame_count(inspection)
    if frame_count is None:
        raise ExtractionError(
            f"cannot plan sampling for {inspection.path}: neither an observed nor a "
            "declared frame count is available. Inspect with count_frames=True, or fix "
            "the file."
        )

    notes: list[str] = []
    if source is FrameCountSource.DECLARED:
        notes.append(
            f"sampled from the container's declared frame count ({frame_count}) without "
            "verifying it by decoding. Re-inspect with count_frames=True to make the "
            "selection reproducible across decoders."
        )

    try:
        indices = selector(frame_count, frames_per_video=frames_per_video, **strategy_kwargs)
    except TypeError:
        # A custom selector may not accept every strategy-specific keyword. Retry with
        # only what it declares, so registering a two-argument selector works.
        indices = selector(frame_count)

    # A registered selector is arbitrary code, so its output is checked rather than
    # trusted. Out-of-range or unsorted indices would produce frame ids for frames that
    # do not exist, or would silently extract the wrong ones.
    _validate_indices(indices, frame_count, resolved)

    if len(indices) < frames_per_video:
        notes.append(
            f"requested {frames_per_video} frame(s) but the video has only {frame_count}; "
            f"selected {len(indices)}"
        )

    return SamplingPlan(
        source_path=inspection.path,
        strategy=resolved,
        frame_count=frame_count,
        frame_count_source=source,
        requested=frames_per_video,
        indices=tuple(indices),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Frame writing
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameWriter(Protocol):
    """Writes decoded pixels to a file."""

    def write(self, image: Any, path: Path) -> None:
        """Write ``image`` to ``path``, creating parent directories."""
        ...

    @property
    def format(self) -> str:
        """Format this writer produces, e.g. ``"png"``."""
        ...


class NullFrameWriter:
    """Counts frames and writes nothing.

    For ``--dry-run``: the selection, quality verdicts, and frame counts are all
    exercised, so a dry run verifies the plan without writing images. Callers must treat
    its output as unpersisted.
    """

    @property
    def format(self) -> str:
        return "none"

    def write(self, image: Any, path: Path) -> None:
        """Do nothing."""


class OpenCVFrameWriter:
    """Writes frames with OpenCV's image encoder.

    Args:
        image_format: Extension without the dot, e.g. ``"png"``.
        jpeg_quality: Quality for lossy formats, 1-100. Ignored for PNG.
    """

    def __init__(self, image_format: str = "png", *, jpeg_quality: int = 95) -> None:
        normalized = image_format.strip().lower().lstrip(".")
        if not normalized.isalnum():
            raise ExtractionError(f"invalid image format: {image_format!r}")
        if not 1 <= jpeg_quality <= 100:
            raise ExtractionError(f"jpeg_quality must be within 1-100, got {jpeg_quality}")
        self._format = normalized
        self._quality = jpeg_quality

    @property
    def format(self) -> str:
        return self._format

    def write(self, image: Any, path: Path) -> None:
        """Write one frame.

        Raises:
            ExtractionError: If OpenCV is missing or the encoder fails.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ExtractionError(
                "opencv-python-headless is not installed, so frames cannot be written. "
                "Install it, or use NullFrameWriter for a dry run."
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        params: list[int] = []
        if self._format in {"jpg", "jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        if not cv2.imwrite(str(path), image, params):
            raise ExtractionError(f"failed to encode frame to {path}")


def default_frame_writer(image_format: str = "png") -> FrameWriter | None:
    """Return an OpenCV writer, or ``None`` when OpenCV is unavailable."""
    try:
        import cv2  # noqa: F401  (probe only)
    except ImportError:
        return None
    return OpenCVFrameWriter(image_format)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """How many frames to take per video, and how to write them.

    Attributes:
        frames_per_video: Number of frames to sample from each video.
        strategy: Selection rule.
        strategy_kwargs: Extra keywords for the selector, e.g. ``{"interval": 5}`` for
            :attr:`SamplingStrategy.EVERY_N`.
        image_format: Output image format.
        jpeg_quality: Quality for lossy output formats.
        quality_policy: Frame-quality thresholds. ``None`` uses the conservative default.
    """

    frames_per_video: int = 16
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    strategy_kwargs: Mapping[str, int] = field(default_factory=dict)
    image_format: str = "png"
    jpeg_quality: int = 95
    quality_policy: QualityPolicy | None = None

    def __post_init__(self) -> None:
        if self.frames_per_video < 1:
            raise ExtractionError(
                f"frames_per_video must be at least 1, got {self.frames_per_video}"
            )
        object.__setattr__(self, "strategy", SamplingStrategy.parse(self.strategy))
        object.__setattr__(self, "strategy_kwargs", dict(self.strategy_kwargs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_per_video": self.frames_per_video,
            "strategy": self.strategy.value,
            "strategy_kwargs": dict(self.strategy_kwargs),
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality,
            "quality_policy": (self.quality_policy or QualityPolicy()).to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ExtractionConfig:
        """Build a config from a mapping, ignoring unknown keys."""
        raw_policy = raw.get("quality_policy")
        policy = QualityPolicy.from_dict(raw_policy) if isinstance(raw_policy, Mapping) else None
        return cls(
            frames_per_video=int(raw.get("frames_per_video", 16)),
            strategy=SamplingStrategy.parse(raw.get("strategy", SamplingStrategy.UNIFORM)),
            strategy_kwargs={
                str(key): int(value) for key, value in (raw.get("strategy_kwargs") or {}).items()
            },
            image_format=str(raw.get("image_format", "png")),
            jpeg_quality=int(raw.get("jpeg_quality", 95)),
            quality_policy=policy,
        )


DEFAULT_EXTRACTION_CONFIG = ExtractionConfig()


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    """One frame that survived selection and quality filtering.

    Attributes:
        index: Position in the source video.
        statistics: Pixel summary used for the quality verdict.
        relative_path: Output path relative to the extraction root, as posix text. Kept
            relative so a manifest stays portable across machines.
        verdict: The quality decision and its reasons.
    """

    index: int
    statistics: FrameStatistics
    relative_path: str
    verdict: QualityVerdict

    @property
    def is_kept(self) -> bool:
        return self.verdict.accepted


@dataclass(frozen=True, slots=True)
class FrameExtractionResult:
    """Outcome of extracting frames from one video.

    Rejected frames are kept in the result rather than discarded, so a run can report
    why a frame was dropped and by how much the corpus shrank.

    Attributes:
        source_path: Video that was processed.
        plan: The sampling plan actually used.
        frames: Selected frames, kept and rejected alike.
        warnings: Non-fatal problems, including every plan note.
        written: Whether frame images were persisted.
    """

    source_path: str
    plan: SamplingPlan
    frames: tuple[ExtractedFrame, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    written: bool = False

    @property
    def kept(self) -> tuple[ExtractedFrame, ...]:
        return tuple(frame for frame in self.frames if frame.is_kept)

    @property
    def rejected(self) -> tuple[ExtractedFrame, ...]:
        return tuple(frame for frame in self.frames if not frame.is_kept)

    @property
    def kept_indices(self) -> tuple[int, ...]:
        return tuple(frame.index for frame in self.kept)

    @property
    def verdicts(self) -> tuple[QualityVerdict, ...]:
        return tuple(frame.verdict for frame in self.frames)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "plan": self.plan.to_dict(),
            "selected": len(self.frames),
            "kept": len(self.kept),
            "rejected": len(self.rejected),
            "written": self.written,
            "frames": [
                {
                    "index": frame.index,
                    "relative_path": frame.relative_path,
                    "accepted": frame.is_kept,
                    "reasons": list(frame.verdict.reason_values),
                    "statistics": frame.statistics.to_dict(),
                }
                for frame in self.frames
            ],
            "warnings": list(self.warnings),
        }


class VideoFrameExtractor:
    """Selects and extracts frames from one video at a time.

    Sequential single pass: frames are decoded in order, and only the planned indices are
    written. Frames the decoder skips mid-stream simply never arrive, so a corrupt file
    yields fewer frames than planned — surfaced through the plan's notes and the quality
    summary rather than by silently padding.

    Args:
        decoder: Decoder used to read frames.
        writer: Where images go. :class:`NullFrameWriter` gives a dry run.
        config: Sampling and output settings.
    """

    def __init__(
        self,
        decoder: VideoDecoder,
        writer: FrameWriter,
        config: ExtractionConfig | None = None,
    ) -> None:
        self._decoder = decoder
        self._writer = writer
        self.config = DEFAULT_EXTRACTION_CONFIG if config is None else config
        self._quality = FrameQualityFilter(self.config.quality_policy)

    @property
    def writer(self) -> FrameWriter:
        return self._writer

    @property
    def writes_images(self) -> bool:
        return not isinstance(self._writer, NullFrameWriter)

    @property
    def quality_policy(self) -> QualityPolicy:
        """The policy actually applied, with defaults resolved.

        Exposed so reporting summarises verdicts under the same thresholds that produced
        them. Reading the defaults again at report time would silently mislabel a run that
        used stricter thresholds.
        """
        return self._quality.policy

    def frame_relative_path(self, source_video_id: str, index: int) -> str:
        """Deterministic output path for one frame, relative to the extraction root.

        Zero-padded to six digits so lexical order matches numeric order — a directory
        listing of ``frame_000100.png`` sorts before ``frame_000099.png`` without it,
        which is a quiet way to make a review pass miss frames.

        Raises:
            ExtractionError: If the index is negative.
        """
        if index < 0:
            raise ExtractionError(f"frame index must be non-negative, got {index}")
        extension = "png" if self._writer.format == "none" else self._writer.format
        return f"{source_video_id}/frame_{index:06d}.{extension}"

    def extract(
        self,
        inspection: MediaInspection,
        *,
        source_video_id: str,
        output_root: Path | None = None,
    ) -> FrameExtractionResult:
        """Extract the planned frames from one video.

        Args:
            inspection: A usable inspection of the video.
            source_video_id: Stable parent-video identity, used for output paths.
            output_root: Directory images are written under. Required unless the writer
                is a :class:`NullFrameWriter`.

        Returns:
            Kept and rejected frames together with the plan used.

        Raises:
            ExtractionError: If the inspection is unusable, the plan cannot be built, or
                writing was requested without an output root.
        """
        usable = inspection.require_usable()
        if self.writes_images and output_root is None:
            raise ExtractionError(
                f"output_root is required to write frames for {usable.path}; pass one, "
                "or use NullFrameWriter for a dry run"
            )

        plan = plan_uniform_sampling(
            usable,
            self.config.frames_per_video,
            strategy=self.config.strategy,
            **self.config.strategy_kwargs,
        )
        warnings = list(plan.notes)
        wanted = set(plan.indices)
        results: list[ExtractedFrame] = []

        if wanted:
            seen: set[int] = set()
            for frame in self._decode(usable.path):
                if frame.index not in wanted:
                    continue
                seen.add(frame.index)
                results.append(self._handle_frame(frame, source_video_id, output_root))
                if len(seen) == len(wanted):
                    break

            missing = sorted(wanted - seen)
            if missing:
                warnings.append(
                    f"{len(missing)} planned frame(s) were never decoded from {usable.path} "
                    f"(first missing index: {missing[0]}). The decoder stopped early, which "
                    "usually means a truncated or partially corrupt file."
                )
                results.sort(key=lambda item: item.index)

        return FrameExtractionResult(
            source_path=usable.path,
            plan=plan,
            frames=tuple(results),
            warnings=tuple(warnings),
            written=self.writes_images,
        )

    def _decode(self, path: str) -> Iterator[DecodedFrame]:
        """Decode sequentially, keeping pixels only when they will be written."""
        yield from self._decoder.read_frames(Path(path), keep_images=self.writes_images)

    def _handle_frame(
        self,
        frame: DecodedFrame,
        source_video_id: str,
        output_root: Path | None,
    ) -> ExtractedFrame:
        relative = self.frame_relative_path(source_video_id, frame.index)
        verdict = self._quality.evaluate(frame.statistics)
        if verdict.accepted and self.writes_images and output_root is not None:
            # Rejected frames are not written: their image bytes are not needed to audit
            # the decision, which the recorded statistics already support.
            self._writer.write(frame.image, output_root / relative)
        return ExtractedFrame(
            index=frame.index,
            statistics=frame.statistics,
            relative_path=relative,
            verdict=verdict,
        )


def plan_many(
    inspections: Iterable[MediaInspection],
    frames_per_video: int,
    **kwargs: Any,
) -> tuple[SamplingPlan, ...]:
    """Plan sampling for several videos, skipping ones that cannot be planned.

    Returns plans only for videos that yielded one, so a single undecodable file does not
    abort a whole batch. The caller compares counts against the input to notice.
    """
    plans: list[SamplingPlan] = []
    for inspection in inspections:
        try:
            plans.append(plan_uniform_sampling(inspection, frames_per_video, **kwargs))
        except ExtractionError:
            continue
    return tuple(plans)
