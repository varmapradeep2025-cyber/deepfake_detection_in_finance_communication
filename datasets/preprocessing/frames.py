"""Frame identity and the frame manifest record.

Why frame identity is derived, not generated
--------------------------------------------
Frame ids are computed from the parent video's identity and the frame's index. They are
never UUIDs, timestamps, or counter values. The requirement is not tidiness — it is that
re-running preprocessing over unchanged inputs produces an unchanged manifest, because the
frame ids feed the frozen split and a single moved id silently changes what the test set
is while every published number still claims to describe it.

Parent linkage is mandatory
--------------------------
A frame record without a resolvable parent video is refused by
:meth:`FrameSample.require_parent_linkage`. Without a parent identity there is no way to
group frames by source video, and grouping is the only thing standing between the model and
near-duplicate frames on both sides of the split. So it is a construction-time invariant,
not a convention.

Cross-manipulation grouping is the interesting case
--------------------------------------------------
In FaceForensics++ the genuine clip and its forgeries all descend from one recording, laid
out as ``<manipulation>/<recording>/<name>.mp4``. Grouping by file path therefore puts the
genuine clip in train and its own forgery in test — leakage through a shared recording,
and the most likely way a split on this dataset goes wrong without anyone noticing.

Two identities, because one is not enough
-----------------------------------------
That layout forces a distinction that is easy to miss:

``source_video_id``
    The identity of **one file**, e.g. ``ffpp__DeepFakes__000__001``. Unique, and what a
    frame id is built from.
``source_recording_id``
    The identity of the **underlying recording**, e.g. ``ffpp__000__001``. Several
    ``source_video_id`` values map to one of these, and it is what frames are grouped by.

Collapsing them into a single field does not work. Grouping by the file identity leaks,
because a clip and its forgeries become different groups. Grouping by the recording
identity *and* deriving frame ids from it collides, because the genuine clip and its
forgery both produce a frame 0 and both want the id ``ffpp__000__001__f000000`` — and a
duplicate ``sample_id`` is rejected outright by
:func:`datasets.schema.validate_samples`. Keeping both means each frame is uniquely
identifiable *and* every manipulation of a recording shares a group.

:class:`SourceIdentityLayout` chooses the recording identity. It errs toward
over-linking, deliberately: if the layout cannot tell two recordings apart, merging their
groups only costs a little test-set size, whereas splitting them costs the validity of
every number reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from datasets.schema import (
    ManifestValidationError,
    ManipulationType,
    MediaSample,
    Modality,
    SampleLabel,
    SplitName,
)

__all__ = [
    "FrameManifestError",
    "FrameSample",
    "SourceIdentityLayout",
    "derive_frame_id",
    "derive_source_video_id",
    "frame_id_separator",
    "normalise_relative_path",
]


class FrameManifestError(ManifestValidationError):
    """Raised when a frame record is missing required parent linkage."""


#: Separates the parts of a derived frame id. Double underscore keeps the composite id
#: unambiguous when a dataset name or filename contains a single underscore, which is
#: common enough that single-underscore joining is not worth the ambiguity.
_ID_SEPARATOR = "__"


def frame_id_separator() -> str:
    """The separator used inside derived ids, exposed so tests can assert on it."""
    return _ID_SEPARATOR


class SourceIdentityLayout(str, Enum):
    """How a manifest path maps to a *recording* identity.

    Attributes:
        FULL_RELATIVE_PATH: The whole relative path, extension removed. One recording per
            file, so every manipulation of a source recording is its own group. This is the
            default because it cannot merge unrelated recordings — but on FaceForensics++
            it does *not* prevent a clip and its forgeries from landing in different splits.
        DROP_FIRST_SEGMENT: The path with its first component removed, which is the
            manipulation directory in the FaceForensics++ layout. Groups one recording
            together across every manipulation of it.
    """

    FULL_RELATIVE_PATH = "full_relative_path"
    DROP_FIRST_SEGMENT = "drop_first_segment"

    @classmethod
    def parse(cls, value: Any) -> SourceIdentityLayout:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise FrameManifestError(
            f"{value!r} is not a valid source identity layout; expected one of "
            f"{', '.join(member.value for member in cls)}"
        )


def normalise_relative_path(relative_path: Path | str) -> PurePosixPath:
    """Normalise a manifest path to a comparable relative POSIX path.

    Backslashes become forward slashes so a manifest written on Windows resolves to the
    same identity as one written on Linux. Extension is preserved here; identity
    derivation drops it separately.

    Raises:
        FrameManifestError: If the path is empty, absolute, or escapes its root. A path
            containing ``..`` cannot be turned into a stable identity, and silently
            normalising it away would let two different files claim one identity.
    """
    raw = str(relative_path).strip().replace("\\", "/")
    if not raw:
        raise FrameManifestError("relative path must not be empty")
    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise FrameManifestError(
            f"expected a relative path, got absolute {relative_path!r}; source identity "
            "must be independent of where the dataset happens to be mounted"
        )
    if ".." in pure.parts:
        raise FrameManifestError(f"relative path must not contain '..': {relative_path!r}")
    return pure


def derive_source_video_id(dataset_name: str, relative_path: Path | str) -> str:
    """Compute a stable identity for **one video file** from a dataset name and path.

    Always includes the full relative path, because frame ids must be unique across every
    file in the dataset. The extension is excluded, so re-encoding ``clip.mp4`` to
    ``clip.mkv`` keeps the same identity and its frames stay the same frames rather than
    appearing to come from a second video.

    For the identity used to *group* frames, see :func:`derive_source_recording_id`.

    Args:
        dataset_name: Dataset the video belongs to.
        relative_path: The video's path within the dataset.

    Returns:
        An id of the form ``<dataset>__<path parts>``, unique per file.

    Raises:
        FrameManifestError: If the dataset name is empty, or the path is unusable.

    Examples:
        >>> derive_source_video_id("ffpp", "original/000/001.mp4")
        'ffpp__original__000__001'
        >>> derive_source_video_id("ffpp", "DeepFakes/000/001.mp4")
        'ffpp__DeepFakes__000__001'
    """
    name = str(dataset_name).strip()
    if not name:
        raise FrameManifestError("dataset_name must not be empty")

    pure = normalise_relative_path(relative_path)
    stem = pure.stem
    if not stem:
        raise FrameManifestError(
            f"cannot derive a source identity from {relative_path!r}: no filename stem"
        )

    parts = [name, *pure.parent.parts, stem]
    return _ID_SEPARATOR.join(part.strip() for part in parts if part.strip())


def derive_source_recording_id(
    dataset_name: str,
    relative_path: Path | str,
    *,
    layout: SourceIdentityLayout | str = SourceIdentityLayout.FULL_RELATIVE_PATH,
) -> str:
    """Compute the identity of the **source recording** a video descends from.

    This is the grouping key. Several videos map to one recording id — on FaceForensics++
    the genuine clip and every forgery of it — which is exactly what keeps them in one
    split.

    Args:
        dataset_name: Dataset the video belongs to.
        relative_path: The video's path within the dataset.
        layout: Which parts of the path identify the recording rather than the
            manipulation of it.

    Returns:
        An id of the form ``<dataset>__<recording parts>``.

    Raises:
        FrameManifestError: If the dataset name is empty, or the path cannot satisfy the
            requested layout.

    Examples:
        With the default layout this matches
        :func:`derive_source_video_id` exactly:

        >>> derive_source_recording_id("ffpp", "original/000/001.mp4")
        'ffpp__original__000__001'

        Dropping the first segment collapses every manipulation of recording ``000``:

        >>> derive_source_recording_id("ffpp", "DeepFakes/000/001.mp4", layout="drop_first_segment")
        'ffpp__000__001'
    """
    name = str(dataset_name).strip()
    if not name:
        raise FrameManifestError("dataset_name must not be empty")

    # Parse first. Comparing the raw argument with `is` would silently skip the layout
    # branch whenever a plain string was passed, since a str-Enum member and its value are
    # equal but not identical.
    resolved_layout = SourceIdentityLayout.parse(layout)

    pure = normalise_relative_path(relative_path)
    stem = pure.stem
    if not stem:
        raise FrameManifestError(
            f"cannot derive a recording identity from {relative_path!r}: no filename stem"
        )

    parents = [part for part in pure.parent.parts if part not in {".", "/"}]
    if resolved_layout is SourceIdentityLayout.DROP_FIRST_SEGMENT:
        if not parents:
            raise FrameManifestError(
                f"layout {resolved_layout.value} needs at least one directory above the "
                f"filename, but {relative_path!r} has none"
            )
        parents = parents[1:]

    parts = [name, *parents, stem]
    return _ID_SEPARATOR.join(part.strip() for part in parts if part.strip())


def derive_frame_id(source_video_id: str, frame_index: int) -> str:
    """Compute a frame's id from its parent video identity and index.

    Args:
        source_video_id: Identity from :func:`derive_source_video_id`.
        frame_index: Zero-based position in the source video.

    Returns:
        ``<source_video_id>__f<index>`` with a six-digit zero-padded index.

    Raises:
        FrameManifestError: If the parent id is empty or the index is negative.
    """
    parent = str(source_video_id).strip()
    if not parent:
        raise FrameManifestError("source_video_id must not be empty")
    if frame_index < 0:
        raise FrameManifestError(f"frame_index must be non-negative, got {frame_index}")
    return f"{parent}{_ID_SEPARATOR}f{frame_index:06d}"


# ---------------------------------------------------------------------------
# Frame record
# ---------------------------------------------------------------------------

#: Metadata key under which all frame-specific fields live. Namespacing keeps them out of
#: the core :class:`MediaSample` field set, so the schema stays unchanged and the frame
#: detail round-trips through the existing JSONL machinery intact.
_FRAME_METADATA_KEY = "frame"


@dataclass(frozen=True, slots=True)
class FrameSample:
    """One extracted frame, with the parent video it came from.

    Attributes:
        frame_id: Derived frame identity; becomes ``MediaSample.sample_id``.
        source_video_id: Identity of the parent video **file**. Required, and what
            :attr:`frame_id` is derived from.
        source_recording_id: Identity of the underlying source recording. Required. Several
            ``source_video_id`` values share one recording id, and this is the key frames
            are grouped by, which is what stops a clip and its forgeries from landing in
            different splits.
        frame_index: Zero-based position in the parent video. Required.
        frame_path: Frame image path relative to the extraction root.
        dataset_name: Dataset the parent video belongs to.
        label: Ground truth, carried from the parent video's adapter-assigned label.
            Never inferred from a path.
        manipulation_type: Manipulation family, carried from the parent.
        source_video_sha256: Digest of the parent video's bytes. Required, so a frame can
            always be traced to an exact file.
        license: Licence of the parent dataset, required by :class:`MediaSample`.
        source: Provenance of the video, e.g. ``"original"``.
        split: Assigned split. Authoritative assignments live in the frozen split.
        width, height: Frame dimensions in pixels.
        frame_sha256: Digest of the frame image, or ``None`` when images were not written.
        statistics: Pixel summary as written by the quality filter.
        quality_reasons: Rejection reasons; empty when kept.
        sampling: The sampling plan used, so the selection is reconstructible.
        modality: Image or video. Frames are images.
    """

    frame_id: str
    source_video_id: str
    source_recording_id: str
    frame_index: int
    frame_path: str
    dataset_name: str
    label: SampleLabel
    source_video_sha256: str
    license: str
    manipulation_type: ManipulationType = ManipulationType.UNKNOWN
    source: str | None = None
    split: SplitName | None = None
    width: int | None = None
    height: int | None = None
    frame_sha256: str | None = None
    statistics: dict[str, Any] = field(default_factory=dict)
    quality_reasons: tuple[str, ...] = ()
    sampling: dict[str, Any] = field(default_factory=dict)
    modality: Modality = Modality.IMAGE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", str(self.frame_id).strip())
        object.__setattr__(self, "source_video_id", str(self.source_video_id).strip())
        object.__setattr__(self, "source_recording_id", str(self.source_recording_id).strip())
        object.__setattr__(self, "dataset_name", str(self.dataset_name).strip())
        object.__setattr__(self, "label", SampleLabel.parse(self.label))
        object.__setattr__(
            self,
            "manipulation_type",
            ManipulationType.parse(self.manipulation_type),
        )
        object.__setattr__(
            self,
            "source",
            self.source.strip() if isinstance(self.source, str) else None,
        )
        object.__setattr__(
            self, "split", SplitName.parse(self.split) if self.split is not None else None
        )
        object.__setattr__(self, "modality", Modality.parse(self.modality))
        object.__setattr__(self, "quality_reasons", tuple(self.quality_reasons))

        expected_id = derive_frame_id(self.source_video_id, self.frame_index)
        if self.frame_id != expected_id:
            raise FrameManifestError(
                f"frame_id {self.frame_id!r} does not match its parent and index; "
                f"expected {expected_id!r}. Derive it with derive_frame_id() so ids stay "
                "a function of the data rather than of the caller."
            )
        if not self.source_recording_id:
            # Checked here rather than left to the grouping pass: a frame with no recording
            # identity cannot be grouped, and cannot therefore be kept out of a second
            # split. Refusing at construction means that frame never reaches a manifest.
            raise FrameManifestError(
                f"{self.frame_id}: source_recording_id must not be empty. Derive it with "
                "derive_source_recording_id() so frames can be grouped by recording."
            )

    # --- validation -------------------------------------------------------

    def require_parent_linkage(self) -> FrameSample:
        """Assert the parent video is fully identified, and return self.

        Checks the fields that make grouping possible and verifiable: a unique video
        identity, a recording identity to group by, and a parent digest.

        Raises:
            FrameManifestError: If any is missing, naming which.
        """
        missing: list[str] = []
        if not self.source_video_id:
            missing.append("source_video_id")
        if not self.source_recording_id:
            missing.append("source_recording_id")
        if not self.source_video_sha256:
            missing.append("source_video_sha256")
        if missing:
            raise FrameManifestError(
                f"{self.frame_id}: missing parent linkage: {', '.join(missing)}. "
                "source_video_sha256 is required so every frame is traceable to an exact "
                "source file, and the ids are required so frames can be grouped by "
                "recording."
            )
        return self

    @property
    def has_parent_linkage(self) -> bool:
        """Whether parent identity, recording identity, and digest are all present."""
        return bool(self.source_video_id and self.source_recording_id and self.source_video_sha256)

    # --- serialization ----------------------------------------------------

    def _frame_metadata(self) -> dict[str, Any]:
        """Frame-specific payload stored under the ``frame`` metadata key."""
        payload: dict[str, Any] = {
            "frame_index": self.frame_index,
            "source_video_id": self.source_video_id,
            "source_recording_id": self.source_recording_id,
            "source_video_sha256": self.source_video_sha256,
            "frame_sha256": self.frame_sha256,
            "quality_reasons": list(self.quality_reasons),
        }
        if self.width is not None:
            payload["width"] = self.width
        if self.height is not None:
            payload["height"] = self.height
        if self.statistics:
            payload["statistics"] = dict(self.statistics)
        if self.sampling:
            payload["sampling"] = dict(self.sampling)
        return payload

    def to_media_sample(self) -> MediaSample:
        """Convert to the manifest record this project already knows how to write.

        Going through :class:`MediaSample` means frames are written by
        :func:`datasets.manifest.write_manifest`, hashed by
        :func:`datasets.manifest.calculate_manifest_hash`, and validated by the same code
        as any other sample — no second manifest format to keep in sync.

        Raises:
            FrameManifestError: If the frame image digest is unknown. A manifest is a
                claim about bytes on disk, so a record with no digest of its own cannot
                make that claim. A stand-in value would be worse than an error: it would
                satisfy digest verification while attesting to a file it never hashed,
                turning a corruption check into a no-op.
        """
        if not self.frame_sha256:
            raise FrameManifestError(
                f"{self.frame_id}: cannot build a manifest record without the frame image "
                "digest. Hash the extracted file and set frame_sha256. A dry run writes no "
                "images, so it must not write a frame manifest."
            )
        metadata: dict[str, Any] = dict(self.metadata)
        metadata[_FRAME_METADATA_KEY] = self._frame_metadata()
        return MediaSample(
            sample_id=self.frame_id,
            dataset_name=self.dataset_name,
            filepath=self.frame_path,
            modality=self.modality,
            label=self.label,
            # The frame image digest is this sample's own identity. The parent video's
            # digest lives in metadata, because it identifies provenance rather than this
            # file.
            sha256=self.frame_sha256,
            license=self.license,
            manipulation_type=self.manipulation_type,
            source=self.source,
            split=self.split,
            metadata=metadata,
        )

    @classmethod
    def from_media_sample(cls, sample: MediaSample) -> FrameSample:
        """Rebuild a frame record from a manifest record.

        Raises:
            FrameManifestError: If the record is not a frame record, or its frame id
                disagrees with its parent and index.
        """
        raw = sample.metadata.get(_FRAME_METADATA_KEY)
        if not isinstance(raw, Mapping):
            raise FrameManifestError(
                f"{sample.sample_id}: not a frame record; expected a "
                f"{_FRAME_METADATA_KEY!r} metadata block"
            )
        frame_index = raw.get("frame_index")
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise FrameManifestError(
                f"{sample.sample_id}: frame_index must be an integer, got {frame_index!r}"
            )
        source_video_id = raw.get("source_video_id")
        if not isinstance(source_video_id, str) or not source_video_id.strip():
            raise FrameManifestError(
                f"{sample.sample_id}: source_video_id is required in the "
                f"{_FRAME_METADATA_KEY!r} metadata block"
            )
        source_recording_id = raw.get("source_recording_id")
        if not isinstance(source_recording_id, str) or not source_recording_id.strip():
            # Falling back to the video identity here would silently regroup frames that
            # should share a recording, reintroducing exactly the leakage this layout
            # exists to prevent. A manifest written before this field is a hard error.
            raise FrameManifestError(
                f"{sample.sample_id}: source_recording_id is required in the "
                f"{_FRAME_METADATA_KEY!r} metadata block; it cannot be inferred from "
                "source_video_id"
            )
        statistics = raw.get("statistics")
        sampling = raw.get("sampling")
        extra = {
            key: value
            for key, value in sample.metadata.items()
            if key not in {_FRAME_METADATA_KEY, "_extra"}
        }
        if "_extra" in sample.metadata and isinstance(sample.metadata["_extra"], Mapping):
            extra["_extra"] = dict(sample.metadata["_extra"])

        return cls(
            frame_id=sample.sample_id,
            source_video_id=source_video_id,
            source_recording_id=source_recording_id,
            frame_index=frame_index,
            frame_path=sample.filepath,
            dataset_name=sample.dataset_name,
            label=sample.label,
            source_video_sha256=str(raw.get("source_video_sha256") or ""),
            license=sample.license,
            manipulation_type=sample.manipulation_type,
            source=sample.source,
            split=sample.split,
            width=_optional_int(raw.get("width")),
            height=_optional_int(raw.get("height")),
            frame_sha256=_optional_str(raw.get("frame_sha256")),
            statistics=dict(statistics) if isinstance(statistics, Mapping) else {},
            quality_reasons=tuple(raw.get("quality_reasons") or ()),
            sampling=dict(sampling) if isinstance(sampling, Mapping) else {},
            modality=sample.modality,
            metadata=extra,
        )

    def with_split(self, split: SplitName | str | None) -> FrameSample:
        """Return a copy assigned to a split."""
        return replace(self, split=SplitName.parse(split) if split is not None else None)

    def to_dict(self) -> dict[str, Any]:
        """Frame-shaped dict, for reports. Manifest IO uses :meth:`to_media_sample`."""
        return {
            "frame_id": self.frame_id,
            "source_video_id": self.source_video_id,
            "source_recording_id": self.source_recording_id,
            "frame_index": self.frame_index,
            "frame_path": self.frame_path,
            "dataset_name": self.dataset_name,
            "label": self.label.value,
            "manipulation_type": self.manipulation_type.value,
            "source_video_sha256": self.source_video_sha256,
            "frame_sha256": self.frame_sha256,
            "width": self.width,
            "height": self.height,
            "split": self.split.value if self.split is not None else None,
            "quality_reasons": list(self.quality_reasons),
            "statistics": dict(self.statistics),
            "sampling": dict(self.sampling),
        }


def frames_to_media_samples(frames: Iterable[FrameSample]) -> list[MediaSample]:
    """Convert frames to manifest records, asserting parent linkage first.

    The assertion runs here so a manifest cannot be written from frames that would
    silently break grouping later.
    """
    return [frame.require_parent_linkage().to_media_sample() for frame in frames]


def frames_from_media_samples(samples: Iterable[MediaSample]) -> list[FrameSample]:
    """Convert manifest records back to frame records."""
    return [FrameSample.from_media_sample(sample) for sample in samples]


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
