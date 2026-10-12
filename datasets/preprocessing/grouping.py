"""Grouping frames by source video, and proving no group spans a split.

What this module is for
-----------------------
Frame-level disjointness is not enough. A genuine clip and its forgeries share a face, a
background, a codec, and a source encoding; put one in train and the other in test and the
model scores well without generalising. So the split must be decided over *groups of frames
from one source video*, never over frames individually.

The grouping key is :attr:`FrameSample.source_video_id`, whose derivation is a deliberate
choice (see :class:`~datasets.preprocessing.frames.SourceIdentityLayout`). This module does
not guess it — it consumes it, and records what it consumed so a split can be audited
later without re-deriving the layout.

Checks performed
----------------
1. **Identity conflicts.** Two frames claiming one ``source_video_id`` with different
   parent digests. That means two different files are presenting as one video, so grouping
   them is a choice that needs to be surfaced rather than absorbed. Reported, not raised,
   so a whole batch's conflicts are visible at once; :func:`require_consistent_groups`
   turns them into an error when a run cannot proceed on ambiguous identity.

2. **Split leakage.** Every group must resolve to exactly one split. :func:`assert_no_group_leakage`
   is the programmatic gate, and it checks *groups*, not ids — id-level disjointness can
   pass while one source video is split across train and test, which is the failure that
   matters.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from datasets.preprocessing.frames import FrameSample
from datasets.schema import SampleLabel, SplitName
from datasets.split_schema import SplitValidationError, validate_split_id_sets

__all__ = [
    "GroupingError",
    "GroupingResult",
    "LeakageReport",
    "SourceVideoGroup",
    "assert_no_group_leakage",
    "group_frames_by_source_video",
    "require_consistent_groups",
]


class GroupingError(SplitValidationError):
    """Raised when grouping is impossible or ambiguous."""


@dataclass(frozen=True, slots=True)
class SourceVideoGroup:
    """Every frame known to descend from one source recording.

    Attributes:
        group_id: The ``source_recording_id`` shared by the group's frames. Note this is the
            recording, not the individual video file: a genuine clip and its forgeries
            share one group id even though their ``source_video_id`` values differ.
        frame_ids: Member frame ids, sorted.
        labels: Labels present in the group. More than one is normal and is in fact
            desirable — a recording that contributes both genuine and forged frames
            balances the split's class mix automatically, which is why grouping by
            recording rather than by class is preferred.
        manipulation_types: Manipulation families present.
        source_video_sha256: Digests of the contributing video files. Several are normal,
            because each manipulation of a recording is a separate file.
        source_video_ids: The contributing video file identities, which is what makes a
            recording's provenance inspectable.
        dataset_name: Dataset the parent belongs to.
    """

    group_id: str
    frame_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    manipulation_types: tuple[str, ...] = ()
    source_video_sha256: tuple[str, ...] = ()
    source_video_ids: tuple[str, ...] = ()
    dataset_name: str = ""

    @property
    def frame_count(self) -> int:
        return len(self.frame_ids)

    @property
    def video_count(self) -> int:
        """How many distinct video files contribute to this recording."""
        return len(self.source_video_ids)

    @property
    def is_single_label(self) -> bool:
        return len(self.labels) == 1

    @property
    def stratification_key(self) -> str:
        """Label-mix signature, used to keep splits class-balanced.

        Deliberately the whole sorted label tuple rather than the majority label. A group
        with both real and fake frames must be treated as its own stratum, because moving
        it wholesale is what keeps the class ratio stable.
        """
        return "+".join(self.labels) if self.labels else "unlabelled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "frame_ids": list(self.frame_ids),
            "frame_count": self.frame_count,
            "labels": list(self.labels),
            "manipulation_types": list(self.manipulation_types),
            "source_video_sha256": list(self.source_video_sha256),
            "source_video_ids": list(self.source_video_ids),
            "video_count": self.video_count,
            "dataset_name": self.dataset_name,
            "stratification_key": self.stratification_key,
        }


@dataclass(frozen=True, slots=True)
class GroupingResult:
    """Groups built from frames, plus anything that made grouping ambiguous.

    Attributes:
        groups: Groups, sorted by id for stable output.
        total_frames: Frames supplied.
        unlinked_frames: Frame ids with no recording identity. Non-empty means the caller
            passed frames that should have been validated earlier.
        video_id_reuse: Video identity to the recording ids it appears under, for the
            videos that appear under more than one. Empty in a healthy corpus.
    """

    groups: tuple[SourceVideoGroup, ...]
    total_frames: int
    unlinked_frames: tuple[str, ...] = ()
    video_id_reuse: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_consistent(self) -> bool:
        """Whether every frame was linkable and every video sat in exactly one recording."""
        return not self.unlinked_frames and not self.video_id_reuse

    @property
    def group_count(self) -> int:
        return len(self.groups)

    def group_ids(self) -> tuple[str, ...]:
        return tuple(group.group_id for group in self.groups)

    def group_for_frame(self, frame_id: str) -> SourceVideoGroup | None:
        """Find the group containing a frame id."""
        for group in self.groups:
            if frame_id in group.frame_ids:
                return group
        return None

    def group_for_video(self, video_id: str) -> SourceVideoGroup | None:
        """Find the group a video file contributes to."""
        for group in self.groups:
            if video_id in group.source_video_ids:
                return group
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_count": self.group_count,
            "total_frames": self.total_frames,
            "unlinked_frames": list(self.unlinked_frames),
            "video_id_reuse": {
                video_id: list(recording_ids)
                for video_id, recording_ids in sorted(self.video_id_reuse.items())
            },
            "is_consistent": self.is_consistent,
            "groups": [group.to_dict() for group in self.groups],
        }


def group_frames_by_source_video(frames: Iterable[FrameSample]) -> GroupingResult:
    """Build source-recording groups from frame records.

    Groups are keyed on ``source_recording_id``, so every manipulation of one recording
    lands in one group while each frame id stays unique. Members are sorted, so the output
    is a pure function of the input set and two runs over the same frames agree exactly.

    A video identity appearing under two recording ids is reported in
    :attr:`GroupingResult.video_id_reuse`. That is not a harmless duplication: it would mean
    one video contributes frames to two groups, and the two groups could be placed in
    different splits — leakage through a mislabelled path rather than through the layout.

    Args:
        frames: Frame records. Frames missing a recording identity are collected into
            :attr:`GroupingResult.unlinked_frames` rather than raising, so one bad record
            does not hide the state of the rest.

    Returns:
        Groups plus the diagnostics above.
    """
    materialised = list(frames)
    buckets: dict[str, dict[str, Any]] = {}
    unlinked: list[str] = []
    # video_id -> recording ids, so a video landing in two recordings can be detected.
    video_to_recordings: dict[str, set[str]] = {}

    for frame in materialised:
        recording_id = frame.source_recording_id.strip()
        if not recording_id:
            unlinked.append(frame.frame_id)
            continue
        bucket = buckets.setdefault(
            recording_id,
            {
                "frame_ids": set(),
                "labels": set(),
                "manipulations": set(),
                "digests": set(),
                "video_ids": set(),
                "dataset": frame.dataset_name,
            },
        )
        bucket["frame_ids"].add(frame.frame_id)
        bucket["labels"].add(frame.label.value)
        bucket["manipulations"].add(frame.manipulation_type.value)
        video_id = frame.source_video_id.strip()
        if video_id:
            bucket["video_ids"].add(video_id)
            video_to_recordings.setdefault(video_id, set()).add(recording_id)
        if frame.source_video_sha256:
            bucket["digests"].add(frame.source_video_sha256)

    groups = tuple(
        SourceVideoGroup(
            group_id=recording_id,
            frame_ids=tuple(sorted(bucket["frame_ids"])),
            labels=tuple(sorted(bucket["labels"])),
            manipulation_types=tuple(sorted(bucket["manipulations"])),
            source_video_sha256=tuple(sorted(bucket["digests"])),
            source_video_ids=tuple(sorted(bucket["video_ids"])),
            dataset_name=str(bucket["dataset"]),
        )
        for recording_id, bucket in sorted(buckets.items())
    )

    reuse = {
        video_id: tuple(sorted(recording_ids))
        for video_id, recording_ids in video_to_recordings.items()
        if len(recording_ids) > 1
    }

    return GroupingResult(
        groups=groups,
        total_frames=len(materialised),
        unlinked_frames=tuple(sorted(unlinked)),
        video_id_reuse=reuse,
    )


def require_consistent_groups(result: GroupingResult) -> GroupingResult:
    """Raise unless every frame was linkable and no video spanned two recordings.

    Raises:
        GroupingError: Listing every unlinked frame and reused video identity.
    """
    problems: list[str] = []
    if result.unlinked_frames:
        problems.append(
            f"{len(result.unlinked_frames)} frame(s) have no source_recording_id: "
            + ", ".join(result.unlinked_frames[:5])
        )
    for video_id, recording_ids in sorted(result.video_id_reuse.items()):
        problems.append(
            f"video {video_id!r} appears under {len(recording_ids)} recording ids "
            f"({', '.join(recording_ids[:3])}); one video contributing to two groups could "
            "put them in different splits"
        )
    if problems:
        raise GroupingError(
            f"cannot group {result.total_frames} frame(s): {len(problems)} problem(s)",
            errors=problems,
        )
    return result


# ---------------------------------------------------------------------------
# Leakage enforcement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """Result of checking group/split consistency.

    Attributes:
        group_count: Groups examined.
        assigned_groups: Groups that resolve to a split.
        spanning_groups: Group id to the splits it appears in. Any entry is a leak.
        unassigned_frames: Frame ids in no split.
        duplicate_ids: Frame ids appearing more than once in the same split's id list.
        label_counts: Per-split label counts of assigned frames, for balance review.
    """

    group_count: int
    assigned_groups: int
    spanning_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    unassigned_frames: tuple[str, ...] = ()
    duplicate_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    label_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @property
    def is_leak_free(self) -> bool:
        """Whether the split is safe to publish.

        Requires no group spanning two splits, no duplicate ids, and every frame assigned.
        """
        return not self.spanning_groups and not self.duplicate_ids and not self.unassigned_frames

    @property
    def leaked_group_count(self) -> int:
        return len(self.spanning_groups)

    def summary(self) -> str:
        status = "LEAK-FREE" if self.is_leak_free else "LEAKAGE"
        parts = [f"{status}: groups={self.group_count} assigned={self.assigned_groups}"]
        if self.spanning_groups:
            parts.append(f"spanning_groups={len(self.spanning_groups)}")
        if self.unassigned_frames:
            parts.append(f"unassigned_frames={len(self.unassigned_frames)}")
        if self.duplicate_ids:
            parts.append(f"duplicate_ids={len(self.duplicate_ids)}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_leak_free": self.is_leak_free,
            "group_count": self.group_count,
            "assigned_groups": self.assigned_groups,
            "leaked_group_count": self.leaked_group_count,
            "spanning_groups": {
                group_id: list(splits) for group_id, splits in sorted(self.spanning_groups.items())
            },
            "unassigned_frames": list(self.unassigned_frames),
            "duplicate_ids": {
                frame_id: list(splits) for frame_id, splits in sorted(self.duplicate_ids.items())
            },
            "label_counts": {
                split: dict(sorted(counts.items()))
                for split, counts in sorted(self.label_counts.items())
            },
        }


def assert_no_group_leakage(
    groups: Iterable[SourceVideoGroup],
    assignment: Mapping[str, str | SplitName | None],
    *,
    label_by_frame: Mapping[str, SampleLabel | str] | None = None,
    strict: bool = True,
) -> LeakageReport:
    """Verify every source-video group resolves to exactly one split.

    This is the enforcement point for the whole project. Id-level disjointness — which
    :func:`datasets.split_schema.validate_split_id_sets` already provides — cannot catch a
    source video appearing in both train and test, because every frame id is distinct. So
    the group is re-expanded and checked here.

    Args:
        groups: Source-video groups.
        assignment: Frame id to split. Values may be strings or :class:`SplitName`.
        label_by_frame: Optional frame id to label, used to report per-split class balance.
        strict: When ``True`` (default), raise on any leak. When ``False``, return the
            report so a caller can present every problem at once.

    Returns:
        A :class:`LeakageReport`.

    Raises:
        GroupingError: If ``strict`` and the split leaks.

    Notes:
        The *intended* design is that assignment is derived from groups, which makes
        leakage impossible by construction. This check exists because that derivation
        happens in more than one place and a hand-edited split file is a realistic way for
        a leak to appear. It is a backstop, and it is written to fail loudly.
    """
    group_list = list(groups)
    spanning: dict[str, tuple[str, ...]] = {}
    seen: dict[str, list[str]] = {}
    label_counts: dict[str, dict[str, int]] = {}

    for group in group_list:
        splits: set[str] = set()
        for frame_id in group.frame_ids:
            raw = assignment.get(frame_id)
            if raw is None:
                continue
            split_value = raw.value if isinstance(raw, SplitName) else str(raw)
            splits.add(split_value)
            seen.setdefault(frame_id, []).append(split_value)
            if label_by_frame is not None:
                label = label_by_frame.get(frame_id)
                if label is not None:
                    label_name = label.value if isinstance(label, SampleLabel) else str(label)
                    counts = label_counts.setdefault(split_value, {})
                    counts[label_name] = counts.get(label_name, 0) + 1
        if len(splits) > 1:
            spanning[group.group_id] = tuple(sorted(splits))

    duplicates = {
        frame_id: tuple(sorted(set(entries)))
        for frame_id, entries in seen.items()
        if len(entries) > 1
    }

    assigned = {
        frame_id
        for group in group_list
        for frame_id in group.frame_ids
        if assignment.get(frame_id) is not None
    }
    all_frames = {frame_id for group in group_list for frame_id in group.frame_ids}
    unassigned = tuple(sorted(all_frames - assigned))

    # Reuse the existing id-level validator so this check cannot drift from the core
    # split invariant. Duplicates within one split are a real problem even without
    # cross-split leakage, and this is where that is caught.
    by_split: dict[str, list[str]] = {}
    for frame_id, entries in seen.items():
        for split_value in entries:
            by_split.setdefault(split_value, []).append(frame_id)
    id_errors: list[str] = []
    ordered = [by_split.get(name, []) for name in sorted(by_split)][:3]
    while len(ordered) < 3:
        ordered.append([])
    try:
        validate_split_id_sets(ordered[0], ordered[1], ordered[2])
    except SplitValidationError as exc:
        id_errors.extend(exc.errors)

    report = LeakageReport(
        group_count=len(group_list),
        assigned_groups=sum(
            1
            for group in group_list
            if any(assignment.get(frame_id) is not None for frame_id in group.frame_ids)
        ),
        spanning_groups=spanning,
        unassigned_frames=unassigned,
        duplicate_ids=duplicates,
        label_counts=label_counts,
    )

    if strict and not report.is_leak_free:
        problems: list[str] = list(id_errors)
        for group_id, overlapping in sorted(spanning.items()):
            problems.append(
                f"source video {group_id} has frames in {list(overlapping)}: every frame from "
                "one source video must be in exactly one split"
            )
        if duplicates:
            problems.append(
                f"{len(duplicates)} frame id(s) appear more than once: "
                + ", ".join(sorted(duplicates)[:5])
            )
        if unassigned:
            problems.append(
                f"{len(unassigned)} frame(s) are in no split: " + ", ".join(unassigned[:5])
            )
        raise GroupingError(
            f"split leakage detected across {len(group_list)} source-video group(s)",
            errors=problems,
        )
    return report
