"""Exact duplicate detection over frame digests, and leak-safe handling.

Scope
-----
This module finds *identical frame content* using the SHA-256 digests the frame manifest
already carries. It never opens a media file and never reads frame bytes: the digest is
the content identity, and the manifest wrote it from the extracted image. Detection is
therefore cheap, deterministic, and repeatable against a manifest alone, with no media
files in memory at any point.

What counts as a duplicate
--------------------------
A digest appearing on more than one frame. Two cases that must not be conflated:

**Within-source repeats.** The same digest appears several times inside one
``source_video_id`` — a static-camera shot where every sampled frame is pixel-identical.
These are reported, but they cannot leak: they are already inside one group and therefore
one split.

**Cross-source duplicates.** The same digest appears under two or more
``source_video_id`` values. When those videos also belong to different
``source_recording_id`` values, the duplicated content could otherwise land in different
splits — a genuine clip re-encoded or copied under a second path is the canonical case.
These are the findings that drive group coalescing.

Digests that cannot be verified
-------------------------------
A frame with no ``frame_sha256`` (a dry run) or a malformed digest is reported as
``unverifiable`` and never treated as either unique or duplicated. This module refuses to
guess: an unverifiable digest is not evidence of uniqueness.

Leakage prevention
------------------
The splitter assigns *groups* so no frame-level decision can leak. Duplicate detection
extends that unit: :func:`coalesce_duplicate_linked_groups` merges the source-recording
groups of every field linked by a cross-recording duplicate into one group, so the
existing group-aware splitter and its leakage check apply unchanged. Transitive links
(A shares content with B, B with C) are handled by connected components, so all three
end up atomically in one split.

Near-duplicates
---------------
Deliberately out of scope. Perceptual hashing needs decoded pixels and a similarity
threshold whose false-positive rate must be measured against the target media, so it is a
separate, explicitly-labelled feature — never mixed into exact-digest findings.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from datasets.preprocessing.frames import FrameSample
from datasets.preprocessing.grouping import GroupingError, SourceVideoGroup

__all__ = [
    "CoalescedGroups",
    "DuplicateFinding",
    "DuplicateReport",
    "UnverifiableFrame",
    "coalesce_duplicate_linked_groups",
    "duplicate_linked_components",
    "find_exact_duplicates",
    "normalise_frame_digest",
]

_HEX_DIGITS = frozenset("0123456789abcdef")

#: Prefix of the synthetic group id a coalesced duplicate cluster receives. Opaque and
#: deterministic; the member recording ids are available in the report and the split's
#: provenance, which is where a reviewer looks for them.
_GROUP_PREFIX = "dup::"


def normalise_frame_digest(value: str | None) -> tuple[str | None, str | None]:
    """Normalise a frame digest, or say why it cannot be used.

    Returns:
        ``(canonical_digest, reason)``. ``canonical_digest`` is lowercase hex when the
        value is usable; ``reason`` is ``"missing"`` or ``"malformed"`` when it is not.
        Exactly one is populated.

    The same rules as :meth:`datasets.schema`'s SHA-256 validation: 64 hex characters,
    canonicalised to lowercase so ``ABC…`` and ``abc…`` are the same digest, never picked
    up as two.
    """
    if value is None:
        return None, "missing"
    text = str(value).strip()
    if not text:
        return None, "missing"
    lowered = text.lower()
    if len(lowered) != 64 or not _HEX_DIGITS.issuperset(lowered):
        return None, "malformed"
    return lowered, None


@dataclass(frozen=True, slots=True)
class UnverifiableFrame:
    """A frame whose digest cannot be used for duplicate detection.

    Attributes:
        frame_id: The frame's id.
        source_video_id: Parent video, so the reader can find it on disk.
        reason: ``"missing"`` (no digest, e.g. dry run) or ``"malformed"``.
    """

    frame_id: str
    source_video_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "source_video_id": self.source_video_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DuplicateFinding:
    """One digest and every frame carrying it.

    Attributes:
        digest: The shared digest, lowercase hex.
        frame_ids: All frames with this digest, sorted.
        source_video_ids: Distinct parent videos involved, sorted.
        source_recording_ids: Distinct source recordings involved, sorted.
    """

    digest: str
    frame_ids: tuple[str, ...]
    source_video_ids: tuple[str, ...]
    source_recording_ids: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frame_ids)

    @property
    def crosses_sources(self) -> bool:
        """Whether the digest appears under more than one ``source_video_id``."""
        return len(self.source_video_ids) > 1

    @property
    def crosses_recordings(self) -> bool:
        """Whether the digest appears under more than one ``source_recording_id``.

        Only these findings can leak: two recordings sharing identical content must be
        kept in one split, which is what coalescing enforces.
        """
        return len(self.source_recording_ids) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "frame_count": self.frame_count,
            "crosses_sources": self.crosses_sources,
            "crosses_recordings": self.crosses_recordings,
            "frame_ids": list(self.frame_ids),
            "source_video_ids": list(self.source_video_ids),
            "source_recording_ids": list(self.source_recording_ids),
        }


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """Everything exact-duplicate detection found in one frame set.

    Attributes:
        findings: Every digest that appears on more than one frame, sorted by digest.
        unverifiable: Frames whose digest could not be used, sorted by frame id.
        total_frames: Frames examined.
    """

    findings: tuple[DuplicateFinding, ...] = ()
    unverifiable: tuple[UnverifiableFrame, ...] = ()
    total_frames: int = 0

    @property
    def verified_frames(self) -> int:
        return self.total_frames - len(self.unverifiable)

    @property
    def cross_source(self) -> tuple[DuplicateFinding, ...]:
        """Findings that span more than one source video."""
        return tuple(finding for finding in self.findings if finding.crosses_sources)

    @property
    def cross_recording(self) -> tuple[DuplicateFinding, ...]:
        """Findings that span more than one source recording — the leak-relevant ones."""
        return tuple(finding for finding in self.findings if finding.crosses_recordings)

    @property
    def within_source_only(self) -> tuple[DuplicateFinding, ...]:
        """Repeated content confined to one source video, which cannot leak."""
        return tuple(finding for finding in self.findings if not finding.crosses_sources)

    @property
    def unverifiable_count(self) -> int:
        return len(self.unverifiable)

    @property
    def is_clean(self) -> bool:
        """Whether no leak-relevant duplicate exists and every digest was usable.

        ``False`` when a digest is unverifiable, because an unverifiable digest is not
        evidence of uniqueness — the caller must decide how to treat it.
        """
        return not self.cross_recording and not self.unverifiable

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "verified_frames": self.verified_frames,
            "findings": [finding.to_dict() for finding in self.findings],
            "cross_source_findings": len(self.cross_source),
            "cross_recording_findings": len(self.cross_recording),
            "within_source_findings": len(self.within_source_only),
            "unverifiable_frames": [frame.to_dict() for frame in self.unverifiable],
        }


def find_exact_duplicates(frames: Iterable[FrameSample]) -> DuplicateReport:
    """Find every frame digest shared by more than one frame.

    The output is a pure function of the input *set*: frames are bucketed and every list
    is sorted, so reordering the input never changes the report.

    Args:
        frames: Frame records. Frames without a usable digest are reported as
            ``unverifiable``, never silently ignored.

    Returns:
        A :class:`DuplicateReport` with one finding per multi-frame digest.
    """
    buckets: dict[str, dict[str, set[str]]] = {}
    unverifiable: list[UnverifiableFrame] = []
    total = 0

    for frame in frames:
        total += 1
        digest, reason = normalise_frame_digest(frame.frame_sha256)
        if digest is None:
            unverifiable.append(
                UnverifiableFrame(
                    frame_id=frame.frame_id,
                    source_video_id=frame.source_video_id,
                    reason=reason or "missing",
                )
            )
            continue
        bucket = buckets.setdefault(
            digest, {"frame_ids": set(), "video_ids": set(), "recording_ids": set()}
        )
        bucket["frame_ids"].add(frame.frame_id)
        if frame.source_video_id:
            bucket["video_ids"].add(frame.source_video_id)
        if frame.source_recording_id:
            bucket["recording_ids"].add(frame.source_recording_id)

    findings = tuple(
        DuplicateFinding(
            digest=digest,
            frame_ids=tuple(sorted(bucket["frame_ids"])),
            source_video_ids=tuple(sorted(bucket["video_ids"])),
            source_recording_ids=tuple(sorted(bucket["recording_ids"])),
        )
        for digest, bucket in sorted(buckets.items())
        if len(bucket["frame_ids"]) > 1
    )

    return DuplicateReport(
        findings=findings,
        unverifiable=tuple(sorted(unverifiable, key=lambda frame: frame.frame_id)),
        total_frames=total,
    )


def duplicate_linked_components(report: DuplicateReport) -> tuple[frozenset[str], ...]:
    """Connected components of recording ids linked by cross-recording duplicates.

    Two recordings are in one component when identical content appears under both. The
    relationship is transitive: if A shares content with B and B shares content with C,
    all three form one component and must be assigned to one split together.

    Returns:
        The components as frozensets of ``source_recording_id`` values, sorted
        deterministically by their smallest member.
    """
    parent: dict[str, str] = {}
    component_members: set[str] = set()

    def find(key: str) -> str:
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:  # path compression
            parent[key], key = root, parent[key]
        return root

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # Deterministic parent choice: smaller id becomes the root.
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for finding in report.cross_recording:
        for recording_id in finding.source_recording_ids:
            parent.setdefault(recording_id, recording_id)
            component_members.add(recording_id)
        first, *rest = finding.source_recording_ids
        for recording_id in rest:
            union(first, recording_id)

    groups: dict[str, set[str]] = {}
    for recording_id in component_members:
        groups.setdefault(find(recording_id), set()).add(recording_id)

    return tuple(
        sorted((frozenset(members) for members in groups.values()), key=lambda c: sorted(c)[0])
    )


@dataclass(frozen=True, slots=True)
class CoalescedGroups:
    """Source-video groups after duplicate-linked recordings were merged.

    Attributes:
        groups: Groups ready for the splitter. Either the input untouched (when no
            cross-recording duplicate linked anything) or the merged form of it.
        merged_components: Each component of recording ids that was actually merged into
            one group, as group-id sets.
        removed_group_ids: Flat, sorted list of the group ids folded into the merged
            groups, so a reviewer can see exactly which recordings were fused.
    """

    groups: tuple[SourceVideoGroup, ...]
    merged_components: tuple[frozenset[str], ...] = ()
    removed_group_ids: tuple[str, ...] = ()

    @property
    def merged_count(self) -> int:
        return len(self.merged_components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged_components": [
                sorted(members) for members in self.merged_components
            ],
            "removed_group_ids": list(self.removed_group_ids),
        }


def coalesce_duplicate_linked_groups(
    groups: Iterable[SourceVideoGroup],
    report: DuplicateReport,
) -> CoalescedGroups:
    """Merge every group connected to another by a cross-recording duplicate.

    The merged group is a normal :class:`SourceVideoGroup`, so the existing group-aware
    splitter and its leakage check apply to it unchanged. Without any cross-recording
    finding the groups are returned untouched — coalescing is identity-preserving, so
    turning it on is harmless for a corpus with no content duplication.

    Only *cross-recording* duplicates drive merges. A digest shared by different videos of
    one recording is already inside one group, because grouping is per recording; merging
    it again would be a no-op.

    Args:
        groups: Source-video groups, as built by
            :func:`~datasets.preprocessing.grouping.group_frames_by_source_video`.
        report: A :class:`DuplicateReport` over the same frames those groups were built
            from.

    Returns:
        The coalesced group list and the fired merges.
    """
    by_id = {group.group_id: group for group in groups}

    components = []
    for component in duplicate_linked_components(report):
        members = [group_id for group_id in sorted(component) if group_id in by_id]
        if len(members) < 2:
            continue
        components.append(members)

    if not components:
        return CoalescedGroups(
            groups=tuple(sorted(by_id.values(), key=lambda group: group.group_id))
        )

    remaining: dict[str, SourceVideoGroup] = dict(by_id)
    merged_components: list[frozenset[str]] = []
    removed: list[str] = []
    for members in sorted(components, key=lambda component: component[0]):
        merged_components.append(frozenset(members))
        removed.extend(members)
        combined = _merge_groups([by_id[group_id] for group_id in members])
        for group_id in members:
            remaining.pop(group_id, None)
        remaining[combined.group_id] = combined

    return CoalescedGroups(
        groups=tuple(sorted(remaining.values(), key=lambda group: group.group_id)),
        merged_components=tuple(merged_components),
        removed_group_ids=tuple(sorted(removed)),
    )


def _merge_groups(members: Sequence[SourceVideoGroup]) -> SourceVideoGroup:
    """Combine source-video groups into one, unioning every set-valued field."""
    frame_ids = {frame_id for group in members for frame_id in group.frame_ids}
    labels = {label for group in members for label in group.labels}
    manipulations = {manip for group in members for manip in group.manipulation_types}
    digests = {digest for group in members for digest in group.source_video_sha256}
    video_ids = {video_id for group in members for video_id in group.source_video_ids}
    datasets = {group.dataset_name for group in members if group.dataset_name}
    if len(datasets) > 1:
        raise GroupingError(
            "cannot merge duplicate-linked groups from multiple datasets: "
            + ", ".join(sorted(datasets))
        )
    return SourceVideoGroup(
        group_id=_merged_group_id(members),
        frame_ids=tuple(sorted(frame_ids)),
        labels=tuple(sorted(labels)),
        manipulation_types=tuple(sorted(manipulations)),
        source_video_sha256=tuple(sorted(digests)),
        source_video_ids=tuple(sorted(video_ids)),
        dataset_name=datasets.pop() if datasets else "",
    )


def _merged_group_id(members: Sequence[SourceVideoGroup]) -> str:
    """Deterministic id for a coalesced cluster.

    A hash of the sorted member recording ids: stable across runs, collision-free, and
    identical wherever the same recordings are found. Same-member, so same id.
    """
    payload = "|".join(sorted(group.group_id for group in members))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_GROUP_PREFIX}{digest}"
