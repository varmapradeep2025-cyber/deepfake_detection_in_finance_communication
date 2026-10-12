"""Group-aware train/validation/test assignment.

The one idea
------------
A split is chosen over **groups of frames from one source video**, then expanded to frame
ids. Deciding per frame is the mistake this module exists to make impossible: every frame
id is distinct, so a per-frame split passes id-level disjointness while quietly putting
one recording in train and its own forgery in test.

Why not :func:`datasets.split_schema.assign_splits`
---------------------------------------------------
It is a flat shuffle with ``int(size * fraction)`` cut points, and its own docstring warns
that it does not group near-duplicates. Calling it on *group* ids and expanding the result
would be a legitimate reuse — but the truncation cannot keep each stratum balanced. A
stratum of three groups with 0.8/0.1/0.1 gets ``int(3*0.1) == 0`` for validation, so
validation fills up from whichever stratum happens to be walked last, and the class balance
of the result is an accident. This module assigns by remaining capacity instead, which
makes the per-stratum counts exact.

The invariants it relies on, rather than reimplements, are
:class:`~datasets.split_schema.FrozenSplit` and
:func:`~datasets.split_schema.validate_split_id_sets`.

Determinism
-----------
Group order comes from a SHA-256 of ``seed`` and the group id, never Python's built-in
``hash()``, whose salt varies per process. So the same seed and groups produce the same
split on any machine, in any Python build — which is what makes the result freezable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datasets.preprocessing.grouping import (
    GroupingError,
    LeakageReport,
    SourceVideoGroup,
    assert_no_group_leakage,
)
from datasets.schema import SampleLabel, SplitName, utc_timestamp
from datasets.split_schema import FrozenSplit

__all__ = [
    "DEFAULT_GROUP_SPLIT_CONFIG",
    "GroupSplitConfig",
    "GroupSplitResult",
    "assign_groups_to_splits",
    "build_frame_split",
    "expand_groups_to_frames",
    "stable_sort_key",
    "to_frozen_split",
]


@dataclass(frozen=True, slots=True)
class GroupSplitConfig:
    """How groups are divided between splits.

    Attributes:
        train_fraction: Fraction of groups for training.
        validation_fraction: Fraction for validation. The remainder goes to test, so
            leaving this at its default and raising ``train_fraction`` is valid.
        seed: Seed for the deterministic group ordering.
        stratify_by_label: Keep each label-mix group in its own stratum, so splits stay
            class-balanced. On FaceForensics++ this costs nothing and helps a lot: a
            recording contributes both genuine and forged frames, so moving the whole
            recording keeps the class ratio stable by construction.
    """

    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    seed: int = 20260101
    stratify_by_label: bool = True

    def __post_init__(self) -> None:
        for name in ("train_fraction", "validation_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise GroupingError(f"{name} must be within 0.0-1.0, got {value}")
        if self.train_fraction + self.validation_fraction > 1.0:
            raise GroupingError(
                "train_fraction + validation_fraction must be <= 1.0, got "
                f"{self.train_fraction + self.validation_fraction}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise GroupingError("seed must be an integer")

    @property
    def test_fraction(self) -> float:
        """Remaining share, derived rather than configured, so it cannot disagree."""
        return 1.0 - self.train_fraction - self.validation_fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_fraction": self.train_fraction,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "seed": self.seed,
            "stratify_by_label": self.stratify_by_label,
        }


DEFAULT_GROUP_SPLIT_CONFIG = GroupSplitConfig()


def stable_sort_key(seed: int, *parts: str) -> str:
    """A deterministic sort key for grouping and ordering.

    SHA-256 over the seed and the supplied parts. Used instead of ``hash()`` because
    ``hash()`` is salted per process, which would make a split differ between two runs on
    the same machine — the exact failure frozen splits exist to prevent.
    """
    payload = "|".join([str(seed), *parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GroupSplitResult:
    """A completed group-aware split.

    Attributes:
        group_assignment: Group id to split name value.
        frame_assignment: Frame id to split name value.
        dataset_name: Dataset the split covers, inferred from the groups when not given.
        notes: Anything a reader must know — a stratum too small to divide three ways, or
            a split that came out empty.
        config: The configuration used.
        leakage: The verification report, always present even on success.
    """

    group_assignment: Mapping[str, str]
    frame_assignment: Mapping[str, str]
    dataset_name: str = ""
    notes: tuple[str, ...] = ()
    config: GroupSplitConfig = DEFAULT_GROUP_SPLIT_CONFIG
    leakage: LeakageReport | None = None

    def frames_for(self, split_name: SplitName | str) -> tuple[str, ...]:
        """Frame ids in one split, sorted."""
        target = SplitName.parse(split_name).value
        return tuple(
            sorted(frame_id for frame_id, value in self.frame_assignment.items() if value == target)
        )

    def groups_for(self, split_name: SplitName | str) -> tuple[str, ...]:
        """Group ids in one split, sorted."""
        target = SplitName.parse(split_name).value
        return tuple(
            sorted(group_id for group_id, value in self.group_assignment.items() if value == target)
        )

    def counts(self) -> dict[str, int]:
        """Frame counts per split."""
        counts = {member.value: 0 for member in SplitName}
        for value in self.frame_assignment.values():
            counts[value] = counts.get(value, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "counts": self.counts(),
            "group_counts": {split.value: len(self.groups_for(split)) for split in SplitName},
            "notes": list(self.notes),
            "leakage": self.leakage.to_dict() if self.leakage is not None else None,
        }


def assign_groups_to_splits(
    groups: Iterable[SourceVideoGroup],
    config: GroupSplitConfig | None = None,
) -> dict[str, str]:
    """Assign each source-video group to exactly one split.

    Per stratum, group counts are converted to per-split capacities, groups are visited in
    a seed-determined order, and each takes the split with the most remaining capacity.
    Because capacity is exhausted exactly, the per-stratum counts match the requested
    fractions regardless of the order groups happen to arrive in.

    Ties in remaining capacity resolve in a fixed order (train, validation, test), so a
    stratum too small to divide three ways deterministically keeps its groups in the
    larger splits instead of leaving validation and test empty by accident.

    Args:
        groups: Source-video groups. Groups with no frames are ignored.
        config: Fractions, seed, and stratification. Defaults to
            :data:`DEFAULT_GROUP_SPLIT_CONFIG`.

    Returns:
        Group id to split name value.
    """
    settings = DEFAULT_GROUP_SPLIT_CONFIG if config is None else config
    usable = [group for group in groups if group.frame_count > 0]
    if not usable:
        return {}

    strata: dict[str, list[SourceVideoGroup]] = {}
    for group in usable:
        key = group.stratification_key if settings.stratify_by_label else "all"
        strata.setdefault(key, []).append(group)

    assignment: dict[str, str] = {}
    # train, validation, test - the order also breaks capacity ties.
    order = (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)

    for stratum_key, members in sorted(strata.items()):
        size = len(members)
        train_capacity = int(size * settings.train_fraction)
        validation_capacity = int(size * settings.validation_fraction)
        if settings.train_fraction > 0 and train_capacity == 0:
            # int() floors, so any stratum of fewer than 1/train_fraction members would get
            # no training data at all - a lone group would land in test, leaving a corpus
            # with nothing to train on. Reserve one for train instead.
            train_capacity = 1
        capacities = {
            SplitName.TRAIN: train_capacity,
            SplitName.VALIDATION: validation_capacity,
            # Remainder, so the capacities always sum to the stratum size.
            SplitName.TEST: size - train_capacity - validation_capacity,
        }
        ordered = sorted(
            members,
            key=lambda group: (stable_sort_key(settings.seed, stratum_key, group.group_id),),
        )
        for group in ordered:
            target = max(order, key=lambda split: (capacities[split], -order.index(split)))
            if capacities[target] <= 0:
                # Every split for this stratum is full, which only happens when size is
                # smaller than the number of splits. Give it to train rather than
                # dropping the group.
                target = SplitName.TRAIN
            capacities[target] = max(0, capacities[target] - 1)
            assignment[group.group_id] = target.value
    return assignment


def expand_groups_to_frames(
    groups: Iterable[SourceVideoGroup],
    group_assignment: Mapping[str, str],
) -> dict[str, str]:
    """Project a group-level assignment onto frame ids.

    Every frame of a group receives that group's split. That projection is what makes
    leakage structurally impossible: there is no per-frame decision to get wrong.
    """
    frame_assignment: dict[str, str] = {}
    for group in groups:
        split_value = group_assignment.get(group.group_id)
        if split_value is None:
            continue
        for frame_id in group.frame_ids:
            frame_assignment[frame_id] = split_value
    return frame_assignment


def build_frame_split(
    groups: Iterable[SourceVideoGroup],
    config: GroupSplitConfig | None = None,
    *,
    dataset_name: str | None = None,
    source_manifest_hash: str,
    label_by_frame: Mapping[str, SampleLabel | str] | None = None,
    verify: bool = True,
) -> GroupSplitResult:
    """Produce a verified, group-aware split over frames.

    The verification is not optional decoration: :func:`assert_no_group_leakage` is run
    before the result is returned, so a caller cannot accidentally skip it. It passes by
    construction here, and that is the point — the check would catch a future change that
    started assigning frames individually.

    Args:
        groups: Source-video groups to split.
        config: Split configuration.
        dataset_name: Dataset for the split. Inferred from the groups when omitted.
        source_manifest_hash: ``sha256:<hex>`` hash of the frame manifest this split
            covers, so the split is pinned to an exact data definition.
        label_by_frame: Optional frame id to label, for the balance report.
        verify: Set ``False`` only to inspect a deliberately leaky assignment, e.g. in a
            test that is asserting the check fires.

    Returns:
        The assignment, with the leakage report attached.

    Raises:
        GroupingError: On invalid fractions, or when verification finds leakage.
    """
    settings = DEFAULT_GROUP_SPLIT_CONFIG if config is None else config
    group_list = list(groups)
    resolved_dataset = dataset_name or _infer_dataset_name(group_list)

    notes: list[str] = []
    group_assignment = assign_groups_to_splits(group_list, settings)
    frame_assignment = expand_groups_to_frames(group_list, group_assignment)

    empty = [
        split.value
        for split in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)
        if not any(value == split.value for value in frame_assignment.values())
    ]
    if empty:
        notes.append(
            f"{', '.join(empty)} contains no frames. With "
            f"{len(group_list)} source-video group(s) this is expected; increase the "
            "number of source videos before reading anything into it."
        )

    leakage: LeakageReport | None = None
    if verify:
        leakage = assert_no_group_leakage(
            group_list, frame_assignment, label_by_frame=label_by_frame, strict=True
        )

    return GroupSplitResult(
        group_assignment=group_assignment,
        frame_assignment=frame_assignment,
        dataset_name=resolved_dataset,
        notes=tuple(notes),
        config=settings,
        leakage=leakage,
    )


def to_frozen_split(
    result: GroupSplitResult,
    *,
    source_manifest_hash: str,
    dataset_name: str | None = None,
    rationale: str = "",
    extra: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> FrozenSplit:
    """Convert an assignment into the project's existing frozen split artifact.

    Frame ids are expanded here, so :class:`FrozenSplit` receives exactly one id per split
    and its own disjointness enforcement runs as a second, independent check on top of the
    group-level one.

    The grouping policy is recorded in ``rationale`` and ``extra``. That is not
    housekeeping: a split whose id sets look correct is still useless if nobody can tell
    later whether source-video grouping was applied, and that is the first question anyone
    should ask of a reported score.

    Args:
        result: A verified :class:`GroupSplitResult`.
        dataset_name: Dataset the split covers. Falls back to the name
            :func:`build_frame_split` inferred from the groups.
        source_manifest_hash: ``sha256:<hex>`` frame-manifest hash.
        rationale: Why this split is what it is. Prefixed with the grouping guarantee.
        extra: Additional provenance to store on the split.
        created_at: Override the creation timestamp. Supplying it makes the artifact
            byte-reproducible; omitting it stamps the current time.

    Raises:
        GroupingError: If no dataset name can be determined.
    """
    resolved_dataset = dataset_name or result.dataset_name
    if not resolved_dataset:
        raise GroupingError(
            "cannot build a frozen split without a dataset name; pass dataset_name= or "
            "build the assignment with build_frame_split() over groups that carry one"
        )
    leakage = result.leakage
    if leakage is None:
        guarantee = (
            "source-video grouping applied, but the leakage check was NOT run for this "
            "split. Do not report results against it."
        )
    else:
        guarantee = (
            f"source-video grouping applied: {leakage.group_count} group(s) across "
            f"{leakage.assigned_groups} assigned group(s); leakage check "
            f"{'passed' if leakage.is_leak_free else 'FAILED'}. "
            "Every frame derived from one source video is in exactly one split."
        )
    notes = f"{guarantee} {rationale}".strip()

    provenance: dict[str, Any] = dict(extra or {})
    provenance.setdefault("split_config", result.config.to_dict())
    if leakage is not None:
        provenance.setdefault("leakage_report", leakage.to_dict())
    if result.notes:
        provenance.setdefault("notes", list(result.notes))

    # FrozenSplit.create() has no `extra` parameter, so the class is constructed directly
    # to carry provenance. Disjointness is still enforced in __post_init__.
    return FrozenSplit(
        dataset_name=resolved_dataset,
        train_ids=result.frames_for(SplitName.TRAIN),
        validation_ids=result.frames_for(SplitName.VALIDATION),
        test_ids=result.frames_for(SplitName.TEST),
        created_at=created_at or utc_timestamp(),
        seed=result.config.seed,
        source_manifest_hash=source_manifest_hash,
        rationale=notes,
        extra=provenance,
    )


def _infer_dataset_name(groups: Sequence[SourceVideoGroup]) -> str:
    """Pick the dataset name shared by the groups.

    Raises:
        GroupingError: If the groups span more than one dataset, because one split cannot
            meaningfully cover several.
    """
    names = {group.dataset_name for group in groups if group.dataset_name}
    if not names:
        raise GroupingError("cannot infer a dataset name: no group carries one")
    if len(names) > 1:
        raise GroupingError(
            f"groups span {len(names)} datasets ({', '.join(sorted(names))}); build one "
            "split per dataset rather than mixing them"
        )
    return names.pop()
