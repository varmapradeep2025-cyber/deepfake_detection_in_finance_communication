"""End-to-end preprocessing: validated video manifest in, frozen artifacts out.

Artifacts written
-----------------
===============================  ====================================================
``frames.jsonl``                 Frame manifest, *only* when frames were actually written.
                                  Passed through :func:`datasets.manifest.write_manifest`,
                                  so it is validated and deterministically ordered like
                                  any other manifest in the project. Every frame carries
                                  the digest of the image on disk.
``inspection.json``              Every video inspected, including the ones that could not
                                  be decoded and why.
``rejected_frames.json``         Every rejected frame with its reasons and statistics.
                                 Kept out of ``frames.jsonl`` on purpose: the manifest is
                                 the training-data definition, and a rejected frame is not
                                 part of it.
``groups.json``                  Source-video groups and any identity conflicts.
``duplicates.json``              Exact-duplicate findings over the kept frames: every
                                  digest shared by more than one frame, plus frames whose
                                  digest could not be verified. Written even in a dry run,
                                  where the missing digests are reported rather than
                                  assumed unique.
``split.json``                   The frozen split, via the existing
                                  :func:`datasets.split_schema.write_split_definition`.
                                  Skipped in a dry run, since a split must pin a real
                                  frame manifest.
``preprocessing_report.json``    Summary, including the leakage report.
===============================  ====================================================

Ordering is load-bearing
-----------------------
The frame manifest is written and hashed *before* the split is built, so
``source_manifest_hash`` pins the split to the exact frame set it covers. Hashing after
the fact would leave a window where the manifest and the split describe different data.

Duplicate handling is equally load-bearing. Before the split is built, source-recording
groups linked by identical frame content are merged into one group, so exactly-duplicated
content between recordings can never land in different splits. The existing group-aware
splitter and its leakage check then apply to the merged group unchanged.

What a run refuses to do
------------------------
* Extract from a file that would not decode. It is reported and skipped, never coerced.
* Write a frame whose parent video cannot be identified.
* Produce a split in which any source-recording group spans two splits — that raises
  rather than being written and annotated.
* Write a frame manifest for frames that were never written. A digest for a file that does
  not exist would satisfy verification while checking nothing.
* Modify a source video. Raw media is only ever read.
* Let recordings holding identical frame content land in different splits — duplicate
  detection reports them, and the splitter keeps them in one merged group.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets.manifest import (
    calculate_file_sha256,
    calculate_manifest_hash,
    resolve_sample_path,
    write_manifest,
)
from datasets.preprocessing.duplicates import (
    DuplicateReport,
    coalesce_duplicate_linked_groups,
    find_exact_duplicates,
)
from datasets.preprocessing.extraction import (
    ExtractionConfig,
    ExtractionError,
    FrameExtractionResult,
    NullFrameWriter,
    VideoFrameExtractor,
)
from datasets.preprocessing.frames import (
    FrameManifestError,
    FrameSample,
    SourceIdentityLayout,
    derive_frame_id,
    derive_source_recording_id,
    derive_source_video_id,
    frames_to_media_samples,
)
from datasets.preprocessing.grouping import (
    GroupingResult,
    SourceVideoGroup,
    group_frames_by_source_video,
)
from datasets.preprocessing.inspection import MediaInspection, MediaInspector
from datasets.preprocessing.quality import FrameQualityFilter, QualityPolicy, QualitySummary
from datasets.preprocessing.splitting import (
    GroupSplitConfig,
    GroupSplitResult,
    build_frame_split,
    to_frozen_split,
)
from datasets.schema import MediaSample, Modality, SplitName
from datasets.split_schema import FrozenSplit, load_split_definition, write_split_definition

__all__ = [
    "DEFAULT_ARTIFACT_NAMES",
    "ArtifactNames",
    "InspectionBatch",
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "PreprocessingResult",
    "PreprocessingRunError",
]


class PreprocessingRunError(RuntimeError):
    """Raised when a preprocessing run cannot be completed honestly."""


@dataclass(frozen=True, slots=True)
class ArtifactNames:
    """Filenames written by a run. Overridable so a caller can version output dirs."""

    frames_manifest: str = "frames.jsonl"
    inspection: str = "inspection.json"
    rejected_frames: str = "rejected_frames.json"
    groups: str = "groups.json"
    duplicates: str = "duplicates.json"
    split: str = "split.json"
    report: str = "preprocessing_report.json"


DEFAULT_ARTIFACT_NAMES = ArtifactNames()


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Settings for one preprocessing run.

    Attributes:
        extraction: Frame count, strategy, output format, quality policy.
        split: Group-aware split fractions and seed.
        source_identity_layout: How a *recording* identity is derived from a path, which
            determines how frames are grouped and therefore what the split protects
            against. The single most consequential setting here; see
            :class:`~datasets.preprocessing.frames.SourceIdentityLayout`. Frame ids always
            use the per-file path, so this never affects id uniqueness.
        count_frames: Decode each video once up front to count frames, so sampling uses a
            measured count rather than a container's declaration. Costs one extra pass and
            is the difference between a reproducible sampling plan and a hopeful one.
        verify_source_digests: Re-hash each source video and compare with its manifest
            record. Reads every byte of the corpus, so it is off by default.
        dry_run: Select and evaluate frames but write no images. Reports are still
            written, because they describe the plan honestly; the frame manifest and the
            split are not, because both assert facts about image bytes that a dry run
            cannot produce.
        license_filter: Reject samples whose licence matches a blocked value. Empty
            disables the check, which is only appropriate for synthetic fixtures.
    """

    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    split: GroupSplitConfig = field(default_factory=GroupSplitConfig)
    source_identity_layout: SourceIdentityLayout = SourceIdentityLayout.FULL_RELATIVE_PATH
    count_frames: bool = True
    verify_source_digests: bool = False
    dry_run: bool = False
    license_filter: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_identity_layout",
            SourceIdentityLayout.parse(self.source_identity_layout),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction": self.extraction.to_dict(),
            "split": self.split.to_dict(),
            "source_identity_layout": self.source_identity_layout.value,
            "count_frames": self.count_frames,
            "verify_source_digests": self.verify_source_digests,
            "dry_run": self.dry_run,
            "license_filter": list(self.license_filter),
        }


@dataclass(frozen=True, slots=True)
class InspectionBatch:
    """Every inspection record, plus the subset that can be extracted from.

    The two are kept together deliberately. Keeping only the successes is how a run
    silently under-reports its own losses: 40 of 1000 videos fail to decode, and the
    artifact says "1000 videos inspected" with no trace of the 40. So every record is
    retained for reporting, including the failures and their reasons.

    Attributes:
        usable: ``(sample, inspection)`` pairs for videos that can be extracted from.
        records: Every inspection record, in manifest order, whatever its status.
    """

    usable: tuple[tuple[MediaSample, MediaInspection], ...] = ()
    records: tuple[MediaInspection, ...] = ()

    @property
    def failures(self) -> tuple[MediaInspection, ...]:
        """Records for videos that were skipped, with the reason attached."""
        return tuple(record for record in self.records if not record.is_usable)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Everything a run produced, whether or not artifacts were written.

    Attributes:
        frames: Kept frames, ready to be written as a manifest.
        rejected: Rejected frames with their reasons.
        inspections: Per-video inspection records.
        grouping: Source-video groups.
        split_result: Group-level assignment and frame projection.
        frozen_split: The split artifact, or ``None`` when no frames survived.
        artifacts: Paths actually written.
        quality: Aggregate quality outcome across all videos.
        duplicates: Exact-duplicate findings over the kept frames, or ``None`` when no
            frames survived.
        warnings: Run-level warnings.
        config: The configuration used.
    """

    frames: tuple[FrameSample, ...]
    rejected: tuple[FrameSample, ...]
    inspections: tuple[MediaInspection, ...]
    grouping: GroupingResult
    split_result: GroupSplitResult | None
    frozen_split: FrozenSplit | None
    artifacts: Mapping[str, Path] = field(default_factory=dict)
    quality: QualitySummary | None = None
    duplicates: DuplicateReport | None = None
    warnings: tuple[str, ...] = ()
    config: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @property
    def groups(self) -> tuple[SourceVideoGroup, ...]:
        return self.grouping.groups

    @property
    def is_leak_free(self) -> bool:
        """Whether the run produced a verified, leak-free split.

        ``False`` when no split was built, so an empty run cannot be mistaken for a clean
        one.
        """
        leakage = self.split_result.leakage if self.split_result is not None else None
        return leakage is not None and leakage.is_leak_free

    def summary(self) -> str:
        """One-line human summary."""
        parts = [
            f"videos={len(self.inspections)}",
            f"frames_kept={len(self.frames)}",
            f"frames_rejected={len(self.rejected)}",
            f"groups={len(self.groups)}",
        ]
        if self.split_result is not None:
            counts = self.split_result.counts()
            parts.append(
                "splits="
                + ",".join(
                    f"{name}:{counts.get(name.value, 0)}"
                    for name in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.TEST)
                )
            )
            parts.append(f"leak_free={self.is_leak_free}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "config": self.config.to_dict(),
            "frame_count": len(self.frames),
            "rejected_count": len(self.rejected),
            "group_count": len(self.groups),
            "is_leak_free": self.is_leak_free,
            "artifacts": {key: str(path) for key, path in sorted(self.artifacts.items())},
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "grouping": self.grouping.to_dict(),
            "duplicates": self.duplicates.to_dict() if self.duplicates is not None else None,
            "split": self.split_result.to_dict() if self.split_result is not None else None,
            "inspections": [record.to_dict() for record in self.inspections],
            "warnings": list(self.warnings),
        }


class PreprocessingPipeline:
    """Runs the preprocessing stages over a validated video manifest.

    Args:
        inspector: Inspects videos. Holds the decoder.
        extractor: Selects and writes frames.
        config: Run settings.
        names: Artifact filenames.

    The two collaborators are injected rather than constructed, so a test can drive the
    whole pipeline with a synthetic decoder and no image files on disk.
    """

    def __init__(
        self,
        inspector: MediaInspector,
        extractor: VideoFrameExtractor,
        config: PreprocessingConfig | None = None,
        names: ArtifactNames | None = None,
    ) -> None:
        self.inspector = inspector
        self.extractor = extractor
        self.config = PreprocessingConfig() if config is None else config
        self.names = DEFAULT_ARTIFACT_NAMES if names is None else names

    # --- stages -----------------------------------------------------------

    def select_videos(self, samples: Iterable[MediaSample]) -> tuple[MediaSample, ...]:
        """Filter a manifest down to usable video samples.

        Raises:
            PreprocessingRunError: If nothing remains, naming why each candidate was
                dropped rather than reporting an empty run with no cause.
        """
        videos: list[MediaSample] = []
        rejected: list[str] = []
        blocked = {value.strip().lower() for value in self.config.license_filter if value.strip()}

        for sample in samples:
            if sample.modality is not Modality.VIDEO:
                continue
            if blocked and sample.license.strip().lower() in blocked:
                rejected.append(f"{sample.sample_id}: licence {sample.license!r} is blocked")
                continue
            videos.append(sample)

        if not videos:
            detail = "; ".join(rejected) if rejected else "manifest contained no video samples"
            raise PreprocessingRunError(f"no usable video samples: {detail}")
        return tuple(videos)

    def inspect_videos(
        self, videos: Sequence[MediaSample], base_dir: Path | str
    ) -> InspectionBatch:
        """Inspect each video, keeping the pair so labels stay attached.

        A video that fails inspection is reported and skipped rather than aborting the
        batch: one corrupt file should cost one video, not the whole corpus. It is also
        *recorded*, so the run cannot quietly under-report how much data it lost.
        """
        usable: list[tuple[MediaSample, MediaInspection]] = []
        records: list[MediaInspection] = []
        for sample in videos:
            path = resolve_sample_path(sample, base_dir)
            record = self.inspector.inspect(path, count_frames=self.config.count_frames)
            records.append(record)
            if not record.is_usable:
                continue
            if (
                self.config.verify_source_digests
                and sample.sha256
                and record.sha256 is not None
                and record.sha256 != sample.sha256
            ):
                raise PreprocessingRunError(
                    f"{sample.sample_id}: file digest {record.sha256} does not match the "
                    f"manifest's {sample.sha256}. Refusing to extract from media that has "
                    "changed since the manifest was written."
                )
            usable.append((sample, record))
        return InspectionBatch(usable=tuple(usable), records=tuple(records))

    def build_frame_samples(
        self,
        videos: Sequence[MediaSample],
        inspections: Sequence[tuple[MediaSample, MediaInspection]],
        output_root: Path | None,
    ) -> tuple[list[FrameSample], list[FrameSample], list[QualitySummary], list[str]]:
        """Extract frames and turn them into frame records.

        Every kept frame is hashed from the file that was written, so the manifest's digest
        is a claim about real bytes. That costs one extra read of each frame image, which
        is the price of a manifest that can still detect a corrupted frame later. A dry run
        writes no images, so its frames carry no digest and it must not write a frame
        manifest.

        Returns:
            ``(kept, rejected, per-video quality summaries, warnings)``.
        """
        kept: list[FrameSample] = []
        rejected: list[FrameSample] = []
        summaries: list[QualitySummary] = []
        warnings: list[str] = []
        policy = self.extractor.quality_policy

        for sample, inspection in inspections:
            # Two distinct identities: the video file, which makes frame ids unique, and
            # the underlying recording, which is what frames are grouped by. Deriving both
            # from the same manifest path is what lets a clip and its forgeries share a
            # group without colliding on frame ids.
            source_video_id = derive_source_video_id(sample.dataset_name, sample.filepath)
            recording_id = derive_source_recording_id(
                sample.dataset_name,
                sample.filepath,
                layout=self.config.source_identity_layout,
            )
            try:
                result = self.extractor.extract(
                    inspection,
                    source_video_id=source_video_id,
                    output_root=output_root,
                )
            except ExtractionError as exc:
                # One video's planning or encoding failure must not lose the others.
                warnings.append(f"{sample.sample_id}: extraction failed: {exc}")
                continue

            summaries.append(_quality_summary(result, policy))
            for warning in result.warnings:
                warnings.append(f"{sample.sample_id}: {warning}")

            for frame in result.frames:
                reasons = frame.verdict.reason_values
                record = FrameSample(
                    frame_id=derive_frame_id(source_video_id, frame.index),
                    source_video_id=source_video_id,
                    source_recording_id=recording_id,
                    frame_index=frame.index,
                    frame_path=frame.relative_path,
                    dataset_name=sample.dataset_name,
                    label=sample.label,
                    manipulation_type=sample.manipulation_type,
                    source=sample.source,
                    source_video_sha256=inspection.sha256 or sample.sha256,
                    # Only accepted frames have image bytes on disk; rejected ones are
                    # audited from their recorded statistics. Hashing an unwritten path
                    # would fail, which is the correct outcome rather than a workaround.
                    frame_sha256=(
                        _hash_written_frame(
                            output_root, frame.relative_path, written=result.written
                        )
                        if frame.is_kept
                        else None
                    ),
                    license=sample.license,
                    width=frame.statistics.width,
                    height=frame.statistics.height,
                    statistics=frame.statistics.to_dict(),
                    quality_reasons=reasons,
                    sampling=result.plan.to_dict(),
                )
                if frame.is_kept:
                    kept.append(record)
                else:
                    rejected.append(record)

        return kept, rejected, summaries, warnings

    # --- orchestration ----------------------------------------------------

    def run(
        self,
        samples: Iterable[MediaSample],
        base_dir: Path | str,
        output_root: Path | str,
        *,
        write_artifacts: bool = True,
    ) -> PreprocessingResult:
        """Run every stage and optionally write the artifacts.

        Args:
            samples: A validated manifest, as loaded by
                :func:`datasets.manifest.load_manifest`.
            base_dir: Directory that relative ``filepath`` values resolve against.
            output_root: Directory for frames and artifacts. Created when writing.
            write_artifacts: ``False`` computes everything and returns it without touching
                the filesystem, which is the cheapest way to preview a run.

        Returns:
            The full result.

        Raises:
            PreprocessingRunError: If no usable videos, or if the split would leak.
            GroupingError: Propagated from the leakage check.
        """
        root = Path(output_root)
        frames_root = root / "frames"
        warnings: list[str] = []

        videos = self.select_videos(samples)
        batch = self.inspect_videos(videos, base_dir)
        if not batch.usable:
            reasons = "; ".join(f"{record.path}: {record.error}" for record in batch.failures[:5])
            raise PreprocessingRunError(
                f"none of the {len(videos)} video sample(s) could be decoded: {reasons}"
            )
        if batch.failure_count:
            warnings.append(
                f"{batch.failure_count} of {len(videos)} video(s) failed inspection and "
                f"were skipped; see {self.names.inspection} for the reasons"
            )

        writer = self.extractor.writer
        effective_root = None if isinstance(writer, NullFrameWriter) else frames_root
        images_persisted = effective_root is not None and self.extractor.writes_images
        if write_artifacts and effective_root is not None:
            effective_root.mkdir(parents=True, exist_ok=True)

        if write_artifacts and not images_persisted:
            # Said up front rather than leaving a reports-only directory that looks like a
            # finished run: the frame manifest and split are the deliverables, and neither
            # can honestly exist without the frames they describe.
            warnings.append(
                "dry run: no frame images were written, so no frame manifest or split "
                "artifact was written. Reports describe the plan only."
            )

        kept, rejected, summaries, extract_warnings = self.build_frame_samples(
            videos, batch.usable, effective_root
        )
        warnings.extend(extract_warnings)

        grouping = group_frames_by_source_video(kept)
        quality = _merge_summaries(summaries)

        # Guard before anything is written or hashed: a frame with no parent identity
        # cannot be grouped, and grouping is what makes the split leak-free.
        _require_frame_linkage(kept)

        duplicate_report = find_exact_duplicates(kept)
        if duplicate_report.unverifiable:
            warnings.append(
                f"{duplicate_report.unverifiable_count} frame(s) have no usable frame "
                "digest and were excluded from exact-duplicate detection; see "
                f"{self.names.duplicates} for the frames and reasons"
            )
        if duplicate_report.cross_recording:
            warnings.append(
                f"exact-duplicate detection found {len(duplicate_report.cross_source)} "
                "cross-source digest finding(s); "
                f"{len(duplicate_report.cross_recording)} of them link distinct source "
                "recordings, whose groups were merged before splitting so the duplicated "
                "content cannot cross splits; see "
                f"{self.names.duplicates} for the frames and digests"
            )

        artifacts: dict[str, Path] = {}
        frozen: FrozenSplit | None = None
        split_result: GroupSplitResult | None = None

        if not kept:
            warnings.append(
                "no frames survived extraction and quality filtering, so no split was produced"
            )
            if write_artifacts:
                artifacts = self._write_reports(
                    root,
                    frames=(),
                    rejected=rejected,
                    inspections=batch.records,
                    grouping=grouping,
                    split_result=None,
                    frozen_split=None,
                    quality=quality,
                    duplicates=None,
                    warnings=warnings,
                    config=self.config,
                )
            return PreprocessingResult(
                frames=(),
                rejected=tuple(rejected),
                inspections=batch.records,
                grouping=grouping,
                split_result=None,
                frozen_split=None,
                artifacts=artifacts,
                quality=quality,
                warnings=tuple(warnings),
                config=self.config,
            )

        # Exact duplicates first: recordings linked by identical content must be kept in
        # one group so the group-aware splitter cannot separate them. Coalescing is
        # identity-preserving when no cross-recording duplicate exists.
        coalesced = coalesce_duplicate_linked_groups(grouping.groups, duplicate_report)
        split_groups = coalesced.groups

        label_by_frame = {frame.frame_id: frame.label for frame in kept}

        frame_manifest_hash = _preview_manifest_hash(kept, self.config)
        if images_persisted:
            # Written and hashed before the split is built, so the split's
            # source_manifest_hash pins the exact frame set it covers.
            artifacts["frames_manifest"] = write_manifest(
                root / self.names.frames_manifest,
                frames_to_media_samples(kept),
                overwrite=True,
            )
            frame_manifest_hash = calculate_manifest_hash(frames_to_media_samples(kept))

        # build_frame_split verifies leakage and raises on a leak, so a split that reaches
        # the writer is leak-free by construction. It runs over the coalesced groups, which
        # is what makes duplicate-linked recordings atomic: their merged group cannot span
        # two splits without tripping the leakage check.
        split_result = build_frame_split(
            split_groups,
            self.config.split,
            source_manifest_hash=frame_manifest_hash,
            label_by_frame=label_by_frame,
        )
        duplicate_note: list[str] = []
        if coalesced.merged_components:
            duplicate_note.append(
                f"{len(coalesced.merged_components)} cluster(s) of source recordings "
                "linked by exact duplicate content were merged so duplicate-linked "
                "recordings stay in one split"
            )
        frozen = to_frozen_split(
            split_result,
            dataset_name=kept[0].dataset_name,
            source_manifest_hash=frame_manifest_hash,
            rationale=(
                "Group-aware split over source-recording groups. "
                f"Recording identity layout: {self.config.source_identity_layout.value}. "
                f"Frames per video: {self.config.extraction.frames_per_video} "
                f"({self.config.extraction.strategy.value}). "
                + (" ".join(duplicate_note) + "." if duplicate_note else "")
            ),
            extra={
                "frame_manifest_hash": frame_manifest_hash,
                "source_identity_layout": self.config.source_identity_layout.value,
                "duplicate_detection": duplicate_report.to_dict(),
                "duplicate_coalescing": coalesced.to_dict(),
            },
        )

        if write_artifacts:
            if images_persisted:
                artifacts["split"] = root / self.names.split
                write_split_definition(artifacts["split"], frozen)
            artifacts.update(
                self._write_reports(
                    root,
                    frames=kept,
                    rejected=rejected,
                    inspections=batch.records,
                    grouping=grouping,
                    split_result=split_result,
                    frozen_split=frozen,
                    quality=quality,
                    duplicates=duplicate_report,
                    warnings=warnings,
                    config=self.config,
                )
            )

        return PreprocessingResult(
            frames=tuple(kept),
            rejected=tuple(rejected),
            inspections=batch.records,
            grouping=grouping,
            split_result=split_result,
            frozen_split=frozen,
            artifacts=artifacts,
            quality=quality,
            duplicates=duplicate_report,
            warnings=tuple(warnings),
            config=self.config,
        )

    # --- artifacts --------------------------------------------------------

    def _write_reports(
        self,
        root: Path,
        *,
        frames: Sequence[FrameSample],
        rejected: Sequence[FrameSample],
        inspections: Sequence[MediaInspection],
        grouping: GroupingResult,
        split_result: GroupSplitResult | None,
        frozen_split: FrozenSplit | None,
        quality: QualitySummary | None,
        duplicates: DuplicateReport | None,
        warnings: Sequence[str],
        config: PreprocessingConfig,
    ) -> dict[str, Path]:
        """Write the JSON reports beside the manifests."""
        root.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        written["inspection"] = _write_json(
            root / self.names.inspection,
            {
                "decoder": self.inspector.decoder_name,
                "counted_frames": config.count_frames,
                "videos": [record.to_dict() for record in inspections],
            },
        )

        # Rejected frames are kept out of the frame manifest but fully documented here,
        # with their statistics, so a rejection is auditable without re-decoding.
        written["rejected_frames"] = _write_json(
            root / self.names.rejected_frames,
            {
                "rejected_count": len(rejected),
                "frames": [frame.to_dict() for frame in rejected],
            },
        )

        written["groups"] = _write_json(root / self.names.groups, grouping.to_dict())

        # Exact duplicates are reported separately from everything else: a reader must be
        # able to tell identical-content findings (digests match) apart from future
        # suspected near-duplicates (perceptual similarity), which must never be merged
        # into this file. Unverifiable digests are listed explicitly, never implied unique.
        if duplicates is not None:
            written["duplicates"] = _write_json(
                root / self.names.duplicates, duplicates.to_dict()
            )

        if frozen_split is not None and split_result is not None:
            leakage = split_result.leakage
            written["report"] = _write_json(
                root / self.names.report,
                {
                    "summary": (
                        f"videos={len(inspections)} frames={len(frames)} "
                        f"rejected={len(rejected)} groups={len(grouping.groups)} "
                        f"leak_free={leakage.is_leak_free if leakage else False}"
                    ),
                    "config": config.to_dict(),
                    "frame_manifest_hash": frozen_split.source_manifest_hash,
                    "split_content_hash": frozen_split.content_hash(),
                    "quality": quality.to_dict() if quality is not None else None,
                    "split": split_result.to_dict(),
                    "warnings": list(warnings),
                },
            )
        return written

    def load_frozen_split(self, output_root: Path | str) -> FrozenSplit:
        """Read back the split this run wrote.

        Reloading and re-validating is worth doing before training: it proves the artifact
        on disk is a valid split, not just that an in-memory object was.

        Raises:
            FileNotFoundError: If no split was written.
        """
        path = Path(output_root) / self.names.split
        if not path.is_file():
            raise FileNotFoundError(f"no split artifact at {path}")
        return load_split_definition(path)


def _quality_summary(result: FrameExtractionResult, policy: QualityPolicy) -> QualitySummary:
    """Quality summary for one video's selected frames.

    Uses the policy the extractor actually applied. Re-reading the default here would let
    a run with stricter thresholds report itself as if it had run the defaults.
    """
    return FrameQualityFilter(policy).summarize(result.verdicts)


def _hash_written_frame(
    output_root: Path | None, relative_path: str, *, written: bool
) -> str | None:
    """Digest of a frame image that was actually written.

    Returns ``None`` when nothing was persisted. Callers must not substitute a stand-in:
    the manifest is a statement about file contents, and an invented digest would pass
    verification without ever having checked a byte.
    """
    if not written or output_root is None:
        return None
    path = output_root / relative_path
    if not path.is_file():
        raise PreprocessingRunError(
            f"expected frame image at {path}, but it was not written. Refusing to record "
            "a frame whose bytes cannot be hashed."
        )
    return calculate_file_sha256(path)


def _preview_manifest_hash(frames: Sequence[FrameSample], config: PreprocessingConfig) -> str:
    """A clearly-marked stand-in hash for the frame manifest a dry run would write.

    Only used when no images were written, so no real manifest exists to hash. It has to
    satisfy the ``sha256:`` shape that :class:`FrozenSplit` validates, so it is derived
    from the plan and the frame ids, and the resulting split is labelled ``dry_run`` and
    never written to disk.
    """
    import hashlib

    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.frame_id.encode("utf-8"))
    digest.update(repr(sorted(config.to_dict().items())).encode("utf-8"))
    return f"sha256:dry-run-{digest.hexdigest()}"


def _merge_summaries(summaries: Sequence[QualitySummary]) -> QualitySummary:
    """Combine per-video quality summaries into one run-level summary."""
    if not summaries:
        return QualitySummary(total=0, kept=0, rejected=0)
    kept = sum(summary.kept for summary in summaries)
    rejected = sum(summary.rejected for summary in summaries)
    reason_counts: dict[str, int] = {}
    for summary in summaries:
        for reason, count in summary.reason_counts.items():
            reason_counts[reason] = reason_counts.get(reason, 0) + count
    return QualitySummary(
        total=kept + rejected,
        kept=kept,
        rejected=rejected,
        reason_counts=reason_counts,
        policy=summaries[0].policy,
    )


def _write_json(path: Path, payload: Any) -> Path:
    """Write a JSON report with deterministic formatting.

    Sort-keys and a fixed newline so re-running over unchanged data produces a
    byte-identical report, and so the files diff cleanly in review.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def _require_frame_linkage(frames: Iterable[FrameSample]) -> None:
    """Assert every frame names its parent video.

    Raises:
        FrameManifestError: Naming the first offending frame.
    """
    for frame in frames:
        try:
            frame.require_parent_linkage()
        except FrameManifestError as exc:
            raise FrameManifestError(
                f"refusing to build a frame manifest with broken parent linkage: {exc}"
            ) from exc
