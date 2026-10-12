"""Media preprocessing: from a validated video manifest to a leakage-safe frame split.

The pipeline, in order
----------------------
1. **Inspect** — read each video's real stream properties. Unreadable properties are
   recorded as unavailable, never as defaults. (:mod:`~datasets.preprocessing.inspection`)
2. **Extract** — sample frames by a deterministic, seek-free rule so the same file always
   yields the same indices. (:mod:`~datasets.preprocessing.extraction`)
3. **Filter** — drop broken frames under a conservative, explicit policy, recording why.
   (:mod:`~datasets.preprocessing.quality`)
4. **Record** — give each frame a derived id and an unbreakable link to its parent video.
   (:mod:`~datasets.preprocessing.frames`)
5. **Group** — collect every frame from one source video into a single group.
   (:mod:`~datasets.preprocessing.grouping`)
6. **Split** — assign *groups* to train/validation/test, expand to frames, and prove no
   group spans a split. (:mod:`~datasets.preprocessing.splitting`)

The ordering is not arbitrary. Grouping must follow identification, because a group is
defined by parent-video identity; and splitting must follow grouping, because a split
decided per frame cannot be repaired afterwards.

What this package deliberately does not do
------------------------------------------
It never reads or writes raw source videos, never modifies a source file, and never
downloads anything. It derives new files (frames) and new manifests alongside the inputs,
and it does not touch labels: a frame's label is copied from the adapter that read the
dataset, never inferred from a filename, directory, or archive layout.
"""

from __future__ import annotations

from datasets.preprocessing.duplicates import (
    CoalescedGroups,
    DuplicateFinding,
    DuplicateReport,
    UnverifiableFrame,
    coalesce_duplicate_linked_groups,
    duplicate_linked_components,
    find_exact_duplicates,
    normalise_frame_digest,
)
from datasets.preprocessing.extraction import (
    DEFAULT_EXTRACTION_CONFIG,
    ExtractedFrame,
    ExtractionConfig,
    ExtractionError,
    FrameCountSource,
    FrameExtractionResult,
    FrameWriter,
    NullFrameWriter,
    OpenCVFrameWriter,
    SamplingPlan,
    SamplingStrategy,
    VideoFrameExtractor,
    available_strategies,
    default_frame_writer,
    every_n_indices,
    plan_uniform_sampling,
    register_strategy,
    resolve_frame_count,
    uniform_frame_indices,
)
from datasets.preprocessing.frames import (
    FrameManifestError,
    FrameSample,
    SourceIdentityLayout,
    derive_frame_id,
    derive_source_recording_id,
    derive_source_video_id,
    frames_from_media_samples,
    frames_to_media_samples,
)
from datasets.preprocessing.grouping import (
    GroupingError,
    GroupingResult,
    LeakageReport,
    SourceVideoGroup,
    assert_no_group_leakage,
    group_frames_by_source_video,
    require_consistent_groups,
)
from datasets.preprocessing.inspection import (
    AudioDecoder,
    AudioStreamInfo,
    DecodedFrame,
    FrameStatistics,
    ImageDecoder,
    ImageStreamInfo,
    InspectionStatus,
    MediaInspection,
    MediaInspectionError,
    MediaInspector,
    OpenCVDecoder,
    OpenCVImageDecoder,
    VideoDecoder,
    VideoDecoderUnavailableError,
    VideoStreamInfo,
    WaveAudioDecoder,
    compute_frame_statistics,
    default_audio_decoder,
    default_decoder,
    default_image_decoder,
    infer_modality,
    is_audio_candidate,
    is_image_candidate,
    is_video_candidate,
)
from datasets.preprocessing.pipeline import (
    DEFAULT_ARTIFACT_NAMES,
    ArtifactNames,
    PreprocessingConfig,
    PreprocessingPipeline,
    PreprocessingResult,
    PreprocessingRunError,
)
from datasets.preprocessing.quality import (
    DEFAULT_QUALITY_POLICY,
    FrameQualityFilter,
    QualityPolicy,
    QualityReason,
    QualitySummary,
    QualityVerdict,
)
from datasets.preprocessing.splitting import (
    DEFAULT_GROUP_SPLIT_CONFIG,
    GroupSplitConfig,
    GroupSplitResult,
    assign_groups_to_splits,
    build_frame_split,
    expand_groups_to_frames,
    stable_sort_key,
    to_frozen_split,
)

__all__ = [
    "DEFAULT_ARTIFACT_NAMES",
    "DEFAULT_EXTRACTION_CONFIG",
    "DEFAULT_GROUP_SPLIT_CONFIG",
    "DEFAULT_QUALITY_POLICY",
    "ArtifactNames",
    "AudioDecoder",
    "AudioStreamInfo",
    "CoalescedGroups",
    "DecodedFrame",
    "DuplicateFinding",
    "DuplicateReport",
    "ExtractedFrame",
    "ExtractionConfig",
    "ExtractionError",
    "FrameCountSource",
    "FrameExtractionResult",
    "FrameManifestError",
    "FrameQualityFilter",
    "FrameSample",
    "FrameStatistics",
    "FrameWriter",
    "GroupSplitConfig",
    "GroupSplitResult",
    "GroupingError",
    "GroupingResult",
    "ImageDecoder",
    "ImageStreamInfo",
    "InspectionStatus",
    "LeakageReport",
    "MediaInspection",
    "MediaInspectionError",
    "MediaInspector",
    "NullFrameWriter",
    "OpenCVDecoder",
    "OpenCVFrameWriter",
    "OpenCVImageDecoder",
    "PreprocessingConfig",
    "PreprocessingPipeline",
    "PreprocessingResult",
    "PreprocessingRunError",
    "QualityPolicy",
    "QualityReason",
    "QualitySummary",
    "QualityVerdict",
    "SamplingPlan",
    "SamplingStrategy",
    "SourceIdentityLayout",
    "SourceVideoGroup",
    "UnverifiableFrame",
    "VideoDecoder",
    "VideoDecoderUnavailableError",
    "VideoFrameExtractor",
    "VideoStreamInfo",
    "WaveAudioDecoder",
    "assert_no_group_leakage",
    "assign_groups_to_splits",
    "available_strategies",
    "build_frame_split",
    "coalesce_duplicate_linked_groups",
    "compute_frame_statistics",
    "default_audio_decoder",
    "default_decoder",
    "default_frame_writer",
    "default_image_decoder",
    "derive_frame_id",
    "derive_source_recording_id",
    "derive_source_video_id",
    "duplicate_linked_components",
    "every_n_indices",
    "expand_groups_to_frames",
    "find_exact_duplicates",
    "frames_from_media_samples",
    "frames_to_media_samples",
    "group_frames_by_source_video",
    "infer_modality",
    "is_audio_candidate",
    "is_image_candidate",
    "is_video_candidate",
    "normalise_frame_digest",
    "plan_uniform_sampling",
    "register_strategy",
    "require_consistent_groups",
    "resolve_frame_count",
    "stable_sort_key",
    "to_frozen_split",
    "uniform_frame_indices",
]
