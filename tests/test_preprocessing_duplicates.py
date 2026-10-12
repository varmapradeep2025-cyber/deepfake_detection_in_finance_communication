"""Tests for exact-duplicate detection and duplicate-aware splitting.

The property under test throughout: **identical frame content anywhere in the corpus can
never appear in two different splits.** The manifest already carries each frame's SHA-256,
so detection never opens media — and the splitter is extended by *merging* groups linked by
duplicated content, then running the existing group-aware leakage check over the merged
group, rather than by a second splitting system.

Covered here: within-source repeats vs cross-source duplicates, missing/malformed digests,
order independence, transitive links, split isolation, FaceForensics++ compatibility, and
the pipeline wiring end to end.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from datasets.manifest import calculate_file_sha256
from datasets.preprocessing import (
    FrameSample,
    GroupingError,
    GroupSplitConfig,
    SourceIdentityLayout,
    build_frame_split,
    coalesce_duplicate_linked_groups,
    derive_frame_id,
    derive_source_recording_id,
    derive_source_video_id,
    duplicate_linked_components,
    find_exact_duplicates,
    group_frames_by_source_video,
    normalise_frame_digest,
)
from datasets.preprocessing.extraction import ExtractionConfig, VideoFrameExtractor
from datasets.preprocessing.frames import SourceIdentityLayout as Layout
from datasets.preprocessing.grouping import SourceVideoGroup, assert_no_group_leakage
from datasets.preprocessing.inspection import MediaInspector
from datasets.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from datasets.schema import MediaSample, Modality, SampleLabel
from tests.synthetic_preprocessing import FakeVideo, FakeVideoDecoder

PARENT_SHA = "a" * 64
MANIFEST_HASH = f"sha256:{'c' * 64}"


def make_frame(
    source_video_id: str,
    frame_index: int,
    *,
    recording_id: str | None = None,
    label: SampleLabel = SampleLabel.FAKE,
    parent_sha: str = PARENT_SHA,
    frame_sha: str | None = "d" * 64,
    dataset: str = "ffpp",
) -> FrameSample:
    """Build a frame record, defaulting the frame digest to ``d`` hex."""
    return FrameSample(
        frame_id=derive_frame_id(source_video_id, frame_index),
        source_video_id=source_video_id,
        source_recording_id=recording_id or source_video_id,
        frame_index=frame_index,
        frame_path=f"{source_video_id}/frame_{frame_index:06d}.png",
        dataset_name=dataset,
        label=label,
        source_video_sha256=parent_sha,
        license="research-only",
        frame_sha256=frame_sha,
    )


def make_group(group_id: str, *, frame_count: int = 3, dataset: str = "ffpp") -> SourceVideoGroup:
    """A source-video group directly, for splitting tests."""
    return SourceVideoGroup(
        group_id=group_id,
        frame_ids=tuple(derive_frame_id(group_id, index) for index in range(frame_count)),
        labels=("fake",),
        source_video_sha256=(PARENT_SHA,),
        dataset_name=dataset,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestFindExactDuplicates:
    def test_unique_digests_produce_no_findings(self):
        report = find_exact_duplicates(
            [
                make_frame("a", 0, frame_sha="1" * 64),
                make_frame("b", 0, frame_sha="2" * 64),
            ]
        )
        assert report.findings == ()
        assert report.verified_frames == 2
        assert report.is_clean

    def test_identical_digests_within_one_source_are_reported(self):
        report = find_exact_duplicates(
            [
                make_frame("ffpp__000__001", 0),
                make_frame("ffpp__000__001", 1),
            ]
        )
        assert len(report.findings) == 1
        finding = report.findings[0]
        assert finding.crosses_sources is False
        assert finding.crosses_recordings is False
        assert finding.frame_ids == (
            "ffpp__000__001__f000000",
            "ffpp__000__001__f000001",
        )
        assert report.within_source_only == (finding,)
        assert report.cross_source == ()
        assert report.cross_recording == ()

    def test_cross_source_duplicate_shares_one_recording(self):
        # A genuine clip and its forgery, both real duplicates of each other's frames:
        # same recording, so already grouped together — never a split risk.
        frames = [
            make_frame("ffpp__original__000", 0, recording_id="ffpp__000"),
            make_frame("ffpp__DeepFakes__000", 0, recording_id="ffpp__000"),
        ]
        report = find_exact_duplicates(frames)
        finding = report.cross_source[0]
        assert finding.crosses_sources is True
        assert finding.crosses_recordings is False

    def test_cross_recording_duplicate_names_every_identity(self):
        # Two *different* recordings sharing a frame: canonical leakage vector.
        frames = [
            make_frame("ffpp__original__000", 0, recording_id="ffpp__000"),
            make_frame("ffpp__original__001", 0, recording_id="ffpp__001"),
        ]
        report = find_exact_duplicates(frames)
        finding = report.cross_source[0]
        assert finding.crosses_recordings is True
        assert finding.source_video_ids == ("ffpp__original__000", "ffpp__original__001")
        assert finding.source_recording_ids == ("ffpp__000", "ffpp__001")
        assert len(finding.frame_ids) == 2

    def test_report_lists_the_digest(self):
        digest = "ab" * 32
        report = find_exact_duplicates(
            [
                make_frame("a", 0, frame_sha=digest),
                make_frame("b", 0, frame_sha=digest),
            ]
        )
        assert report.findings[0].digest == digest

    def test_output_is_order_independent(self):
        frames = [
            make_frame("a", 0),
            make_frame("b", 0),
            make_frame("a", 1, frame_sha="9" * 64),
        ]
        forward = find_exact_duplicates(frames)
        backward = find_exact_duplicates(list(reversed(frames)))
        assert forward.to_dict() == backward.to_dict()

    def test_report_is_json_serialisable(self):
        report = find_exact_duplicates(
            [
                make_frame("a", 0),
                make_frame("b", 0),
                make_frame("c", 0, frame_sha=None),
            ]
        )
        json.dumps(report.to_dict())  # must not raise


class TestUnverifiableDigests:
    def test_missing_digest_is_reported_not_ignored(self):
        report = find_exact_duplicates([make_frame("a", 0, frame_sha=None)])
        assert report.findings == ()
        assert len(report.unverifiable) == 1
        entry = report.unverifiable[0]
        assert entry.frame_id == "a__f000000"
        assert entry.reason == "missing"
        assert report.is_clean is False

    def test_blank_digest_is_missing(self):
        for blank in ("", "   "):
            report = find_exact_duplicates([make_frame("a", 0, frame_sha=blank)])
            assert report.unverifiable[0].reason == "missing"

    def test_malformed_digest_is_reported(self):
        for bad in ("abc", "z" * 64, "a" * 63, "a" * 65, "a" * 63 + "G"):
            report = find_exact_duplicates([make_frame("a", 0, frame_sha=bad)])
            assert report.unverifiable[0].reason == "malformed", bad

    def test_unverifiable_frames_never_count_as_unique(self):
        # A frame whose digest is unusable must not be treated as distinct content.
        frames = [
            make_frame("a", 0, frame_sha="1" * 64),
            make_frame("b", 0, frame_sha=None),
        ]
        report = find_exact_duplicates(frames)
        assert report.findings == ()
        assert report.unverifiable_count == 1
        assert report.is_clean is False

    def test_uppercase_hex_matches_its_lowercase_twin(self):
        report = find_exact_duplicates(
            [
                make_frame("a", 0, frame_sha="AB" * 32),
                make_frame("b", 0, frame_sha="ab" * 32),
            ]
        )
        assert len(report.findings) == 1
        assert report.findings[0].digest == "ab" * 32

    def test_normalise_frame_digest_helper(self):
        assert normalise_frame_digest(None) == (None, "missing")
        assert normalise_frame_digest("") == (None, "missing")
        assert normalise_frame_digest("xyz") == (None, "malformed")
        assert normalise_frame_digest("A" * 64) == ("a" * 64, None)


# ---------------------------------------------------------------------------
# Connected components and coalescing
# ---------------------------------------------------------------------------


class TestDuplicateLinkedComponents:
    def test_no_cross_recording_findings_yields_no_components(self):
        report = find_exact_duplicates(
            [
                make_frame("ffpp__original__000", 0, recording_id="ffpp__000"),
                make_frame("ffpp__DeepFakes__000", 0, recording_id="ffpp__000"),
            ]
        )
        assert duplicate_linked_components(report) == ()

    def test_two_linked_recordings_form_one_component(self):
        report = find_exact_duplicates(
            [
                make_frame("a__v1", 0, recording_id="rec_a"),
                make_frame("b__v1", 0, recording_id="rec_b"),
            ]
        )
        assert duplicate_linked_components(report) == (frozenset({"rec_a", "rec_b"}),)

    def test_transitive_links_form_one_component(self):
        # a shares content with b, b shares with c -> all three in one component.
        frames = [
            make_frame("a__v0", 0, recording_id="rec_a"),
            make_frame("b__v0", 0, recording_id="rec_b"),
            make_frame("b__v1", 1, recording_id="rec_b"),
            make_frame("c__v0", 0, recording_id="rec_c"),
        ]
        report = find_exact_duplicates(frames)
        components = duplicate_linked_components(report)
        assert len(components) == 1
        assert components[0] == frozenset({"rec_a", "rec_b", "rec_c"})

    def test_disjoint_links_form_separate_components(self):
        frames = [
            make_frame("a__v0", 0, recording_id="rec_a", frame_sha="11" * 32),
            make_frame("b__v0", 0, recording_id="rec_b", frame_sha="11" * 32),
            make_frame("c__v0", 0, recording_id="rec_c", frame_sha="22" * 32),
            make_frame("d__v0", 0, recording_id="rec_d", frame_sha="22" * 32),
        ]
        components = duplicate_linked_components(find_exact_duplicates(frames))
        assert len(components) == 2

    def test_components_are_deterministic(self):
        frames = [
            make_frame("a__v0", 0, recording_id="rec_z"),
            make_frame("b__v0", 0, recording_id="rec_a"),
            make_frame("b__v1", 1, recording_id="rec_a"),
            make_frame("c__v0", 0, recording_id="rec_m"),
        ]
        report = find_exact_duplicates(frames)
        assert duplicate_linked_components(report) == duplicate_linked_components(report)


class TestCoalesceGroups:
    def test_no_cross_recording_duplicate_returns_groups_untouched(self):
        groups = [make_group("rec_a"), make_group("rec_b")]
        report = find_exact_duplicates(
            [
                make_frame("a__v1", 0, recording_id="rec_a"),
                make_frame("a__v1", 1, recording_id="rec_a"),
            ]
        )
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        assert [group.group_id for group in coalesced.groups] == ["rec_a", "rec_b"]
        assert coalesced.merged_components == ()
        assert coalesced.removed_group_ids == ()

    def test_linked_groups_are_merged_into_one(self):
        frames = [
            make_frame("a__v1", 0, recording_id="rec_a", label=SampleLabel.REAL),
            make_frame("a__v1", 1, recording_id="rec_a", label=SampleLabel.REAL),
            make_frame("b__v1", 0, recording_id="rec_b", label=SampleLabel.FAKE),
        ]
        report = find_exact_duplicates(frames)
        groups = [
            group
            for group in group_frames_by_source_video(frames).groups
            if group.frame_count > 0
        ]
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        assert coalesced.merged_count == 1
        assert coalesced.removed_group_ids == ("rec_a", "rec_b")
        merged = coalesced.groups[0]
        assert merged.frame_ids == (
            "a__v1__f000000",
            "a__v1__f000001",
            "b__v1__f000000",
        )
        assert merged.labels == ("fake", "real")
        assert merged.dataset_name == "ffpp"

    def test_transitive_merge_covers_three_recordings(self):
        frames = [
            make_frame(f"{group}__v0", index, recording_id=group)
            for group in ("rec_a", "rec_b", "rec_c")
            for index in (0, 1)
        ]
        report = find_exact_duplicates(frames)
        groups = [
            group for group in group_frames_by_source_video(frames).groups if group.frame_count
        ]
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        assert len(coalesced.groups) == 1
        assert coalesced.merged_count == 1
        assert set(coalesced.groups[0].frame_ids) == {frame.frame_id for frame in frames}

    def test_merge_is_deterministic(self):
        frames = [
            make_frame(f"rec_{group}__v0", index, recording_id=f"rec_{group}")
            for group in "abc"
            for index in (0, 1)
        ]
        report = find_exact_duplicates(frames)
        groups = [
            group for group in group_frames_by_source_video(frames).groups if group.frame_count
        ]
        first = coalesce_duplicate_linked_groups(groups, report)
        second = coalesce_duplicate_linked_groups(groups, report)
        assert first.to_dict() == second.to_dict()
        assert first.groups == second.groups

    def test_components_missing_from_groups_are_ignored(self):
        # The report may know about recording ids that are not in this group set; only the
        # groups actually present may be merged.
        frames = [
            make_frame("a__v0", 0, recording_id="rec_a"),
            make_frame("b__v0", 0, recording_id="rec_b"),
        ]
        report = find_exact_duplicates(frames)
        groups = [make_group("rec_a")]
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        assert coalesced.merged_count == 0
        assert [group.group_id for group in coalesced.groups] == ["rec_a"]


class TestSplitIsolation:
    def test_merged_cluster_stays_in_one_split(self):
        frames = [
            make_frame(f"rec_{group}__v0", index, recording_id=f"rec_{group}")
            for group in ("a", "b")
            for index in (0, 1, 2)
        ]
        report = find_exact_duplicates(frames)
        groups = [
            group for group in group_frames_by_source_video(frames).groups if group.frame_count
        ]
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        result = build_frame_split(
            coalesced.groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH
        )
        assert result.leakage.is_leak_free
        splits = {
            result.frame_assignment[frame_id] for frame_id in coalesced.groups[0].frame_ids
        }
        assert len(splits) == 1

    def test_id_level_check_passes_until_duplicated_content_is_coalesced(self):
        # Two recordings hold byte-identical frames. At the id level each recording is a
        # contiguous split member, so the backstop sees nothing wrong; the content link is
        # only visible after duplicate detection merges them into one group.
        frames = [
            make_frame("rec_a__v0", index, recording_id="rec_a") for index in range(4)
        ] + [make_frame("rec_b__v0", index, recording_id="rec_b") for index in range(4)]
        groups = [
            group for group in group_frames_by_source_video(frames).groups if group.frame_count
        ]
        assignment = {
            frame.frame_id: ("train" if frame.frame_id.startswith("rec_a") else "test")
            for frame in frames
        }
        assert_no_group_leakage(groups, assignment, strict=True)

        report = find_exact_duplicates(frames)
        assert len(report.cross_recording) == 1
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        assert coalesced.merged_count == 1
        with pytest.raises(GroupingError, match="split leakage detected"):
            assert_no_group_leakage(coalesced.groups, assignment, strict=True)


class TestFaceForensicsCompatibility:
    MANIPULATIONS = (
        ("original", SampleLabel.REAL),
        ("DeepFakes", SampleLabel.FAKE),
        ("Face2Face", SampleLabel.FAKE),
    )
    RECORDINGS = 4

    def _frames(self, *, cross_recording: bool = False) -> list[FrameSample]:
        frames: list[FrameSample] = []
        for recording in range(self.RECORDINGS):
            relative = f"0{recording:03d}/clip.mp4"
            for manipulation, label in self.MANIPULATIONS:
                path = f"{manipulation}/{relative}"
                video_id = derive_source_video_id("ffpp", path)
                recording_id = derive_source_recording_id(
                    "ffpp", path, layout=Layout.DROP_FIRST_SEGMENT
                )
                frames.append(
                    FrameSample(
                        frame_id=derive_frame_id(video_id, 0),
                        source_video_id=video_id,
                        source_recording_id=recording_id,
                        frame_index=0,
                        frame_path=f"{video_id}/frame_000000.png",
                        dataset_name="ffpp",
                        label=label,
                        source_video_sha256=PARENT_SHA,
                        license="research-only",
                        # Within a recording every manipulation shares a digest; optionally
                        # the first two recordings also share one, linking them.
                        frame_sha256=(
                            "11" * 32 if cross_recording and recording < 2 else f"{recording:064x}"
                        ),
                    )
                )
        return frames

    def test_ffpp_manipulations_of_one_recording_are_not_merged(self):
        frames = self._frames()
        report = find_exact_duplicates(frames)
        # Sharing content across manipulations of one recording is normal and creates no
        # cross-recording links, so coalescing changes nothing.
        assert report.cross_recording == ()
        frames_list = frames
        groups = group_frames_by_source_video(frames_list).groups
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        assert len(coalesced.groups) == self.RECORDINGS
        assert coalesced.merged_count == 0

    def test_ffpp_layout_with_cross_recording_duplicate_merges_only_those_recordings(self):
        frames = self._frames(cross_recording=True)
        report = find_exact_duplicates(frames)
        assert len(report.cross_recording) == 1
        groups = group_frames_by_source_video(frames).groups
        coalesced = coalesce_duplicate_linked_groups(groups, report)
        # Recordings 0 and 1 merge; 2 and 3 stay as they were.
        assert len(coalesced.groups) == self.RECORDINGS - 1
        result = build_frame_split(
            coalesced.groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH
        )
        assert result.leakage.is_leak_free


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class DuplicatingFrameWriter:
    """Writes distinct bytes per frame, except identical bytes for chosen sources.

    The identical-sources list keys on the source-video directory name, so two videos
    there produce byte-identical frames — an exact copy that duplicate detection must
    catch even under different recording ids.
    """

    format = "png"

    def __init__(self, identical_sources: tuple[str, str] | None = None) -> None:
        self.identical_sources = list(identical_sources or ())

    def write(self, image: object, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        source = path.parent.name
        if source in self.identical_sources:
            payload = b"DUPLICATE-FRAME-BYTES"
        else:
            payload = f"{source}/{path.name}".encode()
        path.write_bytes(payload)


def build_corpus(root: Path, *, recordings: int = 3) -> list[MediaSample]:
    samples: list[MediaSample] = []
    manipulations = (("original", SampleLabel.REAL), ("DeepFakes", SampleLabel.FAKE))
    for index in range(recordings):
        for manipulation, label in manipulations:
            relative = f"{manipulation}/{index:03d}/clip.mp4"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{manipulation}-{index}".encode() * 64)
            samples.append(
                MediaSample(
                    sample_id=f"ffpp::{manipulation}::{index:03d}",
                    dataset_name="ffpp",
                    filepath=relative,
                    modality=Modality.VIDEO,
                    label=label,
                    sha256=calculate_file_sha256(path),
                    license="research-only",
                    source="synthetic",
                )
            )
    return samples


def build_pipeline(
    decoder: FakeVideoDecoder,
    writer: object,
    *,
    base: Path,
    recordings: int = 3,
    dry_run: bool = False,
) -> PreprocessingPipeline:
    for index in range(recordings):
        for manipulation, _, _ in (
            ("original", SampleLabel.REAL, None),
            ("DeepFakes", SampleLabel.FAKE, None),
        ):
            decoder.add(
                base / manipulation / f"{index:03d}" / "clip.mp4",
                FakeVideo(frame_count=20, width=64, height=48),
            )
    extractor = VideoFrameExtractor(
        decoder, writer, ExtractionConfig(frames_per_video=3)  # type: ignore[arg-type]
    )
    return PreprocessingPipeline(
        MediaInspector(decoder),
        extractor,
        PreprocessingConfig(
            extraction=extractor.config,
            split=GroupSplitConfig(train_fraction=0.4, validation_fraction=0.3),
            source_identity_layout=SourceIdentityLayout.DROP_FIRST_SEGMENT,
            count_frames=True,
            dry_run=dry_run,
        ),
    )


class TestPipelineIntegration:
    def test_cross_recording_duplicate_is_merged_into_one_split(self, tmp_path: Path):
        base = tmp_path / "media"
        base.mkdir()
        samples = build_corpus(base)
        # original/000 and original/001 produce byte-identical frames: an exact duplicate
        # between two different recordings.
        writer = DuplicatingFrameWriter(
            identical_sources=("ffpp__original__000__clip", "ffpp__original__001__clip")
        )
        decoder = FakeVideoDecoder()
        pipeline = build_pipeline(decoder, writer, base=base)

        out = tmp_path / "out"
        result = pipeline.run(samples, base, out)

        assert result.is_leak_free
        assert result.duplicates is not None
        assert len(result.duplicates.cross_recording) == 1
        assert result.duplicates.cross_recording[0].digest == hashlib.sha256(
            b"DUPLICATE-FRAME-BYTES"
        ).hexdigest()
        # The two recordings became one cluster, so the split assigns the merged group.
        assert any(key.startswith("dup::") for key in result.split_result.group_assignment)
        assert result.frozen_split.extra["duplicate_coalescing"]["merged_components"]
        assert "linked by exact duplicate content" in result.frozen_split.rationale
        assert "exact-duplicate detection found" in " ".join(result.warnings)
        assert (out / "duplicates.json").is_file()
        report = json.loads((out / "duplicates.json").read_text(encoding="utf-8"))
        assert report["cross_recording_findings"] == 1

    def test_clean_corpus_is_untouched_and_reported(self, tmp_path: Path):
        base = tmp_path / "media"
        base.mkdir()
        samples = build_corpus(base)
        writer = DuplicatingFrameWriter()
        pipeline = build_pipeline(FakeVideoDecoder(), writer, base=base)

        result = pipeline.run(samples, base, tmp_path / "out")

        assert result.is_leak_free
        assert result.duplicates is not None
        assert result.duplicates.findings == ()
        assert result.duplicates.is_clean
        # Distinct frame bytes -> no duplicate findings -> no coalescing, groups intact.
        # Grouping merges original+forgery of one recording, so 3 groups for 6 videos.
        assert len(result.groups) == 3
        assert not any(key.startswith("dup::") for key in result.split_result.group_assignment)
        assert not any("duplicate" in warning.lower() for warning in result.warnings)

    def test_dry_run_reports_unverifiable_digests(self, tmp_path: Path):
        base = tmp_path / "media"
        base.mkdir()
        samples = build_corpus(base, recordings=2)
        from datasets.preprocessing.extraction import NullFrameWriter

        pipeline = build_pipeline(
            FakeVideoDecoder(), NullFrameWriter(), base=base, recordings=2, dry_run=True
        )

        out = tmp_path / "out"
        result = pipeline.run(samples, base, out)

        # Dry-run frames have no digest, so duplicates cannot be found — and that is
        # reported explicitly rather than assumed clean.
        assert result.duplicates is not None
        assert result.duplicates.findings == ()
        assert result.duplicates.unverifiable_count == len(result.frames)
        assert result.duplicates.is_clean is False
        assert (out / "duplicates.json").is_file()
        assert any("no usable frame digest" in warning for warning in result.warnings)

    def test_duplicate_report_round_trips_through_artifacts(self, tmp_path: Path):
        base = tmp_path / "media"
        base.mkdir()
        samples = build_corpus(base, recordings=2)
        writer = DuplicatingFrameWriter(
            identical_sources=("ffpp__original__000__clip", "ffpp__original__001__clip")
        )
        pipeline = build_pipeline(FakeVideoDecoder(), writer, base=base, recordings=2)

        first = pipeline.run(samples, base, tmp_path / "a")
        second = pipeline.run(samples, base, tmp_path / "b")

        assert first.frozen_split.content_hash() == second.frozen_split.content_hash()
        assert (tmp_path / "a" / "duplicates.json").read_bytes() == (
            tmp_path / "b" / "duplicates.json"
        ).read_bytes()
