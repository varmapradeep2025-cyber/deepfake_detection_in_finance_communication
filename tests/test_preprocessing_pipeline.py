"""End-to-end preprocessing: manifest in, artifacts out.

Driven with a fake decoder and, for the frame-writing paths, a real image writer over
numpy arrays. No fixture videos on disk, so these tests are fast and deterministic.

The behaviours pinned here are the ones that would otherwise fail silently:

* a frame manifest contains only frames whose bytes exist and are hashed;
* a video that failed inspection appears in ``inspection.json``;
* a dry run writes no manifest and no split;
* the split is leak-free and names the layout that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from datasets.manifest import (
    calculate_file_sha256,
    calculate_manifest_hash,
    load_manifest,
)
from datasets.preprocessing.extraction import (
    ExtractionConfig,
    NullFrameWriter,
    SamplingStrategy,
    VideoFrameExtractor,
)
from datasets.preprocessing.frames import (
    FrameManifestError,
    SourceIdentityLayout,
    frames_to_media_samples,
)
from datasets.preprocessing.inspection import MediaInspector, default_decoder
from datasets.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
    PreprocessingRunError,
)
from datasets.preprocessing.quality import QualityPolicy
from datasets.preprocessing.splitting import GroupSplitConfig
from datasets.schema import MediaSample, Modality, SampleLabel
from datasets.split_schema import load_split_definition
from tests.synthetic_preprocessing import (
    FakeVideo,
    FakeVideoDecoder,
    textured_statistics,
)

#: Enough recordings for a 70/20/10 group split to populate all three.
RECORDINGS = 10
#: ``(directory, label, manipulation)`` per manipulation of a recording. The genuine clip
#: carries no manipulation, matching the reviewed FF++ adapter.
MANIPULATIONS = (
    ("original", SampleLabel.REAL, None),
    ("DeepFakes", SampleLabel.FAKE, "synthetic_media"),
)
FRAMES_PER_VIDEO = 4


class InMemoryFrameWriter:
    """Writes frame images into a dict, for tests that need bytes but not a codec.

    Stands in for :class:`~datasets.preprocessing.extraction.OpenCVFrameWriter` so these
    tests assert pipeline behaviour rather than PNG encoding. The real writer is exercised
    in ``test_preprocessing_sampling.py``.

    Each frame's bytes are derived from its output path, so no two frames share a digest —
    the manifest is realistic, and exact-duplicate detection over these corpora reports
    nothing unless a test deliberately injects duplicated content. A writer whose every
    output byte was identical would make every frame a duplicate and collapse the whole
    corpus into one group.
    """

    format = "png"

    def __init__(self) -> None:
        self.written: dict[str, bytes] = {}

    def write(self, image: object, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Distinct per (source video, frame index): the digest must differ between
        # frames exactly as a real extraction's would.
        payload = f"{path.parent.name}/{path.name}".encode()
        path.write_bytes(payload)
        self.written[str(path)] = payload


def build_corpus(root: Path, *, recordings: int = RECORDINGS) -> list[MediaSample]:
    """A FaceForensics++-shaped manifest: ``<manip>/<recording>/clip.mp4``.

    Each manipulation of a recording is a separate file with a separate digest, which is
    what makes it a distinct source video while still belonging to one recording group.
    """
    samples: list[MediaSample] = []
    for index in range(recordings):
        for manipulation, label, manipulation_type in MANIPULATIONS:
            relative = f"{manipulation}/{index:03d}/clip.mp4"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Distinct bytes per file, so digests differ as they would for real media.
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
                    manipulation_type=manipulation_type,
                    source="synthetic",
                )
            )
    return samples


def register_corpus(
    decoder: FakeVideoDecoder, base: Path, *, recordings: int = RECORDINGS, **kwargs: Any
) -> None:
    """Register every clip of a corpus with ``base``-absolute paths.

    The pipeline resolves manifest paths against ``base`` before inspecting, so the
    decoder must be keyed by the resolved path. Registering relative paths would leave
    every lookup to the bare-stem fallback, where the last registration wins and a
    one-file difference in behaviour goes unnoticed.
    """
    for index in range(recordings):
        for manipulation, _, _ in MANIPULATIONS:
            decoder.add(base / manipulation / f"{index:03d}" / "clip.mp4", FakeVideo(**kwargs))


def build_pipeline(
    corpus_root: Path,
    *,
    frames_per_video: int = FRAMES_PER_VIDEO,
    layout: SourceIdentityLayout = SourceIdentityLayout.DROP_FIRST_SEGMENT,
    writer: object | None = None,
    quality_policy: QualityPolicy | None = None,
) -> PreprocessingPipeline:
    decoder = FakeVideoDecoder()
    register_corpus(decoder, corpus_root, frame_count=20, width=64, height=48)
    extractor = VideoFrameExtractor(
        decoder,
        NullFrameWriter() if writer is None else writer,  # type: ignore[arg-type]
        ExtractionConfig(
            frames_per_video=frames_per_video,
            strategy=SamplingStrategy.UNIFORM,
            quality_policy=quality_policy,
        ),
    )
    config = PreprocessingConfig(
        extraction=extractor.config,
        split=GroupSplitConfig(),
        source_identity_layout=layout,
        count_frames=True,
        dry_run=writer is None,
    )
    return PreprocessingPipeline(MediaInspector(decoder), extractor, config)


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, list[MediaSample]]:
    base = tmp_path / "media"
    base.mkdir()
    return base, build_corpus(base)


class TestSelectionAndGates:
    def test_audio_rows_are_never_preprocessed(self, corpus, tmp_path):
        base, samples = corpus
        audio = MediaSample(
            sample_id="ffpp::audio::000",
            dataset_name="ffpp",
            filepath="audio/000.wav",
            modality=Modality.AUDIO,
            label=SampleLabel.REAL,
            sha256="a" * 64,
            license="research-only",
        )
        pipeline = build_pipeline(base)

        selected = pipeline.select_videos([*samples, audio])

        assert audio not in selected
        assert len(selected) == len(samples)

    def test_blocked_licence_is_excluded_by_name(self, corpus):
        base, samples = corpus
        pipeline = build_pipeline(base)
        pipeline.config = PreprocessingConfig(
            extraction=pipeline.extractor.config,
            license_filter=("research-only",),
        )

        with pytest.raises(PreprocessingRunError, match="licence"):
            pipeline.select_videos(samples)

    def test_an_empty_manifest_says_so(self, corpus):
        base, _ = corpus
        pipeline = build_pipeline(base)

        with pytest.raises(PreprocessingRunError, match="no usable video samples"):
            pipeline.select_videos([])

    def test_failed_inspection_is_recorded_not_dropped(self, tmp_path):
        # The whole point of InspectionBatch: a lost video must still be visible.
        base = tmp_path / "media"
        base.mkdir()
        samples = build_corpus(base, recordings=2)
        decoder = FakeVideoDecoder()
        for index in range(2):
            for manipulation, _, _ in MANIPULATIONS:
                decoder.add(
                    base / manipulation / f"{index:03d}" / "clip.mp4",
                    FakeVideo(frame_count=8, undecodable=(index == 1)),
                )
        pipeline = PreprocessingPipeline(
            MediaInspector(decoder),
            VideoFrameExtractor(decoder, NullFrameWriter(), ExtractionConfig(frames_per_video=2)),
            PreprocessingConfig(count_frames=False, dry_run=True),
        )

        batch = pipeline.inspect_videos(samples, base)

        assert len(batch.records) == len(samples)
        assert batch.failure_count == 2
        assert all("clip.mp4" in record.error for record in batch.failures)

    def test_changed_media_is_refused_when_verification_is_on(self, corpus):
        # A file that changed since the manifest was written would otherwise be extracted
        # under a digest that no longer describes it, quietly invalidating every frame
        # derived from it.
        base, samples = corpus
        pipeline = build_pipeline(base)
        tampered = MediaSample(
            sample_id=samples[0].sample_id,
            dataset_name=samples[0].dataset_name,
            filepath=samples[0].filepath,
            modality=samples[0].modality,
            label=samples[0].label,
            sha256="f" * 64,
            license=samples[0].license,
        )
        pipeline.config = PreprocessingConfig(
            extraction=pipeline.extractor.config,
            count_frames=False,
            verify_source_digests=True,
            dry_run=True,
        )

        with pytest.raises(PreprocessingRunError, match="does not match the manifest"):
            pipeline.inspect_videos([tampered], base)

    def test_verification_is_off_by_default(self, corpus):
        # Re-hashing a whole corpus is expensive, so it has to be asked for.
        base, samples = corpus
        pipeline = build_pipeline(base)
        batch = pipeline.inspect_videos(samples[:1], base)
        assert batch.usable


class TestDryRun:
    def test_dry_run_writes_no_frame_manifest(self, corpus, tmp_path):
        base, samples = corpus
        out = tmp_path / "out"
        pipeline = build_pipeline(base)

        result = pipeline.run(samples, base, out)

        assert not (out / "frames.jsonl").exists()
        assert not (out / "split.json").exists()
        assert result.frames
        assert result.split_result is not None
        assert any("dry run" in warning for warning in result.warnings)

    def test_dry_run_still_writes_reports(self, corpus, tmp_path):
        base, samples = corpus
        out = tmp_path / "out"
        pipeline = build_pipeline(base)

        result = pipeline.run(samples, base, out)

        assert (out / "inspection.json").is_file()
        assert (out / "groups.json").is_file()
        assert "frames_manifest" not in result.artifacts
        assert "split" not in result.artifacts

    def test_dry_run_frames_carry_no_digest(self, corpus, tmp_path):
        # Nothing was written, so there is nothing to hash. Inventing a digest would let a
        # later verification pass pass without checking a byte.
        base, samples = corpus
        pipeline = build_pipeline(base)

        result = pipeline.run(samples, base, tmp_path / "out")

        assert all(frame.frame_sha256 is None for frame in result.frames)

    def test_a_frame_record_without_a_digest_cannot_be_written(self, corpus):
        base, samples = corpus
        pipeline = build_pipeline(base)
        result = pipeline.run(samples, base, Path("out"))
        frame = result.frames[0]
        assert frame.frame_sha256 is None

        with pytest.raises(FrameManifestError, match="digest"):
            frame.to_media_sample()

    def test_write_artifacts_false_touches_nothing(self, corpus, tmp_path):
        base, samples = corpus
        out = tmp_path / "out"
        pipeline = build_pipeline(base)

        result = pipeline.run(samples, base, out, write_artifacts=False)

        assert not out.exists()
        assert result.frames


class TestFullRunArtifacts:
    @pytest.fixture()
    def written(self, corpus, tmp_path):
        base, samples = corpus
        out = tmp_path / "out"
        writer = InMemoryFrameWriter()
        pipeline = build_pipeline(base, writer=writer)

        result = pipeline.run(samples, base, out)

        return result, out, writer

    def test_frame_manifest_is_written(self, written):
        result, out, _ = written
        assert (out / "frames.jsonl").is_file()
        assert result.artifacts["frames_manifest"].name == "frames.jsonl"

    def test_every_frame_digest_matches_the_bytes_on_disk(self, written):
        # The load-bearing integrity claim: the manifest's digest is a fact about a file.
        result, out, _ = written
        records = load_manifest(out / "frames.jsonl")
        for frame in result.frames:
            assert frame.frame_sha256 is not None
            assert calculate_file_sha256(out / "frames" / frame.frame_path) == (frame.frame_sha256)
        assert len(records) == len(result.frames)

    def test_manifest_round_trips_through_the_frame_schema(self, written):
        result, out, _ = written
        from datasets.preprocessing.frames import FrameSample

        reloaded = [FrameSample.from_media_sample(r) for r in load_manifest(out / "frames.jsonl")]

        # Keyed by id: write_manifest sorts, the run emits in corpus order.
        assert {f.frame_id: f for f in reloaded} == {f.frame_id: f for f in result.frames}

    def test_frame_paths_live_under_the_frame_root(self, written):
        result, out, _ = written
        for frame in result.frames:
            assert (out / "frames" / frame.frame_path).is_file()

    def test_split_is_written_and_loads(self, written):
        result, out, _ = written
        assert (out / "split.json").is_file()
        split = load_split_definition(out / "split.json")
        assert split.content_hash() == result.frozen_split.content_hash()
        assert split.source_manifest_hash == result.frozen_split.source_manifest_hash

    def test_split_pins_the_frame_manifest(self, written):
        result, _, _ = written
        expected = calculate_manifest_hash(frames_to_media_samples(result.frames))
        assert result.frozen_split.source_manifest_hash == expected

    def test_split_covers_exactly_the_kept_frames(self, written):
        result, _, _ = written
        split = result.frozen_split
        assigned = [
            frame_id for name in ("train", "validation", "test") for frame_id in split.ids_for(name)
        ]
        assert sorted(assigned) == sorted(frame.frame_id for frame in result.frames)

    def test_split_is_leak_free(self, written):
        result, _, _ = written
        assert result.is_leak_free

    def test_split_records_the_layout_that_produced_it(self, written):
        result, _, _ = written
        assert result.frozen_split.extra["source_identity_layout"] == (
            SourceIdentityLayout.DROP_FIRST_SEGMENT.value
        )

    def test_inspection_report_lists_every_video(self, written):
        result, out, _ = written
        report = json.loads((out / "inspection.json").read_text(encoding="utf-8"))
        assert len(report["videos"]) == len(result.inspections)

    def test_groups_report_has_no_conflicts(self, written):
        _, out, _ = written
        groups = json.loads((out / "groups.json").read_text(encoding="utf-8"))
        assert groups["video_id_reuse"] == {}

    def test_preprocessing_report_summarises_the_run(self, written):
        result, out, _ = written
        report = json.loads((out / "preprocessing_report.json").read_text(encoding="utf-8"))
        assert report["summary"]
        assert report["frame_manifest_hash"] == result.frozen_split.source_manifest_hash
        assert report["quality"]["total"] == len(result.frames) + len(result.rejected)

    def test_rerunning_produces_identical_reports(self, corpus, tmp_path):
        base, samples = corpus
        first_out, second_out = tmp_path / "a", tmp_path / "b"
        for out in (first_out, second_out):
            pipeline = build_pipeline(base, writer=InMemoryFrameWriter())
            pipeline.run(samples, base, out)

        for name in ("inspection.json", "groups.json"):
            assert (first_out / name).read_bytes() == (second_out / name).read_bytes()

    def test_every_frame_links_to_its_recording(self, written):
        result, _, _ = written
        for frame in result.frames:
            assert frame.source_video_id
            assert frame.source_recording_id
            assert frame.source_video_sha256


class TestLeakageEndToEnd:
    def test_a_clip_and_its_forgery_share_a_split(self, corpus, tmp_path):
        base, samples = corpus
        pipeline = build_pipeline(base, writer=InMemoryFrameWriter())

        result = pipeline.run(samples, base, tmp_path / "out")

        assignment = result.split_result.frame_assignment
        by_recording: dict[str, set[str]] = {}
        for frame in result.frames:
            by_recording.setdefault(frame.source_recording_id, set()).add(
                assignment[frame.frame_id]
            )
        assert all(len(splits) == 1 for splits in by_recording.values())

    def test_per_file_layout_produces_three_times_the_groups(self, corpus, tmp_path):
        base, samples = corpus
        pipeline = build_pipeline(
            base,
            layout=SourceIdentityLayout.FULL_RELATIVE_PATH,
            writer=InMemoryFrameWriter(),
        )

        result = pipeline.run(samples, base, tmp_path / "out")

        assert len(result.groups) == RECORDINGS * len(MANIPULATIONS)

    def test_split_proportions_follow_the_configured_fractions(self, corpus, tmp_path):
        # Group counts, not frame counts: every group here holds the same number of frames,
        # so the two agree and the fraction is checkable exactly.
        base, samples = corpus
        pipeline = build_pipeline(base, writer=InMemoryFrameWriter())

        result = pipeline.run(samples, base, tmp_path / "out")

        counts = result.split_result.counts()
        total = sum(counts.values())
        config = result.config.split
        assert counts["train"] / total == pytest.approx(config.train_fraction, abs=0.1)
        assert counts["test"] > 0
        assert counts["validation"] > 0

    def test_explicit_fractions_are_respected(self, corpus, tmp_path):
        base, samples = corpus
        pipeline = build_pipeline(base, writer=InMemoryFrameWriter())
        pipeline.config = PreprocessingConfig(
            extraction=pipeline.extractor.config,
            split=GroupSplitConfig(train_fraction=0.7, validation_fraction=0.2),
            source_identity_layout=SourceIdentityLayout.DROP_FIRST_SEGMENT,
            dry_run=False,
        )

        result = pipeline.run(samples, base, tmp_path / "out")

        counts = result.split_result.counts()
        total = sum(counts.values())
        assert counts["train"] / total == pytest.approx(0.7, abs=0.02)
        assert counts["validation"] / total == pytest.approx(0.2, abs=0.02)

    def test_a_single_recording_still_yields_a_usable_split(self, corpus, tmp_path):
        # A stratum smaller than 1/train_fraction would otherwise get no training data at
        # all, leaving a corpus with nothing to train on.
        base = tmp_path / "one"
        base.mkdir()
        samples = build_corpus(base, recordings=1)
        decoder = FakeVideoDecoder()
        register_corpus(decoder, base, recordings=1, frame_count=10)
        extractor = VideoFrameExtractor(
            decoder,
            NullFrameWriter(),
            ExtractionConfig(frames_per_video=2),
        )
        pipeline = PreprocessingPipeline(
            MediaInspector(decoder),
            extractor,
            PreprocessingConfig(
                extraction=extractor.config,
                source_identity_layout=SourceIdentityLayout.DROP_FIRST_SEGMENT,
                count_frames=False,
                dry_run=True,
            ),
        )

        result = pipeline.run(samples, base, tmp_path / "out")

        assert result.split_result.counts()["train"] > 0
        assert result.is_leak_free


class TestQualityReporting:
    def test_summary_uses_the_configured_policy(self, corpus, tmp_path):
        # Reporting under the default thresholds would mislabel a stricter run.
        base, samples = corpus
        policy = QualityPolicy(reject_uniform=False, min_mean_intensity=200.0)
        pipeline = build_pipeline(base, writer=InMemoryFrameWriter(), quality_policy=policy)

        result = pipeline.run(samples, base, tmp_path / "out")

        assert result.quality.policy == policy

    def test_rejected_frames_are_kept_out_of_the_manifest(self, corpus, tmp_path):
        base, samples = corpus
        pipeline = build_pipeline(base, writer=InMemoryFrameWriter())

        result = pipeline.run(samples, base, tmp_path / "out")

        manifest_ids = {
            record.sample_id for record in load_manifest(tmp_path / "out" / "frames.jsonl")
        }
        assert manifest_ids.isdisjoint({frame.frame_id for frame in result.rejected})

    def test_rejected_frames_are_written_with_reasons(self, corpus, tmp_path):
        base, samples = corpus
        pipeline = build_pipeline(
            base,
            writer=InMemoryFrameWriter(),
            quality_policy=QualityPolicy(reject_uniform=True),
        )

        result = pipeline.run(samples, base, tmp_path / "out")

        report = json.loads((tmp_path / "out" / "rejected_frames.json").read_text(encoding="utf-8"))
        assert report["rejected_count"] == len(result.rejected)
        assert all("quality_reasons" in frame for frame in report["frames"])


class TestDeterminism:
    def test_same_config_yields_the_same_frame_ids(self, corpus, tmp_path):
        base, samples = corpus
        first = build_pipeline(base, writer=InMemoryFrameWriter()).run(
            samples, base, tmp_path / "a"
        )
        second = build_pipeline(base, writer=InMemoryFrameWriter()).run(
            samples, base, tmp_path / "b"
        )

        assert [f.frame_id for f in first.frames] == [f.frame_id for f in second.frames]
        assert first.frozen_split.content_hash() == second.frozen_split.content_hash()

    def test_selection_is_independent_of_manifest_order(self, corpus, tmp_path):
        base, samples = corpus
        forward = build_pipeline(base, writer=InMemoryFrameWriter()).run(
            samples, base, tmp_path / "a"
        )
        reversed_run = build_pipeline(base, writer=InMemoryFrameWriter()).run(
            list(reversed(samples)), base, tmp_path / "b"
        )

        assert {f.frame_id for f in forward.frames} == {f.frame_id for f in reversed_run.frames}


class TestRealMediaEndToEnd:
    """The whole chain on a real encoded file.

    Everything else here fakes either the decoder or the writer. This class uses the real
    OpenCV decoder *and* the real PNG writer, so it proves the pieces compose: a real
    container decodes, frames are really written, their digests are really computed, and
    the manifest they produce is really valid. Skipped when OpenCV is absent.
    """

    def test_a_real_video_flows_through_to_a_valid_manifest(self, tmp_path):
        from tests.synthetic_preprocessing import opencv_available, write_real_video

        if not opencv_available():
            pytest.skip("opencv-python-headless is not installed")

        from datasets.preprocessing.extraction import OpenCVFrameWriter

        base = tmp_path / "media"
        (base / "original" / "000").mkdir(parents=True)
        (base / "DeepFakes" / "000").mkdir(parents=True)
        samples = []
        for manipulation, label, manipulation_type in MANIPULATIONS:
            relative = f"{manipulation}/000/clip.mp4"
            path = base / relative
            # Distinct frame counts and a gradient that advances per frame, so a wrong
            # frame index would be detectable rather than looking identical.
            write_real_video(path, frame_count=20, width=96, height=72)
            samples.append(
                MediaSample(
                    sample_id=f"ffpp::{manipulation}::000",
                    dataset_name="ffpp",
                    filepath=relative,
                    modality=Modality.VIDEO,
                    label=label,
                    sha256=calculate_file_sha256(path),
                    license="research-only",
                    manipulation_type=manipulation_type,
                    source="synthetic",
                )
            )

        decoder = default_decoder()
        extractor = VideoFrameExtractor(
            decoder,
            OpenCVFrameWriter("png"),
            ExtractionConfig(frames_per_video=4),
        )
        pipeline = PreprocessingPipeline(
            MediaInspector(decoder),
            extractor,
            PreprocessingConfig(
                extraction=extractor.config,
                source_identity_layout=SourceIdentityLayout.DROP_FIRST_SEGMENT,
                count_frames=True,
            ),
        )

        result = pipeline.run(samples, base, tmp_path / "out")

        assert len(result.frames) == 8
        assert result.quality.rejected == 0
        for frame in result.frames:
            assert frame.frame_sha256 is not None
            written = tmp_path / "out" / "frames" / frame.frame_path
            assert written.is_file()
            # The manifest's digest is the file's digest, verified against the real bytes.
            assert calculate_file_sha256(written) == frame.frame_sha256

        # The frame manifest round-trips through the shared schema and loader.
        records = load_manifest(tmp_path / "out" / "frames.jsonl")
        assert len(records) == 8
        assert {r.sample_id for r in records} == {f.frame_id for f in result.frames}

        split = load_split_definition(tmp_path / "out" / "split.json")
        assert split.source_manifest_hash == result.frozen_split.source_manifest_hash

    def test_written_frames_are_identical_across_runs(self, tmp_path):
        from tests.synthetic_preprocessing import opencv_available, write_real_video

        if not opencv_available():
            pytest.skip("opencv-python-headless is not installed")

        from datasets.preprocessing.extraction import OpenCVFrameWriter

        base = tmp_path / "media"
        base.mkdir()
        path = base / "clip.mp4"
        write_real_video(path, frame_count=12)
        sample = MediaSample(
            sample_id="ffpp::clip::000",
            dataset_name="ffpp",
            filepath="clip.mp4",
            modality=Modality.VIDEO,
            label=SampleLabel.REAL,
            sha256=calculate_file_sha256(path),
            license="research-only",
            source="synthetic",
        )

        runs = []
        for name in ("a", "b"):
            decoder = default_decoder()
            extraction = ExtractionConfig(frames_per_video=3)
            extractor = VideoFrameExtractor(decoder, OpenCVFrameWriter("png"), extraction)
            pipeline = PreprocessingPipeline(
                MediaInspector(decoder),
                extractor,
                PreprocessingConfig(extraction=extraction, dry_run=True),
            )
            result = pipeline.run([sample], base, tmp_path / name, write_artifacts=False)
            runs.append([f.frame_id for f in result.frames])

        assert runs[0] == runs[1]


class TestStatisticsAreCarriedThrough:
    def test_frame_records_carry_dimensions_and_statistics(self, corpus, tmp_path):
        base, samples = corpus
        pipeline = build_pipeline(base, writer=InMemoryFrameWriter())

        result = pipeline.run(samples, base, tmp_path / "out")

        frame = result.frames[0]
        assert frame.width == 64
        assert frame.height == 48
        assert frame.statistics["mean_intensity"] == pytest.approx(120.0)
        assert frame.sampling["requested"] == FRAMES_PER_VIDEO
        assert frame.sampling["frame_count_source"] == "observed"

    def test_statistics_helper_is_a_sane_fixture(self):
        stats = textured_statistics()
        assert stats.width == 64
        assert not stats.is_uniform
