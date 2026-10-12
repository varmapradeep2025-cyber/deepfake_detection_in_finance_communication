"""Media inspection and frame-quality filtering.

Two pure stages, so both are testable without a codec or an image file:

* :mod:`datasets.preprocessing.inspection` answers "what is in this file, and can it be
  decoded?" Counting frames is separated from the rest, because it costs a full decode.
* :mod:`datasets.preprocessing.quality` answers "should this frame be kept?" with
  thresholds that are conservative by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from datasets.manifest import calculate_file_sha256
from datasets.preprocessing.inspection import (
    InspectionStatus,
    MediaInspector,
    VideoDecoderUnavailableError,
    default_decoder,
    is_video_candidate,
)
from datasets.preprocessing.quality import (
    DEFAULT_QUALITY_POLICY,
    FrameQualityFilter,
    QualityPolicy,
    QualityReason,
)
from datasets.schema import Modality, SampleLabel
from tests.synthetic_preprocessing import (
    FakeVideo,
    FakeVideoDecoder,
    dark_statistics,
    opencv_available,
    small_statistics,
    textured_statistics,
    uniform_statistics,
    write_real_video,
)

PLACEHOLDER_BYTES = b"synthetic-placeholder-bytes" * 8


def register(decoder: FakeVideoDecoder, path: Path, **kwargs: Any) -> Path:
    """Create a placeholder file and serve it from a fake decoder."""
    path.write_bytes(PLACEHOLDER_BYTES)
    decoder.add(path, FakeVideo(**kwargs))
    return path


@pytest.fixture()
def decoder() -> FakeVideoDecoder:
    return FakeVideoDecoder({"clip": FakeVideo(frame_count=10)})


@pytest.fixture()
def inspector(decoder: FakeVideoDecoder) -> MediaInspector:
    return MediaInspector(decoder)


requires_opencv = pytest.mark.skipif(
    not opencv_available(), reason="opencv-python-headless is not installed"
)


class TestProbeFailures:
    def test_undecodable_file_is_reported_not_raised(self, decoder, inspector, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=1, undecodable=True)

        record = inspector.inspect(path)

        assert record.status is InspectionStatus.DECODE_FAILED
        assert not record.is_usable
        # Still identified, so the failure is traceable to exact bytes.
        assert record.sha256 == calculate_file_sha256(path)
        assert "clip.mp4" in (record.error or "")

    def test_missing_file_names_the_path(self, inspector, tmp_path):
        missing = tmp_path / "absent.mp4"

        record = inspector.inspect(missing)

        assert record.status is InspectionStatus.FILE_UNREADABLE
        assert not record.is_usable
        assert record.sha256 is None
        assert "absent.mp4" in (record.error or "")

    def test_empty_file_is_not_reported_as_a_good_video(self, inspector, tmp_path):
        path = tmp_path / "empty.mp4"
        path.write_bytes(b"")
        record = inspector.inspect(path)
        assert not record.is_usable

    def test_to_dict_round_trips_through_json(self, decoder, inspector, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=8)
        record = inspector.inspect(path, count_frames=True)
        assert json.loads(json.dumps(record.to_dict())) == record.to_dict()

    def test_failed_record_still_carries_what_was_known(self, decoder, inspector, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=4, undecodable=True)
        record = inspector.inspect(path)
        # Size and digest are knowable without decoding, and are exactly what you need to
        # decide whether the file is worth re-acquiring.
        assert record.file_size == path.stat().st_size
        assert record.sha256 is not None


class TestStreamProperties:
    def test_probe_values_are_carried_through(self, decoder, inspector, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=20, width=320, height=240)
        record = inspector.inspect(path)
        assert record.status is InspectionStatus.OK
        assert (record.width, record.height) == (320, 240)
        assert record.codec == "fake_codec"
        assert record.is_usable

    def test_frame_count_is_not_asserted_without_counting(self, decoder, inspector, tmp_path):
        # The declared count is metadata and may be wrong, so it is never quietly promoted
        # to an observed one.
        path = register(decoder, tmp_path / "clip.mp4", frame_count=10, declared_frame_count=999)
        record = inspector.inspect(path, count_frames=False)
        assert record.frame_count is None
        assert record.declared_frame_count == 999

    def test_counting_yields_the_observed_count(self, decoder, inspector, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=10, declared_frame_count=999)
        record = inspector.inspect(path, count_frames=True)
        assert record.frame_count == 10
        assert record.declared_frame_count == 999

    def test_mismatched_count_is_warned_about(self, decoder, inspector, tmp_path):
        # Sampling uses the observed count, so a silent disagreement would make the plan
        # unexplainable after the fact.
        path = register(decoder, tmp_path / "clip.mp4", frame_count=10, declared_frame_count=999)
        record = inspector.inspect(path, count_frames=True)
        assert any("999" in warning for warning in record.warnings)

    def test_agreeing_counts_produce_no_warning(self, decoder, inspector, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=10)
        record = inspector.inspect(path, count_frames=True)
        assert record.warnings == ()

    def test_counting_reads_the_file_exactly_once(self, decoder, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=5)
        decoder.read_calls.clear()

        MediaInspector(decoder).inspect(path, count_frames=True)

        assert len(decoder.read_calls) == 1

    def test_skipping_counting_reads_nothing(self, decoder, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=5)
        decoder.read_calls.clear()

        MediaInspector(decoder).inspect(path, count_frames=False)

        assert decoder.read_calls == []

    def test_counting_is_skipped_for_undecodable_files(self, decoder, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=3, undecodable=True)
        decoder.read_calls.clear()

        MediaInspector(decoder).inspect(path, count_frames=True)

        assert decoder.read_calls == []

    def test_hashing_can_be_switched_off(self, decoder, tmp_path):
        path = register(decoder, tmp_path / "clip.mp4", frame_count=5)

        record = MediaInspector(decoder, compute_hashes=False).inspect(path)

        assert record.sha256 is None
        assert record.file_size == path.stat().st_size


class TestMissingDecoder:
    def test_missing_backend_is_named_rather_than_crashing(self, tmp_path):
        class NoDecoder:
            name = "absent"

            def probe(self, path: Path) -> Any:
                raise VideoDecoderUnavailableError("no backend installed")

            def read_frames(self, path: Path, *, keep_images: bool = False) -> Any:
                raise VideoDecoderUnavailableError("no backend installed")
                yield  # pragma: no cover

        path = tmp_path / "clip.mp4"
        path.write_bytes(b"x" * 16)

        record = MediaInspector(NoDecoder()).inspect(path)

        assert not record.is_usable
        assert "absent" in (record.error or "")

    def test_a_none_decoder_does_not_crash(self, tmp_path):
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"x" * 16)
        record = MediaInspector(None).inspect(path)
        assert not record.is_usable
        assert record.sha256 == calculate_file_sha256(path)


class TestVideoCandidateDetection:
    def test_known_video_extensions_are_candidates(self):
        assert is_video_candidate(Path("clip.mp4"))
        assert is_video_candidate(Path("clip.MKV"))

    def test_unknown_extension_is_not_assumed_to_be_video(self):
        # Guessing would pull sidecar files into a decode pass.
        assert not is_video_candidate(Path("notes.txt"))
        assert not is_video_candidate(Path("labels.json"))


class TestRealCodecRoundTrip:
    """The one place a real encoder is involved.

    A fake decoder can only prove the pipeline's logic. These prove the decoder
    integration: a written file reads back with the frame count it was given, probe
    metadata is populated, and duration is a count over a rate.
    """

    @requires_opencv
    def test_written_video_decodes_with_its_declared_count(self, tmp_path):
        path = write_real_video(tmp_path / "real.mp4", frame_count=24)

        record = MediaInspector(default_decoder()).inspect(path, count_frames=True)

        assert record.status is InspectionStatus.OK
        assert record.frame_count == 24
        assert (record.width, record.height) == (64, 48)

    @requires_opencv
    def test_probe_reports_a_codec_and_a_container_separately(self, tmp_path):
        path = write_real_video(tmp_path / "real.mp4", frame_count=6)

        record = MediaInspector(default_decoder()).inspect(path)

        assert record.container_format == "mp4"
        assert record.codec
        # A FourCC filed as a container would claim a codec name is a file format.
        assert record.codec != record.container_format

    @requires_opencv
    def test_duration_is_frame_count_over_rate(self, tmp_path):
        path = write_real_video(tmp_path / "real.mp4", frame_count=30, frame_rate=30.0)

        record = MediaInspector(default_decoder()).inspect(path)

        assert record.frame_rate == pytest.approx(30.0, rel=0.01)
        assert record.duration_seconds == pytest.approx(1.0, abs=0.05)

    @requires_opencv
    def test_a_non_video_file_fails_inspection_but_keeps_its_digest(self, tmp_path):
        path = tmp_path / "notes.mp4"
        path.write_bytes(b"this is plain text, not a video")

        record = MediaInspector(default_decoder()).inspect(path)

        assert not record.is_usable
        assert record.sha256 == calculate_file_sha256(path)


class TestQualityPolicyDefaults:
    def test_default_policy_only_rejects_unusable_frames(self):
        # Deliberately permissive: rejecting on brightness or contrast by default would
        # discard legitimate footage and quietly shrink the corpus.
        assert DEFAULT_QUALITY_POLICY.min_mean_intensity is None
        assert DEFAULT_QUALITY_POLICY.max_mean_intensity is None
        assert DEFAULT_QUALITY_POLICY.min_std_intensity is None
        assert DEFAULT_QUALITY_POLICY.min_dynamic_range is None

    def test_default_minimum_size_is_not_optional(self):
        assert DEFAULT_QUALITY_POLICY.min_width == 32
        assert DEFAULT_QUALITY_POLICY.min_height == 32

    def test_dimensions_cannot_be_disabled(self):
        # A frame below the minimum cannot support the downstream analysis, so there is no
        # legitimate way to turn the check off.
        with pytest.raises(ValueError):
            QualityPolicy(min_width=0)

    def test_bounds_must_be_ordered(self):
        with pytest.raises(ValueError):
            QualityPolicy(min_mean_intensity=200.0, max_mean_intensity=10.0)

    def test_intensity_bounds_must_be_in_range(self):
        with pytest.raises(ValueError):
            QualityPolicy(min_mean_intensity=300.0)


class TestQualityEvaluation:
    def test_healthy_frame_is_kept(self):
        verdict = FrameQualityFilter().evaluate(textured_statistics())
        assert verdict.accepted
        assert verdict.reasons == ()

    def test_tiny_frame_is_rejected_for_size(self):
        verdict = FrameQualityFilter().evaluate(small_statistics(8, 8))
        assert not verdict.accepted
        assert QualityReason.BELOW_MIN_RESOLUTION in verdict.reasons

    def test_narrow_frame_is_rejected(self):
        verdict = FrameQualityFilter().evaluate(textured_statistics(width=64, height=16))
        assert not verdict.accepted
        assert QualityReason.BELOW_MIN_RESOLUTION in verdict.reasons

    def test_flat_frame_is_rejected_by_default(self):
        verdict = FrameQualityFilter().evaluate(uniform_statistics(value=90))
        assert not verdict.accepted
        assert QualityReason.UNIFORM in verdict.reasons

    def test_flat_check_can_be_disabled(self):
        verdict = FrameQualityFilter(QualityPolicy(reject_uniform=False)).evaluate(
            uniform_statistics(value=90)
        )
        assert verdict.accepted

    def test_bright_frame_is_rejected_when_enabled(self):
        verdict = FrameQualityFilter(QualityPolicy(max_mean_intensity=200.0)).evaluate(
            textured_statistics(mean=250.0)
        )
        assert not verdict.accepted
        assert QualityReason.TOO_BRIGHT in verdict.reasons

    def test_dark_frame_is_rejected_when_enabled(self):
        verdict = FrameQualityFilter(QualityPolicy(min_mean_intensity=20.0)).evaluate(
            dark_statistics(mean=2.0)
        )
        assert not verdict.accepted
        assert QualityReason.TOO_DARK in verdict.reasons

    def test_dark_frame_is_kept_by_default(self):
        # The default policy must not discard low-light footage, only broken frames.
        assert FrameQualityFilter().evaluate(dark_statistics(mean=2.0)).accepted

    def test_all_applicable_reasons_are_reported(self):
        # Stopping at the first failure would make the histogram depend on check order.
        verdict = FrameQualityFilter(
            QualityPolicy(reject_uniform=True, min_mean_intensity=200.0)
        ).evaluate(uniform_statistics(value=5))
        assert len(verdict.reasons) >= 2

    def test_thresholds_are_inclusive_at_the_boundary(self):
        filter_ = FrameQualityFilter(
            QualityPolicy(min_mean_intensity=100.0, max_mean_intensity=200.0)
        )
        assert filter_.evaluate(textured_statistics(mean=100.0)).accepted
        assert filter_.evaluate(textured_statistics(mean=200.0)).accepted
        assert not filter_.evaluate(textured_statistics(mean=201.0)).accepted

    def test_filter_is_stateless(self):
        filter_ = FrameQualityFilter()
        first = filter_.evaluate(textured_statistics())
        second = filter_.evaluate(textured_statistics())
        assert first == second


class TestQualitySummary:
    def test_summary_counts_match_the_verdicts(self):
        verdicts = [
            FrameQualityFilter().evaluate(textured_statistics()),
            FrameQualityFilter().evaluate(textured_statistics()),
            FrameQualityFilter().evaluate(uniform_statistics(value=5)),
        ]

        summary = FrameQualityFilter().summarize(verdicts)

        assert summary.total == 3
        assert summary.kept == 2
        assert summary.rejected == 1
        assert summary.kept_fraction == pytest.approx(2 / 3)

    def test_summary_of_nothing_is_empty_not_an_error(self):
        summary = FrameQualityFilter().summarize([])
        assert summary.total == 0
        assert summary.kept_fraction == 0.0

    def test_summary_records_the_policy_it_applied(self):
        policy = QualityPolicy(min_mean_intensity=50.0)
        filter_ = FrameQualityFilter(policy)

        summary = filter_.summarize([filter_.evaluate(textured_statistics())])

        assert summary.policy == policy

    def test_reason_counts_add_up(self):
        verdicts = [
            FrameQualityFilter().evaluate(uniform_statistics(value=5)),
            FrameQualityFilter().evaluate(small_statistics()),
        ]

        summary = FrameQualityFilter().summarize(verdicts)

        assert sum(summary.reason_counts.values()) == 2

    def test_summary_serialises(self):
        filter_ = FrameQualityFilter()
        summary = filter_.summarize([filter_.evaluate(textured_statistics())])
        assert json.loads(json.dumps(summary.to_dict())) == summary.to_dict()


class TestModalityIsExplicit:
    """The pipeline's first gate: only ``VIDEO`` rows are preprocessed."""

    def test_modalities_are_distinct_values(self):
        assert Modality.VIDEO.value == "video"
        assert Modality.AUDIO.value == "audio"
        assert Modality.IMAGE.value == "image"

    def test_labels_are_distinct_values(self):
        # Nothing in the schema infers a label from a filename, so an adapter that gets it
        # wrong cannot be silently corrected later in the pipeline.
        assert SampleLabel.REAL is not SampleLabel.FAKE
