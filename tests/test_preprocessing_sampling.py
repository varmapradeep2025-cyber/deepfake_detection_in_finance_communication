"""Tests for deterministic sampling, frame identity, and the strategy registry.

The properties asserted here are the ones every downstream guarantee rests on: the same
inputs must produce the same frame indices, and the same video must always produce the
same frame ids.
"""

from __future__ import annotations

import itertools

import pytest

from datasets.preprocessing import (
    ExtractionError,
    FrameCountSource,
    InspectionStatus,
    MediaInspection,
    MediaInspector,
    SamplingStrategy,
    available_strategies,
    derive_frame_id,
    derive_source_recording_id,
    derive_source_video_id,
    every_n_indices,
    plan_uniform_sampling,
    register_strategy,
    resolve_frame_count,
    uniform_frame_indices,
)
from datasets.preprocessing.frames import FrameManifestError, SourceIdentityLayout
from tests.synthetic_preprocessing import FakeVideoDecoder

# ---------------------------------------------------------------------------
# uniform_frame_indices
# ---------------------------------------------------------------------------


class TestUniformFrameIndices:
    @pytest.mark.parametrize(
        ("frame_count", "frames_per_video"),
        [(30, 5), (100, 7), (1000, 16), (17, 4), (50, 3), (7, 3)],
    )
    def test_indices_are_ascending_unique_and_in_range(self, frame_count, frames_per_video):
        indices = uniform_frame_indices(frame_count, frames_per_video)
        assert list(indices) == sorted(indices)
        assert len(set(indices)) == len(indices)
        assert all(0 <= index < frame_count for index in indices)

    def test_is_reproducible_across_repeated_calls(self):
        # The core contract. Compared as a tuple so order matters.
        assert uniform_frame_indices(1000, 16) == uniform_frame_indices(1000, 16)

    @pytest.mark.parametrize(
        ("frame_count", "frames_per_video"),
        [(30, 5), (100, 7), (1000, 16), (17, 4)],
    )
    def test_spans_whole_video_including_both_endpoints(self, frame_count, frames_per_video):
        indices = uniform_frame_indices(frame_count, frames_per_video)
        assert indices[0] == 0
        assert indices[-1] == frame_count - 1

    def test_short_video_yields_every_frame_once(self):
        # Asking for more frames than exist must return all of them, never duplicates.
        assert uniform_frame_indices(5, 20) == (0, 1, 2, 3, 4)

    def test_exact_length_yields_every_frame(self):
        assert uniform_frame_indices(10, 10) == tuple(range(10))

    def test_zero_length_video_yields_nothing(self):
        assert uniform_frame_indices(0, 5) == ()

    def test_single_requested_frame_is_the_first(self):
        assert uniform_frame_indices(30, 1) == (0,)

    def test_sampling_is_evenly_spaced_within_one_frame(self):
        # Even spacing, so no gap between selected frames exceeds one rounding step.
        indices = uniform_frame_indices(100, 7)
        gaps = [b - a for a, b in itertools.pairwise(indices)]
        assert max(gaps) - min(gaps) <= 1

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive_frame_budget(self, bad):
        with pytest.raises(ExtractionError, match="at least 1"):
            uniform_frame_indices(30, bad)

    def test_rejects_negative_frame_count(self):
        with pytest.raises(ExtractionError, match="non-negative"):
            uniform_frame_indices(-1, 5)


class TestEveryNIndices:
    def test_starts_at_first_frame_and_steps(self):
        assert every_n_indices(10, 3) == (0, 3, 6, 9)

    def test_interval_of_one_returns_every_frame(self):
        assert every_n_indices(5, 1) == (0, 1, 2, 3, 4)

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ExtractionError, match="at least 1"):
            every_n_indices(10, 0)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    def test_uniform_and_every_n_are_registered(self):
        available = available_strategies()
        assert SamplingStrategy.UNIFORM in available
        assert SamplingStrategy.EVERY_N in available

    @pytest.mark.parametrize(
        "strategy", [SamplingStrategy.FPS_BASED, SamplingStrategy.TIMESTAMP_BASED]
    )
    def test_declared_but_unimplemented_strategy_fails_loudly(self, strategy, touched_video):
        # Must raise rather than silently fall back to uniform: believing you sampled by
        # timestamp when you sampled by index is the worse outcome.
        inspection = MediaInspection(
            path=str(touched_video),
            frame_count=30,
            declared_frame_count=30,
            status=InspectionStatus.OK,
        )
        with pytest.raises(ExtractionError, match="not implemented"):
            plan_uniform_sampling(inspection, 5, strategy=strategy)

    def test_unknown_strategy_names_the_available_ones(self):
        with pytest.raises(ExtractionError, match="available"):
            SamplingStrategy.parse("nonsense")

    def test_custom_strategy_can_be_registered_and_used(self, touched_video):
        def first_middle_last(frame_count: int, **kwargs: int) -> tuple[int, ...]:
            if frame_count == 0:
                return ()
            return (0, frame_count // 2, frame_count - 1)

        register_strategy(SamplingStrategy.EVERY_N, first_middle_last)
        try:
            inspection = MediaInspection(
                path=str(touched_video), frame_count=10, declared_frame_count=10
            )
            plan = plan_uniform_sampling(inspection, 3, strategy=SamplingStrategy.EVERY_N)
            assert plan.indices == (0, 5, 9)
        finally:
            # Restore the real selector so later tests are unaffected.
            register_strategy(SamplingStrategy.EVERY_N, every_n_indices)

    def test_selector_returning_non_integer_is_rejected(self, touched_video):
        register_strategy(SamplingStrategy.EVERY_N, lambda frame_count, **kwargs: (0, "5"))
        try:
            inspection = MediaInspection(
                path=str(touched_video), frame_count=10, declared_frame_count=10
            )
            with pytest.raises(ExtractionError, match="not a frame index"):
                plan_uniform_sampling(inspection, 2, strategy=SamplingStrategy.EVERY_N)
        finally:
            register_strategy(SamplingStrategy.EVERY_N, every_n_indices)

    def test_selector_returning_out_of_range_index_is_rejected(self, touched_video):
        register_strategy(
            SamplingStrategy.EVERY_N, lambda frame_count, **kwargs: (0, frame_count + 5)
        )
        try:
            inspection = MediaInspection(
                path=str(touched_video), frame_count=10, declared_frame_count=10
            )
            with pytest.raises(ExtractionError, match="outside the video"):
                plan_uniform_sampling(inspection, 2, strategy=SamplingStrategy.EVERY_N)
        finally:
            register_strategy(SamplingStrategy.EVERY_N, every_n_indices)

    def test_selector_returning_unsorted_indices_is_rejected(self, touched_video):
        register_strategy(SamplingStrategy.EVERY_N, lambda frame_count, **kwargs: (5, 1))
        try:
            inspection = MediaInspection(
                path=str(touched_video), frame_count=10, declared_frame_count=10
            )
            with pytest.raises(ExtractionError, match="ascending"):
                plan_uniform_sampling(inspection, 2, strategy=SamplingStrategy.EVERY_N)
        finally:
            register_strategy(SamplingStrategy.EVERY_N, every_n_indices)


# ---------------------------------------------------------------------------
# Frame count resolution
# ---------------------------------------------------------------------------


class TestFrameCountResolution:
    def test_observed_count_is_preferred_over_declared(self):
        inspection = MediaInspection(path="v.mp4", frame_count=29, declared_frame_count=30)
        count, source = resolve_frame_count(inspection)
        assert (count, source) == (29, FrameCountSource.OBSERVED)

    def test_falls_back_to_declared_count(self):
        inspection = MediaInspection(path="v.mp4", declared_frame_count=30)
        count, source = resolve_frame_count(inspection)
        assert (count, source) == (30, FrameCountSource.DECLARED)

    def test_no_count_yields_unavailable(self):
        count, source = resolve_frame_count(MediaInspection(path="v.mp4"))
        assert (count, source) == (None, FrameCountSource.UNAVAILABLE)

    def test_planning_without_any_count_is_refused(self, touched_video):
        inspection = MediaInspector(FakeVideoDecoder()).inspect(touched_video)
        with pytest.raises(ExtractionError, match="frame count is available"):
            plan_uniform_sampling(inspection, 5)

    def test_sampling_from_declared_count_is_noted(self, touched_video):
        # Allowed, because for constant-frame-rate video it is usually right - but never
        # silent, since a wrong declared count changes which frames are extracted.
        inspection = MediaInspection(path=str(touched_video), declared_frame_count=30)
        plan = plan_uniform_sampling(inspection, 4)
        assert plan.frame_count_source is FrameCountSource.DECLARED
        assert any("declared frame count" in note for note in plan.notes)


class TestSamplingPlan:
    def test_plan_is_deterministic(self, touched_video):
        inspection = MediaInspection(
            path=str(touched_video), frame_count=100, declared_frame_count=100
        )
        first = plan_uniform_sampling(inspection, 8)
        second = plan_uniform_sampling(inspection, 8)
        assert first.indices == second.indices
        assert first.to_dict() == second.to_dict()

    def test_short_video_is_flagged_and_noted(self, touched_video):
        inspection = MediaInspection(path=str(touched_video), frame_count=3)
        plan = plan_uniform_sampling(inspection, 10)
        assert plan.is_short
        assert len(plan.indices) == 3
        assert any("requested 10" in note for note in plan.notes)

    def test_plan_reports_coverage(self, touched_video):
        inspection = MediaInspection(
            path=str(touched_video), frame_count=100, declared_frame_count=100
        )
        assert plan_uniform_sampling(inspection, 8).covers_whole_video


# ---------------------------------------------------------------------------
# Identity derivation
# ---------------------------------------------------------------------------


class TestSourceVideoIdentity:
    """The per-file identity, which must be unique across the dataset."""

    def test_includes_the_full_path(self):
        assert derive_source_video_id("ffpp", "original/000/001.mp4") == "ffpp__original__000__001"
        assert (
            derive_source_video_id("ffpp", "DeepFakes/000/001.mp4") == "ffpp__DeepFakes__000__001"
        )

    def test_distinguishes_manipulations_of_one_recording(self):
        # Two different files, so two different identities - which is what keeps their
        # frame ids unique.
        assert derive_source_video_id("ffpp", "original/000/001.mp4") != derive_source_video_id(
            "ffpp", "DeepFakes/000/001.mp4"
        )

    def test_extension_is_excluded_so_reencodes_keep_identity(self):
        assert derive_source_video_id("ds", "clips/a.mp4") == derive_source_video_id(
            "ds", "clips/a.mkv"
        )

    def test_windows_separators_normalise(self):
        assert derive_source_video_id("ds", r"clips\a.mp4") == derive_source_video_id(
            "ds", "clips/a.mp4"
        )

    def test_identity_is_deterministic(self):
        first = derive_source_video_id("ffpp", "original/000/001.mp4")
        second = derive_source_video_id("ffpp", "original/000/001.mp4")
        assert first == second

    def test_takes_no_layout_argument(self):
        # Frame ids depend on file identity alone; the layout governs grouping only.
        # An accidental layout parameter here would reintroduce frame-id collisions.
        import inspect

        assert list(inspect.signature(derive_source_video_id).parameters) == [
            "dataset_name",
            "relative_path",
        ]

    @pytest.mark.parametrize(
        "bad", ["", "/absolute/path.mp4", "../escape/clip.mp4", "clips/../../x.mp4"]
    )
    def test_rejects_paths_that_cannot_be_stably_identified(self, bad):
        with pytest.raises(FrameManifestError):
            derive_source_video_id("ds", bad)

    def test_rejects_empty_dataset_name(self):
        with pytest.raises(FrameManifestError, match="dataset_name"):
            derive_source_video_id("  ", "clips/a.mp4")


class TestSourceRecordingIdentity:
    """The grouping key, which must collapse manipulations of one recording."""

    def test_default_layout_matches_the_file_identity(self):
        assert derive_source_recording_id("ffpp", "original/000/001.mp4") == derive_source_video_id(
            "ffpp", "original/000/001.mp4"
        )

    def test_drop_first_segment_links_manipulations_of_one_recording(self):
        layout = SourceIdentityLayout.DROP_FIRST_SEGMENT
        identities = {
            derive_source_recording_id("ffpp", f"{manip}/000/001.mp4", layout=layout)
            for manip in ("original", "DeepFakes", "Face2Face", "NeuralTextures")
        }
        assert identities == {"ffpp__000__001"}

    def test_drop_first_segment_still_separates_different_recordings(self):
        layout = SourceIdentityLayout.DROP_FIRST_SEGMENT
        assert derive_source_recording_id(
            "ffpp", "DeepFakes/000/001.mp4", layout=layout
        ) != derive_source_recording_id("ffpp", "DeepFakes/002/001.mp4", layout=layout)

    def test_layout_accepts_a_plain_string(self):
        # A str-Enum member and its value compare equal but are not identical; a regression
        # that compared with `is` silently skipped the layout branch for string input.
        assert derive_source_recording_id(
            "ffpp", "DeepFakes/000/001.mp4", layout="drop_first_segment"
        ) == derive_source_recording_id(
            "ffpp",
            "DeepFakes/000/001.mp4",
            layout=SourceIdentityLayout.DROP_FIRST_SEGMENT,
        )

    def test_drop_first_segment_needs_a_directory(self):
        with pytest.raises(FrameManifestError, match="at least one directory"):
            derive_source_recording_id(
                "ds", "clip.mp4", layout=SourceIdentityLayout.DROP_FIRST_SEGMENT
            )

    def test_unknown_layout_is_rejected(self):
        with pytest.raises(FrameManifestError, match="not a valid source identity layout"):
            derive_source_recording_id("ds", "clips/a.mp4", layout="magic")

    def test_rejects_empty_dataset_name(self):
        with pytest.raises(FrameManifestError, match="dataset_name"):
            derive_source_recording_id("  ", "clips/a.mp4")

    def test_grouped_identity_is_still_a_pure_function_of_the_path(self):
        layout = SourceIdentityLayout.DROP_FIRST_SEGMENT
        first = derive_source_recording_id("ffpp", "DeepFakes/000/001.mp4", layout=layout)
        second = derive_source_recording_id("ffpp", "DeepFakes/000/001.mp4", layout=layout)
        assert first == second


class TestFrameIdentity:
    def test_frame_id_is_derived_from_parent_and_index(self):
        assert derive_frame_id("ds__a", 0) == "ds__a__f000000"
        assert derive_frame_id("ds__a", 42) == "ds__a__f000042"

    def test_frame_ids_sort_in_numeric_order(self):
        # Zero padding is what makes a directory listing agree with frame order.
        ids = [derive_frame_id("ds__a", index) for index in (2, 100, 30)]
        assert sorted(ids) == [
            "ds__a__f000002",
            "ds__a__f000030",
            "ds__a__f000100",
        ]

    def test_same_parent_and_index_always_give_same_id(self):
        assert derive_frame_id("ds__a", 7) == derive_frame_id("ds__a", 7)

    def test_rejects_empty_parent(self):
        with pytest.raises(FrameManifestError, match="source_video_id"):
            derive_frame_id("", 0)

    def test_rejects_negative_index(self):
        with pytest.raises(FrameManifestError, match="non-negative"):
            derive_frame_id("ds__a", -1)
