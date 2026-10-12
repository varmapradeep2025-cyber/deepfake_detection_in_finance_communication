"""Tests for grouping, group-aware splitting, and leakage prevention.

The property under test throughout: **every frame from one source video must be in
exactly one split.** Frame-id disjointness cannot express this — every frame id is
distinct, so a per-frame split passes it while putting a genuine clip in train and its own
forgery in test. These tests build the situation deliberately and assert it is caught.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from datasets.preprocessing import (
    FrameSample,
    GroupingError,
    GroupSplitConfig,
    SourceIdentityLayout,
    SourceVideoGroup,
    assert_no_group_leakage,
    assign_groups_to_splits,
    build_frame_split,
    derive_frame_id,
    derive_source_recording_id,
    derive_source_video_id,
    expand_groups_to_frames,
    group_frames_by_source_video,
    require_consistent_groups,
    stable_sort_key,
    to_frozen_split,
)
from datasets.preprocessing.frames import FrameManifestError
from datasets.schema import SampleLabel

PARENT_SHA = "a" * 64
OTHER_SHA = "b" * 64
MANIFEST_HASH = f"sha256:{'c' * 64}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_frame(
    source_video_id: str,
    frame_index: int,
    *,
    recording_id: str | None = None,
    label: SampleLabel = SampleLabel.FAKE,
    sha: str = PARENT_SHA,
    dataset: str = "ffpp",
) -> FrameSample:
    """Build a frame record with a derived id.

    `recording_id` defaults to the video id, which is the per-file grouping case. Pass it
    explicitly to model several videos descending from one recording.
    """
    return FrameSample(
        frame_id=derive_frame_id(source_video_id, frame_index),
        source_video_id=source_video_id,
        source_recording_id=recording_id or source_video_id,
        frame_index=frame_index,
        frame_path=f"{source_video_id}/frame_{frame_index:06d}.png",
        dataset_name=dataset,
        label=label,
        source_video_sha256=sha,
        license="research-only",
    )


def make_group(group_id: str, *, frame_count: int = 4, dataset: str = "ffpp") -> SourceVideoGroup:
    """Build a group directly, for split tests that do not need frames."""
    return SourceVideoGroup(
        group_id=group_id,
        frame_ids=tuple(derive_frame_id(group_id, index) for index in range(frame_count)),
        labels=("fake",),
        source_video_sha256=(PARENT_SHA,),
        dataset_name=dataset,
    )


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_frames_from_one_video_land_in_one_group(self):
        frames = [make_frame("ffpp__000__001", index) for index in range(5)]
        result = group_frames_by_source_video(frames)
        assert result.group_count == 1
        assert result.groups[0].frame_count == 5

    def test_group_membership_is_order_independent(self):
        frames = [make_frame("ffpp__000__001", index) for index in range(4)]
        forward = group_frames_by_source_video(frames)
        backward = group_frames_by_source_video(list(reversed(frames)))
        assert forward.to_dict() == backward.to_dict()

    def test_group_records_labels_and_manipulations(self):
        frames = [
            make_frame("ffpp__000__001", 0, label=SampleLabel.REAL),
            make_frame("ffpp__000__001", 1, label=SampleLabel.FAKE),
        ]
        group = group_frames_by_source_video(frames).groups[0]
        assert group.labels == ("fake", "real")
        assert group.stratification_key == "fake+real"

    def test_blank_recording_identity_is_refused_at_construction(self):
        # Refused early rather than becoming a phantom group: a frame that cannot be
        # grouped is a frame that cannot be protected from leakage.
        with pytest.raises(FrameManifestError):
            FrameSample(
                frame_id=derive_frame_id("orphan", 0),
                source_video_id="orphan",
                source_recording_id="   ",
                frame_index=0,
                frame_path="orphan/frame_000000.png",
                dataset_name="ffpp",
                label=SampleLabel.FAKE,
                source_video_sha256=PARENT_SHA,
                license="research-only",
            )

    def test_recording_id_must_be_supplied(self):
        with pytest.raises(TypeError):
            FrameSample(
                frame_id=derive_frame_id("orphan", 0),
                source_video_id="orphan",
                frame_index=0,
                frame_path="orphan/frame_000000.png",
                dataset_name="ffpp",
                label=SampleLabel.FAKE,
                source_video_sha256=PARENT_SHA,
                license="research-only",
            )

    def test_one_video_under_two_recordings_is_reported(self):
        # The real conflict. A video contributing frames to two groups could put those
        # groups in different splits, which is leakage through a mislabelled path.
        frames = [
            make_frame("ffpp__original__000__001", 0, recording_id="ffpp__000__a"),
            make_frame("ffpp__original__000__001", 1, recording_id="ffpp__000__b"),
        ]
        result = group_frames_by_source_video(frames)
        assert result.group_count == 2
        assert result.video_id_reuse["ffpp__original__000__001"] == (
            "ffpp__000__a",
            "ffpp__000__b",
        )
        assert not result.is_consistent
        with pytest.raises(GroupingError, match=r"2 recording ids"):
            require_consistent_groups(result)

    def test_several_videos_in_one_recording_is_expected(self):
        # A genuine clip plus its forgeries: several video ids, one recording. This must
        # NOT be reported as a conflict.
        frames = [
            make_frame(f"ffpp__{manip}__000__001", index, recording_id="ffpp__000__001")
            for manip in ("original", "DeepFakes")
            for index in range(2)
        ]
        result = group_frames_by_source_video(frames)
        assert result.is_consistent
        assert result.group_count == 1
        assert result.groups[0].video_count == 2

    def test_consistent_groups_pass_the_guard(self):
        result = group_frames_by_source_video([make_frame("ffpp__000__001", 0)])
        assert require_consistent_groups(result).is_consistent

    def test_group_for_frame_round_trips(self):
        frames = [make_frame("ffpp__000__001", index) for index in range(3)]
        result = group_frames_by_source_video(frames)
        frame_id = frames[1].frame_id
        assert result.group_for_frame(frame_id).group_id == "ffpp__000__001"

    def test_frames_are_never_counted_in_two_groups(self):
        frames = [make_frame(f"ffpp__000__00{n}", index) for n in (1, 2) for index in range(3)]
        result = group_frames_by_source_video(frames)
        seen = [fid for group in result.groups for fid in group.frame_ids]
        assert len(seen) == len(set(seen)) == len(frames)


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


class TestLeakageDetection:
    def test_clean_group_assignment_is_leak_free(self):
        groups = [make_group(f"ffpp__000__00{n}") for n in range(4)]
        assignment = expand_groups_to_frames(
            groups, assign_groups_to_splits(groups, GroupSplitConfig())
        )
        report = assert_no_group_leakage(groups, assignment)
        assert report.is_leak_free
        assert not report.spanning_groups

    def test_group_split_across_train_and_test_is_caught(self):
        # The exact failure this project exists to prevent, built by hand.
        groups = [make_group("ffpp__000__001", frame_count=4)]
        assignment = {
            derive_frame_id("ffpp__000__001", index): ("train" if index < 2 else "test")
            for index in range(4)
        }
        report = assert_no_group_leakage(groups, assignment, strict=False)
        assert not report.is_leak_free
        assert report.leaked_group_count == 1
        assert report.spanning_groups["ffpp__000__001"] == ("test", "train")

    def test_leakage_raises_when_strict(self):
        groups = [make_group("ffpp__000__001", frame_count=2)]
        assignment = {
            derive_frame_id("ffpp__000__001", 0): "train",
            derive_frame_id("ffpp__000__001", 1): "validation",
        }
        with pytest.raises(GroupingError, match="split leakage detected"):
            assert_no_group_leakage(groups, assignment, strict=True)

    def test_leak_error_names_the_offending_group(self):
        groups = [make_group("ffpp__000__777", frame_count=2)]
        assignment = {
            derive_frame_id("ffpp__000__777", 0): "train",
            derive_frame_id("ffpp__000__777", 1): "test",
        }
        with pytest.raises(GroupingError) as excinfo:
            assert_no_group_leakage(groups, assignment, strict=True)
        assert "ffpp__000__777" in str(excinfo.value)

    def test_unassigned_frames_are_reported(self):
        groups = [make_group("ffpp__000__001", frame_count=2)]
        assignment = {derive_frame_id("ffpp__000__001", 0): "train"}
        report = assert_no_group_leakage(groups, assignment, strict=False)
        assert not report.is_leak_free
        assert len(report.unassigned_frames) == 1

    def test_report_includes_label_balance(self):
        groups = [make_group("ffpp__000__001", frame_count=2)]
        frame_ids = list(groups[0].frame_ids)
        assignment = {frame_ids[0]: "train", frame_ids[1]: "test"}
        labels = {frame_ids[0]: SampleLabel.REAL, frame_ids[1]: SampleLabel.FAKE}
        report = assert_no_group_leakage(groups, assignment, label_by_frame=labels, strict=False)
        assert report.label_counts["train"] == {"real": 1}
        assert report.label_counts["test"] == {"fake": 1}

    def test_summary_is_readable(self):
        groups = [make_group("ffpp__000__001", frame_count=2)]
        assignment = dict.fromkeys(groups[0].frame_ids, "train")
        report = assert_no_group_leakage(groups, assignment)
        assert "LEAK-FREE" in report.summary()


# ---------------------------------------------------------------------------
# Group-aware assignment
# ---------------------------------------------------------------------------


class TestGroupAwareAssignment:
    def test_every_group_is_assigned_exactly_once(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(20)]
        assignment = assign_groups_to_splits(groups, GroupSplitConfig())
        assert set(assignment) == {group.group_id for group in groups}
        assert len(assignment) == len(groups)

    def test_all_three_splits_are_populated_for_enough_groups(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(20)]
        assignment = assign_groups_to_splits(groups, GroupSplitConfig())
        assert set(assignment.values()) == {"train", "validation", "test"}

    def test_assignment_is_deterministic_for_a_seed(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(20)]
        first = assign_groups_to_splits(groups, GroupSplitConfig(seed=7))
        second = assign_groups_to_splits(groups, GroupSplitConfig(seed=7))
        assert first == second

    def test_different_seeds_give_different_partitions(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(40)]
        assert assign_groups_to_splits(groups, GroupSplitConfig(seed=1)) != assign_groups_to_splits(
            groups, GroupSplitConfig(seed=2)
        )

    def test_stable_sort_key_is_pure_sha256(self):
        assert stable_sort_key(7, "g") == stable_sort_key(7, "g")
        assert stable_sort_key(7, "g") != stable_sort_key(8, "g")

    def test_respects_requested_fractions(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(100)]
        assignment = assign_groups_to_splits(
            groups, GroupSplitConfig(train_fraction=0.7, validation_fraction=0.2)
        )
        counts = dict.fromkeys(("train", "validation", "test"), 0)
        for value in assignment.values():
            counts[value] += 1
        assert counts["train"] == 70
        assert counts["validation"] == 20
        assert counts["test"] == 10

    def test_small_stratum_still_lands_somewhere(self):
        # Two groups cannot fill three splits; they must not be dropped.
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(2)]
        assignment = assign_groups_to_splits(groups, GroupSplitConfig())
        assert len(assignment) == 2

    def test_single_group_is_not_lost(self):
        assignment = assign_groups_to_splits([make_group("ffpp__000__001")])
        assert assignment == {"ffpp__000__001": "train"}

    def test_groups_without_frames_are_ignored(self):
        groups = [make_group("ffpp__000__001"), SourceVideoGroup(group_id="empty")]
        assert set(assign_groups_to_splits(groups)) == {"ffpp__000__001"}

    def test_empty_input_yields_empty_assignment(self):
        assert assign_groups_to_splits([]) == {}

    def test_stratification_keeps_label_mixes_apart(self):
        # Mixed-label groups and pure-label groups must not be interleaved into the same
        # stratum, or the class ratio of the splits drifts.
        mixed = SourceVideoGroup(
            group_id="ffpp__mixed",
            frame_ids=("a", "b"),
            labels=("fake", "real"),
        )
        pure = SourceVideoGroup(group_id="ffpp__pure", frame_ids=("c",), labels=("fake",))
        assert mixed.stratification_key != pure.stratification_key

    def test_expansion_gives_every_frame_of_a_group_the_same_split(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(10)]
        assignment = assign_groups_to_splits(groups, GroupSplitConfig())
        expanded = expand_groups_to_frames(groups, assignment)
        for group in groups:
            splits = {expanded[frame_id] for frame_id in group.frame_ids}
            assert len(splits) == 1


# ---------------------------------------------------------------------------
# Validation and frozen-split output
# ---------------------------------------------------------------------------


class TestGroupSplitConfigValidation:
    def test_rejects_fractions_above_one(self):
        with pytest.raises(GroupingError, match=r"must be <= 1\.0"):
            GroupSplitConfig(train_fraction=0.9, validation_fraction=0.2)

    def test_rejects_negative_fraction(self):
        with pytest.raises(GroupingError, match=r"within 0\.0-1\.0"):
            GroupSplitConfig(train_fraction=-0.1)

    def test_rejects_boolean_seed(self):
        with pytest.raises(GroupingError, match="seed must be an integer"):
            GroupSplitConfig(seed=True)  # type: ignore[arg-type]

    def test_test_fraction_is_derived(self):
        config = GroupSplitConfig(train_fraction=0.6, validation_fraction=0.3)
        assert config.test_fraction == pytest.approx(0.1)


class TestBuildFrameSplit:
    @pytest.fixture
    def ffpp_like_groups(self) -> list[SourceVideoGroup]:
        """Three recordings, each contributing a genuine clip and two forgeries.

        Mirrors the FaceForensics++ layout after grouping: one group per source recording,
        each holding both a real and fake frames.
        """
        groups: list[SourceVideoGroup] = []
        for recording in range(6):
            group_id = f"ffpp__0{recording:02d}__001"
            frames = [
                derive_frame_id(group_id, 0),
                derive_frame_id(group_id, 1),
                derive_frame_id(group_id, 2),
            ]
            groups.append(
                SourceVideoGroup(
                    group_id=group_id,
                    frame_ids=tuple(frames),
                    labels=("fake", "real"),
                    source_video_sha256=(PARENT_SHA,),
                    dataset_name="ffpp",
                )
            )
        return groups

    def test_produces_a_verified_leak_free_split(self, ffpp_like_groups):
        result = build_frame_split(
            ffpp_like_groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH
        )
        assert result.leakage.is_leak_free
        assert result.dataset_name == "ffpp"

    def test_no_group_spans_splits(self, ffpp_like_groups):
        result = build_frame_split(
            ffpp_like_groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH
        )
        for group in ffpp_like_groups:
            splits = {result.frame_assignment[fid] for fid in group.frame_ids}
            assert len(splits) == 1, f"{group.group_id} spans {splits}"

    def test_counts_add_up(self, ffpp_like_groups):
        result = build_frame_split(
            ffpp_like_groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH
        )
        counts = result.counts()
        assert counts["train"] + counts["validation"] + counts["test"] == sum(
            len(group.frame_ids) for group in ffpp_like_groups
        )

    def test_is_reproducible(self, ffpp_like_groups):
        first = build_frame_split(
            ffpp_like_groups, GroupSplitConfig(seed=99), source_manifest_hash=MANIFEST_HASH
        )
        second = build_frame_split(
            ffpp_like_groups, GroupSplitConfig(seed=99), source_manifest_hash=MANIFEST_HASH
        )
        assert first.frame_assignment == second.frame_assignment

    def test_frames_for_filters_correctly(self, ffpp_like_groups):
        result = build_frame_split(
            ffpp_like_groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH
        )
        train = set(result.frames_for("train"))
        assert train
        assert train.isdisjoint(result.frames_for("test"))

    def test_empty_corpus_is_refused_with_a_reason(self):
        # Refused rather than producing an empty split that could be mistaken for a
        # completed one.
        with pytest.raises(GroupingError, match="cannot infer a dataset name"):
            build_frame_split([], GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH)

    def test_groups_from_several_datasets_are_refused(self):
        groups = [
            make_group("a__000__001", dataset="ds_a"),
            make_group("b__000__001", dataset="ds_b"),
        ]
        with pytest.raises(GroupingError, match="span 2 datasets"):
            build_frame_split(groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH)


class TestFrozenSplitOutput:
    @pytest.fixture
    def result(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(10)]
        return build_frame_split(groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH)

    def test_records_the_grouping_guarantee_in_the_rationale(self, result):
        # A split whose rationale does not say grouping was applied is not usable evidence.
        frozen = to_frozen_split(result, source_manifest_hash=MANIFEST_HASH)
        assert "source-video grouping applied" in frozen.rationale
        assert "leakage check passed" in frozen.rationale

    def test_records_leakage_report_and_config_as_provenance(self, result):
        frozen = to_frozen_split(result, source_manifest_hash=MANIFEST_HASH)
        assert frozen.extra["leakage_report"]["is_leak_free"] is True
        assert frozen.extra["split_config"]["seed"] == result.config.seed

    def test_frame_ids_land_in_the_artifact(self, result):
        frozen = to_frozen_split(result, source_manifest_hash=MANIFEST_HASH)
        assert frozen.total_ids == len(result.frame_assignment)

    def test_content_hash_is_stable_across_identical_runs(self):
        groups = [make_group(f"ffpp__000__0{n:02d}") for n in range(10)]
        first = to_frozen_split(
            build_frame_split(groups, GroupSplitConfig(seed=5), source_manifest_hash=MANIFEST_HASH),
            source_manifest_hash=MANIFEST_HASH,
            created_at="2026-01-01T00:00:00+00:00",
        )
        second = to_frozen_split(
            build_frame_split(groups, GroupSplitConfig(seed=5), source_manifest_hash=MANIFEST_HASH),
            source_manifest_hash=MANIFEST_HASH,
            created_at="2026-01-01T00:00:00+00:00",
        )
        assert first.content_hash() == second.content_hash()

    def test_requires_a_dataset_name(self):
        groups = [make_group("ffpp__000__001")]
        result = build_frame_split(groups, GroupSplitConfig(), source_manifest_hash=MANIFEST_HASH)
        object.__setattr__(result, "dataset_name", "")
        with pytest.raises(GroupingError, match="without a dataset name"):
            to_frozen_split(result, source_manifest_hash=MANIFEST_HASH)


# ---------------------------------------------------------------------------
# The real-world leakage scenario
# ---------------------------------------------------------------------------


class TestFaceForensicsLeakageScenario:
    """The situation that motivates grouping, reproduced end to end.

    Models the FaceForensics++ layout as it actually is: `<manipulation>/<recording>/clip`
    for each of three manipulations, with the genuine clip and both forgeries descending
    from one recording.
    """

    LAYOUT = SourceIdentityLayout.DROP_FIRST_SEGMENT
    MANIPULATIONS = (
        ("original", SampleLabel.REAL),
        ("DeepFakes", SampleLabel.FAKE),
        ("Face2Face", SampleLabel.FAKE),
    )
    RECORDINGS = 10
    FRAMES_PER_VIDEO = 4

    def _frames(self) -> list[FrameSample]:
        frames: list[FrameSample] = []
        for recording in range(self.RECORDINGS):
            relative = f"0{recording:03d}/clip.mp4"
            for manipulation, label in self.MANIPULATIONS:
                path = f"{manipulation}/{relative}"
                video_id = derive_source_video_id("ffpp", path)
                recording_id = derive_source_recording_id("ffpp", path, layout=self.LAYOUT)
                for index in range(self.FRAMES_PER_VIDEO):
                    frames.append(
                        FrameSample(
                            frame_id=derive_frame_id(video_id, index),
                            source_video_id=video_id,
                            source_recording_id=recording_id,
                            frame_index=index,
                            frame_path=f"{video_id}/frame_{index:06d}.png",
                            dataset_name="ffpp",
                            label=label,
                            source_video_sha256=f"{recording:02d}{manipulation[:2]}" * 32,
                            license="research-only",
                        )
                    )
        return frames

    def test_frame_ids_are_unique_across_manipulations(self):
        # The bug this scenario exists to pin down: deriving frame ids from the recording
        # identity instead of the video identity collides, because a genuine clip and its
        # forgeries each have a frame 0.
        frame_ids = [frame.frame_id for frame in self._frames()]
        assert len(set(frame_ids)) == len(frame_ids)

    def test_drop_first_segment_groups_a_clip_with_its_forgeries(self):
        frames = self._frames()
        result = group_frames_by_source_video(frames)
        # 10 recordings, not 30 videos: each recording is one group.
        assert result.group_count == self.RECORDINGS
        assert result.is_consistent

    def test_each_group_holds_both_real_and_fake_frames(self):
        for group in group_frames_by_source_video(self._frames()).groups:
            assert group.labels == ("fake", "real")
            assert group.video_count == len(self.MANIPULATIONS)

    def test_frames_per_group(self):
        groups = group_frames_by_source_video(self._frames()).groups
        expected = len(self.MANIPULATIONS) * self.FRAMES_PER_VIDEO
        assert all(group.frame_count == expected for group in groups)

    def test_per_file_layout_would_produce_three_times_as_many_groups(self):
        # Shows why the layout choice is load-bearing: with the per-file recording layout, a
        # clip and its forgeries are three separate groups and could land in different
        # splits.
        frames = self._frames()
        by_file = group_frames_by_source_video(
            [replace(frame, source_recording_id=frame.source_video_id) for frame in frames]
        )
        assert by_file.group_count == self.RECORDINGS * len(self.MANIPULATIONS)

    def test_split_keeps_every_recording_entirely_in_one_split(self):
        frames = self._frames()
        result = build_frame_split(
            group_frames_by_source_video(frames).groups,
            GroupSplitConfig(),
            source_manifest_hash=MANIFEST_HASH,
        )
        assert result.leakage.is_leak_free
        for group in group_frames_by_source_video(frames).groups:
            splits = {result.frame_assignment[fid] for fid in group.frame_ids}
            assert len(splits) == 1

    def test_all_three_splits_receive_data(self):
        result = build_frame_split(
            group_frames_by_source_video(self._frames()).groups,
            GroupSplitConfig(),
            source_manifest_hash=MANIFEST_HASH,
        )
        assert set(result.frame_assignment.values()) == {
            "train",
            "validation",
            "test",
        }

    def test_class_balance_is_preserved_by_grouping(self):
        # Moving whole recordings keeps the real/fake mix in every split, which a per-frame
        # split would not.
        frames = self._frames()
        result = build_frame_split(
            group_frames_by_source_video(frames).groups,
            GroupSplitConfig(),
            source_manifest_hash=MANIFEST_HASH,
            label_by_frame={frame.frame_id: frame.label for frame in frames},
        )
        for split, counts in result.leakage.label_counts.items():
            assert counts.get("real", 0) > 0, f"{split} has no genuine frames"
            assert counts.get("fake", 0) > 0, f"{split} has no forged frames"

    def test_a_per_frame_split_would_have_leaked(self):
        # Counterfactual that gives the grouping its justification: the same frames split
        # per frame rather than per recording do leak.
        frames = self._frames()
        per_frame = {
            frame.frame_id: ("train" if int(frame.frame_id[-6:]) < 2 else "test")
            for frame in frames
        }
        with pytest.raises(GroupingError, match="split leakage detected"):
            assert_no_group_leakage(
                group_frames_by_source_video(frames).groups, per_frame, strict=True
            )
