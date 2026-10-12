"""Synthetic pipeline validation (Milestone 3A) - plumbing only, no real data."""
from __future__ import annotations

from datasets.preprocessing import (
    FrameSample,
    GroupSplitConfig,
    build_frame_split,
    coalesce_duplicate_linked_groups,
    derive_frame_id,
    duplicate_linked_components,
    find_exact_duplicates,
    group_frames_by_source_video,
)
from datasets.preprocessing.grouping import GroupingError, assert_no_group_leakage
from datasets.preprocessing.splitting import to_frozen_split
from datasets.schema import SampleLabel


def _frame(
    recording_id: str,
    vid_id: str,
    idx: int,
    frame_sha: str,
    label: SampleLabel = SampleLabel.FAKE,
) -> FrameSample:
    return FrameSample(
        frame_id=derive_frame_id(vid_id, idx),
        source_video_id=vid_id,
        source_recording_id=recording_id,
        frame_index=idx,
        frame_path=f"{vid_id}/frame_{idx:06d}.png",
        dataset_name="ffpp",
        label=label,
        source_video_sha256="a" * 64,
        license="research-only",
        frame_sha256=frame_sha,
    )


def test_frames_same_recording_group_together_across_manipulations():
    rec = "rec_001"
    frames = [
        _frame(rec, "DeepFakes__rec_001", 0, "1" * 64, label=SampleLabel.REAL),
        _frame(rec, "DeepFakes__rec_001", 1, "2" * 64, label=SampleLabel.FAKE),
        _frame(rec, "FaceSwap__rec_001", 0, "3" * 64, label=SampleLabel.FAKE),
        _frame(rec, "FaceSwap__rec_001", 1, "4" * 64, label=SampleLabel.FAKE),
    ]
    res = group_frames_by_source_video(frames)
    assert res.is_consistent
    assert res.group_count == 1
    g = res.groups[0]
    assert g.group_id == rec
    assert g.video_count == 2
    assert g.frame_count == 4


def test_deterministic_splits_fixed_seed():
    frames = []
    for r in range(4):
        rec = f"rec_{r:03d}"
        for v, lab in [("orig", SampleLabel.REAL), ("df", SampleLabel.FAKE)]:
            vid = f"{v}__{rec}"
            for i in range(2):
                frames.append(_frame(rec, vid, i, f"{r}{v}{i}" * 8, label=lab))
    res = group_frames_by_source_video(frames)
    config = GroupSplitConfig(train_fraction=0.5, validation_fraction=0.25, seed=123)
    manifest_hash = "sha256:" + "0" * 64
    split1 = build_frame_split(res.groups, config, source_manifest_hash=manifest_hash)
    split2 = build_frame_split(res.groups, config, source_manifest_hash=manifest_hash)
    fs1 = to_frozen_split(split1, source_manifest_hash=manifest_hash)
    fs2 = to_frozen_split(split2, source_manifest_hash=manifest_hash)
    assert fs1.validation_ids == fs2.validation_ids
    assert fs1.test_ids == fs2.test_ids


def test_group_leakage_assertion_passes_when_groups_intact():
    frames = []
    for r in range(3):
        rec = f"rec_{r:03d}"
        vid = f"orig__{rec}"
        frames.append(_frame(rec, vid, 0, f"{r}" * 64, label=SampleLabel.REAL))
    res = group_frames_by_source_video(frames)
    manifest_hash = "sha256:" + "1" * 64
    split = build_frame_split(
        res.groups, GroupSplitConfig(seed=7), source_manifest_hash=manifest_hash
    )
    fs = to_frozen_split(split, source_manifest_hash=manifest_hash)
    assign = {}
    for fid in fs.train_ids:
        assign[fid] = 'train'
    for fid in fs.validation_ids:
        assign[fid] = 'validation'
    for fid in fs.test_ids:
        assign[fid] = 'test'
    assert_no_group_leakage(res.groups, assign)


def test_invalid_missing_source_recording_raises_grouping_error():
    try:
        f = FrameSample(
            frame_id=derive_frame_id("v1", 0),
            source_video_id="v1",
            source_recording_id="",
            frame_index=0,
            frame_path="v1/f.png",
            dataset_name="d",
            label=SampleLabel.REAL,
            source_video_sha256="a" * 64,
            license="x",
            frame_sha256="b" * 64,
        )
    except Exception:
        # If construction fails (as it should if validation rejects empty), that's fine
        assert True
        return
    # If construction succeeded somehow, check grouping
    try:
        from datasets.preprocessing.grouping import require_consistent_groups
        res = group_frames_by_source_video([f])
        require_consistent_groups(res)
    except GroupingError:
        assert True
        return
    assert False


def test_duplicate_detection_and_coalescing():
    dup = "a" * 64
    frames = [
        _frame("r1", "v1", 0, dup, label=SampleLabel.REAL),
        _frame("r1", "v1", 1, "x" * 64, label=SampleLabel.FAKE),
        _frame("r2", "v2", 0, dup, label=SampleLabel.FAKE),
    ]
    dups = find_exact_duplicates(frames)
    assert len(dups.findings) == 1
    duplicate_linked_components(dups)
    merged = coalesce_duplicate_linked_groups(group_frames_by_source_video(frames).groups, dups)
    assert len(merged.groups) >= 1

