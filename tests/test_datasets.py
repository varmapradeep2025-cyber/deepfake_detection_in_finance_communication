"""Tests for :mod:`datasets.schema` and :mod:`datasets.manifest`.

Covers record validation, enum rejection, duplicate detection, JSONL round-tripping,
and chunked file hashing.
"""

from __future__ import annotations

import json

import pytest

from datasets.manifest import (
    calculate_file_sha256,
    calculate_manifest_hash,
    count_labels,
    count_modalities,
    describe_manifest,
    load_manifest,
    resolve_sample_path,
    validate_manifest_file,
    validate_sample_file,
    write_manifest,
)
from datasets.schema import (
    ManifestValidationError,
    ManipulationType,
    MediaSample,
    Modality,
    SampleLabel,
    SplitName,
    validate_samples,
)
from tests.conftest import make_file, make_sample, sha256_bytes

VALID_SHA = "a" * 64


def base_record(tmp_path, **overrides):
    """Minimal valid manifest record as a plain dict."""
    payload = make_file(tmp_path / "media" / "a.bin", b"bytes-a")
    record = {
        "sample_id": "s1",
        "dataset_name": "ds",
        "filepath": str(payload.relative_to(tmp_path)).replace("\\", "/"),
        "modality": "image",
        "label": "fake",
        "sha256": sha256_bytes(b"bytes-a"),
        "license": "CC0-1.0",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Enum parsing
# ---------------------------------------------------------------------------


class TestModality:
    def test_accepts_all_three_values(self):
        assert Modality.parse("image") is Modality.IMAGE
        assert Modality.parse("audio") is Modality.AUDIO
        assert Modality.parse("video") is Modality.VIDEO

    def test_values_are_exactly_the_three_modalities(self):
        assert set(Modality.values()) == {"image", "audio", "video"}

    @pytest.mark.parametrize("bad", ["text", "imag", "", "images", "IMAGE2", None, 7, ["image"]])
    def test_rejects_invalid_modality(self, bad):
        with pytest.raises(ManifestValidationError, match="invalid modality"):
            Modality.parse(bad)

    def test_case_is_normalized(self):
        assert Modality.parse("IMAGE") is Modality.IMAGE
        assert Modality.parse("  Video ") is Modality.VIDEO


class TestSampleLabel:
    def test_accepts_real_and_fake(self):
        assert SampleLabel.parse("real") is SampleLabel.REAL
        assert SampleLabel.parse("fake") is SampleLabel.FAKE

    def test_values_are_exactly_real_and_fake(self):
        assert set(SampleLabel.values()) == {"real", "fake"}

    @pytest.mark.parametrize("bad", ["tampered", "synthetic", "", "REALISH", None, 0])
    def test_rejects_invalid_label(self, bad):
        with pytest.raises(ManifestValidationError, match="invalid label"):
            SampleLabel.parse(bad)

    def test_is_fake_property(self):
        assert MediaSample(
            sample_id="x",
            dataset_name="d",
            filepath="f",
            modality="image",
            label="fake",
            sha256=VALID_SHA,
            license="CC0-1.0",
        ).is_fake
        assert not MediaSample(
            sample_id="y",
            dataset_name="d",
            filepath="f",
            modality="image",
            label="real",
            sha256=VALID_SHA,
            license="CC0-1.0",
        ).is_fake


class TestSplitName:
    def test_accepts_canonical_names_and_val_alias(self):
        assert SplitName.parse("train") is SplitName.TRAIN
        assert SplitName.parse("val") is SplitName.VALIDATION
        assert SplitName.parse("test") is SplitName.TEST

    def test_rejects_unknown_split(self):
        with pytest.raises(ManifestValidationError, match="invalid split"):
            SplitName.parse("holdout")


class TestManipulationType:
    def test_none_maps_to_unknown_when_absent(self):
        assert ManipulationType.parse(None) is ManipulationType.UNKNOWN

    def test_normalizes_separators(self):
        assert ManipulationType.parse("face-swap") is ManipulationType.FACE_SWAP
        assert ManipulationType.parse("Face Swap") is ManipulationType.FACE_SWAP

    def test_rejects_unknown_family(self):
        with pytest.raises(ManifestValidationError, match="invalid manipulation_type"):
            ManipulationType.parse("telepathy")


# ---------------------------------------------------------------------------
# MediaSample
# ---------------------------------------------------------------------------


class TestMediaSample:
    def test_builds_with_minimal_required_fields(self, tmp_path):
        sample = make_sample(tmp_path, "s1")
        assert sample.sample_id == "s1"
        assert sample.manipulation_type is ManipulationType.UNKNOWN
        assert sample.split is None

    def test_optional_fields_may_be_absent(self, tmp_path):
        sample = make_sample(tmp_path, "s1")
        assert sample.source is None
        assert sample.which_fields_are_populated() == ()

    def test_reports_populated_optional_fields(self, tmp_path):
        sample = make_sample(
            tmp_path,
            "s1",
            source="original",
            split="train",
            manipulation_type="face_swap",
            metadata={"width": 512},
        )
        populated = sample.which_fields_are_populated()
        assert set(populated) == {"manipulation_type", "metadata", "source", "split"}

    def test_ignores_empty_metadata_in_populated_report(self, tmp_path):
        assert "metadata" not in make_sample(tmp_path, "s1").which_fields_are_populated()

    def test_rejects_empty_sample_id(self):
        with pytest.raises(ManifestValidationError, match="sample_id"):
            MediaSample(
                sample_id="  ",
                dataset_name="d",
                filepath="f",
                modality="image",
                label="real",
                sha256=VALID_SHA,
                license="L",
            )

    @pytest.mark.parametrize("bad", ["abc", "z" * 63, "z" * 65, "g" * 64, 123, None])
    def test_rejects_invalid_sha256(self, bad):
        with pytest.raises(ManifestValidationError, match="sha256"):
            MediaSample(
                sample_id="s",
                dataset_name="d",
                filepath="f",
                modality="image",
                label="real",
                sha256=bad,
                license="L",
            )

    def test_normalizes_sha256_case_and_whitespace(self):
        sample = MediaSample(
            sample_id="s",
            dataset_name="d",
            filepath="f",
            modality="image",
            label="real",
            sha256="A" * 64,
            license="L",
        )
        assert sample.sha256 == "a" * 64

    def test_requires_license(self):
        with pytest.raises(ManifestValidationError, match="license"):
            MediaSample(
                sample_id="s",
                dataset_name="d",
                filepath="f",
                modality="image",
                label="real",
                sha256=VALID_SHA,
                license="  ",
            )

    def test_rejects_non_mapping_metadata(self, tmp_path):
        with pytest.raises(ManifestValidationError, match="metadata"):
            make_sample(tmp_path, "s1", metadata=["not", "a", "mapping"])

    def test_round_trips_through_dict(self, tmp_path):
        original = make_sample(tmp_path, "s1", metadata={"duration_s": 1.5}, source="original")
        restored = MediaSample.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_to_dict_uses_canonical_enum_strings(self, tmp_path):
        payload = make_sample(tmp_path, "s1", split="train").to_dict()
        assert payload["modality"] == "image"
        assert payload["label"] == "fake"
        assert payload["split"] == "train"
        assert payload["manipulation_type"] == "unknown"

    def test_from_dict_preserves_unknown_keys_under_extra(self, tmp_path):
        record = base_record(tmp_path, future_field="keep-me")
        sample = MediaSample.from_dict(record)
        assert sample.metadata["_extra"]["future_field"] == "keep-me"

    @pytest.mark.parametrize(
        "missing", ["sample_id", "dataset_name", "modality", "label", "sha256"]
    )
    def test_from_dict_reports_missing_required_fields(self, tmp_path, missing):
        record = base_record(tmp_path)
        del record[missing]
        with pytest.raises(ManifestValidationError, match="missing required field"):
            MediaSample.from_dict(record)

    def test_from_dict_rejects_non_object(self):
        with pytest.raises(ManifestValidationError, match="must be an object"):
            MediaSample.from_dict(["not", "an", "object"])


class TestDuplicateDetection:
    def test_detects_duplicate_sample_ids(self, tmp_path):
        samples = [make_sample(tmp_path, "dup"), make_sample(tmp_path, "dup")]
        with pytest.raises(ManifestValidationError) as exc:
            validate_samples(samples)
        assert exc.value.errors

    def test_distinct_ids_are_accepted(self, tmp_path):
        validate_samples([make_sample(tmp_path, "a"), make_sample(tmp_path, "b")])

    def test_same_id_in_different_datasets_is_allowed(self, tmp_path):
        validate_samples(
            [
                make_sample(tmp_path, "shared", dataset_name="ds1"),
                make_sample(tmp_path, "shared", dataset_name="ds2"),
            ]
        )

    def test_detects_duplicate_in_loaded_manifest(self, tmp_path):
        record = base_record(tmp_path)
        path = tmp_path / "m.jsonl"
        path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
        report = validate_manifest_file(path)
        assert not report.is_valid
        assert report.duplicate_sample_ids


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class TestFileHashing:
    def test_hashes_empty_file(self, tmp_path):
        empty = make_file(tmp_path / "empty.bin", b"")
        assert calculate_file_sha256(empty) == sha256_bytes(b"")

    def test_matches_hashlib_for_known_content(self, tmp_path):
        payload = b"deepfake-research-payload"
        target = make_file(tmp_path / "f.bin", payload)
        assert calculate_file_sha256(target) == sha256_bytes(payload)

    def test_is_chunk_size_independent(self, tmp_path):
        payload = bytes(range(256)) * 4096  # 1 MiB, larger than any chunk below
        target = make_file(tmp_path / "big.bin", payload)
        assert calculate_file_sha256(target, chunk_size=7) == calculate_file_sha256(
            target, chunk_size=1 << 20
        )

    def test_supports_alternative_algorithms(self, tmp_path):
        target = make_file(tmp_path / "f.bin", b"abc")
        assert len(calculate_file_sha256(target, algorithm="sha512")) == 128

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing file"):
            calculate_file_sha256(tmp_path / "nope.bin")

    def test_rejects_non_positive_chunk_size(self, tmp_path):
        target = make_file(tmp_path / "f.bin", b"abc")
        with pytest.raises(ValueError, match="chunk_size"):
            calculate_file_sha256(target, chunk_size=0)

    def test_rejects_unknown_algorithm(self, tmp_path):
        target = make_file(tmp_path / "f.bin", b"abc")
        with pytest.raises(ValueError, match="unsupported hash algorithm"):
            calculate_file_sha256(target, algorithm="not-a-hash")


class TestSampleVerification:
    def test_verifies_matching_file(self, tmp_path):
        sample = make_sample(tmp_path, "s1")
        ok, reason = validate_sample_file(sample, tmp_path)
        assert ok and reason is None

    def test_detects_content_tampering(self, tmp_path):
        sample = make_sample(tmp_path, "s1")
        (tmp_path / sample.filepath).write_bytes(b"tampered")
        ok, reason = validate_sample_file(sample, tmp_path)
        assert not ok
        assert "sha256 mismatch" in reason

    def test_detects_missing_file(self, tmp_path):
        sample = make_sample(tmp_path, "s1")
        (tmp_path / sample.filepath).unlink()
        ok, reason = validate_sample_file(sample, tmp_path)
        assert not ok
        assert "missing file" in reason

    def test_detects_same_size_different_content(self, tmp_path):
        """A one-byte edit must not pass: that is the point of hashing."""
        sample = make_sample(tmp_path, "s1", content=b"AAAA")
        (tmp_path / sample.filepath).write_bytes(b"AAAB")
        ok, reason = validate_sample_file(sample, tmp_path)
        assert not ok and "mismatch" in reason

    def test_resolves_relative_path_against_base_dir(self, tmp_path):
        sample = make_sample(tmp_path, "s1")
        assert resolve_sample_path(sample, tmp_path) == (tmp_path / sample.filepath).resolve()

    def test_manifest_level_file_check(self, tmp_path):
        samples = [make_sample(tmp_path, "s1"), make_sample(tmp_path, "s2")]
        path = write_manifest(tmp_path / "m.jsonl", samples)
        assert validate_manifest_file(path, check_files=True, base_dir=tmp_path).is_valid


class TestManifestHash:
    def test_hash_is_stable_across_calls(self, tmp_path):
        samples = [make_sample(tmp_path, "a"), make_sample(tmp_path, "b")]
        assert calculate_manifest_hash(samples) == calculate_manifest_hash(samples)

    def test_hash_is_order_independent(self, tmp_path):
        first = [make_sample(tmp_path, "a"), make_sample(tmp_path, "b")]
        second = [make_sample(tmp_path, "b"), make_sample(tmp_path, "a")]
        assert calculate_manifest_hash(first) == calculate_manifest_hash(second)

    def test_hash_changes_when_content_changes(self, tmp_path):
        a = make_sample(tmp_path, "a")
        b = make_sample(tmp_path, "a", content=b"different")
        assert calculate_manifest_hash([a]) != calculate_manifest_hash([b])

    def test_hash_is_prefixed_with_algorithm(self, tmp_path):
        digest = calculate_manifest_hash([make_sample(tmp_path, "a")])
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64

    def test_empty_manifest_has_a_hash(self):
        assert calculate_manifest_hash([]).startswith("sha256:")


# ---------------------------------------------------------------------------
# JSONL IO
# ---------------------------------------------------------------------------


class TestManifestIO:
    def test_round_trips(self, tmp_path):
        samples = [make_sample(tmp_path, "b"), make_sample(tmp_path, "a")]
        path = write_manifest(tmp_path / "m.jsonl", samples)
        loaded = load_manifest(path)
        # write_manifest sorts by sample_id, so file order is "a", "b".
        assert [s.sample_id for s in loaded] == ["a", "b"]
        by_id = {s.sample_id: s for s in samples}
        assert {s.sample_id: s.sha256 for s in loaded} == {k: v.sha256 for k, v in by_id.items()}
        assert loaded[0].to_dict() == by_id["a"].to_dict()

    def test_write_is_byte_deterministic(self, tmp_path):
        samples = [make_sample(tmp_path, "a"), make_sample(tmp_path, "b")]
        first = write_manifest(tmp_path / "one.jsonl", samples).read_bytes()
        second = write_manifest(tmp_path / "two.jsonl", list(reversed(samples))).read_bytes()
        assert first == second

    def test_refuses_to_overwrite_without_permission(self, tmp_path):
        path = write_manifest(tmp_path / "m.jsonl", [make_sample(tmp_path, "a")])
        with pytest.raises(FileExistsError, match="overwrite"):
            write_manifest(path, [make_sample(tmp_path, "b")])

    def test_overwrite_when_permitted(self, tmp_path):
        path = write_manifest(tmp_path / "m.jsonl", [make_sample(tmp_path, "a")])
        write_manifest(path, [make_sample(tmp_path, "b")], overwrite=True)
        assert [s.sample_id for s in load_manifest(path)] == ["b"]

    def test_rejects_duplicates_before_writing(self, tmp_path):
        dupes = [make_sample(tmp_path, "same"), make_sample(tmp_path, "same")]
        with pytest.raises(ManifestValidationError, match="duplicate"):
            write_manifest(tmp_path / "m.jsonl", dupes)

    def test_failed_write_leaves_no_file(self, tmp_path):
        target = tmp_path / "m.jsonl"
        with pytest.raises(ManifestValidationError):
            write_manifest(target, [make_sample(tmp_path, "same")] * 2)
        assert not target.exists()

    def test_skips_blank_lines(self, tmp_path):
        path = write_manifest(tmp_path / "m.jsonl", [make_sample(tmp_path, "a")])
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
        assert len(load_manifest(path)) == 1

    def test_rejects_malformed_json_with_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"sample_id": "a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ManifestValidationError, match=r"bad\.jsonl:2"):
            load_manifest(path)

    def test_rejects_non_object_record(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ManifestValidationError, match="must be a JSON object"):
            load_manifest(path)

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(ManifestValidationError, match="manifest not found"):
            load_manifest(tmp_path / "absent.jsonl")

    def test_validation_report_is_valid_for_good_manifest(self, tmp_path):
        path = write_manifest(tmp_path / "m.jsonl", [make_sample(tmp_path, "a")])
        report = validate_manifest_file(path)
        assert report.is_valid
        assert report.total_records == 1
        assert "VALID" in report.summary()

    def test_validation_report_collects_every_bad_record(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            json.dumps({"sample_id": "a"}) + "\n" + json.dumps({"sample_id": "b"}) + "\n",
            encoding="utf-8",
        )
        report = validate_manifest_file(path)
        assert not report.is_valid
        assert len(report.errors) == 2
        assert report.to_dict()["is_valid"] is False


class TestDescribeManifest:
    def test_summarizes_composition(self, tmp_path):
        samples = [
            make_sample(tmp_path, "a", modality="image", label="real"),
            make_sample(tmp_path, "b", modality="audio", label="fake"),
            make_sample(tmp_path, "c", modality="audio", label="fake"),
        ]
        summary = describe_manifest(samples)
        assert summary.total_samples == 3
        assert summary.modalities == {"audio": 2, "image": 1}
        assert summary.labels == {"fake": 2, "real": 1}
        assert summary.manifest_hash.startswith("sha256:")

    def test_summary_is_json_serializable(self, tmp_path):
        summary = describe_manifest([make_sample(tmp_path, "a")])
        assert json.loads(json.dumps(summary.to_dict()))

    def test_counting_helpers(self, tmp_path):
        samples = [
            make_sample(tmp_path, "a", label="real"),
            make_sample(tmp_path, "b", label="fake"),
        ]
        assert count_labels(samples) == {"real": 1, "fake": 1}
        assert count_modalities(samples) == {"image": 2}
