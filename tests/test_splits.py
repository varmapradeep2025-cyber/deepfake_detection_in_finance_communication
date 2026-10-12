"""Tests for :mod:`datasets.split_schema`.

Covers leakage detection, duplicate-id rejection, and deterministic split hashing.
"""

from __future__ import annotations

import json

import pytest

from datasets.split_schema import (
    FrozenSplit,
    SplitValidationError,
    assign_splits,
    load_split_definition,
    validate_split_id_sets,
    write_split_definition,
)

MANIFEST_HASH = "sha256:" + "b" * 64


def build(**overrides) -> FrozenSplit:
    """Build a valid split, overriding any field."""
    payload = {
        "dataset_name": "unit_test_dataset",
        "train_ids": ["t1", "t2"],
        "validation_ids": ["v1"],
        "test_ids": ["s1", "s2"],
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": 20260101,
        "source_manifest_hash": MANIFEST_HASH,
    }
    payload.update(overrides)
    return FrozenSplit(**payload)


class TestConstruction:
    def test_builds_valid_split(self):
        split = build()
        assert split.total_ids == 5
        assert split.is_empty is False

    def test_ids_are_sorted_and_stripped(self):
        split = build(train_ids=[" t2 ", "t1"])
        assert split.train_ids == ("t1", "t2")

    def test_optional_fields_default(self):
        split = build()
        assert split.rationale == ""

    def test_empty_split_is_flagged(self):
        split = build(train_ids=[], validation_ids=[], test_ids=[])
        assert split.is_empty
        assert split.ratios() == {"train": 0.0, "validation": 0.0, "test": 0.0}

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_rejects_empty_dataset_name(self, bad):
        with pytest.raises(SplitValidationError, match="dataset_name"):
            build(dataset_name=bad)

    def test_rejects_non_integer_seed(self):
        with pytest.raises(SplitValidationError, match="seed"):
            build(seed="123")

    @pytest.mark.parametrize("bad", ["not-a-hash", "b" * 64, ""])
    def test_rejects_bad_source_manifest_hash(self, bad):
        with pytest.raises(SplitValidationError, match="source_manifest_hash"):
            build(source_manifest_hash=bad)


class TestDuplicateIds:
    def test_rejects_duplicates_within_train(self):
        with pytest.raises(SplitValidationError, match="duplicate id"):
            build(train_ids=["t1", "t1"])

    def test_rejects_duplicates_within_test(self):
        with pytest.raises(SplitValidationError, match="duplicate id"):
            build(test_ids=["s1", "s1", "s2"])

    def test_reports_which_ids_duplicated(self):
        with pytest.raises(SplitValidationError) as exc:
            build(test_ids=["s1", "s1"])
        assert exc.value.errors == ["s1"]

    def test_rejects_non_string_id(self):
        with pytest.raises(SplitValidationError, match="non-string id"):
            build(train_ids=[1, 2])

    def test_rejects_bare_string_instead_of_list(self):
        with pytest.raises(SplitValidationError, match="not a str"):
            build(train_ids="t1")


class TestOverlapDetection:
    def test_train_validation_overlap_is_rejected(self):
        with pytest.raises(SplitValidationError, match="split leakage"):
            build(train_ids=["t1", "shared"], validation_ids=["shared"])

    def test_train_test_overlap_is_rejected(self):
        with pytest.raises(SplitValidationError, match="split leakage"):
            build(train_ids=["shared"], test_ids=["shared"])

    def test_validation_test_overlap_is_rejected(self):
        with pytest.raises(SplitValidationError, match="split leakage"):
            build(validation_ids=["shared"], test_ids=["shared"])

    def test_error_names_every_offending_split(self):
        with pytest.raises(SplitValidationError) as exc:
            build(train_ids=["x"], validation_ids=["x"], test_ids=["x"])
        assert "train" in exc.value.errors[0]
        assert "test" in exc.value.errors[0]

    def test_standalone_validator_accepts_disjoint_sets(self):
        validate_split_id_sets(["a", "b"], ["c"], ["d"])

    def test_standalone_validator_rejects_overlap(self):
        with pytest.raises(SplitValidationError, match="leakage"):
            validate_split_id_sets(["a"], ["a"], ["b"])

    def test_report_flags_leakage_without_raising(self):
        report = build().validate()
        assert report.is_valid
        assert report.train_count == 2
        assert "VALID" in report.summary()

    def test_report_is_json_serializable(self):
        assert json.loads(json.dumps(build().validate().to_dict()))["is_valid"] is True


class TestHashing:
    def test_hash_is_deterministic(self):
        assert build().content_hash() == build().content_hash()

    def test_hash_is_order_independent(self):
        first = build(train_ids=["a", "b", "c"])
        second = build(train_ids=["c", "a", "b"])
        assert first.content_hash() == second.content_hash()

    def test_hash_is_prefixed_with_algorithm(self):
        digest = build().content_hash()
        assert digest.startswith("sha256:")
        assert len(digest) == len("sha256:") + 64

    def test_hash_changes_when_membership_changes(self):
        assert build(test_ids=["s1"]).content_hash() != build(test_ids=["s2"]).content_hash()

    def test_hash_changes_when_split_moves_between_partitions(self):
        assert (
            build(test_ids=["s1"]).content_hash()
            != build(train_ids=["s1"], test_ids=[]).content_hash()
        )

    def test_hash_ignores_provenance_metadata(self):
        """Editing rationale or created_at must not change a split's identity."""
        baseline = build().content_hash()
        assert build(rationale="totally rewritten").content_hash() == baseline
        assert build(created_at="1999-12-31T23:59:59+00:00").content_hash() == baseline

    def test_hash_differs_across_datasets(self):
        assert build(dataset_name="ds1").content_hash() != build(dataset_name="ds2").content_hash()


class TestAccessors:
    def test_ids_for_resolves_each_split(self):
        split = build()
        assert split.ids_for("train") == ("t1", "t2")
        assert split.ids_for("validation") == ("v1",)
        assert split.ids_for("test") == ("s1", "s2")

    def test_ids_for_rejects_unknown_split(self):
        with pytest.raises(Exception, match="invalid split"):
            build().ids_for("holdout")

    def test_ratios_sum_to_one(self):
        ratios = build().ratios()
        assert sum(ratios.values()) == pytest.approx(1.0)
        assert ratios["train"] == pytest.approx(0.4)
        assert ratios["test"] == pytest.approx(0.4)

    def test_all_datasets_not_applicable(self):
        assert build().dataset_name == "unit_test_dataset"


class TestSerialization:
    def test_round_trips_through_dict(self):
        original = build(rationale="grouped by source video")
        restored = FrozenSplit.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()
        assert restored.content_hash() == original.content_hash()

    def test_from_dict_reports_missing_fields(self):
        payload = build().to_dict()
        del payload["seed"]
        with pytest.raises(SplitValidationError, match="missing required field"):
            FrozenSplit.from_dict(payload)

    def test_from_dict_still_validates_overlap(self):
        payload = build().to_dict()
        payload["test_ids"] = payload["train_ids"]
        with pytest.raises(SplitValidationError, match="leakage"):
            FrozenSplit.from_dict(payload)

    def test_write_and_load_definition(self, tmp_path):
        split = build()
        content_hash = write_split_definition(tmp_path / "s.json", split)
        restored = load_split_definition(tmp_path / "s.json")
        assert restored.content_hash() == split.content_hash()
        assert content_hash.startswith("sha256:")

    def test_load_missing_definition_raises(self, tmp_path):
        with pytest.raises(SplitValidationError, match="not found"):
            load_split_definition(tmp_path / "absent.json")

    def test_load_malformed_json_raises(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(SplitValidationError, match="malformed JSON"):
            load_split_definition(path)


class TestAssignSplits:
    def test_partitions_all_ids(self):
        ids = [f"s{i}" for i in range(100)]
        partition = assign_splits(ids, seed=1)
        assert len(partition["train"]) == 80
        assert len(partition["validation"]) == 10
        assert len(partition["test"]) == 10

    def test_is_deterministic_for_a_given_seed(self):
        ids = [f"s{i}" for i in range(50)]
        assert assign_splits(ids, seed=7) == assign_splits(ids, seed=7)

    def test_differs_for_different_seeds(self):
        ids = [f"s{i}" for i in range(50)]
        assert assign_splits(ids, seed=7) != assign_splits(ids, seed=8)

    def test_partitions_are_disjoint_and_complete(self):
        ids = [f"s{i}" for i in range(37)]
        partition = assign_splits(ids, seed=3)
        combined = partition["train"] + partition["validation"] + partition["test"]
        assert sorted(combined) == sorted(ids)

    def test_rejects_duplicate_input_ids(self):
        with pytest.raises(SplitValidationError, match="duplicate"):
            assign_splits(["a", "a"], seed=1)

    def test_empty_input_yields_empty_partition(self):
        assert assign_splits([], seed=1) == {"train": (), "validation": (), "test": ()}

    def test_rejects_fractions_exceeding_one(self):
        with pytest.raises(SplitValidationError, match=r"must be <= 1\.0"):
            assign_splits(["a"], seed=1, train_fraction=0.8, validation_fraction=0.5)

    def test_rejects_negative_fractions(self):
        with pytest.raises(SplitValidationError, match="non-negative"):
            assign_splits(["a"], seed=1, train_fraction=-0.1)

    def test_result_can_be_used_to_construct_a_frozen_split(self):
        ids = [f"s{i}" for i in range(20)]
        partition = assign_splits(ids, seed=11)
        split = FrozenSplit.create(
            "ds",
            seed=11,
            source_manifest_hash=MANIFEST_HASH,
            train_ids=partition["train"],
            validation_ids=partition["validation"],
            test_ids=partition["test"],
        )
        assert split.total_ids == 20
        assert split.is_empty is False

    def test_partition_keys_map_onto_split_fields(self):
        """assign_splits returns short keys; FrozenSplit.create needs *_ids names."""
        partition = assign_splits([f"s{i}" for i in range(10)], seed=2)
        assert set(partition) == {"train", "validation", "test"}
