"""Tests for the local acquisition and registration layer.

What is under test
------------------
The layer's whole purpose is refusal, so most of these tests assert that it refuses. A
directory being present is not permission; a verified licence is not a version; a version is
not an access date. Each of those gaps gets its own test, because each is a way the layer
could otherwise produce a confident-sounding record that nobody is cleared to stand behind.

The one test that *does* record an acquisition uses a synthetic tree of tiny non-video
files and a fixture registry whose licence fields are placeholders. It asserts that the
mechanics work — a record exists, is deterministic, copies nothing — and deliberately
asserts nothing about whether any real dataset may be used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasets.acquisition import (
    AcquisitionBlockedError,
    AcquisitionPlan,
    AcquisitionRecord,
    AcquisitionRequest,
    AcquisitionRequestError,
    load_acquisition_record,
    plan_acquisition,
    record_acquisition,
    record_path_for,
    registry_block_for,
    write_record,
)
from datasets.adapters.faceforensics_plus_plus import FaceForensicsPlusPlusAdapter
from datasets.registry import RegistryError
from tests.synthetic_faceforensics import (
    SYNTHETIC_DATASET_ID,
    build_faceforensics_tree,
    build_verified_adapter_config,
    verified_registry_yaml,
)

FIXTURE_LICENSE = "test_fixture_license"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A complete, resolved synthetic FaceForensics++ tree."""
    return build_faceforensics_tree(tmp_path / "raw" / SYNTHETIC_DATASET_ID)


@pytest.fixture
def adapter_config(tmp_path: Path) -> Path:
    """An adapter configuration with an explicit licence, so the licence gate is passed."""
    return build_verified_adapter_config(tmp_path / "adapter.yaml", license_id=FIXTURE_LICENSE)


@pytest.fixture
def adapter(adapter_config: Path) -> FaceForensicsPlusPlusAdapter:
    built = FaceForensicsPlusPlusAdapter.from_config_file(adapter_config)
    assert built is not None
    return built


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """A fixture registry entry that has cleared every gate."""
    path = tmp_path / "registry.yaml"
    path.write_text(verified_registry_yaml(), encoding="utf-8")
    return path


def _request(
    tree: Path,
    adapter: FaceForensicsPlusPlusAdapter,
    registry: Path,
    **overrides: object,
) -> AcquisitionRequest:
    values: dict[str, object] = {
        "dataset_id": SYNTHETIC_DATASET_ID,
        "source_dir": tree,
        "adapter": adapter,
        "registry_path": registry,
    }
    values.update(overrides)
    return AcquisitionRequest(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Planning is read-only
# ---------------------------------------------------------------------------


class TestPlanningIsReadOnly:
    def test_writes_nothing_even_when_the_plan_is_clean(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        before = sorted(path.name for path in tmp_file_list(tree.parent))
        plan_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-15",
                dataset_version="0.0.0-test_fixture",
            )
        )
        assert sorted(path.name for path in tmp_file_list(tree.parent)) == before

    def test_modifies_no_media(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        digests = {path: path.stat().st_mtime_ns for path in tmp_file_list(tree)}
        plan_acquisition(_request(tree, adapter, registry))
        assert {path: path.stat().st_mtime_ns for path in tmp_file_list(tree)} == digests

    def test_reports_a_usable_layout_without_a_licence(
        self, tmp_path: Path, tree: Path, adapter_config: Path
    ) -> None:
        shipped = FaceForensicsPlusPlusAdapter.from_config_file(
            Path("configs/adapters/faceforensics_plus_plus.yaml")
        )
        assert shipped is not None
        plan = plan_acquisition(
            AcquisitionRequest(
                dataset_id=SYNTHETIC_DATASET_ID,
                source_dir=tree,
                adapter=shipped,
                registry_path=tmp_path / "absent-registry.yaml",
            )
        )
        # Layout inspection is answerable with no licence at all, which is the point: a
        # person checking an unpacked archive should not have to clear it to be told
        # whether it looks right.
        assert plan.is_usable
        assert plan.is_blocked


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_refuses_when_the_licence_is_not_verified(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, tmp_path: Path
    ) -> None:
        registry = tmp_path / "registry.yaml"
        registry.write_text(
            verified_registry_yaml(license_status="pending_verification"), encoding="utf-8"
        )
        plan = plan_acquisition(_request(tree, adapter, registry))
        assert plan.is_blocked
        assert any("license_status is 'pending_verification'" in b for b in plan.blockers)

    @pytest.mark.parametrize("status", ["unknown", "not_verified", "forbidden"])
    def test_refuses_every_non_verified_licence_status(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, tmp_path: Path, status: str
    ) -> None:
        registry = tmp_path / "registry.yaml"
        registry.write_text(verified_registry_yaml(license_status=status), encoding="utf-8")
        assert plan_acquisition(_request(tree, adapter, registry)).is_blocked

    def test_refuses_an_unregistered_dataset(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(_request(tree, adapter, registry, dataset_id="not_registered"))
        assert any("not registered" in blocker for blocker in plan.blockers)

    @pytest.mark.parametrize("status", ["blocked", "deprecated"])
    def test_refuses_a_status_that_forbids_acquisition(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, tmp_path: Path, status: str
    ) -> None:
        registry = tmp_path / "registry.yaml"
        registry.write_text(verified_registry_yaml(status=status), encoding="utf-8")
        plan = plan_acquisition(_request(tree, adapter, registry))
        assert any(f"status {status!r} forbids acquisition" in b for b in plan.blockers)

    def test_refuses_when_the_adapter_licence_is_still_a_marker(
        self, tree: Path, registry: Path
    ) -> None:
        shipped = FaceForensicsPlusPlusAdapter.from_config_file(
            Path("configs/adapters/faceforensics_plus_plus.yaml")
        )
        assert shipped is not None
        plan = plan_acquisition(_request(tree, shipped, registry))
        assert any("unverified marker" in blocker for blocker in plan.blockers)

    def test_refuses_when_a_required_directory_is_missing(
        self, tmp_path: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        partial = build_faceforensics_tree(
            tmp_path / "raw" / SYNTHETIC_DATASET_ID, manipulations=("DeepFakes",)
        )
        plan = plan_acquisition(_request(partial, adapter, registry))
        assert plan.is_blocked
        assert any("original" in blocker for blocker in plan.blockers)

    def test_refuses_when_the_source_directory_does_not_exist(
        self, tmp_path: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(_request(tmp_path / "nowhere", adapter, registry))
        assert plan.is_blocked
        assert any("not found" in blocker for blocker in plan.blockers)

    def test_refuses_an_impossible_access_date(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(_request(tree, adapter, registry, access_date="2026-13-45"))
        assert any("2026-13-45" in blocker for blocker in plan.blockers)

    def test_refuses_an_unverified_dataset_version(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(_request(tree, adapter, registry, dataset_version="unknown"))
        assert any("unverified marker" in blocker for blocker in plan.blockers)

    def test_refuses_when_the_registry_cannot_be_read(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, tmp_path: Path
    ) -> None:
        broken = tmp_path / "broken.yaml"
        broken.write_text("datasets: [not, a, mapping]\n", encoding="utf-8")
        plan = plan_acquisition(_request(tree, adapter, broken))
        assert any(blocker.startswith("registry:") for blocker in plan.blockers)

    def test_a_refusal_names_every_blocker_at_once(
        self, tmp_path: Path, adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        registry = tmp_path / "registry.yaml"
        registry.write_text(
            verified_registry_yaml(license_status="pending_verification", status="blocked"),
            encoding="utf-8",
        )
        plan = plan_acquisition(
            _request(
                tmp_path / "nowhere",
                adapter,
                registry,
                dataset_version="unknown",
                access_date="nonsense",
            )
        )
        # One run reports everything wrong, rather than making the operator fix problems
        # one at a time in the order they happen to trip.
        assert len(plan.blockers) >= 4

    def test_recording_refuses_and_writes_nothing(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, tmp_path: Path
    ) -> None:
        registry = tmp_path / "registry.yaml"
        registry.write_text(
            verified_registry_yaml(license_status="pending_verification"), encoding="utf-8"
        )
        with pytest.raises(AcquisitionBlockedError) as excinfo:
            record_acquisition(
                _request(
                    tree,
                    adapter,
                    registry,
                    access_date="2026-01-15",
                    dataset_version="0.0.0-test_fixture",
                )
            )
        assert "Nothing was written" in str(excinfo.value)
        assert not list(tmp_path.glob("*.acquisition.json"))


class TestWarningsThatDoNotBlock:
    def test_warns_when_no_access_date_was_supplied(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(_request(tree, adapter, registry))
        assert any("--access-date" in warning for warning in plan.warnings)

    def test_warns_when_no_version_was_supplied(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(_request(tree, adapter, registry))
        assert any("--dataset-version" in warning for warning in plan.warnings)

    def test_warns_when_terms_acceptance_is_still_required(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, tmp_path: Path
    ) -> None:
        registry = tmp_path / "registry.yaml"
        registry.write_text(verified_registry_yaml(), encoding="utf-8")
        plan = plan_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-15",
                dataset_version="0.0.0-test_fixture",
            )
        )
        assert plan.blockers == ()
        assert any("requires_terms_acceptance" in warning for warning in plan.warnings)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecording:
    def _recorded(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> AcquisitionRecord:
        return record_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-15",
                dataset_version="0.0.0-test_fixture",
            )
        )

    def test_records_the_supplied_facts(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        record = self._recorded(tree, adapter, registry)
        assert record.dataset_id == SYNTHETIC_DATASET_ID
        assert record.access_date == "2026-01-15"
        assert record.dataset_version == "0.0.0-test_fixture"
        assert record.license == FIXTURE_LICENSE
        assert record.registry_status == "approved"

    def test_states_plainly_that_nothing_was_copied(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        payload = self._recorded(tree, adapter, registry).to_dict()
        assert payload["bytes_copied"] is False
        assert payload["media_modified"] is False
        assert payload["acquisition_method"] == "local_directory"

    def test_includes_the_layout_it_recorded_against(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        payload = self._recorded(tree, adapter, registry).to_dict()
        assert payload["layout"]["is_resolved"] is True
        assert payload["totals"]["candidate_files"] == payload["layout"]["total_files"]
        assert payload["totals"]["total_bytes"] > 0

    def test_records_the_path_as_given_rather_than_resolved(
        self, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        tree = build_faceforensics_tree(Path("relative") / SYNTHETIC_DATASET_ID)
        record = record_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-15",
                dataset_version="0.0.0-test_fixture",
            )
        )
        # A tracked record should not carry one machine's absolute layout.
        assert not Path(record.source_dir).is_absolute()

    def test_is_deterministic(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        first = self._recorded(tree, adapter, registry)
        second = self._recorded(tree, adapter, registry)
        assert first.content_hash() == second.content_hash()
        assert first.to_json() == second.to_json()

    def test_the_hash_changes_when_a_fact_changes(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        first = self._recorded(tree, adapter, registry)
        second = record_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-16",
                dataset_version="0.0.0-test_fixture",
            )
        )
        assert first.content_hash() != second.content_hash()

    def test_round_trips_through_disk(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path, tmp_path: Path
    ) -> None:
        record = self._recorded(tree, adapter, registry)
        path = write_record(record, tmp_path / "record.json")
        assert load_acquisition_record(path) == record

    def test_a_written_record_is_stable_on_disk(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path, tmp_path: Path
    ) -> None:
        record = self._recorded(tree, adapter, registry)
        destination = tmp_path / "record.json"
        write_record(record, destination)
        first = destination.read_text(encoding="utf-8")
        write_record(record, destination)
        assert destination.read_text(encoding="utf-8") == first

    def test_records_to_the_default_path(
        self, tmp_path: Path, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        path = write_record(
            self._recorded(tree, adapter, registry),
            record_path_for(SYNTHETIC_DATASET_ID, base_dir=tmp_path / "manifests"),
        )
        assert path.name == f"{SYNTHETIC_DATASET_ID}.acquisition.json"

    def test_says_it_copied_nothing_after_a_round_trip(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path, tmp_path: Path
    ) -> None:
        record = self._recorded(tree, adapter, registry)
        path = write_record(record, tmp_path / "record.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["bytes_copied"] is False
        assert load_acquisition_record(path).to_dict() == record.to_dict()


class TestRegistryPatch:
    def test_emits_the_acquisition_block_without_applying_it(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        before = registry.read_text(encoding="utf-8")
        record = record_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-15",
                dataset_version="0.0.0-test_fixture",
            )
        )
        block = registry_block_for(record)
        # The registry is a reviewed file. The block is emitted so a person can paste it,
        # not applied so a script can decide.
        assert registry.read_text(encoding="utf-8") == before
        assert "acquired: true" in block
        assert "access_date: 2026-01-15" in block
        assert "version: 0.0.0-test_fixture" in block
        assert record.content_hash() in block

    def test_points_at_the_record_that_backs_the_block(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        record = record_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-15",
                dataset_version="0.0.0-test_fixture",
            )
        )
        assert f"{SYNTHETIC_DATASET_ID}.acquisition.json" in registry_block_for(record)


class TestRecordValidation:
    def _valid_fields(self) -> dict[str, object]:
        return {
            "dataset_id": "example",
            "dataset_name": "Example",
            "dataset_version": "1.0.0",
            "access_date": "2026-01-15",
            "source_dir": "datasets/raw/example",
            "license": "example_license",
            "adapter_name": "faceforensics_plus_plus",
            "adapter_version": "0.1.0",
            "adapter_config": None,
            "layout": {},
            "registry_status": "approved",
        }

    def test_accepts_a_complete_record(self) -> None:
        assert AcquisitionRecord(**self._valid_fields()).access_date == "2026-01-15"

    @pytest.mark.parametrize("bad", ["15-01-2026", "2026/01/15", "yesterday", "", "20260115"])
    def test_rejects_a_non_iso_access_date(self, bad: str) -> None:
        with pytest.raises(AcquisitionRequestError):
            AcquisitionRecord(**{**self._valid_fields(), "access_date": bad})

    def test_rejects_an_impossible_date(self) -> None:
        with pytest.raises(AcquisitionRequestError):
            AcquisitionRecord(**{**self._valid_fields(), "access_date": "2026-02-30"})

    def test_rejects_an_unverified_version(self) -> None:
        with pytest.raises(AcquisitionRequestError) as excinfo:
            AcquisitionRecord(**{**self._valid_fields(), "dataset_version": "unknown"})
        assert "unverified marker" in str(excinfo.value)

    def test_rejects_an_unverified_licence(self) -> None:
        with pytest.raises(AcquisitionRequestError):
            AcquisitionRecord(**{**self._valid_fields(), "license": "pending_verification"})

    def test_rejects_a_blank_dataset_id(self) -> None:
        with pytest.raises(AcquisitionRequestError):
            AcquisitionRecord(**{**self._valid_fields(), "dataset_id": "  "})


class TestRecordLoading:
    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(AcquisitionRequestError) as excinfo:
            load_acquisition_record(tmp_path / "absent.json")
        assert "not found" in str(excinfo.value)

    def test_rejects_malformed_json(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(AcquisitionRequestError) as excinfo:
            load_acquisition_record(path)
        assert "malformed" in str(excinfo.value)

    def test_rejects_a_record_missing_a_field(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        path.write_text(json.dumps({"bytes_copied": False}), encoding="utf-8")
        with pytest.raises(AcquisitionRequestError) as excinfo:
            load_acquisition_record(path)
        assert "missing required field" in str(excinfo.value)

    def test_rejects_a_record_claiming_media_was_copied(self, tmp_path: Path) -> None:
        path = tmp_path / "record.json"
        path.write_text(json.dumps({"bytes_copied": True}), encoding="utf-8")
        with pytest.raises(AcquisitionRequestError) as excinfo:
            load_acquisition_record(path)
        assert "never copies dataset media" in str(excinfo.value)


class TestRequestValidation:
    def test_rejects_a_blank_dataset_id(self, tmp_path: Path) -> None:
        with pytest.raises(AcquisitionRequestError):
            AcquisitionRequest(dataset_id="  ", source_dir=tmp_path, adapter=object())  # type: ignore[arg-type]

    def test_refuses_an_adapter_that_cannot_inspect(self, tmp_path: Path, registry: Path) -> None:
        class Blind:
            name = "blind"

            adapter_version = "0.0.0"

        with pytest.raises(AcquisitionRequestError) as excinfo:
            plan_acquisition(
                AcquisitionRequest(
                    dataset_id="example",
                    source_dir=tmp_path,
                    adapter=Blind(),  # type: ignore[arg-type]
                    registry_path=registry,
                )
            )
        assert "cannot inspect" in str(excinfo.value)


class TestRendering:
    def test_a_blocked_plan_says_nothing_was_written(
        self, tmp_path: Path, adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        registry = tmp_path / "registry.yaml"
        registry.write_text(
            verified_registry_yaml(license_status="pending_verification"), encoding="utf-8"
        )
        plan = plan_acquisition(_request(tmp_path / "raw", adapter, registry))
        assert "Nothing was written" in plan.render()

    def test_a_clean_plan_says_it_is_ready(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(
            _request(
                tree,
                adapter,
                registry,
                access_date="2026-01-15",
                dataset_version="0.0.0-test_fixture",
            )
        )
        assert plan.is_usable
        assert "ready to record" in plan.render()

    def test_renders_as_json(
        self, tree: Path, adapter: FaceForensicsPlusPlusAdapter, registry: Path
    ) -> None:
        plan = plan_acquisition(_request(tree, adapter, registry))
        payload = json.loads(json.dumps(plan.to_dict()))
        assert payload["dataset_id"] == SYNTHETIC_DATASET_ID
        assert payload["layout_usable"] is True


def tmp_file_list(root: Path) -> list[Path]:
    """Every file under ``root``, for asserting nothing changed."""
    return [path for path in root.rglob("*") if path.is_file()]


def test_registry_error_type_is_importable() -> None:
    # Guards the import the module relies on for its failure path.
    assert issubclass(RegistryError, Exception)
    assert AcquisitionPlan is not None
