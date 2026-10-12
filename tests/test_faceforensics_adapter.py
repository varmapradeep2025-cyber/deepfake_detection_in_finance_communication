"""Tests for the FaceForensics++ adapter and its layout resolution.

What is under test
------------------
The adapter's decisions are all decisions about *configuration versus directory tree*:
which directories were declared, which of those exist, what label each one carries, and
what happens when they disagree. Those are testable against a synthetic tree of tiny
non-video files (see :mod:`tests.synthetic_faceforensics`) without any dataset material.

The property that gets the most attention here is that labels come from the configuration
and never from a directory name. A directory called ``DeepFakes`` gets label ``fake``
because the configuration says so; a configuration that says otherwise is honoured, and
that is asserted directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from datasets.adapters import adapter_names, resolve_adapter
from datasets.adapters.base import AdapterConfigurationError
from datasets.adapters.faceforensics_plus_plus import (
    FaceForensicsPlusPlusAdapter,
    LayoutResolutionError,
    is_unverified_license,
)
from datasets.ingest import IngestionRequest, ingest_dataset
from datasets.manifest import load_manifest
from datasets.schema import ManipulationType
from tests.synthetic_faceforensics import (
    SYNTHETIC_BYTES,
    SYNTHETIC_DATASET_ID,
    build_faceforensics_tree,
    build_verified_adapter_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "adapters" / "faceforensics_plus_plus.yaml"


def _shipped_adapter() -> FaceForensicsPlusPlusAdapter:
    """The adapter as it ships, built from the reviewed configuration."""
    adapter = FaceForensicsPlusPlusAdapter.from_config_file(SHIPPED_CONFIG)
    assert adapter is not None
    return adapter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestShippedConfiguration:
    """The configuration in the repository is the adapter's only source of truth."""

    def test_loads(self) -> None:
        adapter = _shipped_adapter()
        assert adapter.dataset_id == SYNTHETIC_DATASET_ID

    def test_declares_every_known_manipulation(self) -> None:
        declared = set(_shipped_adapter().label_mapping().as_dict())
        assert declared == {"original", "deepfakes", "face2face", "faceswap", "neuraltextures"}

    def test_maps_original_to_real_and_the_rest_to_fake(self) -> None:
        labels = _shipped_adapter().label_mapping().as_dict()
        assert labels["original"] == "real"
        assert {labels[name] for name in labels} == {"real", "fake"}

    def test_declares_a_manipulation_type_for_each_fake_family(self) -> None:
        manipulations = _shipped_adapter().manipulation_mapping().as_dict()
        assert manipulations == {
            "deepfakes": ManipulationType.SYNTHETIC_MEDIA.value,
            "face2face": ManipulationType.FACE_REENACTMENT.value,
            "faceswap": ManipulationType.FACE_SWAP.value,
            "neuraltextures": ManipulationType.FACIAL_ATTRIBUTE_EDIT.value,
        }

    def test_covers_both_canonical_labels(self) -> None:
        assert set(_shipped_adapter().label_coverage()) == {"real", "fake"}

    def test_ships_an_unverified_licence_so_acquisition_stays_blocked(self) -> None:
        adapter = _shipped_adapter()
        assert is_unverified_license(adapter.license_id)

    def test_refuses_to_ingest_while_the_licence_is_a_marker(self) -> None:
        with pytest.raises(AdapterConfigurationError) as excinfo:
            _shipped_adapter().validate_configuration()
        assert "pending_verification" in str(excinfo.value)

    def test_is_registered(self) -> None:
        assert "faceforensics_plus_plus" in adapter_names()
        assert resolve_adapter("faceforensics_plus_plus") is FaceForensicsPlusPlusAdapter


class TestConfigurationRejections:
    """Malformed configurations fail at load time, not halfway through an ingestion run."""

    def _write(self, tmp_path: Path, **overrides: object) -> Path:
        config = {
            "adapter": "faceforensics_plus_plus",
            "dataset_id": SYNTHETIC_DATASET_ID,
            "dataset_name": "Fixture",
            "license": "fixture_license",
            "manipulations": {
                "original": {"label_token": "original", "label": "real"},
                "DeepFakes": {
                    "label_token": "deepfakes",
                    "label": "fake",
                    "manipulation": "deepfakes",
                },
            },
            "required_manipulations": ["original"],
        }
        config.update(overrides)
        path = tmp_path / "config.yaml"
        path.write_text(_to_yaml(config), encoding="utf-8")
        return path

    def test_rejects_unknown_top_level_key(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, unexpected="value")
        with pytest.raises(AdapterConfigurationError) as excinfo:
            FaceForensicsPlusPlusAdapter.from_config_file(path)
        assert "unexpected" in str(excinfo.value)

    def test_rejects_a_label_outside_the_canonical_pair(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            manipulations={
                "original": {"label_token": "original", "label": "genuine"},
                "DeepFakes": {"label_token": "deepfakes", "label": "synthetic"},
            },
        )
        with pytest.raises(AdapterConfigurationError) as excinfo:
            FaceForensicsPlusPlusAdapter.from_config_file(path)
        assert "genuine" in str(excinfo.value)

    def test_rejects_a_manipulation_type_outside_the_vocabulary(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            manipulations={
                "original": {"label_token": "original", "label": "real"},
                "DeepFakes": {
                    "label_token": "deepfakes",
                    "label": "fake",
                    "manipulation": "reenactment_of_entire_body",
                },
            },
        )
        with pytest.raises(AdapterConfigurationError) as excinfo:
            FaceForensicsPlusPlusAdapter.from_config_file(path)
        assert "reenactment_of_entire_body" in str(excinfo.value)

    def test_rejects_requiring_a_directory_it_does_not_declare(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, required_manipulations=["original", "FaceSwap"])
        with pytest.raises(AdapterConfigurationError) as excinfo:
            FaceForensicsPlusPlusAdapter.from_config_file(path)
        assert "faceswap" in str(excinfo.value)

    def test_rejects_an_explicitly_empty_required_list(self, tmp_path: Path) -> None:
        # An explicit [] must not be quietly read as [original]: a configuration that says
        # "nothing is required" should fail rather than mean something else than it says.
        path = self._write(tmp_path, required_manipulations=[])
        with pytest.raises(AdapterConfigurationError) as excinfo:
            FaceForensicsPlusPlusAdapter.from_config_file(path)
        assert "empty" in str(excinfo.value)

    def test_defaults_required_manipulations_to_original_when_absent(self, tmp_path: Path) -> None:
        path = self._write(tmp_path)
        adapter = FaceForensicsPlusPlusAdapter.from_config_file(path)
        assert adapter is not None
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("DeepFakes",))
        # Nothing declares `original` as required in the file, and no `original` directory
        # exists, so the default requirement is the one that goes unmet.
        assert adapter.inspect(tree).missing_required == ("original",)

    def test_an_absent_source_directory_reports_absence_not_a_per_directory_miss(
        self, tmp_path: Path
    ) -> None:
        path = self._write(tmp_path)
        adapter = FaceForensicsPlusPlusAdapter.from_config_file(path)
        assert adapter is not None
        report = adapter.inspect(tmp_path / "nowhere")
        # Naming every required directory as missing would bury the real problem, which is
        # that there is no tree here at all.
        assert report.declared == ()
        assert any("not found" in message for message in report.errors)

    def test_rejects_a_duplicate_label_token(self, tmp_path: Path) -> None:
        path = self._write(
            tmp_path,
            manipulations={
                "original": {"label_token": "original", "label": "real"},
                "DeepFakes": {"label_token": "original", "label": "fake"},
            },
        )
        with pytest.raises(AdapterConfigurationError):
            FaceForensicsPlusPlusAdapter.from_config_file(path)


# ---------------------------------------------------------------------------
# Labels come from configuration, not from names
# ---------------------------------------------------------------------------


class TestLabelsComeFromConfiguration:
    """The central claim: nothing is inferred from a directory's name."""

    def test_a_directory_named_like_a_fake_is_real_when_configured_so(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            _to_yaml(
                {
                    "adapter": "faceforensics_plus_plus",
                    "dataset_id": SYNTHETIC_DATASET_ID,
                    "dataset_name": "Fixture",
                    "license": "fixture_license",
                    "manipulations": {
                        "DeepFakes": {"label_token": "control", "label": "real"},
                        "original": {"label_token": "treatment", "label": "fake"},
                    },
                    "required_manipulations": ["DeepFakes"],
                }
            ),
            encoding="utf-8",
        )
        adapter = FaceForensicsPlusPlusAdapter.from_config_file(config)
        assert adapter is not None
        build_faceforensics_tree(
            tmp_path / "tree", manipulations=("DeepFakes", "original"), flat=True
        )

        labels = {
            sample.path.parent.name: sample.label_token
            for sample in adapter.discover(tmp_path / "tree")
        }
        assert labels == {"DeepFakes": "control", "original": "treatment"}

    def test_an_undeclared_directory_yields_samples_with_no_label(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        build_faceforensics_tree(
            tmp_path / "tree",
            manipulations=("original",),
            extra_files={"FaceShifter": ("unknown_family.mp4",)},
        )
        undeclared = [
            sample
            for sample in adapter.discover(tmp_path / "tree")
            if sample.path.parent.name == "FaceShifter"
        ]
        assert len(undeclared) == 1
        assert undeclared[0].label_token is None
        assert undeclared[0].manipulation_token is None


# ---------------------------------------------------------------------------
# Layout resolution
# ---------------------------------------------------------------------------


class TestLayoutResolution:
    def test_reports_every_declared_directory_present(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree")
        report = adapter.inspect(tree)
        assert report.is_resolved
        assert report.missing_required == ()
        assert "NeuralTextures" in report.missing_optional
        assert report.errors == ()

    def test_a_missing_required_directory_is_an_error(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        build_faceforensics_tree(tmp_path / "tree", manipulations=("DeepFakes", "Face2Face"))
        report = adapter.inspect(tmp_path / "tree")
        assert not report.is_resolved
        assert report.missing_required == ("original",)
        assert any("original" in message for message in report.errors)

    def test_a_missing_optional_directory_is_a_warning_not_an_error(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        report = adapter.inspect(tree)
        assert report.is_resolved
        assert report.missing_optional
        assert any("FaceSwap" in message for message in report.warnings)

    def test_an_empty_present_directory_is_reported_not_padded(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        (tree / "DeepFakes").mkdir()
        report = adapter.inspect(tree)
        assert not report.is_resolved
        assert "DeepFakes" in report.empty_present

    def test_an_absent_source_directory_is_an_error(self, tmp_path: Path) -> None:
        report = _shipped_adapter().inspect(tmp_path / "nowhere")
        assert not report.is_resolved
        assert any("not found" in message for message in report.errors)

    def test_undeclared_directories_are_reported_and_counted(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(
            tmp_path / "tree",
            manipulations=("original",),
            extra_files={"FaceShifter": ("a.mp4", "b.mp4")},
        )
        report = adapter.inspect(tree)
        assert report.undeclared_directories == (("FaceShifter", 2, 2 * SYNTHETIC_BYTES),)
        assert report.unattributable_files == 2
        # The totals describe the directory on disk, so unattributable files are included
        # rather than quietly excluded: an understated size is a wrong size.
        assert report.total_files == report.file_count_for("original") + 2
        assert report.total_bytes == report.total_files * SYNTHETIC_BYTES

    def test_a_flat_layout_is_flagged_but_still_discovered(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",), flat=True)
        report = adapter.inspect(tree)
        assert report.is_resolved
        assert any("directly in the directory" in message for message in report.warnings)
        assert len(list(adapter.discover(tree))) == 2

    def test_reports_a_tree_with_no_authentic_directory(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("DeepFakes",))
        report = adapter.inspect(tree)
        assert any("genuine material" in message for message in report.warnings)

    def test_discovery_raises_when_a_required_directory_is_missing(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        build_faceforensics_tree(tmp_path / "tree", manipulations=("DeepFakes",))
        with pytest.raises(LayoutResolutionError) as excinfo:
            list(adapter.discover(tmp_path / "tree"))
        assert "original" in str(excinfo.value)

    def test_discovery_continues_past_a_missing_optional_directory(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        assert len(list(adapter.discover(tree))) == 4


# ---------------------------------------------------------------------------
# Determinism and serialization
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_yields_the_same_samples_in_the_same_order_every_time(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree")
        first = [(sample.sample_id, str(sample.path)) for sample in adapter.discover(tree)]
        second = [(sample.sample_id, str(sample.path)) for sample in adapter.discover(tree)]
        assert first == second

    def test_orders_samples_within_a_directory_by_relative_path(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        ids = [sample.sample_id for sample in adapter.discover(tree)]
        assert ids == sorted(ids)

    def test_sample_ids_are_relative_to_the_source_directory(self, tmp_path: Path) -> None:
        adapter = _shipped_adapter()
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        for sample in adapter.discover(tree):
            assert not Path(sample.sample_id).is_absolute()
            assert str(tree) not in sample.sample_id

    def test_honours_source_token_from_the_parent_directory(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            _to_yaml(
                {
                    "adapter": "faceforensics_plus_plus",
                    "dataset_id": SYNTHETIC_DATASET_ID,
                    "dataset_name": "Fixture",
                    "license": "fixture_license",
                    "source_token_from_parent_directory": True,
                    "manipulations": {"original": {"label_token": "original", "label": "real"}},
                    "required_manipulations": ["original"],
                }
            ),
            encoding="utf-8",
        )
        adapter = FaceForensicsPlusPlusAdapter.from_config_file(config)
        assert adapter is not None
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        sources = {sample.source_token for sample in adapter.discover(tree)}
        assert sources == {"000", "003"}

    def test_leaves_source_tokens_unset_by_default(self, tmp_path: Path) -> None:
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        for sample in _shipped_adapter().discover(tree):
            assert sample.source_token is None

    def test_report_serializes_to_json(self, tmp_path: Path) -> None:
        report = _shipped_adapter().inspect(build_faceforensics_tree(tmp_path / "tree"))
        assert json.loads(json.dumps(report.to_dict()))["is_resolved"] is True

    def test_report_renders_without_raising(self, tmp_path: Path) -> None:
        report = _shipped_adapter().inspect(build_faceforensics_tree(tmp_path / "tree"))
        rendered = report.render()
        assert "declared layout" in rendered
        assert "original" in rendered

    def test_declares_a_verified_adapter_that_ingests(self, tmp_path: Path) -> None:
        config = build_verified_adapter_config(
            tmp_path / "config.yaml", license_id="fixture_license"
        )
        adapter = FaceForensicsPlusPlusAdapter.from_config_file(config)
        assert adapter is not None
        adapter.validate_configuration()
        assert not is_unverified_license(adapter.license_id)


# ---------------------------------------------------------------------------
# End to end through the real ingestion pipeline
# ---------------------------------------------------------------------------


class TestThroughTheIngestionPipeline:
    """The adapter's output has to survive normalization, hashing, and manifest writing.

    Adapter tests can all pass while the pipeline rejects what the adapter emits, so these
    run the real `ingest_dataset` against a synthetic tree. Nothing is decoded: the fixture
    files are placeholders, and the pipeline only hashes them.
    """

    def _request(
        self,
        tree: Path,
        adapter: FaceForensicsPlusPlusAdapter,
        output_dir: Path,
        *,
        dry_run: bool,
    ) -> IngestionRequest:
        return IngestionRequest(
            dataset_id=SYNTHETIC_DATASET_ID,
            adapter=adapter,
            source_dir=tree,
            output_dir=output_dir,
            dry_run=dry_run,
        )

    @pytest.fixture
    def licensed_adapter(self, tmp_path: Path) -> FaceForensicsPlusPlusAdapter:
        """An adapter with an explicit licence, since the shipped one refuses to ingest."""
        config = build_verified_adapter_config(
            tmp_path / "adapter.yaml", license_id="test_fixture_license"
        )
        built = FaceForensicsPlusPlusAdapter.from_config_file(config)
        assert built is not None
        return built

    def test_a_dry_run_reports_the_label_split_from_the_configuration(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        tree = build_faceforensics_tree(
            tmp_path / "tree", manipulations=("original", "DeepFakes", "Face2Face", "FaceSwap")
        )
        summary = ingest_dataset(self._request(tree, licensed_adapter, tmp_path, dry_run=True))
        # 4 genuine files and 4 per fake family, from the declared layout.
        assert summary.real_samples == 4
        assert summary.fake_samples == 12

    def test_manipulation_types_reach_the_manifest(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        tree = build_faceforensics_tree(
            tmp_path / "tree", manipulations=("original", "DeepFakes", "Face2Face")
        )
        summary = ingest_dataset(self._request(tree, licensed_adapter, tmp_path, dry_run=True))
        assert summary.manipulation_counts[ManipulationType.SYNTHETIC_MEDIA.value] == 4
        assert summary.manipulation_counts[ManipulationType.FACE_REENACTMENT.value] == 4

    def test_genuine_samples_record_no_manipulation_rather_than_a_fake_one(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        summary = ingest_dataset(self._request(tree, licensed_adapter, tmp_path, dry_run=True))
        # `unknown` is a statement that the dataset said nothing, which differs from
        # `other`, meaning it said something that does not fit.
        assert summary.manipulation_counts["unknown"] == 4

    def test_writes_a_manifest_and_provenance_record(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original", "DeepFakes"))
        output = tmp_path / "manifests"
        summary = ingest_dataset(self._request(tree, licensed_adapter, output, dry_run=False))
        samples = load_manifest(summary.manifest_path)
        assert len(samples) == 8
        assert all(sample.sha256 for sample in samples)
        assert summary.provenance_path.is_file()

    def test_the_manifest_carries_the_digest_of_the_file_on_disk(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        import hashlib

        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        output = tmp_path / "manifests"
        summary = ingest_dataset(self._request(tree, licensed_adapter, output, dry_run=False))
        for sample in load_manifest(summary.manifest_path):
            actual = hashlib.sha256((tree / sample.filepath).read_bytes()).hexdigest()
            assert sample.sha256 == actual

    def test_the_manifest_hash_is_reproducible_across_runs(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original", "DeepFakes"))
        first = ingest_dataset(self._request(tree, licensed_adapter, tmp_path, dry_run=True))
        second = ingest_dataset(self._request(tree, licensed_adapter, tmp_path, dry_run=True))
        assert first.manifest_hash == second.manifest_hash

    def test_refuses_to_ingest_while_the_shipped_licence_is_a_marker(self, tmp_path: Path) -> None:
        tree = build_faceforensics_tree(tmp_path / "tree")
        with pytest.raises(AdapterConfigurationError):
            ingest_dataset(self._request(tree, _shipped_adapter(), tmp_path, dry_run=True))

    def test_reports_an_undeclared_directory_as_unlabelled_rather_than_guessing(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        tree = build_faceforensics_tree(
            tmp_path / "tree",
            manipulations=("original",),
            extra_files={"FaceShifter": ("unknown.mp4",)},
        )
        summary = ingest_dataset(self._request(tree, licensed_adapter, tmp_path, dry_run=True))
        # The file is visible and the run refuses it, naming it. Silently skipping it
        # would produce a clean-looking manifest that quietly omits data.
        assert summary.errors
        assert "FaceShifter" in summary.errors[0]
        assert "no label" in summary.errors[0]

    def test_modifies_no_source_media(
        self, tmp_path: Path, licensed_adapter: FaceForensicsPlusPlusAdapter
    ) -> None:
        tree = build_faceforensics_tree(tmp_path / "tree", manipulations=("original",))
        before = {
            path: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in tree.rglob("*")
            if path.is_file()
        }
        ingest_dataset(self._request(tree, licensed_adapter, tmp_path / "out", dry_run=False))
        after = {
            path: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in tree.rglob("*")
            if path.is_file()
        }
        assert after == before


def _to_yaml(value: object) -> str:
    """Render a small mapping as YAML. Avoids a second YAML dependency in fixtures."""
    import yaml

    return yaml.safe_dump(value, sort_keys=False, default_flow_style=False)
