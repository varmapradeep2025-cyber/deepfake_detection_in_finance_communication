"""Tests for the dataset ingestion layer.

Covers the adapter contract, the controlled normalization layer, the ingestion
pipeline and its summary, dry-run behaviour, determinism, and the CLI.

Every test operates on the synthetic fixture in :mod:`tests.synthetic_dataset`, written
into pytest's ``tmp_path``. That fixture is a handful of bytes and is **not research
data**; see that module's docstring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from datasets.adapters import (
    AdapterConfigurationError,
    DatasetAdapter,
    FaceForensicsPlusPlusAdapter,
    RawSample,
    TemplateAdapter,
    adapter_names,
    register_adapter,
    resolve_adapter,
    validate_dataset_id,
)
from datasets.ingest import (
    HashVerificationStatus,
    IngestionError,
    IngestionRequest,
    build_media_sample,
    ingest_dataset,
    main,
)
from datasets.manifest import calculate_file_sha256, load_manifest
from datasets.normalization import (
    LabelMapping,
    ManipulationMapping,
    ModalityMapping,
    NoLabelMappingError,
    NormalizationError,
    UnknownLabelError,
    UnknownManipulationError,
    UnknownModalityError,
    normalize_token,
)
from datasets.provenance import (
    DatasetProvenance,
    ProvenanceError,
    load_provenance,
    provenance_path_for,
)
from datasets.schema import ManipulationType, MediaSample, Modality, SampleLabel
from tests.synthetic_dataset import (
    SYNTHETIC_LICENSE,
    SYNTHETIC_RELATIVE_FILES,
    build_synthetic_dataset,
    synthetic_template_config,
)

DATASET_ID = "synthetic_unit_test_fixture"
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """The synthetic fixture dataset, freshly written into ``tmp_path``."""
    return build_synthetic_dataset(tmp_path / "raw" / DATASET_ID)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Destination for manifests. Does not exist until a run creates it."""
    return tmp_path / "manifests"


@pytest.fixture
def adapter() -> TemplateAdapter:
    """A template adapter configured for the synthetic fixture."""
    return TemplateAdapter.from_config(synthetic_template_config(), dataset_id=DATASET_ID)


@pytest.fixture
def request_factory(output_dir: Path):
    """Build an :class:`IngestionRequest` with sensible test defaults."""

    def make(
        adapter: DatasetAdapter,
        source: Path,
        **overrides: object,
    ) -> IngestionRequest:
        payload: dict[str, object] = {
            "dataset_id": DATASET_ID,
            "adapter": adapter,
            "source_dir": source,
            "output_dir": output_dir,
        }
        payload.update(overrides)
        return IngestionRequest(**payload)  # type: ignore[arg-type]

    return make


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------


class IncompleteAdapter(DatasetAdapter):
    """Deliberately incomplete, to prove the base class refuses to instantiate it."""

    name = "incomplete"


class MinimalAdapter(DatasetAdapter):
    """Smallest usable adapter, used to test the contract itself."""

    name = "minimal"
    adapter_version = "7"

    def __init__(self, **kwargs: object) -> None:
        options: dict[str, object] = {
            "dataset_name": "minimal",
            "license_id": "CC0-1.0",
            "dataset_version": "1",
        }
        options.update(kwargs)
        super().__init__(**options)  # type: ignore[arg-type]
        self._labels = LabelMapping({"real": "real", "fake": "fake"})

    def label_mapping(self) -> LabelMapping:
        return self._labels

    def discover(self, source_dir: Path):
        for path in sorted(source_dir.rglob("*.bin")):
            yield RawSample(sample_id=path.stem, path=path, label_token="real")


class TestAdapterInterface:
    def test_abstract_adapter_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteAdapter(dataset_name="d", license_id="L")

    def test_discovery_is_abstract(self):
        assert DatasetAdapter.discover.__isabstractmethod__ is True
        assert DatasetAdapter.label_mapping.__isabstractmethod__ is True

    def test_manipulation_mapping_defaults_to_empty(self):
        adapter = MinimalAdapter()
        assert adapter.manipulation_mapping().tokens == ()

    def test_modality_mapping_defaults_to_empty(self):
        assert MinimalAdapter().modality_mapping().tokens == ()

    def test_requires_a_license(self):
        with pytest.raises(AdapterConfigurationError, match="license_id"):
            MinimalAdapter(license_id="   ")

    def test_accepts_optional_source_reference_as_none(self):
        assert MinimalAdapter().source_reference is None

    def test_blank_source_reference_becomes_none(self):
        assert MinimalAdapter(source_reference="  ").source_reference is None

    def test_describe_reports_declared_vocabulary(self):
        described = MinimalAdapter().describe()
        assert described["adapter"] == "minimal"
        assert described["adapter_version"] == "7"
        assert described["label_mappings"] == {"fake": "fake", "real": "real"}
        assert described["label_coverage"] == ["fake", "real"]

    def test_validate_configuration_passes_for_a_declared_adapter(self):
        MinimalAdapter().validate_configuration()

    def test_validate_configuration_rejects_an_adapter_with_no_labels(self):
        class NoLabels(MinimalAdapter):
            name = "no-labels"

            def label_mapping(self) -> LabelMapping:
                return LabelMapping({})

        with pytest.raises(AdapterConfigurationError, match="no label mapping"):
            NoLabels().validate_configuration()

    def test_raw_sample_rejects_empty_sample_id(self):
        with pytest.raises(AdapterConfigurationError, match="sample_id"):
            RawSample(sample_id="  ", path=Path("x"))

    def test_raw_sample_coerces_path_and_metadata(self):
        raw = RawSample(sample_id="s", path="a/b.bin")  # type: ignore[arg-type]
        assert isinstance(raw.path, Path)
        assert raw.metadata == {}
        assert raw.label_token is None


class TestAdapterRegistry:
    #: Every adapter this project ships. Adding a dataset adapter is a deliberate decision
    #: about how a dataset is read, so it means editing this list too — which puts the
    #: decision in a diff rather than letting a new adapter appear silently.
    EXPECTED_ADAPTERS = ("faceforensics_plus_plus", "template")

    def test_template_is_registered(self):
        assert "template" in adapter_names()

    def test_ships_exactly_the_expected_adapters(self):
        """A new adapter may not appear without this list being updated on purpose."""
        assert adapter_names() == self.EXPECTED_ADAPTERS

    def test_every_dataset_adapter_ships_a_configuration_file(self):
        """A dataset adapter is driven by a reviewed file, so one must exist for each.

        The template is excluded: it is the worked example of the configuration format,
        exercised from test fixtures, not a dataset anyone reads.
        """
        configs = REPO_ROOT / "configs" / "adapters"
        on_disk = {path.stem for path in configs.glob("*.yaml")}
        dataset_adapters = set(self.EXPECTED_ADAPTERS) - {"template"}
        assert on_disk == dataset_adapters

    def test_resolve_returns_the_class(self):
        assert resolve_adapter("template") is TemplateAdapter

    def test_resolve_returns_the_faceforensics_adapter(self):
        assert resolve_adapter("faceforensics_plus_plus") is FaceForensicsPlusPlusAdapter

    def test_unknown_adapter_lists_the_options(self):
        with pytest.raises(AdapterConfigurationError, match="template"):
            resolve_adapter("nonexistent")

    def test_registration_rejects_a_duplicate_name(self):
        with pytest.raises(AdapterConfigurationError, match="already registered"):
            register_adapter(TemplateAdapter)

    def test_registration_rejects_an_empty_name(self):
        with pytest.raises(AdapterConfigurationError, match="non-empty"):
            register_adapter(TemplateAdapter, name=" ")


class TestDatasetIdValidation:
    @pytest.mark.parametrize("bad", ["", "   ", ".", "..", "a/b", "a\\b", "c:name"])
    def test_rejects_unsafe_dataset_ids(self, bad):
        with pytest.raises(AdapterConfigurationError, match="dataset_id"):
            validate_dataset_id(bad)

    def test_trims_surrounding_whitespace(self):
        assert validate_dataset_id("  demo  ") == "demo"


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


class TestLabelNormalization:
    def test_resolves_declared_tokens(self):
        mapping = LabelMapping({"original": "real", "forged": "fake"})
        assert mapping.resolve("original") is SampleLabel.REAL
        assert mapping.resolve("forged") is SampleLabel.FAKE

    def test_token_shape_is_normalized_not_meaning(self):
        """Case and separators fold; that is formatting, not interpretation."""
        mapping = LabelMapping({"Face Swap": "fake"})
        assert mapping.resolve("face-swap") is SampleLabel.FAKE
        assert mapping.resolve("  FACE_SWAP ") is SampleLabel.FAKE

    def test_unknown_label_is_an_error(self):
        mapping = LabelMapping({"original": "real", "forged": "fake"})
        with pytest.raises(UnknownLabelError) as exc:
            mapping.resolve("spoofed")
        assert "spoofed" in str(exc.value)
        assert "original" in str(exc.value)

    def test_absent_label_is_an_error(self):
        with pytest.raises(UnknownLabelError, match="no label"):
            LabelMapping({"original": "real"}).resolve(None)

    def test_empty_mapping_says_so_explicitly(self):
        with pytest.raises(NoLabelMappingError, match="explicitly"):
            LabelMapping({}).resolve("real")

    def test_synonyms_are_not_inferred(self):
        """A vocabulary word that sounds meaningful is still undeclared."""
        mapping = LabelMapping({"original": "real"})
        for token in ("authentic", "synthetic", "manipulated", "forged"):
            with pytest.raises(UnknownLabelError):
                mapping.resolve(token)

    def test_rejects_a_target_that_is_not_a_label(self):
        with pytest.raises(NormalizationError, match="real"):
            LabelMapping({"weird": "maybe"})

    def test_rejects_two_spellings_of_one_token(self):
        with pytest.raises(NormalizationError, match="more than once"):
            LabelMapping({"face swap": "fake", "face-swap": "real"})

    def test_rejects_a_non_string_token(self):
        with pytest.raises(NormalizationError, match="token"):
            LabelMapping({7: "fake"})  # type: ignore[dict-item]

    def test_rejects_an_empty_token(self):
        with pytest.raises(NormalizationError, match="non-empty"):
            LabelMapping({"  ": "fake"})

    def test_rejects_a_non_mapping(self):
        with pytest.raises(NormalizationError, match="mapping"):
            LabelMapping(["real", "fake"])  # type: ignore[arg-type]

    def test_is_read_only_as_a_mapping(self):
        mapping = LabelMapping({"original": "real"})
        assert mapping["ORIGINAL"] == "real"
        assert "original" in mapping
        assert len(mapping) == 1


# ---------------------------------------------------------------------------
# Modality normalization
# ---------------------------------------------------------------------------


class TestModalityNormalization:
    def test_resolves_declared_tokens(self):
        mapping = ModalityMapping({"picture": "image", "sound": "audio"})
        assert mapping.resolve("picture") is Modality.IMAGE
        assert mapping.resolve("sound") is Modality.AUDIO

    def test_invalid_modality_token_is_an_error(self):
        with pytest.raises(UnknownModalityError, match="hologram"):
            ModalityMapping({"picture": "image"}).resolve("hologram")

    def test_rejects_a_modality_outside_the_canonical_set(self):
        with pytest.raises(NormalizationError, match="image"):
            ModalityMapping({"clip": "hologram"})

    def test_absent_modality_is_an_error(self):
        with pytest.raises(UnknownModalityError, match="never inferred"):
            ModalityMapping({}).resolve(None)

    def test_only_three_canonical_modalities_exist(self):
        assert set(Modality.values()) == {"image", "audio", "video"}


class TestModalityResolutionOrder:
    """Modality comes from authoritative structure first, and never from a filename."""

    def test_authoritative_modality_wins_over_a_token(self, source_dir: Path, adapter):
        raw = RawSample(
            sample_id="s",
            path=source_dir / "audio" / "real" / "aud_real_001.wav",
            label_token="original",
            modality=Modality.AUDIO,
            modality_token="conflicting-token",
        )
        sample = build_media_sample(raw, adapter=adapter, dataset_name="ds", source_root=source_dir)
        assert sample.modality is Modality.AUDIO

    def test_token_is_used_when_there_is_no_authoritative_modality(self, source_dir: Path):
        mapping_adapter = TemplateAdapter.from_config(
            {**synthetic_template_config(), "modalities": {"picture": "image"}},
            dataset_id=DATASET_ID,
        )
        raw = RawSample(
            sample_id="s",
            path=source_dir / "image" / "real" / "img_real_001.png",
            label_token="original",
            modality_token="picture",
        )
        sample = build_media_sample(
            raw,
            adapter=mapping_adapter,
            dataset_name="ds",
            source_root=source_dir,
            license_id=SYNTHETIC_LICENSE,
        )
        assert sample.modality is Modality.IMAGE
        assert mapping_adapter.modality_mapping().tokens == ("picture",)

    def test_unresolvable_modality_flags_the_sample_rather_than_guessing(self, source_dir: Path):
        """No modality anywhere means an error, not a guess from ``.wav``."""
        adapter = TemplateAdapter.from_config(synthetic_template_config(), dataset_id=DATASET_ID)
        raw = RawSample(
            sample_id="s",
            path=source_dir / "audio" / "real" / "aud_real_001.wav",
            label_token="original",
        )
        with pytest.raises(UnknownModalityError, match="extension"):
            build_media_sample(raw, adapter=adapter, dataset_name="ds", source_root=source_dir)


# ---------------------------------------------------------------------------
# Manipulation types
# ---------------------------------------------------------------------------


class TestManipulationTypes:
    def test_vocabulary_includes_the_documented_families(self):
        values = set(ManipulationType.values())
        assert {
            "face_swap",
            "face_reenactment",
            "lip_sync",
            "synthetic_media",
            "voice_cloning",
            "voice_conversion",
            "audio_manipulation",
            "unknown",
        } <= values

    def test_absent_manipulation_resolves_to_unknown(self):
        assert ManipulationMapping({}).resolve(None) is ManipulationType.UNKNOWN
        assert ManipulationMapping({}).resolve("  ") is ManipulationType.UNKNOWN

    def test_declared_token_resolves(self):
        mapping = ManipulationMapping({"faceswap": "face_swap"})
        assert mapping.resolve("FaceSwap") is ManipulationType.FACE_SWAP

    def test_present_but_unmapped_token_is_an_error(self):
        """Silence here would discard a specific claim the dataset made."""
        with pytest.raises(UnknownManipulationError, match="deepfake"):
            ManipulationMapping({}).resolve("deepfake")

    def test_error_lists_what_was_declared(self):
        mapping = ManipulationMapping({"faceswap": "face_swap"})
        with pytest.raises(UnknownManipulationError, match="faceswap"):
            mapping.resolve("deepfake")

    def test_rejects_a_family_outside_the_vocabulary(self):
        with pytest.raises(NormalizationError, match="ManipulationType"):
            ManipulationMapping({"weird": "telepathy"})

    def test_dataset_wording_is_preserved_when_it_resolves_to_unknown(
        self, source_dir: Path, adapter
    ):
        """A declared-but-unmapped word is recorded verbatim, not discarded."""
        raw = RawSample(
            sample_id="s",
            path=source_dir / "audio" / "real" / "aud_real_001.wav",
            label_token="original",
            modality=Modality.AUDIO,
            manipulation_token="telepathy",
        )
        other = TemplateAdapter.from_config(synthetic_template_config(), dataset_id=DATASET_ID)
        object.__setattr__(other, "_manipulations", ManipulationMapping({"telepathy": "unknown"}))
        sample = build_media_sample(raw, adapter=other, dataset_name="ds", source_root=source_dir)
        assert sample.manipulation_type is ManipulationType.UNKNOWN
        assert sample.metadata["source_manipulation_label"] == "telepathy"
        assert adapter.label_mapping().resolve("forged") is SampleLabel.FAKE


# ---------------------------------------------------------------------------
# Template adapter behaviour
# ---------------------------------------------------------------------------


class TestTemplateAdapter:
    def test_discovers_every_fixture_file(self, source_dir: Path, adapter):
        found = sorted(
            raw.path.relative_to(source_dir).as_posix() for raw in adapter.discover(source_dir)
        )
        assert found == sorted(SYNTHETIC_RELATIVE_FILES)

    def test_discovery_order_is_deterministic(self, source_dir: Path, adapter):
        first = [raw.sample_id for raw in adapter.discover(source_dir)]
        second = [raw.sample_id for raw in adapter.discover(source_dir)]
        assert first == second

    def test_sample_ids_come_from_the_relative_path(self, source_dir: Path, adapter):
        ids = {raw.sample_id for raw in adapter.discover(source_dir)}
        assert "image/fake/img_fake_001" in ids

    def test_modality_comes_from_the_directory_not_the_extension(self, source_dir: Path, adapter):
        by_id = {raw.sample_id: raw for raw in adapter.discover(source_dir)}
        wav = by_id["audio/real/aud_real_001"]
        assert wav.modality is Modality.AUDIO
        assert wav.path.suffix == ".wav"

    def test_unmatched_files_are_yielded_unlabelled(self, source_dir: Path, adapter):
        stray = source_dir / "unlisted" / "mystery.bin"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"mystery")
        by_id = {raw.sample_id: raw for raw in adapter.discover(source_dir)}
        assert by_id["unlisted/mystery"].label_token is None
        assert by_id["unlisted/mystery"].modality is None

    def test_default_rule_attributes_unmatched_files(self, source_dir: Path):
        adapter = TemplateAdapter.from_config(
            {
                **synthetic_template_config(),
                "default_rule": {"label": "original", "modality": "image"},
            },
            dataset_id=DATASET_ID,
        )
        stray = source_dir / "unlisted" / "mystery.bin"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"mystery")
        stray_sample = next(
            raw for raw in adapter.discover(source_dir) if raw.sample_id == "unlisted/mystery"
        )
        assert stray_sample.label_token == "original"
        assert stray_sample.modality is Modality.IMAGE

    def test_extension_filter_restricts_candidates(self, source_dir: Path):
        adapter = TemplateAdapter.from_config(
            {**synthetic_template_config(), "extensions": [".png"]}, dataset_id=DATASET_ID
        )
        assert len(list(adapter.discover(source_dir))) == 2

    def test_extension_filter_normalizes_separators_and_case(self, source_dir: Path):
        adapter = TemplateAdapter.from_config(
            {**synthetic_template_config(), "extensions": ["PNG"]}, dataset_id=DATASET_ID
        )
        assert len(list(adapter.discover(source_dir))) == 2

    def test_hidden_files_are_skipped(self, source_dir: Path, adapter):
        (source_dir / ".DS_Store").write_bytes(b"junk")
        assert ".DS_Store" not in {raw.sample_id for raw in adapter.discover(source_dir)}

    def test_glob_rules_match(self, source_dir: Path):
        adapter = TemplateAdapter.from_config(
            {
                "license": "CC0-1.0",
                "labels": {"original": "real", "forged": "fake"},
                "directory_rules": [{"path": "audio/*", "label": "forged", "modality": "audio"}],
                "default_rule": {"label": "original", "modality": "image"},
            },
            dataset_id=DATASET_ID,
        )
        by_id = {raw.sample_id: raw for raw in adapter.discover(source_dir)}
        assert by_id["audio/fake/aud_fake_001"].label_token == "forged"
        assert by_id["video/real/vid_real_001"].modality is Modality.IMAGE

    def test_config_without_a_license_is_rejected(self):
        with pytest.raises(AdapterConfigurationError, match="license"):
            TemplateAdapter.from_config({"labels": {"real": "real"}}, dataset_id=DATASET_ID)

    def test_config_rejects_unknown_rule_keys(self):
        with pytest.raises(AdapterConfigurationError, match="unknown key"):
            TemplateAdapter.from_config(
                {
                    "license": "CC0-1.0",
                    "labels": {"real": "real"},
                    "directory_rules": [{"path": "a", "label": "real", "typo": 1}],
                },
                dataset_id=DATASET_ID,
            )

    def test_config_rejects_a_bad_modality(self):
        with pytest.raises(AdapterConfigurationError, match="modality"):
            TemplateAdapter.from_config(
                {
                    "license": "CC0-1.0",
                    "labels": {"real": "real"},
                    "directory_rules": [{"path": "a", "label": "real", "modality": "hologram"}],
                },
                dataset_id=DATASET_ID,
            )

    def test_config_rejects_a_non_mapping_vocabulary(self):
        with pytest.raises(AdapterConfigurationError, match="labels"):
            TemplateAdapter.from_config(
                {"license": "CC0-1.0", "labels": ["real", "fake"]}, dataset_id=DATASET_ID
            )

    def test_config_file_round_trip(self, tmp_path: Path, source_dir: Path):
        config_path = tmp_path / "adapter.json"
        config_path.write_text(json.dumps(synthetic_template_config()), encoding="utf-8")
        loaded = TemplateAdapter.from_config_file(config_path, dataset_id=DATASET_ID)
        assert loaded.dataset_name == "synthetic_unit_test_fixture"
        assert loaded.license_id == SYNTHETIC_LICENSE
        assert len(list(loaded.discover(source_dir))) == 6

    def test_missing_config_file_is_reported(self, tmp_path: Path):
        with pytest.raises(AdapterConfigurationError, match="adapter config not found"):
            TemplateAdapter.from_config_file(tmp_path / "absent.json", dataset_id=DATASET_ID)

    def test_malformed_config_file_is_reported(self, tmp_path: Path):
        config_path = tmp_path / "adapter.json"
        config_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(AdapterConfigurationError, match="malformed JSON"):
            TemplateAdapter.from_config_file(config_path, dataset_id=DATASET_ID)

    def test_config_file_must_hold_an_object(self, tmp_path: Path):
        config_path = tmp_path / "adapter.json"
        config_path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(AdapterConfigurationError, match="JSON object"):
            TemplateAdapter.from_config_file(config_path, dataset_id=DATASET_ID)


# ---------------------------------------------------------------------------
# Successful ingestion
# ---------------------------------------------------------------------------


class TestSuccessfulIngestion:
    def test_writes_a_manifest_and_provenance(
        self, request_factory, source_dir, adapter, output_dir
    ):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert summary.is_successful
        assert summary.total_samples == 6
        assert output_dir.joinpath(f"{DATASET_ID}.jsonl").is_file()
        assert output_dir.joinpath(f"{DATASET_ID}.provenance.json").is_file()

    def test_manifest_holds_one_record_per_sample(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert len(load_manifest(summary.manifest_path)) == 6

    def test_source_media_is_not_modified(self, request_factory, source_dir, adapter):
        before = {
            path.relative_to(source_dir).as_posix(): calculate_file_sha256(path)
            for path in sorted(source_dir.rglob("*"))
            if path.is_file()
        }
        ingest_dataset(request_factory(adapter, source_dir))
        after = {
            path.relative_to(source_dir).as_posix(): calculate_file_sha256(path)
            for path in sorted(source_dir.rglob("*"))
            if path.is_file()
        }
        assert before == after

    def test_recorded_paths_are_relative_and_posix(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        for sample in load_manifest(summary.manifest_path):
            assert not Path(sample.filepath).is_absolute()
            assert "\\" not in sample.filepath

    def test_hashes_match_the_files_on_disk(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        for sample in load_manifest(summary.manifest_path):
            expected = hashlib.sha256((source_dir / sample.filepath).read_bytes()).hexdigest()
            assert sample.sha256 == expected

    def test_dataset_name_and_license_are_recorded(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        samples = load_manifest(summary.manifest_path)
        assert {s.dataset_name for s in samples} == {"synthetic_unit_test_fixture"}
        assert {s.license for s in samples} == {SYNTHETIC_LICENSE}

    def test_split_is_left_unassigned(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert all(sample.split is None for sample in load_manifest(summary.manifest_path))

    def test_dataset_wording_is_preserved_alongside_the_canonical_label(
        self, request_factory, source_dir, adapter
    ):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        by_id = {s.sample_id: s for s in load_manifest(summary.manifest_path)}
        assert by_id["image/fake/img_fake_001"].metadata["source_label"] == "forged"
        assert by_id["image/real/img_real_001"].metadata["source_label"] == "original"

    def test_refuses_to_overwrite_without_permission(self, request_factory, source_dir, adapter):
        ingest_dataset(request_factory(adapter, source_dir))
        with pytest.raises(FileExistsError, match="overwrite"):
            ingest_dataset(request_factory(adapter, source_dir))

    def test_overwrites_when_permitted(self, request_factory, source_dir, adapter):
        ingest_dataset(request_factory(adapter, source_dir))
        summary = ingest_dataset(request_factory(adapter, source_dir, overwrite=True))
        assert summary.is_successful


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestIngestionSummary:
    @pytest.fixture
    def summary(self, request_factory, source_dir, adapter):
        return ingest_dataset(request_factory(adapter, source_dir))

    def test_counts_labels(self, summary):
        assert summary.real_samples == 3
        assert summary.fake_samples == 3
        assert summary.total_samples == 6

    def test_counts_modalities(self, summary):
        assert summary.modality_counts == {"audio": 2, "image": 2, "video": 2}

    def test_counts_manipulations_including_unknown(self, summary):
        """The three ``real`` directories declare no manipulation, so they are ``unknown``."""
        assert summary.manipulation_counts == {"face_swap": 3, "unknown": 3}

    def test_counts_missing_metadata(self, summary):
        """No sample states an origin, and the three ``real`` ones state no manipulation."""
        assert summary.missing_metadata_counts == {"source": 6, "manipulation_type": 3}

    def test_reports_no_metadata_gaps_when_the_adapter_declares_everything(self, summary):
        assert summary.missing_metadata_keys == {}

    def test_reports_metadata_keys_present_on_only_some_samples(self, request_factory, source_dir):
        """A key some samples carry and others do not is a dataset-wide gap, not a detail."""

        class PartlyAnnotatedAdapter(TemplateAdapter):
            def discover(self, source_dir: Path):
                for raw in super().discover(source_dir):
                    if "real" in raw.path.parts:
                        raw.metadata["capture_note"] = "synthetic"
                    yield raw

        summary = ingest_dataset(
            request_factory(
                PartlyAnnotatedAdapter.from_config(
                    synthetic_template_config(), dataset_id=DATASET_ID
                ),
                source_dir,
            )
        )
        assert summary.missing_metadata_keys == {"capture_note": 3}

    def test_hash_verification_is_reported(self, summary):
        assert summary.hash_verification is HashVerificationStatus.VERIFIED

    def test_manifest_hash_is_recorded(self, summary):
        assert summary.manifest_hash.startswith("sha256:")
        assert len(summary.manifest_hash) == len("sha256:") + 64

    def test_records_versioning_fields(self, summary):
        assert summary.dataset_id == DATASET_ID
        assert summary.dataset_version == "fixture-1"
        assert summary.ingestion_script_version
        assert summary.ingestion_timestamp

    def test_is_json_serializable(self, summary):
        assert json.loads(summary.to_json())["total_samples"] == 6

    def test_render_reports_the_essentials(self, summary):
        """``render`` returns the text the CLI prints; printing is covered by TestCli."""
        printed = summary.render()
        assert DATASET_ID in printed
        assert summary.manifest_hash in printed

    def test_hash_verification_can_be_disabled(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir, verify_hashes=False))
        assert summary.hash_verification is HashVerificationStatus.NOT_REQUESTED


class TestHashVerificationFailure:
    def test_reports_failure_when_a_file_changes(
        self, request_factory, source_dir, adapter, output_dir, monkeypatch
    ):
        """Verification must notice bytes that changed between hashing and writing."""
        import datasets.ingest as ingest_module

        original = ingest_module.write_manifest

        def tampering_write(path, samples, **kwargs):
            result = original(path, samples, **kwargs)
            (source_dir / "image" / "real" / "img_real_001.png").write_bytes(b"tampered")
            return result

        monkeypatch.setattr(ingest_module, "write_manifest", tampering_write)
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert summary.hash_verification is HashVerificationStatus.FAILED
        assert output_dir.is_dir()


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


class TestIngestionValidation:
    def test_unknown_label_token_is_reported_not_guessed(self, request_factory, source_dir):
        config = synthetic_template_config()
        config["labels"] = {"original": "real"}  # 'forged' deliberately unmapped
        adapter = TemplateAdapter.from_config(config, dataset_id=DATASET_ID)
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert not summary.is_successful
        assert len(summary.errors) == 3
        assert any("unknown label 'forged'" in error for error in summary.errors)
        assert summary.total_samples == 3

    def test_unmatched_files_are_flagged_not_dropped(self, request_factory, source_dir):
        stray = source_dir / "unlisted" / "mystery.bin"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"mystery")
        summary = ingest_dataset(
            request_factory(
                TemplateAdapter.from_config(synthetic_template_config(), dataset_id=DATASET_ID),
                source_dir,
            )
        )
        assert not summary.is_successful
        assert any("unlisted/mystery" in error for error in summary.errors)

    def test_no_manifest_is_written_when_a_sample_fails(
        self, request_factory, source_dir, output_dir
    ):
        config = synthetic_template_config()
        config["labels"] = {"original": "real"}
        adapter = TemplateAdapter.from_config(config, dataset_id=DATASET_ID)
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert summary.errors
        assert summary.manifest_path is None
        assert not output_dir.joinpath(f"{DATASET_ID}.jsonl").exists()

    def test_duplicate_sample_ids_are_reported(self, request_factory, source_dir):
        class DuplicateIds(TemplateAdapter):
            name = "duplicate-ids"

            def discover(self, source_dir: Path):
                yield from super().discover(source_dir)
                yield next(iter(super().discover(source_dir)))

        adapter = DuplicateIds.from_config(synthetic_template_config(), dataset_id=DATASET_ID)
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert not summary.is_successful
        assert any("duplicate sample_id" in error for error in summary.errors)

    def test_missing_source_directory_raises(self, request_factory, adapter, tmp_path):
        with pytest.raises(IngestionError, match="source directory not found"):
            ingest_dataset(request_factory(adapter, tmp_path / "absent"))

    def test_unsafe_dataset_id_raises(self, request_factory, source_dir, adapter):
        with pytest.raises(AdapterConfigurationError, match="dataset_id"):
            ingest_dataset(request_factory(adapter, source_dir, dataset_id="../escape"))

    def test_adapter_without_a_label_mapping_raises(self, request_factory, source_dir):
        adapter = TemplateAdapter(
            dataset_name=DATASET_ID,
            license_id=SYNTHETIC_LICENSE,
            labels={},
        )
        with pytest.raises(AdapterConfigurationError, match="no label mapping"):
            ingest_dataset(request_factory(adapter, source_dir))

    def test_missing_media_file_is_reported_per_sample(self, request_factory, source_dir):
        class VanishingAdapter(TemplateAdapter):
            name = "vanishing"

            def discover(self, source_dir: Path):
                for raw in super().discover(source_dir):
                    if raw.sample_id == "image/real/img_real_001":
                        raw.path.unlink(missing_ok=True)
                    yield raw

        adapter = VanishingAdapter.from_config(synthetic_template_config(), dataset_id=DATASET_ID)
        summary = ingest_dataset(request_factory(adapter, source_dir))
        assert any("not found" in error for error in summary.errors)

    def test_path_outside_the_source_root_is_refused(self, source_dir, adapter, tmp_path):
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"elsewhere")
        raw = RawSample(
            sample_id="outside",
            path=outside,
            label_token="original",
            modality=Modality.AUDIO,
        )
        with pytest.raises(Exception, match="outside the dataset root"):
            build_media_sample(raw, adapter=adapter, dataset_name="ds", source_root=source_dir)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    @pytest.fixture
    def dry_summary(self, request_factory, source_dir, adapter):
        return ingest_dataset(request_factory(adapter, source_dir, dry_run=True))

    def test_reports_the_same_counts_as_a_real_run(
        self, request_factory, source_dir, adapter, dry_summary
    ):
        real = ingest_dataset(request_factory(adapter, source_dir))
        assert dry_summary.total_samples == real.total_samples
        assert dry_summary.modality_counts == real.modality_counts
        assert dry_summary.manipulation_counts == real.manipulation_counts

    def test_manifest_hash_matches_a_real_run(
        self, request_factory, source_dir, adapter, dry_summary
    ):
        real = ingest_dataset(request_factory(adapter, source_dir))
        assert dry_summary.manifest_hash == real.manifest_hash

    def test_writes_nothing(self, request_factory, source_dir, adapter, dry_summary, output_dir):
        assert not output_dir.exists()

    def test_reports_no_manifest_path(self, dry_summary):
        assert dry_summary.manifest_path is None
        assert dry_summary.provenance_path is None
        assert dry_summary.manifest_written is False

    def test_is_marked_as_a_dry_run(self, dry_summary):
        assert dry_summary.dry_run is True

    def test_does_not_modify_source_media(self, request_factory, source_dir, adapter, tmp_path):
        before = calculate_file_sha256(source_dir / "audio" / "fake" / "aud_fake_001.wav")
        ingest_dataset(request_factory(adapter, source_dir, dry_run=True))
        assert calculate_file_sha256(source_dir / "audio" / "fake" / "aud_fake_001.wav") == before

    def test_still_reports_errors(self, request_factory, source_dir):
        config = synthetic_template_config()
        config["labels"] = {"original": "real"}
        adapter = TemplateAdapter.from_config(config, dataset_id=DATASET_ID)
        summary = ingest_dataset(request_factory(adapter, source_dir, dry_run=True))
        assert not summary.is_successful
        assert summary.errors

    def test_verify_hashes_can_run_in_a_dry_run(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(
            request_factory(adapter, source_dir, dry_run=True, verify_hashes=True)
        )
        assert summary.hash_verification is HashVerificationStatus.VERIFIED


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_manifest_bytes_are_identical_across_runs(
        self, request_factory, source_dir, adapter, tmp_path
    ):
        first_dir = tmp_path / "out-a"
        second_dir = tmp_path / "out-b"
        first = ingest_dataset(request_factory(adapter, source_dir, output_dir=first_dir))
        second = ingest_dataset(request_factory(adapter, source_dir, output_dir=second_dir))
        assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
        assert first.manifest_hash == second.manifest_hash

    def test_manifest_does_not_contain_a_timestamp(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        text = summary.manifest_path.read_text(encoding="utf-8")
        assert summary.ingestion_timestamp not in text
        assert "ingestion" not in text

    def test_provenance_records_the_timestamp(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        record = load_provenance(summary.provenance_path)
        assert record.ingestion_timestamp == summary.ingestion_timestamp

    def test_provenance_content_hash_ignores_the_timestamp(
        self, request_factory, source_dir, adapter, tmp_path
    ):
        first = load_provenance(
            ingest_dataset(
                request_factory(adapter, source_dir, output_dir=tmp_path / "out-a")
            ).provenance_path
        )
        second = load_provenance(
            ingest_dataset(
                request_factory(adapter, source_dir, output_dir=tmp_path / "out-b")
            ).provenance_path
        )
        assert first.content_hash() == second.content_hash()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_records_the_required_fields(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        record = load_provenance(summary.provenance_path)
        assert record.dataset_id == DATASET_ID
        assert record.dataset_name == "synthetic_unit_test_fixture"
        assert record.version == "fixture-1"
        assert record.license == SYNTHETIC_LICENSE
        assert record.access_date is None
        assert record.source_reference is None
        assert record.ingestion_script_version
        assert record.manifest_hash == summary.manifest_hash

    def test_unknown_fields_are_listed_not_filled(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        record = load_provenance(summary.provenance_path)
        assert set(record.unknown_fields) == {"source", "source_reference", "access_date"}

    def test_records_a_supplied_source_reference(self, source_dir):
        adapter = TemplateAdapter.from_config(
            {
                **synthetic_template_config(),
                "source_reference": "local-fixture-directory",
                "access_date": "2026-01-01",
            },
            dataset_id=DATASET_ID,
        )
        summary = ingest_dataset(
            IngestionRequest(
                dataset_id=DATASET_ID,
                adapter=adapter,
                source_dir=source_dir,
                output_dir=source_dir.parent / "manifests",
            )
        )
        record = load_provenance(summary.provenance_path)
        assert record.source_reference == "local-fixture-directory"
        assert record.access_date == "2026-01-01"
        assert record.unknown_fields == ("source",)

    def test_path_pairs_with_the_manifest(self):
        assert provenance_path_for("m/demo.jsonl").name == "demo.provenance.json"

    def test_requires_a_sha256_prefixed_hash(self):
        with pytest.raises(ProvenanceError, match="sha256"):
            DatasetProvenance(
                dataset_id="d",
                dataset_name="d",
                version="1",
                license="L",
                ingestion_script_version="1",
                manifest_hash="deadbeef",
                ingestion_timestamp="2026-01-01T00:00:00+00:00",
            )

    def test_requires_a_hex_digest(self):
        with pytest.raises(ProvenanceError, match="hex"):
            DatasetProvenance(
                dataset_id="d",
                dataset_name="d",
                version="1",
                license="L",
                ingestion_script_version="1",
                manifest_hash="sha256:zzzz",
                ingestion_timestamp="2026-01-01T00:00:00+00:00",
            )

    def test_rejects_a_non_object_payload(self):
        with pytest.raises(ProvenanceError, match="JSON object"):
            DatasetProvenance.from_dict(["not", "a", "record"])

    def test_reports_missing_required_fields(self):
        with pytest.raises(ProvenanceError, match="missing required field"):
            DatasetProvenance.from_dict({"dataset_id": "d"})

    def test_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ProvenanceError, match="not found"):
            load_provenance(tmp_path / "absent.json")

    def test_malformed_file_is_reported(self, tmp_path):
        bad = tmp_path / "p.json"
        bad.write_text("{", encoding="utf-8")
        with pytest.raises(ProvenanceError, match="malformed JSON"):
            load_provenance(bad)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    @pytest.fixture
    def config_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "adapter.json"
        path.write_text(json.dumps(synthetic_template_config()), encoding="utf-8")
        return path

    def test_help_is_available(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        printed = capsys.readouterr().out
        for option in (
            "--dataset-id",
            "--adapter",
            "--source",
            "--output",
            "--dry-run",
            "--verify-hashes",
        ):
            assert option in printed

    def test_lists_registered_adapters(self, capsys):
        assert main(["--dataset-id", DATASET_ID, "--list-adapters"]) == 0
        assert "template" in capsys.readouterr().out

    def test_dry_run_succeeds_without_writing(self, source_dir, config_path, output_dir, capsys):
        code = main(
            [
                "--dataset-id",
                DATASET_ID,
                "--adapter",
                "template",
                "--adapter-config",
                str(config_path),
                "--source",
                str(source_dir),
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        assert code == 0
        printed = capsys.readouterr().out
        assert "dry-run" in printed
        assert not output_dir.exists()

    def test_ingest_writes_and_prints_the_summary(
        self, source_dir, config_path, output_dir, capsys
    ):
        code = main(
            [
                "--dataset-id",
                DATASET_ID,
                "--adapter-config",
                str(config_path),
                "--source",
                str(source_dir),
                "--output",
                str(output_dir),
            ]
        )
        assert code == 0
        assert output_dir.joinpath(f"{DATASET_ID}.jsonl").is_file()
        assert DATASET_ID in capsys.readouterr().out

    def test_json_output_is_machine_readable(self, source_dir, config_path, output_dir, capsys):
        main(
            [
                "--dataset-id",
                DATASET_ID,
                "--adapter-config",
                str(config_path),
                "--source",
                str(source_dir),
                "--output",
                str(output_dir),
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["total_samples"] == 6
        assert payload["is_successful"] is True

    def test_sample_errors_exit_one(self, source_dir, config_path, output_dir, capsys):
        config = synthetic_template_config()
        config["labels"] = {"original": "real"}
        config_path.write_text(json.dumps(config), encoding="utf-8")
        code = main(
            [
                "--dataset-id",
                DATASET_ID,
                "--adapter-config",
                str(config_path),
                "--source",
                str(source_dir),
                "--output",
                str(output_dir),
                "--dry-run",
            ]
        )
        assert code == 1
        assert "errors" in capsys.readouterr().out

    def test_missing_source_exits_two(self, config_path, tmp_path, capsys):
        code = main(
            [
                "--dataset-id",
                DATASET_ID,
                "--adapter-config",
                str(config_path),
                "--source",
                str(tmp_path / "absent"),
            ]
        )
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_unknown_adapter_exits_two(self, source_dir, capsys):
        code = main(["--dataset-id", DATASET_ID, "--adapter", "nope", "--source", str(source_dir)])
        assert code == 2
        assert "unknown adapter" in capsys.readouterr().err

    def test_adapter_without_a_config_exits_two(self, source_dir, capsys):
        code = main(["--dataset-id", DATASET_ID, "--source", str(source_dir)])
        assert code == 2
        assert "--adapter-config" in capsys.readouterr().err

    def test_existing_manifest_exits_two(self, source_dir, config_path, output_dir, capsys):
        args = [
            "--dataset-id",
            DATASET_ID,
            "--adapter-config",
            str(config_path),
            "--source",
            str(source_dir),
            "--output",
            str(output_dir),
        ]
        assert main(args) == 0
        capsys.readouterr()
        assert main(args) == 2
        assert "overwrite" in capsys.readouterr().err

    def test_overwrite_flag_allows_a_rerun(self, source_dir, config_path, output_dir, capsys):
        args = [
            "--dataset-id",
            DATASET_ID,
            "--adapter-config",
            str(config_path),
            "--source",
            str(source_dir),
            "--output",
            str(output_dir),
        ]
        assert main(args) == 0
        assert main([*args, "--overwrite"]) == 0

    def test_source_defaults_to_the_raw_directory_convention(
        self, config_path, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.chdir(tmp_path)
        build_synthetic_dataset(tmp_path / "datasets" / "raw" / DATASET_ID)
        assert (
            main(["--dataset-id", DATASET_ID, "--adapter-config", str(config_path), "--dry-run"])
            == 0
        )
        assert "samples            : 6" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


class TestNormalizeToken:
    def test_folds_case_whitespace_and_separators(self):
        assert normalize_token("  Face-Swap Name ") == "face_swap_name"

    def test_collapses_repeated_separators(self):
        assert normalize_token("face__swap") == "face_swap"

    def test_rejects_non_strings(self):
        with pytest.raises(NormalizationError, match="token"):
            normalize_token(None)

    def test_rejects_empty(self):
        with pytest.raises(NormalizationError, match="non-empty"):
            normalize_token(" - ")


class TestSampleContract:
    def test_manifest_records_use_canonical_vocabulary(self, request_factory, source_dir, adapter):
        summary = ingest_dataset(request_factory(adapter, source_dir))
        samples = load_manifest(summary.manifest_path)
        assert {s.modality for s in samples} <= set(Modality)
        assert {s.label for s in samples} <= set(SampleLabel)
        assert all(isinstance(s, MediaSample) for s in samples)
