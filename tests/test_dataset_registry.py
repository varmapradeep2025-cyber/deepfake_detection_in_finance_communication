"""Tests for the dataset registry.

Two kinds of test live here, and the distinction matters:

* **Fixture tests** write small registry files into pytest's ``tmp_path`` to exercise
  each validation rule in isolation — one rule broken, one error expected. They are the
  reason a rule that was deleted would fail loudly rather than quietly stop protecting
  anything.
* **Contract tests** read the real ``configs/datasets.yaml`` and assert the properties
  this project claims about itself: nothing is downloaded, nothing is approved, no
  licence or reference is invented, no intelligence source is a training corpus, and no
  generalization result is claimed.

No test performs network access, downloads anything, or writes outside ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from datasets.registry import (
    GENERALIZATION_GROUPS,
    REGISTRY_SCHEMA_VERSION,
    UNVERIFIED_MARKERS,
    AccessType,
    DatasetEntry,
    GeneralizationPair,
    GeneralizationStatus,
    LabelScheme,
    LicenseStatus,
    RegistryError,
    RegistryModality,
    RegistryNotFoundError,
    RegistryReport,
    RegistryStatus,
    RegistryValidationError,
    ResearchRole,
    SourceType,
    default_registry_path,
    is_unverified,
    load_registry,
    main,
    validate_registry,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "datasets.yaml"


# ---------------------------------------------------------------------------
# Fixtures: small, deliberately invalid-on-purpose registries
# ---------------------------------------------------------------------------


def _source_entry(dataset_id: str = "source_corpus", **overrides: Any) -> dict[str, Any]:
    """A minimal valid source-domain entry, overridable field by field."""
    entry: dict[str, Any] = {
        "dataset_id": dataset_id,
        "name": "Source Corpus",
        "modality": "video",
        "source_type": "research_dataset",
        "research_role": ["source_domain", "training"],
        "status": "candidate",
        "access_type": "unknown",
        "license_status": "pending_verification",
        "source_reference": "not_verified",
        "version": "unknown",
        "train_allowed": True,
        "validation_allowed": True,
        "test_allowed": True,
        "unseen_test_allowed": False,
        "generalization_groups": ["video_deepfake"],
        "label_schema": {"scheme": "binary_real_fake", "values": ["real", "fake"]},
        "manipulation_types": "pending_verification",
        "requires_request": "unknown",
        "requires_terms_acceptance": "unknown",
        "redistribution_allowed": "pending_verification",
        "acquisition": {"acquired": False, "access_date": None, "version": None},
        "notes": "Fixture entry.",
    }
    entry.update(overrides)
    return entry


def _target_entry(dataset_id: str = "target_corpus", **overrides: Any) -> dict[str, Any]:
    """A minimal valid held-out target-domain entry."""
    entry = _source_entry(
        dataset_id,
        name="Target Corpus",
        research_role=["target_domain", "unseen_test"],
        train_allowed=False,
        validation_allowed=False,
        test_allowed=False,
        unseen_test_allowed=True,
        generalization_groups=["video_deepfake"],
        notes="Fixture target entry.",
    )
    entry.update(overrides)
    return entry


def _intelligence_entry(dataset_id: str = "some_feed", **overrides: Any) -> dict[str, Any]:
    """A minimal valid public-intelligence entry."""
    entry = _source_entry(
        dataset_id,
        name="Some Feed",
        modality="threat_intelligence",
        source_type="intelligence_feed",
        research_role=["threat_intelligence", "external_intelligence"],
        train_allowed=False,
        validation_allowed=False,
        test_allowed=False,
        unseen_test_allowed=False,
        generalization_groups=[],
        label_schema={"scheme": "evidence_state", "values": ["observed"]},
        notes="Fixture intelligence entry.",
    )
    entry.update(overrides)
    return entry


def _pair(**overrides: Any) -> dict[str, Any]:
    """A minimal valid generalization pair."""
    pair: dict[str, Any] = {
        "group_id": "video_deepfake",
        "source": "source_corpus",
        "target": "target_corpus",
        "status": "planned",
        "executed": False,
        "notes": "Fixture pair.",
    }
    pair.update(overrides)
    return pair


def write_registry(
    tmp_path: Path,
    datasets: dict[str, Any] | None = None,
    generalization: list[dict[str, Any]] | None = None,
    *,
    registry_version: Any = REGISTRY_SCHEMA_VERSION,
    filename: str = "datasets.yaml",
) -> Path:
    """Write a registry document to ``tmp_path`` and return its path."""
    document: dict[str, Any] = {
        "registry_version": registry_version,
        "datasets": datasets
        if datasets is not None
        else {"source_corpus": _source_entry(), "target_corpus": _target_entry()},
        "generalization": generalization if generalization is not None else [_pair()],
    }
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def write_raw_registry(tmp_path: Path, text: str, *, filename: str = "datasets.yaml") -> Path:
    """Write verbatim YAML, for malformed and duplicate-key cases."""
    path = tmp_path / filename
    path.write_text(text, encoding="utf-8")
    return path


def errors_of(path: Path) -> list[str]:
    return validate_registry(path).errors


def assert_mentions(messages: list[str], *fragments: str) -> None:
    """Assert some message mentions each fragment, with the failing list in the report."""
    joined = "\n".join(messages)
    missing = [fragment for fragment in fragments if fragment not in joined]
    assert not missing, f"expected a message mentioning {missing} in:\n{joined}"


# ---------------------------------------------------------------------------
# The real registry is the contract
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> Any:
    return load_registry(REGISTRY_PATH)


def test_the_repository_registry_is_valid() -> None:
    report = validate_registry(REGISTRY_PATH)
    assert report.is_valid, "\n".join(report.errors)


def test_the_repository_registry_is_the_configured_file() -> None:
    assert default_registry_path() == REGISTRY_PATH.resolve()


def test_registry_declares_every_planned_source(registry: Any) -> None:
    assert set(registry.ids()) == {
        "faceforensics_plus_plus",
        "celeb_df_v2",
        "asvspoof_2021_df",
        "fakeavceleb",
        "cic_trap4phish",
        "phishtank",
        "urlhaus",
        "tranco",
    }


def test_nothing_is_approved_or_acquired(registry: Any) -> None:
    """A candidate registry must not contain an approved or locally available entry."""
    for entry in registry:
        assert entry.status is not RegistryStatus.APPROVED, entry.dataset_id
        assert entry.status is not RegistryStatus.AVAILABLE_LOCALLY, entry.dataset_id
        assert not entry.is_acquired, entry.dataset_id


def test_no_licence_reference_or_version_is_invented(registry: Any) -> None:
    """Unverified information must remain an explicit marker, not a plausible guess."""
    for entry in registry:
        assert entry.license_status is not LicenseStatus.VERIFIED, entry.dataset_id
        assert is_unverified(entry.source_reference), entry.dataset_id
        assert is_unverified(entry.version), entry.dataset_id
        assert is_unverified(str(entry.approximate_size)), entry.dataset_id


def test_every_entry_reports_what_is_still_unknown(registry: Any) -> None:
    for entry in registry:
        assert entry.unverified_fields(), entry.dataset_id


def test_intelligence_sources_are_never_training_corpora(registry: Any) -> None:
    feeds = [entry for entry in registry if entry.is_intelligence_source]
    assert {entry.dataset_id for entry in feeds} == {"phishtank", "urlhaus", "tranco"}
    for entry in feeds:
        assert not entry.has_training_intent, entry.dataset_id
        assert entry.label_schema is not None
        assert entry.label_schema.scheme.value in {"evidence_state", "unlabeled"}, entry.dataset_id


def test_tranco_is_declared_unlabeled(registry: Any) -> None:
    """A popularity ranking carries no labels, and must not pretend otherwise."""
    tranco = registry.get("tranco")
    assert tranco is not None
    assert tranco.label_schema is not None
    assert tranco.label_schema.scheme.value == "unlabeled"
    assert tranco.label_schema.values == ()


def test_target_domains_are_never_trained_on(registry: Any) -> None:
    for pair in registry.generalization:
        target = registry.get(pair.target)
        assert target is not None
        assert not target.train_allowed, pair.group_id
        assert target.unseen_test_allowed, pair.group_id
        assert target.has_role(ResearchRole.TARGET_DOMAIN), pair.group_id


def test_source_domains_are_never_the_unseen_test(registry: Any) -> None:
    for pair in registry.generalization:
        source = registry.get(pair.source)
        assert source is not None
        assert not source.unseen_test_allowed, pair.group_id
        assert source.train_allowed, pair.group_id


def test_no_generalization_claim_has_been_executed(registry: Any) -> None:
    """`executed: true` without data behind it is the easiest claim here to fake."""
    for pair in registry.generalization:
        assert pair.executed is False, pair.group_id
        assert pair.status is GeneralizationStatus.PLANNED, pair.group_id


def test_every_planned_group_has_an_explicit_pair(registry: Any) -> None:
    """A group an entry claims must name a source and a target."""
    for entry in registry:
        for group in entry.generalization_groups:
            assert group in GENERALIZATION_GROUPS
            matching = [
                pair for pair in registry.pairs_for(entry.dataset_id) if pair.group_id == group
            ]
            assert matching, f"{entry.dataset_id} claims {group} with no pair"


def test_the_image_and_video_experiments_are_separate_groups(registry: Any) -> None:
    """Same dataset pair, different experiment: the metrics are not interchangeable."""
    groups = {pair.group_id for pair in registry.generalization}
    assert {"image_deepfake", "video_deepfake", "audio_deepfake"} <= groups


def test_multimodal_entry_declares_its_components(registry: Any) -> None:
    entry = registry.get("fakeavceleb")
    assert entry is not None
    assert RegistryModality.MULTIMODAL in entry.modality
    assert {value.value for value in entry.derived_modalities} == {"audio", "video"}


def test_qr_benchmark_declares_both_artefacts(registry: Any) -> None:
    entry = registry.get("cic_trap4phish")
    assert entry is not None
    assert {value.value for value in entry.modality} == {"qr", "url"}
    assert entry.status is RegistryStatus.PENDING_VERIFICATION
    assert not any(entry.usage_flags.values())


def test_loading_is_deterministic(registry: Any) -> None:
    """Two loads of an unchanged file serialize identically, so review is meaningful."""
    first = load_registry(REGISTRY_PATH).to_dict()
    second = load_registry(REGISTRY_PATH).to_dict()
    assert first == second
    assert list(first["datasets"]) == sorted(first["datasets"])


def test_the_registry_describes_nothing_on_disk(registry: Any) -> None:
    """A registry that says a local path exists is asserting something about the machine."""
    for entry in registry:
        assert entry.acquisition.local_path is None, entry.dataset_id
        assert entry.acquisition.access_date is None, entry.dataset_id


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vocabulary", "accepted"),
    [
        (
            RegistryModality,
            ["image", "audio", "video", "qr", "url", "multimodal", "threat_intelligence"],
        ),
        (ResearchRole, ["training", "unseen_test", "target_domain", "source_domain"]),
        (
            RegistryStatus,
            [
                "candidate",
                "approved",
                "pending_verification",
                "available_locally",
                "blocked",
                "deprecated",
            ],
        ),
        (AccessType, ["unknown", "public_download"]),
        (LicenseStatus, ["pending_verification", "verified"]),
        (GeneralizationStatus, ["planned", "blocked", "executed"]),
    ],
)
def test_vocabularies_are_closed(vocabulary: Any, accepted: list[str]) -> None:
    assert set(accepted) <= set(vocabulary.values())


def test_unknown_vocabulary_value_is_rejected_with_the_alternatives() -> None:
    with pytest.raises(RegistryValidationError) as excinfo:
        RegistryStatus.parse("almost_approved")
    assert "candidate" in str(excinfo.value)


def test_vocabulary_parsing_is_case_and_separator_insensitive() -> None:
    assert RegistryModality.parse(" Image ") is RegistryModality.IMAGE
    assert ResearchRole.parse("in-domain-test") is ResearchRole.IN_DOMAIN_TEST


def test_unverified_markers_are_the_documented_ones() -> None:
    assert {"unknown", "pending", "pending_verification", "not_verified"} <= UNVERIFIED_MARKERS
    assert is_unverified("  Not_Verified ")
    assert not is_unverified("CC-BY-4.0")


def test_generalization_groups_are_the_three_planned_experiments() -> None:
    assert set(GENERALIZATION_GROUPS) == {"image_deepfake", "audio_deepfake", "video_deepfake"}


DOC_PATH = ROOT / "docs" / "dataset_registry.md"


@pytest.mark.parametrize(
    "vocabulary",
    [
        RegistryModality,
        ResearchRole,
        RegistryStatus,
        SourceType,
        AccessType,
        LicenseStatus,
        LabelScheme,
        GeneralizationStatus,
    ],
)
def test_the_documentation_lists_exactly_the_live_vocabulary(vocabulary: Any) -> None:
    """A documented vocabulary that drifted from the code is a lie in a safety document."""
    text = DOC_PATH.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith(f"| `{vocabulary.__name__}`"))
    documented = {cell.strip().strip("`") for cell in row.split("|")[2].split(",") if cell.strip()}
    assert documented == {value.value for value in vocabulary}


def test_the_documentation_lists_exactly_the_live_markers() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "`unknown`, `pending`, `pending_verification`, `not_verified`, `not_applicable`" in text


def test_the_documentation_lists_exactly_the_live_groups() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "Three groups are declared: `image_deepfake`, `audio_deepfake`, `video_deepfake`" in text


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


def test_a_minimal_registry_validates(tmp_path: Path) -> None:
    path = write_registry(tmp_path)
    report = validate_registry(path)
    assert report.is_valid, "\n".join(report.errors)
    assert report.dataset_count == 2
    assert report.generalization_count == 1


def test_loading_returns_entries_sorted_by_id(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={
            "zebra": _source_entry("zebra"),
            "alpha": _target_entry("alpha"),
            "middle": _target_entry("middle"),
        },
        generalization=[],
    )
    registry = load_registry(path)
    assert list(registry.ids()) == ["alpha", "middle", "zebra"]
    assert len(registry) == 3


def test_missing_file_is_reported_not_raised(tmp_path: Path) -> None:
    report = validate_registry(tmp_path / "absent.yaml")
    assert not report.is_valid
    assert "not found" in report.errors[0]


def test_missing_file_raises_for_load(tmp_path: Path) -> None:
    with pytest.raises(RegistryNotFoundError):
        load_registry(tmp_path / "absent.yaml")


def test_malformed_yaml_is_reported_with_the_path(tmp_path: Path) -> None:
    path = write_raw_registry(tmp_path, "registry_version: 1\ndatasets: [unclosed\n")
    report = validate_registry(path)
    assert not report.is_valid
    assert "could not parse YAML" in report.errors[0]


def test_empty_file_is_reported(tmp_path: Path) -> None:
    path = write_raw_registry(tmp_path, "")
    assert "empty" in errors_of(path)[0]


def test_duplicate_yaml_keys_are_refused(tmp_path: Path) -> None:
    """PyYAML would keep the last value silently; a registry may not disagree with itself."""
    text = "registry_version: 1\ndatasets:\n  one:\n    dataset_id: one\n    dataset_id: two\n"
    report = validate_registry(write_raw_registry(tmp_path, text))
    assert not report.is_valid
    assert_mentions(report.errors, "duplicate key")


def test_unsupported_registry_version_is_refused(tmp_path: Path) -> None:
    path = write_registry(tmp_path, registry_version=99)
    assert_mentions(errors_of(path), "registry_version 99 is not supported")


def test_non_mapping_root_is_refused(tmp_path: Path) -> None:
    path = write_raw_registry(tmp_path, "- just\n- a\n- list\n")
    assert_mentions(errors_of(path), "root must be a mapping")


def test_empty_registry_is_refused(tmp_path: Path) -> None:
    """An empty registry would validate trivially while describing nothing."""
    path = write_registry(tmp_path, datasets={}, generalization=[])
    assert_mentions(errors_of(path), "must not be empty")


def test_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    path = write_registry(tmp_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["datset_registry"] = {}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert_mentions(errors_of(path), "unknown field 'datset_registry'")


def test_unknown_entry_field_is_refused(tmp_path: Path) -> None:
    """A typo in a field name would otherwise look like a configured, working field."""
    path = write_registry(
        tmp_path, datasets={"source_corpus": _source_entry(generalisation_groups=[])}
    )
    assert_mentions(errors_of(path), "unknown field 'generalisation_groups'")


def test_every_missing_required_field_is_reported_at_once(tmp_path: Path) -> None:
    entry = _source_entry()
    for field in ("name", "status", "train_allowed", "notes"):
        del entry[field]
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    messages = errors_of(path)
    for field in ("name", "status", "train_allowed", "notes"):
        assert_mentions(messages, f"missing required field '{field}'")


def test_null_is_not_accepted_for_a_required_text_field(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={"source_corpus": _source_entry(source_reference=None)},
        generalization=[],
    )
    assert_mentions(errors_of(path), "must not be null; use an unverified marker instead")


def test_missing_generalization_group_field_is_refused(tmp_path: Path) -> None:
    entry = _source_entry()
    del entry["generalization_groups"]
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "missing required field 'generalization_groups'")


def test_singular_generalization_group_is_accepted(tmp_path: Path) -> None:
    """One group may be written without a list; the field name is not a formality."""
    entry = _source_entry()
    entry["generalization_group"] = entry.pop("generalization_groups")
    path = write_registry(
        tmp_path,
        datasets={"source_corpus": entry, "target_corpus": _target_entry()},
        generalization=[],
    )
    assert load_registry(path).get("source_corpus").generalization_groups == ("video_deepfake",)


def test_declaring_both_group_spellings_is_refused(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path, datasets={"source_corpus": _source_entry(generalization_group="video_deepfake")}
    )
    assert_mentions(errors_of(path), "declare only one of")


def test_duplicate_dataset_id_is_refused(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={
            "first_key": _source_entry("shared_id"),
            "second_key": _source_entry("shared_id"),
        },
        generalization=[],
    )
    assert_mentions(errors_of(path), "duplicate dataset_id")


def test_unknown_group_is_refused_and_names_the_alternatives(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path, datasets={"source_corpus": _source_entry(generalization_groups=["qr_phishing"])}
    )
    assert_mentions(errors_of(path), "unknown group 'qr_phishing'", "image_deepfake")


def test_unknown_modality_is_refused(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path, datasets={"source_corpus": _source_entry(modality="hologram")}, generalization=[]
    )
    assert_mentions(errors_of(path), "modality", "invalid RegistryModality value")


def test_non_boolean_usage_flag_is_refused(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path, datasets={"source_corpus": _source_entry(train_allowed="yes")}, generalization=[]
    )
    assert_mentions(errors_of(path), "train_allowed must be true or false")


def test_optional_flag_may_hold_a_marker_but_not_anything_else(tmp_path: Path) -> None:
    ok = write_registry(
        tmp_path,
        datasets={"source_corpus": _source_entry(requires_request="unknown")},
        generalization=[],
    )
    assert validate_registry(ok).is_valid

    bad = write_registry(
        tmp_path,
        datasets={"source_corpus": _source_entry(requires_request="probably")},
        generalization=[],
        filename="bad.yaml",
    )
    assert_mentions(errors_of(bad), "requires_request must be true, false")


def test_manipulation_marker_means_unrecorded_not_a_one_item_list(tmp_path: Path) -> None:
    entry = load_registry(
        write_registry(tmp_path, datasets={"source_corpus": _source_entry()}, generalization=[])
    ).get("source_corpus")
    assert entry.manipulation_types is None
    assert "manipulation_types" in entry.unverified_fields()


def test_manipulation_list_is_kept_verbatim(tmp_path: Path) -> None:
    entry = load_registry(
        write_registry(
            tmp_path,
            datasets={"source_corpus": _source_entry(manipulation_types=["face_swap"])},
            generalization=[],
        )
    ).get("source_corpus")
    assert entry.manipulation_types == ("face_swap",)
    assert "manipulation_types" not in entry.unverified_fields()


def test_label_scheme_must_be_known(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={"source_corpus": _source_entry(label_schema={"scheme": "vibes"})},
        generalization=[],
    )
    assert_mentions(errors_of(path), "invalid LabelScheme value")


def test_access_date_must_look_like_a_date(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={
            "source_corpus": _source_entry(
                acquisition={
                    "acquired": True,
                    "access_date": "last tuesday",
                    "version": "v1",
                    "local_path": "datasets/raw/source_corpus",
                }
            )
        },
        generalization=[],
    )
    assert_mentions(errors_of(path), "access_date must be YYYY-MM-DD")


def test_unknown_key_inside_a_nested_block_is_refused(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={"source_corpus": _source_entry(acquisition={"acquired": False, "who": "me"})},
        generalization=[],
    )
    assert_mentions(errors_of(path), "unknown field 'who'")


# ---------------------------------------------------------------------------
# Semantic gates
# ---------------------------------------------------------------------------


def test_training_on_a_target_domain_is_refused(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={"target_corpus": _target_entry(train_allowed=True)},
        generalization=[],
    )
    assert_mentions(errors_of(path), "voids the unseen-test claim")


def test_validating_against_a_target_domain_is_refused(tmp_path: Path) -> None:
    """Tuning against the target leaks it just as thoroughly as training on it."""
    path = write_registry(
        tmp_path,
        datasets={"target_corpus": _target_entry(validation_allowed=True)},
        generalization=[],
    )
    assert_mentions(errors_of(path), "tuning", "voids the unseen-test claim")


def test_unseen_test_without_the_target_role_is_refused(tmp_path: Path) -> None:
    entry = _target_entry(research_role=["unseen_test"])
    path = write_registry(tmp_path, datasets={"target_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "requires research_role 'target_domain'")


def test_target_role_without_unseen_test_is_refused(tmp_path: Path) -> None:
    entry = _target_entry(unseen_test_allowed=False)
    path = write_registry(tmp_path, datasets={"target_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "'target_domain' requires unseen_test_allowed")


def test_target_domain_may_not_double_as_an_in_domain_test(tmp_path: Path) -> None:
    entry = _target_entry(test_allowed=True)
    path = write_registry(tmp_path, datasets={"target_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "cannot also be an in-domain test set")


def test_source_role_without_training_is_refused(tmp_path: Path) -> None:
    entry = _source_entry(train_allowed=False)
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "'source_domain' requires train_allowed")


def test_approved_status_requires_a_verified_licence(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={"source_corpus": _source_entry(status="approved")},
        generalization=[],
    )
    assert_mentions(errors_of(path), "status 'approved' requires license_status 'verified'")


def test_blocked_status_refuses_usage(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path, datasets={"source_corpus": _source_entry(status="blocked")}, generalization=[]
    )
    assert_mentions(errors_of(path), "'blocked' requires every usage flag to be false")


def test_available_locally_requires_an_acquisition_record(tmp_path: Path) -> None:
    """Possession is the one thing the registry may assert, and only with evidence."""
    entry = _source_entry(status="available_locally")
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "requires acquisition.acquired to be true")


def test_available_locally_requires_a_complete_acquisition_record(tmp_path: Path) -> None:
    entry = _source_entry(
        status="available_locally",
        acquisition={"acquired": True, "access_date": "2026-01-01", "version": "v1"},
    )
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "complete acquisition record", "local_path")


def test_acquired_data_must_be_recorded_as_available_locally(tmp_path: Path) -> None:
    entry = _source_entry(
        acquisition={
            "acquired": True,
            "access_date": "2026-01-01",
            "version": "v1",
            "local_path": "x",
        }
    )
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "must be recorded as 'available_locally'")


def test_multimodal_requires_two_components(tmp_path: Path) -> None:
    entry = _source_entry(modality="multimodal", derived_modalities=["audio"])
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "requires derived_modalities naming at least two")


def test_intelligence_source_cannot_declare_class_labels(tmp_path: Path) -> None:
    entry = _intelligence_entry(label_schema={"scheme": "binary_real_fake", "values": ["real"]})
    path = write_registry(tmp_path, datasets={"some_feed": entry}, generalization=[])
    assert_mentions(errors_of(path), "cannot declare class labels")


def test_intelligence_source_cannot_be_a_training_corpus(tmp_path: Path) -> None:
    entry = _intelligence_entry(train_allowed=True)
    path = write_registry(tmp_path, datasets={"some_feed": entry}, generalization=[])
    assert_mentions(errors_of(path), "cannot be a training or validation corpus")


def test_detection_corpus_requires_a_label_scheme(tmp_path: Path) -> None:
    entry = _source_entry()
    del entry["label_schema"]
    path = write_registry(tmp_path, datasets={"source_corpus": entry}, generalization=[])
    assert_mentions(errors_of(path), "no label_schema declared")


# ---------------------------------------------------------------------------
# Generalization relationships
# ---------------------------------------------------------------------------


def test_pair_with_an_unregistered_source_is_refused(tmp_path: Path) -> None:
    path = write_registry(tmp_path, generalization=[_pair(source="absent_corpus")])
    assert_mentions(errors_of(path), "source 'absent_corpus' is not a registered dataset")


def test_pair_pointing_at_itself_is_refused(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={"source_corpus": _source_entry()},
        generalization=[_pair(target="source_corpus")],
    )
    assert_mentions(errors_of(path), "source and target are the same dataset")


def test_pair_source_must_be_trainable(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={
            "source_corpus": _source_entry(research_role=["source_domain"], train_allowed=False),
            "target_corpus": _target_entry(),
        },
    )
    assert_mentions(errors_of(path), "cannot be trained on")


def test_pair_target_must_be_held_out(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={
            "source_corpus": _source_entry(),
            "target_corpus": _target_entry(unseen_test_allowed=False),
        },
    )
    assert_mentions(errors_of(path), "not usable as a held-out target")


def test_both_endpoints_must_declare_the_group(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={
            "source_corpus": _source_entry(generalization_groups=[]),
            "target_corpus": _target_entry(),
        },
    )
    assert_mentions(errors_of(path), "does not declare group 'video_deepfake'")


def test_duplicate_group_pair_is_refused(tmp_path: Path) -> None:
    path = write_registry(tmp_path, generalization=[_pair(), _pair(target="target_corpus")])
    assert_mentions(errors_of(path), "duplicate group_id")


def test_unknown_pair_group_is_refused(tmp_path: Path) -> None:
    path = write_registry(tmp_path, generalization=[_pair(group_id="qr_phishing")])
    assert_mentions(errors_of(path), "is not a declared generalization group")


def test_executed_flag_requires_available_endpoints(tmp_path: Path) -> None:
    path = write_registry(tmp_path, generalization=[_pair(status="executed", executed=True)])
    assert_mentions(errors_of(path), "executed: true requires the source", "available_locally")


def test_executed_status_requires_the_flag(tmp_path: Path) -> None:
    path = write_registry(tmp_path, generalization=[_pair(status="executed")])
    assert_mentions(errors_of(path), "status 'executed' requires executed: true")


def test_intelligence_source_cannot_take_part_in_a_pair(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        datasets={
            "some_feed": _intelligence_entry("some_feed"),
            "target_corpus": _target_entry(),
        },
        generalization=[_pair(source="some_feed")],
    )
    assert_mentions(errors_of(path), "cannot take part in a generalization experiment")


def test_blocked_endpoint_blocks_the_pair(tmp_path: Path) -> None:
    blocked = _source_entry(
        status="blocked",
        train_allowed=False,
        validation_allowed=False,
        test_allowed=False,
        research_role=["source_domain"],
    )
    path = write_registry(
        tmp_path,
        datasets={"source_corpus": blocked, "target_corpus": _target_entry()},
        generalization=[_pair()],
    )
    assert_mentions(errors_of(path), "no result can be reported for this pair")


def test_pairs_for_reports_every_relationship(tmp_path: Path) -> None:
    registry = load_registry(write_registry(tmp_path))
    assert len(registry.pairs_for("source_corpus")) == 1
    assert len(registry.pairs_for("target_corpus")) == 1
    assert registry.pairs_for("absent") == ()


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_warnings_do_not_make_a_registry_invalid(tmp_path: Path) -> None:
    report = validate_registry(write_registry(tmp_path))
    assert report.is_valid
    assert any("usage flags state intent only" in message for message in report.warnings)


def test_intelligence_source_warns_against_collecting_before_verification(tmp_path: Path) -> None:
    report = validate_registry(
        write_registry(
            tmp_path,
            datasets={"some_feed": _intelligence_entry()},
            generalization=[],
        )
    )
    assert report.is_valid
    assert any("no collector may be configured" in message for message in report.warnings)


def test_redistribution_conflict_is_reported(tmp_path: Path) -> None:
    report = validate_registry(
        write_registry(
            tmp_path,
            datasets={"source_corpus": _source_entry(redistribution_allowed=True)},
            generalization=[],
        )
    )
    assert any("never redistributing" in message for message in report.warnings)


def test_unverified_census_is_reported_once(tmp_path: Path) -> None:
    report = validate_registry(write_registry(tmp_path))
    census = [message for message in report.warnings if message.startswith("registry: unverified")]
    assert len(census) == 1
    assert "version=2" in census[0]


def test_group_without_a_pair_is_reported(tmp_path: Path) -> None:
    report = validate_registry(
        write_registry(
            tmp_path,
            datasets={
                "source_corpus": _source_entry(
                    generalization_groups=["video_deepfake", "audio_deepfake"]
                ),
                "target_corpus": _target_entry(),
            },
        )
    )
    assert any("no generalization pair defined" in message for message in report.warnings)


# ---------------------------------------------------------------------------
# The record objects
# ---------------------------------------------------------------------------


def test_entry_to_dict_round_trips_through_json() -> None:
    entry = load_registry(REGISTRY_PATH).get("tranco")
    payload = entry.to_dict()
    assert json.loads(json.dumps(payload))["dataset_id"] == "tranco"
    assert payload["status"] == "candidate"
    assert payload["acquisition"]["acquired"] is False


def test_entry_render_states_the_unknown_fields(tmp_path: Path, capsys: Any) -> None:
    entry = load_registry(REGISTRY_PATH).get("phishtank")
    text = entry.render()
    assert "unverified fields" in text
    assert "source_reference  : not_verified" in text
    assert "acquired locally  : False" in text


def test_entry_construction_rejects_a_non_boolean_flag() -> None:
    with pytest.raises(RegistryValidationError):
        DatasetEntry(
            dataset_id="x",
            name="X",
            modality=(RegistryModality.VIDEO,),
            source_type="research_dataset",
            research_role=(ResearchRole.TRAINING,),
            status=RegistryStatus.CANDIDATE,
            access_type=AccessType.UNKNOWN,
            license_status=LicenseStatus.PENDING_VERIFICATION,
            source_reference="not_verified",
            version="unknown",
            train_allowed="yes",  # type: ignore[arg-type]
            validation_allowed=False,
            test_allowed=False,
            unseen_test_allowed=False,
            generalization_groups=(),
        )


def test_pair_description_states_the_direction() -> None:
    pair = GeneralizationPair(group_id="video_deepfake", source="a", target="b")
    assert pair.describe() == "video_deepfake     a -> b [planned]"


def test_report_summary_states_validity(tmp_path: Path) -> None:
    report = validate_registry(write_registry(tmp_path))
    assert report.summary().startswith("VALID")
    assert "datasets=2" in report.summary()
    assert report.to_dict()["is_valid"] is True


def test_a_report_is_invalid_exactly_when_it_has_errors(tmp_path: Path) -> None:
    clean = RegistryReport(path=tmp_path / "x.yaml", registry_version=1, dataset_count=2)
    assert clean.is_valid
    assert clean.summary().startswith("VALID")

    broken = RegistryReport(path=tmp_path / "x.yaml", errors=["datasets.x: something is wrong"])
    assert not broken.is_valid
    assert broken.summary().startswith("INVALID")
    assert broken.to_dict()["errors"] == ["datasets.x: something is wrong"]


def test_registry_error_is_the_common_base() -> None:
    assert issubclass(RegistryNotFoundError, RegistryError)
    assert issubclass(RegistryValidationError, RegistryError)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_without_an_action_prints_help_and_exits_two(capsys: Any) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out


def test_cli_validate_returns_zero_for_the_repository_registry(capsys: Any) -> None:
    assert main(["--validate"]) == 0
    assert "VALID" in capsys.readouterr().out


def test_cli_validate_returns_one_for_an_invalid_registry(tmp_path: Path, capsys: Any) -> None:
    path = write_registry(tmp_path, datasets={"source_corpus": _source_entry(modality="hologram")})
    assert main(["--registry", str(path), "--validate"]) == 1
    assert "error   :" in capsys.readouterr().out


def test_cli_validate_json_is_machine_readable(tmp_path: Path, capsys: Any) -> None:
    path = write_registry(tmp_path)
    assert main(["--registry", str(path), "--validate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["is_valid"] is True
    assert payload["dataset_count"] == 2


def test_cli_list_prints_one_line_per_source(capsys: Any) -> None:
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "faceforensics_plus_plus" in out
    assert "audio_deepfake" in out


def test_cli_list_json_round_trips(capsys: Any) -> None:
    assert main(["--list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_version"] == REGISTRY_SCHEMA_VERSION
    assert payload["datasets"][0]["dataset_id"]


def test_cli_show_prints_the_entry_and_its_pairs(capsys: Any) -> None:
    assert main(["--show", "celeb_df_v2"]) == 0
    out = capsys.readouterr().out
    assert "unseen_test_allowed: True" in out
    assert "image_deepfake" in out


def test_cli_show_rejects_an_unknown_id(capsys: Any) -> None:
    assert main(["--show", "not_registered"]) == 2
    assert "unknown dataset id" in capsys.readouterr().err


def test_cli_reports_a_missing_file_as_a_bad_request(tmp_path: Path, capsys: Any) -> None:
    assert main(["--registry", str(tmp_path / "absent.yaml"), "--list"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_accepts_an_explicit_registry_path(tmp_path: Path, capsys: Any) -> None:
    path = write_registry(tmp_path)
    assert main(["--registry", str(path), "--validate"]) == 0
    assert str(path) in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The module never touches the network
# ---------------------------------------------------------------------------


def test_registry_source_contains_no_network_or_media_imports() -> None:
    forbidden = ("import requests", "import httpx", "urllib.request", "urlopen", "import socket")
    text = (ROOT / "datasets" / "registry.py").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in text.splitlines()
        if not line.strip().startswith("#") and any(token in line for token in forbidden)
    ]
    assert not offenders, f"network access found in the registry module: {offenders}"
