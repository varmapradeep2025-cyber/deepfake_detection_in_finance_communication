"""Integrity tests for the research project foundation.

These tests assert the *structure* and *reproducibility conventions* established in
configs/project.yaml and .gitignore. They are deliberately cheap: no ML imports,
no dataset access, no network.

Run with:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip(
        "PyYAML is not installed; run: pip install -r requirements.txt",
        allow_module_level=True,
    )

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.yaml"

REQUIRED_DIRECTORIES = (
    "backend",
    "frontend",
    "ml",
    "ml/image",
    "ml/audio",
    "ml/video",
    "ml/fusion",
    "ml/generalization",
    "ml/explainability",
    "ml/risk",
    "forensic",
    "forensic/metadata",
    "forensic/hashing",
    "forensic/provenance",
    "forensic/chain_of_custody",
    "forensic/evidence",
    "threat_intelligence",
    "threat_intelligence/collectors",
    "threat_intelligence/source_registry",
    "threat_intelligence/fingerprinting",
    "threat_intelligence/similarity",
    "threat_intelligence/campaign",
    "threat_intelligence/correlation",
    "datasets",
    "datasets/raw",
    "datasets/processed",
    "datasets/manifests",
    "datasets/splits",
    "datasets/adapters",
    "configs/adapters",
    "experiments",
    "experiments/image",
    "experiments/audio",
    "experiments/video",
    "experiments/cross_dataset",
    "experiments/multimodal",
    "experiments/ablation",
    "reports",
    "notebooks",
    "scripts",
    "tests",
    "configs",
    "docs",
)

REQUIRED_ROOT_FILES = (
    "README.md",
    ".gitignore",
    "requirements.txt",
    "configs/project.yaml",
    "configs/datasets.yaml",
    "ruff.toml",
    "mypy.ini",
)

# Python modules that must exist for the reproducibility + schema layer.
REQUIRED_MODULES = (
    "ml/__init__.py",
    "ml/reproducibility/__init__.py",
    "ml/reproducibility/seeds.py",
    "ml/reproducibility/environment.py",
    "ml/reproducibility/config.py",
    "datasets/__init__.py",
    "datasets/schema.py",
    "datasets/manifest.py",
    "datasets/split_schema.py",
    "datasets/normalization.py",
    "datasets/provenance.py",
    "datasets/ingest.py",
    "datasets/registry.py",
    "datasets/adapters/__init__.py",
    "datasets/adapters/base.py",
    "datasets/adapters/template.py",
    "experiments/__init__.py",
    "experiments/schema.py",
    "experiments/results.py",
    "experiments/run_artifacts.py",
)

# The five files that constitute a completed run record.
REQUIRED_RUN_ARTIFACTS = (
    "config.json",
    "metrics.json",
    "environment.json",
    "dataset_info.json",
    "README.md",
)

# Heavy ML stacks must stay commented out during the foundation stage.
HEAVY_DEPS = ("torch", "torchvision", "transformers", "timm", "tensorflow", "fastapi")

# Patterns that must be ignored so media and weights never enter git.
REQUIRED_IGNORE_PATTERNS = (
    "datasets/raw/**",
    "datasets/processed/**",
    "artifacts/",
    "*.ckpt",
    "*.pt",
    "*.onnx",
    "experiments/runs/",
    ".env",
)


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def gitignore_text() -> str:
    return (ROOT / ".gitignore").read_text(encoding="utf-8")


# --- Structure --------------------------------------------------------------


@pytest.mark.parametrize("relative", REQUIRED_DIRECTORIES)
def test_required_directory_exists(relative: str) -> None:
    assert (ROOT / relative).is_dir(), f"missing directory: {relative}"


@pytest.mark.parametrize("relative", REQUIRED_ROOT_FILES)
def test_required_root_file_exists(relative: str) -> None:
    assert (ROOT / relative).is_file(), f"missing file: {relative}"


@pytest.mark.parametrize("relative", REQUIRED_MODULES)
def test_required_module_exists(relative: str) -> None:
    assert (ROOT / relative).is_file(), f"missing module: {relative}"


def test_runs_directory_exists() -> None:
    assert (ROOT / "experiments" / "runs").is_dir()


def test_run_artifact_layout_is_documented() -> None:
    """The run layout must be stated in code, not only in prose."""
    from experiments.run_artifacts import REQUIRED_RUN_FILES

    assert set(REQUIRED_RUN_FILES) == set(REQUIRED_RUN_ARTIFACTS)


def test_no_run_directories_exist_yet() -> None:
    """Importing the package must not fabricate experiment runs."""
    runs_root = ROOT / "experiments" / "runs"
    produced = [child.name for child in runs_root.iterdir() if child.is_dir()]
    assert produced == [], f"unexpected run directories: {produced}"


def test_empty_directories_are_placeheld_for_git() -> None:
    """Every directory that holds no code must carry a .gitkeep so git preserves it."""
    missing = [
        rel
        for rel in REQUIRED_DIRECTORIES
        if not any((ROOT / rel).iterdir()) and not (ROOT / rel / ".gitkeep").exists()
    ]
    assert not missing, f"empty dirs without .gitkeep: {missing}"


def test_no_heavy_ml_code_is_present() -> None:
    """Guard against a heavy dependency creeping into a module import."""
    forbidden = (
        "import torch",
        "import cv2",
        "import librosa",
        "import fastapi",
        "import transformers",
        "import requests",
        "import httpx",
        "urllib.request",
    )
    offenders: list[str] = []
    for relative in REQUIRED_MODULES:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(token in stripped for token in forbidden):
                offenders.append(f"{relative}: {stripped}")
    assert not offenders, f"heavy or networked imports found: {offenders}"


# --- Configuration ----------------------------------------------------------


def test_config_parses_as_mapping(config: dict) -> None:
    assert isinstance(config, dict)


def test_config_declares_stage(config: dict) -> None:
    assert config["project"]["stage"] == "foundation"


def test_config_is_seeded(config: dict) -> None:
    reproducibility = config["reproducibility"]
    assert isinstance(reproducibility["global_seed"], int)
    assert reproducibility["deterministic_algorithms"] is True


def test_config_experiments_use_multiple_seeds(config: dict) -> None:
    """Single-seed results are not acceptable for this project."""
    seeds = config["experiments"]["seeds"]
    assert isinstance(seeds, list)
    assert len(seeds) >= 3


def test_config_requires_ablation_and_forbids_test_tuning(config: dict) -> None:
    assert config["experiments"]["ablation_required"] is True
    assert config["evaluation"]["prohibit_test_set_tuning"] is True


def test_config_verifies_data_hashes_on_load(config: dict) -> None:
    assert config["data"]["verify_hashes_on_load"] is True
    assert config["data"]["hashing"]["algorithm"] == "sha256"


def test_declared_source_paths_exist(config: dict) -> None:
    for name, relative in config["paths"]["source"].items():
        assert (ROOT / relative).exists(), f"paths.source.{name} missing: {relative}"


def test_declared_generated_paths_are_gitignored(config: dict, gitignore_text: str) -> None:
    """Runtime output dirs must exist in .gitignore and must not be committed."""
    for name, relative in config["paths"]["generated"].items():
        assert f"{relative}/" in gitignore_text, f"paths.generated.{name} not ignored: {relative}"
        assert not (ROOT / relative).exists(), (
            f"paths.generated.{name} should not be committed: {relative}"
        )


def test_declared_module_paths_exist(config: dict) -> None:
    for name, module in config["modules"].items():
        assert (ROOT / module["path"]).is_dir(), f"modules.{name} missing: {module['path']}"


# --- Ingestion --------------------------------------------------------------


def test_ingestion_requires_a_license_before_writing(config: dict) -> None:
    """Licensing governs whether the data may be used, so it is never assumed."""
    assert config["ingestion"]["require_license"] is True


def test_ingestion_is_declared_deterministic(config: dict) -> None:
    """The manifest carries no timestamp so identical inputs reproduce identical bytes."""
    assert config["ingestion"]["deterministic_manifest"] is True


def test_ingestion_paths_match_the_configured_convention(config: dict) -> None:
    ingestion = config["ingestion"]
    assert ingestion["raw_dir"] == config["paths"]["source"]["datasets_raw"]
    assert ingestion["manifest_dir"] == config["paths"]["source"]["datasets_manifests"]


def test_no_dataset_has_been_selected_yet(config: dict) -> None:
    """Plumbing exists for a dataset; no dataset has been chosen for the experiments yet."""
    assert config["ingestion"]["datasets_selected"] == []


# --- Dataset adapters ---------------------------------------------------------


def test_no_dataset_is_acquired_yet() -> None:
    """An adapter for a dataset is not a licence to use it, and neither is a directory.

    Asserted against the registry rather than the filesystem: what matters is what the
    project claims, and the claim lives in the reviewed file. A dataset on someone's disk
    that nobody has declared is not data this project may use.
    """
    from datasets.registry import load_registry

    registry = load_registry(ROOT / "configs" / "datasets.yaml")
    acquired = [
        entry.dataset_id
        for entry in registry
        if entry.is_acquired or entry.acquisition.is_complete()
    ]
    assert acquired == []


def test_the_faceforensics_adapter_ships_with_a_configuration() -> None:
    from datasets.adapters import adapter_names

    assert "faceforensics_plus_plus" in adapter_names()
    config = ROOT / "configs" / "adapters" / "faceforensics_plus_plus.yaml"
    assert config.is_file()


def test_the_faceforensics_adapter_ships_without_a_verified_licence() -> None:
    """The shipped licence is a marker, so ingestion and acquisition both stay closed."""
    from datasets.adapters.faceforensics_plus_plus import (
        FaceForensicsPlusPlusAdapter,
        is_unverified_license,
    )

    adapter = FaceForensicsPlusPlusAdapter.from_config_file(
        ROOT / "configs" / "adapters" / "faceforensics_plus_plus.yaml"
    )
    assert adapter is not None
    assert is_unverified_license(adapter.license_id)


def test_acquisition_layer_exists() -> None:
    assert (ROOT / "datasets" / "acquisition.py").is_file()
    assert (ROOT / "docs" / "dataset_acquisition.md").is_file()


def test_no_acquisition_record_ships_with_the_repository() -> None:
    """A record in git would claim an acquisition that never happened."""
    manifests = ROOT / "datasets" / "manifests"
    if manifests.is_dir():
        assert list(manifests.glob("*.acquisition.json")) == []


def test_ingestion_documentation_exists() -> None:
    assert (ROOT / "docs" / "dataset_ingestion.md").is_file()


# --- Dataset registry --------------------------------------------------------


def test_registry_documentation_exists() -> None:
    assert (ROOT / "docs" / "dataset_registry.md").is_file()


def test_project_config_points_at_the_registry(config: dict) -> None:
    registry = config["dataset_registry"]
    assert registry["file"] == "configs/datasets.yaml"
    assert (ROOT / registry["file"]).is_file()


def test_declared_registry_version_matches_the_code(config: dict) -> None:
    """A version in the config that the loader does not understand would validate nothing."""
    from datasets.registry import REGISTRY_SCHEMA_VERSION

    assert config["dataset_registry"]["registry_version"] == REGISTRY_SCHEMA_VERSION
    with (ROOT / "configs" / "datasets.yaml").open(encoding="utf-8") as handle:
        assert yaml.safe_load(handle)["registry_version"] == REGISTRY_SCHEMA_VERSION


def test_the_default_registry_is_the_configured_file(config: dict) -> None:
    from datasets.registry import default_registry_path

    assert default_registry_path() == (ROOT / config["dataset_registry"]["file"]).resolve()


def test_no_dataset_is_approved_or_acquired_in_the_config(config: dict) -> None:
    """Approval and acquisition are separate deliberate steps; neither has happened."""
    registry = config["dataset_registry"]
    assert registry["approved_datasets"] == []
    assert registry["datasets_acquired"] == []


def test_the_configured_registry_validates() -> None:
    from datasets.registry import default_registry_path, validate_registry

    report = validate_registry(default_registry_path())
    assert report.is_valid, "\n".join(report.errors)


def test_the_registry_documents_the_same_command(config: dict) -> None:
    """The command recorded in the config must be the one the documentation shows."""
    assert config["dataset_registry"]["command"] in (ROOT / "README.md").read_text(encoding="utf-8")
    assert config["dataset_registry"]["command"] in (
        ROOT / "docs" / "dataset_registry.md"
    ).read_text(encoding="utf-8")


# --- Safety gates -----------------------------------------------------------


def test_no_planned_module_is_enabled_yet(config: dict) -> None:
    enabled = [name for name, module in config["modules"].items() if module["enabled"]]
    assert not enabled, f"modules unexpectedly enabled at foundation stage: {enabled}"


def test_network_access_is_off_by_default(config: dict) -> None:
    threat_intel = config["threat_intelligence"]
    assert threat_intel["enabled"] is False
    assert threat_intel["allow_network_access"] is False
    assert threat_intel["require_opt_in"] is True


def test_services_are_disabled(config: dict) -> None:
    services = config["services"]
    assert services["api"]["enabled"] is False
    assert services["database"]["enabled"] is False
    assert services["frontend"]["enabled"] is False


# --- Requirements -----------------------------------------------------------


def test_heavy_ml_dependencies_stay_commented_out() -> None:
    """Foundation stage must not require the large ML installs."""
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    active = [
        line.strip().lower() for line in lines if line.strip() and not line.strip().startswith("#")
    ]
    for dep in HEAVY_DEPS:
        assert not any(entry.startswith(dep) for entry in active), (
            f"{dep} must remain commented out during the foundation stage"
        )


# --- Ignore rules -----------------------------------------------------------


@pytest.mark.parametrize("pattern", REQUIRED_IGNORE_PATTERNS)
def test_gitignore_excludes_data_and_weights(gitignore_text: str, pattern: str) -> None:
    assert pattern in gitignore_text, f"missing .gitignore entry: {pattern}"


def test_manifests_and_splits_remain_tracked(gitignore_text: str) -> None:
    """Manifests are the reproducible record of the data; they must not be ignored."""
    assert "datasets/manifests/**" in gitignore_text
    assert "datasets/splits/**" in gitignore_text
