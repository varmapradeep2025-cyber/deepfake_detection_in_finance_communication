"""Tests for :mod:`ml.reproducibility.config` and :mod:`ml.reproducibility.environment`.

Confirms the config contract holds and that environment capture is JSON-serializable
and reports missing information as ``None`` rather than inventing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ml.reproducibility.config import (
    ConfigNotFoundError,
    ConfigValidationError,
    ProjectConfig,
    load_config,
    load_project_config,
    project_root,
)
from ml.reproducibility.environment import (
    OPTIONAL_PACKAGES,
    capture_environment,
    capture_package_versions,
    git_commit_hash,
    git_is_dirty,
    python_version_string,
)


class TestProjectConfig:
    def test_loads_default_config(self, config):
        assert config.stage == "foundation"

    def test_exposes_global_seed(self, config):
        assert isinstance(config.global_seed, int)
        assert config.global_seed > 0

    def test_exposes_multiple_experiment_seeds(self, config):
        assert len(config.experiment_seeds) >= 3

    def test_dotted_get(self, config):
        assert config.get("data.hashing.algorithm") == "sha256"

    def test_dotted_get_with_default(self, config):
        assert config.get("does.not.exist", default="fallback") == "fallback"

    def test_dotted_get_raises_without_default(self, config):
        with pytest.raises(KeyError, match="not found"):
            config.get("nope.nope")

    def test_get_path_is_absolute_and_exists(self, config):
        resolved = config.get_path("paths.source.datasets_raw", must_exist=True)
        assert resolved.is_absolute()
        assert resolved.is_dir()

    def test_get_path_rejects_missing_key(self, config):
        with pytest.raises(ConfigValidationError, match="is missing"):
            config.get_path("paths.source.absent_key")

    def test_get_path_rejects_nonexistent_location(self, tmp_path):
        raw = load_project_config().raw
        raw["paths"]["source"]["absent_dir"] = "no/such/place"
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="missing location"):
            load_config(path).get_path("paths.source.absent_dir", must_exist=True)

    def test_get_path_rejects_non_string(self, config):
        with pytest.raises(ConfigValidationError, match="path string"):
            config.get_path("reproducibility.global_seed")

    def test_sections_are_mappings(self, config):
        for name in ("project", "reproducibility", "data", "experiments", "modules"):
            assert isinstance(config.get(name), dict)

    def test_no_modules_enabled_at_foundation_stage(self, config):
        assert config.enabled_modules() == ()

    def test_unknown_module_reports_disabled(self, config):
        assert config.is_module_enabled("not_a_module") is False

    def test_all_declared_source_paths_exist(self, config):
        for name, relative in config.raw["paths"]["source"].items():
            assert (config.root / relative).exists(), f"paths.source.{name}: {relative}"

    def test_all_declared_module_paths_exist(self, config):
        for name, module in config.raw["modules"].items():
            assert (Path(config.root) / module["path"]).is_dir(), f"modules.{name}"

    def test_config_is_immutable(self, config):
        with pytest.raises((AttributeError, TypeError)):
            config.raw = {}  # type: ignore[misc]

    def test_raw_exposes_the_full_mapping(self, config):
        assert config.raw["project"]["stage"] == "foundation"
        assert "reproducibility" in config.raw

    def test_project_root_contains_expected_layout(self):
        root = project_root()
        assert (root / "configs" / "project.yaml").is_file()
        assert (root / "ml" / "reproducibility").is_dir()


class TestConfigErrors:
    def test_missing_file_raises_not_found(self, tmp_path):
        with pytest.raises(ConfigNotFoundError, match="project config not found"):
            load_config(tmp_path / "absent.yaml")

    def test_malformed_yaml_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="could not parse YAML"):
            load_config(path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="empty"):
            load_config(path)

    def test_non_mapping_root_raises(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="must be a mapping"):
            load_config(path)

    def test_missing_sections_are_all_reported(self, tmp_path):
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"project": {"stage": "x"}}), encoding="utf-8")
        with pytest.raises(ConfigValidationError) as exc:
            load_config(path)
        assert len(exc.value.errors) >= 6

    def test_invalid_seed_type_is_reported(self, tmp_path):
        good = load_project_config().raw
        good["reproducibility"]["global_seed"] = "not-an-int"
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(good), encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="global_seed"):
            load_config(path)

    def test_missing_paths_group_is_reported(self, tmp_path):
        good = load_project_config().raw
        del good["paths"]["source"]
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(good), encoding="utf-8")
        with pytest.raises(ConfigValidationError, match=r"paths\.source"):
            load_config(path)

    def test_empty_seed_list_is_rejected(self, tmp_path):
        good = load_project_config().raw
        good["experiments"]["seeds"] = []
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(good), encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="seeds"):
            load_config(path)

    def test_valid_config_round_trips(self, tmp_path):
        good = load_project_config().raw
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump(good), encoding="utf-8")
        reloaded = load_config(path)
        assert reloaded.global_seed == load_project_config().global_seed

    def test_project_config_is_constructible_directly(self):
        config = ProjectConfig(
            raw={"project": {"stage": "x"}, "reproducibility": {"global_seed": 1}},
            source_path=Path("configs/project.yaml"),
            root=project_root(),
        )
        assert config.stage == "x"


class TestEnvironmentCapture:
    def test_returns_report(self):
        report = capture_environment()
        assert report.python_version
        assert report.operating_system

    def test_is_json_serializable(self):
        payload = capture_environment().as_dict()
        assert json.loads(json.dumps(payload))

    def test_to_json_is_stable(self):
        first = capture_environment().to_json()
        second = capture_environment().to_json()
        assert json.loads(first)["platform"] == json.loads(second)["platform"]

    def test_records_python_version(self):
        report = capture_environment()
        assert report.python_version.startswith("3.")
        assert report.python_implementation

    def test_records_byte_order_and_encoding(self):
        report = capture_environment()
        assert report.byte_order in {"little", "big"}
        assert report.filesystem_encoding

    def test_core_packages_reported(self):
        packages = capture_environment().packages
        assert packages["numpy"]
        assert packages["PyYAML"]

    def test_missing_package_is_none_not_guessed(self):
        assert capture_package_versions(("definitely-not-installed-xyz",)) == {
            "definitely-not-installed-xyz": None
        }

    def test_optional_packages_recorded_as_absent(self):
        """Heavy stacks are reported as unavailable, documenting their absence."""
        optional = capture_environment().optional_packages
        assert set(optional) == set(OPTIONAL_PACKAGES)
        assert optional["torch"] is None

    def test_optional_packages_can_be_excluded(self):
        assert capture_environment(include_optional_packages=False).optional_packages == {}

    def test_git_fields_are_none_outside_a_checkout(self, tmp_path):
        assert git_commit_hash(tmp_path) is None
        assert git_is_dirty(tmp_path) is None

    def test_timestamp_can_be_supplied(self):
        report = capture_environment(timestamp="2026-01-01T00:00:00+00:00")
        assert report.captured_at == "2026-01-01T00:00:00+00:00"

    def test_captures_pyhashseed_when_set(self):
        import os

        os.environ["PYTHONHASHSEED"] = "12345"
        assert capture_environment().relevant_env_vars["PYTHONHASHSEED"] == "12345"

    def test_python_version_string_is_single_line(self):
        assert "\n" not in python_version_string()


@pytest.mark.parametrize("path_key", ["paths.source.datasets_raw", "paths.source.configs"])
def test_config_paths_are_relative_to_root(config: ProjectConfig, path_key: str):
    assert config.get_path(path_key).is_relative_to(config.root)
