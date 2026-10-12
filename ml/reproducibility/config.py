"""Safe, typed access to ``configs/project.yaml``.

``configs/project.yaml`` is the single source of truth for paths, seeds, and module
toggles. This module loads it defensively: a malformed or missing file raises a
specific error instead of yielding a half-populated object that fails later and far
from the cause.

Example
-------
>>> config = load_project_config()  # doctest: +SKIP
>>> config.global_seed  # doctest: +SKIP
20260101
>>> config.get_path("datasets.raw")  # doctest: +SKIP
'datasets/raw'
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "ConfigNotFoundError",
    "ConfigValidationError",
    "ProjectConfig",
    "load_config",
    "load_project_config",
    "project_root",
]

# Repository root is this file's parent-parent-parent:
#   <root>/ml/reproducibility/config.py
_DEFAULT_CONFIG_RELPATH = Path("configs") / "project.yaml"


class ConfigError(Exception):
    """Base class for configuration problems."""


class ConfigNotFoundError(ConfigError):
    """Raised when the configuration file cannot be located."""


class ConfigValidationError(ConfigError):
    """Raised when the configuration file is present but structurally invalid.

    ``str(error)`` includes every problem found, so a traceback alone is enough to
    diagnose the failure without inspecting :attr:`errors`.
    """

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        if self.errors:
            message = f"{message}: " + "; ".join(self.errors)
        super().__init__(message)


def project_root() -> Path:
    """Absolute path to the repository root.

    Derived from this module's location rather than the working directory, so
    config loads identically regardless of where a command is invoked from.
    """
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Read-only view over the parsed project configuration.

    The raw mapping is held in :attr:`raw`; the section shortcuts below expose the
    individual top-level sections. Dotted lookups
    (``config.get("reproducibility.global_seed")``) are supported so that callers
    can reach nested settings without a bespoke accessor per key.

    Attributes:
        raw: The complete parsed mapping, unmodified.
        source_path: Absolute path of the file this config was read from.
        root: Repository root that relative paths resolve against.
    """

    raw: dict[str, Any]
    source_path: Path
    root: Path

    # --- section shortcuts -------------------------------------------------

    @property
    def project(self) -> dict[str, Any]:
        """The ``project:`` section."""
        return self._section("project")

    @property
    def reproducibility(self) -> dict[str, Any]:
        """The ``reproducibility:`` section."""
        return self._section("reproducibility")

    @property
    def data(self) -> dict[str, Any]:
        """The ``data:`` section."""
        return self._section("data")

    @property
    def experiments(self) -> dict[str, Any]:
        """The ``experiments:`` section."""
        return self._section("experiments")

    @property
    def modules(self) -> dict[str, Any]:
        """The ``modules:`` section."""
        return self._section("modules")

    @property
    def services(self) -> dict[str, Any]:
        """The ``services:`` section."""
        return self._section("services")

    @property
    def threat_intelligence(self) -> dict[str, Any]:
        """The ``threat_intelligence:`` section."""
        return self._section("threat_intelligence")

    # --- derived values ----------------------------------------------------

    @property
    def global_seed(self) -> int:
        """The configured ``reproducibility.global_seed``."""
        value = self.reproducibility.get("global_seed")
        if not isinstance(value, int):
            raise ConfigValidationError("reproducibility.global_seed must be an integer")
        return value

    @property
    def experiment_seeds(self) -> tuple[int, ...]:
        """The configured seed list used for repeated runs."""
        seeds = self.experiments.get("seeds", [])
        if not isinstance(seeds, list) or not seeds:
            raise ConfigValidationError("experiments.seeds must be a non-empty list")
        return tuple(int(seed) for seed in seeds)

    @property
    def stage(self) -> str:
        """The configured project stage, e.g. ``foundation``."""
        return str(self.project.get("stage", "unknown"))

    # --- accessors ---------------------------------------------------------

    def _section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigValidationError(f"config section {name!r} is missing or not a mapping")
        return value

    def get(self, dotted_key: str, default: Any = ...) -> Any:
        """Fetch a nested value by dotted path.

        Args:
            dotted_key: e.g. ``"reproducibility.global_seed"``.
            default: Returned when the key is absent. When omitted, a missing key
                raises :class:`KeyError` instead of silently yielding ``None``.

        Raises:
            KeyError: If the key is absent and no ``default`` was supplied.
            ConfigValidationError: If an intermediate segment is not a mapping.
        """
        current: Any = self.raw
        walked: list[str] = []
        for part in dotted_key.split("."):
            walked.append(part)
            if not isinstance(current, dict) or part not in current:
                if default is not ...:
                    return default
                raise KeyError(f"config key not found: {'.'.join(walked)}")
            current = current[part]
        return current

    def get_path(self, dotted_key: str, *, must_exist: bool = False) -> Path:
        """Resolve a configured path to an absolute :class:`Path`.

        Args:
            dotted_key: Dotted path to a path value, e.g. ``"paths.source.datasets_raw"``.
            must_exist: Raise if the resolved location is absent. Use for paths that
                must be present in a checkout; leave ``False`` for runtime output
                directories that are created on demand.

        Raises:
            ConfigValidationError: If the key is missing, the value is not a string,
                or ``must_exist`` is set and the location does not exist.
        """
        try:
            raw = self.get(dotted_key)
        except KeyError as exc:
            raise ConfigValidationError(f"config key {dotted_key!r} is missing") from exc
        if not isinstance(raw, str):
            raise ConfigValidationError(
                f"config key {dotted_key!r} must be a path string, got {type(raw).__name__}"
            )
        resolved = (self.root / raw).resolve()
        if must_exist and not resolved.exists():
            raise ConfigValidationError(
                f"config key {dotted_key!r} points to a missing location: {raw}"
            )
        return resolved

    def is_module_enabled(self, module_name: str) -> bool:
        """Whether a planned module is enabled.

        Unknown module names return ``False``: an unregistered module must not be
        treated as active.
        """
        entry = self.modules.get(module_name)
        if not isinstance(entry, dict):
            return False
        return bool(entry.get("enabled", False))

    def enabled_modules(self) -> tuple[str, ...]:
        """Sorted names of all enabled modules."""
        return tuple(sorted(name for name, entry in self.modules.items() if entry.get("enabled")))


def _validate(data: dict[str, Any]) -> None:
    """Structural checks that must pass before the config is handed out."""
    errors: list[str] = []

    for section in (
        "project",
        "reproducibility",
        "paths",
        "data",
        "experiments",
        "modules",
        "evaluation",
        "services",
    ):
        if section not in data:
            errors.append(f"missing required section: {section}")
        elif not isinstance(data[section], dict):
            errors.append(f"section {section!r} must be a mapping")

    if not errors:
        repro = data["reproducibility"]
        if not isinstance(repro.get("global_seed"), int):
            errors.append("reproducibility.global_seed must be an integer")

        paths = data["paths"]
        for group in ("source", "generated"):
            if group not in paths:
                errors.append(f"paths.{group} is required (source vs generated)")
            elif not isinstance(paths[group], dict):
                errors.append(f"paths.{group} must be a mapping")

        seeds = data["experiments"].get("seeds")
        if not isinstance(seeds, list) or len(seeds) < 1:
            errors.append("experiments.seeds must be a non-empty list")
        elif not all(isinstance(seed, int) for seed in seeds):
            errors.append("experiments.seeds must contain only integers")

    if errors:
        raise ConfigValidationError(
            f"invalid project configuration ({len(errors)} problem(s))", errors=errors
        )


def load_config(path: Path | str | None = None) -> ProjectConfig:
    """Load and validate the project configuration.

    Args:
        path: Explicit config file. Defaults to ``<repo root>/configs/project.yaml``.

    Returns:
        A validated :class:`ProjectConfig`.

    Raises:
        ConfigNotFoundError: If the file does not exist.
        ConfigValidationError: If the YAML is malformed or fails structural checks.
    """
    root = project_root()
    config_path = Path(path).resolve() if path else root / _DEFAULT_CONFIG_RELPATH

    if not config_path.is_file():
        raise ConfigNotFoundError(f"project config not found: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"could not parse YAML at {config_path}: {exc}") from exc

    if loaded is None:
        raise ConfigValidationError(f"config file is empty: {config_path}")
    if not isinstance(loaded, dict):
        raise ConfigValidationError(f"config root must be a mapping, got {type(loaded).__name__}")

    _validate(loaded)
    return ProjectConfig(raw=loaded, source_path=config_path, root=root)


def load_project_config() -> ProjectConfig:
    """Convenience wrapper loading the default ``configs/project.yaml``."""
    return load_config()
