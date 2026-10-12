"""A generic, configuration-driven adapter: the template for real dataset adapters.

Purpose
-------
This is the reference implementation of :class:`~datasets.adapters.base.DatasetAdapter`
and the only adapter currently registered. It exists so that writing an adapter for a
real dataset is a matter of *declaring a vocabulary*, not writing traversal code. It is
not an adapter for any particular dataset, and it is not research data.

What it is good for
-------------------
Datasets that encode authenticity and modality in their directory structure, which is
the common case::

    <source>/
    |-- image/real/*.png
    |-- image/fake/*.png
    |-- audio/real/*.wav
    |-- audio/fake/*.wav

Configuration
-------------
A JSON mapping with these keys:

``dataset_name``
    Human-readable name. Defaults to ``dataset_id``.
``license``
    Required. SPDX identifier or dataset-specific licence string.
``dataset_version``
    Dataset release identifier. Defaults to ``"unversioned"``.
``source_reference``
    Where the dataset came from. Optional, and left ``null`` rather than invented.
``access_date``
    When the data was obtained. Optional.
``labels``
    Required. ``{"<dataset token>": "real"|"fake", ...}``.
``modalities``
    Optional token table for datasets that spell modality as a word rather than
    recording it structurally. Directory rules usually make this unnecessary.
``manipulations``
    Optional ``{"<dataset token>": "<ManipulationType>", ...}``.
``directory_rules``
    Ordered list of ``{"path": ..., "label": ..., "modality": ..., "manipulation": ...,
    "source": ...}``. ``label`` and ``modality`` are each required unless
    ``default_rule`` supplies them. ``path`` is a POSIX relative directory matched
    exactly, or a ``fnmatch`` pattern, or ``"**"`` for the default rule.
``default_rule``
    A single rule applied to any file no ``directory_rules`` entry matched. Without
    one, unmatched files are still yielded — with no label and no modality — so the
    pipeline flags them instead of the adapter quietly dropping them.
``extensions``
    Optional filename-extension filter, e.g. ``[".png", ".jpg"]``. Omit to consider
    every file a candidate.

Example configuration for a directory-structured corpus::

    {
        "license": "CC-BY-4.0",
        "dataset_version": "2024-01",
        "labels": {"original": "real", "forged": "fake"},
        "manipulations": {"faceswap": "face_swap"},
        "directory_rules": [
            {"path": "image/real", "label": "original", "modality": "image"},
            {
                "path": "image/fake",
                "label": "forged",
                "modality": "image",
                "manipulation": "faceswap",
            },
            {"path": "audio/**", "label": "original", "modality": "audio"},
        ],
        "default_rule": {"label": "forged", "modality": "audio"},
        "extensions": [".wav"],
    }

Note what this does *not* contain: no sample labelled by filename, no modality derived
from ``.wav``, no fabricated ``device`` or ``duration``. Everything absent stays absent.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from datasets.adapters.base import (
    AdapterConfigurationError,
    DatasetAdapter,
    RawSample,
    load_adapter_config_file,
    validate_dataset_id,
)
from datasets.normalization import LabelMapping, ManipulationMapping, ModalityMapping
from datasets.schema import Modality

__all__ = ["DirectoryRule", "TemplateAdapter"]

#: Rule key that matches every remaining file.
CATCH_ALL_RULE = "**"

#: Rule keys an adapter may declare. Anything else is a typo and is rejected rather
#: than ignored, because a silently dropped ``manipulation`` key would look like a
#: dataset that does not disclose manipulations.
_ALLOWED_RULE_KEYS = frozenset({"path", "label", "modality", "manipulation", "source"})


@dataclass(frozen=True, slots=True)
class DirectoryRule:
    """One directory-to-attribution rule.

    Attributes:
        pattern: POSIX relative directory to match. ``"**"`` matches everything.
        label_token: Dataset label token for files here, e.g. ``"forged"``.
        modality: Canonical modality when the directory structure determines it.
        modality_token: Dataset modality token, when it must be translated.
        manipulation_token: Dataset manipulation token, or ``None`` for "not stated".
        source_token: Free-form origin marker, or ``None``.
    """

    pattern: str
    label_token: str | None = None
    modality: Modality | None = None
    modality_token: str | None = None
    manipulation_token: str | None = None
    source_token: str | None = None

    def matches(self, relative_dir: str) -> bool:
        """Whether a POSIX relative directory is governed by this rule.

        Uses :func:`fnmatch.fnmatchcase` rather than :func:`fnmatch.fnmatch`: the
        latter folds case on Windows and not on Linux, which would make discovery
        platform-dependent. Ingestion has to be identical on every machine.

        Args:
            relative_dir: Directory of the file relative to the dataset root,
                ``"."`` for the root itself.

        Returns:
            ``True`` if this rule governs that directory.
        """
        if self.pattern in {CATCH_ALL_RULE, "."}:
            return True
        if relative_dir == ".":
            return False
        return self.pattern == relative_dir or fnmatchcase(relative_dir, self.pattern)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, where: str) -> DirectoryRule:
        """Build a rule from its JSON form, rejecting unknown keys.

        Raises:
            AdapterConfigurationError: On a non-mapping, an unknown key, a missing
                ``path``, or a modality value that is not canonical.
        """
        if not isinstance(raw, Mapping):
            raise AdapterConfigurationError(f"{where} must be a mapping, got {type(raw).__name__}")
        unknown = sorted(set(raw) - _ALLOWED_RULE_KEYS)
        if unknown:
            raise AdapterConfigurationError(
                f"{where} has unknown key(s): {', '.join(unknown)}; "
                f"allowed keys are {', '.join(sorted(_ALLOWED_RULE_KEYS))}"
            )
        pattern = raw.get("path", CATCH_ALL_RULE)
        if not isinstance(pattern, str) or not pattern.strip():
            raise AdapterConfigurationError(f"{where}.path must be a non-empty string")

        modality: Modality | None = None
        raw_modality = raw.get("modality")
        if raw_modality is not None:
            if not isinstance(raw_modality, str):
                raise AdapterConfigurationError(f"{where}.modality must be a string")
            try:
                modality = Modality.parse(raw_modality)
            except ValueError as exc:
                raise AdapterConfigurationError(str(exc)) from exc

        return cls(
            pattern=pattern.strip().strip("/"),
            label_token=_optional_str(raw.get("label"), f"{where}.label"),
            modality=modality,
            modality_token=_optional_str(raw.get("modality_token"), f"{where}.modality_token"),
            manipulation_token=_optional_str(raw.get("manipulation"), f"{where}.manipulation"),
            source_token=_optional_str(raw.get("source"), f"{where}.source"),
        )


class TemplateAdapter(DatasetAdapter):
    """A fully declarative adapter driven by a JSON configuration.

    Construct it either directly::

        adapter = TemplateAdapter.from_config({"license": "CC0-1.0", ...})

    or, usually, from a file on disk via the ``--adapter-config`` CLI option.

    Determinism
    -----------
    ``discover`` walks the tree in sorted order and returns files in that order, so the
    same directory always yields the same samples in the same sequence. This is one of
    the two halves of reproducible ingestion; the other is the pipeline's canonical
    sort before writing.
    """

    name = "template"
    adapter_version = "1"

    def __init__(
        self,
        *,
        dataset_name: str,
        license_id: str,
        dataset_version: str = "unversioned",
        source_reference: str | None = None,
        access_date: str | None = None,
        labels: Mapping[str, Any] | None = None,
        modalities: Mapping[str, Any] | None = None,
        manipulations: Mapping[str, Any] | None = None,
        directory_rules: tuple[DirectoryRule, ...] = (),
        default_rule: DirectoryRule | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            license_id=license_id,
            dataset_version=dataset_version,
            source_reference=source_reference,
            access_date=access_date,
        )
        self._labels = LabelMapping(labels or {})
        self._modalities = ModalityMapping(modalities or {})
        self._manipulations = ManipulationMapping(manipulations or {})
        self._rules = directory_rules
        self._default_rule = default_rule
        self._extensions = _normalize_extensions(extensions)

    # --- vocabulary ---------------------------------------------------------

    def label_mapping(self) -> LabelMapping:
        return self._labels

    def modality_mapping(self) -> ModalityMapping:
        return self._modalities

    def manipulation_mapping(self) -> ManipulationMapping:
        return self._manipulations

    # --- discovery ----------------------------------------------------------

    def discover(self, source_dir: Path) -> Iterator[RawSample]:
        """Yield a :class:`RawSample` for every candidate file, in sorted order.

        Files under no matching rule and no default rule are still yielded, with
        ``label_token`` and modality left unset. The pipeline flags them, which is the
        behaviour we want: a file the adapter cannot account for is a question for a
        human, not a row to omit.
        """
        root = Path(source_dir)
        for path in sorted(_iter_files(root)):
            if not self._accepts(path):
                continue
            rule = self._rule_for(path, root)
            yield RawSample(
                sample_id=sample_id_for(path, root),
                path=path,
                label_token=rule.label_token if rule else None,
                modality=rule.modality if rule else None,
                modality_token=rule.modality_token if rule else None,
                manipulation_token=rule.manipulation_token if rule else None,
                source_token=rule.source_token if rule else None,
            )

    def _accepts(self, path: Path) -> bool:
        """Whether a path is a candidate sample given the extension filter."""
        if self._extensions is None:
            return True
        return path.suffix.lower() in self._extensions

    def _rule_for(self, path: Path, root: Path) -> DirectoryRule | None:
        """First matching rule for ``path``, else the default rule, else ``None``.

        Rules are matched against the directory *relative to the dataset root*, because
        a configuration states dataset layout, not where the dataset happens to be
        unpacked on this machine.
        """
        relative_dir = path.relative_to(root).parent.as_posix()
        if relative_dir == ".":
            relative_dir = ""
        for rule in self._rules:
            if rule.matches(relative_dir):
                return rule
        return self._default_rule

    # --- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        dataset_id: str = "",
    ) -> TemplateAdapter:
        """Build an adapter from a configuration mapping.

        Args:
            config: The mapping described in this module's docstring.
            dataset_id: Dataset identifier from the command line, used for
                ``dataset_name`` when the config does not state one.

        Returns:
            A configured adapter.

        Raises:
            AdapterConfigurationError: If a required key is missing, has the wrong
                type, or a declared vocabulary is malformed.
        """
        if not isinstance(config, Mapping):
            raise AdapterConfigurationError(
                f"adapter config must be a mapping, got {type(config).__name__}"
            )

        raw_rules = config.get("directory_rules") or []
        if not isinstance(raw_rules, list):
            raise AdapterConfigurationError("directory_rules must be a list")
        rules = tuple(
            DirectoryRule.from_mapping(rule, where=f"directory_rules[{index}]")
            for index, rule in enumerate(raw_rules)
        )

        raw_default = config.get("default_rule")
        default_rule = (
            DirectoryRule.from_mapping(raw_default, where="default_rule")
            if isinstance(raw_default, Mapping)
            else None
        )

        license_id = _text(config, "license")
        if not license_id:
            raise AdapterConfigurationError(
                "adapter config must declare 'license'. Licensing governs whether the data "
                "may be used at all, so ingestion will not assume one."
            )

        return cls(
            dataset_name=_text(config, "dataset_name") or dataset_id or "unnamed-dataset",
            license_id=license_id,
            dataset_version=_text(config, "dataset_version") or "unversioned",
            source_reference=_text(config, "source_reference") or None,
            access_date=_text(config, "access_date") or None,
            labels=_sub_mapping(config, "labels"),
            modalities=_sub_mapping(config, "modalities"),
            manipulations=_sub_mapping(config, "manipulations"),
            directory_rules=rules,
            default_rule=default_rule,
            extensions=_extension_tuple(config.get("extensions")),
        )

    @classmethod
    def from_config_file(cls, path: Path | str, *, dataset_id: str = "") -> TemplateAdapter:
        """Build an adapter from a JSON configuration file.

        Reading is delegated to
        :func:`~datasets.adapters.base.load_adapter_config_file`, so the JSON path is
        shared with adapters that prefer YAML and the error messages match.

        Raises:
            AdapterConfigurationError: If the file is missing, is not JSON, does not
                hold an object, or names a ``dataset_id`` that is not a safe path
                segment.
        """
        return cls.from_config(
            load_adapter_config_file(path),
            dataset_id=validate_dataset_id(dataset_id) if dataset_id else "",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_id_for(path: Path, source_root: Path) -> str:
    """Derive a stable identifier from a file's position in the dataset.

    Uses the POSIX relative path without its extension, so the id is independent of
    the machine, the mount point, and the order in which a directory happens to be
    walked. Adapters for a dataset with real upstream identifiers should prefer those
    and use this only as a fallback.
    """
    relative = Path(path).resolve().relative_to(Path(source_root).resolve())
    return relative.with_suffix("").as_posix()


def _iter_files(root: Path) -> Iterator[Path]:
    """Yield every non-hidden file under ``root``, in sorted order.

    Hidden files and directories are skipped: they hold editor state and partial
    downloads, not samples. Sorting at every level is what makes discovery
    order-independent of the filesystem.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        base = Path(dirpath)
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            yield base / filename


def _normalize_extensions(extensions: tuple[str, ...] | None) -> frozenset[str] | None:
    """Lower-case and dot-prefix extension filters.

    ``None`` and an empty sequence both mean "accept every file", because an empty
    filter list is far more likely to be an oversight than an intent to select nothing.
    """
    if not extensions:
        return None
    normalized = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in extensions
        if ext.strip()
    }
    return frozenset(normalized) or None


def _extension_tuple(raw: Any) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise AdapterConfigurationError("extensions must be a list of strings")
    return tuple(raw)


def _sub_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    """Read an optional vocabulary mapping out of a config mapping."""
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AdapterConfigurationError(f"{key} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _text(config: Mapping[str, Any], key: str) -> str:
    """Read an optional string out of a config mapping, rejecting wrong types."""
    value = config.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise AdapterConfigurationError(f"{key} must be a string, got {type(value).__name__}")
    return value.strip()


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdapterConfigurationError(
            f"{field_name} must be a string, got {type(value).__name__}"
        )
    return value.strip() or None
