"""One strict YAML reader, shared by every configuration file in this project.

Why this module exists
----------------------
PyYAML's ``safe_load`` keeps the **last** value for a repeated key and says nothing at
all. For a configuration that a human reviews in a pull request, that is silent data
loss: a duplicated ``license_status`` that quietly wins, or a second ``labels`` block that
overrides the first, produces a file that parses cleanly and means something other than
what it says.

Every versioned YAML file in this project — the dataset registry
(``configs/datasets.yaml``) and the dataset adapter configurations
(``configs/adapters/*.yaml``) — is therefore read through here, so there is exactly one
answer to "what happens when a key appears twice".

Two rules
---------
1. **A duplicate key is an error**, with both the key and its line reported.
2. **An empty document is distinguishable from a mapping.** ``safe_load`` returns
   ``None`` for an empty file, which is easy to mistake for "parsed fine"; callers get an
   explicit error naming the file instead.

Nothing here validates the *meaning* of a document. That belongs to the module that owns
the schema: :mod:`datasets.registry` validates the registry, and
:func:`datasets.adapters.base.load_adapter_config_file` validates an adapter config.
This module only decides how bytes become a mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "StrictYamlError",
    "load_yaml_mapping",
    "strict_yaml_load",
]


class StrictYamlError(ValueError):
    """Raised when a YAML document is malformed, duplicated, or not a mapping."""


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    """Build a mapping, refusing a repeated key.

    Refusing is the whole point: PyYAML's default behaviour is to overwrite, which is how
    a file ends up disagreeing with itself without anything reporting it.
    """
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=False)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                context="while constructing a mapping",
                context_mark=node.start_mark,
                problem=f"duplicate key {key!r}",
                problem_mark=key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=False)
    return mapping


class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that additionally rejects duplicate mapping keys."""


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def strict_yaml_load(text: str, *, where: str = "<string>") -> Any:
    """Parse YAML text, rejecting duplicate keys.

    Args:
        text: The document.
        where: Human-readable source name used in error messages.

    Returns:
        The parsed document. Any type — the caller owns schema validation.

    Raises:
        StrictYamlError: If the text is malformed or contains a duplicate key.
    """
    try:
        return yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise StrictYamlError(f"could not parse YAML at {where}: {exc}") from exc


def load_yaml_mapping(path: Path | str) -> dict[str, Any]:
    """Read a YAML file that must hold a mapping.

    Args:
        path: File to read.

    Returns:
        The document as a mapping.

    Raises:
        StrictYamlError: If the file is missing, unreadable, empty, malformed, has a
            duplicate key, or does not hold a mapping.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise StrictYamlError(f"file not found: {file_path}")
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StrictYamlError(f"could not read {file_path}: {exc}") from exc
    loaded = strict_yaml_load(text, where=str(file_path))
    if loaded is None:
        raise StrictYamlError(f"file is empty: {file_path}")
    if not isinstance(loaded, dict):
        raise StrictYamlError(f"root must be a mapping, got {type(loaded).__name__}: {file_path}")
    return loaded
