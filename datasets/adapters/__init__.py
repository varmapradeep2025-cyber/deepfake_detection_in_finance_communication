"""Dataset adapters: the seam between a foreign dataset and our manifest format.

Every dataset gets one adapter. An adapter discovers files and declares how its
dataset's own vocabulary maps onto :mod:`datasets.schema`; it never decides what a
sample means and never fills in a value the dataset did not provide.

Registered adapters
-------------------
``template``
    :class:`~datasets.adapters.template.TemplateAdapter`, a configuration-driven
    reference implementation, and not an adapter for any real dataset.
``faceforensics_plus_plus``
    :class:`~datasets.adapters.faceforensics_plus_plus.FaceForensicsPlusPlusAdapter`,
    the first adapter for a named research dataset. It reads its layout, vocabulary, and
    terms from ``configs/adapters/faceforensics_plus_plus.yaml``, and **refuses to ingest
    while that file still carries an unverified licence marker**. Layout inspection needs
    no licence and stays available, so an installation can be described before anything
    asserts it was used.

The distinction matters: an adapter existing is not the same as a dataset being usable.
Registering ``faceforensics_plus_plus`` records that this project's layout assumptions
about FaceForensics++ are written down and reviewable. It does not record that the
dataset has been acquired, verified, or used — that is
``configs/datasets.yaml``, and it still says none of those things.

What an adapter must never do
-----------------------------
Guess a label, infer modality from a filename, invent metadata, or modify the source
media. See :mod:`datasets.adapters.base` for the contract and
:mod:`datasets.normalization` for the vocabulary rules.
"""

from __future__ import annotations

from datasets.adapters.base import (
    AdapterConfigurationError,
    AdapterError,
    DatasetAdapter,
    RawSample,
    load_adapter_config_file,
    validate_dataset_id,
)
from datasets.adapters.faceforensics_plus_plus import (
    DECLARED_FACEFORENSICS_MANIPULATIONS,
    DirectoryDeclaration,
    FaceForensicsPlusPlusAdapter,
    LayoutReport,
    LayoutResolutionError,
)
from datasets.adapters.template import DirectoryRule, TemplateAdapter
from datasets.normalization import (
    LabelMapping,
    ManipulationMapping,
    ModalityMapping,
    NoLabelMappingError,
    NormalizationError,
    UnknownLabelError,
    UnknownManipulationError,
    UnknownModalityError,
)

__all__ = [
    "ADAPTER_REGISTRY",
    "DECLARED_FACEFORENSICS_MANIPULATIONS",
    "AdapterConfigurationError",
    "AdapterError",
    "DatasetAdapter",
    "DirectoryDeclaration",
    "DirectoryRule",
    "FaceForensicsPlusPlusAdapter",
    "LabelMapping",
    "LayoutReport",
    "LayoutResolutionError",
    "ManipulationMapping",
    "ModalityMapping",
    "NoLabelMappingError",
    "NormalizationError",
    "RawSample",
    "TemplateAdapter",
    "UnknownLabelError",
    "UnknownManipulationError",
    "UnknownModalityError",
    "adapter_names",
    "load_adapter_config_file",
    "register_adapter",
    "resolve_adapter",
    "validate_dataset_id",
]

#: Adapter name -> adapter class. Keyed by the value the CLI's ``--adapter`` accepts.
#:
#: Types are ``type[DatasetAdapter]``; the registry intentionally holds classes rather
#: than instances because construction needs per-run arguments (licence, version,
#: source) that are supplied on the command line.
ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    FaceForensicsPlusPlusAdapter.name: FaceForensicsPlusPlusAdapter,
    TemplateAdapter.name: TemplateAdapter,
}


def register_adapter(adapter_class: type[DatasetAdapter], *, name: str | None = None) -> None:
    """Register an adapter class under ``name`` (defaults to its ``name`` attribute).

    Args:
        adapter_class: A concrete :class:`DatasetAdapter` subclass.
        name: Registry key. Must not shadow an existing entry.

    Raises:
        AdapterConfigurationError: If ``name`` is empty or already registered.
    """
    key = name or adapter_class.name
    if not key.strip():
        raise AdapterConfigurationError("adapter name must be a non-empty string")
    if key in ADAPTER_REGISTRY:
        raise AdapterConfigurationError(f"adapter {key!r} is already registered")
    ADAPTER_REGISTRY[key] = adapter_class


def adapter_names() -> tuple[str, ...]:
    """Registered adapter names, sorted. Used by ``--list-adapters``."""
    return tuple(sorted(ADAPTER_REGISTRY))


def resolve_adapter(name: str) -> type[DatasetAdapter]:
    """Look up an adapter class by registry name.

    Args:
        name: Value passed to ``--adapter``.

    Returns:
        The registered adapter class.

    Raises:
        AdapterConfigurationError: If the name is not registered. The message lists
            the available names, because the remedy is always one of them.
    """
    if name not in ADAPTER_REGISTRY:
        available = ", ".join(adapter_names()) or "(none registered)"
        raise AdapterConfigurationError(f"unknown adapter {name!r}; available: {available}")
    return ADAPTER_REGISTRY[name]
