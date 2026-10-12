"""The dataset adapter contract.

What an adapter is for
----------------------
Datasets disagree about everything: how files are laid out, what a label is called,
where the manipulation type is recorded, whether authenticity is a column or a folder
name. Rather than teach the ingestion pipeline about each dataset, every dataset gets
one small class that answers the same six questions:

===========================  ==========================================================
Question                     Answered by
===========================  ==========================================================
Which files are samples?      :meth:`DatasetAdapter.discover`
What is each file's modality? ``RawSample.modality`` / :meth:`~DatasetAdapter.modality_mapping`
Is it real or fake?           ``RawSample.label_token`` / :meth:`~DatasetAdapter.label_mapping`
What was manipulated?        ``RawSample.manipulation_token``
Where did it come from?       ``RawSample.source_token``
What else does the dataset    ``RawSample.metadata``
say about it?
===========================  ==========================================================

The adapter's second job, and the one that matters more, is to *declare* how its
dataset's own words map onto :mod:`datasets.schema`. Those declarations live in
:mod:`datasets.normalization` tables, are introspectable, and are the only sanctioned
route from a foreign vocabulary to a canonical one.

What an adapter must never do
-----------------------------
* guess a label, a modality, or a manipulation type,
* infer modality from a filename extension,
* invent a value to fill a schema field,
* modify, move, or copy the source media.

Those constraints are not stylistic. An ingestion layer that guesses produces a
manifest that looks complete and is quietly wrong, which is far more damaging than one
that reports what it does not know.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from datasets.normalization import LabelMapping, ManipulationMapping, ModalityMapping
from datasets.schema import Modality
from datasets.yaml_io import StrictYamlError, load_yaml_mapping

__all__ = [
    "AdapterConfigurationError",
    "AdapterError",
    "DatasetAdapter",
    "RawSample",
    "load_adapter_config_file",
    "validate_dataset_id",
]


class AdapterError(RuntimeError):
    """Base class for adapter problems."""


class AdapterConfigurationError(AdapterError):
    """Raised when an adapter is misconfigured.

    Covers a missing licence, a dataset id that is not a safe path segment, and
    declared vocabularies that contradict themselves.
    """


# Declared once and shared: a mapping is read-only, and "declared nothing" must read
# the same to every caller so that an absent declaration is never mistaken for a bug.
_EMPTY_MODALITY_MAPPING = ModalityMapping({})
_EMPTY_MANIPULATION_MAPPING = ManipulationMapping({})


@dataclass(frozen=True, slots=True)
class RawSample:
    """One discovered file, before hashing and canonicalization.

    A ``RawSample`` records what the *dataset* says, using the dataset's own words.
    Nothing here is canonicalized: ``label_token`` is whatever the dataset called it,
    and it is :mod:`datasets.normalization` that translates. Fields the dataset does
    not supply stay ``None`` — an adapter that fills them in is fabricating evidence.

    Attributes:
        sample_id: Stable identifier, unique within the dataset. A positional index is
            not acceptable: it shifts whenever a dataset is repackaged, which
            invalidates every split and every published number referencing it.
        path: Location of the media file on disk.
        label_token: The dataset's own real/fake wording, or ``None`` if it gave none.
        modality: Canonical modality, when the adapter determined it from the
            dataset's authoritative structure. ``None`` means "not determined".
        modality_token: The dataset's own modality wording, when it used words rather
            than a value this project already understands.
        manipulation_token: The dataset's manipulation wording, or ``None``.
        source_token: Free-form origin marker, e.g. ``"original"``,
            ``"platform_download"``. ``None`` when the dataset does not say.
        metadata: Optional extras the dataset provided. Never invented here.
    """

    sample_id: str
    path: Path
    label_token: str | None = None
    modality: Modality | None = None
    modality_token: str | None = None
    manipulation_token: str | None = None
    source_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise AdapterConfigurationError(
                f"sample_id must be a non-empty string, got {self.sample_id!r}"
            )
        object.__setattr__(self, "sample_id", self.sample_id.strip())
        object.__setattr__(self, "path", Path(self.path))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        elif not isinstance(self.metadata, Mapping):
            raise AdapterConfigurationError(
                f"metadata must be a mapping, got {type(self.metadata).__name__}"
            )
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))


def validate_dataset_id(dataset_id: str) -> str:
    """Validate a dataset identifier used as a directory and file name.

    A dataset id becomes a directory name (``datasets/raw/<dataset_id>/``) and a file
    stem (``datasets/manifests/<dataset_id>.jsonl``), so it is restricted to a single
    safe path segment. This is a guard against path traversal, not a style rule.

    Args:
        dataset_id: Candidate identifier.

    Returns:
        The trimmed identifier.

    Raises:
        AdapterConfigurationError: If it is empty, contains a path separator, a drive
            letter, or is ``.``/``..``.
    """
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise AdapterConfigurationError(f"dataset_id must be a non-empty string: {dataset_id!r}")
    trimmed = dataset_id.strip()
    if trimmed in {".", ".."} or "/" in trimmed or "\\" in trimmed or ":" in trimmed:
        raise AdapterConfigurationError(
            f"dataset_id must be a single path segment without separators or ':'; "
            f"got {dataset_id!r}"
        )
    return trimmed


def load_adapter_config_file(path: Path | str) -> dict[str, Any]:
    """Read an adapter configuration file as a mapping.

    JSON and YAML are both accepted, chosen by suffix, because the choice is a property of
    the file rather than of the caller: ``configs/adapters/*.json`` predates this and is
    still supported, while ``configs/adapters/*.yaml`` reads better for a configuration
    that carries comments explaining what was verified.

    Whichever format is used, the result goes through the same strict reader as the
    dataset registry: a duplicate key is an error rather than a silently winning last
    value, and an empty or non-mapping document is an error rather than ``None``.

    Args:
        path: Configuration file.

    Returns:
        The configuration mapping. Schema validation belongs to the adapter.

    Raises:
        AdapterConfigurationError: If the file is missing, unreadable, empty, malformed,
            has a duplicate key, does not hold a mapping, or has an unknown suffix.
    """
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AdapterConfigurationError(f"adapter config not found: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise AdapterConfigurationError(f"malformed JSON at {config_path}: {exc.msg}") from exc
        if not isinstance(loaded, dict):
            raise AdapterConfigurationError(
                f"adapter config must be a JSON object, got {type(loaded).__name__}: {config_path}"
            )
        return loaded
    if suffix in {".yaml", ".yml"}:
        try:
            return load_yaml_mapping(config_path)
        except StrictYamlError as exc:
            raise AdapterConfigurationError(f"{exc}") from exc
    raise AdapterConfigurationError(
        f"unsupported adapter config format {suffix or '(no suffix)'!r}: {config_path}. "
        "Use .json, .yaml, or .yml."
    )


class DatasetAdapter(ABC):
    """Abstract base for every dataset adapter.

    An adapter is declarative about vocabulary and procedural about discovery. It
    reports what it found; it never decides what a sample *means*. Meaning arrives
    through the declared mappings below, which the ingestion pipeline reads.

    Subclasses must implement :meth:`discover` and normally :meth:`label_mapping`.
    :meth:`modality_mapping` may legitimately be empty for a dataset whose structure
    already determines modality, or whose modality is simply absent — the pipeline
    flags those samples rather than guessing.

    Attributes:
        name: Registry key used by ``--adapter``.
        adapter_version: Adapter revision, recorded in provenance. Bump it whenever
            discovery or mapping behaviour changes, because the manifest hash changes
            with it and a reader needs to know why.
    """

    name: ClassVar[str] = "unnamed"
    adapter_version: ClassVar[str] = "1"

    def __init__(
        self,
        *,
        dataset_name: str,
        license_id: str,
        dataset_version: str = "unversioned",
        source_reference: str | None = None,
        access_date: str | None = None,
    ) -> None:
        """
        Args:
            dataset_name: Human-readable dataset name for provenance.
            license_id: SPDX identifier or dataset-specific licence string. Required:
                licensing governs whether the data may be used at all, so an adapter
                cannot start without one.
            dataset_version: Version or release identifier of the dataset. Datasets
                change; results pinned to "whatever was on the site" are not
                reproducible.
            source_reference: Where the dataset came from. ``None`` when unknown.
                Never fabricate a URL — an invented link is worse than an honest gap.
            access_date: When the data was accessed, RFC 3339 or ``YYYY-MM-DD``.
                ``None`` when not recorded.

        Raises:
            AdapterConfigurationError: If a required identifier is missing.
        """
        self.dataset_name = _require_text(dataset_name, "dataset_name")
        self.license_id = _require_text(license_id, "license_id")
        self.dataset_version = _require_text(dataset_version, "dataset_version")
        self.source_reference = _optional_text(source_reference, "source_reference")
        self.access_date = _optional_text(access_date, "access_date")

    # --- construction hooks -------------------------------------------------

    @classmethod
    def from_dataset_id(cls, dataset_id: str) -> DatasetAdapter | None:
        """Construct this adapter from its hardcoded vocabulary, or ``None``.

        Adapters that declare their dataset's licence and vocabulary in code return an
        instance here. Adapters that need per-dataset configuration return ``None``, and
        the CLI then asks for ``--adapter-config`` instead.

        Returning ``None`` rather than raising lets one mechanism serve both adapter
        styles without the CLI having to know which is which.
        """
        return None

    @classmethod
    def from_config_file(cls, path: Path, *, dataset_id: str = "") -> DatasetAdapter | None:
        """Construct this adapter from a JSON configuration file, or ``None``.

        ``None`` means "this adapter does not take a configuration file", which is
        distinct from a configuration file that failed to load: a malformed file is an
        error, an unsupported adapter is a usage mistake, and both deserve their own
        message.
        """
        return None

    # --- declared vocabulary ------------------------------------------------

    @abstractmethod
    def label_mapping(self) -> LabelMapping:
        """Return this dataset's explicit real/fake vocabulary.

        The returned table is a claim about one specific dataset. It is never
        defaulted, never inferred from the label strings themselves, and never shared
        between datasets: ``"original"`` means one thing in one corpus and nothing in
        particular in another.
        """

    def modality_mapping(self) -> ModalityMapping:
        """Return this dataset's explicit modality vocabulary.

        May be empty when modality comes from the dataset's structure instead, or when
        the dataset provides no modality information at all. Both cases are honest;
        guessing from a file extension is not.
        """
        return _EMPTY_MODALITY_MAPPING

    def manipulation_mapping(self) -> ManipulationMapping:
        """Return this dataset's explicit manipulation vocabulary.

        Empty by default, which means manipulation types resolve to ``unknown`` rather
        than being invented. Datasets that describe manipulations map them here.
        """
        return _EMPTY_MANIPULATION_MAPPING

    # --- discovery ----------------------------------------------------------

    @abstractmethod
    def discover(self, source_dir: Path) -> Iterator[RawSample]:
        """Yield one :class:`RawSample` per media file under ``source_dir``.

        Implementations must:

        * read only — never copy, move, rename, or rewrite source media;
        * be deterministic, yielding the same files in the same order for the same
          directory, so an identical ingest reproduces an identical manifest;
        * yield stable ``sample_id`` values, ideally derived from the dataset's own
          identifiers or from the relative path;
        * yield *every* candidate file, including ones whose label or modality cannot
          be determined, so the pipeline can flag them. Returning a silently shortened
          list would hide exactly the problems worth reporting.

        Args:
            source_dir: Root of the dataset as it exists on disk.

        Yields:
            Discovered samples, using the dataset's own vocabulary.
        """

    # --- self-checks --------------------------------------------------------

    def validate_configuration(self) -> None:
        """Check the adapter's own declarations before any file is read.

        Called by the ingestion pipeline on every run, including dry runs, so that a
        misconfigured adapter fails immediately and unambiguously instead of producing
        a manifest full of flagged samples.

        Raises:
            AdapterConfigurationError: If a required identifier is missing or no label
                mapping was declared.
        """
        _require_text(self.license_id, "license_id")
        _require_text(self.dataset_name, "dataset_name")
        _require_text(self.dataset_version, "dataset_version")

        labels = self.label_mapping()
        if not labels:
            raise AdapterConfigurationError(
                f"adapter {self.name!r} declares no label mapping. Every adapter must state "
                "explicitly which of its dataset's labels mean 'real' and which mean 'fake'. "
                "Ingestion refuses to infer a label from its wording."
            )

    def label_coverage(self) -> tuple[str, ...]:
        """Canonical label values this adapter can produce, sorted.

        Reported by the dry run. A single-class result (``("real",)``) is legitimate
        for a genuine-media-only corpus but useless as a detection benchmark, so it is
        surfaced rather than blocked.
        """
        return self.label_mapping().targets

    def describe(self) -> dict[str, Any]:
        """Machine-readable description of the adapter and its declared vocabulary.

        Included in the dry-run report so that a reader can see exactly which mappings
        were in force, without opening the adapter's source.
        """
        return {
            "adapter": self.name,
            "adapter_version": self.adapter_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "license": self.license_id,
            "source_reference": self.source_reference,
            "access_date": self.access_date,
            "label_mappings": self.label_mapping().as_dict(),
            "label_coverage": list(self.label_coverage()),
            "modality_mappings": self.modality_mapping().as_dict(),
            "manipulation_mappings": self.manipulation_mapping().as_dict(),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, version={self.adapter_version!r})"


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterConfigurationError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


def _optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AdapterConfigurationError(
            f"{field_name} must be a string or None, got {type(value).__name__}"
        )
    trimmed = value.strip()
    return trimmed or None
