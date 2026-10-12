"""Dataset registry: what we intend to evaluate, and on what terms.

What this module is for
-----------------------
:mod:`datasets.ingest` answers "what is in each file of a dataset that is already on
this machine". The registry answers the question that has to be answered *before* that:
**which datasets and public-intelligence sources do we intend to work with, in what role,
and what is still unknown about each one?**

So the registry is a planning and configuration layer, and the file it reads
(``configs/datasets.yaml``) is versioned while the data it describes is not. Nothing here
downloads, collects, decodes, or transforms anything. ``python -m datasets.registry
--validate`` reads one YAML file and reports what it does and does not claim.

The three commitments that shape the design
-------------------------------------------
**Absence is data.** Access route, licence, version, size, and label wording are the
fields most likely to be unknown at this stage, so unknown values are first-class
markers (:data:`UNVERIFIED_MARKERS`) rather than empty strings, ``None``, or a plausible
guess. ``not_verified`` is a materially different statement from ``unknown``: the first
says "we looked and did not find it confirmed", the second says "we have not looked".

**Intent grants nothing.** ``train_allowed`` and friends state what an entry *would* be
used for once it is verified and acquired. They do not authorize anything. Acquisition is
gated on :class:`RegistryStatus`, and ``status: approved`` is refused while the licence
is still unverified.

**A closed vocabulary, one authority.** Every enumeration in this module — modality,
research role, status, source type, access type, licence status, label scheme,
generalization group — is defined here and nowhere else. An unrecognized value is an
error naming the accepted ones, never a value quietly stored as written. That is what
stops a typo (``generalisation_groups``) from disabling a safety rule while appearing to
work.

Extending a vocabulary
----------------------
Adding a member to an enum or to :data:`GENERALIZATION_GROUPS` is the sanctioned way to
extend one, and it costs every future reader of every published result: a new modality
has to be handled by the whole pipeline. New members need a definition and a reason
here, in the code, before they are used in YAML.

Usage::

    python -m datasets.registry --validate      # check configs/datasets.yaml
    python -m datasets.registry --list          # one line per registered source
    python -m datasets.registry --show tranco   # one entry in full
    python -m datasets.registry --validate --json

Nothing in this module performs I/O beyond reading the registry file itself.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from datasets.yaml_io import StrictYamlError, strict_yaml_load

__all__ = [
    "DEFAULT_REGISTRY_RELPATH",
    "GENERALIZATION_GROUPS",
    "REGISTRY_SCHEMA_VERSION",
    "UNVERIFIED_MARKERS",
    "AccessType",
    "AcquisitionRecord",
    "DatasetEntry",
    "DatasetRegistry",
    "GeneralizationPair",
    "GeneralizationStatus",
    "LabelSchema",
    "LabelScheme",
    "LicenseStatus",
    "RegistryError",
    "RegistryModality",
    "RegistryNotFoundError",
    "RegistryReport",
    "RegistryStatus",
    "RegistryValidationError",
    "ResearchRole",
    "SourceType",
    "default_registry_path",
    "is_unverified",
    "load_registry",
    "main",
    "validate_registry",
]

#: Registry format version this module understands. A file declaring anything else is
#: rejected rather than parsed on a guess at what the newer fields mean.
REGISTRY_SCHEMA_VERSION = 1

#: Default location of the registry file, relative to the repository root.
DEFAULT_REGISTRY_RELPATH = Path("configs") / "datasets.yaml"

#: Dotted key in ``configs/project.yaml`` naming the registry file.
_REGISTRY_CONFIG_KEY = "dataset_registry.file"

#: Accepted stand-ins for "we do not know", "we have not checked", and "the dataset did
#: not say". Allowed wherever a value is genuinely unknown; rejected wherever a value is
#: required to be real, such as a confirmed licence.
#:
#: ``not_applicable`` is a claim that the question does not arise — a URL feed has no
#: manipulation types — which is different from, and stronger than, not knowing.
UNVERIFIED_MARKERS = frozenset(
    {
        "unknown",
        "pending",
        "pending_verification",
        "not_verified",
        "not_applicable",
    }
)

#: Generalization groups the project intends to measure. Closed like the enums below: a
#: new group means a new experiment family, so it is added here with a definition rather
#: than invented in YAML.
GENERALIZATION_GROUPS = frozenset({"image_deepfake", "audio_deepfake", "video_deepfake"})

#: ``access_date`` shape. Deliberately a shape check and not a date parse: an unparsed
#: string keeps this module free of clock and timezone handling, which have no place in
#: a file that describes data we do not yet have.
_ACCESS_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ].*)?$")

#: Roles that make an entry a source of evidence about URLs rather than ML training data.
_INTELLIGENCE_ROLES = frozenset({"threat_intelligence", "external_intelligence"})

#: Label schemes an intelligence source may declare. Class labels would invite turning a
#: feed match into a training target, which is exactly what must not happen.
_INTELLIGENCE_LABEL_SCHEMES = frozenset({"evidence_state", "unlabeled", "unknown"})


class RegistryError(Exception):
    """Base class for registry problems."""


class RegistryNotFoundError(RegistryError):
    """Raised when the registry file cannot be located."""


class RegistryValidationError(RegistryError):
    """Raised when the registry is structurally or semantically invalid.

    Collects every problem found rather than only the first, so a malformed registry can
    be fixed in one pass. ``str(error)`` includes the full list, which means a traceback
    alone is enough to diagnose the failure.
    """

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        if self.errors:
            message = f"{message}: " + "; ".join(self.errors)
        super().__init__(message)


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

_V = TypeVar("_V", bound="_Vocabulary")


class _Vocabulary(str, Enum):
    """Base class for the registry's closed vocabularies.

    ``str`` mixin so a member compares and serializes exactly like the plain YAML string
    it came from, which keeps ``configs/datasets.yaml`` readable without importing this
    package.
    """

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Every accepted value, in declaration order."""
        return tuple(member.value for member in cls)

    @classmethod
    def parse(cls: type[_V], value: Any) -> _V:
        """Coerce ``value`` to a member, raising :class:`RegistryValidationError`.

        Matching is case-insensitive and treats ``-`` and space as ``_``, so ``"Image"``
        and ``"binary-real-fake"`` resolve. An unknown value is an error naming the
        accepted ones rather than a value stored as written.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
            for member in cls.__members__.values():
                if member.value == candidate:
                    return member
        raise RegistryValidationError(
            f"invalid {cls.__name__} value {value!r}; expected one of {', '.join(cls.values())}"
        )


class RegistryModality(_Vocabulary):
    """What kind of evidence the source holds.

    ``IMAGE``, ``AUDIO``, and ``VIDEO`` are media; ``QR`` and ``URL`` are the phishing
    artefacts a QR or URL detector consumes; ``MULTIMODAL`` marks a corpus whose audio
    and video components are evaluated separately; ``THREAT_INTELLIGENCE`` marks an
    evidence source, which is never a training corpus.
    """

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    QR = "qr"
    URL = "url"
    MULTIMODAL = "multimodal"
    THREAT_INTELLIGENCE = "threat_intelligence"


class ResearchRole(_Vocabulary):
    """How an entry is intended to be used in an experiment.

    ``SOURCE_DOMAIN`` / ``TARGET_DOMAIN`` are what make a generalization pair meaningful:
    the source is trained on, the target is held out and never seen. They are roles
    rather than assumptions, so an entry cannot quietly become a target domain by being
    mentioned in a generalization block.
    """

    TRAINING = "training"
    VALIDATION = "validation"
    IN_DOMAIN_TEST = "in_domain_test"
    UNSEEN_TEST = "unseen_test"
    EXTERNAL_BENCHMARK = "external_benchmark"
    THREAT_INTELLIGENCE = "threat_intelligence"
    EXTERNAL_INTELLIGENCE = "external_intelligence"
    REFERENCE = "reference"
    OPTIONAL = "optional"
    SOURCE_DOMAIN = "source_domain"
    TARGET_DOMAIN = "target_domain"


class RegistryStatus(_Vocabulary):
    """Where an entry stands in the approval process.

    ``CANDIDATE`` and ``PENDING_VERIFICATION`` mean nothing may be acquired yet.
    ``APPROVED`` additionally requires a verified licence. ``AVAILABLE_LOCALLY`` is the
    only status that asserts data is on this machine, and it requires a completed
    acquisition record. ``BLOCKED`` and ``DEPRECATED`` are for entries we must not or no
    longer should use.
    """

    CANDIDATE = "candidate"
    APPROVED = "approved"
    PENDING_VERIFICATION = "pending_verification"
    AVAILABLE_LOCALLY = "available_locally"
    BLOCKED = "blocked"
    DEPRECATED = "deprecated"


class SourceType(_Vocabulary):
    """What kind of thing the entry is.

    The distinction that matters is between a corpus someone published for research and
    a live feed that keeps changing: a benchmark version can be pinned, a feed cannot.
    """

    RESEARCH_DATASET = "research_dataset"
    BENCHMARK_DATASET = "benchmark_dataset"
    INTELLIGENCE_FEED = "intelligence_feed"
    REFERENCE_LIST = "reference_list"


class AccessType(_Vocabulary):
    """How access is obtained. ``UNKNOWN`` is the honest default here."""

    UNKNOWN = "unknown"
    PUBLIC_DOWNLOAD = "public_download"
    REGISTRATION_REQUIRED = "registration_required"
    REQUEST_REQUIRED = "request_required"
    API = "api"
    FEED = "feed"
    MANUAL = "manual"


class LicenseStatus(_Vocabulary):
    """Whether the terms of use have been established.

    ``VERIFIED`` is the only value that authorizes acquisition, and it must come from
    reading the licence rather than from remembering it.
    """

    UNKNOWN = "unknown"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    RESTRICTED = "restricted"
    PROHIBITED_FOR_USE = "prohibited_for_use"


class LabelScheme(_Vocabulary):
    """The kind of labelling an entry states.

    ``BINARY_REAL_FAKE`` is the deepfake convention. ``BINARY_BENIGN_MALICIOUS`` is the
    phishing convention. ``EVIDENCE_STATE`` records observations about a URL rather than
    a class per sample, and ``UNLABELED`` says the source carries no labels at all — a
    popularity ranking, for instance. These are not interchangeable: an unlabelled source
    cannot be scored, and an evidence state is not a class label.
    """

    BINARY_REAL_FAKE = "binary_real_fake"
    BINARY_BENIGN_MALICIOUS = "binary_benign_malicious"
    EVIDENCE_STATE = "evidence_state"
    UNLABELED = "unlabeled"
    UNKNOWN = "unknown"


class GeneralizationStatus(_Vocabulary):
    """State of a planned cross-dataset experiment."""

    PLANNED = "planned"
    BLOCKED = "blocked"
    EXECUTED = "executed"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelSchema:
    """How an entry states its labels, at the level of the scheme only.

    Attributes:
        scheme: One of :class:`LabelScheme`.
        values: The canonical values for the scheme. Empty for an unlabelled source.
        description: What the source actually says, or why it is not yet known. Free
            text: the point is that the reader sees the caveat, not that it is parsed.
    """

    scheme: LabelScheme
    values: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheme", LabelScheme.parse(self.scheme))
        object.__setattr__(self, "values", tuple(sorted(dict.fromkeys(self.values))))
        object.__setattr__(self, "description", str(self.description).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme.value,
            "values": list(self.values),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    """What it would take to have this data locally, and whether that has happened.

    Attributes:
        acquired: Whether the data is on this machine. The only field in the registry
            allowed to assert possession.
        access_date: When it was accessed, ``YYYY-MM-DD`` or RFC 3339. ``None`` when
            nothing has been accessed.
        version: The exact release acquired. ``None`` until known.
        local_path: Where it lives, relative to the repository root. ``None`` until
            acquired.
    """

    acquired: bool = False
    access_date: str | None = None
    version: str | None = None
    local_path: str | None = None

    def is_complete(self) -> bool:
        """Whether a *claimed* acquisition is fully recorded.

        Complete means acquired, dated, versioned, and located. A half-filled record is
        a gap, not an acquisition: a hash computed against the wrong release is worse
        than no hash.
        """
        return bool(self.acquired and self.access_date and self.version and self.local_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "acquired": self.acquired,
            "access_date": self.access_date,
            "version": self.version,
            "local_path": self.local_path,
        }


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """One dataset or public-intelligence source, and what is known about it.

    Attributes:
        dataset_id: Stable identifier, used in manifests, splits, and run records.
        name: Human-readable name.
        modality: What kind of evidence the source holds. May be several values; use
            :attr:`RegistryModality.MULTIMODAL` for a corpus evaluated per component.
        derived_modalities: Modalities obtained from this source by conversion rather
            than by acquisition, e.g. frames derived from a video corpus. Declaring it
            keeps "we would extract frames" distinct from "the dataset contains frames".
        source_type: What kind of thing it is.
        research_role: How it is intended to be used.
        status: Where it stands in approval. See :class:`RegistryStatus`.
        access_type: How access is obtained, or ``unknown``.
        license_status: Whether terms of use are established.
        source_reference: Where it is published. ``not_verified`` until someone reads it.
        version: The release we would pin. ``unknown`` until read.
        train_allowed: Intent to train on it.
        validation_allowed: Intent to tune on it.
        test_allowed: Intent to evaluate in-domain on it.
        unseen_test_allowed: Intent to evaluate out-of-distribution on it. Requires
            :attr:`ResearchRole.TARGET_DOMAIN` and forbids a training role.
        generalization_groups: Groups this entry participates in.
        approximate_size: Scale of the corpus, or an unverified marker. Optional,
            because a dataset we have not read has no size we can state.
        label_schema: The labelling scheme, when the source states one.
        manipulation_types: Canonical manipulation families present, or ``None`` when
            the source has not been read. Deliberately *not* normalized here: a
            dataset's own manipulation vocabulary belongs to its adapter, and asserting
            one now would be a guess.
        requires_request: Whether access needs an agreement. May be unverified.
        requires_terms_acceptance: Whether access needs accepted terms. May be unverified.
        redistribution_allowed: Whether the terms permit redistribution. May be
            unverified. This project redistributes nothing regardless.
        acquisition: Whether the data has actually been acquired.
        notes: What a reader must not get wrong about this entry.
    """

    dataset_id: str
    name: str
    modality: tuple[RegistryModality, ...]
    source_type: SourceType
    research_role: tuple[ResearchRole, ...]
    status: RegistryStatus
    access_type: AccessType
    license_status: LicenseStatus
    source_reference: str
    version: str
    train_allowed: bool
    validation_allowed: bool
    test_allowed: bool
    unseen_test_allowed: bool
    generalization_groups: tuple[str, ...]
    acquisition: AcquisitionRecord = field(default_factory=AcquisitionRecord)
    derived_modalities: tuple[RegistryModality, ...] = ()
    approximate_size: str | None = None
    label_schema: LabelSchema | None = None
    manipulation_types: tuple[str, ...] | None = None
    requires_request: bool | str = "unknown"
    requires_terms_acceptance: bool | str = "unknown"
    redistribution_allowed: bool | str = "pending_verification"
    notes: str = ""

    # --- normalization ------------------------------------------------------

    def __post_init__(self) -> None:
        for name in ("dataset_id", "name", "source_reference", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RegistryValidationError(f"{name} must be a non-empty string, got {value!r}")
            object.__setattr__(self, name, value.strip())

        for name in ("train_allowed", "validation_allowed", "test_allowed", "unseen_test_allowed"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise RegistryValidationError(f"{name} must be true or false, got {value!r}")

        object.__setattr__(self, "modality", _sort_enum_tuple(RegistryModality, self.modality))
        object.__setattr__(
            self, "derived_modalities", _sort_enum_tuple(RegistryModality, self.derived_modalities)
        )
        object.__setattr__(
            self, "research_role", _sort_enum_tuple(ResearchRole, self.research_role)
        )
        object.__setattr__(self, "source_type", SourceType.parse(self.source_type))
        object.__setattr__(self, "status", RegistryStatus.parse(self.status))
        object.__setattr__(self, "access_type", AccessType.parse(self.access_type))
        object.__setattr__(self, "license_status", LicenseStatus.parse(self.license_status))

        if not self.modality:
            raise RegistryValidationError(f"{self.dataset_id}: modality must not be empty")
        if not self.research_role:
            raise RegistryValidationError(f"{self.dataset_id}: research_role must not be empty")

        object.__setattr__(
            self, "generalization_groups", tuple(sorted(set(self.generalization_groups)))
        )
        object.__setattr__(self, "notes", str(self.notes).strip())
        if self.approximate_size is not None:
            object.__setattr__(self, "approximate_size", str(self.approximate_size).strip())

    # --- derived ------------------------------------------------------------

    @property
    def is_intelligence_source(self) -> bool:
        """Whether this entry is an evidence source rather than a detection corpus.

        Intelligence sources cannot become training data, whatever a future edit to this
        file might say.
        """
        return RegistryModality.THREAT_INTELLIGENCE in self.modality or any(
            role.value in _INTELLIGENCE_ROLES for role in self.research_role
        )

    @property
    def usage_flags(self) -> dict[str, bool]:
        """The four intended-use flags, keyed by field name."""
        return {
            "train_allowed": self.train_allowed,
            "validation_allowed": self.validation_allowed,
            "test_allowed": self.test_allowed,
            "unseen_test_allowed": self.unseen_test_allowed,
        }

    @property
    def has_training_intent(self) -> bool:
        """Whether any training or validation use is intended."""
        return self.train_allowed or self.validation_allowed

    @property
    def is_acquired(self) -> bool:
        """Whether the data is on this machine. Never inferred from :attr:`status`."""
        return self.acquisition.acquired

    def has_role(self, role: ResearchRole) -> bool:
        """Whether this entry declares ``role``."""
        return role in self.research_role

    def unverified_fields(self) -> tuple[str, ...]:
        """Fields still holding an unverified marker, sorted.

        Stated explicitly so a reader can see at a glance which questions are still open
        instead of inferring it from a field being null.
        """
        candidates: dict[str, Any] = {
            "access_type": self.access_type.value,
            "license_status": self.license_status.value,
            "source_reference": self.source_reference,
            "version": self.version,
            "approximate_size": self.approximate_size,
            "requires_request": self.requires_request,
            "requires_terms_acceptance": self.requires_terms_acceptance,
            "redistribution_allowed": self.redistribution_allowed,
            "manipulation_types": "unknown" if self.manipulation_types is None else None,
        }
        return tuple(
            sorted(
                name
                for name, value in candidates.items()
                if value is not None and str(value).strip().lower() in UNVERIFIED_MARKERS
            )
        )

    # --- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form with canonical lowercase values."""
        payload: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "modality": [value.value for value in self.modality],
            "derived_modalities": [value.value for value in self.derived_modalities],
            "source_type": self.source_type.value,
            "research_role": [value.value for value in self.research_role],
            "status": self.status.value,
            "access_type": self.access_type.value,
            "license_status": self.license_status.value,
            "source_reference": self.source_reference,
            "version": self.version,
            "train_allowed": self.train_allowed,
            "validation_allowed": self.validation_allowed,
            "test_allowed": self.test_allowed,
            "unseen_test_allowed": self.unseen_test_allowed,
            "generalization_groups": list(self.generalization_groups),
            "approximate_size": self.approximate_size,
            "label_schema": self.label_schema.to_dict() if self.label_schema else None,
            "manipulation_types": (
                list(self.manipulation_types) if self.manipulation_types is not None else None
            ),
            "requires_request": self.requires_request,
            "requires_terms_acceptance": self.requires_terms_acceptance,
            "redistribution_allowed": self.redistribution_allowed,
            "acquisition": self.acquisition.to_dict(),
            "notes": self.notes,
        }
        return dict(sorted(payload.items()))

    def describe(self) -> str:
        """One-line summary for ``--list``."""
        return (
            f"{self.dataset_id:<22} {self.status.value:<20} "
            f"{','.join(value.value for value in self.modality):<22} "
            f"{','.join(value.value for value in self.research_role)}"
        )

    def render(self) -> str:
        """Multi-line, reader-facing description of this entry."""
        lines = [
            f"dataset_id        : {self.dataset_id}",
            f"name              : {self.name}",
            f"modality          : {', '.join(value.value for value in self.modality)}",
        ]
        if self.derived_modalities:
            lines.append(
                f"derived modalities: {', '.join(v.value for v in self.derived_modalities)}"
            )
        lines += [
            f"source_type       : {self.source_type.value}",
            f"research_role     : {', '.join(value.value for value in self.research_role)}",
            f"status            : {self.status.value}",
            f"access_type       : {self.access_type.value}",
            f"license_status    : {self.license_status.value}",
            f"source_reference  : {self.source_reference}",
            f"version           : {self.version}",
            f"approximate_size  : {self.approximate_size or '(not stated)'}",
            "labels            : "
            + (
                f"{self.label_schema.scheme.value} {list(self.label_schema.values)}"
                if self.label_schema
                else "(not stated)"
            ),
            "manipulation types: "
            + (", ".join(self.manipulation_types) if self.manipulation_types else "(unverified)"),
            f"generalization    : {', '.join(self.generalization_groups) or '(none)'}",
            f"acquired locally  : {self.acquisition.acquired}",
        ]
        for flag, value in self.usage_flags.items():
            lines.append(f"{flag:<19}: {value}")
        lines.append(
            "unverified fields : "
            + (", ".join(self.unverified_fields()) if self.unverified_fields() else "(none)")
        )
        lines.append(f"notes             : {self.notes}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GeneralizationPair:
    """One intended train-on-A / test-on-B relationship.

    Declarative. It records the experiment we mean to run and nothing about its result;
    ``executed`` stays ``False`` until both endpoints are actually present, because a
    generalization number without data behind it is the easiest claim in this project to
    fabricate by accident.

    Attributes:
        group_id: Which group this pair belongs to; must be in :data:`GENERALIZATION_GROUPS`.
        source: Dataset id trained on.
        target: Dataset id held out.
        status: State of the intended experiment.
        executed: Whether the experiment has been run. Requires both endpoints to be
            available locally.
        notes: What the experiment is meant to show.
    """

    group_id: str
    source: str
    target: str
    status: GeneralizationStatus = GeneralizationStatus.PLANNED
    executed: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("group_id", "source", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RegistryValidationError(f"{name} must be a non-empty string, got {value!r}")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.executed, bool):
            raise RegistryValidationError(f"executed must be true or false, got {self.executed!r}")
        object.__setattr__(self, "status", GeneralizationStatus.parse(self.status))
        object.__setattr__(self, "notes", str(self.notes).strip())

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form with canonical lowercase values."""
        return dict(
            sorted(
                {
                    "group_id": self.group_id,
                    "source": self.source,
                    "target": self.target,
                    "status": self.status.value,
                    "executed": self.executed,
                    "notes": self.notes,
                }.items()
            )
        )

    def describe(self) -> str:
        return f"{self.group_id:<18} {self.source} -> {self.target} [{self.status.value}]"


@dataclass(frozen=True, slots=True)
class DatasetRegistry:
    """The whole registry: entries, planned cross-dataset pairs, and where they came from.

    Attributes:
        registry_version: Format version of the file that was read.
        entries: Every entry, sorted by ``dataset_id`` so the object is order-independent.
        generalization: Every planned pair, sorted by ``group_id``.
        source_path: The file this registry was read from.
    """

    registry_version: int
    entries: tuple[DatasetEntry, ...]
    generalization: tuple[GeneralizationPair, ...]
    source_path: Path

    def __iter__(self) -> Iterator[DatasetEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def ids(self) -> tuple[str, ...]:
        """Every registered ``dataset_id``, sorted."""
        return tuple(entry.dataset_id for entry in self.entries)

    def get(self, dataset_id: str) -> DatasetEntry | None:
        """The entry with this id, or ``None``."""
        for entry in self.entries:
            if entry.dataset_id == dataset_id:
                return entry
        return None

    def pairs_for(self, dataset_id: str) -> tuple[GeneralizationPair, ...]:
        """Planned pairs in which this dataset is the source or the target."""
        return tuple(
            pair for pair in self.generalization if dataset_id in (pair.source, pair.target)
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form. Keys sorted, so two loads serialize identically."""
        return {
            "registry_version": self.registry_version,
            "datasets": {entry.dataset_id: entry.to_dict() for entry in self.entries},
            "generalization": [pair.to_dict() for pair in self.generalization],
        }


@dataclass(frozen=True, slots=True)
class RegistryReport:
    """Outcome of checking a registry file.

    Returned rather than raised, because the useful behaviour is "tell me everything
    wrong with this file", which a fail-fast exception cannot express.

    Attributes:
        path: The registry file that was checked.
        registry_version: Version declared by the file, or ``None`` when unreadable.
        dataset_count: Entries successfully parsed.
        generalization_count: Pairs successfully parsed.
        errors: Problems that make the registry invalid; empty means valid.
        warnings: Statements worth reading that do not make it invalid.
    """

    path: Path
    registry_version: int | None = None
    dataset_count: int = 0
    generalization_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """``True`` when no errors were found. Warnings do not affect validity."""
        return not self.errors

    def summary(self) -> str:
        """One-line status, as printed by the CLI."""
        status = "VALID" if self.is_valid else "INVALID"
        version = self.registry_version if self.registry_version is not None else "unknown"
        return (
            f"{status}: {self.path} | registry_version={version} "
            f"datasets={self.dataset_count} generalization={self.generalization_count} "
            f"errors={len(self.errors)} warnings={len(self.warnings)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "registry_version": self.registry_version,
            "dataset_count": self.dataset_count,
            "generalization_count": self.generalization_count,
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Problem collection
# ---------------------------------------------------------------------------

#: Returned by :meth:`_Problems.required` for an absent key, so the coercion that
#: follows stays quiet instead of reporting the same problem a second time.
_MISSING: Any = object()

#: Maps a required vocabulary field to its enum. One authority, so the parser cannot
#: drift from the schema documentation.
_VOCABULARIES: dict[str, type[_Vocabulary]] = {
    "source_type": SourceType,
    "status": RegistryStatus,
    "access_type": AccessType,
    "license_status": LicenseStatus,
}


class _Problems:
    """Accumulates validation problems for one section of the file.

    A registry with three mistakes should produce three messages, not one traceback at
    a time, so every coercion appends to a shared list and returns ``None``.

    :meth:`required` returns the :data:`_MISSING` sentinel rather than ``None`` for an
    absent key, so the coercion that follows reports nothing extra: a field that is not
    there is one problem, and saying ``must be true or false, got None`` as well would
    make the file look twice as broken as it is.
    """

    def __init__(self, location: str) -> None:
        self.location = location
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(f"{self.location}: {message}")

    def as_mapping(self, value: Any, what: str = "entry") -> Mapping[str, Any] | None:
        if not isinstance(value, dict):
            self.error(f"{what} must be a mapping, got {_typename(value)}")
            return None
        return value

    def required(self, payload: Mapping[str, Any], key: str) -> Any:
        """Fetch a key that must be present, reporting absence rather than raising."""
        if key not in payload:
            self.error(f"missing required field {key!r}")
            return _MISSING
        return payload[key]

    def text(self, value: Any, field_name: str) -> str | None:
        if value is _MISSING:
            return None
        if value is None:
            self.error(f"{field_name} must not be null; use an unverified marker instead")
            return None
        if not isinstance(value, str) or not value.strip():
            self.error(f"{field_name} must be a non-empty string, got {value!r}")
            return None
        return value.strip()

    def boolean(self, value: Any, field_name: str) -> bool | None:
        if value is _MISSING:
            return None
        if not isinstance(value, bool):
            self.error(f"{field_name} must be true or false, got {value!r}")
            return None
        return value

    def flag(self, value: Any, field_name: str) -> bool | str | None:
        """A flag that may legitimately be unverified."""
        if value is _MISSING:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in UNVERIFIED_MARKERS:
            return value.strip().lower()
        self.error(f"{field_name} must be true, false, or one of {_marker_list()}, got {value!r}")
        return None

    def vocabulary(self, vocabulary: type[_Vocabulary], value: Any, field_name: str) -> Any | None:
        if value is _MISSING:
            return None
        try:
            return vocabulary.parse(value)
        except RegistryValidationError as exc:
            self.error(f"{field_name} {exc}")
            return None

    def vocabulary_list(
        self, vocabulary: type[_Vocabulary], value: Any, field_name: str, *, allow_empty: bool
    ) -> tuple[Any, ...] | None:
        """Parse a single value or a list into a tuple of vocabulary members."""
        if value is _MISSING:
            return None
        if not isinstance(value, list | str | vocabulary):
            self.error(f"{field_name} must be a string or list, got {_typename(value)}")
            return None
        raw_values = value if isinstance(value, list) else [value]
        parsed: list[Any] = []
        for item in raw_values:
            member = self.vocabulary(vocabulary, item, field_name)
            if member is not None:
                parsed.append(member)
        if not parsed and not allow_empty:
            self.error(f"{field_name} must list at least one value")
            return None
        return tuple(parsed)

    def group_list(self, value: Any, field_name: str) -> tuple[str, ...] | None:
        """Parse a generalization group list, which may be empty."""
        if value is _MISSING:
            return None
        if not isinstance(value, list | str):
            self.error(f"{field_name} must be a string or list, got {_typename(value)}")
            return None
        raw_values = value if isinstance(value, list) else [value]
        groups: list[str] = []
        for item in raw_values:
            if not isinstance(item, str) or not item.strip():
                self.error(f"{field_name} entries must be non-empty strings, got {item!r}")
                continue
            group = item.strip()
            if group not in GENERALIZATION_GROUPS:
                self.error(
                    f"{field_name} contains unknown group {group!r}; expected one of "
                    f"{_sorted_values(GENERALIZATION_GROUPS)}. Add it to "
                    "datasets.registry.GENERALIZATION_GROUPS to introduce a new group."
                )
                continue
            groups.append(group)
        return tuple(groups)

    def reject_unknown_keys(self, payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
        """Report keys the schema does not define.

        An unrecognized key is almost always a typo, and a typo is the dangerous kind of
        mistake here: ``generalisation_groups`` would look like a configured field and
        silently disable a check.
        """
        for key in sorted(set(payload) - allowed):
            self.error(f"unknown field {key!r}; expected one of {_sorted_values(allowed)}")


def _typename(value: Any) -> str:
    """YAML-flavoured type name, so a message says ``null`` rather than ``NoneType``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "mapping"
    return type(value).__name__


def _sorted_values(values: Any) -> str:
    return ", ".join(repr(value) for value in sorted(values))


def _marker_list() -> str:
    return ", ".join(sorted(UNVERIFIED_MARKERS))


def _sort_enum_tuple(vocabulary: type[_Vocabulary], values: Sequence[Any]) -> tuple[Any, ...]:
    """De-duplicate and sort a vocabulary list, so serialized output is order-independent."""
    parsed = tuple(vocabulary.parse(value) for value in values)
    order = {member.value: index for index, member in enumerate(vocabulary)}
    return tuple(sorted(dict.fromkeys(parsed), key=lambda member: order[member.value]))


# ---------------------------------------------------------------------------
# Loading and validation
#
# YAML itself is read through datasets/yaml_io.py, which owns the single strict
# policy shared with the adapter configuration files. A duplicate key is an error there,
# because a registry that silently keeps the last value is a registry that disagrees with
# itself.
# ---------------------------------------------------------------------------

_REQUIRED_TEXT_FIELDS = ("dataset_id", "name", "source_reference", "version", "notes")
_REQUIRED_VOCABULARY_FIELDS = ("source_type", "status", "access_type", "license_status")
_REQUIRED_BOOLEAN_FIELDS = (
    "train_allowed",
    "validation_allowed",
    "test_allowed",
    "unseen_test_allowed",
)
_OPTIONAL_FLAG_FIELDS = (
    "requires_request",
    "requires_terms_acceptance",
    "redistribution_allowed",
)

_ENTRY_KEYS = frozenset(
    {
        "dataset_id",
        "name",
        "modality",
        "derived_modalities",
        "source_type",
        "research_role",
        "status",
        "access_type",
        "license_status",
        "source_reference",
        "version",
        "train_allowed",
        "validation_allowed",
        "test_allowed",
        "unseen_test_allowed",
        "generalization_groups",
        "generalization_group",
        "approximate_size",
        "label_schema",
        "manipulation_types",
        "requires_request",
        "requires_terms_acceptance",
        "redistribution_allowed",
        "acquisition",
        "notes",
    }
)

_PAIR_KEYS = frozenset({"group_id", "source", "target", "status", "executed", "notes"})

_ROOT_KEYS = frozenset({"registry_version", "datasets", "generalization"})


def default_registry_path() -> Path:
    """Where the registry file lives, according to ``configs/project.yaml``.

    The configured path wins so that the config stays the single source of truth; the
    literal default exists only so the module still works outside a checkout.

    The config is imported lazily rather than at module scope, mirroring
    :mod:`datasets.ingest`: the registry stays usable when the config layer is not
    importable, and the dependency never becomes a cycle at import time.
    """
    from ml.reproducibility.config import load_project_config, project_root

    try:
        config = load_project_config()
    except Exception:  # pragma: no cover - exercised only outside a valid checkout
        return (project_root() / DEFAULT_REGISTRY_RELPATH).resolve()

    configured = config.get(_REGISTRY_CONFIG_KEY, None)
    if isinstance(configured, str) and configured.strip():
        return (config.root / configured.strip()).resolve()
    return (config.root / DEFAULT_REGISTRY_RELPATH).resolve()


def _resolve_path(path: Path | str | None) -> Path:
    if path is None:
        return default_registry_path()
    return Path(path).resolve()


def _read_document(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse the YAML file, returning ``(document, errors)``.

    Never raises for content problems: a missing file, a syntax error, or a duplicate
    key all come back as messages, because ``--validate`` must report the problem rather
    than crash.
    """
    if not path.is_file():
        return None, [f"registry file not found: {path}"]
    try:
        loaded = strict_yaml_load(path.read_text(encoding="utf-8"), where=str(path))
    except StrictYamlError as exc:
        return None, [str(exc)]
    if loaded is None:
        return None, [f"registry file is empty: {path}"]
    if not isinstance(loaded, dict):
        return None, [f"registry root must be a mapping, got {_typename(loaded)}"]
    return loaded, []


def load_registry(path: Path | str | None = None) -> DatasetRegistry:
    """Load, parse, and validate the registry.

    Args:
        path: Registry file. Defaults to the path configured by ``dataset_registry.file``
            in ``configs/project.yaml``.

    Returns:
        A validated :class:`DatasetRegistry`, with entries sorted by ``dataset_id``.

    Raises:
        RegistryNotFoundError: If the file does not exist.
        RegistryValidationError: If the YAML is malformed, a duplicate key is present, or
            any validation error is found. Every error is attached.
    """
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise RegistryNotFoundError(f"registry file not found: {resolved}")

    document, errors = _read_document(resolved)
    if document is None:
        raise RegistryValidationError(f"could not read registry {resolved}", errors=errors)

    registry, errors, _warnings = _build(document, resolved)
    if registry is None:
        raise RegistryValidationError(
            f"invalid dataset registry ({len(errors)} problem(s))", errors=errors
        )
    return registry


def validate_registry(path: Path | str | None = None) -> RegistryReport:
    """Check the registry and report everything that is wrong with it.

    The non-raising counterpart of :func:`load_registry`, for CI and for ``--validate``.

    Args:
        path: Registry file. Defaults to the configured location.

    Returns:
        A :class:`RegistryReport`. ``report.is_valid`` is ``False`` when any error was
        found; warnings never affect validity.
    """
    resolved = _resolve_path(path)
    document, errors = _read_document(resolved)
    if document is None:
        return RegistryReport(path=resolved, errors=list(errors))

    registry, errors, warnings = _build(document, resolved)
    # Read the version off the raw document rather than from the built registry: a report
    # for an invalid file is most useful when it says which version declared itself.
    declared = document.get("registry_version")
    version = declared if isinstance(declared, int) and not isinstance(declared, bool) else None
    if registry is None:
        return RegistryReport(
            path=resolved, registry_version=version, errors=errors, warnings=warnings
        )
    return RegistryReport(
        path=resolved,
        registry_version=registry.registry_version,
        dataset_count=len(registry.entries),
        generalization_count=len(registry.generalization),
        errors=errors,
        warnings=warnings,
    )


def _build(
    document: dict[str, Any], path: Path
) -> tuple[DatasetRegistry | None, list[str], list[str]]:
    """Build a registry, collecting errors and warnings rather than raising.

    Returns ``(None, errors, warnings)`` when anything is invalid: a partially validated
    registry is not handed out, because a caller that ignores the report would otherwise
    silently act on half a file.
    """
    errors: list[str] = []
    warnings: list[str] = []

    root = _Problems("registry")
    root.reject_unknown_keys(document, _ROOT_KEYS)

    version = document.get("registry_version")
    checked_version = 0
    if version is None:
        root.error("missing required field 'registry_version'")
    elif not isinstance(version, int) or isinstance(version, bool):
        root.error(f"registry_version must be an integer, got {version!r}")
    elif version != REGISTRY_SCHEMA_VERSION:
        root.error(
            f"registry_version {version} is not supported by this module "
            f"(expects {REGISTRY_SCHEMA_VERSION})"
        )
    else:
        checked_version = version
    errors.extend(root.errors)

    raw_datasets = document.get("datasets")
    if raw_datasets is None:
        errors.append("registry: missing required section 'datasets'")
        return None, errors, warnings
    if not isinstance(raw_datasets, dict):
        errors.append(f"registry: 'datasets' must be a mapping, got {_typename(raw_datasets)}")
        return None, errors, warnings
    if not raw_datasets:
        errors.append("registry: 'datasets' must not be empty; an empty registry claims nothing")
        return None, errors, warnings

    entries: list[DatasetEntry] = []
    seen_ids: dict[str, str] = {}
    for key in sorted(raw_datasets):
        entry, entry_errors = _parse_entry(key, raw_datasets[key])
        errors.extend(entry_errors)
        if entry is None:
            continue
        location = f"datasets.{entry.dataset_id}"
        check_errors, check_warnings = _check_entry(entry, location=location)
        errors.extend(check_errors)
        warnings.extend(check_warnings)
        if entry.dataset_id in seen_ids:
            errors.append(
                f"{location}: duplicate dataset_id, already declared under "
                f"datasets.{seen_ids[entry.dataset_id]}"
            )
            continue
        seen_ids[entry.dataset_id] = key
        entries.append(entry)

    pairs, pair_errors = _parse_pairs(document.get("generalization", []))
    errors.extend(pair_errors)
    for pair in pairs:
        pair_errors, pair_warnings = _check_pair(
            pair, entries=entries, location=f"generalization.{pair.group_id}"
        )
        errors.extend(pair_errors)
        warnings.extend(pair_warnings)

    warnings.extend(_cross_check_groups(entries, pairs))
    warnings.extend(_unverified_census(entries))

    if errors:
        return None, errors, warnings

    registry = DatasetRegistry(
        registry_version=checked_version,
        entries=tuple(sorted(entries, key=lambda entry: entry.dataset_id)),
        generalization=tuple(sorted(pairs, key=lambda pair: pair.group_id)),
        source_path=path,
    )
    return registry, errors, warnings


def _parse_entry(key: str, raw: Any) -> tuple[DatasetEntry | None, list[str]]:
    """Build one :class:`DatasetEntry`, collecting every problem in it.

    Returns ``(None, errors)`` when the entry cannot be constructed; a partially valid
    entry is never returned, since a half-parsed entry would hide which fields were
    trusted.
    """
    problems = _Problems(f"datasets.{key}")
    payload = problems.as_mapping(raw)
    if payload is None:
        return None, problems.errors
    problems.reject_unknown_keys(payload, _ENTRY_KEYS)

    values: dict[str, Any] = {}

    for name in _REQUIRED_TEXT_FIELDS:
        text = problems.text(problems.required(payload, name), name)
        if text is not None:
            values[name] = text

    for name in _REQUIRED_VOCABULARY_FIELDS:
        member = problems.vocabulary(_VOCABULARIES[name], problems.required(payload, name), name)
        if member is not None:
            values[name] = member

    for name in _REQUIRED_BOOLEAN_FIELDS:
        flag = problems.boolean(problems.required(payload, name), name)
        if flag is not None:
            values[name] = flag

    for name in _OPTIONAL_FLAG_FIELDS:
        if name in payload:
            maybe_flag = problems.flag(payload[name], name)
            if maybe_flag is not None:
                values[name] = maybe_flag

    modality = problems.vocabulary_list(
        RegistryModality, problems.required(payload, "modality"), "modality", allow_empty=False
    )
    if modality is not None:
        values["modality"] = modality

    roles = problems.vocabulary_list(
        ResearchRole,
        problems.required(payload, "research_role"),
        "research_role",
        allow_empty=False,
    )
    if roles is not None:
        values["research_role"] = roles

    if "derived_modalities" in payload:
        derived = problems.vocabulary_list(
            RegistryModality, payload["derived_modalities"], "derived_modalities", allow_empty=True
        )
        if derived is not None:
            values["derived_modalities"] = derived

    # Accept either spelling of the group field; the plural is canonical because an entry
    # can belong to more than one group. Declaring both at once is ambiguous rather than
    # redundant, so it is an error instead of one silently winning.
    group_key = _group_field_name(payload)
    if group_key is None:
        both = [
            name for name in ("generalization_groups", "generalization_group") if name in payload
        ]
        if both:
            problems.error(
                f"declare only one of {', '.join(repr(name) for name in both)}; "
                "'generalization_groups' is canonical"
            )
        else:
            problems.error(
                "missing required field 'generalization_groups' (a single value may also "
                "be written as 'generalization_group')"
            )
    else:
        groups = problems.group_list(payload[group_key], group_key)
        if groups is not None:
            values["generalization_groups"] = groups

    if "approximate_size" in payload:
        parsed = problems.text(payload["approximate_size"], "approximate_size")
        if parsed is not None:
            values["approximate_size"] = parsed

    if "manipulation_types" in payload:
        manipulations = _parse_manipulation_types(payload["manipulation_types"], problems)
        if manipulations is not None:
            values["manipulation_types"] = manipulations

    if "label_schema" in payload:
        label_schema, schema_errors = _parse_label_schema(payload["label_schema"], key)
        problems.errors.extend(schema_errors)
        if label_schema is not None:
            values["label_schema"] = label_schema

    acquisition = _parse_acquisition(payload.get("acquisition", {}), key, problems)
    if acquisition is not None:
        values["acquisition"] = acquisition

    if problems.errors:
        return None, problems.errors

    try:
        return DatasetEntry(**values), []
    except RegistryValidationError as exc:  # pragma: no cover - defensive
        return None, [*problems.errors, f"datasets.{key}: {exc}"]


def _group_field_name(payload: Mapping[str, Any]) -> str | None:
    """Resolve which spelling of the generalization group field the entry uses."""
    present = [
        name for name in ("generalization_groups", "generalization_group") if name in payload
    ]
    if len(present) > 1:
        return None
    return present[0] if present else None


def _parse_manipulation_types(value: Any, problems: _Problems) -> tuple[str, ...] | None:
    """Parse manipulation families, or record that the source has not been read.

    A marker string is not a manipulation list: it means the source has not been read,
    so the honest result is ``None`` rather than a one-element list containing the word
    ``pending_verification``.
    """
    if isinstance(value, str):
        if value.strip().lower() in UNVERIFIED_MARKERS:
            return None
        problems.error(
            "manipulation_types must be a list of canonical manipulation families, or an "
            f"unverified marker ({_marker_list()}), got {value!r}"
        )
        return None
    if not isinstance(value, list):
        problems.error(f"manipulation_types must be a list or marker, got {_typename(value)}")
        return None
    families: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            problems.error(f"manipulation_types entries must be non-empty strings, got {item!r}")
            continue
        families.append(item.strip())
    return tuple(families)


def _parse_label_schema(raw: Any, key: str) -> tuple[LabelSchema | None, list[str]]:
    location = f"datasets.{key}.label_schema"
    if not isinstance(raw, dict):
        return None, [f"{location}: must be a mapping, got {_typename(raw)}"]
    allowed = frozenset({"scheme", "values", "description"})
    problems = _Problems(location)
    problems.reject_unknown_keys(raw, allowed)

    scheme = problems.vocabulary(LabelScheme, raw.get("scheme"), "scheme")
    if "scheme" not in raw:
        problems.error("missing required field 'scheme'")

    values: tuple[str, ...] = ()
    raw_values = raw.get("values", [])
    if not isinstance(raw_values, list) or not all(isinstance(item, str) for item in raw_values):
        problems.error(f"values must be a list of strings, got {raw_values!r}")
    else:
        values = tuple(item.strip() for item in raw_values if item.strip())

    description = raw.get("description", "")
    if not isinstance(description, str):
        problems.error(f"description must be a string, got {_typename(description)}")
        description = ""

    if problems.errors or scheme is None:
        return None, problems.errors
    return LabelSchema(scheme=scheme, values=values, description=description), []


def _parse_acquisition(raw: Any, key: str, problems: _Problems) -> AcquisitionRecord | None:
    location = f"datasets.{key}.acquisition"
    if raw is None:
        return AcquisitionRecord()
    if not isinstance(raw, dict):
        problems.error(f"acquisition must be a mapping, got {_typename(raw)}")
        return None
    allowed = frozenset({"acquired", "access_date", "version", "local_path"})
    acquisition_problems = _Problems(location)
    acquisition_problems.reject_unknown_keys(raw, allowed)

    acquired = raw.get("acquired", False)
    if not isinstance(acquired, bool):
        acquisition_problems.error(f"acquired must be true or false, got {acquired!r}")
        acquired = None

    text_fields: dict[str, str | None] = {}
    for name in ("access_date", "version", "local_path"):
        value = raw.get(name)
        if value is None:
            text_fields[name] = None
            continue
        if not isinstance(value, str) or not value.strip():
            acquisition_problems.error(f"{name} must be a non-empty string or null, got {value!r}")
            text_fields[name] = None
            continue
        text_fields[name] = value.strip()

    access_date = text_fields["access_date"]
    if access_date is not None and not _ACCESS_DATE_PATTERN.match(access_date):
        acquisition_problems.error(
            f"access_date must be YYYY-MM-DD or an RFC 3339 timestamp, got {access_date!r}"
        )
        access_date = None

    problems.errors.extend(acquisition_problems.errors)
    if acquired is None or acquisition_problems.errors:
        return None
    return AcquisitionRecord(
        acquired=acquired,
        access_date=access_date,
        version=text_fields["version"],
        local_path=text_fields["local_path"],
    )


def _parse_pairs(raw: Any) -> tuple[tuple[GeneralizationPair, ...], list[str]]:
    problems = _Problems("generalization")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        return (), [f"generalization: must be a list, got {_typename(raw)}"]

    pairs: list[GeneralizationPair] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(raw):
        location = f"generalization[{index}]"
        if not isinstance(item, dict):
            problems.errors.append(f"{location}: must be a mapping, got {_typename(item)}")
            continue
        pair_problems = _Problems(location)
        pair_problems.reject_unknown_keys(item, _PAIR_KEYS)

        values: dict[str, Any] = {}
        for name in ("group_id", "source", "target"):
            parsed = pair_problems.text(item.get(name), name)
            if name not in item:
                pair_problems.error(f"missing required field {name!r}")
            elif parsed is not None:
                values[name] = parsed

        group_id = values.get("group_id", f"#{index}")
        if "group_id" in values and values["group_id"] not in GENERALIZATION_GROUPS:
            pair_problems.error(
                f"group_id {values['group_id']!r} is not a declared generalization group; "
                f"expected one of {_sorted_values(GENERALIZATION_GROUPS)}"
            )

        if "status" in item:
            status = pair_problems.vocabulary(GeneralizationStatus, item["status"], "status")
            if status is not None:
                values["status"] = status
        if "executed" in item:
            executed = pair_problems.boolean(item["executed"], "executed")
            if executed is not None:
                values["executed"] = executed
        notes = item.get("notes", "")
        if not isinstance(notes, str):
            pair_problems.error(f"notes must be a string, got {_typename(notes)}")
        else:
            values["notes"] = notes

        problems.errors.extend(pair_problems.errors)
        if pair_problems.errors:
            continue
        if group_id in seen:
            problems.errors.append(
                f"{location}: duplicate group_id {group_id!r}, already declared at "
                f"generalization[{seen[group_id]}]"
            )
            continue
        seen[group_id] = index
        try:
            pairs.append(GeneralizationPair(**values))
        except RegistryValidationError as exc:
            problems.errors.append(f"{location}: {exc}")

    return tuple(pairs), problems.errors


# ---------------------------------------------------------------------------
# Semantics: what an entry is allowed to claim
# ---------------------------------------------------------------------------


def _check_entry(entry: DatasetEntry, *, location: str) -> tuple[list[str], list[str]]:
    """Cross-field rules that make a claim meaningful rather than merely present.

    Returns ``(errors, warnings)``. An error means the file misstates something; a
    warning means it is correct, but the reader should know a gap exists.
    """
    errors: list[str] = []
    warnings: list[str] = []

    def fail(message: str) -> None:
        errors.append(f"{location}: {message}")

    def warn(message: str) -> None:
        warnings.append(f"{location}: {message}")

    # 1. An entry claiming to be multimodal has to say which components it holds.
    if RegistryModality.MULTIMODAL in entry.modality and len(entry.derived_modalities) < 2:
        fail(
            "modality 'multimodal' requires derived_modalities naming at least two "
            f"components, got {_sorted_values([m.value for m in entry.derived_modalities])}"
        )

    # 2. The unseen-test claim. Training on the target domain voids it, so the two
    #    cannot both be true.
    if entry.unseen_test_allowed and entry.train_allowed:
        fail(
            "unseen_test_allowed and train_allowed cannot both be true: training on a "
            "target domain voids the unseen-test claim"
        )
    if entry.unseen_test_allowed and entry.validation_allowed:
        fail(
            "unseen_test_allowed and validation_allowed cannot both be true: tuning "
            "against a target domain voids the unseen-test claim"
        )
    if entry.unseen_test_allowed and ResearchRole.TRAINING in entry.research_role:
        fail("unseen_test_allowed conflicts with research_role 'training'")
    if entry.unseen_test_allowed and not entry.has_role(ResearchRole.TARGET_DOMAIN):
        fail("unseen_test_allowed requires research_role 'target_domain'")
    if ResearchRole.TARGET_DOMAIN in entry.research_role and not entry.unseen_test_allowed:
        fail("research_role 'target_domain' requires unseen_test_allowed")
    if ResearchRole.TARGET_DOMAIN in entry.research_role and entry.test_allowed:
        fail(
            "a target domain cannot also be an in-domain test set: test_allowed makes it "
            "neither held out nor unseen"
        )
    if ResearchRole.SOURCE_DOMAIN in entry.research_role and not entry.train_allowed:
        fail("research_role 'source_domain' requires train_allowed")

    # 3. Intelligence sources are evidence, never training data.
    if entry.is_intelligence_source:
        if entry.train_allowed or entry.validation_allowed:
            fail("an intelligence source cannot be a training or validation corpus")
        if (
            entry.label_schema
            and entry.label_schema.scheme.value not in _INTELLIGENCE_LABEL_SCHEMES
        ):
            fail(
                "an intelligence source cannot declare class labels; use evidence_state, "
                f"unlabeled, or unknown, got {entry.label_schema.scheme.value!r}"
            )
    elif entry.label_schema is None:
        fail("no label_schema declared; a detection corpus needs a scheme before ingestion")

    # 4. Status gates acquisition, not the other way round.
    if entry.status is RegistryStatus.AVAILABLE_LOCALLY:
        if not entry.is_acquired:
            fail("status 'available_locally' requires acquisition.acquired to be true")
        elif not entry.acquisition.is_complete():
            missing = [
                name
                for name in ("access_date", "version", "local_path")
                if getattr(entry.acquisition, name) is None
            ]
            fail(
                "status 'available_locally' requires a complete acquisition record; missing "
                + ", ".join(missing)
            )
    elif entry.is_acquired:
        fail(
            f"acquisition.acquired is true while status is {entry.status.value!r}; acquired "
            "data must be recorded as 'available_locally'"
        )

    if (
        entry.status is RegistryStatus.APPROVED
        and entry.license_status is not LicenseStatus.VERIFIED
    ):
        fail("status 'approved' requires license_status 'verified'")
    if entry.status is RegistryStatus.BLOCKED and any(entry.usage_flags.values()):
        fail("status 'blocked' requires every usage flag to be false")

    if entry.status in {RegistryStatus.BLOCKED, RegistryStatus.DEPRECATED} and (
        entry.generalization_groups
    ):
        fail(f"status {entry.status.value!r} cannot participate in a generalization group")

    # Gaps that would silently weaken a later result.
    if entry.has_training_intent and entry.license_status in {
        LicenseStatus.UNKNOWN,
        LicenseStatus.PENDING_VERIFICATION,
    }:
        warn(
            f"training intent declared while license_status is "
            f"{entry.license_status.value!r}; nothing may be acquired until it is verified"
        )
    if entry.has_training_intent and entry.status in {
        RegistryStatus.CANDIDATE,
        RegistryStatus.PENDING_VERIFICATION,
    }:
        warn(
            "usage flags state intent only; acquisition stays gated until status leaves "
            f"{entry.status.value!r}"
        )
    if entry.redistribution_allowed is True:
        warn(
            "redistribution_allowed is true, which conflicts with this project's policy of "
            "never redistributing dataset media"
        )
    if entry.is_intelligence_source and entry.access_type is AccessType.UNKNOWN:
        warn(
            "access_type is unknown; no collector may be configured against this source "
            "until it is verified"
        )
    return errors, warnings


def _check_pair(
    pair: GeneralizationPair, *, entries: Sequence[DatasetEntry], location: str
) -> tuple[list[str], list[str]]:
    """A generalization pair must be executable, or it must admit that it is not."""
    errors: list[str] = []
    warnings: list[str] = []
    by_id = {entry.dataset_id: entry for entry in entries}

    source = by_id.get(pair.source)
    target = by_id.get(pair.target)
    if source is None:
        errors.append(f"{location}: source {pair.source!r} is not a registered dataset")
    if target is None:
        errors.append(f"{location}: target {pair.target!r} is not a registered dataset")
    if source is not None and source is target:
        errors.append(f"{location}: source and target are the same dataset")
    if pair.status is GeneralizationStatus.EXECUTED and not pair.executed:
        errors.append(f"{location}: status 'executed' requires executed: true")
    if pair.executed and pair.status is not GeneralizationStatus.EXECUTED:
        errors.append(f"{location}: executed: true requires status 'executed'")

    for label, endpoint in (("source", source), ("target", target)):
        if endpoint is None:
            continue
        if endpoint.is_intelligence_source:
            errors.append(
                f"{location}: the {label} {endpoint.dataset_id!r} is an intelligence "
                "source, which cannot take part in a generalization experiment"
            )
        if endpoint.status in {RegistryStatus.BLOCKED, RegistryStatus.DEPRECATED}:
            errors.append(
                f"{location}: the {label} {endpoint.dataset_id!r} is {endpoint.status.value}, "
                "so no result can be reported for this pair"
            )

    if source is not None:
        if not source.has_role(ResearchRole.SOURCE_DOMAIN):
            errors.append(
                f"{location}: source {source.dataset_id!r} does not declare "
                "research_role 'source_domain'"
            )
        if not source.train_allowed:
            errors.append(
                f"{location}: source {source.dataset_id!r} cannot be trained on "
                "(train_allowed is false)"
            )
        if pair.group_id not in source.generalization_groups:
            errors.append(
                f"{location}: source {source.dataset_id!r} does not declare group "
                f"{pair.group_id!r} in generalization_groups"
            )

    if target is not None:
        if not target.has_role(ResearchRole.TARGET_DOMAIN):
            errors.append(
                f"{location}: target {target.dataset_id!r} does not declare "
                "research_role 'target_domain'"
            )
        if not target.unseen_test_allowed:
            errors.append(
                f"{location}: target {target.dataset_id!r} is not usable as a held-out "
                "target (unseen_test_allowed must be true)"
            )
        if pair.group_id not in target.generalization_groups:
            errors.append(
                f"{location}: target {target.dataset_id!r} does not declare group "
                f"{pair.group_id!r} in generalization_groups"
            )

    if pair.executed:
        for label, endpoint in (("source", source), ("target", target)):
            if endpoint is not None and endpoint.status is not RegistryStatus.AVAILABLE_LOCALLY:
                errors.append(
                    f"{location}: executed: true requires the {label} "
                    f"{endpoint.dataset_id!r} to be available_locally, but its status is "
                    f"{endpoint.status.value!r}"
                )

    if pair.status is GeneralizationStatus.PLANNED:
        unverified = sorted(
            endpoint.dataset_id
            for endpoint in (source, target)
            if endpoint is not None and endpoint.license_status is not LicenseStatus.VERIFIED
        )
        if unverified:
            warnings.append(
                f"{location}: the intended claim depends on unverified licences for "
                f"{', '.join(unverified)}; nothing may be acquired yet"
            )
    return errors, warnings


def _cross_check_groups(
    entries: Sequence[DatasetEntry], pairs: Sequence[GeneralizationPair]
) -> list[str]:
    """Report groups an entry claims membership of but no pair actually uses.

    Not an error: declaring an intended group before the experiment is spelled out is
    reasonable. It is a warning because a group with no pair is usually a stale claim,
    and a stale claim is how a reader ends up believing an experiment is planned when
    nobody has written it down.
    """
    planned = {pair.group_id for pair in pairs}
    warnings: list[str] = []
    for entry in entries:
        orphans = sorted(set(entry.generalization_groups) - planned)
        if orphans:
            warnings.append(
                f"datasets.{entry.dataset_id}: declares group "
                f"{', '.join(orphans)} with no generalization pair defined for it"
            )
    return warnings


def _unverified_census(entries: Sequence[DatasetEntry]) -> list[str]:
    """One line stating how much of the registry is still unverified.

    Deliberately a single aggregate rather than one warning per entry per field: a file
    where nothing has been verified yet would otherwise emit dozens of identical lines
    and hide the one that matters.
    """
    total = len(entries)
    if total == 0:
        return []
    counts = {
        "source_reference": sum(1 for entry in entries if is_unverified(entry.source_reference)),
        "version": sum(1 for entry in entries if is_unverified(entry.version)),
        "license_status": sum(
            1
            for entry in entries
            if entry.license_status in {LicenseStatus.UNKNOWN, LicenseStatus.PENDING_VERIFICATION}
        ),
        "access_type": sum(1 for entry in entries if entry.access_type is AccessType.UNKNOWN),
    }
    open_fields = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()) if count)
    if not open_fields:
        return []
    return [
        (
            f"registry: unverified across {total} registered source(s) ({open_fields}); no "
            "result can yet be pinned to a version, a licence, or a verified reference"
        )
    ]


def is_unverified(value: str) -> bool:
    """Whether ``value`` is an unverified marker rather than a real value.

    Exposed because callers deciding what a result may cite need the same definition
    the loader uses; two implementations of "unknown" would eventually disagree.
    """
    return str(value).strip().lower() in UNVERIFIED_MARKERS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog="python -m datasets.registry",
        description=(
            "Inspect the dataset and public-intelligence registry: what we intend to "
            "evaluate, in what role, and what is still unverified."
        ),
        epilog=(
            "This command reads one configuration file. It never downloads data, "
            "contacts a feed, or writes anything."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry file. Defaults to dataset_registry.file in configs/project.yaml.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check the registry and report every problem and warning found.",
    )
    parser.add_argument(
        "--list",
        dest="list_entries",
        action="store_true",
        help="Print one line per registered dataset or source.",
    )
    parser.add_argument(
        "--show",
        metavar="DATASET_ID",
        default=None,
        help="Print one entry in full, with the generalization pairs it participates in.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of text.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Returns:
        ``0`` on success, ``1`` when validation found an error, ``2`` for a bad
        request (no action, missing file, unknown dataset id).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.validate or args.list_entries or args.show):
        parser.print_help()
        return 2

    try:
        if args.validate:
            return _run_validate(args)
        if args.list_entries:
            return _run_list(args)
        return _run_show(args)
    except (RegistryError, OSError, StrictYamlError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_validate(args: argparse.Namespace) -> int:
    report = validate_registry(args.registry)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.summary())
        for message in report.errors:
            print(f"  error   : {message}")
        for message in report.warnings:
            print(f"  warning : {message}")
    return 0 if report.is_valid else 1


def _run_list(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    if args.json:
        print(
            json.dumps(
                {
                    "registry_version": registry.registry_version,
                    "datasets": [entry.to_dict() for entry in registry],
                    "generalization": [pair.to_dict() for pair in registry.generalization],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"registry: {registry.source_path} (version {registry.registry_version})")
    print(f"datasets: {len(registry.entries)}")
    for entry in registry:
        print(f"  {entry.describe()}")
    if registry.generalization:
        print(f"generalization: {len(registry.generalization)}")
        for pair in registry.generalization:
            print(f"  {pair.describe()}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    entry = registry.get(args.show)
    if entry is None:
        known = ", ".join(registry.ids())
        print(f"error: unknown dataset id {args.show!r}; registered ids: {known}", file=sys.stderr)
        return 2

    pairs = registry.pairs_for(entry.dataset_id)
    if args.json:
        print(
            json.dumps(
                {
                    "entry": entry.to_dict(),
                    "generalization": [pair.to_dict() for pair in pairs],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(entry.render())
    if pairs:
        print("generalization:")
        for pair in pairs:
            position = "source" if pair.source == entry.dataset_id else "target"
            print(f"  {pair.describe()} ({position}, executed={pair.executed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
