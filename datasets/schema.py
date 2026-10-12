"""Dataset domain models: samples, manifests, and their validation.

This module defines *what a sample is* and *what makes a manifest valid*. The
operational helpers (JSONL I/O, hashing, verification) live in
:mod:`datasets.manifest`.

Optionality is explicit
-----------------------
Real-world datasets are heterogeneous. A face-forensics dataset may have no audio
metadata; an audio corpus may have no per-sample device information. So most
fields are ``None``-able, and :meth:`MediaSample.which_fields_are_populated`
reports what is actually present.

What is *not* optional is identity and provenance-critical data: ``sample_id``,
``dataset_name``, ``modality``, ``label``, ``filepath``, and ``sha256`` are
required. A sample without a verifiable content hash cannot support the
reproducibility contract this project claims, so it is rejected at construction.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields
from datetime import UTC, datetime
from enum import Enum
from typing import Any

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ManipulationType",
    "ManifestValidationError",
    "MediaSample",
    "Modality",
    "SampleLabel",
    "SplitName",
    "SplitSchemaVersion",
]

MANIFEST_SCHEMA_VERSION = 1
SplitSchemaVersion = 1


class ManifestValidationError(ValueError):
    """Raised when a manifest or sample record is invalid.

    Collects every problem found rather than only the first, so a malformed
    manifest can be fixed in one pass. ``str(error)`` includes the full list, so a
    traceback alone is enough to diagnose the failure.
    """

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        if self.errors:
            message = f"{message}: " + "; ".join(self.errors)
        super().__init__(message)


class Modality(str, Enum):
    """Media type of a sample. These are the only three accepted values."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def parse(cls, value: Any) -> Modality:
        """Coerce ``value`` to a :class:`Modality`, raising on anything else.

        Matching is case-insensitive on input but the canonical stored form is
        always lowercase, so manifests do not fragment on ``"Image"`` vs ``"image"``.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().lower()
            for member in cls:
                if member.value == candidate:
                    return member
        raise ManifestValidationError(
            f"invalid modality {value!r}; expected one of {', '.join(cls.values())}"
        )


class SampleLabel(str, Enum):
    """Ground-truth class of a sample."""

    REAL = "real"
    FAKE = "fake"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def parse(cls, value: Any) -> SampleLabel:
        """Coerce ``value`` to a :class:`SampleLabel`, raising on anything else."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().lower()
            for member in cls:
                if member.value == candidate:
                    return member
        raise ManifestValidationError(
            f"invalid label {value!r}; expected one of {', '.join(cls.values())}"
        )


class SplitName(str, Enum):
    """Canonical split identifiers used across manifests and split definitions.

    Four splits are modelled, not three:

    ``TRAIN``
        Fit model parameters.
    ``VALIDATION``
        Model selection and threshold tuning. Never used for gradient updates.
    ``TEST``
        In-distribution held-out evaluation.
    ``UNSEEN_TEST``
        Out-of-distribution evaluation. Resolved against a dataset the model was
        never trained or tuned on, and typically frozen until the model is final.
        Kept separate from ``TEST`` so that repeatedly consulting ``TEST`` while
        iterating cannot silently consume the only out-of-distribution measurement.

    ``UNSEEN_TEST`` is unused by many experiments and legitimately empty.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNSEEN_TEST = "unseen_test"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def parse(cls, value: Any) -> SplitName:
        """Coerce ``value`` to a :class:`SplitName`, raising on anything else.

        Accepts the common aliases ``val``/``valid`` for validation and
        ``unseen``/``ood_test`` for unseen test.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
            aliases: dict[str, SplitName] = {
                "val": SplitName.VALIDATION,
                "valid": SplitName.VALIDATION,
                "unseen": SplitName.UNSEEN_TEST,
                "ood": SplitName.UNSEEN_TEST,
                "ood_test": SplitName.UNSEEN_TEST,
                "held_out": SplitName.UNSEEN_TEST,
            }
            if candidate in aliases:
                return aliases[candidate]
            for member in cls:
                if member.value == candidate:
                    return member
        raise ManifestValidationError(
            f"invalid split {value!r}; expected one of {', '.join(cls.values())}"
        )


class ManipulationType(str, Enum):
    """Known manipulation families.

    This is a *closed* vocabulary on purpose: free-text manipulation labels defeat
    aggregation and make cross-dataset comparison impossible. Datasets with labels
    outside this set must map to :attr:`OTHER` and record the original wording in
    :attr:`MediaSample.metadata`.

    Extending the vocabulary
    ------------------------
    Adding a member here is the sanctioned way to extend it. The cost of adding one
    is that every reader of every published result must handle it, so a new member
    needs a definition and a reason — it is not a place to park one dataset's private
    wording. Adapters still map their own tokens to members *explicitly*
    (:mod:`datasets.normalization`); adding a member does not let an adapter guess.

    ``UNKNOWN`` is not a dumping ground. It means the dataset did not say, and it is
    a materially different statement from :attr:`OTHER`, which means the dataset said
    something that does not fit here.
    """

    NONE = "none"
    FACE_SWAP = "face_swap"
    FACE_REENACTMENT = "face_reenactment"
    FACIAL_ATTRIBUTE_EDIT = "facial_attribute_edit"
    FULL_BODY_SYNTHESIS = "full_body_synthesis"
    SYNTHETIC_MEDIA = "synthetic_media"
    AUDIO_VOCODER = "audio_vocoder"
    VOICE_CLONING = "voice_cloning"
    VOICE_CONVERSION = "voice_conversion"
    AUDIO_MANIPULATION = "audio_manipulation"
    SPEECH_INPAINTING = "speech_inpainting"
    LIP_SYNC = "lip_sync"
    VIDEO_COMPRESSION_ARTIFACT = "video_compression_artifact"
    GAN_IMAGE = "gan_image"
    DIFFUSION_IMAGE = "diffusion_image"
    OTHER = "other"
    UNKNOWN = "unknown"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def parse(cls, value: Any) -> ManipulationType:
        """Coerce ``value`` to a :class:`ManipulationType`.

        ``None`` maps to :attr:`UNKNOWN` rather than failing, because absent
        manipulation metadata is common and legitimate.

        Spellings listed in :data:`_MANIPULATION_ALIASES` are accepted and resolve to
        their canonical member.
        """
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        if isinstance(value, str):
            candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
            candidate = _MANIPULATION_ALIASES.get(candidate, candidate)
            for member in cls:
                if member.value == candidate:
                    return member
        raise ManifestValidationError(
            f"invalid manipulation_type {value!r}; expected one of {', '.join(cls.values())}"
        )


_MANIPULATION_ALIASES: dict[str, str] = {
    # Datasets and papers write both spellings for one manipulation family. The
    # canonical member stays VOICE_CLONING; the shorter spelling is accepted on input
    # so that an adapter can pass through a dataset's own wording, rather than becoming
    # a second member that would split one family into two identities in every count.
    "voice_clone": "voice_cloning",
}


_HEX_DIGITS = frozenset("0123456789abcdef")


def _validate_sha256(value: Any, field_name: str = "sha256") -> str:
    """Normalize and validate a lowercase hex SHA-256 digest."""
    if not isinstance(value, str):
        raise ManifestValidationError(
            f"{field_name} must be a hex string, got {type(value).__name__}"
        )
    normalized = value.strip().lower()
    if len(normalized) != 64:
        raise ManifestValidationError(
            f"{field_name} must be 64 hex characters, got {len(normalized)}"
        )
    if not _HEX_DIGITS.issuperset(normalized):
        raise ManifestValidationError(f"{field_name} is not valid hex: {value!r}")
    return normalized


def _require_non_empty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field_name} must be a non-empty string, got {value!r}")
    return value.strip()


@dataclass(frozen=True, slots=True)
class MediaSample:
    """One media sample in a dataset manifest.

    Media bytes are never stored in the manifest. ``filepath`` locates the file and
    ``sha256`` proves its identity, so a changed file is detectable
    (:mod:`datasets.manifest`).

    Attributes:
        sample_id: Unique within a dataset. Prefer a stable upstream identifier
            over a positional index, which shifts whenever a dataset is repackaged.
        dataset_name: Dataset this sample belongs to, e.g. ``"faceforensics_pp"``.
        filepath: Path to the media file. Relative to the dataset root when
            possible, so a checkout stays portable across machines.
        modality: One of :class:`Modality`.
        label: Ground truth, ``real`` or ``fake``.
        manipulation_type: Manipulation family. Defaults to ``unknown``.
        source: Where the sample came from, e.g. ``"original"``, ``"reencoded"``,
            ``"platform_download"``, ``"web_scrape"``.
        split: Assigned split. ``None`` means the manifest does not assign one;
            authoritative assignments live in the frozen split definition.
        sha256: Hex digest of the file bytes. Required.
        license: SPDX identifier or dataset-specific licence string. Required, since
            licensing governs whether the sample may be used at all.
        metadata: Free-form, dataset-specific extras (duration, frame count,
            resolution, encoder, device). Never interpreted by the core schema.
    """

    sample_id: str
    dataset_name: str
    filepath: str
    modality: Modality
    label: SampleLabel
    sha256: str
    license: str
    manipulation_type: ManipulationType = ManipulationType.UNKNOWN
    source: str | None = None
    split: SplitName | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate on construction, so the type is always valid."""
        object.__setattr__(self, "sample_id", _require_non_empty_str(self.sample_id, "sample_id"))
        object.__setattr__(
            self, "dataset_name", _require_non_empty_str(self.dataset_name, "dataset_name")
        )
        object.__setattr__(self, "filepath", _require_non_empty_str(self.filepath, "filepath"))
        object.__setattr__(self, "modality", Modality.parse(self.modality))
        object.__setattr__(self, "label", SampleLabel.parse(self.label))
        object.__setattr__(
            self, "manipulation_type", ManipulationType.parse(self.manipulation_type)
        )
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256))
        object.__setattr__(self, "license", _require_non_empty_str(self.license, "license"))
        object.__setattr__(
            self, "source", self.source.strip() if isinstance(self.source, str) else None
        )
        object.__setattr__(
            self, "split", SplitName.parse(self.split) if self.split is not None else None
        )
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        elif not isinstance(self.metadata, dict):
            raise ManifestValidationError(
                f"metadata must be a mapping, got {type(self.metadata).__name__}"
            )

    # --- derived information ----------------------------------------------

    @property
    def is_fake(self) -> bool:
        """``True`` when the label is ``fake``."""
        return self.label is SampleLabel.FAKE

    def which_fields_are_populated(self) -> tuple[str, ...]:
        """Names of the optional fields that carry a value.

        Lets a manifest consumer assert coverage without hardcoding assumptions,
        e.g. ``assert "source" in sample.which_fields_are_populated()``.
        """
        populated: list[str] = []
        for f in fields(self):
            if f.name in {
                "sample_id",
                "dataset_name",
                "filepath",
                "modality",
                "label",
                "sha256",
                "license",
            }:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "metadata" and not value:
                continue
            if f.name == "manipulation_type" and value is ManipulationType.UNKNOWN:
                continue
            populated.append(f.name)
        return tuple(sorted(populated))

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict with canonical lowercase enum values."""
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "filepath": self.filepath,
            "modality": self.modality.value,
            "label": self.label.value,
            "manipulation_type": self.manipulation_type.value,
            "source": self.source,
            "split": self.split.value if self.split is not None else None,
            "sha256": self.sha256,
            "license": self.license,
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MediaSample:
        """Build a sample from a manifest record.

        Only fields without defaults are required. Optional fields may be absent, but
        an explicit ``null`` is honoured, so ``"split": null`` and a missing ``split``
        key both mean "unassigned".

        Unknown keys are preserved into ``metadata`` under ``"_extra"`` so that
        reading a manifest written by a newer schema version does not lose
        information silently.

        Raises:
            ManifestValidationError: If a required field is missing or invalid.
        """
        if not isinstance(raw, dict):
            raise ManifestValidationError(
                f"sample record must be an object, got {type(raw).__name__}"
            )

        known = {f.name for f in fields(cls)}
        payload: dict[str, Any] = {key: raw[key] for key in raw if key in known}

        extra = {key: value for key, value in raw.items() if key not in known}
        metadata = dict(payload.get("metadata") or {})
        if extra:
            metadata["_extra"] = extra
        payload["metadata"] = metadata

        missing = [
            f.name
            for f in fields(cls)
            if f.name not in payload and f.default is MISSING and f.default_factory is MISSING
        ]
        if missing:
            raise ManifestValidationError(
                f"sample record is missing required field(s): {', '.join(sorted(missing))}",
                errors=sorted(missing),
            )

        try:
            return cls(**payload)
        except TypeError as exc:
            raise ManifestValidationError(f"could not build sample: {exc}") from exc


def utc_timestamp() -> str:
    """Current UTC time as an RFC 3339 string, second precision."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def validate_samples(samples: list[MediaSample]) -> None:
    """Validate a collection of samples for problems no single sample can detect.

    Checks that ``sample_id`` is unique within ``dataset_name``.

    Raises:
        ManifestValidationError: If duplicate identifiers are found.
    """
    seen: dict[tuple[str, str], int] = {}
    duplicates: list[str] = []
    for index, sample in enumerate(samples):
        key = (sample.dataset_name, sample.sample_id)
        if key in seen:
            duplicates.append(f"{sample.dataset_name}/{sample.sample_id}")
        else:
            seen[key] = index
    if duplicates:
        raise ManifestValidationError(
            f"duplicate sample_id detected ({len(duplicates)} occurrence(s))",
            errors=sorted(set(duplicates)),
        )


def validate_modality_counts(samples: list[MediaSample]) -> dict[Modality, int]:
    """Count samples per modality, for dataset composition reporting."""
    counts = dict.fromkeys(Modality, 0)
    for sample in samples:
        counts[sample.modality] += 1
    return counts


def as_json(payload: Any) -> str:
    """Serialize ``payload`` deterministically: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
