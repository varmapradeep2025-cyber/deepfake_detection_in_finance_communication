"""Frozen split definitions and their determinism guarantees.

Why frozen splits
-----------------
A split is the single most load-bearing decision in a detection benchmark. If a test
set changes between submissions, reported numbers are not comparable and the
experiment is worthless. So a split is:

* **frozen** - once written, its content hash is recorded; any later change is a
  different split and must be reported as such;
* **disjoint** - a sample id may appear in at most one of train/validation/test;
* **deterministic** - the same inputs always produce the same hash, so two
  researchers can confirm they used identical splits without exchanging data.

Leakage is the specific failure this module exists to prevent. A manipulated frame
extracted from a source video, present in both train and test, inflates scores and
hides real generalization gaps. Note that id-level disjointness is *necessary but
not sufficient*: near-duplicate content across source recordings must also be
grouped by source video before splitting. That grouping is out of scope here but is
recorded in the split's ``rationale`` field.

Named ``split_schema.py`` rather than ``splits.py`` because ``datasets/splits/``
already exists as the directory holding these definitions.
"""

from __future__ import annotations

import hashlib
import json
import random as _random
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datasets.schema import ManifestValidationError, SplitName, as_json

__all__ = [
    "SPLIT_SCHEMA_VERSION",
    "FrozenSplit",
    "SplitValidationError",
    "SplitValidationReport",
    "assign_splits",
    "validate_split_id_sets",
]

SPLIT_SCHEMA_VERSION = 1


class SplitValidationError(ManifestValidationError):
    """Raised when a split definition is structurally invalid."""


def _clean_ids(values: Any, field_name: str) -> tuple[str, ...]:
    """Validate a collection of sample ids and return them as a sorted tuple.

    Sorting is what makes :meth:`FrozenSplit.content_hash` order-independent: the
    hash must reflect *which* samples are in a split, not the order someone happened
    to list them.
    """
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise SplitValidationError(
            f"{field_name} must be a list of ids, not a {type(values).__name__}"
        )
    if not isinstance(values, Iterable):
        raise SplitValidationError(f"{field_name} must be an iterable of ids")

    ids: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise SplitValidationError(f"{field_name} contains a non-string id: {value!r}")
        stripped = value.strip()
        if stripped in seen:
            duplicates.append(stripped)
        else:
            seen.add(stripped)
            ids.append(stripped)

    if duplicates:
        raise SplitValidationError(
            f"{field_name} contains {len(duplicates)} duplicate id(s)",
            errors=sorted(set(duplicates)),
        )
    return tuple(sorted(ids))


def validate_split_id_sets(
    train_ids: Iterable[str],
    validation_ids: Iterable[str],
    test_ids: Iterable[str],
) -> None:
    """Assert that the three split id sets are mutually disjoint.

    Raises:
        SplitValidationError: If any id appears in more than one set. The message
            names each overlapping id and every set it appears in, so the leakage is
            identifiable without re-deriving the sets.
    """
    sets: dict[str, set[str]] = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids),
    }
    names = list(sets)
    conflicts: dict[str, list[str]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for overlap in sorted(sets[left] & sets[right]):
                conflicts.setdefault(overlap, []).extend((left, right))

    if conflicts:
        details = [
            f"{sample_id} appears in {sorted(set(groups))}"
            for sample_id, groups in sorted(conflicts.items())
        ]
        raise SplitValidationError(
            f"split leakage: {len(conflicts)} id(s) appear in more than one split",
            errors=details,
        )


@dataclass(frozen=True, slots=True)
class SplitValidationReport:
    """Result of a non-raising split validation pass."""

    train_count: int
    validation_count: int
    test_count: int
    overlaps: dict[str, list[str]]
    duplicate_ids: dict[str, list[str]]

    @property
    def is_valid(self) -> bool:
        return not self.overlaps and not self.duplicate_ids

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [
            (
                f"{status}: train={self.train_count} validation={self.validation_count} "
                f"test={self.test_count}"
            )
        ]
        if self.overlaps:
            parts.append(f"leaked_ids={len(self.overlaps)}")
        if self.duplicate_ids:
            parts.append(f"duplicate_ids={len(self.duplicate_ids)}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "test_count": self.test_count,
            "overlaps": {k: sorted(set(v)) for k, v in sorted(self.overlaps.items())},
            "duplicate_ids": {k: sorted(set(v)) for k, v in sorted(self.duplicate_ids.items())},
            "is_valid": self.is_valid,
        }


def _collect_issues(raw: dict[str, tuple[str, ...]]) -> SplitValidationReport:
    """Build a report from already-normalized id tuples, without raising."""
    sets = {
        "train": set(raw["train"]),
        "validation": set(raw["validation"]),
        "test": set(raw["test"]),
    }
    names = list(sets)
    overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for overlap in sorted(sets[left] & sets[right]):
                overlaps.setdefault(overlap, []).extend((left, right))
    return SplitValidationReport(
        train_count=len(raw["train"]),
        validation_count=len(raw["validation"]),
        test_count=len(raw["test"]),
        overlaps=overlaps,
        duplicate_ids={},  # duplicates are raised during normalization
    )


@dataclass(frozen=True, slots=True)
class FrozenSplit:
    """An immutable, frozen train/validation/test partition for one dataset.

    Attributes:
        dataset_name: Dataset the partition applies to.
        train_ids: Sample ids used for training, sorted.
        validation_ids: Sample ids used for model selection, sorted.
        test_ids: Sample ids held out for final evaluation, sorted. Never consulted
            during training or hyperparameter selection.
        created_at: RFC 3339 timestamp of definition creation.
        seed: Seed used to produce the partition, for provenance.
        source_manifest_hash: ``sha256:<hex>`` hash of the manifest this split was
            derived from, tying the split to the exact data definition it covers.
        rationale: Documented justification. Required in practice: an unexplained
            split is not reproducible science. A prominent note here should record
            whether near-duplicate grouping was performed.
        split_schema_version: Schema version, for forward compatibility.
    """

    dataset_name: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    created_at: str
    seed: int
    source_manifest_hash: str
    rationale: str = ""
    split_schema_version: int = SPLIT_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize ids and enforce disjointness at construction time.

        Validation happens here, not at read time, so an invalid split object
        cannot exist in memory.
        """
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise SplitValidationError("dataset_name must be a non-empty string")
        object.__setattr__(self, "dataset_name", self.dataset_name.strip())

        train = _clean_ids(self.train_ids, "train_ids")
        validation = _clean_ids(self.validation_ids, "validation_ids")
        test = _clean_ids(self.test_ids, "test_ids")
        validate_split_id_sets(train, validation, test)

        object.__setattr__(self, "train_ids", train)
        object.__setattr__(self, "validation_ids", validation)
        object.__setattr__(self, "test_ids", test)

        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise SplitValidationError("seed must be an integer")
        if not isinstance(self.source_manifest_hash, str) or not self.source_manifest_hash.strip():
            raise SplitValidationError("source_manifest_hash must be a non-empty string")
        if not self.source_manifest_hash.strip().lower().startswith("sha256:"):
            raise SplitValidationError(
                "source_manifest_hash must be prefixed with 'sha256:', "
                f"got {self.source_manifest_hash!r}"
            )
        object.__setattr__(self, "source_manifest_hash", self.source_manifest_hash.strip())
        object.__setattr__(self, "created_at", str(self.created_at).strip())

    # --- derived ----------------------------------------------------------

    @property
    def total_ids(self) -> int:
        """Total number of samples across all three splits."""
        return len(self.train_ids) + len(self.validation_ids) + len(self.test_ids)

    @property
    def is_empty(self) -> bool:
        """``True`` when no samples are assigned. Useful as a guard in tests."""
        return self.total_ids == 0

    def ratios(self) -> dict[str, float]:
        """Fraction of assigned samples in each split.

        Returns all-zero values for an empty split rather than dividing by zero.
        """
        total = self.total_ids
        if total == 0:
            return {"train": 0.0, "validation": 0.0, "test": 0.0}
        return {
            "train": len(self.train_ids) / total,
            "validation": len(self.validation_ids) / total,
            "test": len(self.test_ids) / total,
        }

    def ids_for(self, split_name: SplitName | str) -> tuple[str, ...]:
        """Return the ids for one split.

        Raises:
            SplitValidationError: If ``split_name`` is not a known split.
        """
        resolved = SplitName.parse(split_name)
        if resolved is SplitName.TRAIN:
            return self.train_ids
        if resolved is SplitName.VALIDATION:
            return self.validation_ids
        return self.test_ids

    def validate(self) -> SplitValidationReport:
        """Re-run validation and return a report instead of raising."""
        return _collect_issues(
            {
                "train": self.train_ids,
                "validation": self.validation_ids,
                "test": self.test_ids,
            }
        )

    # --- hashing ----------------------------------------------------------

    def content_hash(self) -> str:
        """Deterministic hash of the split *content*.

        Computed over canonical JSON of the three sorted id sets plus
        ``dataset_name``. Order-independent, and stable across Python versions
        because the serialization is explicitly canonical.

        Intentionally **excludes** ``created_at`` and ``rationale``: editing the
        prose justification of a split must not invalidate its identity, or
        documentation edits would silently invalidate every run pinned to it.

        Returns:
            ``"sha256:<hex>"``.
        """
        payload = {
            "dataset_name": self.dataset_name,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
        }
        return f"sha256:{hashlib.sha256(as_json(payload).encode('utf-8')).hexdigest()}"

    # --- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict form."""
        return {
            "dataset_name": self.dataset_name,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "created_at": self.created_at,
            "seed": self.seed,
            "source_manifest_hash": self.source_manifest_hash,
            "rationale": self.rationale,
            "split_schema_version": self.split_schema_version,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FrozenSplit:
        """Build a split from a parsed definition.

        Raises:
            SplitValidationError: If any field is missing or validation fails.
        """
        if not isinstance(raw, dict):
            raise SplitValidationError(
                f"split definition must be an object, got {type(raw).__name__}"
            )
        known = {
            "dataset_name",
            "train_ids",
            "validation_ids",
            "test_ids",
            "created_at",
            "seed",
            "source_manifest_hash",
            "rationale",
            "split_schema_version",
            "extra",
        }
        missing = sorted(known - set(raw) - {"rationale", "split_schema_version", "extra"})
        if missing:
            raise SplitValidationError(
                f"split definition missing required field(s): {', '.join(missing)}",
                errors=missing,
            )
        return cls(**{key: raw[key] for key in known if key in raw})

    @classmethod
    def create(
        cls,
        dataset_name: str,
        *,
        train_ids: Iterable[str] = (),
        validation_ids: Iterable[str] = (),
        test_ids: Iterable[str] = (),
        seed: int,
        source_manifest_hash: str,
        rationale: str = "",
        created_at: str | None = None,
    ) -> FrozenSplit:
        """Construct a split, stamping ``created_at`` when not supplied."""
        return cls(
            dataset_name=dataset_name,
            train_ids=tuple(train_ids),
            validation_ids=tuple(validation_ids),
            test_ids=tuple(test_ids),
            created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
            seed=seed,
            source_manifest_hash=source_manifest_hash,
            rationale=rationale,
        )


def assign_splits(
    sample_ids: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, tuple[str, ...]]:
    """Partition sample ids into train/validation/test deterministically.

    Uses Python's :class:`random.Random` rather than NumPy so the partitioning is
    stable across NumPy versions and requires no deep-learning stack.

    Fractions are of the total and must sum to at most 1.0; the remainder goes to
    test, so omitting ``test_fraction`` is a valid and common choice.

    Determinism caveat
    ------------------
    This is a *random* partition. It does **not** group near-duplicates by source
    video, so leakage across source recordings remains possible. For real
    experiments, prefer an explicit, documented, group-aware partition. This helper
    exists for smoke tests and for datasets known to be duplicate-free.

    Args:
        sample_ids: Sample identifiers. Duplicates are rejected.
        seed: Partition seed.
        train_fraction: Target fraction for training.
        validation_fraction: Target fraction for validation.

    Returns:
        Mapping of split name to sorted id tuple.

    Raises:
        SplitValidationError: On duplicate ids or invalid fractions.
    """
    if train_fraction < 0 or validation_fraction < 0:
        raise SplitValidationError("fractions must be non-negative")
    if train_fraction + validation_fraction > 1.0:
        raise SplitValidationError(
            f"train_fraction + validation_fraction must be <= 1.0, got "
            f"{train_fraction + validation_fraction}"
        )

    ids = _clean_ids(sample_ids, "sample_ids")
    if not ids:
        return {"train": (), "validation": (), "test": ()}

    shuffled = list(ids)
    _random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_cut = int(total * train_fraction)
    validation_cut = train_cut + int(total * validation_fraction)

    partition = {
        "train": tuple(sorted(shuffled[:train_cut])),
        "validation": tuple(sorted(shuffled[train_cut:validation_cut])),
        "test": tuple(sorted(shuffled[validation_cut:])),
    }
    validate_split_id_sets(partition["train"], partition["validation"], partition["test"])
    return partition


def write_split_definition(path: Path | str, split: FrozenSplit) -> str:
    """Write a split definition as pretty JSON, returning the file's content hash.

    The returned value is the SHA-256 of the serialized bytes, which differs from
    :meth:`FrozenSplit.content_hash` (content-level). Both are useful: the former
    pins the exact file, the latter pins the semantics.

    Args:
        path: Destination file. Parent directories are created.
        split: The validated split to serialize.

    Returns:
        ``"sha256:<hex>"`` digest of the written bytes.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(split.to_dict(), indent=2, sort_keys=True) + "\n"
    destination.write_text(payload, encoding="utf-8", newline="\n")
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def load_split_definition(path: Path | str) -> FrozenSplit:
    """Load and validate a split definition from JSON.

    Args:
        path: JSON file containing a split definition.

    Returns:
        A validated :class:`FrozenSplit`.

    Raises:
        SplitValidationError: If the file is missing, malformed, or invalid.
    """
    source = Path(path)
    if not source.is_file():
        raise SplitValidationError(f"split definition not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SplitValidationError(f"malformed JSON at {source}: {exc.msg}") from exc
    return FrozenSplit.from_dict(raw)
