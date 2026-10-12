"""Experiment configuration schema.

An experiment configuration is the complete, self-contained declaration of one
research condition: what was trained, on what, under which seed, and at which
point in the model lifecycle. It must be serializable before a run starts, because
a run whose configuration cannot be reconstructed afterwards is not reproducible.

Four experiment families are modelled explicitly, rather than as free-form strings,
because each carries different obligations:

===============================  ==========================================
Family                          Additional obligation
===============================  ==========================================
:attr:`ExperimentType.KNOWN_DATASET`  Baseline: train and test on one dataset.
:attr:`ExperimentType.CROSS_DATASET`  Requires ``target_dataset`` and only
                                     permits a held-out, non-training eval set.
:attr:`ExperimentType.ABLATION`       Requires ``ablation_of`` and a description
                                     of what was removed or changed.
:attr:`ExperimentType.MULTIMODAL`     Requires per-modality dataset mapping.
===============================  ==========================================

Metric computation lives in :mod:`experiments.results`; no model or metric is
implemented here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from datasets.schema import Modality, as_json

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentConfig",
    "ExperimentType",
    "ExperimentValidationError",
    "validate_experiment_config",
]

EXPERIMENT_SCHEMA_VERSION = 1

# Identifiers double as directory names under experiments/runs/, so the character set
# is restricted to be path-safe and portable across Windows, Linux, and macOS.
# Minimum length is 2, which accommodates terse but legitimate names such as "v0".
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")


class ExperimentValidationError(ValueError):
    """Raised when an experiment configuration is invalid.

    Carries the full list of problems so a malformed config can be corrected in one
    pass rather than one exception at a time. ``str(error)`` includes the list, so a
    traceback alone is enough to diagnose the failure.
    """

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        if self.errors:
            message = f"{message}: " + "; ".join(self.errors)
        super().__init__(message)


class ExperimentType(str, Enum):
    """Research design of an experiment.

    Modelled as a closed enum because these imply different validation rules. An
    unrecognised design should fail loudly at configuration time, not silently
    produce numbers that are not comparable to anything.
    """

    KNOWN_DATASET = "known_dataset"
    CROSS_DATASET = "cross_dataset"
    ABLATION = "ablation"
    MULTIMODAL = "multimodal"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def parse(cls, value: Any) -> ExperimentType:
        """Coerce ``value`` to an :class:`ExperimentType`, raising on anything else."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            candidate = value.strip().lower().replace("-", "_")
            for member in cls:
                if member.value == candidate:
                    return member
        raise ExperimentValidationError(
            f"invalid experiment_type {value!r}; expected one of {', '.join(cls.values())}"
        )


def _identifier_problem(value: Any, field_name: str) -> str | None:
    """Return an error message if ``value`` is not a safe slug, else ``None``.

    Returning rather than raising lets :meth:`ExperimentConfig.__post_init__`
    accumulate every problem in one pass, consistent with the other checks.
    """
    if not isinstance(value, str) or not value.strip():
        return f"{field_name} must be a non-empty string, got {value!r}"
    stripped = value.strip()
    if not _ID_PATTERN.match(stripped):
        return f"{field_name} must match {_ID_PATTERN.pattern} (lowercase slug), got {stripped!r}"
    return None


def _require_identifier(value: Any, field_name: str) -> str:
    """Validate a slug and return the normalized value. Raises on failure."""
    problem = _identifier_problem(value, field_name)
    if problem:
        raise ExperimentValidationError(problem)
    return value.strip()


def _int_problem(value: Any, field_name: str, *, minimum: int = 0) -> str | None:
    """Return an error message if ``value`` is not an integer within range."""
    if isinstance(value, bool) or not isinstance(value, int):
        return f"{field_name} must be an integer, got {type(value).__name__}"
    if value < minimum:
        return f"{field_name} must be >= {minimum}, got {value}"
    return None


def _require_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    problem = _int_problem(value, field_name, minimum=minimum)
    if problem:
        raise ExperimentValidationError(problem)
    return value


def _require_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ExperimentValidationError(
            f"{field_name} must be a non-empty string when provided, got {value!r}"
        )
    return value.strip()


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Immutable declaration of one experimental condition.

    Attributes:
        experiment_id: Unique slug identifying the experiment across the project.
            Becomes the run directory name, so it must be path-safe.
        name: Human-readable short title.
        description: What this experiment tests and why. Required: an undocumented
            experiment cannot be interpreted or reproduced by anyone else.
        modality: Primary modality under test. See :class:`~datasets.schema.Modality`.
        experiment_type: Research design. Drives family-specific validation.
        model_version: Version of the model or pipeline under test, e.g. ``"v0"``.
        training_dataset: Dataset the model is fit on.
        validation_dataset: Dataset used for model selection.
        test_dataset: Dataset used for final reporting. Must differ from the
            training dataset for cross-dataset designs, which is enforced below.
        seed: Run seed, derived from the project base seed via
            :func:`ml.reproducibility.seeds.derive_seed`.
        run_number: Zero-based index of this run within the seed set.
        config_hash: ``sha256:<hex>`` digest pinning the configuration content.
        created_at: RFC 3339 creation timestamp.
        target_dataset: For cross-dataset designs, the unseen evaluation dataset.
        ablation_of: For ablation designs, the ``experiment_id`` being ablated.
        ablation_description: What component was removed, disabled, or replaced.
        modality_datasets: For multimodal designs, modality name to dataset name.
        tags: Free-form labels for filtering result tables.
        splits_used: Split content hashes (``sha256:<hex>``) per dataset, pinning
            the exact partitions used.
        manifests_used: Manifest content hashes per dataset.
        config: Extra model or training hyperparameters, stored verbatim.
    """

    experiment_id: str
    name: str
    description: str
    modality: Modality
    model_version: str
    training_dataset: str
    validation_dataset: str
    test_dataset: str
    seed: int
    run_number: int
    config_hash: str
    created_at: str
    experiment_type: ExperimentType = ExperimentType.KNOWN_DATASET
    target_dataset: str | None = None
    ablation_of: str | None = None
    ablation_description: str | None = None
    modality_datasets: dict[str, str] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    splits_used: dict[str, str] = field(default_factory=dict)
    manifests_used: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    experiment_schema_version: int = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Normalize and validate. Failures raise immediately."""
        errors: list[str] = []

        if not isinstance(self.description, str) or not self.description.strip():
            errors.append("description must be a non-empty string")
        else:
            object.__setattr__(self, "description", self.description.strip())

        try:
            object.__setattr__(self, "modality", Modality.parse(self.modality))
        except Exception as exc:
            errors.append(str(exc))

        try:
            object.__setattr__(self, "experiment_type", ExperimentType.parse(self.experiment_type))
        except ExperimentValidationError as exc:
            errors.append(str(exc))

        for field_name in (
            "experiment_id",
            "name",
            "model_version",
            "training_dataset",
            "validation_dataset",
            "test_dataset",
        ):
            problem = _identifier_problem(getattr(self, field_name), field_name)
            if problem:
                errors.append(problem)
            else:
                object.__setattr__(self, field_name, getattr(self, field_name).strip())

        for field_name in ("seed", "run_number"):
            problem = _int_problem(getattr(self, field_name), field_name)
            if problem:
                errors.append(problem)

        hash_value = self.config_hash
        if not isinstance(hash_value, str) or not hash_value.strip().lower().startswith("sha256:"):
            errors.append("config_hash must be a string prefixed with 'sha256:'")
        else:
            object.__setattr__(self, "config_hash", hash_value.strip())

        object.__setattr__(self, "created_at", str(self.created_at).strip())

        # Family-specific obligations. Guarded so that a missing target_dataset does
        # not cascade into a misleading "self-target" complaint.
        if self.experiment_type is ExperimentType.CROSS_DATASET:
            target = _require_optional_str(self.target_dataset, "target_dataset")
            object.__setattr__(self, "target_dataset", target)
            if not target:
                errors.append("cross_dataset experiments require target_dataset")
            elif target == self.training_dataset:
                errors.append(
                    "cross_dataset experiments must set target_dataset different from "
                    "training_dataset; a self-target is a known_dataset experiment"
                )

        if self.experiment_type is ExperimentType.ABLATION:
            parent = _require_optional_str(self.ablation_of, "ablation_of")
            if not parent:
                errors.append("ablation experiments require ablation_of")
            elif parent == self.experiment_id:
                errors.append("ablation_of must not reference the experiment itself")
            object.__setattr__(self, "ablation_of", parent)
            note = _require_optional_str(self.ablation_description, "ablation_description")
            if not note:
                errors.append("ablation experiments require ablation_description")
            object.__setattr__(self, "ablation_description", note)

        if self.experiment_type is ExperimentType.MULTIMODAL:
            if not isinstance(self.modality_datasets, dict) or not self.modality_datasets:
                errors.append(
                    "multimodal experiments require a non-empty modality_datasets mapping"
                )
            else:
                for key, value in self.modality_datasets.items():
                    if not isinstance(key, str) or not key.strip():
                        errors.append("modality_datasets keys must be non-empty strings")
                        break
                    try:
                        Modality.parse(key)
                    except Exception as exc:
                        errors.append(
                            f"modality_datasets key {key!r} is not a valid modality: {exc}"
                        )
                        break
                    if not isinstance(value, str) or not value.strip():
                        errors.append(
                            f"modality_datasets[{key!r}] must be a non-empty dataset name"
                        )
                        break
        elif self.modality_datasets:
            errors.append("modality_datasets is only meaningful for multimodal experiments")

        if not isinstance(self.config, dict):
            errors.append("config must be a mapping")
        for mapping_name in ("splits_used", "manifests_used"):
            mapping = getattr(self, mapping_name)
            if not isinstance(mapping, dict):
                errors.append(f"{mapping_name} must be a mapping")
                continue
            for key, value in mapping.items():
                if not isinstance(value, str) or not value.strip().lower().startswith("sha256:"):
                    errors.append(f"{mapping_name}[{key!r}] must be a 'sha256:' digest string")
                    break

        if not isinstance(self.tags, (tuple, list)):
            errors.append("tags must be a list of strings")
        else:
            object.__setattr__(
                self, "tags", tuple(str(tag).strip() for tag in self.tags if str(tag).strip())
            )

        if errors:
            raise ExperimentValidationError(
                f"invalid experiment configuration ({len(errors)} problem(s))", errors=errors
            )

    # --- derived ----------------------------------------------------------

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """Every dataset referenced, de-duplicated and sorted."""
        names = {self.training_dataset, self.validation_dataset, self.test_dataset}
        if self.target_dataset:
            names.add(self.target_dataset)
        names.update(self.modality_datasets.values())
        return tuple(sorted(names))

    @property
    def is_cross_dataset(self) -> bool:
        return self.experiment_type is ExperimentType.CROSS_DATASET

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict form."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "experiment_type": self.experiment_type.value,
            "modality": self.modality.value,
            "model_version": self.model_version,
            "training_dataset": self.training_dataset,
            "validation_dataset": self.validation_dataset,
            "test_dataset": self.test_dataset,
            "target_dataset": self.target_dataset,
            "ablation_of": self.ablation_of,
            "ablation_description": self.ablation_description,
            "modality_datasets": dict(self.modality_datasets),
            "seed": self.seed,
            "run_number": self.run_number,
            "config_hash": self.config_hash,
            "created_at": self.created_at,
            "tags": list(self.tags),
            "splits_used": dict(self.splits_used),
            "manifests_used": dict(self.manifests_used),
            "config": dict(self.config),
            "experiment_schema_version": self.experiment_schema_version,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build a config from a parsed mapping.

        Raises:
            ExperimentValidationError: If a required field is missing or invalid.
        """
        if not isinstance(raw, dict):
            raise ExperimentValidationError(
                f"experiment config must be an object, got {type(raw).__name__}"
            )
        known = {f.name for f in fields(cls)}
        missing = sorted(
            name
            for name in known
            if name not in raw
            and name
            not in {
                "experiment_type",
                "target_dataset",
                "ablation_of",
                "ablation_description",
                "modality_datasets",
                "tags",
                "splits_used",
                "manifests_used",
                "config",
                "experiment_schema_version",
            }
        )
        if missing:
            raise ExperimentValidationError(
                f"experiment config missing required field(s): {', '.join(missing)}",
                errors=missing,
            )
        return cls(**{key: raw[key] for key in known if key in raw})

    def compute_config_hash(self) -> str:
        """Compute the content hash over the semantic configuration.

        Excludes ``config_hash`` itself and ``created_at``, so a config can be
        stamped and hashed in either order, and re-running the same condition yields
        an identical hash.

        Returns:
            ``"sha256:<hex>"``.
        """
        payload = self.to_dict()
        payload.pop("config_hash", None)
        payload.pop("created_at", None)
        return f"sha256:{hashlib.sha256(as_json(payload).encode('utf-8')).hexdigest()}"


def validate_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Validate a mapping and return the typed config.

    Raises:
        ExperimentValidationError: If validation fails.
    """
    return ExperimentConfig.from_dict(raw)


def create_experiment_config(
    *,
    experiment_id: str,
    name: str,
    description: str,
    modality: Modality | str,
    model_version: str,
    training_dataset: str,
    validation_dataset: str,
    test_dataset: str,
    seed: int,
    run_number: int,
    experiment_type: ExperimentType | str = ExperimentType.KNOWN_DATASET,
    created_at: str | None = None,
    **kwargs: Any,
) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig`, hashing the content automatically.

    Callers pass plain values and receive a validated config whose ``config_hash``
    already reflects its contents, so a config can never carry a stale hash.

    Returns:
        A validated :class:`ExperimentConfig`.
    """
    config = ExperimentConfig(
        experiment_id=experiment_id,
        name=name,
        description=description,
        experiment_type=ExperimentType.parse(experiment_type),
        modality=Modality.parse(modality),
        model_version=model_version,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        seed=seed,
        run_number=run_number,
        config_hash="sha256:pending",
        created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        **kwargs,
    )
    return ExperimentConfig(
        **{
            **config.to_dict(),
            "modality": config.modality.value,
            "experiment_type": config.experiment_type.value,
            "config_hash": config.compute_config_hash(),
        }
    )


def experiment_config_to_json(config: ExperimentConfig, *, indent: int | None = 2) -> str:
    """Serialize an experiment config to deterministic JSON."""
    return json.dumps(config.to_dict(), indent=indent, sort_keys=True)
