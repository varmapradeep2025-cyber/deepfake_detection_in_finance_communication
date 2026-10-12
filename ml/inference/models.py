"""Model descriptors, outputs, and the registry that finds and verifies them.

Why this is a registry and not a hard-coded import
--------------------------------------------------
Every detector in this project is an optional, versioned artifact: a set of weights
plus a small description of what they are and how to load them. Baking that into an
import would mean the codebase cannot describe a model it does not have — which is
exactly the state of a foundation-stage checkout. So the models live in a directory
tree, each described by a ``model.json`` "model card", and this module scans that
tree, verifies what it can, and reports honestly when nothing is there.

The honest-loading rule
-----------------------
:func:`load_model_backend` never fabricates a model. It checks, in order, that the
weights exist, that their digest matches the recorded one, that the declared
framework is importable, and that a declared entry point resolves — and if any step
fails it raises :class:`ModelUnavailableError` with a ``reason`` that names the step.
A caller turns that into ``analysis_complete`` or ``model_unavailable``; it never
turns a failure into a score.

Weights are hashed with :func:`datasets.manifest.calculate_file_sha256`, the single
file-identity implementation in the project, rather than a second one here.

Directory convention
--------------------
::

    <artifacts>/models/
        <model_id>/
            model.json        # the model card read here
            <weights file>    # named by the card's "weights_file"

The ``<artifacts>`` root is the ``paths.generated.artifacts`` path from
``configs/project.yaml``; this module takes the resolved ``models`` directory so it
has no opinion about where artifacts live.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from datasets.manifest import DEFAULT_CHUNK_SIZE, calculate_file_sha256
from datasets.schema import Modality

__all__ = [
    "ModelBackend",
    "ModelDescriptor",
    "ModelDescriptorError",
    "ModelOutput",
    "ModelRegistry",
    "ModelUnavailableError",
    "load_model_backend",
]

#: The per-model card file name.
MODEL_CARD_FILENAME = "model.json"


class ModelDescriptorError(ValueError):
    """Raised when a model card cannot be parsed into a :class:`ModelDescriptor`."""


class ModelUnavailableError(RuntimeError):
    """Raised when a model cannot be turned into a usable backend.

    Attributes:
        reason: Machine-readable cause, one of ``weights_missing``,
            ``weights_digest_mismatch``, ``dependency_missing``, ``no_entrypoint``,
            ``entrypoint_malformed``, ``entrypoint_missing``,
            ``entrypoint_attribute_missing`` or ``load_failed``.
        model_id: The model the failure concerns, when known.
    """

    def __init__(self, message: str, *, reason: str, model_id: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.model_id = model_id


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """A model card: what a detector is and how to load it.

    Attributes:
        model_id: Directory name and primary identifier, e.g. ``"resnet50-ffpp"``.
        version: Version string, e.g. ``"1.0.0"``.
        modality: The media type the model consumes.
        framework: Importable module the model runs on, e.g. ``"torch"``. Checked for
            importability before loading so a missing dependency is reported as such.
        entrypoint: ``"module.path:factory"`` naming a callable that builds the
            backend. ``None`` is allowed and means "described but not loadable here",
            which is reported as ``no_entrypoint`` rather than guessed.
        weights_file: Weights path relative to the model directory, or ``None`` for a
            model with no separate weights (e.g. a bundled checkpoint).
        weights_sha256: Expected lowercase hex digest of the weights file, or
            ``None`` when the card does not pin it. When present it is verified.
        input_size: Expected ``(height, width)`` in pixels, or ``None``.
        color_space: Channel order the model expects, e.g. ``"RGB"``.
        score_name: Name of the score the model emits.
        score_kind: What the score is, e.g. ``"uncalibrated_score"`` or
            ``"probability"``. Carried onto every output so consumers cannot mistake
            an uncalibrated number for a probability.
        score_semantics: Prose describing what a higher score means.
        decision_threshold: Operating point on the score, or ``None``.
        calibrated: Whether the score may be read as a probability.
        labels: Class labels, in the model's own output order, when stated.
        description: Free-text description of the model and its training data.
    """

    model_id: str
    version: str
    modality: Modality
    framework: str
    entrypoint: str | None = None
    weights_file: str | None = None
    weights_sha256: str | None = None
    input_size: tuple[int, int] | None = None
    color_space: str = "RGB"
    score_name: str = "manipulation_score"
    score_kind: str = "uncalibrated_score"
    score_semantics: str = ""
    decision_threshold: float | None = None
    calibrated: bool = False
    labels: tuple[str, ...] = ()
    description: str = ""

    @property
    def qualified_id(self) -> str:
        """``"<model_id>:<version>"``, the stable identifier for a loaded model."""
        return f"{self.model_id}:{self.version}"

    def model_dir(self, models_dir: Path) -> Path:
        """Directory holding this model's card and weights."""
        return models_dir / self.model_id

    def weights_path(self, models_dir: Path) -> Path | None:
        """Resolved weights path, or ``None`` when the card names no weights file."""
        if self.weights_file is None:
            return None
        return self.model_dir(models_dir) / self.weights_file

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelDescriptor:
        """Build a descriptor from a parsed model card.

        Raises:
            ModelDescriptorError: If a required field is missing or malformed.
        """
        if not isinstance(payload, Mapping):
            raise ModelDescriptorError(
                f"model card root must be a JSON object, got {type(payload).__name__}"
            )

        def require_str(key: str) -> str:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ModelDescriptorError(f"model card field {key!r} must be a non-empty string")
            return value.strip()

        def optional_str(key: str) -> str | None:
            value = payload.get(key)
            if value is None:
                return None
            if not isinstance(value, str):
                raise ModelDescriptorError(f"model card field {key!r} must be a string or null")
            return value

        model_id = require_str("model_id")
        version = require_str("version")
        framework = require_str("framework")

        raw_modality = payload.get("modality")
        if not isinstance(raw_modality, str):
            raise ModelDescriptorError(f"model {model_id!r} is missing a modality string")
        try:
            modality = Modality.parse(raw_modality)
        except Exception as exc:  # pragma: no cover - Modality.parse raises a domain error
            raise ModelDescriptorError(f"model {model_id!r}: {exc}") from exc

        raw_size = payload.get("input_size")
        input_size: tuple[int, int] | None = None
        if raw_size is not None:
            valid_size = (
                isinstance(raw_size, (list, tuple))
                and len(raw_size) == 2
                and all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in raw_size
                )
            )
            if not valid_size:
                raise ModelDescriptorError(
                    f"model {model_id!r}: input_size must be [height, width] integers"
                )
            input_size = (int(raw_size[0]), int(raw_size[1]))

        raw_threshold = payload.get("decision_threshold")
        decision_threshold: float | None = None
        if raw_threshold is not None:
            if not isinstance(raw_threshold, (int, float)) or isinstance(raw_threshold, bool):
                raise ModelDescriptorError(
                    f"model {model_id!r}: decision_threshold must be a number or null"
                )
            decision_threshold = float(raw_threshold)

        raw_labels = payload.get("labels", [])
        if not isinstance(raw_labels, (list, tuple)) or not all(
            isinstance(label, str) for label in raw_labels
        ):
            raise ModelDescriptorError(f"model {model_id!r}: labels must be a list of strings")

        raw_calibrated = payload.get("calibrated", False)
        if not isinstance(raw_calibrated, bool):
            raise ModelDescriptorError(
                f"model {model_id!r}: calibrated must be a boolean"
            )

        return cls(
            model_id=model_id,
            version=version,
            modality=modality,
            framework=framework,
            entrypoint=optional_str("entrypoint"),
            weights_file=optional_str("weights_file"),
            weights_sha256=optional_str("weights_sha256"),
            input_size=input_size,
            color_space=optional_str("color_space") or "RGB",
            score_name=optional_str("score_name") or "manipulation_score",
            score_kind=optional_str("score_kind") or "uncalibrated_score",
            score_semantics=optional_str("score_semantics") or "",
            decision_threshold=decision_threshold,
            calibrated=raw_calibrated,
            labels=tuple(raw_labels),
            description=optional_str("description") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "modality": self.modality.value,
            "framework": self.framework,
            "entrypoint": self.entrypoint,
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "input_size": list(self.input_size) if self.input_size else None,
            "color_space": self.color_space,
            "score_name": self.score_name,
            "score_kind": self.score_kind,
            "score_semantics": self.score_semantics,
            "decision_threshold": self.decision_threshold,
            "calibrated": self.calibrated,
            "labels": list(self.labels),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """What a backend produced for one input.

    The score's semantics travel with it. ``confidence`` is separate from
    ``raw_score`` and is ``None`` unless the model honestly provides one: an
    uncalibrated score is not a confidence, and copying one into the other is the
    failure this record exists to prevent.

    Attributes:
        model_id, version: Identify the model that produced the output.
        score_name: Name of the score.
        raw_score: The number the model emitted, or ``None`` if it emitted none.
        score_kind: What the score is, carried from the descriptor.
        score_semantics: Prose describing what a higher score means.
        calibrated: Whether the score may be read as a probability.
        decision_threshold: The model's operating point, if stated.
        confidence: ``0..1`` when the model provides a real confidence, else ``None``.
        labels: Output class labels, when the model states them.
        extra: Any additional named outputs, recorded verbatim.
        limitations: Known limitations of this output, surfaced to the report.
    """

    model_id: str
    version: str
    score_name: str
    raw_score: float | None
    score_kind: str
    score_semantics: str = ""
    calibrated: bool = False
    decision_threshold: float | None = None
    confidence: float | None = None
    labels: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"model confidence must be within [0, 1] or None, got {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "score_name": self.score_name,
            "raw_score": self.raw_score,
            "score_kind": self.score_kind,
            "score_semantics": self.score_semantics,
            "calibrated": self.calibrated,
            "decision_threshold": self.decision_threshold,
            "confidence": self.confidence,
            "labels": list(self.labels),
            "extra": dict(self.extra),
            "limitations": list(self.limitations),
        }


@runtime_checkable
class ModelBackend(Protocol):
    """What an analyzer needs from a loaded model.

    Deliberately tiny: describe what you are, and score one preprocessed sample. A
    backend holds no request state, so it can be reused across samples.
    """

    #: The card this backend was loaded from.
    descriptor: ModelDescriptor

    def predict(self, sample: Any) -> ModelOutput:
        """Score one preprocessed sample.

        Args:
            sample: Modality-specific preprocessed input produced by an analyzer.

        Returns:
            The model's output. Implementations must not invent a score they did not
            compute.

        Raises:
            Exception: On any failure; the caller records the run as failed rather
                than producing a score.
        """
        ...


class ModelRegistry:
    """The models discovered under one artifacts directory.

    Built with :meth:`discover`, which never raises for a missing or empty directory
    — "no models installed" is a normal, reported state, not an error. Cards that
    cannot be parsed are collected in :attr:`problems` rather than aborting the scan,
    so one bad card does not hide the good ones.

    Attributes:
        models_dir: The directory that was scanned.
        descriptors: The parsed model cards, ordered by ``model_id``.
        problems: Human-readable descriptions of cards that were skipped.
    """

    def __init__(
        self,
        models_dir: Path,
        descriptors: tuple[ModelDescriptor, ...] = (),
        problems: tuple[str, ...] = (),
    ) -> None:
        self.models_dir = models_dir
        self.descriptors = descriptors
        self.problems = problems
        self._by_modality: dict[Modality, tuple[ModelDescriptor, ...]] = {}
        for modality in Modality:
            self._by_modality[modality] = tuple(
                descriptor for descriptor in descriptors if descriptor.modality is modality
            )

    @classmethod
    def discover(cls, models_dir: Path | str) -> ModelRegistry:
        """Scan ``models_dir`` for ``*/model.json`` cards.

        A missing directory yields an empty registry. A card that fails to parse is
        recorded in ``problems`` and skipped.
        """
        root = Path(models_dir)
        descriptors: list[ModelDescriptor] = []
        problems: list[str] = []

        if root.is_dir():
            for child in sorted(p for p in root.iterdir() if p.is_dir()):
                card = child / MODEL_CARD_FILENAME
                if not card.is_file():
                    continue
                try:
                    payload = json.loads(card.read_text(encoding="utf-8"))
                    descriptors.append(ModelDescriptor.from_dict(payload))
                except (OSError, json.JSONDecodeError, ModelDescriptorError) as exc:
                    problems.append(f"{card}: {exc}")

        descriptors.sort(key=lambda descriptor: descriptor.model_id)
        return cls(root, tuple(descriptors), tuple(problems))

    @property
    def is_empty(self) -> bool:
        """Whether no model cards were found."""
        return not self.descriptors

    def get(self, model_id: str) -> ModelDescriptor | None:
        """Return the descriptor with this ``model_id``, or ``None``."""
        for descriptor in self.descriptors:
            if descriptor.model_id == model_id:
                return descriptor
        return None

    def for_modality(self, modality: Modality | str) -> tuple[ModelDescriptor, ...]:
        """Return every descriptor for a modality, in registry order."""
        resolved = Modality.parse(modality)
        return self._by_modality.get(resolved, ())

    def __len__(self) -> int:
        return len(self.descriptors)


def load_model_backend(descriptor: ModelDescriptor, *, models_dir: Path) -> ModelBackend:
    """Load the backend described by ``descriptor``, or explain why it cannot be.

    Performs the checks in the order that yields the most actionable reason: weights
    presence, weights digest, framework importability, entry-point resolution. Any
    failure raises :class:`ModelUnavailableError` rather than returning a stub.

    Args:
        descriptor: The model card to load.
        models_dir: Directory the card was discovered under.

    Raises:
        ModelUnavailableError: With a ``reason`` naming the failed check.
    """
    model_dir = descriptor.model_dir(models_dir)

    weights = descriptor.weights_path(models_dir)
    if weights is not None:
        if not weights.is_file():
            raise ModelUnavailableError(
                f"model {descriptor.qualified_id!r} declares weights {descriptor.weights_file!r} "
                f"but {weights} is not present",
                reason="weights_missing",
                model_id=descriptor.model_id,
            )
        if descriptor.weights_sha256 is not None:
            actual = calculate_file_sha256(weights, chunk_size=DEFAULT_CHUNK_SIZE)
            if actual.lower() != descriptor.weights_sha256.lower():
                raise ModelUnavailableError(
                    f"model {descriptor.qualified_id!r} weights digest mismatch: the card "
                    f"records {descriptor.weights_sha256} but {weights} hashes to {actual}",
                    reason="weights_digest_mismatch",
                    model_id=descriptor.model_id,
                )

    try:
        importlib.import_module(descriptor.framework)
    except ImportError as exc:
        raise ModelUnavailableError(
            f"model {descriptor.qualified_id!r} needs framework {descriptor.framework!r}, "
            f"which is not installed: {exc}",
            reason="dependency_missing",
            model_id=descriptor.model_id,
        ) from exc

    if descriptor.entrypoint is None:
        raise ModelUnavailableError(
            f"model {descriptor.qualified_id!r} declares no entrypoint, so no backend can be "
            "constructed from it",
            reason="no_entrypoint",
            model_id=descriptor.model_id,
        )

    module_name, separator, attribute = descriptor.entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise ModelUnavailableError(
            f"model {descriptor.qualified_id!r} entrypoint {descriptor.entrypoint!r} is not "
            "of the form 'module.path:factory'",
            reason="entrypoint_malformed",
            model_id=descriptor.model_id,
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelUnavailableError(
            f"model {descriptor.qualified_id!r} entrypoint module {module_name!r} could not be "
            f"imported: {exc}",
            reason="entrypoint_missing",
            model_id=descriptor.model_id,
        ) from exc

    factory = getattr(module, attribute, None)
    if factory is None:
        raise ModelUnavailableError(
            f"model {descriptor.qualified_id!r} entrypoint module {module_name!r} has no "
            f"attribute {attribute!r}",
            reason="entrypoint_attribute_missing",
            model_id=descriptor.model_id,
        )

    try:
        backend = factory(descriptor=descriptor, model_dir=model_dir, weights_path=weights)
    except Exception as exc:
        raise ModelUnavailableError(
            f"model {descriptor.qualified_id!r} entrypoint raised while constructing the "
            f"backend: {exc}",
            reason="load_failed",
            model_id=descriptor.model_id,
        ) from exc

    if not isinstance(backend, ModelBackend):
        raise ModelUnavailableError(
            f"model {descriptor.qualified_id!r} entrypoint returned {type(backend).__name__}, "
            "which does not satisfy the ModelBackend protocol",
            reason="load_failed",
            model_id=descriptor.model_id,
        )

    return backend
