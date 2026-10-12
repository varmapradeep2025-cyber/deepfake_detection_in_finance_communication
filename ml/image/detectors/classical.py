"""Model backend for the classical (numpy) image baseline.

What this backend is
--------------------
The loader in :mod:`ml.inference.models` turns a model card into a backend by
importing its ``framework`` and calling its ``entrypoint``. For the classical
baseline the framework import is ``numpy`` and the entrypoint is
``ml.image.detectors.classical:build``. The backend computes the hand-crafted
features from :mod:`ml.image.features`, applies the scaler and linear coefficients
that were fitted during training, and returns one :class:`ModelOutput`.

Why the weights are plain JSON
------------------------------
The fitting step produces two arrays and two scalars. Serializing those as JSON
(rather than pickle or joblib) means an artifact is human-readable, diffable, and —
importantly — cannot execute arbitrary code when loaded. A model artifact is data,
and this format keeps it that way. scikit-learn is therefore needed to *train* the
baseline but never to *run* it.

Honesty about scores
--------------------
The number this backend emits is an uncalibrated decision score: higher means "more
likely manipulated", but it is not a probability and is never presented as one.
``calibrated`` is always ``False`` and ``confidence`` is always ``None``. The score's
meaning is defined entirely by the feature vector it consumes, so a vector computed
by a *different* extractor or config is rejected rather than scored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ml.features.contract import FeatureAvailability
from ml.image.features import (
    IMAGE_FEATURE_NAMES,
    ImageFeatureConfig,
    ImageFeatureExtractor,
)
from ml.image.training import FittedClassifier
from ml.inference.models import ModelDescriptor, ModelOutput

__all__ = [
    "CLASSICAL_WEIGHTS_SCHEMA_VERSION",
    "ClassicalImageBackend",
    "ClassicalWeights",
    "ClassicalWeightsError",
    "build",
    "load_classical_weights",
    "write_classical_weights",
]

#: Version of the on-disk weights layout.
CLASSICAL_WEIGHTS_SCHEMA_VERSION = 1


class ClassicalWeightsError(RuntimeError):
    """Raised when a classical weights file is missing, malformed, or inconsistent."""


@dataclass(frozen=True, slots=True)
class ClassicalWeights:
    """Everything needed to score with the classical baseline, as plain numbers.

    Attributes:
        feature_config: Parameters of the feature transform the coefficients expect.
        feature_names: Feature names in coefficient order.
        scaler_mean, scaler_scale: Standardization applied before the linear model.
        coefficients, intercept: Linear decision function on scaled features. Oriented
            so that a larger value means more likely to be :attr:`positive_class`.
        labels: Output class labels in canonical order.
        positive_class: The label a higher score supports.
        threshold: Decision threshold recorded at training time, or ``None``.
        training: Free-form provenance (dataset, split, seed) for traceability.
        created_at: ISO-8601 UTC timestamp of when the file was written.
        schema_version: Version of this layout.
    """

    feature_config: ImageFeatureConfig
    feature_names: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    labels: tuple[str, ...]
    positive_class: str
    threshold: float | None
    training: Mapping[str, Any]
    created_at: str
    schema_version: int = CLASSICAL_WEIGHTS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.feature_names != IMAGE_FEATURE_NAMES:
            raise ClassicalWeightsError(
                "feature_names do not match the current feature extractor; the weights "
                "were trained on a different feature layout"
            )
        for label, values in (
            ("scaler_mean", self.scaler_mean),
            ("scaler_scale", self.scaler_scale),
        ):
            if len(values) != len(self.coefficients):
                raise ClassicalWeightsError(
                    f"{label} has {len(values)} values for {len(self.coefficients)} coefficients"
                )
        if len(self.feature_names) != len(self.coefficients):
            raise ClassicalWeightsError(
                f"feature_names has {len(self.feature_names)} names for "
                f"{len(self.coefficients)} coefficients"
            )
        if not all(value > 0.0 for value in self.scaler_scale):
            raise ClassicalWeightsError("scaler_scale values must be positive")
        if self.positive_class not in self.labels:
            raise ClassicalWeightsError(
                f"positive_class {self.positive_class!r} is not among labels {self.labels!r}"
            )

    @property
    def feature_dim(self) -> int:
        """Number of features the weights expect."""
        return len(self.coefficients)

    def score(self, vector: npt.NDArray[np.float64]) -> float:
        """Score one feature vector, oriented so higher means :attr:`positive_class`."""
        mean = np.asarray(self.scaler_mean, dtype=np.float64)
        scale = np.asarray(self.scaler_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        scaled = (np.asarray(vector, dtype=np.float64) - mean) / scale
        return float(scaled @ coefficients + float(self.intercept))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "weights_schema_version": int(self.schema_version),
            "created_at": self.created_at,
            "feature_config": self.feature_config.to_dict(),
            "feature_names": list(self.feature_names),
            "scaler": {
                "mean": [float(v) for v in self.scaler_mean],
                "scale": [float(v) for v in self.scaler_scale],
            },
            "classifier": {
                "coefficients": [float(v) for v in self.coefficients],
                "intercept": float(self.intercept),
                "labels": list(self.labels),
                "positive_class": self.positive_class,
            },
            "threshold": self.threshold,
            "training": dict(self.training),
        }

    @classmethod
    def from_fitted(
        cls,
        fitted: FittedClassifier,
        *,
        feature_config: ImageFeatureConfig | None = None,
        threshold: float | None = None,
        training: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> ClassicalWeights:
        """Build weights from a fitted classifier.

        Coefficients are re-oriented so a larger score always means the positive
        class, regardless of the class order scikit-learn happened to use.
        """
        coefficients = fitted.coefficients
        intercept = float(fitted.intercept)
        if fitted.positive_index != 1:
            coefficients = tuple(-value for value in coefficients)
            intercept = -intercept
        return cls(
            feature_config=feature_config or ImageFeatureConfig(),
            feature_names=tuple(fitted.feature_names),
            scaler_mean=tuple(float(v) for v in fitted.scaler_mean),
            scaler_scale=tuple(float(v) for v in fitted.scaler_scale),
            coefficients=tuple(float(v) for v in coefficients),
            intercept=intercept,
            labels=tuple(fitted.classes),
            positive_class=str(fitted.config.positive_label),
            threshold=threshold,
            training=dict(training or {}),
            created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ClassicalWeights:
        """Build weights from parsed JSON.

        Raises:
            ClassicalWeightsError: If required fields are missing or inconsistent.
        """
        if not isinstance(raw, Mapping):
            raise ClassicalWeightsError(
                f"weights root must be a JSON object, got {type(raw).__name__}"
            )
        if int(raw.get("weights_schema_version", 0)) != CLASSICAL_WEIGHTS_SCHEMA_VERSION:
            found = raw.get("weights_schema_version")
            raise ClassicalWeightsError(
                f"unsupported classical weights schema version {found!r}; "
                f"expected {CLASSICAL_WEIGHTS_SCHEMA_VERSION}"
            )
        try:
            scaler = raw["scaler"]
            classifier = raw["classifier"]
            return ClassicalWeights(
                feature_config=ImageFeatureConfig.from_dict(raw["feature_config"]),
                feature_names=tuple(str(v) for v in raw["feature_names"]),
                scaler_mean=tuple(float(v) for v in scaler["mean"]),
                scaler_scale=tuple(float(v) for v in scaler["scale"]),
                coefficients=tuple(float(v) for v in classifier["coefficients"]),
                intercept=float(classifier["intercept"]),
                labels=tuple(str(v) for v in classifier["labels"]),
                positive_class=str(classifier["positive_class"]),
                threshold=(
                    None if raw.get("threshold") is None else float(raw["threshold"])
                ),
                training=dict(raw.get("training", {})),
                created_at=str(raw["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ClassicalWeightsError(f"malformed classical weights: {exc}") from exc


def write_classical_weights(weights: ClassicalWeights, path: Path | str) -> Path:
    """Write ``weights`` to ``path`` as UTF-8 JSON, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(weights.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_classical_weights(path: Path | str) -> ClassicalWeights:
    """Read a classical weights file.

    Raises:
        ClassicalWeightsError: If the file is missing or malformed.
    """
    source = Path(path)
    if not source.is_file():
        raise ClassicalWeightsError(f"classical weights file not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClassicalWeightsError(f"could not read {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ClassicalWeightsError(
            f"{source} must contain a JSON object, got {type(payload).__name__}"
        )
    return ClassicalWeights.from_dict(payload)


class ClassicalImageBackend:
    """A :class:`~ml.inference.models.ModelBackend` for the classical baseline.

    Args:
        descriptor: The model card that describes this backend.
        weights: The loaded, validated weights.
        extractor: Feature extractor; built from the weights' config when omitted.
    """

    def __init__(
        self,
        descriptor: ModelDescriptor,
        weights: ClassicalWeights,
        extractor: ImageFeatureExtractor | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._weights = weights
        self._extractor = extractor or ImageFeatureExtractor(weights.feature_config)

    @property
    def descriptor(self) -> ModelDescriptor:
        """The model card this backend was built from."""
        return self._descriptor

    @property
    def weights(self) -> ClassicalWeights:
        """The weights this backend scores with."""
        return self._weights

    def predict(self, sample: Any) -> ModelOutput:
        """Score one preprocessed image and return a :class:`ModelOutput`.

        Raises:
            ClassicalWeightsError: If the sample cannot be turned into a feature
                vector. The analyzer maps this to ``analysis_failed``; it is never
                turned into a score.
        """
        representation = self._extractor.extract_features(sample)
        if representation.availability is not FeatureAvailability.AVAILABLE:
            raise ClassicalWeightsError(
                "could not extract features for scoring: "
                f"{representation.failure_reason or representation.availability.value}"
            )
        if representation.feature_dim != self._weights.feature_dim:
            raise ClassicalWeightsError(
                f"extracted {representation.feature_dim} features but the model expects "
                f"{self._weights.feature_dim}"
            )
        names = tuple(representation.alignment_metadata.get("feature_names", ()))
        if names and names != self._weights.feature_names:
            raise ClassicalWeightsError(
                "feature names do not match the trained model; refusing to score"
            )

        vector = np.asarray(representation.feature_vector, dtype=np.float64)
        score = self._weights.score(vector)
        if not np.isfinite(score):
            raise ClassicalWeightsError("computed score is not finite; refusing to report it")

        return ModelOutput(
            model_id=self._descriptor.model_id,
            version=self._descriptor.version,
            score_name=self._descriptor.score_name,
            raw_score=score,
            score_kind=self._descriptor.score_kind,
            score_semantics=self._descriptor.score_semantics,
            calibrated=False,
            decision_threshold=self._weights.threshold,
            confidence=None,
            labels=self._weights.labels,
            extra={
                "feature_schema": IMAGE_FEATURE_NAMES,
                "feature_dim": self._weights.feature_dim,
                "feature_values": list(representation.feature_vector or ()),
                "feature_source_digest": getattr(sample, "digest", None),
            },
            limitations=(
                (
                    "Classical baseline: a linear model over hand-crafted residual "
                    "and DCT features. It is a mechanism demonstrator, not a validated "
                    "detector."
                ),
                "The score is uncalibrated and is not a probability or a confidence.",
            ),
        )


def build(
    descriptor: ModelDescriptor,
    *,
    model_dir: Path,
    weights_path: Path | None,
) -> ClassicalImageBackend:
    """Construct a :class:`ClassicalImageBackend` from a model card.

    Called by :func:`ml.inference.models.load_model_backend` with the descriptor and
    resolved paths.

    Raises:
        ClassicalWeightsError: If no weights are declared, the file is missing, or
            the weights are inconsistent with the descriptor.
    """
    del model_dir  # only used to describe the file; weights_path is authoritative
    if weights_path is None:
        raise ClassicalWeightsError(
            f"model {descriptor.qualified_id!r} declares no weights file, so the "
            "classical backend cannot be built"
        )
    weights = load_classical_weights(weights_path)

    if descriptor.labels and tuple(descriptor.labels) != weights.labels:
        raise ClassicalWeightsError(
            f"model {descriptor.qualified_id!r} card labels {tuple(descriptor.labels)!r} "
            f"do not match the weights labels {weights.labels!r}"
        )
    if (
        descriptor.decision_threshold is not None
        and weights.threshold is not None
        and not np.isclose(descriptor.decision_threshold, weights.threshold)
    ):
        raise ClassicalWeightsError(
            f"model {descriptor.qualified_id!r} card threshold "
            f"{descriptor.decision_threshold} does not match the weights file "
            f"threshold {weights.threshold}"
        )
    return ClassicalImageBackend(descriptor, weights)
