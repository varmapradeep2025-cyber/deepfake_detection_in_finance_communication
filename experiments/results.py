"""Experiment result and metrics schema.

Non-negotiable rule: **a metric that was not measured is not reported.**

Every metric field is ``float | None``, defaulting to ``None``. There is no default
of 0.0 and no implicit computation, because a missing value silently rendered as
``0.0`` is indistinguishable from a genuinely catastrophic score, and a fabricated
``roc_auc`` on a run where ranking failed would be read as a real result. Absence is
carried explicitly.

Range policy
------------
* :class:`ScalarMetric` covers rates in ``[0, 1]`` (``accuracy`` ... ``pr_auc``).
  Its validator rejects out-of-range values, since those are always computation bugs.
* :class:`RateMetric` covers quantities that are **rates over a count**, such as
  ``false_positive_rate``. There is no ``[0, 1]`` bound: over a denominator of 1,
  4 false positives means a rate of 4.0. Hard-coding a ``[0, 1]`` clamp here would
  make such a value unrepresentable.
* Confusion matrix cells and generalization gaps are plain, validated per-field.

Related design decisions
------------------------
``generalization_gap`` is carried as explicit in-domain and cross-domain values
rather than computed implicitly, because its sign convention differs between
literatures (train-minus-test vs test-minus-train). Storing the operands avoids the
ambiguity; :attr:`GeneralizationGap.difference` then states the convention it uses
(train minus test, positive means degradation).

Calibration metrics live in :class:`CalibrationMetrics` and are ``None`` unless
explicitly measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "CalibrationMetrics",
    "ConfusionMatrix",
    "ExperimentResult",
    "GeneralizationGap",
    "MetricOutOfRangeError",
    "RateMetric",
    "ResultValidationError",
    "ScalarMetric",
    "METRIC_NAMES",
]

RESULT_SCHEMA_VERSION = 1


class ResultValidationError(ValueError):
    """Raised when a result record violates the schema.

    ``str(error)`` includes every problem found, so a traceback alone is enough to
    diagnose the failure without inspecting :attr:`errors`.
    """

    def __init__(self, message: str, *, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        if self.errors:
            message = f"{message}: " + "; ".join(self.errors)
        super().__init__(message)


class MetricOutOfRangeError(ResultValidationError):
    """Raised when a bounded metric falls outside ``[0, 1]``."""


METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "pr_auc",
)


def _validate_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{name} must be a float or None, got {type(value).__name__}")
    return float(value)


@dataclass(frozen=True, slots=True)
class ScalarMetric:
    """A performance rate bounded to ``[0, 1]``.

    Args:
        name: One of :data:`METRIC_NAMES`.
        value: The measured value, or ``None`` when unavailable.

    Raises:
        MetricOutOfRangeError: If ``value`` falls outside ``[0, 1]``.
    """

    name: str
    value: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _validate_optional_float(self.value, self.name))
        if self.value is not None and not 0.0 <= self.value <= 1.0:
            raise MetricOutOfRangeError(f"{self.name} must be within [0, 1], got {self.value}")

    @property
    def is_available(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {self.name: self.value}


def scalar_metrics(**values: float | None) -> dict[str, ScalarMetric]:
    """Build the bounded metric set from keyword arguments.

    Unmentioned metrics default to ``None``, which keeps "not measured" explicit
    instead of defaulting to a plausible number.

    Example:
        >>> scalar_metrics(accuracy=0.91)["roc_auc"].is_available
        False
    """
    unknown = sorted(set(values) - set(METRIC_NAMES))
    if unknown:
        raise ResultValidationError(f"unknown metric name(s): {', '.join(unknown)}")
    return {name: ScalarMetric(name=name, value=values.get(name)) for name in METRIC_NAMES}


@dataclass(frozen=True, slots=True)
class RateMetric:
    """A count-over-denominator quantity with no upper bound.

    Used for rates such as ``false_positive_rate`` where a small denominator makes
    values above 1.0 correct. A ``None`` value means unmeasured.

    Args:
        name: Metric name.
        value: Measured value, or ``None``.
    """

    name: str
    value: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ResultValidationError("rate metric name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "value", _validate_optional_float(self.value, self.name))

    @property
    def is_available(self) -> bool:
        return self.value is not None

    def as_dict(self) -> dict[str, Any]:
        return {self.name: self.value}


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Binary confusion matrix.

    ``positive_class`` records what the matrix treats as positive, because a matrix
    whose positive class is unstated cannot be interpreted. Default ``"fake"``:
    detecting manipulation is this project's positive outcome.

    Row/column convention is explicitly TN, FP, FN, TP, so there is no ambiguity
    about which cell is a missed deepfake.
    """

    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    positive_class: str = "fake"

    def __post_init__(self) -> None:
        for name in ("true_negatives", "false_positives", "false_negatives", "true_positives"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ResultValidationError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
            if value < 0:
                raise ResultValidationError(f"{name} must be non-negative, got {value}")
        if not isinstance(self.positive_class, str) or not self.positive_class.strip():
            raise ResultValidationError("positive_class must be a non-empty string")
        object.__setattr__(self, "positive_class", self.positive_class.strip())

    @property
    def total(self) -> int:
        """Number of samples the matrix accounts for."""
        return (
            self.true_negatives + self.false_positives + self.false_negatives + self.true_positives
        )

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    def to_matrix(self) -> list[list[int]]:
        """Return as ``[[TN, FP], [FN, TP]]`` with rows = actual, columns = predicted."""
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ConfusionMatrix:
        known = {
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "positive_class",
        }
        return cls(**{key: value for key, value in raw.items() if key in known})


@dataclass(frozen=True, slots=True)
class GeneralizationGap:
    """Degradation when moving from the training distribution to another one.

    Stores the operands rather than a single signed number, because "the gap" has
    opposite conventions in different parts of the literature.

    Args:
        in_domain: Metric value on the training/validation distribution.
        cross_domain: Metric value on the target (unseen) distribution.
        metric_name: Which metric the values refer to, e.g. ``"accuracy"``.
        domain_a: Label for ``in_domain``, default ``"in_domain"``.
        domain_b: Label for ``cross_domain``, default ``"cross_domain"``.
    """

    in_domain: float | None
    cross_domain: float | None
    metric_name: str = "accuracy"
    domain_a: str = "in_domain"
    domain_b: str = "cross_domain"

    def __post_init__(self) -> None:
        object.__setattr__(self, "in_domain", _validate_optional_float(self.in_domain, "in_domain"))
        object.__setattr__(
            self, "cross_domain", _validate_optional_float(self.cross_domain, "cross_domain")
        )
        if not isinstance(self.metric_name, str) or not self.metric_name.strip():
            raise ResultValidationError("metric_name must be a non-empty string")
        object.__setattr__(self, "metric_name", self.metric_name.strip())
        for name in ("domain_a", "domain_b"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ResultValidationError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())

    @property
    def difference(self) -> float | None:
        """``in_domain - cross_domain``.

        Convention: **positive means degradation** under domain shift, and negative
        means the target domain scored better. ``None`` if either operand is
        unavailable, so a gap is never computed from partial data.
        """
        if self.in_domain is None or self.cross_domain is None:
            return None
        return self.in_domain - self.cross_domain

    @property
    def is_available(self) -> bool:
        return self.difference is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form. ``difference`` is included when computable."""
        payload = asdict(self)
        payload["difference"] = self.difference
        return payload


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Confidence-calibration measures. ``None`` unless explicitly measured.

    A detector that is 95% accurate but miscalibrated is unsafe to act on, so
    calibration is tracked separately from discrimination.
    """

    expected_calibration_error: float | None = None
    brier_score: float | None = None

    def __post_init__(self) -> None:
        for name in ("expected_calibration_error", "brier_score"):
            value = _validate_optional_float(getattr(self, name), name)
            if value is not None and value < 0.0:
                raise ResultValidationError(f"{name} must be non-negative, got {value}")
            object.__setattr__(self, name, value)

    @property
    def is_available(self) -> bool:
        return self.expected_calibration_error is not None or self.brier_score is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Metrics and interpretation for one completed run.

    Every metric defaults to ``None``. A run that failed partway therefore produces a
    valid result object with the absent metrics still ``None``, instead of a
    plausible-looking row of zeros.

    Attributes:
        experiment_id: The experiment this result belongs to.
        run_id: Unique run identifier, matching the run directory name.
        seed: Seed that governed the run.
        run_number: Index of the run within the seed set.
        evaluation_split: Which split was scored, e.g. ``"test"``.
        positive_class: What the evaluation treated as positive.
        evaluated_samples: Number of samples scored. ``0`` means nothing was scored.
        metrics: Bounded performance metrics keyed by name.
        confusion_matrix: Counts per outcome, if computed.
        generalization_gap: In-domain vs cross-domain comparison, where applicable.
        calibration: Calibration measures, where computed.
        threshold: Decision threshold used, where applicable.
        notes: Free-text caveats. Should state anything that limits interpretation.
        completed_at: RFC 3339 completion timestamp.
    """

    experiment_id: str
    run_id: str
    seed: int
    run_number: int = 0
    evaluation_split: str = "test"
    positive_class: str = "fake"
    evaluated_samples: int = 0
    metrics: dict[str, ScalarMetric] = field(default_factory=dict)
    confusion_matrix: ConfusionMatrix | None = None
    generalization_gap: GeneralizationGap | None = None
    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    threshold: float | None = None
    notes: tuple[str, ...] = ()
    completed_at: str = ""
    result_schema_version: int = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        for name in ("experiment_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty string")
            else:
                object.__setattr__(self, name, value.strip())

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            errors.append("seed must be an integer")
        if isinstance(self.run_number, bool) or not isinstance(self.run_number, int):
            errors.append("run_number must be an integer")
        elif self.run_number < 0:
            errors.append("run_number must be non-negative")

        if isinstance(self.evaluated_samples, bool) or not isinstance(self.evaluated_samples, int):
            errors.append("evaluated_samples must be an integer")
        elif self.evaluated_samples < 0:
            errors.append("evaluated_samples must be non-negative")

        unknown = sorted(set(self.metrics) - set(METRIC_NAMES))
        if unknown:
            errors.append(f"unknown metric name(s): {', '.join(unknown)}")

        for key, metric in self.metrics.items():
            if not isinstance(metric, ScalarMetric):
                errors.append(
                    f"metrics[{key!r}] must be a ScalarMetric, got {type(metric).__name__}"
                )
            elif metric.name != key:
                errors.append(f"metrics[{key!r}] has mismatched name {metric.name!r}")

        if self.confusion_matrix is not None and not isinstance(
            self.confusion_matrix, ConfusionMatrix
        ):
            errors.append("confusion_matrix must be a ConfusionMatrix or None")
        if self.generalization_gap is not None and not isinstance(
            self.generalization_gap, GeneralizationGap
        ):
            errors.append("generalization_gap must be a GeneralizationGap or None")
        if not isinstance(self.calibration, CalibrationMetrics):
            errors.append("calibration must be a CalibrationMetrics")

        object.__setattr__(self, "threshold", _validate_optional_float(self.threshold, "threshold"))
        if isinstance(self.notes, (list, tuple)):
            object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        else:
            errors.append("notes must be a list of strings")

        object.__setattr__(
            self,
            "completed_at",
            self.completed_at or datetime.now(UTC).isoformat(timespec="seconds"),
        )

        if errors:
            raise ResultValidationError(
                f"invalid experiment result ({len(errors)} problem(s))", errors=errors
            )

    # --- access -----------------------------------------------------------

    def value(self, name: str) -> float | None:
        """Metric value by name, or ``None`` when absent or unmeasured.

        Raises:
            ResultValidationError: If ``name`` is not a recognized metric.
        """
        if name not in METRIC_NAMES:
            raise ResultValidationError(
                f"unknown metric {name!r}; expected one of {', '.join(METRIC_NAMES)}"
            )
        metric = self.metrics.get(name)
        return metric.value if metric is not None else None

    @property
    def measured_metrics(self) -> tuple[str, ...]:
        """Names of metrics that carry an actual value."""
        return tuple(sorted(name for name, metric in self.metrics.items() if metric.is_available))

    @property
    def missing_metrics(self) -> tuple[str, ...]:
        """Names of metrics that are absent or ``None``.

        Surfaced explicitly so a report can state what was not measured rather than
        quietly omitting it.
        """
        return tuple(sorted(name for name in METRIC_NAMES if self.value(name) is None))

    @property
    def has_metrics(self) -> bool:
        """Whether any metric was actually measured."""
        return bool(self.measured_metrics)

    @property
    def generalization_gap_value(self) -> float | None:
        """The generalization gap, or ``None`` when not applicable or unmeasured."""
        return self.generalization_gap.difference if self.generalization_gap else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict, preserving ``None`` rather than dropping keys."""
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "run_number": self.run_number,
            "evaluation_split": self.evaluation_split,
            "positive_class": self.positive_class,
            "evaluated_samples": self.evaluated_samples,
            "metrics": {name: self.value(name) for name in METRIC_NAMES},
            "measured_metrics": list(self.measured_metrics),
            "missing_metrics": list(self.missing_metrics),
            "confusion_matrix": self.confusion_matrix.to_dict() if self.confusion_matrix else None,
            "generalization_gap": self.generalization_gap.to_dict()
            if self.generalization_gap
            else None,
            "calibration": self.calibration.to_dict(),
            "threshold": self.threshold,
            "notes": list(self.notes),
            "completed_at": self.completed_at,
            "result_schema_version": self.result_schema_version,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentResult:
        """Rebuild a result from :meth:`to_dict` output."""
        metrics = {
            name: ScalarMetric(name=name, value=raw.get("metrics", {}).get(name))
            for name in METRIC_NAMES
        }
        raw_cm = raw.get("confusion_matrix")
        raw_gap = raw.get("generalization_gap")
        return cls(
            experiment_id=raw.get("experiment_id", ""),
            run_id=raw.get("run_id", ""),
            seed=raw.get("seed", 0),
            run_number=raw.get("run_number", 0),
            evaluation_split=raw.get("evaluation_split", "test"),
            positive_class=raw.get("positive_class", "fake"),
            evaluated_samples=raw.get("evaluated_samples", 0),
            metrics=metrics,
            confusion_matrix=ConfusionMatrix.from_dict(raw_cm) if raw_cm else None,
            generalization_gap=(
                GeneralizationGap(
                    in_domain=raw_gap.get("in_domain"),
                    cross_domain=raw_gap.get("cross_domain"),
                    metric_name=raw_gap.get("metric_name", "accuracy"),
                    domain_a=raw_gap.get("domain_a", "in_domain"),
                    domain_b=raw_gap.get("domain_b", "cross_domain"),
                )
                if raw_gap
                else None
            ),
            calibration=CalibrationMetrics(
                expected_calibration_error=raw.get("calibration", {}).get(
                    "expected_calibration_error"
                ),
                brier_score=raw.get("calibration", {}).get("brier_score"),
            ),
            threshold=raw.get("threshold"),
            notes=tuple(raw.get("notes", ())),
            completed_at=raw.get("completed_at", ""),
            result_schema_version=raw.get("result_schema_version", RESULT_SCHEMA_VERSION),
        )
