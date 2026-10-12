"""Train and evaluate the classical image baseline classifier.

Scope
-----
This module turns a feature matrix into a *fitted* classifier and into honest
evaluations of that classifier. It deliberately contains no data loading, no file
I/O, and no knowledge of datasets: the caller supplies training data that it has
already restricted to the training split. Keeping the split decision outside this
module is what stops evaluation leakage from creeping in through the back door.

The model
---------
A :class:`~sklearn.preprocessing.StandardScaler` fitted on the training split only,
followed by an L2-regularized :class:`~sklearn.linear_model.LogisticRegression`.
That is the smallest model that is (a) fitted only on training data, (b) exactly
reproducible from a seed, (c) trivially serializable as plain numbers, and (d) has
coefficients a human can read.

Honest evaluation
-----------------
The module never invents a metric. ROC-AUC and PR-AUC are reported only when the
evaluated slice contains both classes *and* the scores actually vary; otherwise they
are ``None`` with a note explaining why. Every metric goes through
:func:`experiments.results.scalar_metrics`, so a missing metric stays missing
(``None``) rather than silently becoming ``0.0``.

Thresholds are selected on a validation slice by the caller, never on the test
slice. :func:`select_threshold` exists so that decision can be made explicitly and
recorded; it does not know or care which split it was handed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from experiments.results import ConfusionMatrix, ScalarMetric, scalar_metrics

__all__ = [
    "ClassifierConfig",
    "ClassifierEvaluation",
    "ClassifierTrainingError",
    "FittedClassifier",
    "ThresholdSelection",
    "evaluate_classifier",
    "predict_scores",
    "select_threshold",
    "train_classifier",
]


class ClassifierTrainingError(ValueError):
    """Raised when the inputs cannot be used to train or evaluate a classifier."""


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Hyper-parameters of the classical classifier.

    Attributes:
        c: Inverse L2 regularization strength (scikit-learn's ``C``). Larger values
            fit the training data more closely.
        max_iter: Maximum optimizer iterations.
        solver: scikit-learn solver name. ``"lbfgs"`` is deterministic here.
        positive_label: The label that means "manipulated". Scores are oriented so
            that larger values mean *more likely* to be this label.
        negative_label: The label that means "authentic".
    """

    c: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    positive_label: str = "fake"
    negative_label: str = "real"

    def __post_init__(self) -> None:
        if self.c <= 0.0:
            raise ClassifierTrainingError(f"c must be positive, got {self.c}")
        if self.max_iter < 1:
            raise ClassifierTrainingError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.positive_label == self.negative_label:
            raise ClassifierTrainingError("positive_label and negative_label must differ")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "c": float(self.c),
            "max_iter": int(self.max_iter),
            "solver": self.solver,
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
        }


@dataclass(frozen=True, slots=True)
class FittedClassifier:
    """A trained linear classifier, stored as plain numbers.

    The scaling and the linear coefficients are all that is needed to score a
    feature vector, which is why a fitted model serializes to JSON without pickle
    or joblib. An artifact written from a fitted classifier can be read back without
    scikit-learn installed.
    """

    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    classes: tuple[str, ...]
    positive_index: int
    feature_names: tuple[str, ...]
    config: ClassifierConfig
    seed: int
    trained_samples: int
    trained_at: str
    schema_version: int = 1

    @property
    def feature_dim(self) -> int:
        """Number of features the classifier expects."""
        return len(self.coefficients)

    def score(self, matrix: Any) -> npt.NDArray[np.float64]:
        """Return one decision score per row of ``matrix``.

        Larger means more likely to be the positive label. The sign is flipped when
        the positive class is not the one scikit-learn placed at index 1.
        """
        features = _as_feature_matrix(matrix, expected_dim=self.feature_dim)
        mean = np.asarray(self.scaler_mean, dtype=np.float64)
        scale = np.asarray(self.scaler_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        scaled = (features - mean) / scale
        raw = scaled @ coefficients + float(self.intercept)
        if self.positive_index == 1:
            return np.asarray(raw, dtype=np.float64)
        return np.asarray(-raw, dtype=np.float64)

    def predict(
        self,
        matrix: Any,
        *,
        threshold: float = 0.0,
    ) -> npt.NDArray[np.int64]:
        """Return a 0/1 prediction per row (1 = positive label)."""
        scores = self.score(matrix)
        return np.asarray(scores >= threshold, dtype=np.int64)

    def predict_labels(
        self,
        matrix: Any,
        *,
        threshold: float = 0.0,
    ) -> list[str]:
        """Return the predicted label *name* per row."""
        predicted = self.predict(matrix, threshold=threshold)
        return [self.classes[int(value)] for value in predicted]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, sufficient to reconstruct the classifier."""
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "scaler": {
                "mean": list(self.scaler_mean),
                "scale": list(self.scaler_scale),
            },
            "coefficients": list(self.coefficients),
            "intercept": float(self.intercept),
            "classes": list(self.classes),
            "positive_index": int(self.positive_index),
            "config": self.config.to_dict(),
            "seed": int(self.seed),
            "trained_samples": int(self.trained_samples),
            "trained_at": self.trained_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FittedClassifier:
        """Rebuild a classifier from :meth:`to_dict` output."""
        scaler = raw["scaler"]
        config = raw["config"]
        return cls(
            scaler_mean=tuple(float(v) for v in scaler["mean"]),
            scaler_scale=tuple(float(v) for v in scaler["scale"]),
            coefficients=tuple(float(v) for v in raw["coefficients"]),
            intercept=float(raw["intercept"]),
            classes=tuple(str(v) for v in raw["classes"]),
            positive_index=int(raw["positive_index"]),
            feature_names=tuple(str(v) for v in raw["feature_names"]),
            config=ClassifierConfig(
                c=float(config["c"]),
                max_iter=int(config["max_iter"]),
                solver=str(config["solver"]),
                positive_label=str(config["positive_label"]),
                negative_label=str(config["negative_label"]),
            ),
            seed=int(raw["seed"]),
            trained_samples=int(raw["trained_samples"]),
            trained_at=str(raw["trained_at"]),
        )


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    """Result of choosing a decision threshold on one slice.

    Attributes:
        threshold: Score at or above which a row is called positive.
        balanced_accuracy: Mean of recall and specificity at :attr:`threshold`.
        youden_j: ``recall + specificity - 1`` at :attr:`threshold`.
        candidates: Number of candidate thresholds considered.
    """

    threshold: float
    balanced_accuracy: float
    youden_j: float
    candidates: int


@dataclass(frozen=True, slots=True)
class ClassifierEvaluation:
    """Evaluation of a fitted classifier on one slice.

    Attributes:
        split_name: Which slice this was (for example ``"validation"``).
        threshold: Threshold the confusion matrix and metrics were computed at.
        positive_class: The label treated as positive.
        evaluated_samples: Number of rows evaluated.
        confusion_matrix: Confusion matrix at :attr:`threshold`.
        metrics: Metric name to :class:`~experiments.results.ScalarMetric`.
        notes: Explanations for any metric left unmeasured.
    """

    split_name: str
    threshold: float
    positive_class: str
    evaluated_samples: int
    confusion_matrix: ConfusionMatrix
    metrics: dict[str, ScalarMetric] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def metric_value(self, name: str) -> float | None:
        """The numeric value of a metric, or ``None`` if unmeasured."""
        metric = self.metrics.get(name)
        return None if metric is None else metric.value


def _as_feature_matrix(
    matrix: Any,
    *,
    expected_dim: int | None = None,
) -> npt.NDArray[np.float64]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ClassifierTrainingError(f"expected a 2D feature matrix, got {array.ndim} dims")
    if array.size == 0:
        raise ClassifierTrainingError("feature matrix is empty")
    if not np.isfinite(array).all():
        raise ClassifierTrainingError("feature matrix contains non-finite values")
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise ClassifierTrainingError(
            f"expected {expected_dim} features per row, got {array.shape[1]}"
        )
    return array


def _require_sklearn() -> tuple[Any, Any, Any]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise ClassifierTrainingError(
            "scikit-learn is required to train and evaluate the classical image "
            "baseline; install it before running training"
        ) from exc
    return StandardScaler, LogisticRegression, (roc_auc_score, average_precision_score)


def train_classifier(
    matrix: Any,
    labels: Sequence[str],
    *,
    feature_names: Sequence[str],
    config: ClassifierConfig | None = None,
    seed: int = 0,
    trained_at: str | None = None,
) -> FittedClassifier:
    """Fit a classifier on a training feature matrix.

    Args:
        matrix: ``(n_samples, n_features)`` feature matrix.
        labels: One label per row; values other than ``config.positive_label`` are
            treated as the negative class.
        feature_names: Feature names in column order.
        config: Hyper-parameters.
        seed: Recorded for traceability. ``lbfgs`` is deterministic here, so the
            seed does not change the fitted coefficients.
        trained_at: Override the training timestamp (useful for tests).

    Raises:
        ClassifierTrainingError: If inputs are malformed, the feature count does not
            match ``feature_names``, or the training data lacks either class.
    """
    resolved = config or ClassifierConfig()
    names = tuple(str(name) for name in feature_names)
    features = _as_feature_matrix(matrix)
    if features.shape[1] != len(names):
        raise ClassifierTrainingError(
            f"feature matrix has {features.shape[1]} columns but {len(names)} names"
        )
    if len(labels) != features.shape[0]:
        raise ClassifierTrainingError(f"got {len(labels)} labels for {features.shape[0]} rows")

    label_array = np.asarray(list(labels))
    positive_mask = label_array == resolved.positive_label
    if positive_mask.all() or not positive_mask.any():
        raise ClassifierTrainingError(
            "training labels must contain both the positive and negative class"
        )

    scaler_cls, model_cls, _ = _require_sklearn()
    encoded = positive_mask.astype(np.int64)

    scaler = scaler_cls()
    scaled = scaler.fit_transform(features)
    model = model_cls(
        C=resolved.c,
        max_iter=resolved.max_iter,
        solver=resolved.solver,
        random_state=seed,
    )
    model.fit(scaled, encoded)

    coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
    classes = tuple(
        resolved.negative_label if int(value) == 0 else resolved.positive_label
        for value in np.asarray(model.classes_).reshape(-1)
    )
    positive_index = int(np.where(np.asarray(model.classes_).reshape(-1) == 1)[0][0])
    stamp = trained_at or datetime.now(UTC).isoformat(timespec="seconds")

    return FittedClassifier(
        scaler_mean=tuple(float(v) for v in np.asarray(scaler.mean_, dtype=np.float64)),
        scaler_scale=tuple(float(v) for v in np.asarray(scaler.scale_, dtype=np.float64)),
        coefficients=tuple(float(v) for v in coefficients),
        intercept=intercept,
        classes=classes,
        positive_index=positive_index,
        feature_names=names,
        config=resolved,
        seed=int(seed),
        trained_samples=int(features.shape[0]),
        trained_at=stamp,
    )


def select_threshold(
    scores: Any,
    labels: Sequence[str],
    *,
    positive_label: str = "fake",
) -> ThresholdSelection:
    """Choose a decision threshold on one slice.

    The threshold maximizes balanced accuracy over the observed score values. Ties
    keep the lowest score value, so the choice is deterministic.

    Args:
        scores: One score per row.
        labels: One label per row.
        positive_label: The label treated as positive.

    Raises:
        ClassifierTrainingError: If lengths differ or the slice has only one class.
    """
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_array = np.asarray(list(labels)).reshape(-1)
    if label_array.shape[0] != score_array.shape[0]:
        raise ClassifierTrainingError(
            f"got {label_array.shape[0]} labels for {score_array.shape[0]} scores"
        )
    positive = (label_array == str(positive_label)).astype(np.int64)
    if positive.all() or not positive.any():
        raise ClassifierTrainingError(
            "threshold selection requires both classes in the slice"
        )

    candidates = np.unique(score_array)
    best_threshold = float(candidates[-1]) + 1.0
    best_balanced = -1.0
    best_youden = -1.0
    for candidate in candidates:
        predicted = (score_array >= candidate).astype(np.int64)
        tp = float(np.sum((positive == 1) & (predicted == 1)))
        fn = float(np.sum((positive == 1) & (predicted == 0)))
        tn = float(np.sum((positive == 0) & (predicted == 0)))
        fp = float(np.sum((positive == 0) & (predicted == 1)))
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        balanced = 0.5 * (recall + specificity)
        if balanced > best_balanced:
            best_balanced = balanced
            best_youden = recall + specificity - 1.0
            best_threshold = float(candidate)

    return ThresholdSelection(
        threshold=best_threshold,
        balanced_accuracy=best_balanced,
        youden_j=best_youden,
        candidates=int(candidates.size),
    )


def evaluate_classifier(
    fitted: FittedClassifier,
    matrix: Any,
    labels: Sequence[str],
    *,
    threshold: float,
    positive_label: str | None = None,
    split_name: str = "test",
) -> ClassifierEvaluation:
    """Evaluate a fitted classifier on one slice.

    Args:
        fitted: The fitted classifier.
        matrix: ``(n_samples, n_features)`` feature matrix.
        labels: One label per row.
        threshold: Score at or above which a row is called positive.
        positive_label: Override the positive label (defaults to the classifier's).
        split_name: Recorded on the result for traceability.

    Raises:
        ClassifierTrainingError: If shapes are inconsistent.
    """
    positive_class = str(positive_label or fitted.config.positive_label)
    features = _as_feature_matrix(matrix, expected_dim=fitted.feature_dim)
    label_array = np.asarray(list(labels)).reshape(-1)
    if label_array.shape[0] != features.shape[0]:
        raise ClassifierTrainingError(
            f"got {label_array.shape[0]} labels for {features.shape[0]} rows"
        )

    scores = fitted.score(features)
    positive = (label_array == positive_class).astype(np.int64)
    predicted = (scores >= float(threshold)).astype(np.int64)

    true_positives = int(np.sum((positive == 1) & (predicted == 1)))
    false_negatives = int(np.sum((positive == 1) & (predicted == 0)))
    false_positives = int(np.sum((positive == 0) & (predicted == 1)))
    true_negatives = int(np.sum((positive == 0) & (predicted == 0)))
    confusion = ConfusionMatrix(
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        false_positives=false_positives,
        true_positives=true_positives,
    )

    total = int(positive.shape[0])
    positives = true_positives + false_negatives
    negatives = true_negatives + false_positives
    predicted_positives = true_positives + false_positives

    accuracy = (true_positives + true_negatives) / total if total else None
    precision = true_positives / predicted_positives if predicted_positives else None
    recall = true_positives / positives if positives else None
    specificity = true_negatives / negatives if negatives else None
    f1 = None
    if precision is not None and recall is not None:
        denominator = precision + recall
        f1 = (2.0 * precision * recall / denominator) if denominator else 0.0

    notes: list[str] = []
    roc_auc: float | None = None
    pr_auc: float | None = None
    both_classes = positives > 0 and negatives > 0
    if not both_classes:
        notes.append(
            "ROC-AUC and PR-AUC require both classes in the evaluated slice; only "
            "one class was present."
        )
    elif np.unique(scores).size < 2:
        notes.append(
            "ROC-AUC and PR-AUC require varying decision scores; all scores were "
            "identical."
        )
    else:
        _, _, (roc_auc_score, average_precision_score) = _require_sklearn()
        roc_auc = float(roc_auc_score(positive, scores))
        pr_auc = float(average_precision_score(positive, scores))

    if predicted_positives == 0:
        notes.append("Precision is unmeasured because no row was predicted positive.")

    metrics = scalar_metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
    )
    return ClassifierEvaluation(
        split_name=split_name,
        threshold=float(threshold),
        positive_class=positive_class,
        evaluated_samples=total,
        confusion_matrix=confusion,
        metrics=metrics,
        notes=tuple(notes),
    )


def predict_scores(
    fitted: FittedClassifier,
    matrix: Any,
) -> npt.NDArray[np.float64]:
    """Return decision scores for ``matrix`` (thin wrapper over ``fitted.score``)."""
    return fitted.score(matrix)
