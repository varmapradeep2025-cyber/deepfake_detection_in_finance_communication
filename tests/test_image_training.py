"""Tests for the classical classifier training and evaluation layer.

The data here is synthetic and separable by construction, so the tests can assert
that the plumbing *works* — scaling is fitted, scores are oriented, metrics are
recorded or honestly omitted — without making any claim about real detection
performance.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.image.features import IMAGE_FEATURE_NAMES
from ml.image.training import (
    ClassifierConfig,
    ClassifierTrainingError,
    FittedClassifier,
    evaluate_classifier,
    predict_scores,
    select_threshold,
    train_classifier,
)

_DIM = len(IMAGE_FEATURE_NAMES)


def _separable(seed: int = 0, per_class: int = 40):
    rng = np.random.default_rng(seed)
    real = rng.normal(0.0, 1.0, (per_class, _DIM))
    fake = rng.normal(3.0, 1.0, (per_class, _DIM))
    matrix = np.vstack([real, fake])
    labels = ["real"] * per_class + ["fake"] * per_class
    order = rng.permutation(len(labels))
    return matrix[order], [labels[i] for i in order]


def _fitted(seed: int = 0) -> FittedClassifier:
    matrix, labels = _separable(seed)
    return train_classifier(
        matrix,
        labels,
        feature_names=IMAGE_FEATURE_NAMES,
        seed=seed,
        trained_at="2026-01-01T00:00:00+00:00",
    )


class TestTraining:
    def test_fits_and_scores_the_training_data(self):
        matrix, labels = _separable()
        fitted = _fitted()
        scores = fitted.score(matrix)
        assert scores.shape == (len(labels),)
        positive_scores = scores[np.asarray(labels) == "fake"]
        negative_scores = scores[np.asarray(labels) == "real"]
        assert positive_scores.mean() > negative_scores.mean()

    def test_records_feature_names_and_dimension(self):
        fitted = _fitted()
        assert fitted.feature_names == IMAGE_FEATURE_NAMES
        assert fitted.feature_dim == _DIM
        assert fitted.trained_samples == 80
        assert fitted.positive_index == 1
        assert fitted.classes == ("real", "fake")

    def test_predict_returns_zero_or_one(self):
        matrix, _ = _separable()
        predictions = _fitted().predict(matrix)
        assert set(np.unique(predictions)) <= {0, 1}

    def test_predict_labels_returns_names(self):
        names = _fitted().predict_labels(_separable()[0])
        assert set(names) <= {"real", "fake"}

    def test_single_class_training_is_refused(self):
        matrix = np.zeros((4, _DIM))
        with pytest.raises(ClassifierTrainingError):
            train_classifier(
                matrix, ["real"] * 4, feature_names=IMAGE_FEATURE_NAMES
            )

    def test_feature_name_count_must_match(self):
        matrix, labels = _separable()
        with pytest.raises(ClassifierTrainingError):
            train_classifier(matrix, labels, feature_names=("only_one",))

    def test_label_count_must_match(self):
        matrix, _ = _separable()
        with pytest.raises(ClassifierTrainingError):
            train_classifier(matrix, ["fake"], feature_names=IMAGE_FEATURE_NAMES)

    def test_non_finite_matrix_is_refused(self):
        matrix, labels = _separable()
        matrix = matrix.copy()
        matrix[0, 0] = np.inf
        with pytest.raises(ClassifierTrainingError):
            train_classifier(matrix, labels, feature_names=IMAGE_FEATURE_NAMES)


class TestSerialization:
    def test_round_trip_scores_identically(self):
        fitted = _fitted()
        matrix, _ = _separable()
        rebuilt = FittedClassifier.from_dict(fitted.to_dict())
        assert np.allclose(fitted.score(matrix), rebuilt.score(matrix))
        assert rebuilt.classes == fitted.classes
        assert rebuilt.config == fitted.config

    def test_config_validation(self):
        with pytest.raises(ClassifierTrainingError):
            ClassifierConfig(c=0.0)
        with pytest.raises(ClassifierTrainingError):
            ClassifierConfig(positive_label="x", negative_label="x")


class TestThresholdSelection:
    def test_selects_a_separating_threshold(self):
        matrix, labels = _separable()
        fitted = _fitted()
        selection = select_threshold(fitted.score(matrix), labels, positive_label="fake")
        predictions = (fitted.score(matrix) >= selection.threshold).astype(int)
        expected = (np.asarray(labels) == "fake").astype(int)
        assert (predictions == expected).mean() > 0.95
        assert selection.balanced_accuracy > 0.95

    def test_is_deterministic(self):
        matrix, labels = _separable()
        scores = _fitted().score(matrix)
        first = select_threshold(scores, labels)
        second = select_threshold(scores, labels)
        assert first == second

    def test_single_class_is_refused(self):
        with pytest.raises(ClassifierTrainingError):
            select_threshold(np.array([0.1, 0.2]), ["real", "real"])

    def test_length_mismatch_is_refused(self):
        with pytest.raises(ClassifierTrainingError):
            select_threshold(np.array([0.1, 0.2]), ["real"])


class TestEvaluation:
    def test_metrics_cover_the_full_contract(self):
        matrix, labels = _separable()
        fitted = _fitted()
        evaluation = evaluate_classifier(
            fitted, matrix, labels, threshold=0.0, split_name="test"
        )
        assert set(evaluation.metrics) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "roc_auc",
            "pr_auc",
        }
        assert evaluation.evaluated_samples == len(labels)
        assert evaluation.metric_value("roc_auc") is not None
        assert evaluation.metric_value("pr_auc") is not None
        assert evaluation.confusion_matrix.total == len(labels)

    def test_metrics_are_bounded(self):
        matrix, labels = _separable()
        evaluation = evaluate_classifier(
            _fitted(), matrix, labels, threshold=0.0
        )
        for name in ("accuracy", "precision", "recall", "f1", "specificity"):
            value = evaluation.metric_value(name)
            assert value is not None and 0.0 <= value <= 1.0

    def test_single_class_slice_omits_auc_with_a_note(self):
        matrix, _ = _separable()
        evaluation = evaluate_classifier(
            _fitted(), matrix, ["real"] * matrix.shape[0], threshold=0.0
        )
        assert evaluation.metric_value("roc_auc") is None
        assert evaluation.metric_value("pr_auc") is None
        assert any("both classes" in note for note in evaluation.notes)

    def test_zero_predicted_positives_omits_precision(self):
        matrix, labels = _separable()
        evaluation = evaluate_classifier(
            _fitted(), matrix, labels, threshold=1e9
        )
        assert evaluation.metric_value("precision") is None
        assert any("predicted positive" in note for note in evaluation.notes)

    def test_predict_scores_wrapper_matches_score(self):
        matrix, _ = _separable()
        fitted = _fitted()
        assert np.allclose(predict_scores(fitted, matrix), fitted.score(matrix))
