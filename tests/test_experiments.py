"""Tests for :mod:`experiments.schema` and :mod:`experiments.results`.

Covers experiment-family validation rules and the rule that unmeasured metrics stay
``None``.
"""

from __future__ import annotations

import json

import pytest

from experiments.results import (
    METRIC_NAMES,
    CalibrationMetrics,
    ConfusionMatrix,
    ExperimentResult,
    GeneralizationGap,
    MetricOutOfRangeError,
    RateMetric,
    ResultValidationError,
    scalar_metrics,
)
from experiments.schema import (
    ExperimentConfig,
    ExperimentType,
    ExperimentValidationError,
    create_experiment_config,
)

MANIFEST_HASH = "sha256:" + "c" * 64
SPLIT_HASH = "sha256:" + "d" * 64


def build_config(**overrides) -> ExperimentConfig:
    payload = {
        "experiment_id": "img-baseline-v0",
        "name": "image-baseline",
        "description": "Establishes the single-dataset image baseline.",
        "modality": "image",
        "model_version": "v0",
        "training_dataset": "dataset_a",
        "validation_dataset": "dataset_a",
        "test_dataset": "dataset_a",
        "seed": 20260101,
        "run_number": 0,
        "config_hash": MANIFEST_HASH,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return ExperimentConfig(**payload)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


class TestExperimentType:
    def test_exposes_four_families(self):
        assert set(ExperimentType.values()) == {
            "known_dataset",
            "cross_dataset",
            "ablation",
            "multimodal",
        }

    def test_parses_canonical_values(self):
        assert ExperimentType.parse("cross_dataset") is ExperimentType.CROSS_DATASET
        assert ExperimentType.parse("Ablation") is ExperimentType.ABLATION

    @pytest.mark.parametrize("bad", ["transfer", "", None, 3, "known-datasets"])
    def test_rejects_unknown_type(self, bad):
        with pytest.raises(ExperimentValidationError, match="invalid experiment_type"):
            ExperimentType.parse(bad)


class TestKnownDatasetConfig:
    def test_builds_valid_config(self):
        config = build_config()
        assert config.experiment_type is ExperimentType.KNOWN_DATASET
        assert config.is_cross_dataset is False

    def test_requires_description(self):
        with pytest.raises(ExperimentValidationError, match="description"):
            build_config(description="   ")

    @pytest.mark.parametrize(
        "bad", ["Bad_ID", "x!", "has space", "-leading", "a" * 200, "", "../escape"]
    )
    def test_rejects_unsafe_experiment_id(self, bad):
        with pytest.raises(ExperimentValidationError, match="experiment_id"):
            build_config(experiment_id=bad)

    @pytest.mark.parametrize("good", ["v0", "v1", "baseline-01", "ab"])
    def test_accepts_short_but_valid_identifiers(self, good):
        assert build_config(model_version=good).model_version == good

    def test_rejects_single_character_identifier(self):
        with pytest.raises(ExperimentValidationError, match="model_version"):
            build_config(model_version="a")

    def test_rejects_negative_run_number(self):
        with pytest.raises(ExperimentValidationError, match="run_number"):
            build_config(run_number=-1)

    def test_rejects_bad_config_hash_prefix(self):
        with pytest.raises(ExperimentValidationError, match="config_hash"):
            build_config(config_hash="deadbeef")

    def test_rejects_invalid_modality(self):
        with pytest.raises(ExperimentValidationError, match="modality"):
            build_config(modality="hologram")

    def test_rejects_invalid_label_like_dataset_name(self):
        with pytest.raises(ExperimentValidationError):
            build_config(training_dataset="")

    def test_rejects_non_mapping_config(self):
        with pytest.raises(ExperimentValidationError, match="config must be a mapping"):
            build_config(config=["not", "a", "mapping"])

    def test_rejects_non_hash_provenance_values(self):
        with pytest.raises(ExperimentValidationError, match="sha256"):
            build_config(manifests_used={"dataset_a": "unknown"})

    def test_accepts_valid_provenance_hashes(self):
        config = build_config(
            manifests_used={"dataset_a": MANIFEST_HASH},
            splits_used={"dataset_a": SPLIT_HASH},
        )
        assert config.manifests_used["dataset_a"] == MANIFEST_HASH

    def test_collects_multiple_problems_at_once(self):
        with pytest.raises(ExperimentValidationError) as exc:
            build_config(experiment_id="X", description="", seed=-5)
        assert len(exc.value.errors) >= 3


class TestCrossDatasetConfig:
    def test_builds_valid_cross_dataset_config(self):
        config = build_config(
            experiment_type="cross_dataset",
            test_dataset="dataset_b",
            target_dataset="dataset_b",
        )
        assert config.is_cross_dataset
        assert config.all_datasets == ("dataset_a", "dataset_b")

    def test_requires_target_dataset(self):
        with pytest.raises(ExperimentValidationError, match="require target_dataset"):
            build_config(experiment_type="cross_dataset", test_dataset="dataset_b")

    def test_rejects_self_target(self):
        with pytest.raises(ExperimentValidationError, match="different from"):
            build_config(
                experiment_type="cross_dataset",
                target_dataset="dataset_a",
                test_dataset="dataset_b",
            )

    def test_is_cross_dataset_property(self):
        assert build_config().is_cross_dataset is False


class TestAblationConfig:
    def test_builds_valid_ablation_config(self):
        config = build_config(
            experiment_type="ablation",
            ablation_of="img-baseline-reference",
            ablation_description="removes frequency-domain artifact branch",
        )
        assert config.experiment_type is ExperimentType.ABLATION
        assert config.ablation_of == "img-baseline-reference"

    def test_requires_ablation_of(self):
        with pytest.raises(ExperimentValidationError, match="require ablation_of"):
            build_config(experiment_type="ablation", ablation_description="removed something")

    def test_requires_description_of_what_was_removed(self):
        with pytest.raises(ExperimentValidationError, match="require ablation_description"):
            build_config(experiment_type="ablation", ablation_of="img-baseline-reference")

    def test_rejects_self_reference(self):
        with pytest.raises(ExperimentValidationError, match="must not reference"):
            build_config(
                experiment_type="ablation",
                ablation_of="img-baseline-v0",
                ablation_description="x",
                experiment_id="img-baseline-v0",
            )


class TestMultimodalConfig:
    def test_builds_valid_multimodal_config(self):
        config = build_config(
            experiment_type="multimodal",
            modality_datasets={"image": "dataset_a", "audio": "dataset_c"},
        )
        assert config.all_datasets == ("dataset_a", "dataset_c")

    def test_requires_modality_datasets(self):
        with pytest.raises(ExperimentValidationError, match="non-empty modality_datasets"):
            build_config(experiment_type="multimodal", modality_datasets={})

    def test_rejects_unknown_modality_key(self):
        with pytest.raises(ExperimentValidationError, match="not a valid modality"):
            build_config(experiment_type="multimodal", modality_datasets={"olfactory": "dataset_a"})

    def test_rejects_modality_datasets_on_non_multimodal_experiment(self):
        with pytest.raises(ExperimentValidationError, match="only meaningful for multimodal"):
            build_config(modality_datasets={"image": "dataset_a"})


class TestConfigHashing:
    def test_hash_is_deterministic(self):
        assert build_config().compute_config_hash() == build_config().compute_config_hash()

    def test_hash_excludes_created_at(self):
        assert (
            build_config().compute_config_hash()
            == build_config(created_at="1999-01-01T00:00:00+00:00").compute_config_hash()
        )

    def test_hash_changes_when_seed_changes(self):
        assert build_config().compute_config_hash() != build_config(seed=7).compute_config_hash()

    def test_hash_changes_when_hyperparameters_change(self):
        assert (
            build_config(config={"lr": 0.001}).compute_config_hash()
            != build_config(config={"lr": 0.01}).compute_config_hash()
        )

    def test_factory_stamps_a_matching_hash(self):
        config = create_experiment_config(
            experiment_id="cross-dataset-v1",
            name="cross-dataset-image",
            description="Train on A, evaluate on B.",
            modality="image",
            model_version="v1",
            training_dataset="dataset_a",
            validation_dataset="dataset_a",
            test_dataset="dataset_b",
            seed=20260102,
            run_number=1,
            experiment_type="cross_dataset",
            target_dataset="dataset_b",
        )
        assert config.config_hash == config.compute_config_hash()

    def test_round_trips_through_dict(self):
        original = build_config(manifests_used={"dataset_a": MANIFEST_HASH})
        restored = ExperimentConfig.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_reports_missing_fields(self):
        payload = build_config().to_dict()
        del payload["model_version"]
        with pytest.raises(ExperimentValidationError, match="missing required field"):
            ExperimentConfig.from_dict(payload)

    def test_to_dict_is_json_serializable(self):
        assert json.loads(json.dumps(build_config().to_dict()))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestScalarMetrics:
    def test_unmentioned_metrics_are_none_not_zero(self):
        metrics = scalar_metrics(accuracy=0.9)
        assert metrics["accuracy"].value == 0.9
        assert metrics["roc_auc"].value is None

    def test_all_seven_metrics_exist(self):
        assert set(scalar_metrics()) == set(METRIC_NAMES)

    def test_boundaries_are_accepted(self):
        for value in (0.0, 1.0):
            assert scalar_metrics(accuracy=value)["accuracy"].value == value

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, 42.0])
    def test_rejects_out_of_range_values(self, bad):
        with pytest.raises(MetricOutOfRangeError, match=r"\[0, 1\]"):
            scalar_metrics(accuracy=bad)

    def test_rejects_unknown_metric_name(self):
        with pytest.raises(ResultValidationError, match="unknown metric"):
            scalar_metrics(not_a_metric=0.5)

    def test_rejects_non_numeric_value(self):
        with pytest.raises(ResultValidationError, match="must be a float or None"):
            scalar_metrics(accuracy="high")

    def test_none_is_explicitly_available_flag_false(self):
        assert scalar_metrics()["f1"].is_available is False

    def test_rate_metric_allows_values_above_one(self):
        """A false-positive count over a denominator of one legitimately exceeds 1.0."""
        assert RateMetric(name="false_positive_rate", value=4.0).value == 4.0

    def test_rate_metric_rejects_non_numeric(self):
        with pytest.raises(ResultValidationError):
            RateMetric(name="r", value="nope")


class TestConfusionMatrix:
    def test_total_and_matrix_shape(self):
        matrix = ConfusionMatrix(
            true_negatives=10,
            false_positives=2,
            false_negatives=3,
            true_positives=85,
        )
        assert matrix.total == 100
        assert matrix.to_matrix() == [[10, 2], [3, 85]]

    def test_is_empty(self):
        assert ConfusionMatrix().is_empty
        assert not ConfusionMatrix(true_positives=1).is_empty

    def test_positive_class_is_stated(self):
        assert ConfusionMatrix().positive_class == "fake"

    def test_rejects_negative_counts(self):
        with pytest.raises(ResultValidationError, match="non-negative"):
            ConfusionMatrix(false_negatives=-1)

    def test_rejects_non_integer_counts(self):
        with pytest.raises(ResultValidationError, match="must be an integer"):
            ConfusionMatrix(true_positives=1.5)

    def test_rejects_empty_positive_class(self):
        with pytest.raises(ResultValidationError, match="positive_class"):
            ConfusionMatrix(positive_class="")

    def test_round_trips(self):
        matrix = ConfusionMatrix(true_negatives=1, true_positives=2)
        assert ConfusionMatrix.from_dict(matrix.to_dict()) == matrix


class TestGeneralizationGap:
    def test_difference_is_in_minus_cross(self):
        gap = GeneralizationGap(in_domain=0.95, cross_domain=0.71)
        assert gap.difference == pytest.approx(0.24)
        assert gap.is_available

    def test_positive_difference_means_degradation(self):
        assert GeneralizationGap(in_domain=0.9, cross_domain=0.7).difference > 0

    def test_absent_when_either_operand_missing(self):
        assert GeneralizationGap(in_domain=0.9, cross_domain=None).difference is None
        assert GeneralizationGap(in_domain=None, cross_domain=0.7).difference is None
        assert GeneralizationGap(in_domain=None, cross_domain=None).is_available is False

    def test_to_dict_includes_computed_difference(self):
        payload = GeneralizationGap(in_domain=0.9, cross_domain=0.7).to_dict()
        assert payload["difference"] == pytest.approx(0.2)

    def test_to_dict_keeps_null_difference(self):
        assert GeneralizationGap(in_domain=0.9, cross_domain=None).to_dict()["difference"] is None

    def test_rejects_empty_metric_name(self):
        with pytest.raises(ResultValidationError, match="metric_name"):
            GeneralizationGap(in_domain=0.9, cross_domain=0.7, metric_name="")

    def test_negative_gap_is_allowed_and_meaningful(self):
        """Better cross-domain than in-domain is possible; it must not be clamped."""
        assert GeneralizationGap(in_domain=0.6, cross_domain=0.8).difference == pytest.approx(-0.2)


class TestCalibrationMetrics:
    def test_defaults_to_none(self):
        calibration = CalibrationMetrics()
        assert calibration.is_available is False

    def test_accepts_measured_values(self):
        calibration = CalibrationMetrics(expected_calibration_error=0.04, brier_score=0.11)
        assert calibration.is_available

    def test_rejects_negative_values(self):
        with pytest.raises(ResultValidationError, match="non-negative"):
            CalibrationMetrics(brier_score=-0.1)


class TestExperimentResult:
    def test_absent_metrics_stay_none(self):
        result = ExperimentResult(experiment_id="e", run_id="r", seed=1)
        assert result.metrics == {}
        assert result.value("accuracy") is None
        assert result.has_metrics is False

    def test_missing_metrics_are_enumerated(self):
        result = ExperimentResult(
            experiment_id="e",
            run_id="r",
            seed=1,
            metrics=scalar_metrics(f1=0.8),
        )
        assert result.measured_metrics == ("f1",)
        assert "accuracy" in result.missing_metrics
        assert len(result.missing_metrics) == len(METRIC_NAMES) - 1

    def test_partial_metrics_are_preserved_as_nulls_on_serialization(self):
        result = ExperimentResult(
            experiment_id="e",
            run_id="r",
            seed=1,
            metrics=scalar_metrics(accuracy=0.9),
        )
        payload = result.to_dict()
        assert payload["metrics"]["accuracy"] == 0.9
        assert payload["metrics"]["roc_auc"] is None
        assert "roc_auc" in payload["metrics"]

    def test_rejects_unknown_metric_key(self):
        with pytest.raises(ResultValidationError, match="unknown metric"):
            ExperimentResult(
                experiment_id="e",
                run_id="r",
                seed=1,
                metrics={"f2": scalar_metrics(f1=0.5)["f1"]},
            )

    def test_rejects_metric_key_name_mismatch(self):
        with pytest.raises(ResultValidationError, match="mismatched name"):
            ExperimentResult(
                experiment_id="e",
                run_id="r",
                seed=1,
                metrics={"accuracy": scalar_metrics(f1=0.5)["f1"]},
            )

    def test_rejects_negative_evaluated_samples(self):
        with pytest.raises(ResultValidationError, match="evaluated_samples"):
            ExperimentResult(experiment_id="e", run_id="r", seed=1, evaluated_samples=-5)

    @pytest.mark.parametrize("field", ["experiment_id", "run_id"])
    def test_rejects_blank_identifiers(self, field):
        payload = {"experiment_id": "e", "run_id": "r", "seed": 1, field: " "}
        with pytest.raises(ResultValidationError, match=field):
            ExperimentResult(**payload)

    def test_generalization_gap_accessor(self):
        result = ExperimentResult(
            experiment_id="e",
            run_id="r",
            seed=1,
            generalization_gap=GeneralizationGap(in_domain=0.9, cross_domain=0.7),
        )
        assert result.generalization_gap_value == pytest.approx(0.2)

    def test_generalization_gap_absent_when_not_applicable(self):
        assert (
            ExperimentResult(experiment_id="e", run_id="r", seed=1).generalization_gap_value is None
        )

    def test_value_rejects_unknown_metric(self):
        with pytest.raises(ResultValidationError, match="unknown metric"):
            ExperimentResult(experiment_id="e", run_id="r", seed=1).value("bleu")

    def test_round_trips_through_dict(self):
        original = ExperimentResult(
            experiment_id="e",
            run_id="r",
            seed=1,
            run_number=2,
            evaluated_samples=100,
            metrics=scalar_metrics(accuracy=0.9, f1=0.8),
            confusion_matrix=ConfusionMatrix(true_negatives=10, true_positives=90),
            generalization_gap=GeneralizationGap(in_domain=0.95, cross_domain=0.8),
            calibration=CalibrationMetrics(brier_score=0.1),
            threshold=0.5,
            notes=("single seed only",),
        )
        restored = ExperimentResult.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_to_json_is_valid_json(self):
        result = ExperimentResult(
            experiment_id="e", run_id="r", seed=1, metrics=scalar_metrics(f1=0.7)
        )
        assert json.loads(result.to_json())["metrics"]["f1"] == 0.7

    def test_failed_run_produces_a_valid_result_with_null_metrics(self):
        """A partial run must serialize honestly, not as a row of zeros."""
        result = ExperimentResult(
            experiment_id="e",
            run_id="r",
            seed=1,
            evaluated_samples=0,
            notes=("run aborted during evaluation",),
        )
        payload = result.to_dict()
        assert payload["evaluated_samples"] == 0
        assert payload["metrics"]["accuracy"] is None
        assert result.has_metrics is False
