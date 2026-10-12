"""Experiment layer: configuration schema, result schema, and run artifact layout.

Module map
----------
:mod:`experiments.schema`
    :class:`~experiments.schema.ExperimentConfig` and the four experiment families
    (known-dataset, cross-dataset, ablation, multimodal).
:mod:`experiments.results`
    :class:`~experiments.results.ExperimentResult` and the metrics types. Absent
    metrics stay ``None``; nothing is inferred or back-filled.
:mod:`experiments.run_artifacts`
    The ``experiments/runs/<experiment_id>/<run_id>/`` layout and its IO.

The per-research-question subdirectories (``experiments/image/``,
``experiments/cross_dataset/``, and so on) currently hold only ``.gitkeep`` and are
awaiting the corresponding modules. No experiment is executed here.
"""

from experiments.results import (
    CalibrationMetrics,
    ConfusionMatrix,
    ExperimentResult,
    GeneralizationGap,
    MetricOutOfRangeError,
    RateMetric,
    ResultValidationError,
    ScalarMetric,
    scalar_metrics,
)
from experiments.run_artifacts import (
    REQUIRED_RUN_FILES,
    RunArtifacts,
    default_run_id,
    run_directory,
    validate_run_directory,
)
from experiments.schema import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentConfig,
    ExperimentType,
    ExperimentValidationError,
    create_experiment_config,
    validate_experiment_config,
)

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "REQUIRED_RUN_FILES",
    "CalibrationMetrics",
    "ConfusionMatrix",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentType",
    "ExperimentValidationError",
    "GeneralizationGap",
    "MetricOutOfRangeError",
    "RateMetric",
    "ResultValidationError",
    "RunArtifacts",
    "ScalarMetric",
    "create_experiment_config",
    "default_run_id",
    "run_directory",
    "scalar_metrics",
    "validate_experiment_config",
    "validate_run_directory",
]
