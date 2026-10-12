"""Tests for :mod:`experiments.run_artifacts`.

Confirms the documented run directory layout is produced exactly, and that a run
record is self-consistent. These tests write only inside ``tmp_path``; no run is
created under ``experiments/runs/``.
"""

from __future__ import annotations

import json

import pytest

from experiments.results import ConfusionMatrix, ExperimentResult, GeneralizationGap, scalar_metrics
from experiments.run_artifacts import (
    REQUIRED_RUN_FILES,
    RunArtifacts,
    default_run_id,
    run_directory,
    validate_run_directory,
)
from experiments.schema import create_experiment_config
from ml.reproducibility.environment import capture_environment


def build_artifacts(directory):
    config = create_experiment_config(
        experiment_id="img-baseline-v0",
        name="image-baseline",
        description="Single-dataset image baseline.",
        modality="image",
        model_version="v0",
        training_dataset="dataset_a",
        validation_dataset="dataset_a",
        test_dataset="dataset_a",
        seed=20260101,
        run_number=0,
        manifests_used={"dataset_a": "sha256:" + "a" * 64},
        splits_used={"dataset_a": "sha256:" + "b" * 64},
    )
    result = ExperimentResult(
        experiment_id=config.experiment_id,
        run_id="seed20260101_run0_20260101T000000Z",
        seed=config.seed,
        run_number=config.run_number,
        evaluated_samples=100,
        metrics=scalar_metrics(accuracy=0.91, f1=0.88),
        confusion_matrix=ConfusionMatrix(
            true_negatives=45, false_positives=5, false_negatives=6, true_positives=44
        ),
        generalization_gap=GeneralizationGap(in_domain=0.91, cross_domain=0.78),
        notes=("single seed; not a final claim",),
    )
    return RunArtifacts(
        directory=directory,
        config=config,
        result=result,
        environment=capture_environment(timestamp="2026-01-01T00:00:00+00:00"),
        dataset_info={
            "datasets": {"dataset_a": {"manifest_hash": "sha256:" + "a" * 64}},
            "counts": {"total": 100, "real": 50, "fake": 50},
        },
    )


class TestLayout:
    def test_writes_all_five_artifacts(self, tmp_path):
        build_artifacts(run_directory(tmp_path, "img-baseline-v0", "run1")).write()
        for name in REQUIRED_RUN_FILES:
            assert (tmp_path / "img-baseline-v0" / "run1" / name).is_file()

    def test_required_files_are_the_documented_five(self):
        assert set(REQUIRED_RUN_FILES) == {
            "config.json",
            "metrics.json",
            "environment.json",
            "dataset_info.json",
            "README.md",
        }

    def test_json_artifacts_parse(self, tmp_path):
        directory = run_directory(tmp_path, "img-baseline-v0", "run1")
        build_artifacts(directory).write()
        for name in ("config.json", "metrics.json", "environment.json", "dataset_info.json"):
            assert json.loads((directory / name).read_text(encoding="utf-8"))

    def test_metrics_preserve_explicit_nulls(self, tmp_path):
        directory = run_directory(tmp_path, "img-baseline-v0", "run1")
        build_artifacts(directory).write()
        payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        assert payload["metrics"]["accuracy"] == 0.91
        assert payload["metrics"]["roc_auc"] is None

    def test_environment_records_git_absence_explicitly(self, tmp_path):
        directory = run_directory(tmp_path, "img-baseline-v0", "run1")
        build_artifacts(directory).write()
        payload = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
        assert "git_commit" in payload
        assert payload["python_version"]

    def test_readme_marks_unmeasured_metrics(self, tmp_path):
        directory = run_directory(tmp_path, "img-baseline-v0", "run1")
        build_artifacts(directory).write()
        readme = (directory / "README.md").read_text(encoding="utf-8")
        assert "accuracy" in readme
        assert "_not measured_" in readme

    def test_readme_includes_run_notes(self, tmp_path):
        directory = run_directory(tmp_path, "img-baseline-v0", "run1")
        build_artifacts(directory).write()
        assert "single seed" in (directory / "README.md").read_text(encoding="utf-8")


class TestImmutability:
    def test_refuses_to_overwrite_an_existing_run(self, tmp_path):
        directory = run_directory(tmp_path, "img-baseline-v0", "run1")
        build_artifacts(directory).write()
        with pytest.raises(FileExistsError, match="already exist"):
            build_artifacts(directory).write()

    def test_overwrite_when_explicitly_allowed(self, tmp_path):
        directory = run_directory(tmp_path, "img-baseline-v0", "run1")
        build_artifacts(directory).write()
        build_artifacts(directory).write(overwrite=True)

    def test_distinct_run_ids_do_not_collide(self, tmp_path):
        build_artifacts(run_directory(tmp_path, "img-baseline-v0", "run1")).write()
        build_artifacts(run_directory(tmp_path, "img-baseline-v0", "run2")).write()
        assert validate_run_directory(run_directory(tmp_path, "img-baseline-v0", "run1")) == []
        assert validate_run_directory(run_directory(tmp_path, "img-baseline-v0", "run2")) == []


class TestConsistency:
    def test_rejects_mismatched_experiment_id(self, tmp_path):
        artifacts = build_artifacts(tmp_path)
        bad_result = ExperimentResult(
            experiment_id="other",
            run_id=artifacts.result.run_id,
            seed=artifacts.config.seed,
        )
        with pytest.raises(ValueError, match="experiment_id"):
            RunArtifacts(
                directory=tmp_path,
                config=artifacts.config,
                result=bad_result,
                environment=artifacts.environment,
                dataset_info={},
            )

    def test_rejects_mismatched_seed(self, tmp_path):
        artifacts = build_artifacts(tmp_path)
        bad_result = ExperimentResult(
            experiment_id=artifacts.config.experiment_id,
            run_id="r",
            seed=999,
        )
        with pytest.raises(ValueError, match="seed"):
            RunArtifacts(
                directory=tmp_path,
                config=artifacts.config,
                result=bad_result,
                environment=artifacts.environment,
                dataset_info={},
            )

    def test_rejects_mismatched_run_number(self, tmp_path):
        artifacts = build_artifacts(tmp_path)
        bad_result = ExperimentResult(
            experiment_id=artifacts.config.experiment_id,
            run_id="r",
            seed=artifacts.config.seed,
            run_number=7,
        )
        with pytest.raises(ValueError, match="run_number"):
            RunArtifacts(
                directory=tmp_path,
                config=artifacts.config,
                result=bad_result,
                environment=artifacts.environment,
                dataset_info={},
            )


class TestHelpers:
    def test_default_run_id_encodes_seed_and_run_number(self):
        run_id = default_run_id(seed=20260101, run_number=2, timestamp="2026-01-01T00:00:00Z")
        assert run_id == "seed20260101_run2_20260101T000000Z"

    def test_run_id_is_unique_per_run_number(self):
        assert default_run_id(
            seed=1, run_number=0, timestamp="2026-01-01T00:00:00Z"
        ) != default_run_id(seed=1, run_number=1, timestamp="2026-01-01T00:00:00Z")

    def test_run_directory_nests_experiment_and_run(self, tmp_path):
        assert run_directory(tmp_path, "exp", "run1") == tmp_path / "exp" / "run1"

    @pytest.mark.parametrize("bad", ["", "..", "a/b", "a\\b", "."])
    def test_run_directory_rejects_unsafe_segments(self, tmp_path, bad):
        with pytest.raises(ValueError, match="safe path segment"):
            run_directory(tmp_path, bad, "run1")
        with pytest.raises(ValueError, match="safe path segment"):
            run_directory(tmp_path, "exp", bad)

    def test_validate_reports_missing_directory(self, tmp_path):
        assert validate_run_directory(tmp_path / "absent")

    def test_validate_reports_missing_artifacts(self, tmp_path):
        problems = validate_run_directory(tmp_path)
        assert any("missing required artifact" in problem for problem in problems)

    def test_validate_reports_malformed_json(self, tmp_path):
        directory = run_directory(tmp_path, "exp", "run1")
        build_artifacts(directory).write()
        (directory / "metrics.json").write_text("{broken", encoding="utf-8")
        assert any("not valid JSON" in problem for problem in validate_run_directory(directory))


def test_no_runs_are_generated_in_the_repository():
    """Importing the module must not create run directories in the repo."""
    from ml.reproducibility.config import project_root

    runs_root = project_root() / "experiments" / "runs"
    produced = [
        child for child in runs_root.iterdir() if child.is_dir() and child.name != "__pycache__"
    ]
    assert produced == [], f"unexpected run directories present: {produced}"
