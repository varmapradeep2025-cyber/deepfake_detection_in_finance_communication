"""Train the classical image baseline from a verified manifest and frozen split.

This script is the only place that turns real dataset bytes into the classical
image baseline. It is built to refuse rather than guess: every precondition that
could silently corrupt an experiment — an unregistered dataset, an unverified
licence, a manifest that does not match the split, overlapping splits, or source
recordings that appear in more than one split — stops the run before a single
feature is computed.

What it writes
--------------
* ``<models_dir>/<model_id>/weights.json`` — the fitted scaler and coefficients,
  as plain JSON.
* ``<models_dir>/<model_id>/model.json`` — the model card the inference loader
  reads, with a SHA-256 digest of the weights.
* With ``--record``, a run directory under the configured ``runs_dir`` holding the
  experiment config, result, environment, and dataset provenance.

What it never does
------------------
No media is copied or modified. No detector performance is claimed: the numbers it
records measure *this run*, on *these splits*, and nothing more.

Real runs require a verified, locally available dataset; the gates below enforce
that and print the exact reason when a precondition fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from datasets.manifest import calculate_file_sha256, calculate_manifest_hash, load_manifest
from datasets.registry import (
    LicenseStatus,
    RegistryStatus,
    default_registry_path,
    load_registry,
)
from datasets.schema import MediaSample, Modality
from datasets.split_schema import FrozenSplit, load_split_definition
from experiments.results import ExperimentResult
from experiments.run_artifacts import RunArtifacts, default_run_id, run_directory
from experiments.schema import ExperimentType, create_experiment_config
from ml.image.detectors.classical import ClassicalWeights, write_classical_weights
from ml.image.features import IMAGE_FEATURE_NAMES, ImageFeatureConfig, ImageFeatureExtractor
from ml.image.preprocess import ImagePreprocessor, default_image_pixel_decoder
from ml.image.training import (
    ClassifierConfig,
    ClassifierEvaluation,
    evaluate_classifier,
    select_threshold,
    train_classifier,
)
from ml.reproducibility.config import load_project_config
from ml.reproducibility.environment import capture_environment
from ml.reproducibility.seeds import set_global_seed

#: Registry statuses a dataset must have to be trained on.
_TRAINABLE_STATUSES = (RegistryStatus.APPROVED, RegistryStatus.AVAILABLE_LOCALLY)

#: The entrypoint advertised in the model card; resolved by the inference loader.
_ENTRYPOINT = "ml.image.detectors.classical:build"


class TrainingRefusedError(RuntimeError):
    """Raised when a precondition for training is not satisfied."""


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Kept separate so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog="python scripts/train_image_baseline.py",
        description=(
            "Train the classical (residual + DCT) image baseline on a verified "
            "manifest and frozen split, and record the run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Path to the image manifest.")
    parser.add_argument("--split", required=True, help="Path to the frozen split JSON.")
    parser.add_argument("--dataset-id", required=True, help="Registry dataset id.")
    parser.add_argument("--experiment-id", required=True, help="Experiment identifier.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--run-number", type=int, default=0, help="Run number for this seed.")
    parser.add_argument(
        "--media-root",
        default=None,
        help="Directory that relative manifest filepaths are resolved against.",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory to write the model artifact into (default from project.yaml).",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Directory to write run artifacts into (default from project.yaml).",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Dataset registry path (default from project.yaml).",
    )
    parser.add_argument(
        "--model-id",
        default="classical-image-baseline",
        help="Model directory / identifier.",
    )
    parser.add_argument("--model-version", default="0.1.0", help="Model version string.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations for logistic regression.",
    )
    parser.add_argument("--c", type=float, default=1.0, help="Inverse L2 strength.")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write a run directory with config, result, environment, and provenance.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary instead of text.",
    )
    return parser


def _refuse(message: str) -> NoReturn:
    raise TrainingRefusedError(message)


def _require_trainable_dataset(registry_path: Path | str | None, dataset_id: str) -> Any:
    registry = load_registry(registry_path)
    entry = registry.get(dataset_id)
    if entry is None:
        _refuse(
            f"dataset {dataset_id!r} is not registered; register and verify it before training"
        )
    if entry.license_status is not LicenseStatus.VERIFIED:
        _refuse(
            f"dataset {dataset_id!r} licence status is {entry.license_status.value!r}, not "
            "'verified'; training is refused until licensing is confirmed"
        )
    if entry.status not in _TRAINABLE_STATUSES:
        _refuse(
            f"dataset {dataset_id!r} status is {entry.status.value!r}; only "
            f"{[status.value for status in _TRAINABLE_STATUSES]} may be trained on"
        )
    return entry


def _check_recording_leakage(grouped: dict[str, tuple[MediaSample, ...]]) -> None:
    owners: dict[str, set[str]] = {}
    for name, samples in grouped.items():
        for sample in samples:
            token = (sample.source or "").strip()
            if not token:
                continue
            owners.setdefault(token, set()).add(name)
    leaking = {token: names for token, names in owners.items() if len(names) > 1}
    if leaking:
        example = ", ".join(
            f"{token!r} -> {sorted(names)}" for token, names in list(leaking.items())[:3]
        )
        _refuse(
            f"{len(leaking)} source recording(s) span more than one split, which leaks "
            f"identity across splits: {example}"
        )


def _check_labels(grouped: dict[str, tuple[MediaSample, ...]]) -> None:
    for name, samples in grouped.items():
        unlabeled = [s.sample_id for s in samples if s.label is None]
        if unlabeled:
            example = ", ".join(repr(i) for i in unlabeled[:3])
            _refuse(f"{len(unlabeled)} sample(s) in split {name!r} have no label: {example}")
        labels = {s.label.value for s in samples if s.label is not None}
        if len(labels) < 2:
            _refuse(
                f"split {name!r} contains only the label(s) {sorted(labels)}; training and "
                "evaluation need both classes in every split"
            )


def _load_split_groups(
    manifest_path: Path,
    split_path: Path,
) -> tuple[FrozenSplit, dict[str, tuple[MediaSample, ...]]]:
    samples = load_manifest(manifest_path)
    split = load_split_definition(split_path)
    if split.source_manifest_hash != calculate_manifest_hash(samples):
        _refuse(
            "the split's source_manifest_hash does not match the manifest; refusing to "
            "train on a split built from different data"
        )

    by_id = {sample.sample_id: sample for sample in samples}
    grouped: dict[str, tuple[MediaSample, ...]] = {}
    for name in ("train", "validation", "test"):
        ids = getattr(split, f"{name}_ids")
        missing = [i for i in ids if i not in by_id]
        if missing:
            example = ", ".join(repr(i) for i in missing[:3])
            _refuse(f"split {name} references {len(missing)} manifest-absent ids: {example}")
        if not ids:
            _refuse(f"split {name} is empty; every split must contain at least one id")
        selected = tuple(by_id[i] for i in ids)
        if any(sample.modality is not Modality.IMAGE for sample in selected):
            _refuse(f"split {name} contains non-image samples; this baseline is image only")
        grouped[name] = selected

    seen: dict[str, str] = {}
    for name, selected in grouped.items():
        for sample in selected:
            previous = seen.get(sample.sample_id)
            if previous is not None:
                _refuse(
                    f"sample {sample.sample_id!r} appears in both {previous!r} and {name!r}"
                )
            seen[sample.sample_id] = name

    _check_recording_leakage(grouped)
    _check_labels(grouped)
    return split, grouped


def _resolve_path(sample: MediaSample, media_root: Path | None, manifest_path: Path) -> Path:
    path = Path(sample.filepath)
    if path.is_absolute():
        return path
    root = media_root if media_root is not None else manifest_path.parent
    return root / path


def _extract_split_features(
    samples: tuple[MediaSample, ...],
    *,
    preprocessor: ImagePreprocessor,
    extractor: ImageFeatureExtractor,
    media_root: Path | None,
    manifest_path: Path,
) -> tuple[np.ndarray, list[str], int]:
    rows: list[tuple[float, ...]] = []
    labels: list[str] = []
    skipped = 0
    for sample in samples:
        path = _resolve_path(sample, media_root, manifest_path)
        try:
            preprocessed = preprocessor.preprocess(path)
        except Exception:
            skipped += 1
            continue
        representation = extractor.extract_features(preprocessed)
        if representation.feature_vector is None:
            skipped += 1
            continue
        rows.append(representation.feature_vector)
        labels.append(sample.label.value)
    if not rows:
        _refuse("no features could be extracted from this split; nothing to train on")
    return np.asarray(rows, dtype=np.float64), labels, skipped


def _summarize_evaluation(evaluation: ClassifierEvaluation) -> dict[str, Any]:
    return {
        "split": evaluation.split_name,
        "threshold": evaluation.threshold,
        "positive_class": evaluation.positive_class,
        "evaluated_samples": evaluation.evaluated_samples,
        "confusion_matrix": evaluation.confusion_matrix.to_dict(),
        "metrics": {name: metric.value for name, metric in evaluation.metrics.items()},
        "notes": list(evaluation.notes),
    }


def _model_card(
    args: argparse.Namespace,
    weights: ClassicalWeights,
    digest: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "model_id": args.model_id,
        "version": args.model_version,
        "modality": Modality.IMAGE.value,
        "framework": "numpy",
        "entrypoint": _ENTRYPOINT,
        "weights_file": "weights.json",
        "weights_sha256": digest,
        "input_size": [224, 224],
        "color_space": "RGB",
        "score_name": "manipulation_score",
        "score_kind": "uncalibrated_score",
        "score_semantics": (
            "Higher values indicate a greater likelihood of manipulation. The score is "
            "uncalibrated and is not a probability."
        ),
        "decision_threshold": threshold,
        "calibrated": False,
        "labels": list(weights.labels),
        "description": (
            "Classical image baseline: L2-regularized logistic regression over "
            "hand-crafted residual and DCT high-frequency features. A mechanism "
            "demonstrator, not a validated detector."
        ),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    project = load_project_config()
    manifest_path = Path(args.manifest).resolve()
    split_path = Path(args.split).resolve()
    media_root = Path(args.media_root).resolve() if args.media_root else None
    registry_path = Path(args.registry) if args.registry else default_registry_path()
    seed = args.seed if args.seed is not None else int(project.global_seed)

    _require_trainable_dataset(registry_path, args.dataset_id)
    split, grouped = _load_split_groups(manifest_path, split_path)

    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        models_dir = project.get_path("paths.generated.artifacts") / "models"
    if args.runs_root:
        runs_root = Path(args.runs_root)
    else:
        runs_root = project.root / str(
            project.get("reproducibility.runs_dir", "experiments/runs")
        )

    set_global_seed(seed)

    decoder = default_image_pixel_decoder()
    if decoder is None:
        _refuse("no image decoder is available (OpenCV is required to read image frames)")
    preprocessor = ImagePreprocessor(decoder)
    feature_config = ImageFeatureConfig()
    extractor = ImageFeatureExtractor(feature_config)

    train_features, train_labels, train_skipped = _extract_split_features(
        grouped["train"],
        preprocessor=preprocessor,
        extractor=extractor,
        media_root=media_root,
        manifest_path=manifest_path,
    )
    validation_features, validation_labels, validation_skipped = _extract_split_features(
        grouped["validation"],
        preprocessor=preprocessor,
        extractor=extractor,
        media_root=media_root,
        manifest_path=manifest_path,
    )
    test_features, test_labels, test_skipped = _extract_split_features(
        grouped["test"],
        preprocessor=preprocessor,
        extractor=extractor,
        media_root=media_root,
        manifest_path=manifest_path,
    )

    config = ClassifierConfig(c=args.c, max_iter=args.max_iter)
    fitted = train_classifier(
        train_features,
        train_labels,
        feature_names=IMAGE_FEATURE_NAMES,
        config=config,
        seed=seed,
    )
    selection = select_threshold(
        fitted.score(validation_features),
        validation_labels,
        positive_label=config.positive_label,
    )
    validation_eval = evaluate_classifier(
        fitted,
        validation_features,
        validation_labels,
        threshold=selection.threshold,
        split_name="validation",
    )
    test_eval = evaluate_classifier(
        fitted,
        test_features,
        test_labels,
        threshold=selection.threshold,
        split_name="test",
    )

    now = datetime.now(UTC).isoformat(timespec="seconds")
    manifest_hash = calculate_manifest_hash(load_manifest(manifest_path))
    weights = ClassicalWeights.from_fitted(
        fitted,
        feature_config=feature_config,
        threshold=selection.threshold,
        training={
            "dataset_id": args.dataset_id,
            "manifest": str(manifest_path),
            "manifest_hash": manifest_hash,
            "split": str(split_path),
            "split_hash": split.content_hash(),
            "seed": seed,
            "run_number": args.run_number,
            "train_samples": int(train_features.shape[0]),
            "validation_samples": int(validation_features.shape[0]),
            "test_samples": int(test_features.shape[0]),
        },
        created_at=now,
    )

    model_dir = models_dir / args.model_id
    weights_path = write_classical_weights(weights, model_dir / "weights.json")
    digest = calculate_file_sha256(weights_path)
    card = _model_card(args, weights, digest, selection.threshold)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "message": (
            f"trained {args.model_id} {args.model_version}: "
            f"{train_features.shape[0]} train / {validation_features.shape[0]} validation / "
            f"{test_features.shape[0]} test samples"
        ),
        "model": {
            "model_id": args.model_id,
            "version": args.model_version,
            "directory": str(model_dir),
            "weights": str(weights_path),
            "weights_sha256": digest,
        },
        "counts": {
            "train": int(train_features.shape[0]),
            "validation": int(validation_features.shape[0]),
            "test": int(test_features.shape[0]),
            "skipped": {
                "train": train_skipped,
                "validation": validation_skipped,
                "test": test_skipped,
            },
        },
        "threshold": selection.threshold,
        "threshold_selection": {
            "balanced_accuracy": selection.balanced_accuracy,
            "youden_j": selection.youden_j,
            "candidates": selection.candidates,
        },
        "validation": _summarize_evaluation(validation_eval),
        "test": _summarize_evaluation(test_eval),
        "recorded": bool(args.record),
    }

    if args.record:
        run_id = default_run_id(seed=seed, run_number=args.run_number)
        experiment_config = create_experiment_config(
            experiment_id=args.experiment_id,
            name=f"Classical image baseline ({args.dataset_id})",
            description=(
                "Residual + DCT hand-crafted features with L2 logistic regression, "
                "trained and evaluated on a frozen split."
            ),
            modality=Modality.IMAGE,
            model_version=args.model_version,
            training_dataset=args.dataset_id,
            validation_dataset=args.dataset_id,
            test_dataset=args.dataset_id,
            seed=seed,
            run_number=args.run_number,
            experiment_type=ExperimentType.KNOWN_DATASET,
            created_at=now,
            tags=("classical", "image", "baseline"),
            manifests_used={str(manifest_path): manifest_hash},
            splits_used={str(split_path): split.content_hash()},
            config={
                "classifier": config.to_dict(),
                "feature_config": feature_config.to_dict(),
                "feature_names": list(IMAGE_FEATURE_NAMES),
                "threshold": selection.threshold,
            },
        )
        experiment_result = ExperimentResult(
            experiment_id=args.experiment_id,
            run_id=run_id,
            seed=seed,
            run_number=args.run_number,
            evaluation_split="test",
            positive_class=config.positive_label,
            evaluated_samples=test_eval.evaluated_samples,
            metrics=test_eval.metrics,
            confusion_matrix=test_eval.confusion_matrix,
            threshold=selection.threshold,
            notes=tuple(list(validation_eval.notes) + list(test_eval.notes)),
            completed_at=now,
        )
        directory = run_directory(runs_root, args.experiment_id, run_id)
        RunArtifacts(
            directory=directory,
            config=experiment_config,
            result=experiment_result,
            environment=capture_environment(timestamp=now),
            dataset_info={
                "dataset_id": args.dataset_id,
                "manifest": str(manifest_path),
                "manifest_hash": manifest_hash,
                "split": str(split_path),
                "split_hash": split.content_hash(),
                "counts": {
                    "train": len(grouped["train"]),
                    "validation": len(grouped["validation"]),
                    "test": len(grouped["test"]),
                },
            },
        ).write()
        summary["run"] = {"run_id": run_id, "directory": str(directory)}

    return summary


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 2 when a precondition refuses the run."""
    args = build_parser().parse_args(argv)
    try:
        summary = _run(args)
    except TrainingRefusedError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(summary.get("message", "training complete"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
