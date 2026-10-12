"""The image analyzer: validation, preprocessing, inference, and reporting.

This is the concrete, end-to-end path for one modality. It composes the pieces — the
inspector for validation, :class:`~ml.image.preprocess.ImagePreprocessor` for the
model input, a :class:`~ml.inference.models.ModelRegistry` for model selection — into
a single call that always returns a report, never an exception and never a
fabricated score.

The failure taxonomy, in order of checking:

* **invalid_input** — the file is missing, empty, over the size limit, not decodable,
  or its pixels cannot be transformed. Nothing could be analyzed.
* **model_unavailable** — the input was fine and preprocessing succeeded, but no
  usable model exists: none is registered, the requested id is unknown, or the model
  cannot be loaded (missing weights, missing framework, missing entry point). The
  report still carries the observed and derived evidence, because the artifact itself
  was successfully read.
* **analysis_failed** — a model was loaded and run, but it raised. The input may be
  fine; the fault is in the model layer.
* **analysis_complete** — a model produced output, which is recorded with its
  semantics, limitations, and a risk assessment derived from it.

Observed evidence is captured before any decision, so even a request answered with
"no model" still records the file's digest, size, and dimensions.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets.preprocessing.inspection import MediaInspector
from datasets.schema import Modality
from ml.image.preprocess import ImagePreprocessor
from ml.inference.evidence import ClaimKind, EvidenceItem, EvidenceType
from ml.inference.modality import AnalysisModality
from ml.inference.models import (
    ModelDescriptor,
    ModelOutput,
    ModelRegistry,
    ModelUnavailableError,
    load_model_backend,
)
from ml.inference.report import DEFAULT_RECOMMENDATION, AnalysisReport
from ml.inference.request import AnalysisRequest, new_request_id, utc_now
from ml.inference.status import AnalysisStatus
from ml.risk.assess import RiskAssessment, RiskPolicy, assess_risk

__all__ = [
    "DEFAULT_MAX_IMAGE_BYTES",
    "ImageAnalyzer",
]

#: Default input ceiling for a still image, 25 MiB. Matches the frontend's image
#: limit; overridable per analyzer so the policy lives with the caller, not the code.
DEFAULT_MAX_IMAGE_BYTES = 25 * 1024 * 1024


class ImageAnalyzer:
    """Analyze one still image and return a complete, honest report.

    Args:
        inspector: Inspector used for validation and observed metadata. Its image
            decoder must be configured for validation to succeed.
        preprocessor: Turns decoded pixels into the model-ready array.
        registry: Supplies the model cards to choose from.
        policy: Risk policy applied to a score. Defaults to the shared default.
        max_bytes: Input size ceiling in bytes.
        clock: Returns an RFC 3339 UTC timestamp. Injected so tests are deterministic.
    """

    def __init__(
        self,
        *,
        inspector: MediaInspector,
        preprocessor: ImagePreprocessor,
        registry: ModelRegistry,
        policy: RiskPolicy | None = None,
        max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self._inspector = inspector
        self._preprocessor = preprocessor
        self._registry = registry
        self._policy = policy
        self._max_bytes = max_bytes
        self._clock = clock

    @classmethod
    def from_project_config(
        cls,
        config: Any | None = None,
        *,
        models_dir: Path | str | None = None,
    ) -> ImageAnalyzer:
        """Build an analyzer wired from ``configs/project.yaml``.

        Resolves the models directory from ``paths.generated.artifacts``, the target
        size from ``data.preprocessing.image.target_size``, and builds the inspector
        and preprocessor with their default (OpenCV-backed) decoders. When OpenCV is
        absent the decoders are ``None`` and analysis reports that honestly rather
        than failing to construct.

        Args:
            config: A loaded project config, or ``None`` to load the default.
            models_dir: Directory to scan for model cards. Defaults to
                ``<paths.generated.artifacts>/models``. Passed in explicitly by a
                caller (such as the backend) that keeps its models elsewhere.
        """
        from ml.image.preprocess import (
            ImagePreprocessConfig,
            ImagePreprocessor,
            default_image_pixel_decoder,
        )

        if config is None:
            from ml.reproducibility.config import load_project_config

            config = load_project_config()

        if models_dir is None:
            artifacts_root = config.get_path("paths.generated.artifacts")
            models_dir = Path(artifacts_root) / "models"
        registry = ModelRegistry.discover(Path(models_dir))

        raw_size = config.get("data.preprocessing.image.target_size", [224, 224])
        target_size = (int(raw_size[0]), int(raw_size[1]))
        color_space = str(config.get("data.preprocessing.image.color_space", "RGB"))
        preprocess_config = ImagePreprocessConfig(
            target_size=target_size, target_color_space=color_space
        )

        decoder = default_image_pixel_decoder()
        if decoder is None:
            raise RuntimeError(
                "no image pixel decoder is available; install opencv-python-headless "
                "('pip install -r requirements.txt') to run image analysis"
            )
        preprocessor = ImagePreprocessor(decoder, preprocess_config)

        return cls(
            inspector=MediaInspector.from_project_config(config),
            preprocessor=preprocessor,
            registry=registry,
        )

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        """Analyze the image named by ``request`` and return a report."""
        request_id = new_request_id()
        started = self._clock()
        path = Path(request.path)
        evidence: list[EvidenceItem] = []
        warnings: list[str] = []
        errors: list[str] = []
        counter = itertools.count(1)

        warnings.extend(self._registry.problems)

        inspection = self._inspector.inspect_image(path)
        if inspection.sha256 is not None:
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.HASH,
                    label="File SHA-256",
                    value=inspection.sha256,
                    source=str(path),
                    method="sha256(file bytes)",
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
        if inspection.file_size is not None:
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label="File size",
                    value=f"{inspection.file_size} bytes",
                    source=str(path),
                    method="stat",
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
        warnings.extend(inspection.warnings)

        if not inspection.is_usable:
            errors.append(inspection.error or f"inspection failed: {inspection.status.value}")
            return self._report(
                request_id,
                AnalysisStatus.INVALID_INPUT,
                request.path,
                started,
                evidence,
                warnings,
                errors,
            )

        if inspection.file_size is not None and inspection.file_size > self._max_bytes:
            errors.append(
                f"file is {inspection.file_size} bytes, over the {self._max_bytes}-byte limit"
            )
            return self._report(
                request_id,
                AnalysisStatus.INVALID_INPUT,
                request.path,
                started,
                evidence,
                warnings,
                errors,
            )

        if inspection.width is not None and inspection.height is not None:
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label="Dimensions",
                    value=f"{inspection.width}x{inspection.height}",
                    source=str(path),
                    method=f"decoder:{self._inspector.image_decoder_name}",
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
        if inspection.channels is not None:
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label="Channel layout",
                    value=f"{inspection.channels} channel(s), {inspection.color_mode or 'unknown'}",
                    source=str(path),
                    method=f"decoder:{self._inspector.image_decoder_name}",
                    claim_kind=ClaimKind.OBSERVED,
                )
            )

        try:
            preprocessed = self._preprocessor.preprocess(path)
        except Exception as exc:
            errors.append(f"could not decode pixels: {exc}")
            return self._report(
                request_id,
                AnalysisStatus.INVALID_INPUT,
                request.path,
                started,
                evidence,
                warnings,
                errors,
            )

        evidence.append(
            self._item(
                counter,
                evidence_type=EvidenceType.HASH,
                label="Preprocessed input SHA-256",
                value=preprocessed.digest,
                source=str(path),
                method=preprocessed.method,
                claim_kind=ClaimKind.DERIVED,
            )
        )
        evidence.append(
            self._item(
                counter,
                evidence_type=EvidenceType.METADATA,
                label="Preprocessed shape",
                value=f"{preprocessed.height}x{preprocessed.width}x{preprocessed.channels} "
                f"{preprocessed.color_space} float32",
                source=str(path),
                method=preprocessed.method,
                claim_kind=ClaimKind.DERIVED,
            )
        )
        if preprocessed.source_channels == 4:
            warnings.append(
                "source image had an alpha channel, which was discarded during conversion "
                "to RGB"
            )

        descriptor, selection_error = self._select_model(request)
        if descriptor is None:
            errors.append(selection_error)
            return self._report(
                request_id,
                AnalysisStatus.MODEL_UNAVAILABLE,
                request.path,
                started,
                evidence,
                warnings,
                errors,
                preprocessing=preprocessed.to_summary(),
            )

        evidence.append(
            self._item(
                counter,
                evidence_type=EvidenceType.PROVENANCE,
                label="Model selected",
                value=descriptor.qualified_id,
                source="model registry",
                method="ModelRegistry.for_modality",
                claim_kind=ClaimKind.OBSERVED,
            )
        )

        try:
            backend = load_model_backend(descriptor, models_dir=self._registry.models_dir)
        except ModelUnavailableError as exc:
            errors.append(f"model {descriptor.qualified_id} could not be loaded: {exc}")
            warnings.append(f"unavailable reason: {exc.reason}")
            return self._report(
                request_id,
                AnalysisStatus.MODEL_UNAVAILABLE,
                request.path,
                started,
                evidence,
                warnings,
                errors,
                model=descriptor,
                preprocessing=preprocessed.to_summary(),
            )

        try:
            output = backend.predict(preprocessed)
        except Exception as exc:
            errors.append(f"model {descriptor.qualified_id} failed to produce an output: {exc}")
            return self._report(
                request_id,
                AnalysisStatus.ANALYSIS_FAILED,
                request.path,
                started,
                evidence,
                warnings,
                errors,
                model=descriptor,
                preprocessing=preprocessed.to_summary(),
            )

        if output.model_id != descriptor.model_id or output.version != descriptor.version:
            warnings.append(
                f"model reported identity {output.model_id}:{output.version}, which does "
                f"not match the loaded card {descriptor.qualified_id}"
            )
        if output.raw_score is None:
            warnings.append("the model produced no score")
        if not output.calibrated:
            warnings.append(
                "the score is uncalibrated and must not be read as a probability; it is an "
                "operating point on the detector, not a likelihood"
            )
        warnings.extend(output.limitations)

        evidence.append(
            self._item(
                counter,
                evidence_type=EvidenceType.SIGNAL,
                label=f"Model score ({output.score_name})",
                value=(
                    f"{output.score_name}={output.raw_score!r} "
                    f"({output.score_kind})"
                ),
                source=f"model:{descriptor.qualified_id}",
                method=f"backend:{descriptor.framework}",
                claim_kind=ClaimKind.INFERRED,
                confidence=output.confidence,
            )
        )

        risk = assess_risk(
            output.raw_score,
            score_kind=output.score_kind,
            calibrated=output.calibrated,
            policy=self._policy,
            unavailable_reason="the model produced no score",
        )

        return self._report(
            request_id,
            AnalysisStatus.ANALYSIS_COMPLETE,
            request.path,
            started,
            evidence,
            warnings,
            errors,
            model=descriptor,
            output=output,
            risk=risk,
            preprocessing=preprocessed.to_summary(),
        )

    # --- internals --------------------------------------------------------

    def _select_model(self, request: AnalysisRequest) -> tuple[ModelDescriptor | None, str]:
        """Choose a model card, or return ``(None, reason)``."""
        if request.model_id is not None:
            descriptor = self._registry.get(request.model_id)
            if descriptor is None:
                return None, f"requested model {request.model_id!r} is not registered"
            if descriptor.modality is not Modality.IMAGE:
                return (
                    None,
                    (
                        f"model {descriptor.qualified_id!r} is for modality "
                        f"{descriptor.modality.value!r}, not image"
                    ),
                )
            return descriptor, ""

        candidates = self._registry.for_modality(Modality.IMAGE)
        if not candidates:
            return (
                None,
                (
                    "no image model is registered; add a model card at "
                    f"{self._registry.models_dir / '<model_id>' / 'model.json'}"
                ),
            )
        return candidates[0], ""

    def _item(
        self,
        counter: itertools.count[int],
        *,
        evidence_type: EvidenceType,
        label: str,
        value: str,
        source: str,
        method: str,
        claim_kind: ClaimKind,
        confidence: float | None = None,
    ) -> EvidenceItem:
        return EvidenceItem(
            id=f"ev-{next(counter):03d}",
            type=evidence_type,
            label=label,
            value=value,
            source=source,
            method=method,
            timestamp=self._clock(),
            confidence=confidence,
            claim_kind=claim_kind,
        )

    def _report(
        self,
        request_id: str,
        status: AnalysisStatus,
        path: str,
        started: str,
        evidence: list[EvidenceItem],
        warnings: list[str],
        errors: list[str],
        *,
        model: ModelDescriptor | None = None,
        output: ModelOutput | None = None,
        risk: RiskAssessment | None = None,
        preprocessing: dict[str, Any] | None = None,
    ) -> AnalysisReport:
        return AnalysisReport(
            request_id=request_id,
            status=status,
            modality=AnalysisModality.IMAGE,
            path=path,
            model=model,
            output=output,
            risk=risk,
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            errors=tuple(errors),
            recommendation=DEFAULT_RECOMMENDATION,
            started_at=started,
            completed_at=self._clock(),
            preprocessing=preprocessing,
        )
