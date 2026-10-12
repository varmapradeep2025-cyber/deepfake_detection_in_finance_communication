"""Synthetic fixtures for the inference layer.

Kept dependency-free on purpose: the fake builder functions here are real import
targets for model cards, so the genuine ``load_model_backend`` path is exercised
without a deep-learning framework installed. The one module that must be importable
is this one, which only needs numpy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from datasets.preprocessing.inspection import ImageStreamInfo
from ml.image.preprocess import PixelImage
from ml.inference.models import ModelDescriptor, ModelOutput

#: Fixed clock so report timestamps are deterministic in tests.
FIXED_TIMESTAMP = "2026-01-01T00:00:00+00:00"


def fixed_clock() -> str:
    """A clock returning a constant timestamp."""
    return FIXED_TIMESTAMP


class FakeImageProbeDecoder:
    """Probe-only image decoder that reports a fixed layout for any readable file.

    Only ``probe`` is implemented, matching the :class:`ImageDecoder` protocol used by
    the inspector. It never touches the pixels, so it needs no codec.
    """

    name = "fake-probe"

    def __init__(
        self,
        width: int = 64,
        height: int = 48,
        channels: int = 3,
        color_mode: str = "BGR",
        container_format: str = "png",
    ) -> None:
        self.width = width
        self.height = height
        self.channels = channels
        self.color_mode = color_mode
        self.container_format = container_format

    def probe(self, path: Path) -> ImageStreamInfo:
        return ImageStreamInfo(
            width=self.width,
            height=self.height,
            channels=self.channels,
            color_mode=self.color_mode,
            format=self.container_format,
        )


class FakeImagePixelDecoder:
    """Pixel decoder returning a pre-built array, with a declared ordering."""

    name = "fake-pixels"

    def __init__(self, array: Any, color_space: str = "BGR") -> None:
        self._array = np.asarray(array)
        self._color_space = color_space

    def decode(self, path: Path) -> PixelImage:
        array = self._array
        height = int(array.shape[0])
        width = int(array.shape[1])
        channels = 1 if array.ndim == 2 else int(array.shape[2])
        return PixelImage(
            array=array,
            color_space=self._color_space,
            width=width,
            height=height,
            channels=channels,
        )


class FakeModelBackend:
    """A backend that returns a fixed, caller-controlled output."""

    def __init__(
        self,
        descriptor: ModelDescriptor,
        *,
        raw_score: float | None = 0.9,
        score_kind: str = "uncalibrated_score",
        calibrated: bool = False,
        confidence: float | None = None,
        labels: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
        extra: dict[str, Any] | None = None,
        calls: list[Any] | None = None,
    ) -> None:
        self.descriptor = descriptor
        self._raw_score = raw_score
        self._score_kind = score_kind
        self._calibrated = calibrated
        self._confidence = confidence
        self._labels = labels
        self._limitations = limitations
        self._extra = extra or {}
        self.calls = calls if calls is not None else []

    def predict(self, sample: Any) -> ModelOutput:
        self.calls.append(sample)
        return ModelOutput(
            model_id=self.descriptor.model_id,
            version=self.descriptor.version,
            score_name=self.descriptor.score_name,
            raw_score=self._raw_score,
            score_kind=self._score_kind,
            score_semantics=self.descriptor.score_semantics,
            calibrated=self._calibrated,
            decision_threshold=self.descriptor.decision_threshold,
            confidence=self._confidence,
            labels=self._labels,
            extra=dict(self._extra),
            limitations=self._limitations,
        )


class CrashingModelBackend:
    """A backend whose ``predict`` raises, to exercise the ``analysis_failed`` path."""

    def __init__(self, descriptor: ModelDescriptor) -> None:
        self.descriptor = descriptor

    def predict(self, sample: Any) -> ModelOutput:
        raise RuntimeError("synthetic backend failure")


# ---------------------------------------------------------------------------
# Entry points referenced by synthetic model cards
# ---------------------------------------------------------------------------


def build_scoring_backend(
    descriptor: ModelDescriptor,
    *,
    model_dir: Path,
    weights_path: Path | None,
) -> FakeModelBackend:
    """Entry point returning an uncalibrated scoring backend."""
    return FakeModelBackend(
        descriptor,
        raw_score=0.93,
        score_kind="uncalibrated_score",
        calibrated=False,
        confidence=None,
        limitations=("synthetic fixture: no real weights were loaded",),
    )


def build_calibrated_backend(
    descriptor: ModelDescriptor,
    *,
    model_dir: Path,
    weights_path: Path | None,
) -> FakeModelBackend:
    """Entry point returning a calibrated-probability backend."""
    return FakeModelBackend(
        descriptor,
        raw_score=0.42,
        score_kind="probability",
        calibrated=True,
        confidence=0.42,
    )


def build_crashing_backend(
    descriptor: ModelDescriptor,
    *,
    model_dir: Path,
    weights_path: Path | None,
) -> CrashingModelBackend:
    """Entry point returning a backend that fails at prediction time."""
    return CrashingModelBackend(descriptor)


def build_raising_entrypoint(
    descriptor: ModelDescriptor,
    *,
    model_dir: Path,
    weights_path: Path | None,
) -> FakeModelBackend:
    """Entry point that raises while constructing the backend."""
    raise RuntimeError("synthetic entrypoint failure")


def build_wrong_object(
    descriptor: ModelDescriptor,
    *,
    model_dir: Path,
    weights_path: Path | None,
) -> object:
    """Entry point returning something that is not a ModelBackend."""
    return object()


#: Module path used in synthetic model cards to reach the entry points above.
ENTRYPOINT_MODULE = "tests.synthetic_inference"


def write_model_card(
    models_dir: Path,
    model_id: str,
    *,
    entrypoint: str | None = None,
    framework: str = "numpy",
    modality: str = "image",
    version: str = "1.0.0",
    **overrides: Any,
) -> Path:
    """Write a ``model.json`` under ``models_dir/<model_id>/`` and return its path."""
    payload: dict[str, Any] = {
        "model_id": model_id,
        "version": version,
        "modality": modality,
        "framework": framework,
        "entrypoint": entrypoint,
    }
    payload.update(overrides)
    card = models_dir / model_id / "model.json"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text(json.dumps(payload), encoding="utf-8")
    return card


def make_image_array(width: int = 8, height: int = 6, channels: int = 3) -> Any:
    """A deterministic gradient image of the requested shape, ``uint8``."""
    if channels == 1:
        base = np.linspace(0, 255, width, dtype=np.uint8)
        return np.repeat(base[None, :], height, axis=0)
    array = np.zeros((height, width, channels), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    array[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    if channels == 4:
        array[:, :, 3] = 255
    return array


def make_image_analyzer(
    *,
    models_dir: Path,
    array: Any,
    color_space: str = "BGR",
    target_size: tuple[int, int] = (4, 4),
    registry: Any = None,
    policy: Any = None,
    clock: Any = fixed_clock,
    max_bytes: int | None = None,
) -> Any:
    """Build an :class:`ImageAnalyzer` wired entirely with fakes.

    The inspector uses a probe-only decoder and the preprocessor a pixel decoder, so
    no codec is needed and the pixel transform is fully controlled by ``array``.
    """
    from datasets.preprocessing.inspection import MediaInspector
    from ml.image.analyzer import ImageAnalyzer
    from ml.image.preprocess import ImagePreprocessConfig, ImagePreprocessor
    from ml.inference.models import ModelRegistry

    arr = np.asarray(array)
    height = int(arr.shape[0])
    width = int(arr.shape[1])
    channels = 1 if arr.ndim == 2 else int(arr.shape[2])
    probe = FakeImageProbeDecoder(
        width=width, height=height, channels=channels, color_mode=color_space
    )
    inspector = MediaInspector(None, image_decoder=probe, compute_hashes=True)
    preprocessor = ImagePreprocessor(
        FakeImagePixelDecoder(array, color_space),
        ImagePreprocessConfig(target_size=target_size),
    )
    if registry is None:
        registry = ModelRegistry.discover(Path(models_dir))

    kwargs: dict[str, Any] = {
        "inspector": inspector,
        "preprocessor": preprocessor,
        "registry": registry,
        "clock": clock,
    }
    if policy is not None:
        kwargs["policy"] = policy
    if max_bytes is not None:
        kwargs["max_bytes"] = max_bytes
    return ImageAnalyzer(**kwargs)
