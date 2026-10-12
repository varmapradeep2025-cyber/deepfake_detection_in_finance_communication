"""Still-image deepfake detection: preprocessing and the analyzer.

The first implemented modality. It provides an injectable pixel decoder, a
deterministic model-agnostic preprocessing transform, and :class:`ImageAnalyzer`,
which composes validation, preprocessing, model inference, risk assessment, and
reporting into one call.
"""

from __future__ import annotations

from ml.image.analyzer import DEFAULT_MAX_IMAGE_BYTES, ImageAnalyzer
from ml.image.preprocess import (
    ImagePixelDecoder,
    ImagePreprocessConfig,
    ImagePreprocessError,
    ImagePreprocessor,
    OpenCVImagePixelDecoder,
    PixelImage,
    PreprocessedImage,
    default_image_pixel_decoder,
)

__all__ = [
    "DEFAULT_MAX_IMAGE_BYTES",
    "ImageAnalyzer",
    "ImagePixelDecoder",
    "ImagePreprocessConfig",
    "ImagePreprocessError",
    "ImagePreprocessor",
    "OpenCVImagePixelDecoder",
    "PixelImage",
    "PreprocessedImage",
    "default_image_pixel_decoder",
]
