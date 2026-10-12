"""Image pixel decoding and preprocessing into a model-ready array.

Inspection tells you *about* an image; this module turns one into the array a model
consumes. The two are kept separate because inspection must run with no image library
installed, while producing pixels obviously cannot.

Decoding is injected
--------------------
Nothing here imports OpenCV at module import time. :class:`OpenCVImagePixelDecoder`
imports it lazily, and the preprocessor takes any :class:`ImagePixelDecoder`, so the
transform is testable with a fake decoder and no codec.

What preprocessing is, and is not
---------------------------------
This is a **model-agnostic default**: decode, convert to a target colour order,
resize to a target size by nearest-neighbour sampling, and scale to ``float32`` in
``[0, 1]``. It is deterministic and dependency-light on purpose. A real detector's
input transform — its normalisation constants, interpolation, cropping — is part of
that model's training contract and must come from the model, not be assumed here.
The backend is therefore handed the preprocessed array *and* told exactly how it was
produced, so it can apply its own transform or reject an unsuitable input.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ImagePixelDecoder",
    "ImagePreprocessConfig",
    "ImagePreprocessError",
    "ImagePreprocessor",
    "OpenCVImagePixelDecoder",
    "PixelImage",
    "PreprocessedImage",
    "default_image_pixel_decoder",
]

#: OpenCV channel count to the ordering name OpenCV uses.
_OPENCV_COLOR_MODES: dict[int, str] = {1: "L", 2: "LA", 3: "BGR", 4: "BGRA"}


class ImagePreprocessError(RuntimeError):
    """Raised when pixels cannot be turned into the target array."""


@dataclass(frozen=True, slots=True)
class PixelImage:
    """Decoded pixels and the ordering the decoder produced them in.

    Attributes:
        array: ``(H, W)`` or ``(H, W, C)`` pixel data as returned by the decoder.
        color_space: The decoder's native ordering, e.g. ``"BGR"`` or ``"L"``.
            Recorded verbatim, because assuming RGB when the decoder produced BGR
            corrupts every downstream pixel.
        width, height, channels: The array's dimensions.
    """

    array: Any
    color_space: str
    width: int
    height: int
    channels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "color_space": self.color_space,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
        }


@runtime_checkable
class ImagePixelDecoder(Protocol):
    """What preprocessing needs from an image decoder: pixels, with their ordering."""

    #: Short stable identifier recorded in the preprocessing summary.
    name: str

    def decode(self, path: Path) -> PixelImage:
        """Return the image's pixels.

        Raises:
            Exception: If the file cannot be decoded.
        """
        ...


class OpenCVImagePixelDecoder:
    """An :class:`ImagePixelDecoder` backed by headless OpenCV.

    ``imread`` with ``IMREAD_UNCHANGED`` keeps the stored channel count, so a
    grayscale image stays one channel and an RGBA image keeps its alpha rather than
    being silently flattened. The reported :attr:`PixelImage.color_space` is OpenCV's
    native ordering (``"BGR"``/``"BGRA"``), not the RGB a human might assume.
    """

    name = "opencv"

    def __init__(self) -> None:
        self._cv2: Any | None = None

    def _module(self) -> Any:
        if self._cv2 is None:
            try:
                import cv2
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImagePreprocessError(
                    "opencv-python-headless is not installed. Install it with "
                    "'pip install -r requirements.txt', or inject another "
                    "ImagePixelDecoder."
                ) from exc
            self._cv2 = cv2
        return self._cv2

    def decode(self, path: Path) -> PixelImage:
        cv2 = self._module()
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ImagePreprocessError(f"OpenCV could not decode {path} as an image")
        shape = np.asarray(array).shape
        height = int(shape[0]) if len(shape) >= 1 else 0
        width = int(shape[1]) if len(shape) >= 2 else 0
        channels = 1 if np.asarray(array).ndim == 2 else int(shape[2])
        if width <= 0 or height <= 0:
            raise ImagePreprocessError(f"OpenCV reported a zero-sized image for {path}")
        return PixelImage(
            array=array,
            color_space=_OPENCV_COLOR_MODES.get(channels, f"{channels}-channel"),
            width=width,
            height=height,
            channels=channels,
        )


def default_image_pixel_decoder() -> ImagePixelDecoder | None:
    """Return the OpenCV pixel decoder, or ``None`` when OpenCV is not installed.

    Same contract as the inspection decoders: ``None`` is a first-class answer that
    lets a caller report "no decoder available" instead of failing to import.
    """
    try:
        import cv2  # noqa: F401  (probe only)
    except ImportError:
        return None
    return OpenCVImagePixelDecoder()


@dataclass(frozen=True, slots=True)
class ImagePreprocessConfig:
    """Target size and colour order for preprocessing.

    Attributes:
        target_size: ``(height, width)`` to resize to.
        target_color_space: Channel order to produce. Only ``"RGB"`` is supported.
    """

    target_size: tuple[int, int] = (224, 224)
    target_color_space: str = "RGB"

    def __post_init__(self) -> None:
        height, width = self.target_size
        if height <= 0 or width <= 0:
            raise ValueError(f"target_size must be positive, got {self.target_size}")
        if self.target_color_space.strip().upper() != "RGB":
            raise ValueError(
                "only an RGB target is supported; a model needing another order must "
                f"be handed the source ordering, not {self.target_color_space!r}"
            )


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """A model-ready array plus the record of how it was produced.

    Attributes:
        array: ``float32`` array of shape ``(H, W, 3)`` in ``[0, 1]``.
        width, height, channels: The array's dimensions.
        color_space: The ordering produced, ``"RGB"``.
        source_width, source_height, source_channels, source_color_space: The decoded
            image's dimensions and ordering, kept so an input can be traced back.
        digest: SHA-256 of the array's bytes, a stable fingerprint of the exact
            tensor a model would see.
        method: Human-readable description of the transform applied.
    """

    array: Any
    width: int
    height: int
    channels: int
    color_space: str
    source_width: int
    source_height: int
    source_channels: int
    source_color_space: str
    digest: str
    method: str

    def value_range(self) -> tuple[float, float]:
        """``(min, max)`` of the array, for the preprocessing summary."""
        flat = np.asarray(self.array, dtype=np.float32)
        return (float(flat.min()), float(flat.max()))

    def to_summary(self) -> dict[str, Any]:
        """JSON-serializable summary, without the pixel buffer itself."""
        low, high = self.value_range()
        return {
            "method": self.method,
            "color_space": self.color_space,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_channels": self.source_channels,
            "source_color_space": self.source_color_space,
            "dtype": "float32",
            "value_range": [low, high],
            "digest_sha256": self.digest,
        }


class ImagePreprocessor:
    """Turns a decoded image into a fixed-size ``float32`` RGB array.

    Args:
        decoder: Pixel decoder. Injected so the transform is testable without a codec.
        config: Target size and colour order.
    """

    def __init__(
        self,
        decoder: ImagePixelDecoder,
        config: ImagePreprocessConfig | None = None,
    ) -> None:
        self._decoder = decoder
        self._config = config or ImagePreprocessConfig()

    @property
    def decoder_name(self) -> str:
        """Name of the injected pixel decoder."""
        return self._decoder.name

    def preprocess(self, path: Path | str) -> PreprocessedImage:
        """Decode and transform one image.

        Raises:
            ImagePreprocessError: If the pixels cannot be transformed. Decoder
                failures propagate from the injected decoder unchanged.
        """
        source = self._decoder.decode(Path(path))
        array = np.asarray(source.array)
        if array.ndim < 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ImagePreprocessError(
                f"decoded image for {path} has no pixels (shape {array.shape})"
            )

        rgb = _to_rgb(array, source.color_space)
        target_height, target_width = self._config.target_size
        resized = _resize_nearest(rgb, target_height, target_width)
        scaled = (resized.astype(np.float32) / np.float32(255.0))
        scaled = np.ascontiguousarray(scaled)
        digest = hashlib.sha256(scaled.tobytes()).hexdigest()

        return PreprocessedImage(
            array=scaled,
            width=target_width,
            height=target_height,
            channels=int(scaled.shape[2]),
            color_space=self._config.target_color_space.strip().upper(),
            source_width=source.width,
            source_height=source.height,
            source_channels=source.channels,
            source_color_space=source.color_space,
            digest=digest,
            method=f"{self._decoder.name}+nearest-resize+scale01",
        )


def _to_rgb(array: Any, color_space: str) -> Any:
    """Convert decoded pixels to three-channel RGB, or raise if ordering is unknown."""
    normalized = color_space.strip().upper() if isinstance(color_space, str) else ""
    channels = 1 if array.ndim == 2 else int(array.shape[2])

    if channels == 1:
        return np.repeat(array[:, :, np.newaxis], 3, axis=2)
    if channels == 2:
        return np.repeat(array[:, :, :1], 3, axis=2)
    if channels == 3:
        if normalized == "RGB":
            return array
        if normalized == "BGR":
            return array[:, :, ::-1]
        raise ImagePreprocessError(
            f"cannot convert {channels}-channel data to RGB: decoder reported colour "
            f"order {color_space!r}, which is neither 'RGB' nor 'BGR'"
        )
    if channels == 4:
        if normalized == "RGBA":
            return array[:, :, :3]
        if normalized == "BGRA":
            return array[:, :, :3][:, :, ::-1]
        raise ImagePreprocessError(
            f"cannot convert {channels}-channel data to RGB: decoder reported colour "
            f"order {color_space!r}, which is neither 'RGBA' nor 'BGRA'"
        )
    raise ImagePreprocessError(f"cannot convert {channels}-channel data to RGB")


def _resize_nearest(array: Any, target_height: int, target_width: int) -> Any:
    """Resize by nearest-neighbour sampling.

    Nearest-neighbour over an area average because it is exactly reproducible with no
    floating-point accumulation order to vary between machines, and the choice is
    recorded in the preprocessing summary rather than hidden.
    """
    source_height = int(array.shape[0])
    source_width = int(array.shape[1])
    if source_height == target_height and source_width == target_width:
        return array
    row_index = (np.arange(target_height) * source_height / target_height).astype(np.intp)
    column_index = (np.arange(target_width) * source_width / target_width).astype(np.intp)
    return array[row_index][:, column_index]
