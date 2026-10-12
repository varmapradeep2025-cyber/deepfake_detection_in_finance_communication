"""Deterministic hand-crafted image features for the classical baseline.

Why hand-crafted first
----------------------
The foundation stage deliberately ships no deep-learning stack. A baseline that
learns from *explainable, dependency-light* features gives the pipeline something
honest to measure before a neural encoder exists, and it fixes the evaluation
contract (splits, metrics, run records) independently of whichever encoder arrives
later.

Two feature families only
-------------------------
The project's plan named exactly two families to start with, and this module adds
no third:

``residual``
    Statistics of a fixed Laplacian high-pass response of the luminance channel.
    Manipulation and re-compression tend to disturb the local high-frequency
    structure that this response isolates.
``dct``
    Per-block, orthonormal DCT-II energies, summarized by how much energy sits in
    the high-frequency coefficients relative to the whole block.

Both are computed with NumPy alone. The DCT is the standard orthonormal DCT-II
matrix, built and applied here rather than pulled from a second dependency, so the
exact numbers are a function of this file and NumPy's linear algebra — nothing else.

What this module does not do
----------------------------
* It never fabricates a vector. An input it cannot honestly process yields an
  unavailable :class:`~ml.features.contract.FeatureRepresentation` with a reason,
  exactly as :mod:`ml.features.contract` requires.
* It makes no claim that these features *detect* anything. They are inputs to a
  classifier whose performance is measured separately; the features themselves are
  not evidence.
* It applies no normalization learned from data. Scaling is fitted on the training
  split alone, in :mod:`ml.image.training`, so no test information leaks into the
  representation.

Determinism
-----------
Every step is a fixed arithmetic operation on ``float64``. Optional rounding to a
fixed number of decimal places (``decimal_places``) removes the last-bit platform
variation in the DCT cosine evaluation, so the feature vector that a fitted model
saw during training is reproducible byte-for-byte from a fresh checkout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

from ml.features.contract import (
    FeatureAvailability,
    FeatureRepresentation,
    FeatureRepresentationKind,
    make_unavailable_features,
)
from ml.inference.modality import AnalysisModality

__all__ = [
    "DCT_FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "IMAGE_FEATURE_NAMES",
    "RESIDUAL_FEATURE_NAMES",
    "ImageFeatureConfig",
    "ImageFeatureExtractor",
    "compute_feature_vector",
]

#: Bumped when the meaning, order, or count of the feature vector changes. A stored
#: model artifact pins this so a reader can tell whether old weights still match the
#: current extractor.
FEATURE_SCHEMA_VERSION = 1

#: Name of the extractor, recorded in every representation and model artifact.
EXTRACTOR_NAME = "image-residual-dct"

#: Version of the extractor implementation.
EXTRACTOR_VERSION = "1.0.0"

#: Identifier of the feature space. Two vectors with different ids are not comparable,
#: whatever their dimensions.
MODALITY_SPACE_ID = "image-residual-dct-v1"

#: Rec. 601 luma weights. Fixed constants, not learned: a learned colour transform
#: would have to be fitted on training data and would make the representation
#: split-dependent.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

#: Feature names of the residual family, in vector order.
RESIDUAL_FEATURE_NAMES: tuple[str, ...] = (
    "residual_mean",
    "residual_std",
    "residual_mean_abs",
    "residual_skew",
    "residual_excess_kurtosis",
    "residual_p95_abs",
    "residual_p99_abs",
    "residual_high_fraction",
)

#: Feature names of the DCT high-frequency family, in vector order.
DCT_FEATURE_NAMES: tuple[str, ...] = (
    "dct_high_energy_mean",
    "dct_high_energy_std",
    "dct_high_energy_ratio",
    "dct_dc_energy_mean",
    "dct_block_energy_mean",
)

#: The full feature vector layout: residual family first, then DCT.
IMAGE_FEATURE_NAMES: tuple[str, ...] = (*RESIDUAL_FEATURE_NAMES, *DCT_FEATURE_NAMES)


@dataclass(frozen=True, slots=True)
class ImageFeatureConfig:
    """Parameters of the feature transform.

    Attributes:
        residual_threshold: Absolute residual magnitude above which a pixel counts as
            an edge for ``residual_high_fraction``. On ``[0, 1]`` luminance, ``0.1``
            is a visible local step.
        dct_block_size: Side length of the square DCT blocks. The image is edge-padded
            up to a whole number of blocks, so no pixels are dropped.
        dct_high_frequency_min_radius: Coefficients whose ``(row + column)`` index sum
            reaches this value count as high-frequency. ``None`` means ``block_size``,
            which separates the low-frequency triangle from the rest.
        decimal_places: Round each feature to this many decimal places, or ``None`` to
            keep full ``float64`` precision. Rounding pins the representation against
            last-bit variation in the cosine evaluation across platforms.
    """

    residual_threshold: float = 0.1
    dct_block_size: int = 8
    dct_high_frequency_min_radius: int | None = None
    decimal_places: int | None = 6

    def __post_init__(self) -> None:
        if isinstance(self.residual_threshold, bool) or not isinstance(
            self.residual_threshold, (int, float)
        ):
            raise TypeError("residual_threshold must be a number")
        if float(self.residual_threshold) < 0.0:
            raise ValueError(
                f"residual_threshold must be non-negative, got {self.residual_threshold}"
            )
        if isinstance(self.dct_block_size, bool) or not isinstance(self.dct_block_size, int):
            raise TypeError("dct_block_size must be an integer")
        if self.dct_block_size < 2:
            raise ValueError(f"dct_block_size must be >= 2, got {self.dct_block_size}")
        if self.dct_high_frequency_min_radius is not None:
            if isinstance(self.dct_high_frequency_min_radius, bool) or not isinstance(
                self.dct_high_frequency_min_radius, int
            ):
                raise TypeError("dct_high_frequency_min_radius must be an integer or None")
            if self.dct_high_frequency_min_radius < 0:
                raise ValueError(
                    "dct_high_frequency_min_radius must be non-negative, got "
                    f"{self.dct_high_frequency_min_radius}"
                )
        if self.decimal_places is not None:
            if isinstance(self.decimal_places, bool) or not isinstance(self.decimal_places, int):
                raise TypeError("decimal_places must be an integer or None")
            if self.decimal_places < 0:
                raise ValueError("decimal_places must be non-negative or None")

    @property
    def resolved_high_frequency_min_radius(self) -> int:
        """The radius actually used, resolving ``None`` to the block size."""
        if self.dct_high_frequency_min_radius is None:
            return self.dct_block_size
        return self.dct_high_frequency_min_radius

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, stored inside a model artifact."""
        return {
            "residual_threshold": float(self.residual_threshold),
            "dct_block_size": int(self.dct_block_size),
            "dct_high_frequency_min_radius": self.dct_high_frequency_min_radius,
            "decimal_places": self.decimal_places,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ImageFeatureConfig:
        """Rebuild a config from :meth:`to_dict` output."""
        return cls(
            residual_threshold=float(raw.get("residual_threshold", 0.1)),
            dct_block_size=int(raw.get("dct_block_size", 8)),
            dct_high_frequency_min_radius=raw.get("dct_high_frequency_min_radius"),
            decimal_places=raw.get("decimal_places", 6),
        )


@lru_cache(maxsize=8)
def _dct_matrix(size: int) -> npt.NDArray[np.float64]:
    """The orthonormal DCT-II matrix of the given size.

    ``D[k, n] = alpha_k * cos(pi * (2n + 1) * k / (2N))`` with
    ``alpha_0 = sqrt(1/N)`` and ``alpha_k = sqrt(2/N)``, so ``D @ D.T == I``.
    """
    indices = np.arange(size, dtype=np.float64)
    matrix = np.empty((size, size), dtype=np.float64)
    for frequency in range(size):
        alpha = np.sqrt(1.0 / size)
        if frequency > 0:
            alpha = np.sqrt(2.0 / size)
        angle = np.pi * (2.0 * indices + 1.0) * frequency / (2.0 * size)
        matrix[frequency, :] = alpha * np.cos(angle)
    return matrix


def _to_luma(rgb: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Weighted luminance of an ``(H, W, 3)`` RGB array."""
    return np.asarray(
        np.tensordot(rgb, _LUMA_WEIGHTS, axes=([2], [0])),
        dtype=np.float64,
    )


def _residual_statistics(
    luma: npt.NDArray[np.float64],
    *,
    threshold: float,
) -> dict[str, float]:
    """Statistics of the Laplacian high-pass response of ``luma``.

    The response is ``(4*center - neighbours) / 4`` with edge replication, so the
    result has the same shape as the input and no pixel is invented.
    """
    padded = np.pad(luma, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    response = (
        4.0 * center
        - padded[:-2, 1:-1]
        - padded[2:, 1:-1]
        - padded[1:-1, :-2]
        - padded[1:-1, 2:]
    ) / 4.0

    flat = response.ravel()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    absolute = np.abs(flat)
    if std > 0.0:
        standardized = (flat - mean) / std
        skew = float(np.mean(standardized**3))
        excess_kurtosis = float(np.mean(standardized**4) - 3.0)
    else:
        skew = 0.0
        excess_kurtosis = 0.0
    return {
        "residual_mean": mean,
        "residual_std": std,
        "residual_mean_abs": float(np.mean(absolute)),
        "residual_skew": skew,
        "residual_excess_kurtosis": excess_kurtosis,
        "residual_p95_abs": float(np.percentile(absolute, 95)),
        "residual_p99_abs": float(np.percentile(absolute, 99)),
        "residual_high_fraction": float(np.mean(absolute > threshold)),
    }


def _dct_high_frequency_energy(
    luma: npt.NDArray[np.float64],
    *,
    block_size: int,
    min_radius: int,
) -> dict[str, float]:
    """Per-block DCT energies, summarized by high-frequency share."""
    height, width = int(luma.shape[0]), int(luma.shape[1])
    pad_bottom = (-height) % block_size
    pad_right = (-width) % block_size
    if pad_bottom or pad_right:
        padded = np.pad(luma, ((0, pad_bottom), (0, pad_right)), mode="edge")
    else:
        padded = luma

    rows = padded.shape[0] // block_size
    columns = padded.shape[1] // block_size
    # (rows, block, columns, block) -> (rows * columns, block, block)
    grid = padded.reshape(rows, block_size, columns, block_size).transpose(0, 2, 1, 3)
    blocks = grid.reshape(rows * columns, block_size, block_size)

    matrix = _dct_matrix(block_size)
    coefficients = matrix @ blocks @ matrix.T
    energies = coefficients**2

    axis = np.arange(block_size)
    radius = axis[:, None] + axis[None, :]
    high_mask = radius >= min_radius
    high_energy = energies[:, high_mask].sum(axis=1)
    total_energy = energies.reshape(energies.shape[0], -1).sum(axis=1)

    total_sum = float(total_energy.sum())
    high_sum = float(high_energy.sum())
    ratio = high_sum / total_sum if total_sum > 0.0 else 0.0
    dc_energy = float(energies[:, 0, 0].mean())

    return {
        "dct_high_energy_mean": float(high_energy.mean()),
        "dct_high_energy_std": float(high_energy.std()),
        "dct_high_energy_ratio": ratio,
        "dct_dc_energy_mean": dc_energy,
        "dct_block_energy_mean": float(total_energy.mean()),
    }


def compute_feature_vector(
    rgb: npt.NDArray[np.float64],
    config: ImageFeatureConfig | None = None,
) -> npt.NDArray[np.float64]:
    """Compute the feature vector for an ``(H, W, 3)`` RGB array.

    Args:
        rgb: Float RGB array. Expected in ``[0, 1]`` as produced by
            :class:`~ml.image.preprocess.ImagePreprocessor`, though the transform is
            defined for any finite values.
        config: Feature parameters. Defaults to :class:`ImageFeatureConfig`.

    Returns:
        A ``float64`` vector of length :data:`IMAGE_FEATURE_NAMES`, in that order.

    Raises:
        ValueError: If the array is not ``(H, W, 3)`` or contains non-finite values.
    """
    resolved = config or ImageFeatureConfig()
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("cannot compute features for an empty image")
    if not np.isfinite(array).all():
        raise ValueError("image contains non-finite values")

    luma = _to_luma(array)
    values = _residual_statistics(luma, threshold=float(resolved.residual_threshold))
    values.update(
        _dct_high_frequency_energy(
            luma,
            block_size=resolved.dct_block_size,
            min_radius=resolved.resolved_high_frequency_min_radius,
        )
    )
    vector = np.array([values[name] for name in IMAGE_FEATURE_NAMES], dtype=np.float64)
    if resolved.decimal_places is not None:
        vector = np.round(vector, resolved.decimal_places)
    return vector


def _resolve_array(sample: Any) -> tuple[Any, str]:
    """Return ``(array, color_space)`` for a preprocessed image or raw array."""
    array = getattr(sample, "array", None)
    if array is not None:
        color_space = getattr(sample, "color_space", None)
        return array, str(color_space) if color_space is not None else "RGB"
    if isinstance(sample, np.ndarray):
        return sample, "RGB"
    raise TypeError(
        "expected a PreprocessedImage or an (H, W, 3) array, got "
        f"{type(sample).__name__}"
    )


class ImageFeatureExtractor:
    """Extract the hand-crafted features from a preprocessed image.

    Conforms to :class:`~ml.features.contract.FeatureExtractorBackend`. Every
    successful result is tagged ``HANDCRAFTED`` — never ``LEARNED_EMBEDDING`` — so a
    consumer cannot mistake these for the output of a trained encoder.

    Args:
        config: Feature parameters.
    """

    extractor_name = EXTRACTOR_NAME
    extractor_version = EXTRACTOR_VERSION

    def __init__(self, config: ImageFeatureConfig | None = None) -> None:
        self._config = config or ImageFeatureConfig()

    @property
    def config(self) -> ImageFeatureConfig:
        """The feature parameters this extractor uses."""
        return self._config

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The feature names in vector order."""
        return IMAGE_FEATURE_NAMES

    def extract_features(self, sample: Any) -> FeatureRepresentation:
        """Extract features, or explain honestly why they are unavailable.

        Never returns a fabricated vector. Inputs that cannot be processed come back
        as ``INCOMPATIBLE_INPUT`` with a ``failure_reason``.
        """
        try:
            array, color_space = _resolve_array(sample)
        except TypeError as exc:
            return self._unavailable(
                reason=str(exc),
                availability=FeatureAvailability.INCOMPATIBLE_INPUT,
            )

        resolved = np.asarray(array)
        if resolved.ndim != 3 or resolved.shape[2] != 3:
            return self._unavailable(
                reason=f"expected an (H, W, 3) RGB image, got shape {resolved.shape}",
                availability=FeatureAvailability.INCOMPATIBLE_INPUT,
            )
        if color_space.strip().upper() != "RGB":
            return self._unavailable(
                reason=(
                    f"expected RGB channel order, got {color_space!r}; the extractor "
                    "does not guess a colour transform"
                ),
                availability=FeatureAvailability.INCOMPATIBLE_INPUT,
            )
        if not np.isfinite(resolved).all():
            return self._unavailable(
                reason="image contains non-finite pixel values",
                availability=FeatureAvailability.EXTRACTION_FAILED,
            )

        try:
            vector = compute_feature_vector(resolved, self._config)
        except ValueError as exc:
            return self._unavailable(
                reason=str(exc),
                availability=FeatureAvailability.INCOMPATIBLE_INPUT,
            )

        feature_tuple = tuple(float(value) for value in vector)
        return FeatureRepresentation(
            availability=FeatureAvailability.AVAILABLE,
            representation_kind=FeatureRepresentationKind.HANDCRAFTED,
            modality=AnalysisModality.IMAGE,
            extractor_name=self.extractor_name,
            extractor_version=self.extractor_version,
            feature_dim=len(feature_tuple),
            feature_shape=(len(feature_tuple),),
            feature_vector=feature_tuple,
            feature_array_shape=(len(feature_tuple),),
            modality_space_id=MODALITY_SPACE_ID,
            cross_modal_compatible=False,
            alignment_metadata={
                "feature_names": list(IMAGE_FEATURE_NAMES),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_config": self._config.to_dict(),
            },
            description=(
                "Hand-crafted image features: Laplacian residual statistics and "
                "per-block DCT high-frequency energy."
            ),
            limitations=(
                "Hand-crafted features only; no deep representation is produced.",
                (
                    "Not a detection result: values are inputs to a separately "
                    "trained classifier and carry no manipulation verdict on their own."
                ),
            ),
        )

    def _unavailable(
        self,
        *,
        reason: str,
        availability: FeatureAvailability,
    ) -> FeatureRepresentation:
        return make_unavailable_features(
            modality=AnalysisModality.IMAGE,
            reason=reason,
            availability=availability,
            extractor_name=self.extractor_name,
            limitations=(
                "No feature vector was produced; nothing is inferred from the input.",
            ),
            extra={"feature_config": self._config.to_dict()},
        )
