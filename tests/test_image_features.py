"""Tests for the hand-crafted image feature transform.

These exercise the *mechanism*: shape and naming consistency, determinism, honest
unavailable results for bad input, and config round-tripping. They deliberately make
no claim about detection quality — the features are not a detector, and a synthetic
gradient proves nothing about real manipulated media.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.features.contract import FeatureAvailability, FeatureRepresentationKind
from ml.image.features import (
    DCT_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    IMAGE_FEATURE_NAMES,
    RESIDUAL_FEATURE_NAMES,
    ImageFeatureConfig,
    ImageFeatureExtractor,
    compute_feature_vector,
)
from ml.image.preprocess import ImagePreprocessConfig, ImagePreprocessor
from ml.inference.modality import AnalysisModality
from tests.synthetic_inference import FakeImagePixelDecoder, make_image_array


def _rgb(seed: int, size: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((size, size, 3), dtype=np.float64)


def _preprocessed(array, color_space: str = "BGR"):
    height, width = array.shape[0], array.shape[1]
    return ImagePreprocessor(
        FakeImagePixelDecoder(array, color_space),
        ImagePreprocessConfig(target_size=(height, width)),
    ).preprocess("synthetic.png")


class TestFeatureLayout:
    def test_vector_length_matches_names(self):
        vector = compute_feature_vector(_rgb(0))
        assert vector.shape == (len(IMAGE_FEATURE_NAMES),)
        assert len(RESIDUAL_FEATURE_NAMES) + len(DCT_FEATURE_NAMES) == len(IMAGE_FEATURE_NAMES)

    def test_all_features_are_finite(self):
        vector = compute_feature_vector(_rgb(1))
        assert np.isfinite(vector).all()

    def test_representation_is_handcrafted_and_available(self):
        representation = ImageFeatureExtractor().extract_features(_rgb(2))
        assert representation.availability is FeatureAvailability.AVAILABLE
        assert representation.representation_kind is FeatureRepresentationKind.HANDCRAFTED
        assert representation.modality is AnalysisModality.IMAGE
        assert representation.feature_dim == len(IMAGE_FEATURE_NAMES)
        assert representation.extractor_name == "image-residual-dct"
        assert representation.extractor_version
        assert representation.modality_space_id == "image-residual-dct-v1"
        assert representation.failure_reason is None

    def test_alignment_metadata_carries_names_and_schema(self):
        representation = ImageFeatureExtractor().extract_features(_rgb(3))
        metadata = representation.alignment_metadata
        assert metadata["feature_names"] == list(IMAGE_FEATURE_NAMES)
        assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert metadata["feature_config"]["dct_block_size"] == 8


class TestDeterminism:
    def test_same_input_gives_identical_vector(self):
        array = _rgb(4)
        first = compute_feature_vector(array)
        second = compute_feature_vector(array)
        assert np.array_equal(first, second)

    def test_independent_extractors_agree(self):
        array = _rgb(5)
        left = ImageFeatureExtractor().extract_features(array)
        right = ImageFeatureExtractor().extract_features(array)
        assert left.feature_vector == right.feature_vector

    def test_different_content_changes_vector(self):
        assert not np.array_equal(compute_feature_vector(_rgb(6)), compute_feature_vector(_rgb(7)))


class TestUnavailableResults:
    def test_grayscale_array_is_incompatible(self):
        representation = ImageFeatureExtractor().extract_features(make_image_array(8, 8, 1))
        assert representation.availability is FeatureAvailability.INCOMPATIBLE_INPUT
        assert representation.feature_vector is None
        assert representation.failure_reason

    def test_four_channel_array_is_incompatible(self):
        representation = ImageFeatureExtractor().extract_features(make_image_array(8, 8, 4))
        assert representation.availability is FeatureAvailability.INCOMPATIBLE_INPUT
        assert representation.feature_vector is None

    def test_non_finite_pixels_are_extraction_failed(self):
        array = _rgb(8)
        array[0, 0, 0] = np.nan
        representation = ImageFeatureExtractor().extract_features(array)
        assert representation.availability is FeatureAvailability.EXTRACTION_FAILED
        assert representation.feature_vector is None

    def test_non_rgb_color_space_is_incompatible(self):
        array = make_image_array(8, 8, 3)
        preprocessed = _preprocessed(array, color_space="BGR")
        object.__setattr__(preprocessed, "color_space", "BGR")
        representation = ImageFeatureExtractor().extract_features(preprocessed)
        assert representation.availability is FeatureAvailability.INCOMPATIBLE_INPUT
        assert representation.feature_vector is None

    def test_unrelated_object_is_incompatible(self):
        representation = ImageFeatureExtractor().extract_features(object())
        assert representation.availability is FeatureAvailability.INCOMPATIBLE_INPUT
        assert representation.feature_vector is None


class TestPreprocessedInput:
    def test_accepts_preprocessed_image(self):
        preprocessed = _preprocessed(make_image_array(16, 12, 3))
        representation = ImageFeatureExtractor().extract_features(preprocessed)
        assert representation.availability is FeatureAvailability.AVAILABLE
        assert representation.feature_dim == len(IMAGE_FEATURE_NAMES)

    def test_preprocessed_and_raw_array_agree(self):
        array = make_image_array(16, 16, 3)
        preprocessed = _preprocessed(array)
        from_array = compute_feature_vector(np.asarray(preprocessed.array, dtype=np.float64))
        representation = ImageFeatureExtractor().extract_features(preprocessed)
        assert tuple(float(v) for v in from_array) == representation.feature_vector


class TestConfig:
    def test_round_trips_through_dict(self):
        config = ImageFeatureConfig(
            residual_threshold=0.25,
            dct_block_size=16,
            dct_high_frequency_min_radius=20,
            decimal_places=4,
        )
        assert ImageFeatureConfig.from_dict(config.to_dict()) == config

    def test_none_radius_resolves_to_block_size(self):
        config = ImageFeatureConfig(dct_block_size=8)
        assert config.resolved_high_frequency_min_radius == 8

    def test_rounding_to_many_places_keeps_determinism(self):
        array = _rgb(9)
        exact = compute_feature_vector(array, ImageFeatureConfig(decimal_places=None))
        rounded = compute_feature_vector(array, ImageFeatureConfig(decimal_places=3))
        assert np.allclose(exact, rounded, atol=5e-3)

    def test_invalid_block_size_rejected(self):
        with pytest.raises(ValueError):
            ImageFeatureConfig(dct_block_size=0)

    def test_invalid_type_rejected(self):
        with pytest.raises(TypeError):
            ImageFeatureConfig(dct_block_size="eight")  # type: ignore[arg-type]

    def test_decimal_places_round_trip_disabled(self):
        config = ImageFeatureConfig(decimal_places=None)
        assert ImageFeatureConfig.from_dict(config.to_dict()).decimal_places is None


class TestInputGuards:
    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            compute_feature_vector(np.zeros((4, 4), dtype=np.float64))

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            compute_feature_vector(np.zeros((0, 0, 3), dtype=np.float64))
