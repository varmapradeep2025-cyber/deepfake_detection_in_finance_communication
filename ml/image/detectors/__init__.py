"""Image detector backends.

This package holds the adapters that turn a stored model artifact into a
:class:`~ml.inference.models.ModelBackend` the image analyzer can call. Each backend
lives in its own module and is addressed by the ``entrypoint`` field of its model
card, e.g. ``ml.image.detectors.classical:build``.
"""

from ml.image.detectors.classical import (
    ClassicalImageBackend,
    ClassicalWeights,
    ClassicalWeightsError,
    build,
    load_classical_weights,
    write_classical_weights,
)

__all__ = [
    "ClassicalImageBackend",
    "ClassicalWeights",
    "ClassicalWeightsError",
    "build",
    "load_classical_weights",
    "write_classical_weights",
]
