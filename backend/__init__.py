"""The HTTP surface over the inference library.

This package is deliberately thin and dependency-free: it adapts the real
:class:`~ml.inference.service.MediaAnalysisService` to a small, honest HTTP+JSON API
using only the standard library. There is no web framework here because none is
installed and none is required to serve the endpoints the frontend already targets.

Public entry points:

* :func:`build_application` — wire the service from ``configs/project.yaml``.
* :class:`BackendApplication` — the framework-agnostic request handlers.
* :func:`build_handler` / :func:`serve` — bind the application to an HTTP server.

Run it with ``python -m backend`` (see :mod:`backend.__main__`).
"""

from __future__ import annotations

from backend.api import (
    BackendApplication,
    build_application,
    build_handler,
    serve,
)

__all__ = [
    "BackendApplication",
    "build_application",
    "build_handler",
    "serve",
]
