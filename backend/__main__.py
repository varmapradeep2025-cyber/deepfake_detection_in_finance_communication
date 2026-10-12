"""Command-line entry point for the inference API: ``python -m backend``.

The service binds loopback by default. It carries no authentication and is meant for
local development behind the frontend, not for exposure on a network.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from backend.api import build_application, serve


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for the server CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m backend",
        description=(
            "Serve the local inference API over the ml.inference library. "
            "An uploaded image with no registered model is reported as "
            "model_unavailable; no score is invented."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="directory of model cards (default: <artifacts>/models)",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=None,
        help="where analyses are persisted (default: <artifacts>/records)",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level name")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the server; returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    application = build_application(models_dir=args.models_dir, store_dir=args.store_dir)
    serve(app=application, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
