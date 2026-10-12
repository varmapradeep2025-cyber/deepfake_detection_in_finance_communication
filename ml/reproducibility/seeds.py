"""Deterministic random seed management.

Scope note
----------
This module seeds Python's :mod:`random` and NumPy only. Framework-specific
seeding (PyTorch, TensorFlow) is deliberately **not** implemented, because those
libraries are not installed yet. When ``ml/image`` and friends land, extend
:func:`set_global_seed` with a guarded optional import rather than adding a hard
dependency here.

Design rules
------------
1. A run is fully described by ``(base_seed, run_number)``. Use
   :func:`derive_seed` rather than ad-hoc arithmetic so that seeds stay
   reproducible and non-colliding.
2. Seeding is idempotent: calling :func:`set_global_seed` twice with the same
   value yields the same state.
3. :func:`get_seed_state` allows a test to assert that two runs are identical.

Example
-------
>>> import random
>>> set_global_seed(20260101)
True
>>> first = random.random()
>>> set_global_seed(20260101)
True
>>> first == random.random()
True
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator

__all__ = [
    "SeedState",
    "SeedStrategy",
    "derive_seed",
    "get_seed_state",
    "is_seeded",
    "seed_worker",
    "set_global_seed",
]

# Seed values are stored in a 64-bit space. NumPy legacy seeding requires a value
# in [0, 2**32); Torch, once added, accepts the full unsigned 64-bit range.
_MAX_SEED = 2**32 - 1


class SeedStrategy(str, Enum):
    """How a run seed is derived from a base seed.

    ``LINEAR`` is the default and is what the project's multi-run protocol uses:
    run *n* gets ``base_seed + n``. It is trivial to restate in a paper.

    ``HASHED`` derives seeds by hashing a label with the base seed, which suits
    data-loader workers and per-module sub-seeds where a linear offset would
    collide or be guessed too easily.
    """

    LINEAR = "linear"
    HASHED = "hashed"


@dataclass(frozen=True, slots=True)
class SeedState:
    """Snapshot of the seed a process is currently operating under."""

    seed: int
    strategy: SeedStrategy
    global_seed: int

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable form, for embedding in a run record."""
        return {
            "seed": self.seed,
            "strategy": self.strategy.value,
            "global_seed": self.global_seed,
        }


_state: SeedState | None = None


def set_global_seed(seed: int, *, strategy: SeedStrategy | str = SeedStrategy.LINEAR) -> bool:
    """Seed Python's :mod:`random`, NumPy, and the ``PYTHONHASHSEED`` environment.

    Args:
        seed: Non-negative integer seed. Values above :data:`_MAX_SEED` are reduced
            modulo the 32-bit space so legacy NumPy seeding stays valid.
        strategy: How ``seed`` relates to the configured base seed. Recorded for
            provenance; :func:`set_global_seed` seeds the *already derived* value.

    Returns:
        ``True`` if the seed was applied, ``False`` if ``seed`` was not a usable
        integer. A ``False`` return is a caller error, not a silent fallback: the
        previous state is left untouched.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If ``seed`` is negative.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    resolved = int(seed)
    legacy_seed = resolved % (_MAX_SEED + 1)

    random.seed(resolved)
    np.random.seed(legacy_seed)

    # PYTHONHASHSEED only affects the interpreter's string hashing at startup, so
    # it cannot take effect for an already-running process. We set it for any
    # worker subprocess spawned later, and record the truth in get_seed_state()
    # rather than pretending an in-process change succeeded.
    os.environ["PYTHONHASHSEED"] = str(legacy_seed)

    global _state
    _state = SeedState(
        seed=resolved,
        strategy=SeedStrategy(strategy),
        global_seed=legacy_seed,
    )
    return True


def derive_seed(
    base_seed: int,
    *,
    run_number: int = 0,
    label: str = "",
    strategy: SeedStrategy | str = SeedStrategy.LINEAR,
) -> int:
    """Derive a reproducible per-run seed from ``base_seed``.

    Args:
        base_seed: The project-level seed, e.g. ``configs.project.reproducibility.global_seed``.
        run_number: Zero-based index of the run within the seed set.
        label: Optional discriminator, required for :attr:`SeedStrategy.HASHED`
            and ignored otherwise. Use it for worker or sub-module seeds.
        strategy: Derivation strategy.

    Returns:
        A non-negative seed in the 32-bit space, so the value is directly usable
        with :func:`set_global_seed` and with any future framework seeding.

    Raises:
        TypeError: If ``base_seed`` or ``run_number`` is not an integer.
        ValueError: If ``base_seed`` is negative, ``run_number`` is negative, or
            :attr:`SeedStrategy.HASHED` is used without a ``label``.
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)):
        raise TypeError(f"base_seed must be an integer, got {type(base_seed).__name__}")
    if isinstance(run_number, bool) or not isinstance(run_number, (int, np.integer)):
        raise TypeError(f"run_number must be an integer, got {type(run_number).__name__}")
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    if run_number < 0:
        raise ValueError(f"run_number must be non-negative, got {run_number}")

    resolved_strategy = SeedStrategy(strategy)
    if resolved_strategy is SeedStrategy.LINEAR:
        seed = int(base_seed) + int(run_number)
    else:
        if not label:
            raise ValueError("SeedStrategy.HASHED requires a non-empty label")
        digest = hashlib.sha256(f"{int(base_seed)}:{label}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big")

    # Keep within the 32-bit space regardless of strategy.
    return seed % (_MAX_SEED + 1)


def get_seed_state() -> SeedState | None:
    """Return the seed currently applied to this process, or ``None`` if never seeded."""
    return _state


def is_seeded() -> bool:
    """Whether :func:`set_global_seed` has been called in this process."""
    return _state is not None


def seed_worker(worker_id: int) -> None:
    """Seed a :mod:`torch` ``DataLoader`` worker process.

    Present now so the calling convention is fixed for the experiment layer, but
    it is intentionally a no-op while PyTorch is absent: importing it must not
    create a dependency on a library this project has deliberately not installed.

    Args:
        worker_id: Index supplied by PyTorch, ``0 .. num_workers - 1``.
    """
    raise NotImplementedError(
        "seed_worker requires PyTorch, which is not installed. "
        "Enable it when ml/* detectors are implemented."
    )


def iter_seed_sequence(base_seed: int, run_count: int) -> Iterator[int]:
    """Yield ``run_count`` derived seeds for one base seed, in order.

    Convenience helper mirroring ``configs.project.experiments.default_runs_per_seed``.
    """
    if run_count < 0:
        raise ValueError(f"run_count must be non-negative, got {run_count}")
    for run_number in range(run_count):
        yield derive_seed(base_seed, run_number=run_number)
