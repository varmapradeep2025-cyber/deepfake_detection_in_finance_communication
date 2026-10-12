"""Tests for :mod:`ml.reproducibility.seeds`.

Covers seed determinism, derivation stability, and input validation.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from ml.reproducibility.seeds import (
    SeedStrategy,
    derive_seed,
    get_seed_state,
    is_seeded,
    iter_seed_sequence,
    seed_worker,
    set_global_seed,
)


class TestSetGlobalSeed:
    def test_seeds_python_random_deterministically(self):
        set_global_seed(20260101)
        first = [random.random() for _ in range(5)]
        set_global_seed(20260101)
        second = [random.random() for _ in range(5)]
        assert first == second

    def test_seeds_numpy_deterministically(self):
        set_global_seed(20260101)
        first = np.random.rand(5).tolist()
        set_global_seed(20260101)
        second = np.random.rand(5).tolist()
        assert first == second

    def test_python_and_numpy_streams_are_independent_but_reproducible(self):
        set_global_seed(7)
        combined_a = (random.random(), float(np.random.rand()))
        set_global_seed(7)
        combined_b = (random.random(), float(np.random.rand()))
        assert combined_a == combined_b

    def test_different_seeds_diverge(self):
        set_global_seed(1)
        first = (random.random(), float(np.random.rand()))
        set_global_seed(2)
        second = (random.random(), float(np.random.rand()))
        assert first != second

    def test_is_idempotent(self):
        assert set_global_seed(123) is True
        assert set_global_seed(123) is True
        assert is_seeded()
        assert get_seed_state() is not None

    def test_records_state(self):
        set_global_seed(4242)
        state = get_seed_state()
        assert state is not None
        assert state.seed == 4242
        assert state.strategy is SeedStrategy.LINEAR
        assert state.as_dict()["seed"] == 4242

    def test_sets_pythonhashseed_env_var(self):
        set_global_seed(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_accepts_numpy_integer_seed(self):
        assert set_global_seed(np.int64(20260101)) is True

    def test_zero_seed_is_valid(self):
        assert set_global_seed(0) is True
        assert get_seed_state().seed == 0

    @pytest.mark.parametrize("bad", [-1, -100])
    def test_rejects_negative_seed(self, bad):
        with pytest.raises(ValueError, match="non-negative"):
            set_global_seed(bad)

    @pytest.mark.parametrize("bad", ["123", 1.5, None, True])
    def test_rejects_non_integer_seed(self, bad):
        with pytest.raises(TypeError, match="must be an integer"):
            set_global_seed(bad)


class TestDeriveSeed:
    def test_linear_is_base_plus_run_number(self):
        assert derive_seed(1000, run_number=0) == 1000
        assert derive_seed(1000, run_number=1) == 1001
        assert derive_seed(1000, run_number=7) == 1007

    def test_linear_derivation_is_reproducible(self):
        assert derive_seed(20260101, run_number=3) == derive_seed(20260101, run_number=3)

    def test_hashed_strategy_is_deterministic(self):
        first = derive_seed(20260101, label="loader", strategy=SeedStrategy.HASHED)
        second = derive_seed(20260101, label="loader", strategy=SeedStrategy.HASHED)
        assert first == second

    def test_hashed_strategy_differs_by_label(self):
        a = derive_seed(20260101, label="loader", strategy=SeedStrategy.HASHED)
        b = derive_seed(20260101, label="augmenter", strategy=SeedStrategy.HASHED)
        assert a != b

    def test_strategies_disagree(self):
        linear = derive_seed(5, run_number=2)
        hashed = derive_seed(5, run_number=2, label="x", strategy=SeedStrategy.HASHED)
        assert linear != hashed

    def test_results_fit_in_numpy_seed_range(self):
        for base in (0, 1, 2**31, 2**32 - 1, 2**40):
            seed = derive_seed(base, run_number=11, label="w", strategy=SeedStrategy.HASHED)
            assert 0 <= seed <= 2**32 - 1
            np.random.seed(seed)  # must not raise

    def test_derived_seed_is_usable_by_set_global_seed(self):
        seed = derive_seed(20260101, run_number=2)
        assert set_global_seed(seed) is True
        first = (random.random(), float(np.random.rand()))
        set_global_seed(seed)
        assert first == (random.random(), float(np.random.rand()))

    def test_hashed_requires_label(self):
        with pytest.raises(ValueError, match="requires a non-empty label"):
            derive_seed(10, label="", strategy=SeedStrategy.HASHED)

    def test_rejects_negative_run_number(self):
        with pytest.raises(ValueError, match="non-negative"):
            derive_seed(10, run_number=-1)

    def test_rejects_non_integer_arguments(self):
        with pytest.raises(TypeError, match="base_seed"):
            derive_seed("10")
        with pytest.raises(TypeError, match="run_number"):
            derive_seed(10, run_number="1")


class TestSeedSequence:
    def test_sequence_length_matches_request(self):
        assert len(list(iter_seed_sequence(20260101, 3))) == 3

    def test_sequence_is_reproducible(self):
        assert list(iter_seed_sequence(20260101, 3)) == list(iter_seed_sequence(20260101, 3))

    def test_sequence_matches_runs_per_seed_convention(self, config):
        base = config.global_seed
        runs = config.experiments.get("default_runs_per_seed", 3)
        expected = [derive_seed(base, run_number=n) for n in range(runs)]
        assert list(iter_seed_sequence(base, runs)) == expected

    def test_rejects_negative_run_count(self):
        with pytest.raises(ValueError, match="non-negative"):
            list(iter_seed_sequence(1, -1))


def test_seed_worker_refuses_to_fake_torch_dependency():
    """The helper must fail loudly rather than silently not seeding a worker."""
    with pytest.raises(NotImplementedError, match="PyTorch"):
        seed_worker(0)
