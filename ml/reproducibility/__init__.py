"""Reproducibility primitives: seeding, environment capture, and config access.

Every experiment in this project must be able to answer, without ambiguity:

* Which seed governed the run?            -> :mod:`ml.reproducibility.seeds`
* What software produced these numbers?    -> :mod:`ml.reproducibility.environment`
* Which configuration settings applied?     -> :mod:`ml.reproducibility.config`

Nothing here performs network access, downloads models, or loads datasets.
"""

from ml.reproducibility.config import ProjectConfig, load_config, load_project_config
from ml.reproducibility.environment import (
    EnvironmentReport,
    capture_environment,
    git_commit_hash,
)
from ml.reproducibility.seeds import (
    SeedStrategy,
    derive_seed,
    get_seed_state,
    seed_worker,
    set_global_seed,
)

__all__ = [
    "EnvironmentReport",
    "ProjectConfig",
    "SeedStrategy",
    "capture_environment",
    "derive_seed",
    "get_seed_state",
    "git_commit_hash",
    "load_config",
    "load_project_config",
    "seed_worker",
    "set_global_seed",
]
