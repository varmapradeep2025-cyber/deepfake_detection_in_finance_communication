"""Environment and provenance capture for reproducible runs.

Every reported metric is meaningless without the software stack that produced it.
This module records that stack into a JSON-serializable structure suitable for
``experiments/runs/<experiment_id>/<run_id>/environment.json``.

Safety
------
This module is strictly read-only with respect to the local machine and never
touches the network. Package versions come from the local import environment;
git metadata comes from invoking the local ``git`` executable. Both are optional:
absence is reported as ``None`` rather than guessed.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "EnvironmentReport",
    "capture_environment",
    "capture_package_versions",
    "git_commit_hash",
    "git_describe",
    "git_is_dirty",
    "python_version_string",
]

# Packages pinned in requirements.txt. These are always reported when installed,
# so a run record is comparable across environments even if the global env differs.
CORE_PACKAGES = (
    "numpy",
    "pandas",
    "PyYAML",
    "tqdm",
    "pytest",
    "ruff",
    "mypy",
)

# Heavy stacks stay absent at this stage. They are recorded as unavailable so the
# report documents their absence instead of leaving a reader to assume it.
OPTIONAL_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "timm",
    "scikit-learn",
    "opencv-python",
    "librosa",
    "soundfile",
    "fastapi",
    "sqlalchemy",
)

_GIT_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Immutable snapshot of the software environment for one run.

    All fields are JSON-serializable. ``git`` and package fields are ``None``
    when the information could not be obtained, never inferred.
    """

    captured_at: str
    python_version: str
    python_implementation: str
    python_executable: str
    operating_system: str
    platform: str
    platform_release: str
    machine: str
    processor: str
    byte_order: str
    filesystem_encoding: str
    cpu_count: int | None
    git_commit: str | None
    git_branch: str | None
    git_describe: str | None
    git_dirty: bool | None
    packages: dict[str, str | None] = field(default_factory=dict)
    optional_packages: dict[str, str | None] = field(default_factory=dict)
    relevant_env_vars: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary form."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to JSON with sorted keys, for stable hashing and diffing."""
        import json

        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


def python_version_string() -> str:
    """Full Python version, e.g. ``3.13.12 (main, ...) [GCC ...]``."""
    return sys.version.replace("\n", " ")


def _package_version(name: str) -> str | None:
    """Return an installed distribution version, or ``None`` if absent.

    Reads installed metadata only; it does not import the package, so a heavy or
    broken dependency cannot crash the capture.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_package_versions(
    names: tuple[str, ...] = CORE_PACKAGES,
) -> dict[str, str | None]:
    """Look up installed versions for ``names``.

    Returns:
        Mapping of distribution name to version string, or ``None`` when not
        installed. Names are sorted for stable output.
    """
    return {name: _package_version(name) for name in sorted(names)}


def _run_git(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a git command, returning stripped stdout or ``None`` on any failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
            cwd=str(cwd) if cwd else None,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def git_commit_hash(repo_root: Path | None = None) -> str | None:
    """Full git commit SHA of ``HEAD``, or ``None`` outside a git checkout."""
    value = _run_git(["rev-parse", "HEAD"], repo_root)
    return value.lower() if value else None


def git_branch(repo_root: Path | None = None) -> str | None:
    """Current branch name, or ``None`` when detached or not a checkout."""
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)


def git_describe(repo_root: Path | None = None) -> str | None:
    """``git describe --tags --always --dirty`` output, or ``None``."""
    return _run_git(["describe", "--tags", "--always", "--dirty"], repo_root)


def git_is_dirty(repo_root: Path | None = None) -> bool | None:
    """Whether the working tree has uncommitted changes.

    ``None`` when git is unavailable or the directory is not a repository. A dirty
    tree is important to record: results from uncommitted code are not
    reproducible from the commit hash alone.
    """
    if _run_git(["rev-parse", "--is-inside-work-tree"], repo_root) != "true":
        return None
    status = _run_git(["status", "--porcelain"], repo_root)
    if status is None:
        return None
    return bool(status)


# Environment variables that can silently change numerical results or paths.
_TRACKED_ENV_VARS = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
)


def _capture_env_vars() -> dict[str, str]:
    """Record reproducibility-relevant environment variables that are set."""
    return {name: os.environ[name] for name in _TRACKED_ENV_VARS if name in os.environ}


def capture_environment(
    repo_root: Path | None = None,
    *,
    include_optional_packages: bool = True,
    timestamp: str | None = None,
) -> EnvironmentReport:
    """Capture a :class:`EnvironmentReport` for the current process.

    Args:
        repo_root: Directory to query for git metadata. Defaults to the current
            working directory.
        include_optional_packages: Also report the heavy stacks that are expected
            to be absent at this stage. Keeping them in the report documents that
            their absence was observed rather than assumed.
        timestamp: Override for ``captured_at``; passed through so a caller can
            supply a single consistent run timestamp.

    Returns:
        A populated, JSON-serializable :class:`EnvironmentReport`.
    """
    from datetime import datetime

    return EnvironmentReport(
        captured_at=timestamp or datetime.now(UTC).isoformat(timespec="seconds"),
        python_version=python_version_string(),
        python_implementation=platform.python_implementation(),
        python_executable=sys.executable,
        operating_system=platform.system(),
        platform=platform.platform(),
        platform_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        byte_order=sys.byteorder,
        filesystem_encoding=sys.getfilesystemencoding(),
        cpu_count=os.cpu_count(),
        git_commit=git_commit_hash(repo_root),
        git_branch=git_branch(repo_root),
        git_describe=git_describe(repo_root),
        git_dirty=git_is_dirty(repo_root),
        packages=capture_package_versions(),
        optional_packages=(
            capture_package_versions(OPTIONAL_PACKAGES) if include_optional_packages else {}
        ),
        relevant_env_vars=_capture_env_vars(),
    )
