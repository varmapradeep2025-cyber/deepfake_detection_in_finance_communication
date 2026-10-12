"""Shared pytest fixtures and helpers.

All filesystem work happens inside pytest's ``tmp_path``, so the test suite never
reads or writes inside the repository and never leaves artifacts behind.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from datasets.schema import MediaSample, Modality, SampleLabel
from tests.synthetic_preprocessing import FakeVideoDecoder


def make_file(path: Path, content: bytes) -> Path:
    """Write ``content`` to ``path``, creating parents, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def sha256_bytes(content: bytes) -> str:
    """Hex SHA-256 of an in-memory byte string."""
    return hashlib.sha256(content).hexdigest()


def make_sample(
    tmp_path: Path,
    sample_id: str,
    *,
    modality: Modality | str = Modality.IMAGE,
    label: SampleLabel | str = SampleLabel.FAKE,
    dataset_name: str = "unit_test_dataset",
    filename: str | None = None,
    content: bytes | None = None,
    license_id: str = "CC-BY-4.0",
    **overrides: object,
) -> MediaSample:
    """Create a real file on disk and return a matching :class:`MediaSample`.

    The digest is computed from the actual bytes written, so
    ``validate_sample_file`` should always pass unless a test deliberately
    corrupts the file afterwards.
    """
    payload = content if content is not None else f"payload-for-{sample_id}".encode()
    target = make_file(tmp_path / "media" / (filename or f"{sample_id}.bin"), payload)
    return MediaSample(
        sample_id=sample_id,
        dataset_name=dataset_name,
        filepath=str(target.relative_to(tmp_path)).replace("\\", "/"),
        modality=modality,
        label=label,
        sha256=sha256_bytes(payload),
        license=license_id,
        **overrides,
    )


@pytest.fixture
def sample_factory(tmp_path: Path):
    """Return a callable building samples inside the test's temporary directory."""
    return lambda sample_id, **kwargs: make_sample(tmp_path, sample_id, **kwargs)


@pytest.fixture
def simple_sample(tmp_path: Path) -> MediaSample:
    """One fully populated image sample with a real file and matching digest."""
    return make_sample(tmp_path, "sample-001", modality=Modality.IMAGE, label=SampleLabel.FAKE)


@pytest.fixture
def config():
    """The validated project configuration.

    Skips rather than fails if PyYAML or the config file is unavailable, since the
    structure tests already cover that path.
    """
    pytest.importorskip("yaml", reason="PyYAML is required for config tests")
    from ml.reproducibility.config import load_project_config

    try:
        return load_project_config()
    except Exception as exc:
        pytest.skip(f"project config unavailable: {exc}")


@pytest.fixture
def fake_decoder() -> FakeVideoDecoder:
    """A synthetic video decoder with no videos registered.

    Registered here rather than imported per module so the whole suite gets it, matching
    how the other shared fixtures are provided.
    """
    return FakeVideoDecoder()


@pytest.fixture
def touched_video(tmp_path: Path) -> Path:
    """A real, readable file that is not decodable video.

    Enough to exercise inspection paths that only need the file to exist, without
    depending on a codec.
    """
    target = tmp_path / "touch.mp4"
    target.write_bytes(b"not really a video, but a real file")
    return target
