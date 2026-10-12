"""A tiny synthetic dataset used only by unit tests.

.. warning::

   **This is not research data.** The files below are a few hundred bytes each,
   generated in-process. They contain no real faces, voices, or footage, they carry no
   forensic value, and they must never appear in a result, a figure, a report, or a
   claim about detector performance. Their only purpose is to let the ingestion tests
   exercise discovery, normalization, hashing, and validation against a directory whose
   layout is known exactly.

File formats
------------
The image and audio files are genuinely valid: a 1x1 PNG and a short PCM WAV, both
constructed byte by byte. The video files are **not** decodable video — no video encoder
is available in this repository's dependency set, and adding one to produce six bytes of
fixture would be the wrong trade. They are minimal ISO base media file format containers
(``ftyp`` + ``mdat``), which is enough for everything the ingestion layer actually does:
walk the tree, attribute a label, hash the bytes, write a manifest. Nothing in
``datasets/`` decodes media, and nothing should start on the strength of a fixture.

Everything is written into pytest's ``tmp_path``. No file produced by this module is
committed to the repository, which also keeps it clear of the ``.gitignore`` rules that
exclude media extensions everywhere.

Layout
------
::

    <root>/
    |-- image/real/img_real_001.png
    |-- image/fake/img_fake_001.png
    |-- audio/real/aud_real_001.wav
    |-- audio/fake/aud_fake_001.wav
    |-- video/real/vid_real_001.mp4
    `-- video/fake/vid_fake_001.mp4

The modality lives in the *directory*, not the extension, on purpose: it lets the tests
prove that the pipeline uses authoritative structural information and never guesses from
a filename. ``aud_fake_001.wav`` sits under ``audio/fake``, and an adapter that
attributed it from the ``.wav`` extension would be attributing image files as audio.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

__all__ = [
    "SYNTHETIC_LICENSE",
    "SYNTHETIC_RELATIVE_FILES",
    "build_synthetic_dataset",
    "synthetic_template_config",
    "tiny_mp4_bytes",
    "tiny_png_bytes",
    "tiny_wav_bytes",
]

#: Stand-in licence for the fixture only. It describes nothing real; a research dataset
#: must have its actual licence checked and recorded.
SYNTHETIC_LICENSE = "SYNTHETIC-FIXTURE-NOT-A-DATASET"

#: Relative paths of the six fixture files, in the order tests expect to find them.
SYNTHETIC_RELATIVE_FILES = (
    "image/real/img_real_001.png",
    "image/fake/img_fake_001.png",
    "audio/real/aud_real_001.wav",
    "audio/fake/aud_fake_001.wav",
    "video/real/vid_real_001.mp4",
    "video/fake/vid_fake_001.mp4",
)


def tiny_png_bytes(rgb: tuple[int, int, int] = (12, 34, 56)) -> bytes:
    """Build a valid 1x1 truecolour PNG.

    Args:
        rgb: The single pixel's colour.

    Returns:
        Complete PNG file bytes.
    """
    raw = b"\x00" + bytes(rgb)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def tiny_wav_bytes(sample_count: int = 64, *, frequency: int = 440, rate: int = 8000) -> bytes:
    """Build a valid mono 16-bit PCM WAV holding a short sine tone.

    Args:
        sample_count: Number of frames. Kept tiny; the fixture is not for listening to.
        frequency: Tone frequency in hertz.
        rate: Sample rate in hertz.

    Returns:
        Complete RIFF/WAVE file bytes.
    """
    samples = bytearray()
    for index in range(sample_count):
        value = int(8000 * math.sin(2 * math.pi * frequency * index / rate))
        samples += struct.pack("<h", value)

    data = bytes(samples)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


def tiny_mp4_bytes(payload: bytes = b"synthetic-video-stand-in") -> bytes:
    """Build a minimal ISO base media file format container.

    Produces a structurally recognisable ``ftyp`` + ``mdat`` pair. It is deliberately
    **not** decodable video; see this module's docstring.

    Args:
        payload: Bytes to place in the media data box.

    Returns:
        Container bytes.
    """

    def box(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", 8 + len(body)) + tag + body

    ftyp = box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
    return ftyp + box(b"mdat", payload)


def _fixture_bytes() -> dict[str, bytes]:
    """Map each fixture's relative path to its bytes."""
    return {
        "image/real/img_real_001.png": tiny_png_bytes((12, 34, 56)),
        "image/fake/img_fake_001.png": tiny_png_bytes((200, 100, 50)),
        "audio/real/aud_real_001.wav": tiny_wav_bytes(frequency=440),
        "audio/fake/aud_fake_001.wav": tiny_wav_bytes(frequency=880),
        "video/real/vid_real_001.mp4": tiny_mp4_bytes(b"synthetic-video-stand-in"),
        "video/fake/vid_fake_001.mp4": tiny_mp4_bytes(b"synthetic-video-stand-in-fake"),
    }


def build_synthetic_dataset(root: Path) -> Path:
    """Write the six-file fixture under ``root`` and return the root.

    Args:
        root: Directory to create. Parent directories are made as needed.

    Returns:
        ``root``, for chaining into a test.
    """
    base = Path(root)
    for relative, payload in _fixture_bytes().items():
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return base


def synthetic_template_config(
    *,
    dataset_version: str = "fixture-1",
    with_manipulations: bool = True,
) -> dict[str, object]:
    """Build a :class:`~datasets.adapters.template.TemplateAdapter` config for the fixture.

    The vocabulary is deliberately *different* from ours — ``"original"`` and
    ``"forged"`` instead of ``"real"`` and ``"fake"`` — so the tests fail if the
    pipeline ever starts reading labels as English instead of translating them.

    Args:
        dataset_version: Version recorded in provenance.
        with_manipulations: When ``False``, the fake directories declare no
            manipulation, exercising the ``unknown`` path.

    Returns:
        A configuration mapping ready for ``TemplateAdapter.from_config``.
    """
    manipulation = "faceswap" if with_manipulations else None
    return {
        "dataset_name": "synthetic_unit_test_fixture",
        "license": SYNTHETIC_LICENSE,
        "dataset_version": dataset_version,
        "source_reference": None,
        "access_date": None,
        "labels": {"original": "real", "forged": "fake"},
        "manipulations": {"faceswap": "face_swap", "vocoder": "audio_vocoder"},
        "directory_rules": [
            {"path": "image/real", "label": "original", "modality": "image"},
            {
                "path": "image/fake",
                "label": "forged",
                "modality": "image",
                "manipulation": manipulation,
            },
            {"path": "audio/real", "label": "original", "modality": "audio"},
            {
                "path": "audio/fake",
                "label": "forged",
                "modality": "audio",
                "manipulation": manipulation,
            },
            {"path": "video/real", "label": "original", "modality": "video"},
            {
                "path": "video/fake",
                "label": "forged",
                "modality": "video",
                "manipulation": manipulation,
            },
        ],
        "default_rule": None,
    }
