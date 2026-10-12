"""A synthetic FaceForensics++-shaped directory, for tests only.

Nothing here is research data
------------------------------
Every "video" this module writes is a few bytes of ASCII. There is no FaceForensics++
material in this repository, and none is needed to test what the adapter and the
acquisition layer actually do: that they read a declared layout, that they map labels
from configuration rather than from names, that they report what is missing, and that
they refuse to record an acquisition the project is not cleared for. Those are all
decisions made from the configuration and the directory tree, and a directory tree of
tiny files exercises every one of them.

Fabricating content that looked like the real dataset would be the wrong instinct. The
tests that matter assert on labels, counts, blockers, and refusals — never on decoded
frames, and never on anything resembling a detection result.

The shape mirrors the real on-disk layout: one top-level directory per manipulation, and
within it one subdirectory per source recording, which is how the dataset groups material
by the video the manipulation was derived from.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "SYNTHETIC_BYTES",
    "SYNTHETIC_DATASET_ID",
    "build_faceforensics_tree",
    "build_verified_adapter_config",
    "verified_registry_yaml",
]

#: Dataset identifier the fixtures use. Matches the real registry entry, so the adapter is
#: exercised against the id it will actually be registered under.
SYNTHETIC_DATASET_ID = "faceforensics_plus_plus"

#: Bytes written per synthetic file. Large enough to be a distinct size per file so byte
#: totals in a layout report are checkable, small enough that a fixture stays trivial.
SYNTHETIC_BYTES = 64

#: Which source recordings a complete tree carries, per manipulation.
_SOURCE_RECORDINGS = ("000", "003")


def build_faceforensics_tree(
    root: Path,
    *,
    manipulations: tuple[str, ...] = ("original", "DeepFakes", "Face2Face", "FaceSwap"),
    flat: bool = False,
    extra_files: dict[str, tuple[str, ...]] | None = None,
) -> Path:
    """Create a FaceForensics++-shaped tree under ``root``.

    Args:
        root: Directory to create the tree in.
        manipulations: Top-level manipulation directories to create, each populated with
            synthetic files.
        flat: Put files directly in the manipulation directory instead of nesting them
            per source recording. Exercises the "unusual layout" warning.
        extra_files: Additional top-level directories to create, each with one file.
            Used to place candidate media where no declaration accounts for it.

    Returns:
        ``root``, for chaining.
    """
    root = Path(root)
    for manipulation in manipulations:
        recordings = (None,) if flat else _SOURCE_RECORDINGS
        for recording in recordings:
            directory = (
                root / manipulation if recording is None else root / manipulation / recording
            )
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                _write_synthetic_file(directory / f"{manipulation}_{index}.mp4")
    for name, filenames in (extra_files or {}).items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            _write_synthetic_file(directory / filename)
    return root


def build_verified_adapter_config(config_path: Path, *, license_id: str) -> Path:
    """Write an adapter configuration whose licence is a real identifier.

    The shipped configuration uses a marker so the licence cannot be recorded by accident.
    A test that wants to get past the gate has to state a licence explicitly, the same way
    a person would after reading the dataset's terms — so the fixture writes one out rather
    than monkeypatching the check.

    Args:
        config_path: Where to write the configuration.
        license_id: The licence identifier to declare. Must not be a marker.

    Returns:
        ``config_path``.
    """
    config_path = Path(config_path)
    config_path.write_text(
        "\n".join(
            [
                "adapter: faceforensics_plus_plus",
                f"dataset_id: {SYNTHETIC_DATASET_ID}",
                "dataset_name: FaceForensics++ (synthetic fixture configuration)",
                f"license: {license_id}",
                "modality: video",
                "source_reference: test_fixture",
                "extensions:",
                "  - .mp4",
                "manipulations:",
                "  original:",
                "    label_token: original",
                "    label: real",
                "  DeepFakes:",
                "    label_token: deepfakes",
                "    label: fake",
                "    manipulation: deepfakes",
                "  Face2Face:",
                "    label_token: face2face",
                "    label: fake",
                "    manipulation: face2face",
                "  FaceSwap:",
                "    label_token: faceswap",
                "    label: fake",
                "    manipulation: faceswap",
                "  NeuralTextures:",
                "    label_token: neuraltextures",
                "    label: fake",
                "    manipulation: neuraltextures",
                "required_manipulations:",
                "  - original",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


SYNTHETIC_DATASET_ID = "faceforensics_plus_plus"


def verified_registry_yaml(
    *,
    dataset_id: str = SYNTHETIC_DATASET_ID,
    license_status: str = "verified",
    status: str = "approved",
) -> str:
    """Registry YAML for a FaceForensics++ entry that has cleared the acquisition gates.

    The license fields state placeholder identifiers rather than a real licence, because a
    test must not assert that this project's use of anything is permitted. What it asserts
    is that the layer reads ``license_status`` as the gate it is documented to be.

    Args:
        dataset_id: Dataset identifier for the entry.
        license_status: The status to declare. Overridden to non-``verified`` values to
            test refusal, which also drops the status back to ``candidate`` because the
            registry requires an approved entry to have a verified licence.
        status: Registry status for the entry.
    """
    if license_status != "verified":
        status = "candidate"
    # The registry holds invariants a real entry would too, so the fixture cannot claim a
    # status the registry would reject: a blocked entry may not be in a generalization group
    # or carry any usage permission, and a deprecated one may not be in a group either.
    groups = "[]" if status in {"blocked", "deprecated"} else "[video_deepfake]"
    permitted = "false" if status == "blocked" else "true"
    # A blocked entry holds no usage permission, and every research role except the
    # passive ones implies some, so it carries only `reference`. The registry enforces
    # this, so the fixture cannot claim a role the registry would reject.
    roles = "[reference]" if status == "blocked" else "[source_domain, training, in_domain_test]"
    license_reference = (
        '"test fixture: placeholder, not a real licence reference"'
        if license_status == "verified"
        else f'"test fixture: placeholder marked {license_status}, not a real licence"'
    )
    return "\n".join(
        [
            "registry_version: 1",
            "datasets:",
            f"  {dataset_id}:",
            f"    dataset_id: {dataset_id}",
            "    name: FaceForensics++ (synthetic fixture entry)",
            "    modality: video",
            "    source_type: research_dataset",
            f"    research_role: {roles}",
            f"    status: {status}",
            "    access_type: request_required",
            f"    license_status: {license_status}",
            f"    source_reference: {license_reference}",
            "    version: 0.0.0-test_fixture",
            "    train_allowed: " + permitted,
            "    validation_allowed: " + permitted,
            "    test_allowed: " + permitted,
            "    unseen_test_allowed: false",
            f"    generalization_groups: {groups}",
            "    approximate_size: unknown",
            "    label_schema:",
            "      scheme: binary_real_fake",
            "      values: [real, fake]",
            "    manipulation_types: pending_verification",
            "    requires_request: false",
            "    requires_terms_acceptance: " + permitted,
            "    redistribution_allowed: not_verified",
            "    acquisition:",
            "      acquired: false",
            "      access_date: null",
            "      version: null",
            "      local_path: null",
            "    notes: >-",
            "      Synthetic entry used only by tests. Not a real dataset record, and its",
            "      license fields state placeholder identifiers rather than a real licence.",
            "",
        ]
    )


def _write_synthetic_file(path: Path) -> None:
    """Write one placeholder file of exactly :data:`SYNTHETIC_BYTES`.

    Not a video, and never decoded as one. The fixed size is what lets a layout report's
    byte total be checked exactly rather than approximately.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = b"SYNTHETIC-NOT-VIDEO:" + path.stem.encode("utf-8")
    if len(body) > SYNTHETIC_BYTES:
        body = body[:SYNTHETIC_BYTES]
    path.write_bytes(body.ljust(SYNTHETIC_BYTES, b" "))
