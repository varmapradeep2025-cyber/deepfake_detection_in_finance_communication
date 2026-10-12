"""Print the project tree and verify the foundation against configs/project.yaml.

Pure standard library. Safe to run before installing any dependency.

Usage:
    python scripts/verify_structure.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

DATA_DIRS = {
    "datasets/raw",
    "datasets/processed",
    "datasets/manifests",
    "datasets/splits",
    "experiments/image",
    "experiments/audio",
    "experiments/video",
    "experiments/cross_dataset",
    "experiments/multimodal",
    "experiments/ablation",
    "threat_intelligence/collectors",
    "threat_intelligence/source_registry",
    "threat_intelligence/fingerprinting",
    "threat_intelligence/similarity",
    "threat_intelligence/campaign",
    "threat_intelligence/correlation",
}


def walk(path: Path, prefix: str = "") -> list[str]:
    """Depth-first listing, sorted, with directories marked."""
    try:
        entries = sorted(
            (p for p in path.iterdir() if p.name not in SKIP_DIRS and not p.name.startswith(".")),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        return []

    rows = [f"{prefix}{path.name}/"]
    for index, child in enumerate(entries):
        last = index == len(entries) - 1
        connector = "`-- " if last else "|-- "
        if child.is_dir():
            rows.extend(walk(child, prefix + ("    " if last else "|   ")))
        else:
            rows.append(f"{prefix}{connector}{child.name}")
    return rows


def strip_marker(row: str) -> str:
    """Remove the trailing separator so the first entry is not mis-indented."""
    return row[2:] if row.startswith(("|", "`")) else row


def relative_dirs() -> set[str]:
    """All real subdirectories, skipping caches and dot-directories."""
    found: set[str] = set()
    for dirpath, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        rel = Path(dirpath).relative_to(ROOT).as_posix()
        if rel != ".":
            found.add(rel)
    return found


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


# Statuses that assert a dataset is usable. None may appear at the foundation stage.
_LIVE_STATUSES = {"approved", "available_locally"}

# Licence strings that state nobody has read the terms. Kept in step with
# datasets/registry.py; the authoritative tests live in tests/test_dataset_registry.py.
_UNVERIFIED_MARKERS = frozenset(
    {"unknown", "not_verified", "pending", "pending_verification", "unverified", ""}
)


def check_registry(config: dict) -> list[bool]:
    """Check the dataset registry's claim gates.

    Standard library plus PyYAML only: this script must run before anything is installed,
    so the registry module itself is not imported. The rules below mirror
    ``datasets/registry.py``; ``tests/test_dataset_registry.py`` is the authoritative suite.
    """
    results: list[bool] = []
    declared = config.get("dataset_registry")
    if not isinstance(declared, dict):
        return [check("dataset_registry declared in configs/project.yaml", False)]

    relative = declared.get("file", "")
    path = ROOT / relative
    results.append(check("dataset_registry.file exists", path.is_file(), relative))
    results.append(
        check(
            "no dataset approved yet",
            declared.get("approved_datasets") == [],
            ", ".join(declared.get("approved_datasets") or []),
        )
    )
    results.append(
        check(
            "no dataset acquired yet",
            declared.get("datasets_acquired") == [],
            ", ".join(declared.get("datasets_acquired") or []),
        )
    )

    if not path.is_file():
        return results

    try:
        with path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except Exception as exc:  # pragma: no cover - malformed registry is caught elsewhere
        return [*results, check("configs/datasets.yaml parses", False, str(exc)[:60])]

    results.append(check("configs/datasets.yaml parses", isinstance(document, dict)))
    results.append(
        check(
            "registry_version is 1",
            isinstance(document, dict) and document.get("registry_version") == 1,
            str(document.get("registry_version")) if isinstance(document, dict) else "",
        )
    )

    entries = document.get("datasets", {}) if isinstance(document, dict) else {}
    pairs = document.get("generalization", []) if isinstance(document, dict) else []
    results.append(check("registry declares datasets", bool(entries), f"{len(entries)} entries"))

    if not isinstance(entries, dict):
        return [*results, check("registry datasets is a mapping", False)]

    live = [
        key
        for key, entry in entries.items()
        if isinstance(entry, dict) and entry.get("status") in _LIVE_STATUSES
    ]
    results.append(
        check("no entry claims to be approved or locally available", not live, ", ".join(live))
    )

    acquired = [
        key
        for key, entry in entries.items()
        if isinstance(entry, dict)
        and isinstance(entry.get("acquisition"), dict)
        and entry["acquisition"].get("acquired") is True
    ]
    results.append(check("no entry records an acquisition", not acquired, ", ".join(acquired)))

    trained_on_feeds = [
        key
        for key, entry in entries.items()
        if isinstance(entry, dict)
        and entry.get("source_type") in {"intelligence_feed", "reference_list"}
        and (entry.get("train_allowed") or entry.get("validation_allowed"))
    ]
    results.append(
        check(
            "no intelligence source is a training corpus",
            not trained_on_feeds,
            ", ".join(trained_on_feeds),
        )
    )

    executed = [
        pair.get("group_id", "?")
        for pair in pairs
        if isinstance(pair, dict) and (pair.get("executed") or pair.get("status") == "executed")
    ]
    results.append(check("no generalization claim is executed", not executed, ", ".join(executed)))

    return results


def check_acquisition() -> list[bool]:
    """Check that the acquisition layer is present and claims nothing.

    A record committed to the repository would assert that a dataset was downloaded on
    some machine at some date, which is not something a structural check can verify and
    not something this project has done.
    """
    results: list[bool] = []
    results.append(
        check("datasets/acquisition.py exists", (ROOT / "datasets" / "acquisition.py").is_file())
    )
    results.append(
        check(
            "docs/dataset_acquisition.md exists",
            (ROOT / "docs" / "dataset_acquisition.md").is_file(),
        )
    )

    manifests = ROOT / "datasets" / "manifests"
    records = sorted(manifests.glob("*.acquisition.json")) if manifests.is_dir() else []
    results.append(
        check(
            "no acquisition record is committed",
            not records,
            ", ".join(path.name for path in records),
        )
    )

    adapter_config = ROOT / "configs" / "adapters" / "faceforensics_plus_plus.yaml"
    results.append(check("FaceForensics++ adapter config exists", adapter_config.is_file()))
    if adapter_config.is_file():
        try:
            document = yaml.safe_load(adapter_config.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - malformed config is caught elsewhere
            return [*results, check("adapter config parses", False, str(exc)[:60])]
        results.append(check("adapter config parses", isinstance(document, dict)))
        license_id = document.get("license") if isinstance(document, dict) else None
        results.append(
            check(
                "adapter ships an unverified licence marker",
                license_id in _UNVERIFIED_MARKERS,
                f"license: {license_id}",
            )
        )

    return results


def check_preprocessing() -> list[bool]:
    """Check that the preprocessing stage is present and honestly pinned.

    These are the invariants a reader cannot check by reading the pipeline: that one media
    backend is pinned, that the decoder protocol is what lets the tests run without it, and
    that the stage still refuses to write raw data.
    """
    results: list[bool] = []

    package = ROOT / "datasets" / "preprocessing"
    required = (
        "__init__.py",
        "inspection.py",
        "extraction.py",
        "quality.py",
        "frames.py",
        "grouping.py",
        "splitting.py",
        "pipeline.py",
    )
    missing = [name for name in required if not (package / name).is_file()]
    results.append(
        check(
            "preprocessing package is complete",
            not missing,
            ", ".join(missing) if missing else f"{len(required)} modules",
        )
    )

    results.append(
        check("preprocessing docs exist", (ROOT / "docs" / "preprocessing.md").is_file())
    )

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    media = [line.strip() for line in requirements.splitlines() if line.strip()]
    active_media = [
        line
        for line in media
        if not line.startswith("#")
        and line.split("==")[0].lower() in {"opencv-python-headless", "av"}
    ]
    results.append(
        check(
            "exactly one media backend is pinned",
            len(active_media) == 1,
            ", ".join(active_media) if active_media else "none active",
        )
    )

    decoder = (package / "inspection.py").read_text(encoding="utf-8")
    results.append(
        check(
            "decoder is injected behind a protocol",
            "class VideoDecoder" in decoder and "def default_decoder" in decoder,
        )
    )

    pipeline = (package / "pipeline.py").read_text(encoding="utf-8")
    results.append(
        check(
            "dry run cannot write a frame manifest",
            "images_persisted" in pipeline
            and "write_artifacts and not images_persisted" in pipeline,
        )
    )

    frames = (package / "frames.py").read_text(encoding="utf-8")
    results.append(
        check(
            "frames carry both a video and a recording identity",
            "source_video_id" in frames and "source_recording_id" in frames,
        )
    )
    results.append(
        check(
            "no invented frame digest",
            "_digest_of_placeholder" not in frames,
        )
    )

    return results


def check_inference() -> list[bool]:
    """Check that the inference stage is present, honest, and ships no weights.

    The invariants a reader cannot confirm from prose: that the loader verifies rather
    than fabricates a model, that evidence is tagged, and that no weights have slipped
    into the repository.
    """
    results: list[bool] = []

    core = ROOT / "ml" / "inference"
    core_required = (
        "__init__.py",
        "status.py",
        "evidence.py",
        "modality.py",
        "models.py",
        "report.py",
        "request.py",
        "service.py",
        "records.py",
        "multimodal.py",
    )
    missing_core = [name for name in core_required if not (core / name).is_file()]
    results.append(
        check(
            "inference core is complete",
            not missing_core,
            ", ".join(missing_core) if missing_core else f"{len(core_required)} modules",
        )
    )

    for package_name, required in (
        ("image", ("__init__.py", "preprocess.py", "analyzer.py")),
        ("video", ("__init__.py", "analyzer.py")),
        ("audio", ("__init__.py", "analyzer.py")),
        ("qr", ("__init__.py", "decode.py", "analyzer.py")),
        ("url", ("__init__.py", "analyze.py")),
        ("risk", ("__init__.py", "assess.py")),
    ):
        package = ROOT / "ml" / package_name
        missing = [name for name in required if not (package / name).is_file()]
        results.append(
            check(
                f"{package_name} package is complete",
                not missing,
                ", ".join(missing) if missing else f"{len(required)} modules",
            )
        )

    results.append(check("inference docs exist", (ROOT / "docs" / "inference.md").is_file()))

    url_module = (ROOT / "ml" / "url" / "analyze.py").read_text(encoding="utf-8")
    results.append(
        check(
            "URL analysis performs no network access",
            "urllib.request" not in url_module
            and "http.client" not in url_module
            and "requests" not in url_module,
        )
    )

    multimodal_module = (core / "multimodal.py").read_text(encoding="utf-8")
    results.append(
        check(
            "multimodal report fuses no scores",
            "scores are not" in multimodal_module and "scored_count" in multimodal_module,
        )
    )

    models = (core / "models.py").read_text(encoding="utf-8")
    results.append(
        check(
            "loader verifies weights rather than fabricating a model",
            "weights_digest_mismatch" in models and "dependency_missing" in models,
        )
    )

    evidence = (core / "evidence.py").read_text(encoding="utf-8")
    analyzer = (ROOT / "ml" / "image" / "analyzer.py").read_text(encoding="utf-8")
    results.append(
        check(
            "evidence is tagged observed/derived/inferred",
            "OBSERVED" in evidence
            and "DERIVED" in evidence
            and "INFERRED" in evidence
            and "claim_kind" in analyzer,
        )
    )

    weight_globs = ("*.pt", "*.pth", "*.ckpt", "*.onnx", "*.safetensors", "*.h5", "*.pb")
    artifacts = ROOT / "artifacts"
    committed = (
        sorted({path.name for pattern in weight_globs for path in artifacts.rglob(pattern)})
        if artifacts.is_dir()
        else []
    )
    results.append(check("no model weights are committed", not committed, ", ".join(committed[:4])))

    return results


def main() -> int:
    print("=" * 78)
    print("deepfake-finance-research - foundation verification")
    print("=" * 78)

    print("\nProject tree\n" + "-" * 78)
    rows = walk(ROOT)
    if rows:
        print(strip_marker(rows[0]))
        for row in rows[1:]:
            print(row)

    print("\nChecks\n" + "-" * 78)
    results: list[bool] = []

    existing = relative_dirs()

    for required_file in (
        "README.md",
        ".gitignore",
        "requirements.txt",
        "configs/project.yaml",
        "configs/datasets.yaml",
    ):
        results.append(check(f"file exists: {required_file}", (ROOT / required_file).is_file()))

    config: dict = {}
    config_loaded = False
    if yaml is None:
        results.append(check("configs/project.yaml parses", False, "PyYAML not installed"))
        print("\n  Install dependencies to enable config checks: pip install -r requirements.txt")
    else:
        try:
            with (ROOT / "configs" / "project.yaml").open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            config_loaded = isinstance(config, dict)
        except Exception as exc:
            results.append(check("configs/project.yaml parses", False, str(exc)[:60]))
        if config_loaded:
            results.append(check("configs/project.yaml parses", True))

    if config_loaded:
        results.append(
            check("project.stage == foundation", config["project"]["stage"] == "foundation")
        )
        results.append(
            check(
                "global seed is set",
                isinstance(config["reproducibility"]["global_seed"], int),
                str(config["reproducibility"]["global_seed"]),
            )
        )
        results.append(check("multiple experiment seeds", len(config["experiments"]["seeds"]) >= 3))

        missing_paths = [
            name for name, rel in config["paths"]["source"].items() if not (ROOT / rel).exists()
        ]
        results.append(check("all source paths exist", not missing_paths, ", ".join(missing_paths)))

        generated = config["paths"]["generated"]
        unignored = [
            name
            for name, rel in generated.items()
            if f"{rel}/" not in (ROOT / ".gitignore").read_text(encoding="utf-8")
        ]
        results.append(check("generated paths are gitignored", not unignored, ", ".join(unignored)))

        missing_modules = [
            name for name, mod in config["modules"].items() if not (ROOT / mod["path"]).is_dir()
        ]
        results.append(
            check("all module paths exist", not missing_modules, ", ".join(missing_modules))
        )

        enabled = [n for n, m in config["modules"].items() if m["enabled"]]
        results.append(check("no modules enabled yet", not enabled, ", ".join(enabled)))

        ingestion = config.get("ingestion", {})
        results.append(
            check(
                "ingestion requires a license and stays deterministic",
                ingestion.get("require_license") is True
                and ingestion.get("deterministic_manifest") is True,
            )
        )
        results.append(
            check(
                "no dataset selected yet",
                ingestion.get("datasets_selected") == [],
                ", ".join(ingestion.get("datasets_selected") or []),
            )
        )
        results.append(
            check(
                "ingestion paths match the configured convention",
                ingestion.get("raw_dir") == config["paths"]["source"]["datasets_raw"]
                and ingestion.get("manifest_dir")
                == config["paths"]["source"]["datasets_manifests"],
            )
        )

        ti = config["threat_intelligence"]
        results.append(
            check(
                "network access disabled by default",
                not ti["enabled"] and not ti["allow_network_access"] and ti["require_opt_in"],
            )
        )

        services = config["services"]
        results.append(
            check(
                "api/database/frontend disabled",
                not services["api"]["enabled"]
                and not services["database"]["enabled"]
                and not services["frontend"]["enabled"],
            )
        )

        results.extend(check_registry(config))
        results.extend(check_acquisition())

        heavy = ("torch", "torchvision", "transformers", "timm", "tensorflow", "fastapi")
        active = [
            line.strip().lower()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        leaked = [d for d in heavy if any(a.startswith(d) for a in active)]
        results.append(check("heavy ML deps still commented out", not leaked, ", ".join(leaked)))

    results.extend(check_preprocessing())
    results.extend(check_inference())

    gitignore_path = ROOT / ".gitignore"
    gitignore = gitignore_path.read_text(encoding="utf-8") if gitignore_path.is_file() else ""
    required_rules = ("datasets/raw/**", "datasets/processed/**", "artifacts/", "*.ckpt")
    missing_rules = [rule for rule in required_rules if rule not in gitignore]
    results.append(
        check("gitignore excludes data/weights", not missing_rules, ", ".join(missing_rules))
    )

    unheld = [d for d in DATA_DIRS if d in existing and not any((ROOT / d).iterdir())]
    results.append(check("empty dirs placeheld with .gitkeep", not unheld, ", ".join(unheld[:4])))

    module_prefixes = ("ml/", "forensic/", "threat_intelligence/")
    tracked_dirs = len([d for d in existing if d.startswith(module_prefixes)])

    print("\nSummary\n" + "-" * 78)
    passed = sum(results)
    total = len(results)
    print(f"  directories : {len(existing)}")
    print(f"  module dirs : {tracked_dirs}")
    print(f"  checks      : {passed}/{total} passed")
    print("=" * 78)

    if passed == total:
        print("\nFoundation verified. Ready for the next increment.")
        return 0
    print("\nSome checks failed. Fix the items marked FAIL above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
