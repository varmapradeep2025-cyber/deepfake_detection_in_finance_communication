"""Local acquisition and registration for an already-acquired dataset.

What this layer is for
----------------------
A dataset must be on this machine, described accurately, before anything may be ingested
from it or reported about it. This module answers two separate questions and refuses to
merge them:

1. **What is actually here?** :func:`plan_acquisition` inspects a local directory through
   the dataset's adapter and reports the layout: which declared manipulation directories
   exist, how many candidate files each holds, which are missing, and what is present that
   the configuration never mentioned. Read-only.
2. **May this project record having acquired it?** :func:`record_acquisition` writes an
   acquisition record, and only when every gate is satisfied.

The gates are the point
-----------------------
A record of acquisition is a claim that a specific dataset exists locally, under terms
this project may act on, at a specific version, on a specific date. Writing one requires:

* the dataset to be **registered** in ``configs/datasets.yaml``,
* its ``license_status`` to be ``verified`` — not pending, not unknown,
* its registry status to permit acquisition (not ``blocked`` or ``deprecated``),
* an **explicit** access date and dataset version, supplied by the operator,
* the adapter's layout to resolve against the directory that is actually present.

Any unmet gate produces a named blocker rather than a partial record. The output of
planning is never written anywhere, so running this module cannot change what the project
believes about a dataset.

Why it does not edit ``configs/datasets.yaml``
----------------------------------------------
Advancing a registry entry is a **human decision** in a reviewed file, so this module
prints the exact ``acquisition:`` block to paste and writes a separate, machine-generated
record beside the manifests it will eventually describe. Rewriting the registry
automatically would mean either losing the comments and ordering that make it reviewable
(YAML round-tripping destroys both) or editing it textually behind your back. The record
is therefore the evidence; the registry edit is the decision.

What it never does
------------------
* **download, copy, move, rename, or modify anything** — the record says so explicitly
  (``bytes_copied: false``), and a test asserts it,
* mark a dataset acquired because a directory exists: a directory proves presence, not
  terms,
* invent an access date, version, licence, reference, or file count,
* proceed while a required manipulation directory is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from datasets.adapters.base import AdapterConfigurationError, DatasetAdapter
from datasets.adapters.faceforensics_plus_plus import (
    FaceForensicsPlusPlusAdapter,
    is_unverified_license,
)
from datasets.registry import (
    LicenseStatus,
    RegistryError,
    RegistryStatus,
    default_registry_path,
    is_unverified,
    load_registry,
)

__all__ = [
    "ACQUISITION_RECORD_SCHEMA_VERSION",
    "AcquisitionBlockedError",
    "AcquisitionPlan",
    "AcquisitionRecord",
    "AcquisitionRequest",
    "AcquisitionRequestError",
    "load_acquisition_record",
    "plan_acquisition",
    "record_acquisition",
    "record_path_for",
    "registry_block_for",
]

#: Schema version of the acquisition record. Bumped when its shape changes.
ACQUISITION_RECORD_SCHEMA_VERSION = 1

#: Where acquisition records live by default. Tracked in git, unlike the media itself:
#: the record is the reproducible evidence that a local dataset exists and under what
#: terms, which is exactly what should survive on someone else's machine.
DEFAULT_RECORD_DIRNAME = "datasets/manifests"

#: Registry statuses that forbid acquisition outright.
_STATUSES_FORBIDDING_ACQUISITION = {RegistryStatus.BLOCKED, RegistryStatus.DEPRECATED}


class AcquisitionRequestError(ValueError):
    """Raised when the request itself is unusable.

    Distinct from :class:`AcquisitionBlockedError`: this is "you did not give me enough to
    work with", while that is "the project is not permitted to do this".
    """


class AcquisitionBlockedError(RuntimeError):
    """Raised when every gate is not satisfied, with each blocker attached."""


@dataclass(frozen=True, slots=True)
class AcquisitionRequest:
    """Everything needed to plan, and possibly record, one local acquisition.

    Attributes:
        dataset_id: The registered dataset this concerns.
        source_dir: Where the dataset is on this machine.
        adapter: The dataset's adapter, carrying its declared layout and vocabulary.
        access_date: When the data was accessed, ``YYYY-MM-DD``. Must be supplied
            explicitly; never inferred from a file timestamp, and never today's date
            standing in for it.
        dataset_version: The release this acquisition corresponds to. Must be supplied
            explicitly, because "whatever was on the site" is not a version.
        registry_path: Registry file to gate against. Defaults to the configured one.
    """

    dataset_id: str
    source_dir: Path
    adapter: DatasetAdapter
    access_date: str | None = None
    dataset_version: str | None = None
    registry_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise AcquisitionRequestError("dataset_id must be a non-empty string")
        object.__setattr__(self, "dataset_id", self.dataset_id.strip())
        object.__setattr__(self, "source_dir", Path(self.source_dir))

    @property
    def registry_file(self) -> Path:
        """Registry to gate against, resolved."""
        if self.registry_path is not None:
            return Path(self.registry_path)
        return default_registry_path()


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    """The read-only result of inspecting a local directory against the registry.

    Contains no claim that the data may be used. It reports what is on disk and which
    gates are unmet.

    Attributes:
        request: The request this plan answers.
        layout: The adapter's layout report for ``source_dir``.
        registered: Whether the dataset appears in the registry at all.
        registry_status: Status recorded there, or ``None`` when unregistered.
        license_status: Licence status recorded there, or ``None`` when unregistered.
        blockers: Reasons acquisition may not be recorded. Empty means it may.
        warnings: Facts worth reading that do not block, such as usage flags still set
            while the status is ``candidate``.
    """

    request: AcquisitionRequest
    layout: Any
    registered: bool
    registry_status: str | None
    license_status: str | None
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_blocked(self) -> bool:
        """Whether acquisition may not be recorded."""
        return bool(self.blockers)

    @property
    def is_usable(self) -> bool:
        """Whether the layout resolves, regardless of the registry gates.

        This is what a person actually wants to know when they point the tool at a
        directory: is the data laid out the way the configuration says? That is
        answerable with no licence at all.
        """
        return bool(self.layout.is_resolved)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return {
            "dataset_id": self.request.dataset_id,
            "source_dir": str(self.request.source_dir),
            "layout_usable": self.is_usable,
            "registered": self.registered,
            "registry_status": self.registry_status,
            "license_status": self.license_status,
            "is_blocked": self.is_blocked,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "layout": self.layout.to_dict(),
        }

    def _own_warnings(self) -> tuple[str, ...]:
        """Warnings raised by the gates rather than by the layout report."""
        from_layout = {f"layout: {message}" for message in self.layout.warnings}
        return tuple(item for item in self.warnings if item not in from_layout)

    def _own_blockers(self) -> tuple[str, ...]:
        """Blockers raised by the gates rather than by the layout report."""
        from_layout = {f"layout: {message}" for message in self.layout.errors}
        return tuple(item for item in self.blockers if item not in from_layout)

    def render(self) -> str:
        """Human-readable report for a terminal.

        Layout findings are left to the layout report's own rendering, and only the gate
        findings are listed separately: repeating each one twice would read as two
        problems where there is one.
        """
        adapter = self.request.adapter
        lines = [
            f"dataset id        : {self.request.dataset_id}",
            f"adapter           : {adapter.name} v{adapter.adapter_version}",
            f"registered        : {self.registered}",
            f"registry status   : {self.registry_status or '(unregistered)'}",
            f"license status    : {self.license_status or '(unregistered)'}",
            "",
            self.layout.render(),
        ]
        for message in self._own_warnings():
            lines.append(f"warning : {message}")
        for message in self._own_blockers():
            lines.append(f"blocker : {message}")
        if not self.blockers:
            lines.append("")
            lines.append(
                "acquisition may be recorded with --record once --access-date and "
                "--dataset-version are supplied"
                if not self._facts_supplied()
                else "ready to record with --record"
            )
        else:
            lines.append("")
            lines.append(
                "Nothing was written and nothing may be acquired yet. This plan only "
                "describes the directory."
            )
        return "\n".join(lines)

    def _facts_supplied(self) -> bool:
        return bool(self.request.access_date and self.request.dataset_version)


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    """A verified, local statement that a dataset exists here, under these terms.

    Deterministic: identical inputs serialize identically, with no timestamp beyond the
    access date the operator supplied. That is what makes it reviewable in git and
    comparable across machines.

    Attributes:
        dataset_id: The registered dataset.
        dataset_name: Its name as recorded by the adapter.
        dataset_version: The release, supplied explicitly.
        access_date: When it was accessed, supplied explicitly.
        source_dir: Where it is, as the operator declared it. Stored as given rather than
            resolved to an absolute path, so a tracked record does not carry one
            machine's directory layout.
        license: The licence identifier from the adapter configuration.
        layout: The layout report at acquisition time.
        registry_status: The registry status observed.
        notes: Operator remarks. Never invented.
    """

    dataset_id: str
    dataset_name: str
    dataset_version: str
    access_date: str
    source_dir: str
    license: str
    adapter_name: str
    adapter_version: str
    adapter_config: str | None
    layout: dict[str, Any]
    registry_status: str
    notes: str = ""
    acquisition_schema_version: int = ACQUISITION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "dataset_name",
            "dataset_version",
            "access_date",
            "source_dir",
            "license",
            "adapter_name",
            "adapter_version",
            "registry_status",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise AcquisitionRequestError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())
        if not _is_iso_date(self.access_date):
            raise AcquisitionRequestError(
                f"access_date must be YYYY-MM-DD, got {self.access_date!r}. The date is a "
                "fact about the acquisition; it is never inferred from a file timestamp or "
                "filled in with today."
            )
        for marker_field, value in (
            ("dataset_version", self.dataset_version),
            ("license", self.license),
        ):
            if is_unverified(value):
                raise AcquisitionRequestError(
                    f"{marker_field} is {value!r}, an unverified marker. A record of "
                    "acquisition has to state what was acquired; replace it with the real "
                    "value first."
                )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form, keys sorted so the file is stable in git."""
        return {
            "acquisition_schema_version": self.acquisition_schema_version,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "access_date": self.access_date,
            "source_dir": self.source_dir,
            "acquisition_method": "local_directory",
            "bytes_copied": False,
            "media_modified": False,
            "license": self.license,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "adapter_config": self.adapter_config,
            "registry_status_at_acquisition": self.registry_status,
            "totals": {
                "candidate_files": self.layout.get("total_files", 0),
                "total_bytes": self.layout.get("total_bytes", 0),
            },
            "layout": self.layout,
            "notes": self.notes,
        }

    def content_hash(self) -> str:
        """``sha256:<hex>`` over the canonical serialization.

        Two acquisitions of the same dataset, terms, and layout agree on this hash, so a
        later change to any of them is visible in review.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def record_path_for(dataset_id: str, *, base_dir: Path | str | None = None) -> Path:
    """Where a dataset's acquisition record lives.

    Args:
        dataset_id: Dataset identifier.
        base_dir: Manifest directory. Defaults to ``datasets/manifests``.

    Returns:
        ``<base>/<dataset_id>.acquisition.json``.
    """
    root = Path(base_dir) if base_dir is not None else Path(DEFAULT_RECORD_DIRNAME)
    return root / f"{dataset_id}.acquisition.json"


def write_record(record: AcquisitionRecord, path: Path | str) -> Path:
    """Write the record as canonical JSON, creating parent directories.

    No timestamp is written, so re-recording the same facts produces the same bytes.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(record.to_json() + "\n", encoding="utf-8")
    return target


def load_acquisition_record(path: Path | str) -> AcquisitionRecord:
    """Read a previously written acquisition record.

    Raises:
        AcquisitionRequestError: If the file is missing, malformed, or lacks required
            fields. ``bytes_copied`` is verified to be ``False`` on read: a record
            claiming this layer copied media describes something this module cannot do.
    """
    source = Path(path)
    if not source.is_file():
        raise AcquisitionRequestError(f"acquisition record not found: {source}")
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AcquisitionRequestError(
            f"malformed acquisition record at {source}: {exc.msg}"
        ) from exc
    if not isinstance(loaded, dict):
        raise AcquisitionRequestError(
            f"acquisition record must be a JSON object, got {type(loaded).__name__}"
        )
    if loaded.get("bytes_copied") is not False:
        raise AcquisitionRequestError(
            f"{source} claims bytes_copied={loaded.get('bytes_copied')!r}. This project "
            "never copies dataset media; a record saying otherwise is not one this layer "
            "wrote."
        )
    try:
        return AcquisitionRecord(
            dataset_id=loaded["dataset_id"],
            dataset_name=loaded["dataset_name"],
            dataset_version=loaded["dataset_version"],
            access_date=loaded["access_date"],
            source_dir=loaded["source_dir"],
            license=loaded["license"],
            adapter_name=loaded["adapter_name"],
            adapter_version=loaded["adapter_version"],
            adapter_config=loaded.get("adapter_config"),
            layout=loaded.get("layout", {}),
            registry_status=loaded.get("registry_status_at_acquisition", "unknown"),
            notes=loaded.get("notes", ""),
            acquisition_schema_version=int(
                loaded.get("acquisition_schema_version", ACQUISITION_RECORD_SCHEMA_VERSION)
            ),
        )
    except KeyError as exc:
        raise AcquisitionRequestError(f"{source} is missing required field {exc}") from exc


def registry_block_for(record: AcquisitionRecord) -> str:
    """The ``acquisition:`` block to paste into ``configs/datasets.yaml``.

    Emitted rather than applied. Advancing a registry entry is a decision about rights
    and provenance, and it belongs in a reviewed diff next to the comments that explain
    the entry — not behind a script that rewrites the file and drops them.
    """
    return "\n".join(
        [
            "    acquisition:",
            "      acquired: true",
            f"      access_date: {record.access_date}",
            f"      version: {record.dataset_version}",
            f"      local_path: {record.source_dir}",
            "",
            "# With the acquisition block complete, set the entry's status to",
            "#   status: available_locally",
            "# and keep license_status as it is. The acquisition record that backs this is",
            f"#   {record_path_for(record.dataset_id).as_posix()}",
            f"# with content hash {record.content_hash()}",
        ]
    )


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_acquisition(request: AcquisitionRequest) -> AcquisitionPlan:
    """Inspect a local directory and report what blocks recording an acquisition.

    Read-only, total, and safe to run at any time. Nothing is written, no media is
    touched, and the registry is not modified. A directory being present proves presence
    only — the registry gates decide whether the project may act on it.

    Args:
        request: What to inspect, with which adapter.

    Returns:
        A plan listing layout findings, warnings, and blockers. Never raises for content
        problems, so one run surfaces everything that is wrong at once.
    """
    adapter = request.adapter
    inspect = getattr(adapter, "inspect", None)
    if not callable(inspect):
        raise AcquisitionRequestError(
            f"adapter {adapter.name!r} cannot inspect a source directory, so an "
            "acquisition cannot be planned against it"
        )
    layout = inspect(request.source_dir)

    blockers: list[str] = []
    warnings: list[str] = []

    for message in layout.errors:
        blockers.append(f"layout: {message}")
    for message in layout.warnings:
        warnings.append(f"layout: {message}")

    if isinstance(adapter, FaceForensicsPlusPlusAdapter) and is_unverified_license(
        adapter.license_id
    ):
        blockers.append(
            f"adapter configuration still declares license: {adapter.license_id}, which is "
            "an unverified marker. Set the real licence in the adapter configuration "
            "before an acquisition can be recorded."
        )

    registered = False
    registry_status: str | None = None
    license_status: str | None = None

    try:
        registry = load_registry(request.registry_file)
    except RegistryError as exc:
        blockers.append(f"registry: {exc}")
    else:
        entry = registry.get(request.dataset_id)
        if entry is None:
            blockers.append(
                f"registry: {request.dataset_id!r} is not registered in "
                f"{request.registry_file.name}. A dataset absent from the registry is not "
                "used, and acquisition cannot be recorded for it. Add an entry first."
            )
        else:
            registered = True
            registry_status = entry.status.value
            license_status = entry.license_status.value

            if entry.status in _STATUSES_FORBIDDING_ACQUISITION:
                blockers.append(f"registry: status {entry.status.value!r} forbids acquisition")
            if entry.license_status is not LicenseStatus.VERIFIED:
                blockers.append(
                    f"registry: license_status is {entry.license_status.value!r}; nothing "
                    "may be acquired until the dataset's terms have been read and recorded "
                    "as verified"
                )
            if entry.is_acquired:
                warnings.append(
                    "registry: an acquisition is already recorded for this dataset; "
                    "writing another record does not change the registry"
                )
            if entry.status is RegistryStatus.CANDIDATE:
                warnings.append(
                    "registry: status is 'candidate'. Acquire it only if the terms permit "
                    "this project's use, then record status: approved alongside "
                    "license_status: verified"
                )
            for flag, value in (
                ("requires_request", entry.requires_request),
                ("requires_terms_acceptance", entry.requires_terms_acceptance),
            ):
                if value is True:
                    warnings.append(
                        f"registry: {flag} is true. Confirm the request or agreement was "
                        "actually made before relying on this acquisition"
                    )
                elif value is None or (isinstance(value, str) and is_unverified(value)):
                    warnings.append(
                        f"registry: {flag} is unverified. Whether access needs a request or "
                        "an accepted agreement is not recorded, so it has not been checked"
                    )

    if request.access_date is None:
        warnings.append(
            "no --access-date supplied. It must be given explicitly: a record of "
            "acquisition cannot leave the date blank or infer it from a file timestamp"
        )
    elif not _is_iso_date(request.access_date):
        blockers.append(
            f"access_date {request.access_date!r} is not a real YYYY-MM-DD date; supply "
            "the date the data was actually accessed"
        )
    if request.dataset_version is None:
        warnings.append(
            "no --dataset-version supplied. It must be given explicitly: 'whatever was on "
            "the site' is not a version, and a result pinned to it cannot be reproduced"
        )
    elif is_unverified(request.dataset_version):
        blockers.append(
            f"dataset_version {request.dataset_version!r} is an unverified marker, so a "
            "record of what was acquired would be empty"
        )

    return AcquisitionPlan(
        request=request,
        layout=layout,
        registered=registered,
        registry_status=registry_status,
        license_status=license_status,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def record_acquisition(request: AcquisitionRequest, *, notes: str = "") -> AcquisitionRecord:
    """Record a verified local acquisition, or refuse.

    Every gate is re-checked here rather than trusting a previous plan, because the point
    of this function is to be the single place that can produce a claim.

    Args:
        request: What was acquired, where, when, and at which version.
        notes: Operator remarks.

    Returns:
        The record. Call :func:`write_record` to persist it.

    Raises:
        AcquisitionBlockedError: If any gate is unmet. Every blocker is in the message.
        AcquisitionRequestError: If a supplied fact is malformed or still a marker.
    """
    plan = plan_acquisition(request)
    if plan.is_blocked:
        listed = "\n  - ".join(plan.blockers)
        raise AcquisitionBlockedError(
            f"acquisition of {request.dataset_id!r} is not permitted:\n  - {listed}\n"
            "Nothing was written."
        )
    if request.access_date is None or request.dataset_version is None:
        raise AcquisitionRequestError(
            "access_date and dataset_version are required to record an acquisition"
        )

    adapter = request.adapter
    return AcquisitionRecord(
        dataset_id=request.dataset_id,
        dataset_name=adapter.dataset_name,
        dataset_version=request.dataset_version,
        access_date=request.access_date,
        source_dir=Path(request.source_dir).as_posix(),
        license=adapter.license_id,
        adapter_name=adapter.name,
        adapter_version=adapter.adapter_version,
        adapter_config=_adapter_config_of(adapter),
        layout=plan.layout.to_dict(),
        registry_status=plan.registry_status or "unknown",
        notes=notes,
    )


def _adapter_config_of(adapter: DatasetAdapter) -> str | None:
    """The configuration file an adapter was built from, when it knows."""
    config_path = getattr(adapter, "config_path", None)
    if config_path is None:
        return None
    return Path(config_path).as_posix()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m datasets.acquisition`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m datasets.acquisition",
        description=(
            "Inspect an already-acquired local dataset and, when every gate is satisfied, "
            "record the acquisition. Downloads nothing and modifies no media."
        ),
    )
    parser.add_argument("--dataset-id", required=True, help="Registered dataset identifier.")
    parser.add_argument(
        "--source",
        required=True,
        help="Directory holding the acquired dataset.",
    )
    parser.add_argument(
        "--adapter",
        default=FaceForensicsPlusPlusAdapter.name,
        help="Adapter to inspect with. Defaults to the FaceForensics++ adapter.",
    )
    parser.add_argument(
        "--adapter-config",
        required=True,
        help="Adapter configuration file, e.g. configs/adapters/faceforensics_plus_plus.yaml",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Registry file to gate against. Defaults to the configured one.",
    )
    parser.add_argument(
        "--access-date",
        default=None,
        help="When the data was accessed (YYYY-MM-DD). Must be supplied explicitly.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Release acquired. Must be supplied explicitly.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Write the acquisition record. Refuses unless every gate is satisfied.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the record. Defaults to datasets/manifests/<id>.acquisition.json",
    )
    parser.add_argument("--notes", default="", help="Operator remarks for the record.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Returns:
        ``0`` when the plan is clean (and the record was written, with ``--record``),
        ``1`` when gates block acquisition, ``2`` for a bad request.
    """
    args = build_parser().parse_args(argv)

    try:
        adapter = _resolve_adapter(args.adapter, Path(args.adapter_config), args.dataset_id)
    except (AdapterConfigurationError, AcquisitionRequestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    request = AcquisitionRequest(
        dataset_id=args.dataset_id,
        source_dir=Path(args.source),
        adapter=adapter,
        access_date=args.access_date,
        dataset_version=args.dataset_version,
        registry_path=Path(args.registry) if args.registry else None,
    )

    try:
        if args.record:
            record = record_acquisition(request, notes=args.notes)
            destination = Path(args.output) if args.output else record_path_for(request.dataset_id)
            path = write_record(record, destination)
            plan = plan_acquisition(request)
            if args.json:
                print(
                    json.dumps(
                        {
                            "record": record.to_dict(),
                            "written_to": str(path),
                            "warnings": list(plan.warnings),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(plan.render())
                print("")
                print(f"acquisition recorded: {path}")
                print(f"content hash       : {record.content_hash()}")
                print("")
                print("Now apply this to configs/datasets.yaml by hand:")
                print(registry_block_for(record))
            return 0

        plan = plan_acquisition(request)
    except AcquisitionBlockedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except AcquisitionRequestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        print(plan.render())
    return 1 if plan.is_blocked else 0


def _resolve_adapter(name: str, config_path: Path, dataset_id: str) -> DatasetAdapter:
    """Build the requested adapter from its configuration file."""
    from datasets.adapters import resolve_adapter

    adapter_class = resolve_adapter(name)
    adapter = adapter_class.from_config_file(config_path, dataset_id=dataset_id)
    if adapter is None:
        raise AcquisitionRequestError(
            f"adapter {name!r} does not take a configuration file; it declares its "
            "vocabulary in code"
        )
    return adapter


def _is_iso_date(value: str) -> bool:
    """Whether a string is a real ``YYYY-MM-DD`` date.

    ``date.fromisoformat`` is used rather than a pattern match so an impossible date like
    ``2026-13-45`` is rejected: a typo is not a fact about an acquisition.
    """
    if not value or len(value) != 10:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
