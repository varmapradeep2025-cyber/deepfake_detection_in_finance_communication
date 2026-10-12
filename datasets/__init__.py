"""Dataset domain: schemas, integrity checks, frozen splits, and the ingestion layer.

Layout note
-----------
``datasets/splits/`` holds split *definition files* (tracked in git).
``datasets/split_schema.py`` is the module that defines, validates, and hashes
them. The two were named apart because a file and a directory cannot share the
name ``splits`` on one filesystem.

Module map
----------
:mod:`datasets.schema`
    :class:`~datasets.schema.MediaSample`, modality/label/manipulation enums, and
    record-level validation.
:mod:`datasets.manifest`
    JSONL I/O, streaming SHA-256, sample verification, manifest hashing.
:mod:`datasets.split_schema`
    :class:`~datasets.split_schema.FrozenSplit`, leakage detection, split hashing.
:mod:`datasets.normalization`
    The controlled label/modality/manipulation vocabulary layer. Nothing is inferred;
    an unmapped token is an error.
:mod:`datasets.adapters`
    :class:`~datasets.adapters.DatasetAdapter`, the per-dataset contract, plus the
    generic ``template`` adapter. No adapter exists for a real dataset yet.
:mod:`datasets.provenance`
    The dataset-level record of origin, licence, version, and manifest hash.
:mod:`datasets.ingest`
    The ingestion pipeline, its summary, and the ``python -m datasets.ingest`` CLI.
:mod:`datasets.registry`
    The version-controlled record of which datasets may be used for what:
    :func:`~datasets.registry.load_registry`, :func:`~datasets.registry.validate_registry`,
    and the ``python -m datasets.registry`` CLI. Nothing is downloaded or verified here
    either; it reads ``configs/datasets.yaml`` and refuses claims it cannot support.

Directory convention
--------------------
::

    datasets/
    |-- raw/<dataset_id>/                  as acquired, never committed
    |-- processed/<dataset_id>/            derived features, never committed
    |-- manifests/<dataset_id>.jsonl       one JSON object per sample (tracked)
    |-- manifests/<dataset_id>.provenance.json   dataset-level record (tracked)
    `-- splits/<dataset_id>.json           frozen split definition (tracked)

The provenance record sits beside its manifest because it describes that manifest;
the documented four-directory layout therefore needs no extra folder.

Nothing in this package downloads data or touches the network.
"""

from datasets.manifest import (
    DEFAULT_CHUNK_SIZE,
    ManifestSummary,
    ManifestValidationReport,
    calculate_file_sha256,
    calculate_manifest_hash,
    count_labels,
    count_modalities,
    describe_manifest,
    load_manifest,
    resolve_sample_path,
    validate_manifest_file,
    validate_sample_file,
    write_manifest,
)
from datasets.normalization import (
    LabelMapping,
    ManipulationMapping,
    ModalityMapping,
    NoLabelMappingError,
    NormalizationError,
    UnknownLabelError,
    UnknownManipulationError,
    UnknownModalityError,
    normalize_token,
)
from datasets.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    DatasetProvenance,
    ProvenanceError,
    load_provenance,
    provenance_path_for,
    write_provenance,
)
from datasets.schema import (
    MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    ManipulationType,
    MediaSample,
    Modality,
    SampleLabel,
    SplitName,
    validate_samples,
)
from datasets.split_schema import (
    SPLIT_SCHEMA_VERSION,
    FrozenSplit,
    SplitValidationError,
    SplitValidationReport,
    assign_splits,
    load_split_definition,
    validate_split_id_sets,
    write_split_definition,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "MANIFEST_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "SPLIT_SCHEMA_VERSION",
    "DatasetProvenance",
    "FrozenSplit",
    "LabelMapping",
    "ManipulationMapping",
    "ManipulationType",
    "ManifestSummary",
    "ManifestValidationError",
    "ManifestValidationReport",
    "MediaSample",
    "Modality",
    "ModalityMapping",
    "NoLabelMappingError",
    "NormalizationError",
    "ProvenanceError",
    "SampleLabel",
    "SplitName",
    "SplitValidationError",
    "SplitValidationReport",
    "UnknownLabelError",
    "UnknownManipulationError",
    "UnknownModalityError",
    "assign_splits",
    "calculate_file_sha256",
    "calculate_manifest_hash",
    "count_labels",
    "count_modalities",
    "describe_manifest",
    "load_manifest",
    "load_provenance",
    "load_split_definition",
    "normalize_token",
    "provenance_path_for",
    "resolve_sample_path",
    "validate_manifest_file",
    "validate_sample_file",
    "validate_samples",
    "validate_split_id_sets",
    "write_manifest",
    "write_provenance",
    "write_split_definition",
]
