# Multimodal Deepfake Detection and Evidence Correlation for Financial Communication

Research prototype for detecting manipulated **image**, **audio**, and **video** media in
financial communication contexts, correlating forensic evidence across sources, and producing
**uncertainty-aware forensic reports**.

> **Status: `foundation`.** This repository currently contains the project skeleton,
> configuration, reproducibility conventions, the dataset registry, the dataset
> ingestion framework, the media-preprocessing pipeline, and a model-agnostic image
> inference pipeline. No trained models, datasets, API, database, or frontend have been
> implemented yet — inference ships no weights, so it returns `model_unavailable` rather
> than a score — and eight candidate datasets are *declared* but none is approved,
> downloaded, or verified. See [Current Scope](#current-scope).

---

## Table of Contents

- [Purpose](#purpose)
- [Current Scope](#current-scope)
- [Directory Guide](#directory-guide)
  - [Top level](#top-level)
  - [`ml/` — Detection models](#ml--detection-models)
  - [`forensic/` — File-level forensics](#forensic--file-level-forensics)
  - [`threat_intelligence/` — Public correlation](#threat_intelligence--public-correlation)
  - [`datasets/` — Data lifecycle](#datasets--data-lifecycle)
  - [`experiments/` — Experiment runs](#experiments--experiment-runs)
  - [`reports/`, `notebooks/`, `scripts/`, `tests/`, `configs/`, `docs/`](#supporting-directories)
- [Reproducibility Contract](#reproducibility-contract)
- [Research Reproducibility](#research-reproducibility)
- [Dataset Registry](#dataset-registry)
- [Dataset Acquisition](#dataset-acquisition)
- [Dataset Ingestion](#dataset-ingestion)
- [Media Preprocessing](#media-preprocessing)
- [Inference](#inference)
- [Dataset Manifest Format (JSONL)](#dataset-manifest-format-jsonl)
- [Planned Module Map](#planned-module-map)
- [Getting Started](#getting-started)
- [Conventions](#conventions)
- [Safety and Data Handling](#safety-and-data-handling)

---

## Purpose

Financial communication is a high-value target for synthetic media: impersonating an
executive on a call, a fabricated press release, a spoofed voice authorising a payment
redirect. This project treats deepfake detection as an **evidence problem**, not just a
classification problem. Every detection is intended to carry:

| Requirement | Meaning |
| --- | --- |
| **Traceability** | Which bytes were analysed, under which hashes. |
| **Provenance** | Where the file came from and how it was handled. |
| **Localization** | *Where* in the media the manipulation evidence lies. |
| **Uncertainty** | How confident the system is, and where it is likely wrong. |
| **Corroboration** | Whether independent signals (metadata, hashes, campaign patterns) agree. |

The intended output is a forensic report that a non-specialist — an analyst, compliance
officer, or journalist — can read and act on, including a clear statement of what the
system *cannot* conclude.

---

## Current Scope

**Implemented now**

- Directory skeleton for all planned subsystems.
- `configs/project.yaml` — single source of truth for paths, seeding, and module toggles.
- `requirements.txt` — pinned lightweight tooling only.
- `.gitignore` — strict separation of tracked source vs. untracked data and weights.
- `ruff.toml`, `mypy.ini` — pinned lint and type-check configuration.
- `ml/reproducibility/` — seeding, environment capture, typed config access.
- `datasets/` — `schema.py` (sample/manifest), `manifest.py` (JSONL IO, SHA-256),
  `split_schema.py` (frozen splits and leakage detection).
- `datasets/normalization.py`, `datasets/adapters/`, `datasets/ingest.py`,
  `datasets/provenance.py` — the ingestion framework: adapter contract, declared-vocabulary
  normalization, deterministic JSONL manifests, and provenance records. See
  [Dataset Ingestion](#dataset-ingestion).
- `configs/datasets.yaml`, `datasets/registry.py` — the dataset registry: eight candidate
  sources declared with every licence, version, reference, and size left an explicit
  unverified marker, plus the three planned generalization pairs. Nothing is approved,
  acquired, or executed. See [Dataset Registry](#dataset-registry).
- `configs/adapters/faceforensics_plus_plus.yaml`, `datasets/adapters/faceforensics_plus_plus.py`,
  `datasets/acquisition.py` — the first dataset adapter and the acquisition gates in front
  of it: declared layout, explicit label mappings, licence-gated refusal. See
  [Dataset Acquisition](#dataset-acquisition).
- `experiments/` — `schema.py` (four experiment families), `results.py` (metrics
  schema), `run_artifacts.py` (run directory layout).
- `ml/inference/`, `ml/image/`, `ml/video/`, `ml/audio/`, `ml/qr/`, `ml/url/`, `ml/risk/`
  — a model-agnostic, end-to-end inference layer for four modalities (image, video,
  audio, QR/URL): a registry that discovers and digest-verifies model cards, an
  injectable image preprocessor, honest risk assessment, and an evidence-tagged
  analysis report. A file-based record store (`ml/inference/records.py`) persists
  results, and a builder (`ml/inference/multimodal.py`) assembles a multimodal report
  that **never fuses** per-modality scores. No weights ship, so image and video report
  `model_unavailable`, non-WAV audio reports `capability_unavailable`, and a QR decode
  completes without a score; see [Inference](#inference).
- `frontend/src/pages/ModalityAnalysis.tsx` and `frontend/src/api/modalities.ts` — a
  service-backed modality-analysis page and typed client for the endpoints above. No
  service implements them yet, so the page renders an explicit offline state rather
  than inventing results.
- `tests/` — structure, config, schemas, hashing, ingestion, registry, preprocessing,
  and inference tests.

**Explicitly not implemented yet** — by explicit design decision, not omission:

- No ML models, no model definitions, no placeholder checkpoints. The inference pipeline
  exists, but there is nothing to load, so analysis returns `model_unavailable` rather
  than a score.
- No deep-learning dependencies (`torch`, `transformers`, etc.) are installed.
- No datasets are downloaded or committed. Eight candidates are *registered as intentions*
  with `status: candidate` and unverified terms; none is `approved`, none exists on disk,
  and `dataset_registry.approved_datasets` is `[]`.
- No dataset has been acquired. The FaceForensics++ adapter ships, and reads a directory if
  one is pointed at it, but it declares `license: pending_verification`, so ingestion and
  acquisition both refuse until a person reads the terms and records them in both the
  adapter config and the registry.
- No licence is verified for any dataset, and no acquisition record exists in the repository.
- No FastAPI service, no PostgreSQL schema, no React/TypeScript application.
- No threat-intelligence collectors or network calls of any kind.

Every planned subsystem is registered in `configs/project.yaml` with `enabled: false`.

---

## Directory Guide

### Top level

| Path | Purpose |
| --- | --- |
| `backend/` | Future FastAPI service layer: request routing, analysis orchestration, report generation, and the database access layer. Empty at this stage. |
| `frontend/` | React + TypeScript client (Vite): dashboard, the demo `Multimodal Analysis` page, and the service-backed `Modality Analysis` page (`src/pages/ModalityAnalysis.tsx`) with its typed inference client (`src/api/modalities.ts`). Node-managed dependencies. |
| `ml/` | Detection and scoring models, organized by modality and research concern. `ml/reproducibility/` (implemented) provides seeding, environment capture, and config access shared by every later module. |
| `forensic/` | File-level forensics that do not depend on a model: metadata, hashing, provenance, custody, evidence storage. |
| `threat_intelligence/` | Optional correlation against public sources: collectors, source registry, fingerprinting, similarity, campaign analysis, correlation. |
| `datasets/` | Dataset lifecycle: raw downloads, processed tensors/features, manifests, and split definitions. Also holds the implemented schema layer (`schema.py`, `manifest.py`, `split_schema.py`). |
| `experiments/` | Experiment code and run outputs, separated by research question. |
| `reports/` | Forensic report templates and generated reports. |
| `notebooks/` | Exploratory analysis notebooks. Exploratory only; anything citable belongs in `experiments/`. |
| `scripts/` | Reproducible automation: environment capture, dataset preparation, re-running experiments end to end. |
| `tests/` | Automated verification of structure, configuration, and (later) model/evidence behaviour. |
| `configs/` | Versioned YAML configuration. `project.yaml` is the root of the config hierarchy. |
| `docs/` | Research documentation: methodology, dataset notes, threat models, findings, and design decisions. |
| `README.md` | This document. |
| `.gitignore` | Versioned code only; data, weights, and renderings are untracked. |
| `requirements.txt` | Pinned lightweight Python dependencies, the one active media backend (opencv-python-headless), plus a documented block of not-yet-installed heavy stacks. |
| `ruff.toml` | Pinned lint ruleset, line length, and exclusions, so `ruff check .` is reproducible. |
| `mypy.ini` | Pinned type-check configuration, so `mypy` is reproducible. |

### Python modules (implemented)

| Path | Purpose |
| --- | --- |
| `ml/reproducibility/seeds.py` | `set_global_seed()` seeds Python `random` and NumPy. `derive_seed()` produces per-run seeds from a base seed. Framework seeding is deliberately absent. |
| `ml/reproducibility/environment.py` | Captures interpreter version, OS, platform, package versions, and git commit into a JSON-serializable report. Missing information is reported as `null`, never inferred. |
| `ml/reproducibility/config.py` | Loads and structurally validates `configs/project.yaml`, exposing typed section access and dotted key lookup. |
| `datasets/schema.py` | `MediaSample` plus the `Modality`, `SampleLabel`, `SplitName`, and `ManipulationType` vocabularies. |
| `datasets/manifest.py` | JSONL load/write/validate, chunked SHA-256, per-sample verification, and order-independent manifest hashing. |
| `datasets/split_schema.py` | `FrozenSplit` with disjointness enforcement and deterministic content hashing. Named `split_schema.py` because `datasets/splits/` already holds split *files*. |
| `datasets/normalization.py` | The only route from a foreign vocabulary to a canonical one: explicit `LabelMapping`, `ModalityMapping`, and `ManipulationMapping` tables, plus token-shape folding. |
| `datasets/adapters/base.py` | The `DatasetAdapter` contract and `RawSample`. What an adapter must never do: guess, infer from a filename, invent, or modify source media. |
| `datasets/adapters/template.py` | `TemplateAdapter`, a configuration-driven adapter for directory-and-label layouts. The worked example of the configuration format; not an adapter for any real dataset. |
| `datasets/adapters/faceforensics_plus_plus.py` | `FaceForensicsPlusPlusAdapter`: reads a declared FaceForensics++ layout, reports what is present, missing, or undeclared, and refuses to ingest while its licence is an unverified marker. |
| `datasets/provenance.py` | `DatasetProvenance`: dataset id/name/version, licence, access date, manifest hash, and ingestion timestamp, with fields that were never recorded listed rather than filled. |
| `datasets/ingest.py` | The pipeline and CLI: discover, normalize, hash, summarize, write manifest + provenance, verify digests. Supports `--dry-run`. |
| `datasets/registry.py` | The dataset registry: closed vocabularies, strict YAML loading with duplicate-key detection, semantic and generalization validation, and the `python -m datasets.registry` CLI. Read-only; imports no network or media library. |
| `datasets/acquisition.py` | Local acquisition gates: read-only layout planning, then a recorded acquisition only when the dataset is registered, the licence is verified, and the version and access date are given explicitly. Copies nothing; emits a registry block for a person to paste. |
| `datasets/preprocessing/inspection.py` | Modality-aware media inspection: the `VideoDecoder`, `ImageDecoder`, and `AudioDecoder` protocols with their OpenCV/`wave` implementations, behind one `MediaInspector`. Records a file's digest even when it cannot be decoded, and keeps the declared frame count separate from the observed one. |
| `datasets/preprocessing/extraction.py` | Deterministic, seek-free frame selection and writing. `UNIFORM` and `EVERY_N` implemented; `FPS_BASED` and `TIMESTAMP_BASED` fail loudly rather than approximating. |
| `datasets/preprocessing/quality.py` | Conservative frame-quality policy. A `None` threshold means disabled, not zero; brightness and contrast checks are opt-in so low-light footage is not discarded. |
| `datasets/preprocessing/frames.py` | `FrameSample` plus the two identities that keep splits honest: `source_video_id` for unique frame ids, `source_recording_id` for grouping. Refuses to write a manifest record with no digest of its own. |
| `datasets/preprocessing/grouping.py` | Recording-level groups and leakage enforcement. Multiple video digests in one recording are expected; one video under two recording ids is reported. |
| `datasets/preprocessing/splitting.py` | Deterministic group-level assignment, label-mix stratification, and conversion to the project's `FrozenSplit`. SHA-256 ordering, never Python's salted `hash()`. |
| `datasets/preprocessing/pipeline.py` | Manifest in, artifacts out: inspect, extract, filter, group, split, verify, write. A dry run writes reports but no frame manifest and no split, because neither can honestly exist without the frames. |
| `experiments/schema.py` | `ExperimentConfig` and the four experiment families: known-dataset, cross-dataset, ablation, multimodal. |
| `experiments/results.py` | `ExperimentResult` and metrics types. Unmeasured metrics stay `null`. |
| `experiments/run_artifacts.py` | The five-file run directory layout and its IO. |
| `ml/inference/status.py` | `AnalysisStatus`: the authority on whether a request produced a score, distinct from missing fields. |
| `ml/inference/evidence.py` | `EvidenceItem` with the `observed`/`derived`/`inferred` claim vocabulary, mirroring the frontend's `evidence.ts`. |
| `ml/inference/models.py` | `ModelDescriptor` (the model card), `ModelOutput`, the `ModelBackend` protocol, `ModelRegistry` (scans `*/model.json`), and `load_model_backend`, which verifies weights by digest and explains every failure rather than fabricating a model. |
| `ml/inference/report.py` | `AnalysisReport`: status, model identity, output, risk, evidence, warnings, errors, and the standing recommendation. |
| `ml/inference/request.py` | `AnalysisRequest` and the request-scoped helpers `utc_now`/`new_request_id`. |
| `ml/inference/service.py` | `MediaAnalysisService`: routes a request to the analyzer for its modality across image, video, audio, and QR; a modality with no analyzer gets an explicit `unsupported_modality` report. |
| `ml/inference/modality.py` | `AnalysisModality` (image/video/audio/qr), whose values reuse the `datasets.schema.Modality` strings so they compare and serialize identically. |
| `ml/inference/records.py` | `AnalysisRecord` (indexed header + full report) and `AnalysisStore`, an atomic, path-safe JSON file store (`SCHEMA_VERSION = 1`). Never copies raw media. |
| `ml/inference/multimodal.py` | `MultimodalReport`, `ReportSection`, and `MultimodalReportBuilder`: assemble a report from chosen records side by side, never fusing or averaging scores. |
| `ml/image/preprocess.py` | Injectable `ImagePixelDecoder` (OpenCV-backed) and a deterministic, model-agnostic transform to a fixed-size `float32` RGB array. |
| `ml/image/analyzer.py` | `ImageAnalyzer`: the concrete end-to-end image path — validate, preprocess, select and load a model, infer, assess risk, and assemble the report. |
| `ml/video/analyzer.py` | `VideoAnalyzer`: probe video stream properties as observed evidence; report `model_unavailable` with no video model, or `capability_unavailable` when frame preprocessing is unavailable. |
| `ml/audio/analyzer.py` | `AudioAnalyzer`: read a WAV via the standard library, recording stream properties; a missing decoder for MP3/FLAC is a `capability_unavailable` gap, not a failure. |
| `ml/qr/decode.py` | `QrDecoder` protocol and the OpenCV-backed `OpenCVQrDecoder`; a missing decoder surfaces as `QrDecoderUnavailableError`. |
| `ml/qr/analyzer.py` | `QrAnalyzer`: decode payloads and grade each URL; completes with `analysis_complete` and no detector score, so completion and scoring stay separate. |
| `ml/url/analyze.py` | `analyze_url`: purely syntactic URL/phishing heuristics (IP-literal host, punycode, userinfo, suspicious TLD, ...). Imports no network module and makes no request. |
| `ml/risk/assess.py` | `assess_risk` and `RiskPolicy`: map a detector score to a severity band, or `unknown` when no score exists, keeping raw score, probability, confidence, and risk distinct. |

### `ml/` — Detection models

| Path | Purpose |
| --- | --- |
| `ml/inference/` | Modality-agnostic inference core (implemented): request, evidence, model registry and loader, report, the service that routes by modality, the `AnalysisModality` vocabulary, the record store, and the multimodal report builder. |
| `ml/image/` | Image deepfake detection (implemented analyzer): preprocessing and `ImageAnalyzer`. The detector itself awaits weights. |
| `ml/audio/` | Audio analysis (implemented analyzer): stream properties from a standard-library WAV read; MP3/FLAC are honestly reported as a missing capability. A detector awaits weights. |
| `ml/video/` | Video analysis (implemented analyzer): stream-property probing and reported capability gaps. A temporal detector awaits weights. |
| `ml/qr/` | QR analysis (implemented): payload decoding via the `QrDecoder` protocol and OpenCV-backed decoder. Completes without a model; URL grading lives in `ml/url/`. |
| `ml/url/` | Syntactic URL/phishing analysis (implemented): offline heuristics over URL text. No network access. |
| `ml/fusion/` | Multimodal evidence fusion: combining image, audio, video, and forensic signals into a single assessment with calibrated confidence. |
| `ml/generalization/` | Cross-dataset generalization: domain-shift handling, model ensembles, and adapter/fine-tuning strategies. |
| `ml/explainability/` | Explainability and evidence localization: saliency, attention, perturbation tests, frame/segment-level localization, and manipulation-boundary detection. |
| `ml/risk/` | Risk and uncertainty assessment: calibration, out-of-distribution detection, abstention/rejection logic, and severity scoring for financial harm. `assess.py` (implemented) maps a score to a severity band honestly. |

### `forensic/` — File-level forensics

| Path | Purpose |
| --- | --- |
| `forensic/metadata/` | Container and EXIF metadata extraction: encoder, software, creation timestamps, device fingerprints, and suspicious-field detection. |
| `forensic/hashing/` | Cryptographic hashing (SHA-256/SHA-512) and perceptual hashing for content-identity matching across re-encodes and re-uploads. |
| `forensic/provenance/` | Provenance tracking: acquisition record, origin attribution, transform history, and C2PA-style content-credential verification. |
| `forensic/chain_of_custody/` | Chain-of-custody tracking: append-only, timestamped, actor-attributed handling events with integrity protection. |
| `forensic/evidence/` | Evidence store: immutable evidence items, claims derived from them, confidence and uncertainty attached to each claim, and report assembly. |

### `threat_intelligence/` — Public correlation

All network access is **disabled by default** and requires explicit opt-in
(`threat_intelligence.enabled`, `allow_network_access` in `configs/project.yaml`).

| Path | Purpose |
| --- | --- |
| `collectors/` | Adapters that fetch structured indicators from public sources, with rate limiting, caching, and provenance capture for every response. |
| `source_registry/` | Declarative registry of public sources: trust tier, update cadence, licence/usage terms, and collection policy. |
| `fingerprinting/` | Content fingerprinting for cross-platform matching: media fingerprints, watermark/logosign detection, audio fingerprinting. |
| `similarity/` | Near-duplicate and reuse detection across observed samples and known corpora, producing similarity links with confidence scores. |
| `campaign/` | Campaign-pattern analysis: clustering of linked samples, shared infrastructure and narrative, temporal sequencing, and actor-behaviour patterns. |
| `correlation/` | Cross-source correlation: joining forensics, model outputs, and threat intelligence into one reasoned assessment with an audit trail. |

### `datasets/` — Data lifecycle

| Path | Purpose | Versioned? |
| --- | --- | --- |
| `datasets/raw/` | As-acquired media, untouched, one subdirectory per dataset. Never modified in place. | No |
| `datasets/processed/` | Derived tensors, frames, spectrograms, embeddings, and cached features. Regenerable from `raw` + manifests. | No |
| `datasets/manifests/` | JSONL records per sample: identifier, modality, label, source, licence, path, hashes, and acquisition metadata. Each manifest sits beside its `<dataset_id>.provenance.json` record. | **Yes** |
| `datasets/splits/` | Train/validation/test split definitions with documented rationale and frozen evaluation folds. | **Yes** |
| `datasets/adapters/` | The adapter contract, the configuration-driven template adapter, and the FaceForensics++ adapter. | **Yes** |

Media bytes are never committed. Manifests, provenance records, and splits are the
reproducible, reviewable record of what was used.

### `experiments/` — Experiment runs

| Path | Purpose |
| --- | --- |
| `experiments/image/` | Image-modality experiments and their ablations. |
| `experiments/audio/` | Audio-modality experiments and their ablations. |
| `experiments/video/` | Video-modality experiments and their ablations. |
| `experiments/cross_dataset/` | Cross-dataset generalization studies: train-on-A/test-on-B matrices and domain-shift diagnosis. |
| `experiments/multimodal/` | Multimodal fusion experiments: score-combination strategies and modality-contribution analysis. |
| `experiments/ablation/` | Component ablations spanning modalities: what each signal actually contributes. |

Each subdirectory will hold experiment definitions and result summaries
(markdown/JSON/CSV, tracked in git); bulky outputs go to the ignored `experiments/runs/`.

### Supporting directories

| Path | Purpose |
| --- | --- |
| `reports/` | `templates/` holds the forensic report layout; generated output is ignored. |
| `notebooks/` | Exploratory notebooks. Reproducible findings must be promoted to `experiments/`. |
| `scripts/` | Reproducible automation: environment capture, dataset preparation, full re-runs, and structure verification. |
| `tests/` | Automated checks: `test_project_structure.py` (skeleton integrity, `configs/project.yaml` parses, `.gitignore` excludes data and weights), `test_ingestion.py` (the ingestion framework against a synthetic fixture), `test_dataset_registry.py` (registry contract, one broken rule per test), `test_faceforensics_adapter.py` (declared layout and explicit label mapping), and `test_acquisition.py` (acquisition gates and refusals). |
| `configs/` | `project.yaml` (root config), `datasets.yaml` (the dataset registry), `adapters/` (per-dataset adapter configurations), plus future per-experiment overrides. |
| `docs/` | Research documentation: `dataset_registry.md`, `dataset_acquisition.md`, `dataset_ingestion.md`, methodology, dataset notes, threat model, ethics, and findings. |

---

## Reproducibility Contract

Reproducible research is a hard requirement of this project. The following are enforced
by `configs/project.yaml` and checked by `tests/test_project_structure.py`.

1. **One config root.** All paths, seeds, thresholds, and module toggles live in
   `configs/project.yaml`. Nothing is hardcoded in module files.
2. **Global seed.** `reproducibility.global_seed: 20260101`, with
   `deterministic_algorithms: true` and `cudnn_deterministic: true`.
3. **Multi-seed protocol.** `experiments.seeds` defines three seeds; the default is three
   runs per seed, so reported numbers carry variance rather than a single lucky run.
4. **Named runs.** `reproducibility.naming_template` gives every run a deterministic
   identifier of the form `{module}/{task}/{dataset}/{seed}_{timestamp}`.
5. **Environment capture.** `record_environment: true` — pip freeze, git SHA, CUDA/driver
   versions, and platform are captured with each run.
6. **Hash-verified data loading.** `data.verify_hashes_on_load: true` against
   `data.hashing.algorithm: sha256`, so a silently changed sample fails loudly.
7. **Frozen splits.** Split definitions are committed under `datasets/splits/`. Tests are
   never tuned against; `evaluation.prohibit_test_set_tuning: true`.
8. **Ablations are mandatory.** `experiments.ablation_required: true` — a proposed model is
   not accepted without component ablations.
9. **Calibration is measured, not assumed.** `experiments.calibration` tracks ECE and
   Brier score alongside discrimination metrics.
10. **Uncertainty is a first-class output.** No model reports a bare score; see
    `ml/risk/`.
11. **Data bytes stay untracked.** `datasets/raw`, `datasets/processed`, weights, and
    embeddings are git-ignored; manifests and splits carry hashes instead.
12. **Network access is off by default.** Threat intelligence requires
    `threat_intelligence.enabled: true` **and** `allow_network_access: true`.

---

## Research Reproducibility

This section documents *how* the guarantees above are implemented, and where each one
lives in code. A number in this project is only meaningful alongside the seed, split,
data version, and software stack that produced it.

### Fixed seeds

`ml/reproducibility/seeds.py`

```python
from ml.reproducibility.seeds import derive_seed, set_global_seed

set_global_seed(derive_seed(20260101, run_number=0))  # run 0 of the base seed
set_global_seed(derive_seed(20260101, run_number=1))  # run 1 of the same base seed
```

- `set_global_seed` seeds Python's `random` and NumPy, and sets `PYTHONHASHSEED` for
  any subprocess spawned later. It is idempotent, and it records its own state in
  `get_seed_state()` so a test can assert that two runs were identical.
- `derive_seed` produces per-run seeds. `SeedStrategy.LINEAR` (the project default) is
  `base_seed + run_number`, which is trivial to restate in a paper.
  `SeedStrategy.HASHED` derives independent seeds for data-loader workers and
  sub-modules by hashing a label with the base seed.
- PyTorch, TensorFlow, and CUDA seeding are **absent by design**, because those
  libraries are not installed. `seed_worker()` therefore raises `NotImplementedError`
  rather than pretending to seed a worker. Extend `set_global_seed` with a guarded
  optional import when `ml/image` lands — do not add a hard dependency now.
- The base seed lives in `configs/project.yaml` as `reproducibility.global_seed`, and the
  seed set as `experiments.seeds`. Neither is hardcoded in a module.

### Frozen splits

`datasets/split_schema.py`, definitions stored under `datasets/splits/`

```python
from datasets.split_schema import FrozenSplit

split = FrozenSplit.create(
    "dataset_a",
    train_ids=[...],
    validation_ids=[...],
    test_ids=[...],
    seed=20260101,
    source_manifest_hash="sha256:...",  # the manifest this split was derived from
    rationale="grouped by source video to prevent near-duplicate leakage",
)
split.content_hash()  # "sha256:..." - identity of the partition itself
```

- Id sets are sorted on construction, so the content hash is **order-independent**: it
  answers "which samples are in this split", not "how was the file written".
- `content_hash()` deliberately **excludes** `created_at` and `rationale`. Rewording
  the justification for a split must not invalidate the identity of every run pinned
  to it.
- Construction **raises** if any id appears twice or in more than one split, so an
  invalid split object cannot exist in memory. `validate()` returns a report instead of
  raising, for use in reports and CI summaries.

### Dataset hashing

`datasets/manifest.py`

- `calculate_file_sha256(path)` streams the file in 1 MiB chunks. Multi-gigabyte video
  is never loaded into memory, and the result is chunk-size independent.
- `validate_sample_file(sample, base_dir)` returns `(ok, reason)` rather than raising,
  so one pass over a manifest can report **every** bad sample.
- `calculate_manifest_hash(samples)` canonicalizes and sorts records before hashing, so
  re-ordering lines in a JSONL file does not change the hash but changing any recorded
  value does.
- Every sample carries a required `sha256`. A sample whose content hash cannot be
  verified is not usable as evidence, so it is rejected at construction.
- `data.verify_hashes_on_load: true` makes ingestion fail loudly on a changed sample
  rather than silently training on different bytes.

### Experiment configuration

`experiments/schema.py`

Every experimental condition is declared *before* the run, as a serializable
`ExperimentConfig`. Four families are modelled explicitly, because each carries a
different obligation:

| Family | Additional requirement |
| --- | --- |
| `known_dataset` | Baseline: train and evaluate within one dataset. |
| `cross_dataset` | Requires `target_dataset`, and rejects a `target_dataset` equal to `training_dataset`. |
| `ablation` | Requires `ablation_of` and an `ablation_description` naming what was removed. |
| `multimodal` | Requires a per-modality `modality_datasets` mapping. |

- `config_hash` is computed over the semantic configuration, excluding `created_at`.
  Re-running the same condition therefore yields an identical hash, which is what makes
  two runs comparable.
- `splits_used` and `manifests_used` record the exact split and manifest digests a run
  depended on. Each must be a `sha256:` string, so a result cannot silently point at a
  dataset version that no longer exists.
- Validation accumulates every problem and raises once, so a malformed config is fixed
  in a single pass rather than one exception at a time.

### Experiment results

`experiments/results.py`

**A metric that was not measured is not reported.** Every metric field is
`float | None`, defaulting to `None`. There is no `0.0` default, because a missing
value rendered as zero is indistinguishable from a genuinely catastrophic score.
`ExperimentResult.missing_metrics` enumerates what was not measured, and a run that
aborted mid-evaluation serializes as a valid record with explicit nulls.

`GeneralizationGap` stores both operands (`in_domain`, `cross_domain`) rather than a
single signed number, because the sign convention differs across the literature.
`difference` reports `in_domain - cross_domain`; **positive means degradation**. It is
`None` if either operand is missing, so a gap is never computed from partial data.

### Environment capture

`ml/reproducibility/environment.py`

`capture_environment()` returns a JSON-serializable record of interpreter version,
implementation, OS, platform, byte order, filesystem encoding, CPU count, the local git
commit / branch / describe / dirty state, and installed package versions.

- Values are read from local metadata only. **No network access**, and heavy packages
  are looked up without importing them, so an absent or broken dependency cannot crash
  the capture.
- Absence is recorded as `null`, never guessed. Outside a git checkout, `git_commit` is
  `null` and the report says so.
- Optional heavy stacks are recorded explicitly (`torch: null` and so on), which
  documents that their absence was *observed* rather than assumed.

### Separation of training and test data

- Split definitions are **committed** under `datasets/splits/`, so the partition is
  reviewable in a pull request rather than regenerated on each machine.
- `datasets/manifests/` is committed; `datasets/raw/` and `datasets/processed/` are
  git-ignored. The manifest records paths and hashes, which is the reproducible record
  of what was used; the bytes are not.
- A cross-dataset experiment is structurally prevented from evaluating on its own
  training set.
- Known limitation: id-level disjointness is **necessary but not sufficient**. A frame
  extracted from a source video can sit in both train and test while sharing an id. Group
  near-duplicates by source recording before splitting, and record what you did in the
  split's `rationale` field. `datasets/manifest.py::assign_splits` is a random
  partition and does *not* group duplicates — it exists for smoke tests and for datasets
  known to be duplicate-free.

### No test-set tuning

- `evaluation.prohibit_test_set_tuning: true` in `configs/project.yaml`.
- `ExperimentConfig` separates `validation_dataset` from `test_dataset`, so model
  selection has a designated home that is not the test set.
- Splits are frozen: changing a split changes its content hash, which invalidates every
  result pinned to the old hash. A revised split is a new split and must be reported as
  one.
- `experiments.ablation_required: true` — a proposed model is not accepted without
  component ablations, so a headline number is never reported without knowing what
  produced it.

### Verification

```powershell
python -m ruff check .      # lint
python -m ruff format --check .
python -m mypy              # types
python -m pytest -q         # tests
python scripts\verify_structure.py
```

---

## Dataset Registry

`datasets/registry.py` and `configs/datasets.yaml` are the record of **which datasets
this project intends to use, for what, and what it has actually verified about them.**
Full documentation: **[`docs/dataset_registry.md`](docs/dataset_registry.md)**.

```powershell
# Validate. Exit code 0 valid, 1 invalid, 2 bad request.
python -m datasets.registry --validate

python -m datasets.registry --list                 # one line per registered source
python -m datasets.registry --show celeb_df_v2      # one entry in full, plus its pairs
python -m datasets.registry --validate --json      # machine-readable
```

```
VALID: <repo>\configs\datasets.yaml | registry_version=1 datasets=8 generalization=3 errors=0 warnings=11
```

A validation run is expected to print warnings. **Eleven warnings is the correct state of
this repository right now**: two sources declare training intent while their licence is
unverified, two declare usage flags while still a candidate, three intelligence feeds have
no verified access route, three planned generalization claims depend on unverified
licences, and one closing census records that `access_type`, `license_status`,
`source_reference`, and `version` are unverified for all eight entries. These are the
honest state of the registry, not defects to be silenced.

**The one rule everything else serves:** a dataset absent from the registry is not used; a
dataset present but unverified is not usable; a dataset usable but absent cannot be used.
This project would rather have an empty registry than a plausible one.

**Eight candidates, declared. Zero approved, acquired, or executed.**

| Registry entry | Role | Groups | Status |
| --- | --- | --- | --- |
| `faceforensics_plus_plus` | source domain, training | image, video | candidate |
| `celeb_df_v2` | target domain, unseen test | image, video | candidate |
| `asvspoof_2021_df` | source domain, training | audio | candidate |
| `fakeavceleb` | target domain, unseen test | audio | candidate |
| `cic_trap4phish` | QR/URL phishing benchmark | — | `pending_verification` |
| `phishtank` | external intelligence | — | candidate |
| `urlhaus` | external intelligence | — | candidate |
| `tranco` | reference list | — | candidate |

| Planned pair | Source → target | Status |
| --- | --- | --- |
| `image_deepfake` | `faceforensics_plus_plus` → `celeb_df_v2` | planned, not executed |
| `video_deepfake` | `faceforensics_plus_plus` → `celeb_df_v2` | planned, not executed |
| `audio_deepfake` | `asvspoof_2021_df` → `fakeavceleb` | planned, not executed |

The registry is a planning layer, so it is built to fail loudly on the specific ways a
claim can be written before the evidence exists:

1. **Nothing unverified is ever invented.** `access_type`, `license_status`,
   `source_reference`, `version`, `approximate_size`, and the manipulation types hold an
   explicit marker (`unknown`, `pending_verification`, `not_verified`, `not_applicable`).
   No URL, licence name, file count, or request procedure is guessed — and an unknown
   count is never written as `0`, which would read as "verified empty".
2. **The held-out claim cannot be voided by configuration.** Training *or tuning* against a
   `target_domain` is an error, and so is using it as an in-domain test set.
3. **A generalization result requires data on both sides.** `executed: true` is refused
   unless the source and target are both `available_locally` with complete acquisition
   records. A paper-shaped claim is the easiest one to write and the hardest to earn.
4. **Intelligence feeds are never training corpora.** A phishing feed or a popularity
   ranking may not declare class labels, may not set `train_allowed`/`validation_allowed`,
   and may not be an endpoint of a generalization experiment. Presence in a ranking is
   never treated as a safety signal.
5. **Approval is gated on rights.** `status: approved` requires
   `license_status: verified`; `available_locally` requires an acquisition record with
   date, version, and path; `blocked` requires every usage flag false.
6. **The vocabulary is closed and the parser strict.** An unknown field is an error
   everywhere, including inside nested blocks, because a misspelled key otherwise looks
   like a configured, working field. Duplicate YAML keys are refused rather than silently
   resolved to the last value.

Advancing an entry is a documented sequence, not a single edit: read the dataset's own
documentation and licence terms by hand, replace the markers with verified values, write
the adapter configuration, then acquire — and only then record
`status: available_locally`. Steps 1–3 are decisions about rights and vocabulary; step 4
is the first step that puts data on disk, and it is gated by licence verification, not by
this file being valid. Step 4 is [`datasets/acquisition.py`](#dataset-acquisition), and it
prints the block to paste rather than writing it.

---

## Dataset Acquisition

`datasets/acquisition.py` stands in front of ingestion. It answers "is this directory laid
out the way the configuration says?" separately from "may this project claim to hold this
dataset, under these terms, at this version, on this date?"

Full documentation: **[`docs/dataset_acquisition.md`](docs/dataset_acquisition.md)**.

```powershell
# Read-only. Reports which manipulation directories are present, which are missing,
# and what is present that no declaration accounts for. Needs no licence.
python -m datasets.acquisition --dataset-id faceforensics_plus_plus `
    --source datasets\raw\faceforensics_plus_plus `
    --adapter-config configs\adapters\faceforensics_plus_plus.yaml

# Record it. Refuses unless every gate passes.
python -m datasets.acquisition --dataset-id faceforensics_plus_plus `
    --source datasets\raw\faceforensics_plus_plus `
    --adapter-config configs\adapters\faceforensics_plus_plus.yaml `
    --access-date 2026-01-15 --dataset-version <release> --record
```

```
dataset id        : faceforensics_plus_plus
adapter           : faceforensics_plus_plus v1
registered        : True
registry status   : candidate
license status    : pending_verification

source directory   : datasets\raw\faceforensics_plus_plus
candidate files    : 0 (0 bytes)
declared layout:
  original         MISSING  required files=0  bytes=0  label=original       manipulation=not_stated
  DeepFakes        MISSING  optional files=0  bytes=0  label=deepfakes      manipulation=deepfakes
  ...
blocker : adapter configuration still declares license: pending_verification, which is an
          unverified marker. Set the real licence in the adapter configuration before an
          acquisition can be recorded.
blocker : registry: license_status is 'pending_verification'; nothing may be acquired
          until the dataset's terms have been read and recorded as verified
```

Both refuse today, deliberately. `license: pending_verification` is not a licence: it says
nobody has read the terms yet, and it keeps every step after reading the directory closed.
An adapter that can read a directory is not permission to use one.

What the layer does **not** do: download anything, copy or modify media, invent a date,
version, licence, or file count, or rewrite `configs/datasets.yaml`. With `--record` it
writes one JSON file under `datasets/manifests/` that says `"bytes_copied": false`, and
prints the registry block for a person to paste — because advancing a registry entry is a
decision about rights, and it belongs in a reviewed diff beside the comments explaining it.

---

## Dataset Ingestion

`datasets/ingest.py` converts a dataset on disk into a standardized JSONL manifest and a
provenance record. Full documentation: **[`docs/dataset_ingestion.md`](docs/dataset_ingestion.md)**;
adapter config reference: **[`configs/adapters/README.md`](configs/adapters/README.md)**.

```powershell
# Preview: discovers, validates, hashes, and counts. Writes nothing.
python -m datasets.ingest --dataset-id demo_corpus `
    --adapter-config configs\adapters\demo_corpus.json --dry-run

# Write datasets\manifests\demo_corpus.jsonl and its provenance record.
python -m datasets.ingest --dataset-id demo_corpus `
    --adapter-config configs\adapters\demo_corpus.json
```

```
dataset            : demo_corpus
adapter            : template v1
dataset version    : demo-1
mode               : dry-run (nothing written)
samples            : 6
  real             : 3
  fake             : 3
modalities         : audio=2 image=2 video=2
manipulations      : face_swap=3 unknown=3
missing metadata   : manipulation_type=3 source=6
hash verification  : verified
manifest hash      : sha256:84b8ebf6fc5b796b0c25eb5a139661bb7a9b37a2a3fc5ebbcddf8d718398545b
```

Four properties are enforced rather than documented-and-hoped-for:

1. **Labels are declared, never inferred.** A dataset's wording reaches the schema only
   through a mapping its own adapter declared. `"forged"` is not `fake` because it sounds
   like it; an undeclared token is an error naming the tokens that *are* declared.
2. **Absence stays absent.** No manipulation stated means `unknown` — a different
   statement from `other`, meaning the dataset said something that does not fit. No
   origin, no access date, no URL: `null`, and listed in the record's `unknown_fields`.
3. **Modality is authoritative or mapped, never read from a file extension.** `.mp4` does
   not prove a file contains video, and a confidently wrong modality is worse than a
   flagged sample.
4. **Source media is read-only, and the manifest is reproducible.** Identical inputs
   produce identical manifest bytes — no timestamp enters it; the ingestion timestamp
   lives in the provenance record alone.

Anything the adapter cannot account for is reported and counted rather than dropped: the
summary states how many samples lack each optional field, and a run with any sample error
writes no manifest at all, because a reader cannot tell from the file which records are
trustworthy.

**State of this stage:** no dataset has been selected or ingested. The only registered
adapter is the configuration-driven `template`. Splitting is *not* part of ingestion —
manifests carry `split: null` and authoritative assignments live in the frozen definitions
under `datasets/splits/`. Group-aware splitting is the next stage down; see
[Media Preprocessing](#media-preprocessing).

---

## Media Preprocessing

`datasets/preprocessing/` turns a validated video manifest into frames and a frozen split,
and inspects all three modalities — video, image, and audio — before anything else.
Full documentation: **[`docs/preprocessing.md`](docs/preprocessing.md)**.

```python
from datasets.manifest import load_manifest
from datasets.preprocessing.extraction import (
    ExtractionConfig,
    OpenCVFrameWriter,
    VideoFrameExtractor,
)
from datasets.preprocessing.frames import SourceIdentityLayout
from datasets.preprocessing.inspection import MediaInspector, default_decoder
from datasets.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline

decoder = default_decoder()
extractor = VideoFrameExtractor(
    decoder, OpenCVFrameWriter("png"), ExtractionConfig(frames_per_video=16)
)

result = PreprocessingPipeline(
    MediaInspector(decoder),
    extractor,
    PreprocessingConfig(
        extraction=extractor.config,
        source_identity_layout=SourceIdentityLayout.DROP_FIRST_SEGMENT,
    ),
).run(
    load_manifest("datasets/manifests/ffpp.jsonl"),
    base_dir="datasets/raw/ffpp",
    output_root="datasets/processed/ffpp",
)

print(result.summary())
assert result.is_leak_free
```

The load-bearing detail is that there are **two** identities, not one. A clip and its
forgeries share a `source_recording_id` and are therefore grouped and split together, but
each is a separate file with its own `source_video_id`, so their frame ids and output
directories stay distinct. Deriving both from the same path is how a genuine clip ends up
in train and its forgery in test, and the model learns the split instead of the
manipulation.

Five properties are enforced rather than documented-and-hoped-for:

1. **Frame ids are unique per file.** Two 4-ary manipulations of one recording each have a
   `frame_000000.png`. A collision overwrites one with the other, silently.
2. **No group spans two splits.** The check runs before the split is returned, so a leaky
   assignment raises instead of being written and annotated.
3. **The declared frame count is never mistaken for the real one.** Both are recorded, a
   disagreement is a warning, and sampling uses the count obtained by decoding.
4. **The frame manifest means it.** Every frame it lists has been written and hashed, so
   the digest is a fact about bytes. A dry run writes no images, so it writes no frame
   manifest and no split — a manifest with invented digests would pass verification
   without checking anything.
5. **Failures stay visible.** A video that cannot be decoded appears in
   `inspection.json` with its reason. Keeping only successes is how a run loses 40 videos
   and reports 1000 inspected.

The same inspector reads stills and audio before frame extraction begins, with one record
type and one rule set. `MediaInspector.inspect_auto` decides modality by extension;
explicit `inspect_image`/`inspect_audio` are the authoritative form when the manifest
already asserts a modality:

```python
from datasets.preprocessing.inspection import MediaInspector

inspector = MediaInspector.from_project_config()  # all three bundled decoders + config
row = inspector.inspect_image("faces/frame.png")   # width, height, channels, colour mode
row = inspector.inspect_audio("calls/call.wav")    # sample rate, channels, duration
```

Images report channels as the decoder sees them (`BGR`, never silently `RGB`); audio uses
the standard-library `wave` reader, exact for WAV and an explicit `decode_failed` for
anything else.

**State of this stage:** verified against synthetic media only. FaceForensics++ is still
`license_status: pending_verification`, so no real footage has been through this pipeline.

---

## Inference

`ml/inference/`, the modality packages (`ml/image/`, `ml/video/`, `ml/audio/`,
`ml/qr/`, `ml/url/`), and `ml/risk/` form one end-to-end analysis path for four
modalities: validate the input, record its observed identity (and preprocess it for
image), select and load a model where one exists, produce a score, assess risk, and
return a report in which **every statement is tagged as observed, derived, or
inferred**. `ml/inference/records.py` stores records and `ml/inference/multimodal.py`
assembles a report from chosen records. Full documentation:
**[`docs/inference.md`](docs/inference.md)**.

```python
from ml.inference.request import AnalysisRequest
from ml.inference.service import MediaAnalysisService

service = MediaAnalysisService.from_project_config()
report = service.analyze(AnalysisRequest(path="suspicious.png", modality="image"))

print(report.status.value)          # image, video, audio, and qr are routed
print(report.has_score)             # scoring is separate from completion
for item in report.evidence:
    print(item.claim_kind.value, item.label, item.value)
```

**The honest output of a checkout with no weights is `model_unavailable`** for image and
video, and `capability_unavailable` for non-WAV audio. This is a successful, verified
result — the file was read, hashed, and dimensioned (and preprocessed for image) — not an
error. A QR analysis decodes and completes with `status="analysis_complete"` but
`has_score=False`: completion and scoring are separate questions, answered by separate
fields. In every case the report carries the observed and derived evidence, states plainly
what could not be produced, and recommends independent verification. What it never does is
invent a confidence score or fuse incomparable per-modality scores into one number.

Models live as artifacts with a small "model card":

```
<artifacts>/models/<model_id>/model.json     # the card
<artifacts>/models/<model_id>/<weights>      # optional, digest-verified
```

`ModelRegistry` scans that tree for `*/model.json`. `load_model_backend` checks, in order,
that the weights exist, that their SHA-256 matches the card, that the declared framework
imports, and that the `"module:factory"` entry point resolves — raising a
`ModelUnavailableError` whose `reason` names the failed step instead of returning a stub.

Five distinctions are enforced, not merely documented:

1. **Status is separate from values.** `score=None` never implies low risk; the
   `AnalysisStatus` says whether a score exists at all.
2. **Raw score is not a probability.** Every `ModelOutput` and `RiskAssessment` carries
   `score_kind` and a `calibrated` flag; an uncalibrated score is warned about and is
   never presented as a likelihood.
3. **Confidence is not the score.** `confidence` is `None` unless the model provides a
   real one, so an uncalibrated score cannot masquerade as confidence.
4. **Risk is a policy, not a probability.** Severity comes from an explicit, named
   `RiskPolicy` operating point; `unknown` means no score, not low risk.
5. **Claims are tagged.** Observed (re-read from the file), derived (deterministic
   transform), and inferred (model output) are recorded distinctly.

**State of this stage:** the pipeline is verified end-to-end with synthetic backends and
both fake and real (OpenCV) pixels. No real detector weights exist in the repository, so
against real media the current output is `model_unavailable`. The smallest legitimate
step to a real score is to drop a detector's weights and a `model.json` — with the
weights' SHA-256, its framework, and a `"module:factory"` entry point — into
`artifacts/models/<model_id>/`; no code change is required.

---

## Dataset Manifest Format (JSONL)

A manifest is one JSON object per line. Media bytes are never committed; the manifest
records the path and the SHA-256 of the content.

**Field reference**

| Field | Required | Type | Notes |
| --- | --- | --- | --- |
| `sample_id` | yes | string | Unique within `dataset_name`. |
| `dataset_name` | yes | string | e.g. `faceforensics_pp`. |
| `filepath` | yes | string | Relative to the dataset base dir, or absolute. |
| `modality` | yes | enum | Exactly one of `image`, `audio`, `video`. |
| `label` | yes | enum | Exactly one of `real`, `fake`. |
| `sha256` | yes | string | 64 lowercase hex characters. |
| `license` | yes | string | SPDX id or dataset-specific string. |
| `manipulation_type` | no | enum | Defaults to `unknown`. See `ManipulationType`. |
| `source` | no | string \| null | e.g. `original`, `reencoded`, `platform_download`. |
| `split` | no | enum \| null | `train`, `validation`, `test`, or `null`. Authoritative assignments live in the frozen split definition. |
| `metadata` | no | object | Free-form extras: duration, resolution, encoder, device. Never interpreted by the core schema. |

**Example — the exact contents of one line of a `datasets/manifests/*.jsonl` file**

```json
{"dataset_name":"demo_forensics","filepath":"manipulated/000142.mp4","label":"fake","license":"CC-BY-4.0","manipulation_type":"face_reenactment","metadata":{"duration_s":8.4,"height":1080,"width":1920},"modality":"video","sample_id":"000142","sha256":"3b1f8c0a5d6e9b2c4a7f0e1d8c3b5a9e2d6f4c8b1a5e9d2c7f3b6a0d4e8c1f5b","source":"original","split":"train"}
```

The same record pretty-printed, plus its matched authentic sample:

```json
{
  "sample_id": "000142",
  "dataset_name": "demo_forensics",
  "filepath": "manipulated/000142.mp4",
  "modality": "video",
  "label": "fake",
  "manipulation_type": "face_reenactment",
  "source": "original",
  "split": "train",
  "sha256": "3b1f8c0a5d6e9b2c4a7f0e1d8c3b5a9e2d6f4c8b1a5e9d2c7f3b6a0d4e8c1f5b",
  "license": "CC-BY-4.0",
  "metadata": {"width": 1920, "height": 1080, "duration_s": 8.4}
}
```

```json
{"sample_id":"000143","dataset_name":"demo_forensics","filepath":"original/000143.mp4","label":"real","license":"CC-BY-4.0","manipulation_type":"none","metadata":{"duration_s":7.9,"height":1080,"width":1920},"modality":"video","sample_id":"000143","sha256":"9c2d5e7a1b3f4a6c8e0d2b4f6a8c0e2d4b6f8a0c2e4d6f8b0a2c4e6d8f0b2a4c6e","source":"original","split":"train"}
```

**Conventions**

- Keys are written in sorted order, so a regenerated manifest is byte-identical and
  reviewable in git. `write_manifest()` enforces this.
- Unknown keys are preserved under `metadata._extra` rather than dropped, so a manifest
  written by a newer schema version does not silently lose information.
- Invalid records do not abort a read. `validate_manifest_file()` returns a report
  listing every problem; `load_manifest(strict=True)` raises with all errors attached.

**Frozen split definition** (stored under `datasets/splits/<dataset_name>.json`)

```json
{
  "dataset_name": "demo_forensics",
  "train_ids": ["000142", "000143"],
  "validation_ids": ["000144"],
  "test_ids": ["000145"],
  "created_at": "2026-01-01T00:00:00+00:00",
  "seed": 20260101,
  "source_manifest_hash": "sha256:6943f74b7eb7f53d62436dd75d15f82a153d3ef3c40d1f87a5e044c125dcf66e",
  "rationale": "Grouped by source recording before splitting to prevent near-duplicate leakage.",
  "split_schema_version": 1,
  "extra": {}
}
```

Its `content_hash()` is `sha256:1f330ae6d970272842197aa30ee9848e318127848326b499c36f5f9089d52c7d`. Note that
this is independent of `created_at` and `rationale`, so rewording the justification does
not invalidate the split's identity.

---

## Planned Module Map

| # | Capability | Location |
| --- | --- | --- |
| 1 | Image deepfake detection | `ml/image/` |
| 2 | Audio deepfake detection | `ml/audio/` |
| 3 | Video deepfake detection | `ml/video/` |
| 4 | Cross-dataset generalization | `ml/generalization/`, `experiments/cross_dataset/` |
| 5 | Multimodal evidence fusion | `ml/fusion/`, `experiments/multimodal/` |
| 6 | Explainability / evidence localization | `ml/explainability/` |
| 7 | File hashing and metadata analysis | `forensic/hashing/`, `forensic/metadata/` |
| 8 | Provenance and chain-of-custody tracking | `forensic/provenance/`, `forensic/chain_of_custody/` |
| 9 | Public threat-intelligence correlation | `threat_intelligence/` |
| 10 | Reused-content and campaign-pattern analysis | `threat_intelligence/similarity/`, `threat_intelligence/campaign/` |
| 11 | Financial communication context analysis | `financial_context` in `configs/project.yaml`, evaluated in `experiments/multimodal/` |
| 12 | Risk and uncertainty assessment | `ml/risk/` |
| 13 | Evidence-based deepfake forensic reports | `forensic/evidence/`, `reports/` |
| 14 | FastAPI backend | `backend/` |
| 15 | PostgreSQL database | `backend/migrations/` |
| 16 | React + TypeScript frontend | `frontend/` |

Items 1 (image detection) and 12 (risk) have working scaffolding in place — a
model-agnostic inference pipeline and a score-to-severity mapping — but the detectors
themselves await weights. See [Inference](#inference).

---

## Getting Started

```powershell
git init
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify the foundation:

```powershell
python scripts\verify_structure.py     # structure + config contract
python -m datasets.registry --validate # dataset registry + claim gates
python -m ruff check .                 # lint
python -m ruff format --check .        # formatting
python -m mypy                         # types
python -m pytest -q                    # tests
```

### Experiment run layout

Every completed run writes one immutable, self-contained directory:

```
experiments/runs/<experiment_id>/<run_id>/
|-- config.json         the exact experiment configuration, incl. config_hash
|-- metrics.json        the result schema, absent metrics kept explicitly null
|-- environment.json    interpreter, OS, platform, packages, git commit
|-- dataset_info.json   manifest and split hashes, dataset composition
`-- README.md           human-readable summary
```

`experiments/runs/` is git-ignored: run output is regenerated by re-running, and is not
the artifact of record. Curated result summaries belong in `reports/`. The layout is
implemented in `experiments/run_artifacts.py`; no run is generated at import time.

---

## Conventions

- **Language layout:** `snake_case` for modules, `PascalCase` for classes,
  `SCREAMING_SNAKE_CASE` for constants. Type hints on public functions.
- **Config access:** modules read `configs/project.yaml`; no duplicated constants.
- **Import style:** intra-project imports are absolute from the repository root
  (`from ml.fusion... import ...`).
- **Evidence first:** every analytic output is written as an *evidence item* with a source,
  a method, a timestamp, and a confidence — never as a bare assertion.
- **Uncertainty is never rounded away:** score, calibration method, and applicable caveats
  travel together.
- **No silent thresholds:** decision thresholds are explicit, sweepable, and recorded with
  the run.
- **No fabricated values:** an unmeasured metric is `null`, never `0`. Absence is data.
- **Lint and types are configured, not defaulted:** `ruff.toml` and `mypy.ini` pin the
  ruleset so results do not drift with a tool upgrade.

---

## Safety and Data Handling

- **Never redistribute media.** Only hashes, manifests, and derived features are committed.
- **Licence compliance.** Each dataset's licence and permitted use are recorded in its
  manifest before use.
- **Consent and privacy.** Real voices and faces of non-consenting individuals are out of
  scope for training. Datasets containing them require documented lawful basis.
- **No fabricated evidence.** Forensic output must distinguish *observed*, *derived*, and
  *inferred* claims, and state when evidence is insufficient.
- **Rate limits and terms.** Threat-intelligence collection respects source terms,
  `robots.txt`, and the configured rate limit; it is off by default.
- **Dual-use.** Manipulation-detection research is dual-use. Detection capability is
  published with mitigations in mind, and offensive tooling is out of scope for this
  repository.

---

## Next Step

The registry now names what this project intends to use and admits what it has not
verified. The natural next increment is **verification, not acquisition**: read one
dataset's licence and access terms by hand, replace its markers in `configs/datasets.yaml`
with verified values, and write the corresponding `configs/adapters/<dataset_id>.json`.
Only then does acquisition become permissible — and it begins by ingesting one dataset
and freezing a split under `datasets/splits/` with a documented near-duplicate grouping
strategy. Still without training anything.

License is intentionally unset at this stage.