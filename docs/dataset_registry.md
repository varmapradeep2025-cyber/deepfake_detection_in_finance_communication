# Dataset registry

What this project intends to use, what it has actually verified, and what it therefore
does **not** claim yet.

Implemented in `datasets/registry.py`, configured by `configs/datasets.yaml`, and pointed
at by `dataset_registry.file` in `configs/project.yaml`.

---

## What this stage does and does not do

**Does**

- Hold one reviewable record per candidate dataset or public-intelligence source.
- Declare, before any data exists, which experiment a dataset is a source for and which
  is its held-out target.
- Refuse a configuration that would make a claim untrue: training on a target domain,
  an intelligence feed used as a training corpus, a generalization result declared
  without data behind it.
- Report, in one place, every field that is still an explicit marker instead of a value.
- Keep the vocabulary closed and the loader strict, so a typo is an error rather than a
  silently ignored field.

**Does not**

- Select, download, verify, or acquire anything. Every entry is `candidate`, nothing is
  `approved`, and `dataset_registry.approved_datasets` is `[]`.
- Know any licence, version, size, download URL, or request procedure. Those fields exist
  so the gaps are visible in review, not so they can be guessed.
- Implement a collector, a licence checker, or a downloader.
- Produce a metric, a benchmark number, or a cross-dataset result.

---

## The single most important rule

> **A dataset not present in this registry is not used. A dataset present but not
> verified is not usable. A dataset usable but not yet in the registry cannot be used.**

`configs/datasets.yaml` is the only place that answers "is this allowed to be used?" for
a dataset. Code does not carry that answer, because code answers it silently.

---

## Claim boundaries

These are the claims the registry is designed to make falsifiable.

| Claim the project wants to make | What the registry must show first |
| --- | --- |
| "Trained on dataset *X*" | `X` registered, `status: available_locally`, `license_status: verified`, `train_allowed: true`, and a complete acquisition record |
| "Evaluated on unseen dataset *Y*" | `Y` registered with `research_role: target_domain` and `unseen_test_allowed: true` |
| "Generalizes from *X* to *Y*" | An explicit `generalization` pair naming both, both declaring the same `group_id`, `executed: false` until the endpoints exist locally |
| "URL *U* appeared in a feed" | A feed entry exists; the observation is recorded as evidence state, never as a class label |
| "Ranking context for domain *D*" | A `reference_list` entry exists; presence in a ranking is never treated as a safety signal |

The inverse also holds. A dataset absent from the registry cannot be claimed at all, and
a *cancelled* entry stays in the file with `status: deprecated` rather than being deleted,
so the history of what was considered survives.

---

## Vocabulary

Closed sets, defined in code and exported from `datasets.registry`. Values are matched
case- and separator-insensitively (`in-domain-test` and `in_domain_test` are the same
role), and anything outside the set is rejected with the alternatives listed.

| Vocabulary | Values |
| --- | --- |
| `RegistryModality` | `image`, `audio`, `video`, `qr`, `url`, `multimodal`, `threat_intelligence` |
| `ResearchRole` | `training`, `validation`, `in_domain_test`, `unseen_test`, `external_benchmark`, `source_domain`, `target_domain`, `threat_intelligence`, `external_intelligence`, `reference`, `optional` |
| `RegistryStatus` | `candidate`, `approved`, `pending_verification`, `available_locally`, `blocked`, `deprecated` |
| `SourceType` | `research_dataset`, `benchmark_dataset`, `intelligence_feed`, `reference_list` |
| `AccessType` | `unknown`, `public_download`, `registration_required`, `request_required`, `api`, `feed`, `manual` |
| `LicenseStatus` | `unknown`, `pending_verification`, `verified`, `restricted`, `prohibited_for_use` |
| `LabelScheme` | `binary_real_fake`, `binary_benign_malicious`, `evidence_state`, `unlabeled`, `unknown` |
| `GeneralizationStatus` | `planned`, `blocked`, `executed` |

Three groups are declared: `image_deepfake`, `audio_deepfake`, `video_deepfake`. The
image and video experiments share a dataset pair but are separate experiments with
separate metrics; conflating them is exactly the kind of overstatement the group field
prevents.

---

## Unverified markers

`unknown`, `pending`, `pending_verification`, `not_verified`, `not_applicable`.

An unverified field holds a marker, never `null` and never a plausible guess. `null` is
reserved for genuinely absent structural values (an acquisition date before anything was
acquired), and is rejected for required text fields.

The distinction between `unknown` and `not_applicable` is load-bearing:
`manipulation_types: not_applicable` on a phishing feed is a claim that the concept does
not apply, while `manipulation_types: unknown` on a face dataset is an admission that it
was never checked.

---

## Entry shape

```yaml
faceforensics_plus_plus:
  dataset_id: faceforensics_plus_plus   # must match the mapping key
  name: FaceForensics++                  # required
  modality: [video, image]
  source_type: research_dataset
  research_role: [source_domain, training]
  status: candidate
  access_type: unknown                   # unknown | public_download | ...
  license_status: pending_verification
  source_reference: not_verified         # URL, DOI, or a marker — never a guess
  version: unknown
  train_allowed: true                    # intent, not permission
  validation_allowed: true
  test_allowed: true
  unseen_test_allowed: false
  generalization_groups: [image_deepfake, video_deepfake]
  approximate_size: unknown
  label_schema:
    scheme: binary_real_fake
    values: [real, fake]
    description: >-
      Intended scheme, unverified. …
  manipulation_types: unknown            # or a list, or a marker
  requires_request: unknown              # true | false | unknown
  requires_terms_acceptance: unknown
  redistribution_allowed: pending_verification   # true | false | marker
  acquisition:
    acquired: false
    access_date: null                    # YYYY-MM-DD once acquired
    version: null
    local_path: null
  notes: >-
    What is planned, and what has not been checked.
```

Unknown fields are rejected everywhere, including inside `label_schema` and
`acquisition`. A misspelled key would otherwise look like a configured, working field.

`generalization_group` (singular) is accepted as an alias for `generalization_groups`;
declaring both is an error.

---

## The rules the loader enforces

Errors make the file invalid; warnings are recorded and reported but do not fail
validation.

**Structure**

- `registry_version` must equal the version the code understands (`1`).
- Duplicate YAML keys are refused rather than silently resolved to the last value.
- The root must be a mapping and must not be empty.
- `dataset_id` must match its mapping key, and must be unique across entries.

**Status and licence**

- `status: approved` requires `license_status: verified`.
- `status: blocked` requires every usage flag to be false.
- `status: available_locally` requires `acquisition.acquired: true` **and** a complete
  record: `access_date` (`YYYY-MM-DD`), `version`, and `local_path`.
- Acquisition recorded in the block requires the status to say `available_locally`. The
  registry will not describe data it has not recorded.

**Usage flags**

- `research_role: target_domain` requires `unseen_test_allowed: true`.
- `unseen_test_allowed: true` forbids `train_allowed`, `validation_allowed`, and the
  `training` role. Training *or tuning* against the target destroys the held-out claim
  it is supposed to support.
- `unseen_test_allowed: true` requires the `target_domain` role.
- A `target_domain` may not also be an in-domain `test_allowed` set; it is then no longer
  unseen.

**Label schemes**

- A detection corpus must declare a `label_schema`; an unlabeled entry is not a
  measurement corpus.
- An intelligence or reference source may not declare class labels, and may not set
  `train_allowed` or `validation_allowed`.
- `modality: multimodal` requires `derived_modalities` naming at least two components.

**Generalization pairs**

- Every pair needs `group_id`, `source`, `target`, and the source and target must be
  registered.
- Source and target must differ.
- The source must be trainable; the target must be usable as a held-out set.
- Both endpoints must declare the pair's `group_id`, and it must be a declared group.
- Intelligence sources may not take part in a generalization experiment.
- A blocked endpoint blocks its pair.
- `executed: true` and `status: executed` require both endpoints to be
  `available_locally`. This is the rule that prevents a claim from being written before
  the experiment exists.

**Warnings**

- Training intent declared against an unverified licence.
- Usage flags set while status is still `candidate` — intent is not permission.
- `access_type: unknown` on an intelligence source: no collector may be configured.
- `redistribution_allowed: true` on a research dataset that will never be redistributed.
- A generalization group an entry claims but no pair defines.
- One closing census of every field that is still a marker across the registry, so the
  total scope of what is unknown is a single number rather than a feeling.

---

## The registry as it stands

Eight candidate sources, zero approved, zero acquired, zero generalization results.

| Dataset | Role | Groups | Status |
| --- | --- | --- | --- |
| `faceforensics_plus_plus` | source domain, training | image, video | candidate |
| `celeb_df_v2` | target domain, unseen test | image, video | candidate |
| `asvspoof_2021_df` | source domain, training | audio | candidate |
| `fakeavceleb` | target domain, unseen test | audio | candidate |
| `cic_trap4phish` | QR and URL phishing benchmark | — | `pending_verification` |
| `phishtank` | external intelligence | — | candidate |
| `urlhaus` | external intelligence | — | candidate |
| `tranco` | reference list | — | candidate |

Declared pairs:

| Group | Source → target | Status |
| --- | --- | --- |
| `image_deepfake` | `faceforensics_plus_plus` → `celeb_df_v2` | planned, not executed |
| `video_deepfake` | `faceforensics_plus_plus` → `celeb_df_v2` | planned, not executed |
| `audio_deepfake` | `asvspoof_2021_df` → `fakeavceleb` | planned, not executed |

`cic_trap4phish` has no usage flags set at all. It is listed so the benchmark to be
built is named and reviewable; it is not a dataset this project may currently use.

---

## Quick start

```powershell
# Validate the registry. Exit code 0 valid, 1 invalid, 2 bad request.
python -m datasets.registry --validate

# List every registered source.
python -m datasets.registry --list

# Show one entry in full, including its unverified fields and its pairs.
python -m datasets.registry --show celeb_df_v2

# Machine-readable output for CI or a script.
python -m datasets.registry --validate --json

# Validate a file that is not the configured one.
python -m datasets.registry --registry configs\datasets.yaml --validate
```

Every command is read-only. The module imports no network or media library, and a test
asserts that no network import can appear in it.

---

## Library use

```python
from datasets.registry import default_registry_path, load_registry, validate_registry

report = validate_registry(default_registry_path())
print(report.summary())
for warning in report.warnings:
    print(" ", warning)

registry = load_registry(default_registry_path())
for entry in registry:
    print(entry.dataset_id, entry.status.value, entry.unverified_fields())
for pair in registry.generalization:
    print(pair.describe())
```

`load_registry` raises on a structurally invalid file; `validate_registry` returns a
report and never raises for content errors, which is what a CLI and a CI check want.

---

## What would move an entry forward

1. Read the dataset's own documentation and licence terms — by hand, from the source.
2. Replace the markers with verified values: `access_type`, `license_status`,
   `source_reference`, `version`, `approximate_size`, and the label scheme description.
3. Write the adapter configuration under `configs/adapters/` (see
   `docs/dataset_ingestion.md`).
4. Acquire the data, then set `status: available_locally` with a complete acquisition
   record. Only then run ingestion.
5. Switch the generalization pair to `executed` once both endpoints exist locally and the
   experiment has actually run.

Steps 1 to 3 are decisions about rights and vocabulary. Step 4 is the first step that puts
data on disk, and it is gated by licence verification — not by this file being valid.

---

## Related

- `docs/dataset_ingestion.md` — turning an acquired dataset into a manifest.
- `configs/adapters/README.md` — per-dataset adapter configuration.
- `configs/project.yaml` — `dataset_registry.file`, `approved_datasets`, `datasets_acquired`.
