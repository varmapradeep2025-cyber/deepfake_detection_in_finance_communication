# Dataset adapter configurations

One configuration file per dataset, consumed by the dataset's adapter via
`python -m datasets.ingest --adapter-config <file>` or
`python -m datasets.acquisition --adapter-config <file>`.

JSON and YAML are both accepted, and both are read strictly: a duplicate key is an
error, not a last-one-wins surprise.

| File | Adapter | Purpose |
| --- | --- | --- |
| `faceforensics_plus_plus.yaml` | `FaceForensicsPlusPlusAdapter` | Reads a locally acquired FaceForensics++ directory. |
| *(template.json — see below)* | `TemplateAdapter` | The configuration-driven reference. |

## Two kinds of adapter

**The template adapter** is the general configuration format. It takes a set of
directory rules and applies the first match to every file. It exists to show the
shape of the format and to exercise the ingestion pipeline in tests. It is not a
dataset anyone reads, so it ships no configuration file.

**A dataset adapter** exists for one specific dataset. It exists only after
someone has read that dataset's own documentation and terms, because the layout it
hard-codes and the labels it states are claims about the dataset. FaceForensics++
is the first, in `faceforensics_plus_plus.yaml`.

A dataset adapter may only *describe* what it found. It never downloads anything,
never edits the raw directory, and never decides whether the data may be used —
that gate belongs to the registry and lives in `datasets/acquisition.py`.

## Template adapter keys

### Required keys

| Key | Meaning |
| --- | --- |
| `license` | SPDX id or the dataset's own licence string. Required — ingestion refuses to run without one. |
| `labels` | The dataset's real/fake wording mapped to `real`/`fake`. Every token used must appear here. |

### Optional keys

| Key | Meaning |
| --- | --- |
| `dataset_name` | Human-readable name for provenance. Defaults to the `--dataset-id`. |
| `dataset_version` | Release or version string. `unversioned` if unstated. |
| `source_reference` | Where the dataset came from. Omit rather than invent a URL. |
| `access_date` | When it was accessed (`YYYY-MM-DD`). |
| `modalities` | The dataset's modality wording mapped to `image`/`audio`/`video`. |
| `manipulations` | The dataset's manipulation wording mapped to a `ManipulationType`. |
| `directory_rules` | Per-directory attributes, applied in order, first match wins. |
| `default_rule` | Attributes for files no rule matched. Without it they are flagged. |
| `extensions` | Candidate file extensions, e.g. `[".mp4", ".wav"]`. Omit to accept all. |

### Rule keys

`path` (directory relative to the dataset root; `*` matches anything, `real/*`
matches one level), `label`, `modality`, `modality_token`, `manipulation`,
`source`.

### Skeleton

```json
{
  "dataset_name": "Example Corpus",
  "dataset_version": "1.0",
  "license": "CC-BY-4.0",
  "source_reference": null,
  "access_date": null,
  "labels": {"original": "real", "manipulated": "fake"},
  "modalities": {},
  "manipulations": {"faceswap": "face_swap"},
  "extensions": [".mp4", ".wav", ".png"],
  "directory_rules": [
    {"path": "original/*", "label": "original", "modality": "video"},
    {"path": "manipulated/faceswap", "label": "manipulated", "modality": "video",
     "manipulation": "faceswap"}
  ],
  "default_rule": null
}
```

## FaceForensics++ keys

The FaceForensics++ adapter does not take directory *rules*. It takes a declared
set of manipulation directories, because the dataset's layout is not negotiable
and matching it loosely is how material silently loses its label.

| Key | Meaning |
| --- | --- |
| `adapter` | Must be `faceforensics_plus_plus`. |
| `dataset_id` | Registry id. Must match the `--dataset-id` used to ingest. |
| `dataset_name` | Human-readable name for provenance. |
| `license` | Licence identifier. **Ships as `pending_verification`**, which is a marker, not a licence — see below. |
| `source_reference` | Where the data came from. Omit rather than invent a URL. |
| `dataset_version`, `access_date` | Release and access date, when known. |
| `modality` | `video` or `image`. |
| `extensions` | Candidate extensions. Files matching nothing are not counted. |
| `manipulations` | The declared layout. One entry per directory. See below. |
| `required_manipulations` | Which of those directories ingestion cannot proceed without. Defaults to `["original"]`; an explicit empty list is rejected. |
| `source_token_from_parent_directory` | `true` to record the source-recording folder as the sample's source token. Off by default. |
| `notes` | Reviewer remarks. |

### Each `manipulations` entry

```yaml
manipulations:
  Face2Face:
    label_token: face2face   # the dataset's own wording; the key the mapping is built on
    label: fake              # the canonical label it translates to
    manipulation: face2face  # the manipulation family, when the directory is one
```

`label_token` is what the mapping is keyed on, and `label` is what it resolves to.
`Face2Face` means `fake` because the line says `label: fake` — not because the
directory name contains the word "Face". Change the `label` and the directory's
meaning changes with it, which is the point: the mapping is a reviewed statement,
not a naming convention the code assumes.

`original` omits `manipulation` on purpose. Genuine material is not a
manipulation, so it resolves to `unknown` in the manifest rather than to a
fabricated "no manipulation" type.

Each directory needs its own `label_token`. Two directories sharing one would
collapse onto a single mapping entry, so half the tree would lose its label; that
is rejected at load time.

## The licence marker

`license: pending_verification` is not a licence. It states that nobody has read
the terms yet, and it keeps the pipeline closed:

* **Layout inspection works.** Point the acquisition tool at a directory and it
  reports which manipulation directories are present, how many files each holds,
  what is missing, and what is present that no declaration accounts for. That
  needs no licence, and it is the right first thing to check.
* **Ingestion refuses.** `validate_configuration()` raises, so no manifest can be
  produced from a dataset whose terms are unknown.
* **Acquisition refuses.** `record_acquisition()` raises, so no record of having
  acquired it can be written.

To open any of those, read the dataset's terms, replace the marker with the real
identifier, and set `license_status: verified` in `configs/datasets.yaml`. That is
a human decision recorded in two reviewed files.

## Checking a layout before ingesting

```powershell
python -m datasets.acquisition --dataset-id faceforensics_plus_plus `
    --source datasets\raw\faceforensics_plus_plus `
    --adapter-config configs\adapters\faceforensics_plus_plus.yaml
```

Read the report, then ingest as a dry run and read the counts again before writing
anything:

```powershell
python -m datasets.ingest --dataset-id faceforensics_plus_plus `
    --adapter-config configs\adapters\faceforensics_plus_plus.yaml --dry-run
```

See `docs/dataset_ingestion.md` for the ingestion contract and
`docs/dataset_acquisition.md` for the acquisition gates.
