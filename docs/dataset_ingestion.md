# Dataset ingestion

How an external dataset becomes a standardized, reviewable, reproducible record —
and what the pipeline refuses to do on its own.

Implemented in `datasets/ingest.py`, `datasets/adapters/`, `datasets/normalization.py`,
and `datasets/provenance.py`.

---

## What this stage does and does not do

**Does**

- Discover media files under a dataset root using a declared **adapter**.
- Translate the dataset's own vocabulary into the canonical schema **only through an
  explicitly declared mapping**.
- Compute a SHA-256 digest per sample, and a manifest-level digest over the record set.
- Write one JSONL manifest plus a provenance record, deterministically.
- Report everything it could not resolve instead of filling the gaps.

**Does not**

- Select, download, or acquire any dataset. No research dataset is configured here;
  `configs/project.yaml` records `ingestion.datasets_selected: []`.
- Modify, move, rename, or copy source media. The raw directory is read-only.
- Split data into train/validation/test. Authoritative assignments live in the frozen
  split definitions under `datasets/splits/`, so ingestion leaves `split` as `null`.
- Guess a label, a modality, or a manipulation type.

---

## Layout

| Path | Contents | Versioned? |
| --- | --- | --- |
| `datasets/raw/<dataset_id>/` | The dataset exactly as acquired. Never modified. | No |
| `datasets/manifests/<dataset_id>.jsonl` | One JSON record per sample. | **Yes** |
| `datasets/manifests/<dataset_id>.provenance.json` | Where the dataset came from, which adapter and script version produced the manifest. | **Yes** |
| `datasets/splits/<dataset_id>.json` | Frozen split definition. Written separately, not by ingestion. | **Yes** |
| `configs/adapters/<dataset_id>.yaml` | Declarative adapter configuration. JSON and YAML are both accepted. | **Yes** |
| `datasets/manifests/<dataset_id>.acquisition.json` | Acquisition record: that the dataset exists locally, under which terms, at which version, as of which date. Written by `datasets/acquisition.py`, never by ingestion. | **Yes** |

Media bytes never enter git. The manifest and the provenance record are the reviewable
record of what was used and where it came from.

---

## Quick start

```powershell
# 1. Acquire the dataset yourself, under terms you have read, and place it where the
#    convention expects it (or pass --source).
#    datasets\raw\<dataset_id>\

# 2. Check the layout before anything else. Read-only, and needs no licence.
python -m datasets.acquisition --dataset-id demo_corpus `
    --source datasets\raw\demo_corpus --adapter-config configs\adapters\demo_corpus.yaml

# 3. Write configs\adapters\<dataset_id>.yaml after reading the dataset's own
#    documentation and licence terms. See configs/adapters/README.md.
#    Then record the acquisition: python -m datasets.acquisition ... --record

# 4. Preview. Nothing is written, nothing is modified.
python -m datasets.ingest --dataset-id demo_corpus `
    --adapter-config configs\adapters\demo_corpus.yaml --dry-run

# 5. Read the counts. Confirm modality, label balance, and what is missing.

# 6. Write the manifest and provenance record.
python -m datasets.ingest --dataset-id demo_corpus `
    --adapter-config configs\adapters\demo_corpus.yaml
```

Steps 2 and 3 are separate because they are separate questions. Step 2 asks whether the
directory looks like what the configuration describes, which is answerable before anyone
has cleared anything. Step 3 is a decision about rights, and ingestion will refuse while
the licence is still a marker. See [`dataset_acquisition.md`](dataset_acquisition.md).

`python -m datasets.ingest --help` lists every option; `--list-adapters` shows which
adapters are registered, and `--json` prints the summary as one line of canonical JSON
for scripting.

**Exit codes:** `0` success, `1` one or more samples failed validation, `2` the request
itself was unusable (unknown adapter, missing source directory, malformed config).

---

## The adapter contract

An adapter is one small class per dataset. It answers *what the files are* and declares
*what the dataset's words mean*. It never decides what a sample is.

`datasets/adapters/base.py`

| Member | Required | Purpose |
| --- | --- | --- |
| `discover(source_dir)` | yes | Yield one `RawSample` per candidate file, deterministically, read-only. |
| `label_mapping()` | yes | The dataset's real/fake wording mapped to `real`/`fake`. |
| `modality_mapping()` | no | Defaults to empty; modality may instead come from the dataset's structure. |
| `manipulation_mapping()` | no | Defaults to empty; an absent manipulation then resolves to `unknown`. |
| `validate_configuration()` | no | Pre-flight self-check, run on every run including dry runs. |
| `describe()` | no | Machine-readable vocabulary dump, included in the dry-run report. |
| `from_dataset_id(dataset_id)` | no | Construct from a hardcoded vocabulary, or return `None`. |
| `from_config_file(path)` | no | Construct from a JSON or YAML configuration file, or return `None`. |

`RawSample` records what the *dataset* says, in the dataset's own words:
`sample_id`, `path`, `label_token`, `modality`, `modality_token`, `manipulation_token`,
`source_token`, `metadata`. A field the dataset does not supply stays `None`.

### Writing a Python adapter

Only worth it when a dataset's layout cannot be expressed as directory rules — nested
archives, per-file sidecar metadata, or a manifest file listing ids. Extend
`DatasetAdapter`, implement `discover` and `label_mapping`, and register it:

```python
from datasets.adapters import DatasetAdapter, RawSample, register_adapter
from datasets.normalization import LabelMapping
from datasets.schema import Modality


class DemoAdapter(DatasetAdapter):
    name = "demo"
    adapter_version = "1"

    def __init__(self) -> None:
        super().__init__(dataset_name="Demo Corpus", license_id="CC-BY-4.0", dataset_version="1")
        self._labels = LabelMapping({"original": "real", "manipulated": "fake"})

    def label_mapping(self) -> LabelMapping:
        return self._labels

    def discover(self, source_dir):
        for path in sorted(source_dir.rglob("*.wav")):
            yield RawSample(
                sample_id=path.stem,
                path=path,
                label_token="original",
                modality=Modality.AUDIO,
            )


register_adapter(DemoAdapter)
```

`discover` here reads modality from the file extension, which is only acceptable because
the adapter *itself* is asserting a documented property of this dataset — "every file in
this corpus is audio". Where that is not true of the whole dataset, use a directory rule
instead and let an unexplained file be flagged.

A licence is mandatory: `DatasetAdapter.__init__` raises `AdapterConfigurationError`
without one. Licensing governs whether the data may be used at all, so it is not a
default and not a CLI flag.

### The template adapter

`datasets/adapters/template.py` implements the same contract from a configuration file,
which is the right choice for a dataset whose layout is "directories plus a label
convention". `configs/adapters/README.md` is the key reference. First matching rule wins;
`default_rule` covers the rest; files matching neither are still yielded, unlabelled, and
flagged by the pipeline.

### The FaceForensics++ adapter

`datasets/adapters/faceforensics_plus_plus.py` is the first adapter for a real dataset. It
takes a declared set of manipulation directories rather than directory rules, because the
dataset's layout is not negotiable and matching it loosely is how material silently loses
its label. It adds three things to the contract:

- **`inspect(source_dir)`** — a read-only layout report: which declared directories exist,
  which are missing, which hold nothing, and what is present that no declaration accounts
  for. This is what `datasets/acquisition.py` plans against.
- **`label` per directory** — each declared directory states the canonical label it maps
  to, so `Face2Face` means `fake` because the configuration says so, not because the name
  contains the word "Face".
- **A licence marker check** — `license: pending_verification` is a marker, not a licence.
  `validate_configuration()` raises while one is in place, so no manifest can be produced
  from a dataset whose terms nobody has read.

`required_manipulations` decides what ingestion cannot proceed without. A missing required
directory raises; a missing optional one is reported and ingestion continues. Neither is
padded with invented samples, and an absent directory is reported differently from an empty
one.

See [`dataset_acquisition.md`](dataset_acquisition.md) for the gates in front of ingestion.

---

## Normalization rules

`datasets/normalization.py`

1. **Nothing is guessed.** A dataset's words are translated only through a table its own
   adapter declared. Nothing concludes that `"forged"` means `fake` because the word
   sounds like it; an undeclared token is an error, not a guess. The error names the
   tokens that *are* declared, so the fix is a one-line table entry.
2. **Token shape is normalized, token meaning is not.** Case, surrounding whitespace,
   and `-` / space / `_` separators fold, so `"Face Swap"` and `"face-swap"` are one
   token rather than two vocabularies.
3. **Uncertainty is preserved.** A dataset that states no manipulation yields
   `unknown` — which is a different statement from `other`, meaning the dataset said
   something that does not fit. A dataset that classifies nothing yields no label, and
   the sample is flagged.
4. **Modality is authoritative or explicitly mapped, never inferred from the extension.**
   Resolution order: a modality the adapter derived from the dataset's own structure, then
   the dataset's modality token through the declared table, then failure. `.mp4` does not
   prove a file contains video, and a confidently wrong modality is worse than a flagged
   sample.

The dataset's own wording is copied into `metadata.source_label` (and
`metadata.source_manipulation_label`) whenever it differs from the canonical value, so a
future re-ingestion can be audited against what the dataset actually said.

---

## Outputs

**Manifest** — one JSON object per line, keys sorted, so identical inputs produce
identical bytes. Format documented in the README under
[Dataset Manifest Format (JSONL)](../README.md#dataset-manifest-format-jsonl).

**Provenance** — written beside the manifest as `<dataset_id>.provenance.json`:

```json
{
  "access_date": null,
  "adapter_name": "template",
  "adapter_version": "1",
  "dataset_id": "demo_corpus",
  "dataset_name": "Demo Corpus",
  "ingestion_script_version": "1",
  "ingestion_timestamp": "2026-10-04T14:31:54+00:00",
  "license": "CC-BY-4.0",
  "manifest_hash": "sha256:84b8ebf6fc5b796b0c25eb5a139661bb7a9b37a2a3fc5ebbcddf8d718398545b",
  "notes": "...",
  "provenance_schema_version": 1,
  "source": null,
  "source_reference": null,
  "version": "demo-1"
}
```

- The manifest carries **no timestamp**, so re-ingesting an unchanged dataset reproduces
  the manifest byte for byte. The timestamp lives here, and only here.
- `content_hash()` covers every field *except* `ingestion_timestamp`, so two runs over the
  same dataset share a content hash and a changed manifest hash is never confused with a
  rerun.
- `source`, `source_reference`, and `access_date` are `null` when unrecorded, and
  `unknown_fields` lists them. An invented URL is worse than an honest gap.

---

## Guarantees

| Guarantee | How it is enforced | Test |
| --- | --- | --- |
| Source media is never modified | Discovery is read-only; hashes are computed, never written back | `test_source_media_is_not_modified` |
| A dry run writes nothing | The write branch is unreachable in a dry run | `test_writes_nothing` |
| Identical inputs produce identical manifests | Canonical sort, no timestamp, sorted keys | `test_manifest_bytes_are_identical_across_runs` |
| Recorded digests match the files | Re-hash after writing; `--verify-hashes`/`--no-verify-hashes` | `test_hashes_match_the_files_on_disk`, `test_reports_failure_when_a_file_changes` |
| A partial dataset never becomes a manifest | Any sample error suppresses writing | `test_still_reports_errors`, `test_writes_nothing` |
| No manifest is silently overwritten | Refuses without `--overwrite` | `test_refuses_to_overwrite_without_permission` |
| Paths stay inside the dataset | A sample outside the source root is refused | `test_path_outside_the_source_root_is_refused` |
| Duplicate ids are reported, not disambiguated | A collision would make a split ambiguous | `test_duplicate_sample_ids_are_reported` |

---

## The summary

Every run reports label balance, modality and manipulation counts, which optional fields
the dataset never stated, and the hash-verification outcome:

```
dataset            : demo_corpus
adapter            : template v1
dataset version    : demo-1
script version     : 1
mode               : dry-run (nothing written)
source             : ...\datasets\raw\demo_corpus
samples            : 6
  real             : 3
  fake             : 3
modalities         : audio=2 image=2 video=2
manipulations      : face_swap=3 unknown=3
missing metadata   : manipulation_type=3 source=6
hash verification  : verified
manifest hash      : sha256:84b8ebf6fc5b796b0c25eb5a139661bb7a9b37a2a3fc5ebbcddf8d718398545b
ingestion time     : 2026-10-04T14:31:44+00:00
```

`missing metadata` is the line worth reading first. A corpus that states no origin and
no manipulation type produces a manifest whose rows are honest about that, which is
exactly what a later experiment needs to know before it draws conclusions.

`label_coverage` in `describe()` is also worth checking during a dry run: a single-class
result (`("real",)`) is legitimate for an authentic-media-only corpus and useless as a
detection benchmark.

---

## Adding a dataset, in order

1. Read the dataset's own documentation. Record its licence, version, and layout.
2. Write `configs/adapters/<dataset_id>.json`. Every label the dataset uses must appear
   in `labels`.
3. Dry run. Read the counts and every reported error.
4. Fix the adapter, not the data. A flagged sample is a question about the adapter's
   declarations.
5. Ingest for real; commit the manifest and the provenance record.
6. Define the frozen split under `datasets/splits/`, grouping near-duplicates by source
   recording, and record the grouping strategy in its `rationale`.

Step 6 is not optional in practice. Id-level disjointness is necessary but not
sufficient: `datasets/manifest.py::assign_splits` is a random partition and does **not**
group duplicates. Group-aware splitting is a documented open item, not a solved problem.

---

## Limitations

- **No dataset is ingested.** The framework is proven against a synthetic six-file fixture
  built in a temporary directory by `tests/synthetic_dataset.py`. Its `.mp4` files are
  ISO-BMFF stand-ins, not decodable video; nothing in this stage decodes media, and these
  fixtures exercise hashing and metadata paths, not codec behaviour.
- **Group-aware splitting lives one stage down.** Ingestion writes `split: null` by design;
  authoritative assignments come from `datasets/preprocessing/`, which groups frames by
  source recording before splitting. See [`preprocessing.md`](preprocessing.md).
- **No sidecar metadata extraction.** Whatever the adapter reads is all that is recorded;
  EXIF, container metadata, and encoder fingerprints belong to `forensic/metadata/`.
- **`TemplateAdapter` covers directory-and-label layouts only.** Archives, sidecar
  manifests, and per-file metadata need a Python adapter.
- **Labels are binary.** `SampleLabel` is `real`/`fake`; a dataset's own finer grading
  belongs in `metadata`, not in a third label value.