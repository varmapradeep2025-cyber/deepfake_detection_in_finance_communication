# Local dataset acquisition

How a dataset comes to be *declared* as belonging to this project.

This is a separate step from ingestion, and it happens first. Ingestion answers
"what is in this directory"; acquisition answers "may this project claim to hold
this dataset, under these terms, as of this date". Conflating the two is how a
directory that somebody downloaded turns into a result this project reports as its
own.

## What this layer does

`datasets/acquisition.py` answers two questions and keeps them apart.

**Plan** — inspect a local directory against the declared layout and report what
is there. Read-only. Writes nothing. Available with no licence at all, because
checking whether an unpacked archive looks right is not a use of the data.

**Record** — write a statement that this dataset exists here, under these terms,
at this version, as of this date. Only when every gate below is satisfied.

```powershell
# What is in this directory? Safe to run at any time.
python -m datasets.acquisition --dataset-id faceforensics_plus_plus `
    --source datasets\raw\faceforensics_plus_plus `
    --adapter-config configs\adapters\faceforensics_plus_plus.yaml

# Claim it, once the gates pass.
python -m datasets.acquisition --dataset-id faceforensics_plus_plus `
    --source datasets\raw\faceforensics_plus_plus `
    --adapter-config configs\adapters\faceforensics_plus_plus.yaml `
    --access-date 2026-01-15 --dataset-version <the release you downloaded> --record
```

Exit codes: `0` clean, `1` blocked, `2` a malformed request. `--json` prints the
plan as JSON.

## The gates

All of them. A record is a claim, so it is all-or-nothing, and every unmet gate is
named rather than the first one found.

| Gate | Why |
| --- | --- |
| The directory is present. | Presence is necessary, and is not sufficient for anything else. |
| The layout resolves. | A missing required manipulation directory means the tree is not what the configuration describes, so the sample set is not what it claims to be. |
| The adapter's `license` is not a marker. | `pending_verification` states that nobody has read the terms. |
| The dataset is in `configs/datasets.yaml`. | An unregistered dataset is not one this project uses. |
| `license_status: verified`. | The registry, not this module, records what the terms permit. |
| `status` is not `blocked` or `deprecated`. | Those statuses exist precisely to forbid use. |
| `--access-date` is a real `YYYY-MM-DD` date. | Supplied explicitly. Never inferred from a file timestamp, never filled in with today. |
| `--dataset-version` is not a marker. | "Whatever was on the site" is not a version, and nothing can be reproduced against it. |

Warnings do not block, but several are worth reading: a dataset still marked
`candidate`, `requires_terms_acceptance: true`, or usage flags that were never
verified.

## What it writes, and where

With `--record`, one file:

```
datasets/manifests/<dataset_id>.acquisition.json
```

It states the dataset id, version, access date, declared local path, licence,
adapter name and version, the full layout report at acquisition time, and two
explicit negatives:

```json
"acquisition_method": "local_directory",
"bytes_copied": false,
"media_modified": false
```

Nothing is copied, moved, renamed, or modified. The record says so, and a test
asserts it — a tracked record claiming this layer copied media describes something
it cannot do, and is rejected on read.

The record is deterministic: identical facts serialize to identical bytes, with
no timestamp beyond the access date you supplied. `content_hash()` is a SHA-256
over the canonical form, so a change to any recorded fact shows up in review.

The local path is stored **as you passed it**, not resolved to an absolute path,
so a tracked record does not carry one machine's directory layout.

## Why it does not edit the registry

`registry_block_for(record)` prints the exact `acquisition:` block to paste into
`configs/datasets.yaml`. It does not apply it.

Advancing a registry entry is a decision about rights and provenance, and it
belongs in a reviewed diff beside the comments that explain the entry. Rewriting
the file programmatically would mean either losing those comments and the
ordering — a YAML round-trip destroys both — or editing the text behind your back.
So the machine writes the evidence and you write the decision:

```yaml
    acquisition:
      acquired: true
      access_date: 2026-01-15
      version: <the release you downloaded>
      local_path: datasets/raw/faceforensics_plus_plus
```

Then set the entry's `status: available_locally`, which the registry requires to
be accompanied by a complete acquisition block. The registry refuses
`acquired: true` under any other status, so the two cannot drift apart.

## What this layer will not do

* **Download anything.** FaceForensics++ must already be acquired, by a person,
  under terms they have read.
* **Copy or modify media.** It reads a directory and writes a JSON file.
* **Mark a dataset acquired because a directory exists.** Presence proves
  presence. The gates decide the rest.
* **Invent a date, version, licence, reference, or file count.** Every one of
  those is either supplied or refused.
* **Proceed past a missing required directory.** Absent is reported as absent and
  empty is reported as empty; neither is padded with invented samples.

## Ordering, relative to the rest of the project

```
acquired locally by a person
  → acquisition plan      (datasets/acquisition.py)   read-only
  → record                (datasets/acquisition.py)   gated
  → registry edit         (configs/datasets.yaml)     by hand
  → ingestion             (datasets/ingest.py)        dry run, then write
  → manifest              (datasets/manifests/)       hashes, no timestamps
  → provenance            (datasets/provenance.py)    what ran, on what
```

The manifest and provenance steps come after this one, and none of them may run
on a dataset that has not cleared these gates. See
[docs/dataset_ingestion.md](dataset_ingestion.md) for the ingestion contract and
[docs/dataset_registry.md](dataset_registry.md) for the registry the gates read.
