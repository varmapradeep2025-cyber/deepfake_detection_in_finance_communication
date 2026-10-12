# Dataset access audit

A documentation-only record of the read-only access and licensing investigation
carried out for the two candidate face-manipulation datasets: **FaceForensics++**
and the **DeepFake Detection Challenge (DFDC)** dataset.

**Audit date: 2026-10-11.**

This note records what was *verified* from official sources and, just as
importantly, what was *not*. Nothing here is an acquisition, an approval, or a
licence determination. No dataset was downloaded, no form was submitted, no
package was installed, and no configuration, registry, adapter, or code file was
changed as part of this investigation.

---

## Scope and method

The investigation was **read-only**. It used only publicly retrievable pages —
official project sites, repository documentation, terms-of-use documents, and the
associated research papers. Where a source could not be retrieved, that is stated
as an unresolved question rather than filled in with an assumption.

Two rules governed the audit:

- A dataset is **not** treated as approved, usable, or legally suitable unless an
  **official** source establishes its terms.
- A claim that could not be confirmed from an official source is recorded as
  **unresolved**, not inferred from third-party summaries.

---

## 1. Official sources

### FaceForensics++

- Project repository: https://github.com/ondyari/FaceForensics
- Dataset documentation (README): https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md
- Terms of Use (PDF): http://kaldir.vc.in.tum.de/faceforensics_tos.pdf
- Project page: http://niessnerlab.org/projects/roessler2019faceforensicspp.html
- Benchmark documentation: https://kaldir.vc.in.tum.de/faceforensics_benchmark/documentation
- Access request form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
- Contact: faceforensics@googlegroups.com

### DFDC

- Official dataset page: https://ai.meta.com/datasets/dfdc/
  (legacy: https://ai.facebook.com/datasets/dfdc/, archived at
  https://web.archive.org/web/20220102014605/https://ai.facebook.com/datasets/dfdc/)
- Full-dataset paper: https://arxiv.org/abs/2006.07397
- Preview-dataset paper: https://arxiv.org/abs/1910.08854
- Challenge / leaderboard: https://www.kaggle.com/c/deepfake-detection-challenge
- Results announcement: https://ai.meta.com/blog/deepfake-detection-challenge-results-an-open-initiative-to-advance-ai/
- Access portal: https://dfdc.ai/ (JavaScript application; content not retrievable
  read-only)

> **Note on a false lead.** A search of "DFDC" surfaced the repository
> `facebookresearch/dcd` carrying a Creative Commons Attribution-NonCommercial 4.0
> licence. That repository is **Dual Curriculum Design** — reinforcement-learning
> environment design — and is **not** the DFDC dataset. It is recorded here only to
> prevent it being cited as DFDC licence evidence.

---

## 2. Verified access procedures and dataset characteristics

### FaceForensics++ — verified

- **Access procedure.** Access is requested by submitting the official Google Form
  listed above. On approval, a download-script link is provided by email. Access is
  therefore **gated by a request/approval step**, not open download.
- **Licence identity.** The data is governed by the *FaceForensics Terms of Use*
  (the PDF above). The code in the repository is MIT-licensed. The Terms of Use
  document carries **no SPDX identifier**.
- **Terms (as stated in the Terms of Use).** Non-commercial research and
  educational use only; the researcher indemnifies the Technical University of
  Munich; material may be shared only with research associates who accept the same
  terms; TUM may terminate access; a for-profit employer is also bound by the terms.
- **Modality.** Video. The released material contains **no audio**; audio is not part
  of the release and any audio would have to be re-fetched from the source platform.
- **Labels.** Binary real/fake with the manipulation method identified.
- **Size.** Approximately: `c40` ~2 GB; `c23` ~10 GB; raw/`c0` ~500 GB. The split is
  720 train / 140 validation+test.
- **Benchmark.** The benchmark test data is for final reporting only, with a single
  submission.

### DFDC — verified

- **Access procedure.** Access requires an **AWS account** with an **IAM user and
  Access Keys**, and an AWS account number in order to sign up at
  https://dfdc.ai/. There is **no** public `registry.opendata.aws/dfdc` entry; the
  dataset is not distributed as an open S3 bucket. Access is gated by the sign-up.
- **Versions.** Two versions exist — a *Preview* dataset (about 5k videos, two facial
  modification algorithms) and the *Full* dataset (about 124k videos, eight facial
  modification algorithms).
- **Modality.** Video only, in ten-second clips. There is **no image dataset**;
  image samples would require extracting frames from the video clips.
- **Labels.** Binary real/fake, with metadata describing the clips (`dataset.json`).
- **Composition.** Training set approximately 119,154 clips across 486 subjects; a
  public test set of about 4,000 clips (50% fake); a private test set of about
  10,000 clips (50% fake), of which the "DFDC-like" portion is released as the final
  test set.
- **Consent.** The videos were recorded with paid actors who entered into
  agreements permitting use and manipulation of their likenesses.
- **Licence identity.** **Not established.** The official pages state only that the
  dataset is intended to have "few restrictions on usage"; the preview paper states
  that users must "request access and agree with the terms of service of the DFDC".
  The actual Terms of Service / licence text is presented inside the gated sign-up
  flow and could **not** be retrieved from any publicly available page during this
  audit.

---

## 3. Comparison

| Aspect | FaceForensics++ | DFDC |
| --- | --- | --- |
| Access path | Google Form → approval → emailed download link | AWS account + IAM user/keys → sign-up at dfdc.ai |
| Access gating | Request/approval required | Account + sign-up required |
| Licence transparency | Terms of Use PDF is **publicly retrievable**; no SPDX id | Licence text **not publicly retrievable**; only behind gated sign-up |
| Licence identity | "FaceForensics Terms of Use" (non-commercial research/education) | **Unestablished** |
| Modality | Video (no audio in release) | Video (10 s clips) |
| Labels | Binary real/fake + manipulation method | Binary real/fake + `dataset.json` metadata |
| Subject consent | Not formally documented in the same way (source material from video platform) | Paid actors with explicit likeness agreements |
| Preprocessing required | Frame extraction for image work | Frame extraction for image work (no image set exists) |
| Relative size | c40 ~2 GB / c23 ~10 GB / raw ~500 GB | Full ~124k videos (large) |
| Citation | Rössler et al., ICCV 2019 | `DFDC2019Preview` (preview) / `DFDC2020` (full) |

---

## 4. Unresolved questions

These remain **open** and must not be treated as settled:

1. **FaceForensics++ derived-frame redistribution rights.** The Terms of Use are
   silent on whether extracted frames are "adapted material" and on whether derived
   frames may be redistributed in an open-source project. The Terms cover the dataset
   itself; the derived-frame question is a human/legal judgement, not something this
   audit could resolve from the document.
2. **FaceForensics++ approval outcome.** Whether a given access request will be
   approved is unknown until the official form is submitted by a person.
3. **FaceForensics++ canonical licence identifier and version.** There is **no**
   SPDX identifier and **no single** official version string stated at the source;
   the release actually downloaded would need to be recorded explicitly.
4. **FaceForensics++ "non-commercial" scope.** Whether financial-communication
   research falls within "non-commercial research/education" is a human/legal
   interpretation, not a fact established by the document.
5. **DFDC terms accessibility.** The DFDC Terms of Service were **inaccessible**
   read-only; the licence identifier, its commercial/non-commercial scope, and any
   redistribution and derived-frame rights are therefore **unknown**.
6. **DFDC image samples.** No standalone binary image set was found; whether one is
   provided anywhere official is **unresolved** (the released material is
   video-only).

---

## 5. Licence terms vs. software/code licences

A deliberate distinction is kept throughout this audit:

- **Data licence** — governs the *dataset media*: who may access it, for what
  purpose (e.g. non-commercial research), and whether it or derived material may be
  shared. This is the governing instrument for using the dataset, and it is the
  instrument whose terms are **unresolved** for DFDC and only partly resolvable for
  FaceForensics++.
- **Software/code licence** — governs the *accompanying source code*. For
  FaceForensics++, the repository code is MIT-licensed. **A permissive code licence
  does not extend to the dataset media.** MIT on the code says nothing about whether
  the videos or frames may be used or redistributed.

Consequently, a permissive code licence (or the absence of one) is **not** evidence
about the dataset's own terms and must never be cited as such.

---

## 6. Decision

**Neither dataset is approved for acquisition.** Both remain candidates whose
licence status is unverified:

- **DFDC** — licence terms are inaccessible read-only, so its status cannot advance
  beyond unverified.
- **FaceForensics++** — its Terms of Use are publicly accessible, which makes its
  access conditions verifiable by a person, but the derived-frame and
  non-commercial-scope questions above remain open.

**FaceForensics++ is the preferred candidate for further access verification**,
solely because its terms are publicly accessible and can be reviewed and recorded;
DFDC's terms cannot currently be reviewed at all. This preference is about the
**traceability of the terms**, not a claim that FaceForensics++ may be used.

No registry entry or configuration change follows from this note. DFDC is not
registered, and FaceForensics++ remains as it was before the audit.

---

## 7. Required citations

Whichever dataset is eventually used, the corresponding work must be cited.

**FaceForensics++**

> Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*,
> ICCV 2019.

**DFDC — preview dataset**

```bibtex
@misc{DFDC2019Preview,
  title={The Deepfake Detection Challenge (DFDC) Preview Dataset},
  author={Brian Dolhansky, Russ Howes, Ben Pflaum, Nicole Baram, Cristian Canton Ferrer},
  year={2019},
  eprint={1910.08854},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

**DFDC — full dataset**

```bibtex
@misc{DFDC2020,
  title={The DeepFake Detection Challenge Dataset},
  author={Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jikuo Lu, Russ Howes, Menglin Wang, Cristian Canton Ferrer},
  year={2020},
  eprint={2006.07397},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

---

## 8. Next steps

The sequence below is the intended path. None of it has been started, and none of it
is performed by this note.

1. **Human review of the FaceForensics++ Terms of Use.** A person reads the Terms of
   Use PDF and resolves, in particular, the derived-frame redistribution question and
   the non-commercial-scope question.
2. **Official access request.** If the review concludes the terms are acceptable for
   this project's use, submit the official FaceForensics++ Google Form.
3. **Approval.** Await and obtain approval; capture the approved terms and the
   release/version information.
4. **Local acquisition.** Acquire the dataset locally by a person, under the reviewed
   terms, recording the explicit access date and dataset version.
5. **Reviewed configuration update.** Only after acquisition, and with a verified
   licence status, update `configs/datasets.yaml` in a reviewed change —
   accompanied by the explanatory comments that belong beside the entry.

Until step 5 is reached and reviewed, the dataset remains **not approved** and
**not usable**, consistent with the registry rule that a dataset not verified is not
usable.

---

## See also

- [docs/dataset_acquisition.md](dataset_acquisition.md) — how a verified dataset is
  recorded locally.
- [docs/dataset_registry.md](dataset_registry.md) — the registry the acquisition
  gates read.
- [docs/dataset_ingestion.md](dataset_ingestion.md) — the ingestion contract that
  follows acquisition.
