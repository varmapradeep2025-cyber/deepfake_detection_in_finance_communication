# Inference

How one media file becomes a tagged, uncertainty-aware analysis report. Implemented in
`ml/inference/` (modality-agnostic core), the per-modality packages `ml/image/`,
`ml/video/`, `ml/audio/`, and `ml/qr/` (with URL grading in `ml/url/`), and `ml/risk/`
(score-to-severity mapping). A file-based record store and a multimodal report builder
live in `ml/inference/records.py` and `ml/inference/multimodal.py`.

This stage sits after media preprocessing and before any backend or UI. It is the point
at which a model — if one exists — turns pixels into a number, and the point at which
this project refuses to turn a number into more certainty than it carries.

Four modalities are routed: **image**, **video**, **audio**, and **qr**. A QR analysis
decodes the payload and grades any URL it contains, so URL/phishing findings surface
through the `qr` modality rather than as a separate route.

---

## The one thing to get right

**A missing model is a reported result, not an error, and a model's score is not a
probability.** The two failures this stage exists to prevent are twin mistakes:

- Inventing a score, a confidence, or a severity when no model ran.
- Presenting a raw, uncalibrated score as though it were the chance that a financial
  communication is fraudulent.

So every request returns an :class:`AnalysisReport`, and its
:class:`AnalysisStatus` — not the presence or absence of fields — says what happened:

| Status | Meaning | Score present? |
| --- | --- | --- |
| `analysis_complete` | The workflow ran to completion. A model score is present **only** when a model ran (a QR decode completes with no score). | maybe |
| `model_unavailable` | Input was fine; no usable model exists. | no |
| `capability_unavailable` | An analyzer exists but a local capability it needs is missing (e.g. no audio decoder, no QR decoder). | no |
| `invalid_input` | The file was missing, empty, over-limit, or undecodable. | no |
| `unsupported_modality` | Recognised modality with no analyzer yet. | no |
| `analysis_failed` | The backend opened the artifact and then raised mid-step (e.g. the QR decoder failed). | no |

`AnalysisReport.is_success` means the workflow completed (`analysis_complete`); it does
**not** mean a score exists. Use `AnalysisReport.has_score` (output present with a
non-null `raw_score`) to ask that separate question. In a checkout with no weights, the
correct and verified answer to an image, video, or audio request is `model_unavailable`;
for audio it is `capability_unavailable` when the file is not WAV, because only a stdlib
WAV reader is installed.

---

## What this stage does and does not do

**Does**

- Validate the input and record its observed identity: SHA-256, size, dimensions,
  channel layout.
- Preprocess the pixels into a fixed-size `float32` RGB array by a deterministic
  transform, and record a digest of the exact array a model would see.
- Discover model cards, select one for the modality, and load it — verifying weights by
  digest and explaining every failure.
- Run the model, record its raw output with its semantics, and attach known limitations.
- Map the score to a risk band through an explicit policy, or `unknown` when there is no
  score.
- Return a report whose every statement is tagged observed, derived, or inferred.

**Does not**

- Define, train, or ship any model or weights.
- Guess a colour order, an input size, or a normalisation the model did not declare.
- Turn an uncalibrated score into a probability or a confidence.
- Produce a risk level without a score.

---

## The report

`AnalysisReport` (in `ml/inference/report.py`) is the unit of output:

| Field | Contents |
| --- | --- |
| `request_id` | Identifier echoed through the request. |
| `status` | An `AnalysisStatus`; the authority on whether a score exists. |
| `modality` | Media type analyzed. |
| `path` | The artifact analyzed. |
| `model` | The `ModelDescriptor` used, or `null`. |
| `output` | The `ModelOutput`, or `null`. Present only on `analysis_complete`. |
| `risk` | The `RiskAssessment`, or `null`. |
| `evidence` | Every `EvidenceItem`, each tagged observed / derived / inferred. |
| `warnings` | Non-fatal observations (uncalibrated score, dropped alpha channel, ...). |
| `errors` | Why the request could not complete. Empty on success. |
| `recommendation` | The standing advice to verify independently. |
| `started_at`, `completed_at` | RFC 3339 UTC timestamps. |
| `preprocessing` | Serialized preprocessing summary, or `null`. |
| `result` | Structured **non-model** output, or `null`: the decoded QR payloads and graded URLs, or the probed video stream properties. |

Two derived properties answer two different questions:

| Property | True when |
| --- | --- |
| `is_success` | `status is analysis_complete` — the workflow finished. |
| `has_score` | `output is not None and output.raw_score is not None` — a detector produced a number. |

A QR analysis is the modality where these genuinely differ: it is a success with no score.

---

## Evidence: every claim is tagged

An :class:`EvidenceItem` mirrors the frontend's `EvidenceItem` contract and carries a
:class:`ClaimKind`:

| Claim kind | Meaning | Reproducible by |
| --- | --- | --- |
| `observed` | Read directly from the artifact. | Re-reading the file. |
| `derived` | Computed from observed data by a deterministic transform. | Re-running the transform. |
| `inferred` | Produced by a model or heuristic. | Re-running the model. |

An `inferred` item carries `confidence` only when an honest number exists; otherwise it
is `null`, so an uncalibrated score cannot masquerade as a confidence.

Field-name correspondence with `frontend/src/types/evidence.ts` (serialized form is
snake_case, per the rest of the repository):

| Frontend (`camelCase`) | Report (`snake_case`) |
| --- | --- |
| `claimKind` | `claim_kind` |
| `type` | `type` |
| `confidence` | `confidence` |

---

## Models are artifacts, not imports

A model is a directory with a card:

```
<artifacts>/models/<model_id>/
    model.json        # required
    <weights file>    # optional; named by "weights_file"
```

`<artifacts>` is `paths.generated.artifacts` from `configs/project.yaml`. A minimal card:

```json
{
  "model_id": "example-image-detector",
  "version": "1.0.0",
  "modality": "image",
  "framework": "torch",
  "entrypoint": "ml.image.detectors.example:build",
  "weights_file": "model.safetensors",
  "weights_sha256": "<64 lowercase hex>",
  "input_size": [224, 224],
  "color_space": "RGB",
  "score_name": "manipulation_score",
  "score_kind": "uncalibrated_score",
  "score_semantics": "Higher means more likely manipulated. Not a calibrated probability.",
  "decision_threshold": 0.5,
  "calibrated": false
}
```

`ModelRegistry.discover(models_dir)` scans `*/model.json`, skips a malformed card into
`problems` rather than aborting, and sorts by `model_id`. `load_model_backend` then
checks, in order:

1. the weights file exists — else `weights_missing`;
2. its SHA-256 matches — else `weights_digest_mismatch`;
3. the declared framework imports — else `dependency_missing`;
4. the `"module:factory"` entry point resolves — else `entrypoint_malformed`,
   `entrypoint_missing`, or `entrypoint_attribute_missing`;
5. the factory returns a `ModelBackend` — else `load_failed`.

The factory is called as `factory(descriptor=..., model_dir=..., weights_path=...)`.
Weights are hashed with `datasets.manifest.calculate_file_sha256`, the project's single
file-identity implementation.

```python
from ml.inference.models import ModelRegistry
from ml.inference.models import load_model_backend

registry = ModelRegistry.discover("artifacts/models")
for descriptor in registry.for_modality("image"):
    backend = load_model_backend(descriptor, models_dir=registry.models_dir)
    print(descriptor.qualified_id, "ready")
```

---

## Preprocessing is model-agnostic and declared

`ImagePreprocessor` decodes, converts to RGB, resizes by nearest-neighbour, and scales to
`float32` in `[0, 1]`. Decoding is injected (`ImagePixelDecoder`), so the transform is
testable with no codec; `OpenCVImagePixelDecoder` imports OpenCV lazily. The decoder
reports its native ordering (`"BGR"`, `"L"`, ...), and the preprocessor converts from that
ordering explicitly — it never assumes RGB.

This is a **default**, not a detector's training contract. A real model's normalisation,
interpolation, and cropping belong to that model and are its backend's responsibility;
the preprocessed array is handed over together with the summary of how it was produced.
The transform is deterministic, so its `digest_sha256` is a stable fingerprint of the
input a model would receive.

---

## Risk is a policy, not a probability

`assess_risk(score, score_kind=..., calibrated=..., policy=...)` returns a
`RiskAssessment`:

- No score → `RiskLevel.UNKNOWN`, with a reason. Never `LOW`, which would be an
  assertion the data does not support.
- A score → a band (`LOW` / `ELEVATED` / `HIGH`) from an explicit, named `RiskPolicy`.

Every assessment records `score_kind` and `calibrated`, so a band derived from an
uncalibrated score is never read as a likelihood. Raw score, probability, confidence, and
risk are four different things here, and the report keeps them apart.

```python
from ml.risk.assess import assess_risk

assess_risk(None).level                      # RiskLevel.UNKNOWN
assess_risk(0.93, score_kind="uncalibrated_score").level   # RiskLevel.HIGH, calibrated=False
assess_risk(0.42, score_kind="probability", calibrated=True).level  # RiskLevel.LOW
```

---

## Usage

```python
from ml.inference.request import AnalysisRequest
from ml.inference.service import MediaAnalysisService

service = MediaAnalysisService.from_project_config()
report = service.analyze(AnalysisRequest(path="suspect.png", modality="image"))

if report.status.value == "model_unavailable":
    print("\n".join(report.errors))            # why nothing could be scored
    for item in report.evidence:               # observed + derived still recorded
        print(item.claim_kind.value, item.label, item.value)
else:
    print(report.output.raw_score, report.risk.level.value)
```

`MediaAnalysisService.analyze` never raises for a problem with the artifact; it raises
`ValueError` only for a modality string the schema does not define. `from_project_config`
builds every analyzer whose local capability is present, so a running service routes all
four modalities without further wiring.

---

## Modality modules

| Modality | Package | What it does | Typical status with no weights |
| --- | --- | --- | --- |
| `image` | `ml/image/` | Preprocess + detector signal. | `model_unavailable` |
| `video` | `ml/video/` | Probe stream properties (codec, frames, fps, resolution) as evidence. | `capability_unavailable` unless a video model exists |
| `audio` | `ml/audio/` | Probe recording properties (rate, channels, frames) via the stdlib WAV reader. | `capability_unavailable` for MP3/FLAC (no decoder installed) |
| `qr` | `ml/qr/` | Decode QR payloads; grade each URL via `ml/url/`. Completes with no model. | `analysis_complete` (or `capability_unavailable` with no OpenCV decoder) |

`ml/url/analyze.analyze_url` is syntactic only: it performs **no network access** and its
findings (IP-literal host, punycode, embedded userinfo, suspicious TLD, phishing keywords,
and more) are inferred, not proof.

---

## The record store

`ml/inference/records.py` persists one analysis per JSON file under
`<root>/records/<report_id>.json`, written atomically (temporary file + rename).

- `AnalysisRecord` holds an indexed header (`report_id`, `created_at`, `modality`,
  `status`, `source_name`) plus the **full** serialized report — the builder and any audit
  need the evidence and warnings, not a summary.
- Raw media is never copied in: a record keeps the artifact's path, digest, and properties,
  so storing a large video does not duplicate it.
- `AnalysisStore.save(overwrite=False)` raises `FileExistsError` rather than clobbering;
  `_safe_stem` rejects identifiers that could escape the directory. The envelope carries
  `schema_version` (`SCHEMA_VERSION = 1`).

## The multimodal report builder — never a fused score

`ml/inference/multimodal.py` assembles a `MultimodalReport` from a chosen sequence of
records. It deliberately produces **no** combined score and no fused risk level: an image
detector's manipulation score, a video probe, an audio probe, and a QR decode measure
different things on different scales, and averaging or summing them would manufacture a
number no component supports. Instead it records, per record:

- `modality`, `status`, `has_score`, `risk_level`, a one-line `headline`, and the counts of
  evidence items, warnings, and errors;
- `notable` per-record observations (an elevated/high risk, a high-severity URL finding, an
  error) collected into a clearly labelled top-level list — described as observations, not
  a verdict.

`MultimodalReportBuilder.build` raises `ValueError` on an empty selection. The
`DEFAULT_MULTIMODAL_DISCLAIMER` states plainly that scores are not fused.

---

## Frontend contract

The dashboard's modality page (`frontend/src/pages/ModalityAnalysis.tsx`) targets the
typed client in `frontend/src/api/modalities.ts`; its wire types live in
`frontend/src/types/modalities.ts` and mirror the Python `to_dict` output (snake_case).

A service implements the read and analyze endpoints in `backend/` using only the standard
library (`http.server`). No framework is used because this checkout ships none — the
project keeps FastAPI/Uvicorn commented out in `requirements.txt` on purpose — and
`configs/project.yaml` (`services.api.enabled: false`) keeps the API disabled by default,
so the server is an opt-in local process, not an enabled foundation module.

| Method and path | Body | Returns |
| --- | --- | --- |
| `POST /analyses` | multipart: `modality`, `file` | `AnalysisRecord` |
| `GET /analyses` | — | `AnalysisRecord[]` |
| `GET /analyses/{report_id}` | — | `AnalysisRecord` |
| `GET /health` | — | `{ "status", "service", "version" }` |
| `OPTIONS` | — | CORS preflight (`204`) |

`MediaAnalysisService` reads an artifact by **path**, so the backend writes the uploaded
bytes to a per-request temporary file and hands the library that path; the stored record's
`path` is then the sanitized client file name, not the deleted temporary path.

Run it locally:

```powershell
.\.venv\Scripts\python.exe -m backend --host 127.0.0.1 --port 8000
```

The application wires only the `image` analyzer (the modality with a real, exercised
workflow). In a checkout with no weights, an image upload is answered with the library's
`model_unavailable` report rather than a fabricated score, and any other modality is
answered with `unsupported_modality`. `POST /reports/multimodal` remains unimplemented.

---

## Known limitations

- **No detector ships.** The verified output against real media is `model_unavailable` for
  image and video, and `capability_unavailable` for non-WAV audio.
- **QR completes without a score.** Decoding is the analysis; `has_score` is `False` even
  though `status` is `analysis_complete`.
- **Video and audio are probes, not detectors.** They record stream/recording properties
  and honest capability gaps; they do not score manipulation.
- **Preprocessing is a default, not a model contract.** A detector needing different
  normalisation or interpolation must apply it in its backend.
- **Nearest-neighbour resize.** Chosen for exact reproducibility; a model that needs area
  or bilinear interpolation must do it itself.
- **Risk thresholds are a place-holder operating point.** The default `RiskPolicy` is
  conservative and must be replaced for a deployment's base rates and cost asymmetry.
- **No calibration or OOD detection.** `ml/risk/` implements severity mapping only;
  calibration, abstention, and distribution-shift detection remain future work.
- **No fused score.** By design: `multimodal.py` never combines per-modality scores.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest `
    tests/test_inference_models.py tests/test_inference_image.py `
    tests/test_inference_video.py tests/test_inference_audio.py `
    tests/test_inference_qr.py tests/test_inference_url.py `
    tests/test_inference_store.py tests/test_inference_multimodal.py `
    tests/test_inference_risk.py -q
```

The model, risk, and most analyzer tests use fakes and need no codec. Integration tests
decode a real PNG/QR with OpenCV and skip when OpenCV is absent; the audio tests use a
written WAV and report the missing MP3 decoder as a capability gap rather than a failure.
