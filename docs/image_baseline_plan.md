# Classical Image Baseline — Plan and Design

This document describes the classical image baseline: the hand-crafted feature
transform, the linear classifier, the on-disk artifact, how it plugs into the
inference service, and — importantly — what it does **not** claim.

It is a design record, not a results report. No detector performance numbers appear
here, because none has been measured on real data yet.

## Purpose

The foundation stage ships no deep-learning stack. The classical baseline gives the
project a complete, runnable image pipeline — verified data in, a fitted model out,
honest metrics reported — before any neural encoder exists. It is a **mechanism
demonstrator**: it proves the plumbing (registry gates, split hygiene, feature
extraction, training, evaluation, artifact loading, inference) works end to end.

It is **not** a validated detector, and nothing derived from it should be presented
as evidence that an image is or is not manipulated.

## Non-goals

- No video, audio, or text analysis.
- No CRFDA/HREN/AEFCD or any neural architecture.
- No claim of detection performance, accuracy, or state-of-the-art parity.
- No dataset download, registration change, or licence decision.
- No automatic dataset approval. Training is *blocked* until a dataset is verified.

## Feature families

Two deterministic, dependency-light families are computed from an
`(H, W, 3)` RGB array in `[0, 1]` (the output of `ml.image.preprocess`). Both are
implemented in `ml/image/features.py`.

### 1. Laplacian residual statistics (8 features)

Luminance is computed with fixed Rec. 601 weights (`0.299, 0.587, 0.114`). The
residual is the edge-padded Laplacian `(4·center − neighbours) / 4`, so the output
has the same shape as the input and no pixel is invented. The eight statistics are:

| Feature | Meaning |
| --- | --- |
| `residual_mean` | Mean residual |
| `residual_std` | Standard deviation |
| `residual_mean_abs` | Mean absolute residual |
| `residual_skew` | Standardized third moment (manual, no SciPy) |
| `residual_excess_kurtosis` | Standardized fourth moment − 3 |
| `residual_p95_abs` | 95th percentile of absolute residual |
| `residual_p99_abs` | 99th percentile |
| `residual_high_fraction` | Fraction with absolute residual > `residual_threshold` |

### 2. DCT high-frequency energy (5 features)

Luminance is edge-padded to a whole number of square blocks. Each block is
transformed with an orthonormal DCT-II matrix built directly in NumPy
(`D[k,n] = α_k · cos(π(2n+1)k / 2N)`; no SciPy). Coefficients whose `(row + column)`
index sum reaches `dct_high_frequency_min_radius` (default: `block_size`) are
"high-frequency". The five features are:

| Feature | Meaning |
| --- | --- |
| `dct_high_energy_mean` | Mean high-frequency energy per block |
| `dct_high_energy_std` | Spread of high-frequency energy |
| `dct_high_energy_ratio` | High-frequency energy ÷ total energy |
| `dct_dc_energy_mean` | Mean DC (average) energy |
| `dct_block_energy_mean` | Mean total block energy |

### Configuration and determinism

`ImageFeatureConfig` fixes `residual_threshold=0.1`, `dct_block_size=8`,
`dct_high_frequency_min_radius=None` (resolved to the block size), and
`decimal_places=6`. Rounding to a fixed number of decimal places removes last-bit
platform variation in the cosine evaluation, so a feature vector is reproducible
byte-for-byte from a fresh checkout. `FEATURE_SCHEMA_VERSION` is bumped whenever the
meaning, order, or count of features changes.

The extractor (`ImageFeatureExtractor`) returns a
`FeatureRepresentation` tagged `HANDCRAFTED`. Inputs it cannot process — wrong
shape, wrong channel order, non-finite pixels — return an *unavailable*
representation with a reason; the extractor never fabricates a vector.

## Classifier

`ml/image/training.py` fits a scikit-learn `StandardScaler` (on the **training
split only**) followed by an L2-regularized `LogisticRegression` (`lbfgs`). The
model is deliberately the smallest that is reproducible, serializable as plain
numbers, and human-readable.

- `ClassifierConfig`: `c=1.0`, `max_iter=1000`, `solver="lbfgs"`,
  `positive_label="fake"`, `negative_label="real"`.
- Labels are encoded as integers; scores are oriented so a **higher score always
  means more likely to be the positive label**, regardless of scikit-learn's class
  ordering.
- `FittedClassifier` stores the scaler and coefficients as plain tuples and can
  score with NumPy alone (`to_dict` / `from_dict`). scikit-learn is required to
  *train* and to compute AUCs, but **not** to run the fitted detector.

### Threshold selection

`select_threshold` chooses the threshold maximizing balanced accuracy on a
validation slice, breaking ties toward the lowest score value for determinism.
Selection happens on validation only; the test split is reserved for final
reporting.

## Evaluation protocol

`evaluate_classifier` returns a `ClassifierEvaluation`. Metrics are built through
`experiments.results.scalar_metrics`, so exactly these seven keys are present:

`accuracy`, `precision`, `recall`, `f1`, `specificity`, `roc_auc`, `pr_auc`.

A metric that cannot be measured honestly is `None`, never `0.0`:

- **ROC-AUC and PR-AUC** are reported only when the evaluated slice contains **both
  classes** *and* the decision scores actually vary. Otherwise they are `None` with
  a note explaining why.
- **Precision** is `None` when no row is predicted positive.

Threshold selection happens on the **validation** split; the **test** split is used
only for the final reported metrics.

## On-disk artifact

The training script writes two files under
`<artifacts>/models/<model_id>/` (default artifacts root from
`configs/project.yaml`):

### `weights.json`

Plain, human-readable JSON — not pickle or joblib — so an artifact is diffable and
cannot execute code when loaded. Structure:

```
weights_schema_version  1
created_at              ISO-8601 UTC
feature_config          {residual_threshold, dct_block_size, ...}
feature_names           [...13 names...]
scaler                  {mean: [...], scale: [...]}
classifier              {coefficients: [...], intercept, labels, positive_class}
threshold               decision score threshold (or null)
training                {dataset_id, manifest, manifest_hash, split, split_hash,
                         seed, run_number, train/validation/test sample counts}
```

### `model.json`

The card read by `ml.inference.models`. For this baseline:

```
modality         "image"
framework        "numpy"           # scikit-learn is NOT needed to run the model
entrypoint       "ml.image.detectors.classical:build"
weights_file     "weights.json"
weights_sha256   <digest of weights.json>
input_size       [224, 224]
color_space      "RGB"
score_kind       "uncalibrated_score"
calibrated       false
labels           ["real", "fake"]
```

The loader (`load_model_backend`) verifies the weights file exists, matches the
recorded digest, that the declared framework imports (`numpy`), and that the
entrypoint resolves — and reports an honest `ModelUnavailableError` at the failing
step otherwise.

## Inference integration

`ml/image/detectors/classical.py` provides the `ModelBackend`. `predict()`:

1. extracts features with the config stored in the weights;
2. rejects a vector whose feature names do not match the trained model;
3. applies the scaler and coefficients with NumPy;
4. returns a `ModelOutput` with `calibrated=False` and `confidence=None`.

The score is an **uncalibrated decision score**, never a probability. The output's
`extra` records the feature schema, dimension, values, and source digest; its
`limitations` state plainly that it is a mechanism demonstrator.

## Data and gating

`scripts/train_image_baseline.py` refuses to run unless *all* of the following hold:

- the dataset is registered, its `license_status` is `verified`, its status is
  `approved` or `available_locally`, and `train_allowed` is true;
- the manifest's computed hash equals the split's `source_manifest_hash`;
- every id referenced by the split exists in the manifest;
- train/validation/test id sets are disjoint;
- no id appears in two splits;
- no `source` recording token appears in more than one split (identity leakage);
- every sample has a label, and each split contains both classes;
- an image decoder is available.

If any check fails the script prints the exact reason and exits with code `2`
without computing a feature or writing a file.

## Reproducibility

- Global seed is set via `ml.reproducibility.seeds.set_global_seed` (seeds Python
  `random` and NumPy only).
- The seed and run number are recorded in the weights' `training` block and in the
  experiment config/result when `--record` is used.
- Deterministic feature rounding plus `lbfgs` make repeated runs on identical data
  reproduce identical coefficients.
- Run artifacts (`config.json`, `metrics.json`, `environment.json`,
  `dataset_info.json`, `README.md`) are written under
  `experiments/runs/<experiment_id>/<run_id>/` when `--record` is passed.

## How to run

Mechanism tests (synthetic only, no real data needed):

```
.venv\Scripts\python.exe -m pytest tests\test_image_features.py tests\test_image_training.py tests\test_image_classical_detector.py -q
```

Real training (blocked until a dataset is verified and acquired):

```
.venv\Scripts\python.exe scripts\train_image_baseline.py ^
  --manifest path\to\image_manifest.jsonl ^
  --split path\to\frozen_split.json ^
  --dataset-id <verified-dataset-id> ^
  --experiment-id classical-image-baseline ^
  --seed 20260101 --run-number 0 ^
  --media-root path\to\media ^
  --model-id classical-image-baseline --model-version 0.1.0 ^
  --record --json
```

## Limitations

- Hand-crafted features only; no deep representation and no learned embedding.
- Uncalibrated scores; no probability or confidence is produced.
- Performance on real manipulated media is **unmeasured**. Any number produced
  before a verified dataset is acquired and a full split evaluated is a
  mechanism check, not a result.
- The linear model is a sanity floor, not a competitive baseline.
- Real-data training is deliberately impossible in the current state: no dataset
  has `license_status: verified`.

## Future work

- Acquire and verify a lawfully usable image dataset (see
  `docs/dataset_access_audit.md`).
- Run the real training command above and record the run.
- Evaluate calibration to justify any future probability output.
- Only then consider a learned encoder, measured against this baseline.
