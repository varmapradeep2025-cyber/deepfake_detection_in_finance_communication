# Media preprocessing

How a validated video manifest becomes frames, and how those frames become a split that
cannot leak. Implemented in `datasets/preprocessing/`.

This stage sits between `datasets/ingest.py` (which records what a dataset *is*) and any
model code (which should only ever see a frozen split).

---

## The one thing to get right

There are two identities, and conflating them breaks the project in a way that will not
announce itself.

| Identity | Derived from | Used for |
| --- | --- | --- |
| `source_video_id` | the full relative path of one file | frame ids, output directories |
| `source_recording_id` | the path, minus its manipulation directory | grouping, splitting |

FaceForensics++ stores `<manipulation>/<recording>/clip.mp4`: one genuine clip and four
forgeries, all from the same recording. Two consequences:

- Frame ids must be unique **per file**. Deriving them from the recording identity would
  give `original/000/clip.mp4` and `DeepFakes/000/clip.mp4` the same `frame_000000.png`,
  and they would overwrite each other on disk.
- Frames must be grouped and split **per recording**. A per-frame split puts the genuine
  clip in train and its forgery in test, and the model learns the split rather than the
  manipulation. That is the failure this stage exists to prevent.

So `SourceIdentityLayout.DROP_FIRST_SEGMENT` drops the first path segment for the
*recording* identity only. With `FULL_RELATIVE_PATH` the FF++ corpus produces 5× as many
groups as it has recordings, and a clip can be separated from its forgeries. Both tests
in `tests/test_preprocessing_pipeline.py::TestLeakageEndToEnd` pin this.

---

## What this stage does and does not do

**Does**

- Inspect every media file: size, SHA-256, and - per modality - codec, dimensions,
  channel layout, frame rate, sample rate, and duration. Unreadable properties are
  recorded as unavailable, never as plausible defaults.
- Count video frames by decoding, so the declared count is never mistaken for the real one.
- Extract a deterministic, seek-free sample of frames.
- Reject unusable frames and record why, with the statistics that justified it.
- Group frames by recording, split by group, and verify no group spans two splits.
- Hash every frame it wrote, so the manifest's digest is a claim about bytes.
- Detect exact duplicates — identical frame digests — across the kept frames, and merge
  recordings linked by that content *before* splitting, so duplicated content can never
  land in two splits.

**Does not**

- Modify, move, or rename source media. The raw directory is read-only.
- Infer a label. Labels come from the adapter and are carried through untouched.
- Write a frame manifest for frames it did not write.
- Fill gaps. Anything it cannot resolve is reported.
- Delete duplicates, or quietly drop a duplicated recording from the split.
- Detect *near*-duplicates (perceptually similar but byte-different frames). Only
  exact, digest-identical duplicates make recordings merge; near-duplicate detection is
  future work (see [Known limitations](#known-limitations)).

---

## Artifacts

Written under an output root you choose:

| File | Contents | Written in a dry run? |
| --- | --- | --- |
| `frames.jsonl` | Frame manifest, through the existing `write_manifest` | **No** |
| `inspection.json` | Every video inspected, including the ones that failed and why | Yes |
| `rejected_frames.json` | Every rejected frame, its reasons and statistics | Yes |
| `groups.json` | Recording groups, plus any video-identity reuse | Yes |
| `duplicates.json` | Exact-duplicate findings over the kept frames: every digest shared by more than one frame, plus frames whose digest could not be verified. Written even in a dry run, where the missing digests are reported rather than implied unique | Yes |
| `split.json` | Frozen split, via `write_split_definition` | **No** |
| `preprocessing_report.json` | Summary, config, quality totals, leakage report | Yes |

A dry run writes no images. It therefore writes no frame manifest and no split: both
assert facts about image bytes that do not exist, and a manifest with invented digests
would pass verification without having checked anything. Reports describe the plan
honestly, and the run says so in its warnings.

Rejected frames are kept out of `frames.jsonl` on purpose. The manifest is the definition
of the training data; a rejected frame is not part of it, and `rejected_frames.json` keeps
the audit trail with the statistics needed to re-check the decision without re-decoding.

---

## Determinism

- Frames are sampled uniformly and endpoint-inclusive, from the **observed** frame count.
- Selection uses sequential decoding, never `CAP_PROP_POS_FRAMES`, which would make a
  frame index a property of the decoder's seek heuristic rather than of the file.
- Group ordering uses SHA-256, not Python's salted `hash()`, so a split is identical
  across processes and machines.
- Reports are written with sorted keys and a fixed newline, so re-running over unchanged
  data produces byte-identical files.
- PNG is the default. JPEG at any quality would inject compression artifacts into the
  pixels a detector is supposed to be judging.

---

## Quality filtering

The default policy is deliberately conservative: it rejects frames below 32×32 and frames
where every pixel is identical, and nothing else. Rejecting on brightness or contrast by
default would discard legitimate low-light footage and quietly shrink the corpus. Those
checks are opt-in:

```python
from datasets.preprocessing.quality import QualityPolicy

policy = QualityPolicy(min_mean_intensity=20.0, max_mean_intensity=235.0)
```

A `None` threshold means *disabled*, not *zero*. The distinction matters: treating `0` as
"off" would reject every frame.

---

## Exact-duplicate detection

The split must not let the same content appear in train and in test. Id-level checks alone
cannot catch that: the genuine clip of recording 0 and the DeepFakes clip of recording 0
already share a group, but two *recordings* could each contain frames copied from the same
source, and no id-derived check would know.

So after extraction, and while the frames are still in memory, the pipeline buckets the
kept frames by their `frame_sha256` digest:

- **Within-source repeats** — the same frame appearing twice inside one video — are found
  and reported, but change nothing: they already live in one group.
- **Cross-source duplicates within one recording** (a genuine clip and its forgery holding
  identical frames) are found and reported, and again change nothing: the recording is
  already one group.
- **Cross-recording duplicates** are the risk. Recordings A and B holding identical frames
  become one *coalesced* group before the group-aware splitter runs, so the duplicated
  content is assigned to exactly one split. If a hand-edited split later splits that
  merged group across train and test, the existing leakage check (now applied to the
  merged group) fails loudly.

Detection is a pure function of the manifest — it never opens media. Results are stable
under any input ordering: frames are bucketed by digest, and findings, connected
components, and merged groups are all emitted sorted.

Because the whole corpus is hashed, an identical digest is a strong claim: `find_exact_duplicates`
will not report a digest unless two kept frames share it, which requires two frames to have
written byte-for-byte identical pixels.

Implemented in `datasets/preprocessing/duplicates.py`:

```python
from datasets.preprocessing import (
    DuplicateReport,
    coalesce_duplicate_linked_groups,
    find_exact_duplicates,
)

report: DuplicateReport = find_exact_duplicates(result.frames)
merged = coalesce_duplicate_linked_groups(result.groups, report)
print(report.cross_recording)  # the leak vectors, with every frame id named
```

Frames whose digest is unusable — missing (dry runs, `frame_sha256=None`) or
malformed (not 64 lower-case hex digits) — are listed in `duplicates.json` under
`unverifiable`, with the reason. They are never treated as *unique*; a split built from
unverifiable digests is reported rather than blessed.

### FaceForensics++ behaviour

The default `DROP_FIRST_SEGMENT` layout puts the original clip and its four forgeries in
one recording group already. Duplicate detection therefore merges recordings *only when
content is shared between different recordings*; an FF++ corpus where each recording's
clips are self-consistent is passed through with groups untouched, and the existing
leakage checks pin that.

### Guarantee

Two recordings considered duplicate-linked (at least one shared digest) can never appear
in different splits. Transitive links are respected: if A duplicates B and B duplicates C,
all three form one component and move as a unit.

---

## Usage

```python
from pathlib import Path

from datasets.manifest import load_manifest
from datasets.preprocessing.extraction import (
    ExtractionConfig,
    OpenCVFrameWriter,
    VideoFrameExtractor,
)
from datasets.preprocessing.inspection import MediaInspector, default_decoder
from datasets.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
)
from datasets.preprocessing.frames import SourceIdentityLayout
from datasets.preprocessing.splitting import GroupSplitConfig

decoder = default_decoder()
extractor = VideoFrameExtractor(
    decoder,
    OpenCVFrameWriter("png"),
    ExtractionConfig(frames_per_video=16),
)

pipeline = PreprocessingPipeline(
    MediaInspector(decoder),
    extractor,
    PreprocessingConfig(
        extraction=extractor.config,
        split=GroupSplitConfig(train_fraction=0.7, validation_fraction=0.2),
        source_identity_layout=SourceIdentityLayout.DROP_FIRST_SEGMENT,
        count_frames=True,
    ),
)

samples = load_manifest("datasets/manifests/ffpp.jsonl")
result = pipeline.run(samples, base_dir="datasets/raw/ffpp", output_root="datasets/processed/ffpp")

print(result.summary())
assert result.is_leak_free
```

Pass `dry_run=True` in the config to preview: selection, quality verdicts, grouping, and
the split proportions are all computed, and nothing is persisted.

---

## The decoder is injectable

`VideoDecoder` is a two-method protocol (`probe`, `read_frames`). Everything above it —
inspection, sampling, quality, grouping, splitting — depends on the protocol, not on
OpenCV. Tests drive the entire pipeline with `tests/synthetic_preprocessing.py`'s
`FakeVideoDecoder` and no image files at all.

Swapping backends means implementing the protocol and changing one construction site.
`opencv-python-headless` is pinned in `requirements.txt` because it is frame-accurate and
seek-free, and because its headless build ships no Qt, which avoids the well-known Windows
conflict where two applications load different Qt versions in one process.

`opencv-python` (the GUI build) is deliberately *not* used. This is a batch job and never
displays a frame.

---

## Inspecting images and audio

Inspection covers all three modalities behind one record type, `MediaInspection`. A video
probe reads container, codec, dimensions, frame rate and declared frame count; an image
probe reads pixel dimensions and channel layout; an audio probe reads sample rate, channel
count and duration.

```python
from datasets.preprocessing.inspection import MediaInspector

inspector = MediaInspector(
    default_decoder(),                       # video (OpenCV)
    image_decoder=default_image_decoder(),   # images (OpenCV)
    audio_decoder=default_audio_decoder(),   # audio (stdlib `wave`)
)

row = inspector.inspect("clips/clip.mp4")          # explicit modality
row = inspector.inspect_image("faces/frame.png")
row = inspector.inspect_audio("calls/call.wav")
row = inspector.inspect_auto("clip.jpg")           # choose by extension
```

Simpler still, `MediaInspector.from_project_config()` wires all three bundled decoders and
reads the hash chunk size from `configs/project.yaml`:

```python
inspector = MediaInspector.from_project_config()
```

Rules that apply across modalities:

- **Fields are never fabricated.** `width=None` plus a status says "could not read", and
  the exact field names that could not be read are listed in `unavailable_fields`.
- **Unsupported and empty files are distinguished.** A zero-byte file is `EMPTY`; a text
  file renamed to `.png` is `DECODE_FAILED` with the decoder's message; a `.txt` passed to
  `inspect_auto` is `UNSUPPORTED` after its size and hash are still recorded.
- **Channels are reported as the decoder sees them.** OpenCV reads colour images as
  `BGR`/`BGRA` and reports them as such, because a consumer that assumed RGB would corrupt
  every downstream pixel.
- **The default audio decoder is `wave`, which handles WAV only.** MP3, FLAC or OGG fed to
  it is a `DECODE_FAILED`, not a guess. Other containers need an injected
  `AudioDecoder` (e.g. one backed by `soundfile`); nothing else changes.

---

## Known limitations

- **OpenCV reports no container.** Only the stream FourCC, which is recorded under `codec`.
  `container_format` comes from the file extension, which is the extension and not a
  verified fact.
- **Duration is frame count over rate.** For variable-frame-rate files this is an
  approximation. `CAP_PROP_POS_MSEC` would be worse: it reports a seek position, not a
  trustworthy total.
- **`FPS_BASED` and `TIMESTAMP_BASED` sampling are not implemented.** They fail loudly
  rather than falling back to uniform.
- **Frame hashing costs a second pass.** Every written frame is read back to be digested.
  That is the price of a manifest that can still detect a corrupted frame later.
- **A recording group may contain both real and fake frames.** That is correct — it is what
  a recording *is* — and the split moves whole recordings, so the class ratio stays stable.
- **Near-duplicates are not merged into the split.** Detection is exact (identical SHA-256
  digests) only. Perceptual near-duplicates — copies that were re-encoded, resized, or
  overlaid with text — produce different digests and are *not* coalesced, and finding them
  is future work. Exact findings stay in their own `duplicates.json` report so a future
  near-duplicate pass can add a separate report without conflating the two.
- **FaceForensics++ is not licensed yet** (`license_status: pending_verification`). Only
  synthetic media has been through this stage.

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_preprocessing_duplicates.py `
    tests/test_preprocessing_inspection.py `
    tests/test_preprocessing_media_inspection.py `
    tests/test_preprocessing_sampling.py `
    tests/test_preprocessing_splitting.py `
    tests/test_preprocessing_pipeline.py -q
```

The two `test_preprocessing_*inspection*.py` files are also the only places a real encoder
is involved, and those tests skip when OpenCV is absent (the audio ones need only the
standard library). The rest use the fake decoders, so they need no media files.
