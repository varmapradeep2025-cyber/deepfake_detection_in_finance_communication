import type {
  CustodyEvent,
  EvidenceItem,
  MetadataField,
  SourceRecommendation,
} from "../types";

export const DEMO_SUBJECT = {
  filename: "ceo_earnings_clip.mp4",
  sha256:
    "3b1f8c0a5d6e9b2c4a7f0e1d8c3b5a9e2d6f4c8b1a5e9d2c7f3b6a0d4e8c1f5b",
  capturedAt: "2026-10-10T08:41:00.000Z",
};

export const DEMO_METADATA: MetadataField[] = [
  { key: "container", value: "MP4 (ISO Base Media v1)" },
  { key: "duration", value: "00:00:42.5" },
  { key: "resolution", value: "1920 × 1080" },
  { key: "video_codec", value: "H.264 (Main@L4.0)" },
  { key: "audio_codec", value: "AAC LC, 48 kHz, stereo" },
  { key: "encoder", value: "unknown — field present but blank" },
  { key: "creation_time", value: "missing", missing: true },
  { key: "device_fingerprint", value: "missing", missing: true },
];

export const DEMO_EVIDENCE: EvidenceItem[] = [
  {
    id: "ev-1",
    type: "hash",
    label: "Content SHA-256",
    value: DEMO_SUBJECT.sha256,
    source: "forensic/hashing",
    method: "streamed SHA-256",
    timestamp: "2026-10-10T08:41:02.000Z",
    confidence: 1,
    claimKind: "observed",
  },
  {
    id: "ev-2",
    type: "metadata",
    label: "Encoder field blank",
    value: "encoder: present but empty",
    source: "forensic/metadata",
    method: "container inspection",
    timestamp: "2026-10-10T08:41:03.000Z",
    confidence: 0.6,
    claimKind: "observed",
  },
  {
    id: "ev-3",
    type: "provenance",
    label: "No creation timestamp",
    value: "creation_time absent from container",
    source: "forensic/provenance",
    method: "explicit absence check",
    timestamp: "2026-10-10T08:41:03.000Z",
    confidence: null,
    claimKind: "observed",
  },
  {
    id: "ev-4",
    type: "signal",
    label: "Temporal flicker placeholder",
    value: "0.74 (demo)",
    source: "frontend/demo",
    method: "mock signal — not inference",
    timestamp: "2026-10-10T08:41:04.000Z",
    confidence: 0.5,
    claimKind: "inferred",
  },
];

export const DEMO_CUSTODY: CustodyEvent[] = [
  {
    id: "co-1",
    timestamp: "2026-10-10T08:30:00.000Z",
    actor: "demo.analyst",
    action: "Evidence received from financial-comms mailbox",
    location: "Analyst workstation",
    hash: null,
  },
  {
    id: "co-2",
    timestamp: "2026-10-10T08:41:02.000Z",
    actor: "system",
    action: "SHA-256 digest recorded",
    location: "forensic/hashing",
    hash: DEMO_SUBJECT.sha256,
  },
  {
    id: "co-3",
    timestamp: "2026-10-10T08:41:05.000Z",
    actor: "system",
    action: "Analysis queued (demo)",
    location: "frontend/demo",
    hash: null,
  },
  {
    id: "co-4",
    timestamp: "2026-10-10T08:41:09.000Z",
    actor: "reviewer",
    action: "Case opened for human review",
    location: "Compliance desk",
    hash: null,
  },
];

export const DEMO_RECOMMENDATIONS: SourceRecommendation[] = [
  {
    id: "re-1",
    title: "Obtain the original distribution copy",
    detail:
      "Request the file directly from the sender's mail server to rule out interim re-encoding.",
    priority: "high",
  },
  {
    id: "re-2",
    title: "Corroborate with call metadata",
    detail:
      "Compare the claimed call schedule with PBX records; absence of a matching leg weakens authenticity.",
    priority: "medium",
  },
  {
    id: "re-3",
    title: "Preserve the chain of custody",
    detail:
      "Seal and hash the original before any further transformation so the digest remains valid.",
    priority: "medium",
  },
];
