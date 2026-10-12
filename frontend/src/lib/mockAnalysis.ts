import type { Modality, RiskLevel } from "../types";

/**
 * DEMO MODE.
 *
 * This module produces a purely deterministic, pseudo-random placeholder result
 * so the analysis workflow can be demonstrated. It performs NO inference, loads
 * NO model, and represents NO experimental result. The same input always yields
 * the same output only so screenshots are reproducible. Every value it returns
 * is labelled `demo: true` and must be shown with the demo warning.
 */
export const DEMO_LABEL = "DEMO DATA — NOT REAL MODEL INFERENCE";

export interface MockSignal {
  name: string;
  /** 0..1 placeholder contribution, not a measured quantity. */
  value: number;
  note: string;
}

export interface MockAnalysisResult {
  demo: true;
  id: string;
  modality: Modality;
  subject: string;
  risk: RiskLevel;
  confidence: number;
  /** Placeholder signals; none is a model output. */
  signals: MockSignal[];
  startedAt: string;
  durationMs: number;
  summary: string;
}

const SIGNAL_POOL: Record<Modality, string[]> = {
  image: ["Compression consistency", "Facial boundary seams", "Noise residual", "EXIF continuity"],
  video: ["Temporal flicker", "Lip-sync drift", "Frame duplication", "Encoder continuity"],
  audio: ["Spectral flatness", "Formant stability", "Phase discontinuity", "Vocoder artifacts"],
  qr: ["Module geometry", "Quiet-zone integrity", "Payload entropy", "Re-encode traces"],
  url: ["Domain age signal", "Redirect chain", "Homoglyph risk", "Credential path pattern"],
};

const RISK_BY_BUCKET: RiskLevel[] = [
  "low",
  "low",
  "elevated",
  "elevated",
  "high",
  "low",
];

function fnv1a(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function unitFrom(seed: number, salt: number): number {
  const mixed = fnv1a(`${seed}:${salt}`);
  return (mixed % 100000) / 100000;
}

export function analyzeMock(
  modality: Modality,
  subject: string,
  now: Date = new Date(),
): MockAnalysisResult {
  const seed = fnv1a(`${modality}::${subject}`);
  const risk = RISK_BY_BUCKET[seed % RISK_BY_BUCKET.length];
  const confidence = 0.51 + unitFrom(seed, 1) * 0.44;
  const names = SIGNAL_POOL[modality];
  const signals: MockSignal[] = names.map((name, index) => ({
    name,
    value: unitFrom(seed, index + 7),
    note: "Placeholder signal — not a model output.",
  }));

  return {
    demo: true,
    id: `demo-${(seed % 0xffffffff).toString(16).padStart(8, "0")}`,
    modality,
    subject,
    risk,
    confidence: Number(confidence.toFixed(3)),
    signals,
    startedAt: now.toISOString(),
    durationMs: 900 + (seed % 1400),
    summary:
      "Simulated result generated locally for interface demonstration. " +
      "It is not the output of any trained detector and must not be cited as an experimental finding.",
  };
}
