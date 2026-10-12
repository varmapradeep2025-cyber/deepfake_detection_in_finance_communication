/**
 * Forensics vocabulary. The repository requires every claim to be tagged as
 * observed, derived, or inferred; nothing here is ever a bare assertion.
 */
export type ClaimKind = "observed" | "derived" | "inferred";

export type EvidenceType =
  | "hash"
  | "metadata"
  | "provenance"
  | "signal"
  | "external";

export interface EvidenceItem {
  id: string;
  type: EvidenceType;
  label: string;
  value: string;
  source: string;
  method: string;
  timestamp: string;
  /** 0..1; null when a value cannot be honestly produced. */
  confidence: number | null;
  claimKind: ClaimKind;
}

export interface MetadataField {
  key: string;
  value: string;
  /** True when a field is absent rather than empty. */
  missing?: boolean;
}

export interface CustodyEvent {
  id: string;
  timestamp: string;
  actor: string;
  action: string;
  location: string;
  /** Digest recorded at that step, or null when none was taken. */
  hash: string | null;
}

export interface SourceRecommendation {
  id: string;
  title: string;
  detail: string;
  priority: "high" | "medium" | "low";
}
