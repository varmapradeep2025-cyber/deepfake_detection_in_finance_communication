/**
 * Wire contract for the inference library's reports, records, and assembled
 * multimodal reports.
 *
 * Field names and string union values mirror the Python serializers exactly
 * (`ml.inference.report.AnalysisReport.to_dict`,
 * `ml.inference.records.AnalysisRecord.to_dict`,
 * `ml.inference.multimodal.MultimodalReport.to_dict`), so a future backend can
 * return these objects verbatim without a field-by-field translation. They are
 * deliberately snake_case here — this module describes the transport shape, not the
 * camelCase view models used elsewhere in the UI.
 *
 * No endpoint serves these types yet: `backend/` is empty. The client in
 * `api/modalities.ts` is the forward contract, and the page renders an explicit
 * offline state until a service exists.
 */
import type { RiskLevel } from "./analysis";
import type { ClaimKind, EvidenceType } from "./evidence";

/** Value set of `ml.inference.modality.AnalysisModality`. */
export type AnalysisModality = "image" | "video" | "audio" | "qr";

/** Value set of `ml.inference.status.AnalysisStatus`. */
export type AnalysisStatus =
  | "analysis_complete"
  | "model_unavailable"
  | "capability_unavailable"
  | "unsupported_modality"
  | "invalid_input"
  | "analysis_failed";

/** One tagged statement in a report (`ml.inference.evidence.EvidenceItem`). */
export interface ApiEvidenceItem {
  id: string;
  type: EvidenceType;
  label: string;
  value: string;
  source: string;
  method: string;
  timestamp: string;
  /** 0..1 when an honest confidence exists, else null. */
  confidence: number | null;
  claim_kind: ClaimKind;
}

/** `ml.risk.assess.RiskAssessment`. `level` is "unknown" when no score existed. */
export interface ApiRiskAssessment {
  level: RiskLevel;
  score: number | null;
  score_kind: string | null;
  calibrated: boolean;
  policy_name: string | null;
  elevated_threshold: number | null;
  high_threshold: number | null;
  rationale: string;
}

/** `ml.inference.models.ModelDescriptor` (a model card, not a live model). */
export interface ApiModelDescriptor {
  model_id: string;
  version: string;
  modality: string;
  framework: string;
  entrypoint: string | null;
  weights_file: string | null;
  weights_sha256: string | null;
  input_size: [number, number] | null;
  color_space: string;
  score_name: string;
  score_kind: string;
  score_semantics: string;
  decision_threshold: number | null;
  calibrated: boolean;
  labels: string[];
  description: string;
}

/** `ml.inference.models.ModelOutput`; `raw_score` is null when a model emitted none. */
export interface ApiModelOutput {
  model_id: string;
  version: string;
  score_name: string;
  raw_score: number | null;
  score_kind: string;
  score_semantics: string;
  calibrated: boolean;
  decision_threshold: number | null;
  confidence: number | null;
  labels: string[];
  extra: Record<string, unknown>;
  limitations: string[];
}

/** A decoded QR payload entry within a QR report's `result.codes`. */
export interface ApiQrCode {
  payload: string;
  kind: "url" | "text";
  url: ApiUrlAnalysis | null;
}

/** One heuristic URL finding (`ml.url.analyze.UrlFinding`). */
export interface ApiUrlFinding {
  code: string;
  severity: "info" | "low" | "medium" | "high";
  message: string;
  detail: string | null;
}

/** `ml.url.analyze.UrlAnalysis`. */
export interface ApiUrlAnalysis {
  url: string;
  scheme: string;
  hostname: string;
  port: number | null;
  path: string;
  host_suffix_hint: string;
  is_ip_literal: boolean;
  is_punycode: boolean;
  has_userinfo: boolean;
  max_severity: string | null;
  findings: ApiUrlFinding[];
}

/** Structured, non-model output. Currently the QR decode result. */
export interface ApiAnalysisResult {
  code_count?: number;
  url_count?: number;
  codes?: ApiCodeEntry[];
  [key: string]: unknown;
}

export interface ApiCodeEntry {
  payload: string;
  kind: "text" | "url";
  url: ApiUrlAnalysis | null;
}

/** `ml.inference.report.AnalysisReport`. */
export interface ApiAnalysisReport {
  request_id: string;
  status: AnalysisStatus;
  modality: string;
  path: string;
  model: ApiModelDescriptor | null;
  output: ApiModelOutput | null;
  risk: ApiRiskAssessment | null;
  evidence: ApiEvidenceItem[];
  warnings: string[];
  errors: string[];
  recommendation: string;
  started_at: string;
  completed_at: string;
  preprocessing: Record<string, unknown> | null;
  result: ApiAnalysisResult | null;
}

/** `ml.inference.records.AnalysisRecord` — the header plus the full report. */
export interface ApiAnalysisRecord {
  schema_version: number;
  report_id: string;
  created_at: string;
  modality: string;
  status: AnalysisStatus;
  source_name: string;
  report: ApiAnalysisReport;
}

/** One record's contribution to a multimodal report. */
export interface ApiReportSection {
  report_id: string;
  source_name: string;
  modality: string;
  status: AnalysisStatus;
  has_score: boolean;
  risk_level: RiskLevel | null;
  headline: string;
  evidence_count: number;
  warning_count: number;
  error_count: number;
  notable: string[];
}

/** `ml.inference.multimodal.MultimodalReport` — analyses side by side, never fused. */
export interface ApiMultimodalReport {
  report_id: string;
  title: string;
  generated_at: string;
  sections: ApiReportSection[];
  modality_counts: Record<string, number>;
  status_counts: Record<string, number>;
  scored_count: number;
  unscored_count: number;
  notable_findings: string[];
  disclaimer: string;
}

/** Whether a report carries a detector score (output present with a raw score). */
export function reportHasScore(report: ApiAnalysisReport): boolean {
  return report.output !== null && report.output.raw_score !== null;
}

/** Human label for an analysis status value. */
export const ANALYSIS_STATUS_LABELS: Record<AnalysisStatus, string> = {
  analysis_complete: "Analysis complete",
  model_unavailable: "Model unavailable",
  capability_unavailable: "Capability unavailable",
  unsupported_modality: "Unsupported modality",
  invalid_input: "Invalid input",
  analysis_failed: "Analysis failed",
};

/** Badge tone for an analysis status value. */
export const ANALYSIS_STATUS_TONES: Record<
  AnalysisStatus,
  "info" | "warning" | "high"
> = {
  analysis_complete: "info",
  model_unavailable: "warning",
  capability_unavailable: "warning",
  unsupported_modality: "warning",
  invalid_input: "high",
  analysis_failed: "high",
};
