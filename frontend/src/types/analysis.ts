export type Modality = "image" | "video" | "audio" | "qr" | "url";

export type RiskLevel = "low" | "elevated" | "high" | "unknown";

/** Mirrors the finite-state workflow used by every analysis panel. */
export type AnalysisState =
  | "idle"
  | "ready"
  | "analyzing"
  | "completed"
  | "error";

export interface ModalityMeta {
  id: Modality;
  label: string;
  /** Accepted MIME types; empty for URL analysis. */
  accept: string[];
  /** Accepted extensions, shown to the user. */
  extensions: string[];
  /** Reasonable client-side limit in bytes. 0 means "no file input". */
  maxBytes: number;
  description: string;
}

export interface AnalysisRecord {
  id: string;
  filename: string;
  modality: Modality;
  risk: RiskLevel;
  /** Reported on a 0..1 scale. Always demo data in this build. */
  confidence: number | null;
  createdAt: string;
  status: "completed" | "failed";
  analyst: string;
  /** Every record in this build is simulated; see DEMO_DATA. */
  demo: true;
}

export interface ModalityBreakdownPoint {
  modality: Modality;
  label: string;
  count: number;
}

export interface RiskDistributionPoint {
  risk: RiskLevel;
  label: string;
  count: number;
  /** CSS color token, resolved in the chart component. */
  color: string;
}

export interface DashboardSummary {
  totalAnalyses: number;
  completedToday: number;
  highRisk: number;
  avgConfidence: number | null;
  modalityBreakdown: ModalityBreakdownPoint[];
  riskDistribution: RiskDistributionPoint[];
  recent: AnalysisRecord[];
}
