import type {
  AnalysisRecord,
  DashboardSummary,
  ModalityBreakdownPoint,
  RiskDistributionPoint,
} from "../types";

/**
 * Illustrative dashboard history. Every record is simulated; the UI shows the
 * DEMO MODE label wherever this data appears.
 */
export const DEMO_ANALYSES: AnalysisRecord[] = [
  {
    id: "an-2041",
    filename: "ceo_earnings_clip.mp4",
    modality: "video",
    risk: "high",
    confidence: 0.87,
    createdAt: "2026-10-10T08:41:00.000Z",
    status: "completed",
    analyst: "demo.analyst",
    demo: true,
  },
  {
    id: "an-2040",
    filename: "wire_transfer_instruction.wav",
    modality: "audio",
    risk: "elevated",
    confidence: 0.72,
    createdAt: "2026-10-10T08:12:00.000Z",
    status: "completed",
    analyst: "demo.analyst",
    demo: true,
  },
  {
    id: "an-2039",
    filename: "investor_deck_page.png",
    modality: "image",
    risk: "low",
    confidence: 0.63,
    createdAt: "2026-10-09T16:55:00.000Z",
    status: "completed",
    analyst: "j.doer",
    demo: true,
  },
  {
    id: "an-2038",
    filename: "https://secure-payout-portal.example",
    modality: "url",
    risk: "high",
    confidence: 0.91,
    createdAt: "2026-10-09T15:30:00.000Z",
    status: "completed",
    analyst: "j.doer",
    demo: true,
  },
  {
    id: "an-2037",
    filename: "invoice_qr.png",
    modality: "qr",
    risk: "elevated",
    confidence: 0.68,
    createdAt: "2026-10-09T11:02:00.000Z",
    status: "completed",
    analyst: "demo.analyst",
    demo: true,
  },
  {
    id: "an-2036",
    filename: "board_call_recording.mp3",
    modality: "audio",
    risk: "low",
    confidence: 0.58,
    createdAt: "2026-10-08T09:20:00.000Z",
    status: "completed",
    analyst: "j.doer",
    demo: true,
  },
  {
    id: "an-2035",
    filename: "press_release_headshot.jpg",
    modality: "image",
    risk: "elevated",
    confidence: 0.7,
    createdAt: "2026-10-08T08:15:00.000Z",
    status: "completed",
    analyst: "demo.analyst",
    demo: true,
  },
  {
    id: "an-2034",
    filename: "quarterly_update.mp4",
    modality: "video",
    risk: "low",
    confidence: 0.61,
    createdAt: "2026-10-07T17:44:00.000Z",
    status: "completed",
    analyst: "j.doer",
    demo: true,
  },
];

export const DEMO_MODALITY_BREAKDOWN: ModalityBreakdownPoint[] = [
  { modality: "video", label: "Video", count: 2 },
  { modality: "audio", label: "Audio", count: 2 },
  { modality: "image", label: "Image", count: 2 },
  { modality: "url", label: "URL", count: 1 },
  { modality: "qr", label: "QR", count: 1 },
];

export const DEMO_RISK_DISTRIBUTION: RiskDistributionPoint[] = [
  { risk: "high", label: "High", count: 2, color: "var(--risk-high)" },
  { risk: "elevated", label: "Elevated", count: 3, color: "var(--risk-elevated)" },
  { risk: "low", label: "Low", count: 3, color: "var(--risk-low)" },
  { risk: "unknown", label: "Unknown", count: 0, color: "var(--text-muted)" },
];

export const DEMO_DASHBOARD_SUMMARY: DashboardSummary = {
  totalAnalyses: DEMO_ANALYSES.length,
  completedToday: DEMO_ANALYSES.filter((a) =>
    a.createdAt.startsWith("2026-10-10"),
  ).length,
  highRisk: DEMO_ANALYSES.filter((a) => a.risk === "high").length,
  avgConfidence:
    DEMO_ANALYSES.reduce((sum, a) => sum + (a.confidence ?? 0), 0) /
    DEMO_ANALYSES.length,
  modalityBreakdown: DEMO_MODALITY_BREAKDOWN,
  riskDistribution: DEMO_RISK_DISTRIBUTION,
  recent: DEMO_ANALYSES,
};
