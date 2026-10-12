import type { CampaignCorrelation, ThreatIndicator } from "../types";

/**
 * Illustrative indicators only. These are fabricated examples for the interface
 * and are NOT the result of any live lookup or threat-intelligence feed. The
 * repository performs no network collection by default.
 */
export const DEMO_INDICATORS: ThreatIndicator[] = [
  {
    id: "ti-1",
    kind: "domain",
    value: "secure-payout-portal.example",
    confidence: 0.82,
    firstSeen: "2026-09-28T10:00:00.000Z",
    source: "example registry (demo)",
    tags: ["phishing", "payment_redirection"],
    demo: true,
  },
  {
    id: "ti-2",
    kind: "url",
    value: "https://secure-payout-portal.example/login",
    confidence: 0.79,
    firstSeen: "2026-09-29T13:20:00.000Z",
    source: "example registry (demo)",
    tags: ["credential_theft"],
    demo: true,
  },
  {
    id: "ti-3",
    kind: "file_hash",
    value: "e3b0c44298fc1c149afbf4c8996fb924...",
    confidence: 0.66,
    firstSeen: "2026-10-01T09:05:00.000Z",
    source: "example registry (demo)",
    tags: ["sample_reuse"],
    demo: true,
  },
  {
    id: "ti-4",
    kind: "handle",
    value: "@quarterly.investor.updates",
    confidence: 0.54,
    firstSeen: "2026-10-02T18:40:00.000Z",
    source: "example registry (demo)",
    tags: ["impersonation"],
    demo: true,
  },
  {
    id: "ti-5",
    kind: "qr_payload",
    value: "https://pay.example/invoice/8842",
    confidence: 0.71,
    firstSeen: "2026-10-03T07:55:00.000Z",
    source: "example registry (demo)",
    tags: ["payment_redirection", "qr"],
    demo: true,
  },
];

export const DEMO_CAMPAIGNS: CampaignCorrelation[] = [
  {
    id: "camp-1",
    name: "Executive payment-redirection cluster (example)",
    narrative:
      "Illustrative grouping of domains and QR payloads sharing a payout theme. Not a verified campaign.",
    confidence: 0.63,
    indicatorIds: ["ti-1", "ti-2", "ti-5"],
    sampleCount: 3,
    firstSeen: "2026-09-28T10:00:00.000Z",
    demo: true,
  },
  {
    id: "camp-2",
    name: "Investor-impersonation handle (example)",
    narrative:
      "Illustrative single-handle cluster used to show correlation layout only.",
    confidence: 0.48,
    indicatorIds: ["ti-4"],
    sampleCount: 1,
    firstSeen: "2026-10-02T18:40:00.000Z",
    demo: true,
  },
];
