export type IndicatorKind =
  | "domain"
  | "url"
  | "file_hash"
  | "qr_payload"
  | "handle";

export interface ThreatIndicator {
  id: string;
  kind: IndicatorKind;
  value: string;
  confidence: number;
  firstSeen: string;
  source: string;
  tags: string[];
  /**
   * This build ships only illustrative indicators. No live lookup is performed
   * and no real feed is queried.
   */
  demo: true;
}

export interface CampaignCorrelation {
  id: string;
  name: string;
  narrative: string;
  confidence: number;
  indicatorIds: string[];
  sampleCount: number;
  firstSeen: string;
  demo: true;
}
