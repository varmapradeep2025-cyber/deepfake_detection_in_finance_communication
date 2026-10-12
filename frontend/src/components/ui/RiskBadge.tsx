import type { RiskLevel } from "../../types";
import { Badge, type BadgeTone } from "./Badge";

const TONE: Record<RiskLevel, BadgeTone> = {
  low: "low",
  elevated: "elevated",
  high: "high",
  unknown: "neutral",
};

const LABEL: Record<RiskLevel, string> = {
  low: "Low",
  elevated: "Elevated",
  high: "High",
  unknown: "Unknown",
};

export function RiskBadge({ risk }: { risk: RiskLevel }) {
  return <Badge tone={TONE[risk]}>{LABEL[risk]} risk</Badge>;
}
