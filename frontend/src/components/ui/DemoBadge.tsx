import { FlaskConical } from "lucide-react";
import { DEMO_LABEL } from "../../lib/mockAnalysis";

interface DemoBadgeProps {
  /** Short label shown on screen; full disclaimer is the title/tooltip. */
  label?: string;
}

/**
 * Mandatory label for any simulated content. Use wherever demo data or mock
 * results are displayed so the provenance of a value is never ambiguous.
 */
export function DemoBadge({ label = "DEMO MODE" }: DemoBadgeProps) {
  return (
    <span className="badge badge--demo" title={DEMO_LABEL} role="note">
      <FlaskConical size={12} aria-hidden />
      {label}
    </span>
  );
}
