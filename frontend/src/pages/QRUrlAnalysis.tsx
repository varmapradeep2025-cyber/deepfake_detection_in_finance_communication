import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { MultimodalAnalysis } from "./MultimodalAnalysis";

export function QRUrlAnalysis() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Threat Intelligence"
        title="QR & URL Analysis"
        description="QR code decoding and URL phishing intelligence. No network access."
        actions={<Badge tone="neutral">Syntactic</Badge>}
      />
      <Card
        title="QR &amp; URL Forensics"
        subtitle="QR decoding and URL grading on text only - never auto-fetches URLs"
      >
        <MultimodalAnalysis />
      </Card>
    </div>
  );
}
