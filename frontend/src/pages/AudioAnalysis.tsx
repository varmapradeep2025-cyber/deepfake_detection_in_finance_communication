import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { MultimodalAnalysis } from "./MultimodalAnalysis";

export function AudioAnalysis() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Forensics"
        title="Audio Analysis"
        description="Audio deepfake detection workflow with evidence tracking."
        actions={<Badge tone="neutral">Service-backed</Badge>}
      />
      <Card
        title="Audio Forensics Workspace"
        subtitle="Speech and recording forensics with honest capability reporting"
      >
        <MultimodalAnalysis />
      </Card>
    </div>
  );
}
