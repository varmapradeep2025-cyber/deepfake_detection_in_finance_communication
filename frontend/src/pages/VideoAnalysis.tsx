import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { MultimodalAnalysis } from "./MultimodalAnalysis";

export function VideoAnalysis() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Forensics"
        title="Video Analysis"
        description="Video deepfake detection workflow with temporal analysis."
        actions={<Badge tone="neutral">Service-backed</Badge>}
      />
      <Card
        title="Video Forensics Workspace"
        subtitle="Temporal consistency analysis with honest capability reporting"
      >
        <MultimodalAnalysis />
      </Card>
    </div>
  );
}
