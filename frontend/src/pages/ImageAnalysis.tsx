import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { ModalityCard } from "../components/modality/ModalityCard";

export function ImageAnalysis() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Forensics"
        title="Image Analysis"
        description="Image deepfake detection workflow with evidence and provenance."
        actions={<Badge tone="neutral">Service-backed</Badge>}
      />
      <Card
        title="Image Forensics Workspace"
        subtitle="Forensic analysis of still images with honest capability reporting"
      >
        <ModalityCard modality="image" />
      </Card>
    </div>
  );
}
