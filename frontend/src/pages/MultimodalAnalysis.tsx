import { useState } from "react";
import { AudioLines, Image as ImageIcon, Link2, QrCode, Video } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { DemoBadge } from "../components/ui/DemoBadge";
import { AnalysisPanel } from "../components/analysis/AnalysisPanel";
import type { Modality } from "../types";

const MODALITIES: {
  id: Modality;
  label: string;
  icon: typeof ImageIcon;
  blurb: string;
}[] = [
  {
    id: "image",
    label: "Image",
    icon: ImageIcon,
    blurb: "Facial and artifact detection for stills and extracted frames.",
  },
  {
    id: "video",
    label: "Video",
    icon: Video,
    blurb: "Temporal consistency across frames of a clip.",
  },
  {
    id: "audio",
    label: "Audio",
    icon: AudioLines,
    blurb: "Speech, vocoder, and re-synthesis artifacts in recordings.",
  },
  {
    id: "qr",
    label: "QR code",
    icon: QrCode,
    blurb: "Payment and onboarding QR images.",
  },
  {
    id: "url",
    label: "URL / phishing",
    icon: Link2,
    blurb: "Suspicious links from email, chat, or documents.",
  },
];

export function MultimodalAnalysis() {
  const [active, setActive] = useState<Modality>("image");
  const activeMeta = MODALITIES.find((m) => m.id === active) ?? MODALITIES[0];

  return (
    <div className="page">
      <PageHeader
        eyebrow="Workflows"
        title="Multimodal Analysis"
        description="Run an analysis workflow per modality. All results in this build are deterministic demo data."
        actions={<DemoBadge />}
      />

      <div className="tabs" role="tablist" aria-label="Analysis modality">
        {MODALITIES.map((item) => {
          const Icon = item.icon;
          const selected = item.id === active;
          return (
            <button
              key={item.id}
              role="tab"
              aria-selected={selected}
              className={`tab${selected ? " tab--active" : ""}`}
              onClick={() => setActive(item.id)}
            >
              <Icon size={15} aria-hidden />
              {item.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid--2">
        <Card
          title={`${activeMeta.label} analysis`}
          subtitle={activeMeta.blurb}
          actions={<DemoBadge label="DEMO MODE" />}
        >
          <AnalysisPanel modality={active} />
        </Card>

        <Card title="How this workflow behaves">
          <ul className="bullet-list">
            <li>
              Choose a file (drag-and-drop or picker) or enter a URL. Files are
              validated locally for type and size and are never uploaded anywhere.
            </li>
            <li>
              A selected file shows its name, size, and modality, and can be
              removed before starting.
            </li>
            <li>
              The workflow moves through idle, ready, analyzing, completed, and
              error states.
            </li>
            <li>
              Analysis is a <strong>deterministic mock</strong> so the flow can be
              demonstrated without trained models.
            </li>
            <li>
              Mock confidence and risk scores are not model outputs and must not be
              reported as experimental results.
            </li>
          </ul>
          <div className="state-banner state-banner--warning" role="note">
            No detection algorithm is implemented in this repository. No model is
            trained, downloaded, or executed.
          </div>
        </Card>
      </div>
    </div>
  );
}
