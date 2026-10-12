import { Link } from "react-router-dom";
import {
  ArrowRight,
  Cpu,
  FileImage,
  FileVideo,
  AudioLines,
  QrCode,
  ScanSearch,
} from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { RiskBadge } from "../components/ui/RiskBadge";
import { Badge } from "../components/ui/Badge";
import { DemoBadge } from "../components/ui/DemoBadge";
import { ModalityChart } from "../components/charts/ModalityChart";
import { RiskDistributionChart } from "../components/charts/RiskDistributionChart";
import { DEMO_DASHBOARD_SUMMARY, DEMO_MODELS } from "../data";
import { formatDateTime, formatPercent } from "../lib/format";
import type { ModelAvailability } from "../types";

const MODEL_TONE: Record<
  ModelAvailability["state"],
  "neutral" | "warning" | "high"
> = {
  not_trained: "warning",
  unavailable: "high",
  declared: "neutral",
};

const MODEL_LABEL: Record<ModelAvailability["state"], string> = {
  not_trained: "Not trained",
  unavailable: "Unavailable",
  declared: "Declared",
};

export function Dashboard() {
  const summary = DEMO_DASHBOARD_SUMMARY;

  return (
    <div className="page">
      <PageHeader
        eyebrow="Forensic Intelligence Center"
        title="Deepfake Forensics Dashboard"
        description="Premium research-grade forensics interface with honest capability reporting."
        actions={<DemoBadge />}
      />

      <div className="stat-grid">
        <StatCard
          label="Total analyses"
          value={summary.totalAnalyses}
          meta="sample history"
        />
        <StatCard
          label="Completed today"
          value={summary.completedToday}
          meta="2026-10-10"
        />
        <StatCard
          label="High risk"
          value={summary.highRisk}
          tone="high"
          meta="requires review"
        />
        <StatCard
          label="Avg. confidence"
          value={formatPercent(summary.avgConfidence)}
          meta="placeholder values"
        />
      </div>

      <div className="grid grid--2">
        <Card
          title="Modality breakdown"
          subtitle="Analyses by media type"
          actions={<DemoBadge label="DEMO DATA" />}
        >
          <ModalityChart data={summary.modalityBreakdown} />
        </Card>
        <Card
          title="Risk distribution"
          subtitle="Counts by assigned risk level"
          actions={<DemoBadge label="DEMO DATA" />}
        >
          <RiskDistributionChart data={summary.riskDistribution} />
        </Card>
      </div>

      <Card
        title="Quick actions"
        subtitle="Jump to dedicated forensic workspaces"
      >
        <div className="dashboard__quick-actions">
          <Link className="btn btn--primary btn--sm" to="/image-analysis">
            <FileImage size={14} aria-hidden /> Image Analysis
          </Link>
          <Link className="btn btn--ghost btn--sm" to="/video-analysis">
            <FileVideo size={14} aria-hidden /> Video Analysis
          </Link>
          <Link className="btn btn--ghost btn--sm" to="/audio-analysis">
            <AudioLines size={14} aria-hidden /> Audio Analysis
          </Link>
          <Link className="btn btn--ghost btn--sm" to="/qr-url-analysis">
            <QrCode size={14} aria-hidden /> QR &amp; URL Analysis
          </Link>
          <Link className="btn btn--ghost btn--sm" to="/analysis">
            <ScanSearch size={14} aria-hidden /> Multimodal Analysis
          </Link>
        </div>
      </Card>

      <Card
        title="Modality workspaces"
        subtitle="Four individually-runnable analyses with honest capability reporting"
        actions={<DemoBadge label="DEMO UI" />}
      >
        <div className="dashboard__modality-cards">
          <div className="dashboard__modality-card dashboard__modality-card--image">
            <div className="dashboard__modality-card__header">
              <div>
                <div className="dashboard__modality-card__title">
                  <div className="dashboard__modality-card__accent">
                    <FileImage size={16} aria-hidden />
                  </div>
                  Image Analysis
                </div>
                <p className="dashboard__modality-card__path">
                  /image-analysis
                </p>
              </div>
              <Badge tone="neutral">Forensics</Badge>
            </div>
            <p className="muted">
              Still-image deepfake forensics with evidence and provenance.
            </p>
            <Link className="btn btn--ghost btn--sm" to="/image-analysis">
              Open <ArrowRight size={14} aria-hidden />
            </Link>
          </div>

          <div className="dashboard__modality-card dashboard__modality-card--video">
            <div className="dashboard__modality-card__header">
              <div>
                <div className="dashboard__modality-card__title">
                  <div className="dashboard__modality-card__accent">
                    <FileVideo size={16} aria-hidden />
                  </div>
                  Video Analysis
                </div>
                <p className="dashboard__modality-card__path">
                  /video-analysis
                </p>
              </div>
              <Badge tone="neutral">Temporal</Badge>
            </div>
            <p className="muted">
              Temporal consistency analysis for video content.
            </p>
            <Link className="btn btn--ghost btn--sm" to="/video-analysis">
              Open <ArrowRight size={14} aria-hidden />
            </Link>
          </div>

          <div className="dashboard__modality-card dashboard__modality-card--audio">
            <div className="dashboard__modality-card__header">
              <div>
                <div className="dashboard__modality-card__title">
                  <div className="dashboard__modality-card__accent">
                    <AudioLines size={16} aria-hidden />
                  </div>
                  Audio Analysis
                </div>
                <p className="dashboard__modality-card__path">
                  /audio-analysis
                </p>
              </div>
              <Badge tone="neutral">Speech</Badge>
            </div>
            <p className="muted">
              Audio forensics with honest capability reporting.
            </p>
            <Link className="btn btn--ghost btn--sm" to="/audio-analysis">
              Open <ArrowRight size={14} aria-hidden />
            </Link>
          </div>

          <div className="dashboard__modality-card dashboard__modality-card--qr">
            <div className="dashboard__modality-card__header">
              <div>
                <div className="dashboard__modality-card__title">
                  <div className="dashboard__modality-card__accent">
                    <QrCode size={16} aria-hidden />
                  </div>
                  QR &amp; URL Analysis
                </div>
                <p className="dashboard__modality-card__path">
                  /qr-url-analysis
                </p>
              </div>
              <Badge tone="neutral">Phishing</Badge>
            </div>
            <p className="muted">
              QR decoding and URL intelligence without network access.
            </p>
            <Link className="btn btn--ghost btn--sm" to="/qr-url-analysis">
              Open <ArrowRight size={14} aria-hidden />
            </Link>
          </div>
        </div>
      </Card>

      <Card
        title="Recent analyses"
        subtitle="Sample analysis history"
        actions={
          <Link className="btn btn--ghost btn--sm" to="/analysis">
            New analysis <ArrowRight size={14} aria-hidden />
          </Link>
        }
      >
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">ID</th>
                <th scope="col">Subject</th>
                <th scope="col">Modality</th>
                <th scope="col">Risk</th>
                <th scope="col">Confidence</th>
                <th scope="col">When</th>
                <th scope="col">Analyst</th>
              </tr>
            </thead>
            <tbody>
              {summary.recent.map((row) => (
                <tr key={row.id}>
                  <td className="mono">{row.id}</td>
                  <td className="table__subject" title={row.filename}>
                    {row.filename}
                  </td>
                  <td className="capitalize">{row.modality}</td>
                  <td>
                    <RiskBadge risk={row.risk} />
                  </td>
                  <td>{formatPercent(row.confidence)}</td>
                  <td className="muted">{formatDateTime(row.createdAt)}</td>
                  <td className="muted">{row.analyst}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-note">
          <DemoBadge label="DEMO DATA" /> Rows in this table are illustrative and
          were not produced by a trained model.
        </p>
      </Card>

      <Card
        title="Model availability"
        subtitle="Declared detector slots — none is trained"
        actions={
          <span className="card__badge">
            <Cpu size={14} aria-hidden /> ml/
          </span>
        }
      >
        <ul className="model-list">
          {DEMO_MODELS.map((model) => (
            <li key={model.id} className="model-list__item">
              <div>
                <span className="model-list__name">{model.name}</span>
                <span className="model-list__detail">{model.detail}</span>
              </div>
              <div className="model-list__meta">
                <span className="muted">{model.modality}</span>
                <Badge tone={MODEL_TONE[model.state]}>
                  {MODEL_LABEL[model.state]}
                </Badge>
              </div>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
