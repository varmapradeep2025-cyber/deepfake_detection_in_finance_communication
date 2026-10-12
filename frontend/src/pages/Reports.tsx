import { Download, FileText } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { DemoBadge } from "../components/ui/DemoBadge";
import { RiskBadge } from "../components/ui/RiskBadge";
import { DEMO_EVIDENCE, DEMO_SUBJECT } from "../data";
import { DEMO_LABEL } from "../lib/mockAnalysis";
import { formatDateTime } from "../lib/format";

const GENERATED_AT = "2026-10-10T09:20:00.000Z";

const REPORT_SECTIONS = [
  {
    title: "Subject",
    body: `${DEMO_SUBJECT.filename} — SHA-256 ${DEMO_SUBJECT.sha256.slice(0, 16)}…`,
  },
  {
    title: "Scope",
    body: "Forensic, model-based, and threat-intelligence findings for a single evidence item.",
  },
  {
    title: "Findings (demo)",
    body: "Placeholder result only. No trained detector produced this assessment.",
  },
  {
    title: "Limitations",
    body: "No inference executed. Confidence values are simulated. Source verification is required before any action.",
  },
];

function buildReportText(): string {
  const lines: string[] = [
    "# Forensic Report (DEMO)",
    "",
    `> ${DEMO_LABEL}`,
    "",
    `Generated: ${GENERATED_AT}`,
    "",
    "## Subject",
    `- File: ${DEMO_SUBJECT.filename}`,
    `- SHA-256: ${DEMO_SUBJECT.sha256}`,
    "",
    "## Findings",
    "",
  ];
  for (const item of DEMO_EVIDENCE) {
    lines.push(
      `- [${item.claimKind}] ${item.label}: ${item.value} (source: ${item.source})`,
    );
  }
  lines.push(
    "",
    "## Limitations",
    "This report is a layout demonstration. It is not the result of any model",
    "inference and must not be cited as an experimental finding.",
    "",
  );
  return lines.join("\n");
}

export function Reports() {
  function download() {
    const blob = new Blob([buildReportText()], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "demo-forensic-report.md";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Deliverables"
        title="Reports"
        description="Preview a forensic report layout and export a demo document."
        actions={<DemoBadge label="DEMO DATA" />}
      />

      <div className="grid grid--2">
        <Card
          title="Report preview"
          subtitle="Layout only — contents are illustrative"
          actions={
            <span className="card__badge">
              <FileText size={14} aria-hidden /> Markdown
            </span>
          }
        >
          <article className="report-preview">
            <header className="report-preview__head">
              <div>
                <h3 className="report-preview__title">
                  Multimodal Forensic Report
                </h3>
                <span className="muted">
                  Generated {formatDateTime(GENERATED_AT)}
                </span>
              </div>
              <RiskBadge risk="high" />
            </header>
            <dl className="kv">
              {REPORT_SECTIONS.map((section) => (
                <div className="kv__row" key={section.title}>
                  <dt className="kv__key">{section.title}</dt>
                  <dd className="kv__value">{section.body}</dd>
                </div>
              ))}
            </dl>
            <p className="table-note">{DEMO_LABEL}</p>
          </article>
        </Card>

        <Card title="Export" subtitle="Client-side download; nothing is transmitted">
          <p className="muted">
            The demo report is assembled in the browser and downloaded directly.
            No data leaves this machine, and no backend endpoint is called.
          </p>
          <button type="button" className="btn btn--primary" onClick={download}>
            <Download size={15} aria-hidden />
            Download demo report
          </button>
          <div className="state-banner state-banner--warning" role="note">
            Exported content is clearly marked as demonstration data.
          </div>
        </Card>
      </div>
    </div>
  );
}
