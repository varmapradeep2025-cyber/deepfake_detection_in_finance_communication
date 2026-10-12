import { useState } from "react";
import { Check, Copy, FileWarning } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { DemoBadge } from "../components/ui/DemoBadge";
import {
  DEMO_CUSTODY,
  DEMO_EVIDENCE,
  DEMO_METADATA,
  DEMO_RECOMMENDATIONS,
  DEMO_SUBJECT,
} from "../data";
import { formatDateTime, formatPercent } from "../lib/format";
import type { ClaimKind } from "../types";

const CLAIM_TONE: Record<ClaimKind, "low" | "neutral" | "warning"> = {
  observed: "low",
  derived: "neutral",
  inferred: "warning",
};

const PRIORITY_TONE = {
  high: "high",
  medium: "elevated",
  low: "low",
} as const;

export function EvidenceProvenance() {
  const [copied, setCopied] = useState(false);

  async function copyHash() {
    try {
      await navigator.clipboard.writeText(DEMO_SUBJECT.sha256);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Forensics"
        title="Evidence & Provenance"
        description="What was analysed, under which hash, and how it was handled."
        actions={<DemoBadge label="DEMO DATA" />}
      />

      <div className="grid grid--2">
        <Card title="Content hash" subtitle="Streamed SHA-256 of the analysed bytes">
          <div className="hash-block">
            <span className="hash-block__label">SHA-256</span>
            <code className="hash-block__value mono" data-testid="sha256">
              {DEMO_SUBJECT.sha256}
            </code>
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={copyHash}
            >
              {copied ? <Check size={14} aria-hidden /> : <Copy size={14} aria-hidden />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <dl className="kv">
            <div className="kv__row">
              <dt className="kv__key">Subject</dt>
              <dd className="kv__value">{DEMO_SUBJECT.filename}</dd>
            </div>
            <div className="kv__row">
              <dt className="kv__key">Analysis timestamp</dt>
              <dd className="kv__value">
                {formatDateTime(DEMO_SUBJECT.capturedAt)}
              </dd>
            </div>
          </dl>
          <p className="table-note">
            Sample digest shown for layout purposes only.
          </p>
        </Card>

        <Card title="Available metadata" subtitle="Container fields observed">
          <div className="table-wrap">
            <table className="table table--compact">
              <tbody>
                {DEMO_METADATA.map((field) => (
                  <tr key={field.key}>
                    <th scope="row" className="mono">
                      {field.key}
                    </th>
                    <td className={field.missing ? "muted" : ""}>
                      {field.missing && (
                        <FileWarning size={13} aria-hidden className="inline-icon" />
                      )}
                      {field.value}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="table-note">
            Explicit absence is recorded rather than guessed.
          </p>
        </Card>
      </div>

      <Card
        title="Chain of custody"
        subtitle="Append-only handling timeline"
        actions={<DemoBadge label="DEMO DATA" />}
      >
        <ol className="timeline">
          {DEMO_CUSTODY.map((event) => (
            <li key={event.id} className="timeline__item">
              <span className="timeline__dot" aria-hidden />
              <div className="timeline__body">
                <div className="timeline__head">
                  <span className="timeline__action">{event.action}</span>
                  <span className="muted timeline__time">
                    {formatDateTime(event.timestamp)}
                  </span>
                </div>
                <div className="timeline__meta">
                  <span>Actor: {event.actor}</span>
                  <span>Location: {event.location}</span>
                  {event.hash && <span className="mono">{event.hash}</span>}
                </div>
              </div>
            </li>
          ))}
        </ol>
      </Card>

      <div className="grid grid--2">
        <Card title="Evidence items" subtitle="Claims tagged by origin">
          <ul className="evidence-list">
            {DEMO_EVIDENCE.map((item) => (
              <li key={item.id} className="evidence-item">
                <div className="evidence-item__head">
                  <span className="evidence-item__label">{item.label}</span>
                  <Badge tone={CLAIM_TONE[item.claimKind]}>
                    {item.claimKind}
                  </Badge>
                </div>
                <span className="evidence-item__value">{item.value}</span>
                <div className="evidence-item__meta muted">
                  <span>{item.source}</span>
                  <span>{item.method}</span>
                  <span>conf: {formatPercent(item.confidence, 0)}</span>
                </div>
              </li>
            ))}
          </ul>
        </Card>

        <Card
          title="Source-verification recommendations"
          subtitle="Next steps for a human analyst"
        >
          <ul className="reco-list">
            {DEMO_RECOMMENDATIONS.map((reco) => (
              <li key={reco.id} className="reco-item">
                <div className="reco-item__head">
                  <span className="reco-item__title">{reco.title}</span>
                  <Badge tone={PRIORITY_TONE[reco.priority]}>
                    {reco.priority}
                  </Badge>
                </div>
                <p className="reco-item__detail">{reco.detail}</p>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
