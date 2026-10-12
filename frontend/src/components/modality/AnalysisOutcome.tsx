import { CircleCheck, TriangleAlert } from "lucide-react";
import type {
  AnalysisModality,
  AnalysisStatus,
  ApiAnalysisRecord,
  ApiCodeEntry,
  ApiUrlAnalysis,
} from "../../types";
import {
  ANALYSIS_STATUS_LABELS,
  ANALYSIS_STATUS_TONES,
  reportHasScore,
} from "../../types";
import { Badge } from "../ui/Badge";
import { RiskBadge } from "../ui/RiskBadge";
import { formatNumber } from "../../lib/format";

const MODALITY_LABELS: Record<string, string> = {
  image: "Image",
  video: "Video",
  audio: "Audio",
  qr: "QR & URL",
};

function modalityLabel(modality: string): string {
  return MODALITY_LABELS[modality as AnalysisModality] ?? modality;
}

function SeverityBadge({ severity }: { severity: string }) {
  const tone =
    severity === "high"
      ? "high"
      : severity === "medium"
        ? "elevated"
        : severity === "low"
          ? "warning"
          : "info";
  return <Badge tone={tone}>{severity}</Badge>;
}

function UrlBlock({ url }: { url: ApiUrlAnalysis }) {
  return (
    <div className="modality-url">
      <dl className="kv">
        <div className="kv__row">
          <dt className="kv__key">Destination</dt>
          <dd className="kv__value mono">{url.url}</dd>
        </div>
        <div className="kv__row">
          <dt className="kv__key">Host</dt>
          <dd className="kv__value mono">
            {url.hostname}
            {url.port !== null ? `:${url.port}` : ""} · {url.scheme}
          </dd>
        </div>
      </dl>
      {url.findings.length === 0 ? (
        <p className="muted">
          No known URL pattern matched. This is not a statement that the link is safe.
        </p>
      ) : (
        <ul className="evidence-list">
          {url.findings.map((finding) => (
            <li className="evidence-item" key={`${url.url}:${finding.code}`}>
              <span className="evidence-item__head">
                <span className="evidence-item__label">
                  <SeverityBadge severity={finding.severity} /> {finding.message}
                </span>
              </span>
              <span className="evidence-item__meta muted">
                <span className="mono">{finding.code}</span>
                {finding.detail ? <span>{finding.detail}</span> : null}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function CodesBlock({ codes }: { codes: ApiCodeEntry[] }) {
  return (
    <div className="modality-codes">
      {codes.length === 0 ? (
        <p className="muted">No QR code was detected in this image.</p>
      ) : (
        codes.map((code, index) => (
          <div className="modality-code" key={`${index}:${code.payload}`}>
            <div className="modality-code__head">
              <span className="muted">Payload #{index + 1}</span>
              <Badge tone={code.kind === "url" ? "warning" : "neutral"}>
                {code.kind}
              </Badge>
            </div>
            <p className="modality-code__payload mono">{code.payload}</p>
            {code.url ? <UrlBlock url={code.url} /> : null}
          </div>
        ))
      )}
    </div>
  );
}

function statusBadge(status: AnalysisStatus) {
  return <Badge tone={ANALYSIS_STATUS_TONES[status]}>{ANALYSIS_STATUS_LABELS[status]}</Badge>;
}

/** Renders one stored analysis record, including the honest "no score" states. */
export function AnalysisOutcome({ record }: { record: ApiAnalysisRecord }) {
  const { report } = record;
  const hasScore = reportHasScore(report);
  const risk = report.risk;
  const riskKnown = risk !== null && risk.level !== "unknown";
  const codes = report.result?.codes;

  return (
    <div className="modality-outcome" data-testid="modality-outcome">
      <div className="result__head">
        <div>
          <span className="result__eyebrow">{modalityLabel(record.modality)} analysis</span>
          <h4 className="result__subject mono" title={record.source_name}>
            {record.source_name}
          </h4>
        </div>
        <div className="modality-outcome__badges">
          {statusBadge(report.status)}
          {riskKnown ? <RiskBadge risk={risk.level} /> : null}
        </div>
      </div>

      <dl className="kv">
        <div className="kv__row">
          <dt className="kv__key">Detector score</dt>
          <dd className="kv__value">
            {hasScore ? (
              <>
                <span className="mono">{formatNumber(report.output?.raw_score ?? null)}</span>{" "}
                <span className="muted">
                  ({report.output?.score_kind}
                  {report.output?.calibrated ? ", calibrated" : ", uncalibrated"})
                </span>
              </>
            ) : (
              <span className="muted">
                No detector score produced — completion and scoring are separate.
              </span>
            )}
          </dd>
        </div>
        {risk ? (
          <div className="kv__row">
            <dt className="kv__key">Risk assessment</dt>
            <dd className="kv__value">{risk.rationale}</dd>
          </div>
        ) : null}
        <div className="kv__row">
          <dt className="kv__key">Request</dt>
          <dd className="kv__value mono">{report.request_id}</dd>
        </div>
      </dl>

      {report.errors.length > 0 ? (
        <div className="state-banner state-banner--error" role="alert">
          <TriangleAlert size={15} aria-hidden />
          <span>{report.errors.join(" ")}</span>
        </div>
      ) : null}

      {report.warnings.length > 0 ? (
        <div className="state-banner state-banner--warning" role="note">
          {report.warnings.map((warning) => (
            <span key={warning}>{warning}</span>
          ))}
        </div>
      ) : null}

      {codes ? <CodesBlock codes={codes} /> : null}

      {report.evidence.length > 0 ? (
        <div className="result__signals">
          <h5 className="result__eyebrow">Evidence ({report.evidence.length})</h5>
        <ul className="evidence-list">
          {report.evidence.map((item) => (
            <li className="evidence-item" key={item.id}>
              <span className="evidence-item__head">
                <span className="evidence-item__label">{item.label}</span>
                <span className="evidence-item__label muted">{item.claim_kind}</span>
              </span>
              <span className="evidence-item__value mono">{item.value}</span>
              <span className="evidence-item__meta muted">
                <span className="mono">{item.method}</span>
                <span>{item.source}</span>
              </span>
            </li>
          ))}
        </ul>
        </div>
      ) : null}

      <p className="result__summary">
        <CircleCheck size={14} aria-hidden /> {report.recommendation}
      </p>
    </div>
  );
}
