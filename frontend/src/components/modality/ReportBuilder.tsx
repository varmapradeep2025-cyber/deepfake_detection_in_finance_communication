import { useEffect, useMemo, useState } from "react";
import { FileStack, LoaderCircle, TriangleAlert } from "lucide-react";
import type { ApiAnalysisRecord, ApiMultimodalReport } from "../../types";
import { ANALYSIS_STATUS_LABELS } from "../../types";
import { buildMultimodalReport, listAnalysisRecords } from "../../api/modalities";
import { Badge } from "../ui/Badge";
import { Card } from "../ui/Card";
import { EmptyState } from "../ui/EmptyState";
import { RiskBadge } from "../ui/RiskBadge";

interface ReportBuilderProps {
  /** Records produced live by the modality cards on this page. */
  liveRecords: ApiAnalysisRecord[];
}

function MultimodalReportView({ report }: { report: ApiMultimodalReport }) {
  return (
    <div className="multimodal-report" data-testid="multimodal-report">
      <div className="result__head">
        <div>
          <span className="result__eyebrow">Assembled report</span>
          <h4 className="result__subject">{report.title}</h4>
        </div>
        <Badge tone="info">{report.sections.length} analyses</Badge>
      </div>

      <dl className="kv">
        <div className="kv__row">
          <dt className="kv__key">Scored</dt>
          <dd className="kv__value">
            {report.scored_count} scored · {report.unscored_count} without a score
          </dd>
        </div>
        <div className="kv__row">
          <dt className="kv__key">Modalities</dt>
          <dd className="kv__value">
            {Object.entries(report.modality_counts)
              .map(([name, count]) => `${name} ×${count}`)
              .join(", ")}
          </dd>
        </div>
      </dl>

      <div className="multimodal-report__sections">
        {report.sections.map((section) => (
          <div className="multimodal-report__section" key={section.report_id}>
            <div className="result__head">
              <div>
                <span className="result__eyebrow">{section.modality}</span>
                <h5 className="result__subject mono">{section.source_name}</h5>
              </div>
              <div className="modality-outcome__badges">
                <Badge tone="neutral">{ANALYSIS_STATUS_LABELS[section.status]}</Badge>
                {section.risk_level !== null ? (
                  <RiskBadge risk={section.risk_level} />
                ) : null}
              </div>
            </div>
            <p className="muted">{section.headline}</p>
            <p className="muted">
              {section.evidence_count} evidence items · {section.warning_count} warnings
              · {section.error_count} errors
            </p>
            {section.notable.length > 0 ? (
              <ul className="bullet-list">
                {section.notable.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ))}
      </div>

      {report.notable_findings.length > 0 ? (
        <div className="state-banner state-banner--warning" role="note">
          <span>
            Top-level observations (not a combined verdict):{" "}
            {report.notable_findings.join("; ")}
          </span>
        </div>
      ) : null}

      <p className="table-note">{report.disclaimer}</p>
    </div>
  );
}

/**
 * Assemble a report from selected stored analyses.
 *
 * Loads the record list from the inference client, merges in any analyses run live
 * on this page, and lets the analyst choose the subset. When the service is
 * unreachable the list cannot be fetched and that is stated plainly; no placeholder
 * record is invented, and the assembled report never fuses scores.
 */
export function ReportBuilder({ liveRecords }: ReportBuilderProps) {
  const [stored, setStored] = useState<ApiAnalysisRecord[]>([]);
  const [loadState, setLoadState] = useState<"loading" | "error" | "ready">(
    "loading",
  );
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [building, setBuilding] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);
  const [report, setReport] = useState<ApiMultimodalReport | null>(null);

  useEffect(() => {
    let cancelled = false;
    listAnalysisRecords().then((result) => {
      if (cancelled) return;
      if (result.ok) {
        setStored(result.data);
        setLoadState("ready");
      } else {
        setLoadError(result.error);
        setLoadState("error");
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const records = useMemo(() => {
    const byId = new Map<string, ApiAnalysisRecord>();
    for (const record of [...stored, ...liveRecords]) {
      byId.set(record.report_id, record);
    }
    return [...byId.values()];
  }, [stored, liveRecords]);

  function toggle(reportId: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(reportId)) next.delete(reportId);
      else next.add(reportId);
      return next;
    });
    setReport(null);
    setBuildError(null);
  }

  async function handleBuild() {
    if (selected.size === 0) return;
    setBuilding(true);
    setBuildError(null);
    const result = await buildMultimodalReport([...selected]);
    setBuilding(false);
    if (result.ok) {
      setReport(result.data);
      return;
    }
    setBuildError(result.error);
  }

  return (
    <Card
      title="Multimodal report builder"
      subtitle="Combine selected analyses side by side. Scores are never fused."
    >
      <div data-testid="report-builder">
        {loadState === "loading" ? (
          <div className="state-banner state-banner--info" role="status">
            <LoaderCircle size={15} className="spin" aria-hidden />
            Looking for stored analyses…
          </div>
        ) : null}

        {loadState === "error" ? (
          <div className="state-banner state-banner--warning" role="note">
            <TriangleAlert size={15} aria-hidden />
            <span>
              Stored analyses could not be listed: {loadError}. The local inference
              service is not running, so there is nothing to assemble.
            </span>
          </div>
        ) : null}

        {loadState === "ready" && records.length === 0 ? (
          <EmptyState
            icon={<FileStack size={22} aria-hidden />}
            title="No stored analyses"
            message="Run an analysis above, or store one through the library, to build a report."
          />
        ) : null}

        {records.length > 0 ? (
          <>
            <ul className="report-builder__list">
              {records.map((record) => (
                <li className="report-builder__item" key={record.report_id}>
                  <label className="report-builder__label">
                    <input
                      type="checkbox"
                      checked={selected.has(record.report_id)}
                      onChange={() => toggle(record.report_id)}
                    />
                    <span className="report-builder__name">{record.source_name}</span>
                    <Badge tone="neutral">{record.modality}</Badge>
                    <span className="muted">
                      {ANALYSIS_STATUS_LABELS[record.status]}
                    </span>
                  </label>
                </li>
              ))}
            </ul>

            <div className="analysis-panel__actions">
              <button
                type="button"
                className="btn btn--primary"
                onClick={handleBuild}
                disabled={selected.size === 0 || building}
              >
                Build report ({selected.size})
              </button>
            </div>
          </>
        ) : null}

        {buildError ? (
          <div className="state-banner state-banner--error" role="alert">
            <TriangleAlert size={15} aria-hidden />
            <span>Report could not be assembled: {buildError}.</span>
          </div>
        ) : null}

        {report ? <MultimodalReportView report={report} /> : null}
      </div>
    </Card>
  );
}
