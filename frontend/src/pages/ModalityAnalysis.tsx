import { useCallback, useState } from "react";
import { WifiOff } from "lucide-react";
import type { AnalysisModality, ApiAnalysisRecord } from "../types";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { ModalityCard } from "../components/modality/ModalityCard";
import { ReportBuilder } from "../components/modality/ReportBuilder";
import { useBackendStatus } from "../context/backendStatus";

const MODALITIES: AnalysisModality[] = ["image", "video", "audio", "qr"];

/**
 * Individual-modality analysis backed by the inference library, with a report
 * builder over the stored results.
 *
 * This page calls the typed client in `api/modalities.ts`. No service implements
 * those endpoints yet, so in this build every run resolves to a network failure
 * that the UI states plainly — it never substitutes demo data for a real result.
 * The four cards and the builder are real: point them at a running service and they
 * work unchanged.
 */
export function ModalityAnalysis() {
  const { status } = useBackendStatus();
  const [liveRecords, setLiveRecords] = useState<ApiAnalysisRecord[]>([]);

  const handleResult = useCallback((record: ApiAnalysisRecord) => {
    setLiveRecords((current) => {
      const without = current.filter((item) => item.report_id !== record.report_id);
      return [...without, record];
    });
  }, []);

  return (
    <div className="page">
      <PageHeader
        eyebrow="Workflows"
        title="Modality Analysis"
        description="Run image, video, audio, and QR/URL analyses through the inference library, then assemble a report from the stored results."
        actions={
          <Badge tone={status.state === "connected" ? "info" : "warning"}>
            {status.state === "connected" ? "Service connected" : "Service offline"}
          </Badge>
        }
      />

      {status.state !== "connected" ? (
        <div className="state-banner state-banner--warning" role="note">
          <WifiOff size={15} aria-hidden />
          <span>
            The local inference service is not reachable at {status.baseUrl}. Every
            analysis below will fail with a network error until a service is running —
            no result is invented in its place. The analysis code itself lives in
            <code> ml/</code> and is exercised by the project's tests.
          </span>
        </div>
      ) : null}

      <div className="grid grid--2">
        {MODALITIES.map((modality) => (
          <ModalityCard key={modality} modality={modality} onResult={handleResult} />
        ))}
      </div>

      <Card title="How this differs from the demo page">
        <ul className="bullet-list">
          <li>
            This page targets the real inference client. The
            <strong> Multimodal Analysis</strong> page is a labelled demonstration
            driven by deterministic mock data.
          </li>
          <li>
            Each modality sends one artifact to the service and renders the report it
            returns, including the honest cases: no model available, a missing
            capability, an input that could not be read, or a QR decode that completes
            without a score.
          </li>
          <li>
            The report builder presents selected analyses side by side. It never
            averages or sums their scores, because each modality measures a different
            thing on a different scale.
          </li>
        </ul>
      </Card>

      <ReportBuilder liveRecords={liveRecords} />
    </div>
  );
}
