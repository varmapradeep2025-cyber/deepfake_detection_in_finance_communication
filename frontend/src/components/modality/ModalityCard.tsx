import { useState } from "react";
import { TriangleAlert } from "lucide-react";
import type { AnalysisModality, ApiAnalysisRecord } from "../../types";
import { analyzeModality } from "../../api/modalities";
import type { ApiResult } from "../../api/client";
import { getModalityMeta, validateFile } from "../../lib/validation";
import { Card } from "../ui/Card";
import { FileDropzone } from "../analysis/FileDropzone";
import { AnalysisOutcome } from "./AnalysisOutcome";

interface ModalityCardProps {
  modality: AnalysisModality;
  /** Called with the record whenever a run succeeds, for the report builder. */
  onResult?: (record: ApiAnalysisRecord) => void;
}

type CardState = "idle" | "ready" | "analyzing" | "error";

function describeFailure(result: Extract<ApiResult<unknown>, { ok: false }>): string {
  switch (result.kind) {
    case "timeout":
      return `No response within the client timeout. The inference service is not running, so nothing was analyzed. (${result.error})`;
    case "network":
      return `${result.error} The local inference service is not running, so nothing was analyzed.`;
    case "http":
      return `${result.error} The service rejected the request.`;
    case "parse":
      return `${result.error} The service returned an unexpected response.`;
  }
}

/**
 * One modality's analysis card. Selects and locally validates a file, sends it to
 * the inference client, and renders the real returned report. When no service is
 * running the failure is shown verbatim rather than replaced with a placeholder.
 */
export function ModalityCard({ modality, onResult }: ModalityCardProps) {
  const meta = getModalityMeta(modality);
  const [state, setState] = useState<CardState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [record, setRecord] = useState<ApiAnalysisRecord | null>(null);

  function reset() {
    setState("idle");
    setFile(null);
    setError(null);
    setRecord(null);
  }

  function handleFile(selected: File) {
    const validation = validateFile(selected, modality);
    if (!validation.valid) {
      setFile(null);
      setError(validation.error ?? "Invalid file.");
      setState("error");
      return;
    }
    setFile(selected);
    setError(null);
    setRecord(null);
    setState("ready");
  }

  async function handleAnalyze() {
    if (!file) return;
    setError(null);
    setRecord(null);
    setState("analyzing");
    const result = await analyzeModality(modality, file);
    if (result.ok) {
      setRecord(result.data);
      setState("idle");
      onResult?.(result.data);
      return;
    }
    setError(describeFailure(result));
    setState("error");
  }

  const analyzing = state === "analyzing";

  return (
    <Card
      title={`${meta.label} analysis`}
      subtitle={meta.description}
      actions={<span className="card__badge">{record ? "Live" : "Service-backed"}</span>}
    >
      <div className="modality-card" data-testid={`modality-card-${modality}`}>
        {record ? (
          <>
            <AnalysisOutcome record={record} />
            <div className="analysis-panel__actions">
              <button type="button" className="btn btn--ghost" onClick={reset}>
                Analyze another file
              </button>
            </div>
          </>
        ) : (
          <>
            <FileDropzone
              modality={modality}
              file={file}
              disabled={analyzing}
              onFile={handleFile}
              onClear={reset}
            />

            {error ? (
              <div className="state-banner state-banner--error" role="alert">
                <TriangleAlert size={15} aria-hidden />
                <span>{error}</span>
              </div>
            ) : null}

            {analyzing ? (
              <div className="state-banner state-banner--info" role="status">
                Running analysis…
              </div>
            ) : null}

            <div className="analysis-panel__actions">
              <button
                type="button"
                className="btn btn--primary"
                onClick={handleAnalyze}
                disabled={!file || analyzing}
              >
                Analyze
              </button>
              {file && !analyzing ? (
                <button type="button" className="btn btn--ghost" onClick={reset}>
                  Clear
                </button>
              ) : null}
            </div>
          </>
        )}
      </div>
    </Card>
  );
}
