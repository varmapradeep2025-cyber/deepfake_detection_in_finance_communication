import { useEffect, useRef, useState } from "react";
import { LoaderCircle, Play, TriangleAlert } from "lucide-react";
import type { AnalysisState, Modality } from "../../types";
import {
  analyzeMock,
  type MockAnalysisResult,
} from "../../lib/mockAnalysis";
import { getModalityMeta, validateFile, validateUrl } from "../../lib/validation";
import { FileDropzone } from "./FileDropzone";
import { UrlInput } from "./UrlInput";
import { ResultView } from "./ResultView";

interface AnalysisPanelProps {
  modality: Modality;
}

export function AnalysisPanel({ modality }: AnalysisPanelProps) {
  const meta = getModalityMeta(modality);
  const [state, setState] = useState<AnalysisState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MockAnalysisResult | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function clearTimer() {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }

  function reset() {
    clearTimer();
    setState("idle");
    setFile(null);
    setUrl("");
    setError(null);
    setResult(null);
  }

  // Reset the whole workflow when the user switches modality.
  useEffect(() => {
    reset();
    return clearTimer;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modality]);

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
    setResult(null);
    setState("ready");
  }

  function handleUrlChange(value: string) {
    setUrl(value);
    setError(null);
    setResult(null);
    setState(value.trim() ? "ready" : "idle");
  }

  function handleStart() {
    let subject: string;
    if (modality === "url") {
      const validation = validateUrl(url);
      if (!validation.valid) {
        setError(validation.error ?? "Invalid URL.");
        setState("error");
        return;
      }
      subject = url.trim();
    } else {
      if (!file) return;
      subject = file.name;
    }

    const outcome = analyzeMock(modality, subject);
    setError(null);
    setResult(null);
    setState("analyzing");
    clearTimer();
    timerRef.current = setTimeout(() => {
      setResult(outcome);
      setState("completed");
      timerRef.current = null;
    }, outcome.durationMs);
  }

  const isUrl = modality === "url";
  const analyzing = state === "analyzing";
  const canStart =
    !analyzing &&
    (isUrl ? url.trim().length > 0 : file !== null);

  return (
    <div
      className="analysis-panel"
      data-testid="analysis-panel"
      data-state={state}
    >
      <p className="analysis-panel__desc">{meta.description}</p>

      {state === "completed" && result ? (
        <ResultView result={result} onReset={reset} />
      ) : (
        <>
          {isUrl ? (
            <UrlInput value={url} onChange={handleUrlChange} disabled={analyzing} />
          ) : (
            <FileDropzone
              modality={modality}
              file={file}
              disabled={analyzing}
              onFile={handleFile}
              onClear={reset}
            />
          )}

          {error && (
            <div className="state-banner state-banner--error" role="alert">
              <TriangleAlert size={15} aria-hidden />
              <span>{error}</span>
            </div>
          )}

          {analyzing && (
            <div className="state-banner state-banner--info" role="status">
              <LoaderCircle size={15} className="spin" aria-hidden />
              <span>Running deterministic demo analysis…</span>
            </div>
          )}

          <div className="analysis-panel__actions">
            <button
              type="button"
              className="btn btn--primary"
              onClick={handleStart}
              disabled={!canStart}
            >
              <Play size={15} aria-hidden />
              Run demo analysis
            </button>
            {(file || url) && !analyzing && (
              <button type="button" className="btn btn--ghost" onClick={reset}>
                Clear
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
