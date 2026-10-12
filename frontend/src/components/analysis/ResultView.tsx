import { ShieldAlert, TriangleAlert } from "lucide-react";
import type { MockAnalysisResult } from "../../lib/mockAnalysis";
import { DEMO_LABEL } from "../../lib/mockAnalysis";
import { formatPercent, truncateMiddle } from "../../lib/format";
import { getModalityMeta } from "../../lib/validation";
import { RiskBadge } from "../ui/RiskBadge";
import { DemoBadge } from "../ui/DemoBadge";

interface ResultViewProps {
  result: MockAnalysisResult;
  onReset: () => void;
}

export function ResultView({ result, onReset }: ResultViewProps) {
  const meta = getModalityMeta(result.modality);

  return (
    <div className="result" data-testid="analysis-result">
      <div className="demo-banner" role="alert">
        <TriangleAlert size={16} aria-hidden />
        <span>{DEMO_LABEL}</span>
      </div>

      <div className="result__head">
        <div>
          <span className="result__eyebrow">{meta.label} analysis</span>
          <h3 className="result__subject" title={result.subject}>
            {truncateMiddle(result.subject, 28, 14)}
          </h3>
        </div>
        <RiskBadge risk={result.risk} />
      </div>

      <dl className="kv">
        <div className="kv__row">
          <dt className="kv__key">Demo result id</dt>
          <dd className="kv__value mono">{result.id}</dd>
        </div>
        <div className="kv__row">
          <dt className="kv__key">Placeholder confidence</dt>
          <dd className="kv__value">{formatPercent(result.confidence)}</dd>
        </div>
        <div className="kv__row">
          <dt className="kv__key">Simulated duration</dt>
          <dd className="kv__value">{(result.durationMs / 1000).toFixed(1)} s</dd>
        </div>
        <div className="kv__row">
          <dt className="kv__key">Generated at</dt>
          <dd className="kv__value">{result.startedAt}</dd>
        </div>
      </dl>

      <div className="result__signals">
        <h4 className="result__section-title">Placeholder signals</h4>
        <ul>
          {result.signals.map((signal) => (
            <li key={signal.name} className="signal">
              <div className="signal__row">
                <span className="signal__name">{signal.name}</span>
                <span className="signal__value mono">
                  {signal.value.toFixed(3)}
                </span>
              </div>
              <div className="signal__track" aria-hidden>
                <span
                  className="signal__fill"
                  style={{ width: `${Math.round(signal.value * 100)}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      </div>

      <p className="result__summary">
        <ShieldAlert size={14} aria-hidden /> {result.summary}
      </p>

      <div className="result__foot">
        <DemoBadge label="DEMO DATA" />
        <button type="button" className="btn btn--primary btn--sm" onClick={onReset}>
          Start over
        </button>
      </div>
    </div>
  );
}
