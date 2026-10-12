import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { DemoBadge } from "../components/ui/DemoBadge";
import { StatCard } from "../components/ui/StatCard";
import { CalibrationChart } from "../components/charts/CalibrationChart";
import { PerformanceChart } from "../components/charts/PerformanceChart";
import {
  DEMO_CALIBRATION,
  DEMO_DATASET_COMPARISON,
  DEMO_METRIC_CARDS,
} from "../data";
import { formatNumber, formatPercent } from "../lib/format";

export function DatasetPerformance() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Evaluation"
        title="Dataset Performance"
        description="Accuracy, precision, recall, F1, calibration, and generalization gap."
        actions={<DemoBadge label="NOT MEASURED" />}
      />

      <div className="state-banner state-banner--warning" role="note">
        No model has been trained and no dataset has been acquired, so every metric
        is unmeasured. Cards display an em dash, never a fabricated zero.
      </div>

      <div className="grid grid--3">
        {DEMO_METRIC_CARDS.map((metric) => (
          <StatCard
            key={metric.key}
            label={metric.label}
            value={
              metric.value === null
                ? "—"
                : metric.format === "percent"
                  ? formatPercent(metric.value)
                  : formatNumber(metric.value)
            }
            meta={`${metric.note} · ${metric.direction.replace("-", " ")}`}
          />
        ))}
      </div>

      <div className="grid grid--2">
        <Card
          title="Known vs. unseen"
          subtitle="Generalization comparison across splits"
          actions={<DemoBadge label="NOT MEASURED" />}
        >
          <PerformanceChart data={DEMO_DATASET_COMPARISON} />
        </Card>

        <Card
          title="Calibration"
          subtitle="Reliability diagram placeholder"
          actions={<DemoBadge label="ILLUSTRATIVE" />}
        >
          <CalibrationChart data={DEMO_CALIBRATION} />
          <p className="table-note">
            Diagonal line is the ideal reference. Calibration is measured with ECE
            and Brier score, never assumed.
          </p>
        </Card>
      </div>
    </div>
  );
}
