import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge, type BadgeTone } from "../components/ui/Badge";
import { DemoBadge } from "../components/ui/DemoBadge";
import { PerformanceChart } from "../components/charts/PerformanceChart";
import { DEMO_DATASET_COMPARISON, DEMO_EXPERIMENTS } from "../data";
import { formatNumber } from "../lib/format";
import type { ExperimentStatus } from "../types";

const STATUS_TONE: Record<ExperimentStatus, BadgeTone> = {
  planned: "neutral",
  running: "info",
  completed: "low",
  aborted: "high",
};

const FAMILY_LABEL: Record<string, string> = {
  known_dataset: "Known dataset",
  cross_dataset: "Cross dataset",
  ablation: "Ablation",
  multimodal: "Multimodal",
};

export function Experiments() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Research"
        title="Experiments"
        description="Declared experiment records and the known-versus-unseen comparison."
        actions={<DemoBadge label="NO RUNS EXECUTED" />}
      />

      <div className="state-banner state-banner--warning" role="note">
        No experiment has been executed. Every record below is a planning
        placeholder; seeds, config hashes, and metrics are intentionally empty.
      </div>

      <Card title="Experiment records" subtitle="All statuses are 'planned'">
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Experiment</th>
                <th scope="col">Family</th>
                <th scope="col">Dataset</th>
                <th scope="col">Status</th>
                <th scope="col">Seed</th>
                <th scope="col">Config hash</th>
              </tr>
            </thead>
            <tbody>
              {DEMO_EXPERIMENTS.map((experiment) => (
                <tr key={experiment.id}>
                  <td>
                    <span className="table__subject">{experiment.name}</span>
                    <span className="table__sub muted">{experiment.notes}</span>
                  </td>
                  <td>{FAMILY_LABEL[experiment.family] ?? experiment.family}</td>
                  <td className="muted">{experiment.dataset}</td>
                  <td>
                    <Badge tone={STATUS_TONE[experiment.status]}>
                      {experiment.status}
                    </Badge>
                  </td>
                  <td className="mono">{experiment.seed ?? "—"}</td>
                  <td className="mono">{experiment.configHash ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid grid--2">
        <Card
          title="Known vs. unseen"
          subtitle="In-domain against cross-domain performance"
          actions={<DemoBadge label="NOT MEASURED" />}
        >
          <PerformanceChart data={DEMO_DATASET_COMPARISON} />
          <p className="table-note">
            Empty bars are correct: no metric has been measured, so every value is
            null rather than zero.
          </p>
        </Card>

        <Card title="Metric table" subtitle="Explicit nulls until runs exist">
          <div className="table-wrap">
            <table className="table table--compact">
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  <th scope="col">Known</th>
                  <th scope="col">Unseen</th>
                </tr>
              </thead>
              <tbody>
                {DEMO_DATASET_COMPARISON.map((row) => (
                  <tr key={row.metric}>
                    <td>{row.label}</td>
                    <td className="mono">{formatNumber(row.known)}</td>
                    <td className="mono">{formatNumber(row.unseen)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
