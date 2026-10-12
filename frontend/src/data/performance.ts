export interface MetricCard {
  key: string;
  label: string;
  /** null means "not measured", rendered as "—", never as 0. */
  value: number | null;
  format: "percent" | "number";
  direction: "higher-better" | "lower-better";
  note: string;
}

/**
 * Metric placeholders. The repository's contract is explicit: an unmeasured
 * metric is null, never zero. No model has been trained, so all are null.
 */
export const DEMO_METRIC_CARDS: MetricCard[] = [
  {
    key: "accuracy",
    label: "Accuracy",
    value: null,
    format: "percent",
    direction: "higher-better",
    note: "Not measured — no trained model.",
  },
  {
    key: "precision",
    label: "Precision",
    value: null,
    format: "percent",
    direction: "higher-better",
    note: "Not measured.",
  },
  {
    key: "recall",
    label: "Recall",
    value: null,
    format: "percent",
    direction: "higher-better",
    note: "Not measured.",
  },
  {
    key: "f1",
    label: "F1",
    value: null,
    format: "percent",
    direction: "higher-better",
    note: "Not measured.",
  },
  {
    key: "ece",
    label: "Calibration (ECE)",
    value: null,
    format: "number",
    direction: "lower-better",
    note: "Calibration is measured, never assumed.",
  },
  {
    key: "generalization_gap",
    label: "Generalization gap",
    value: null,
    format: "number",
    direction: "lower-better",
    note: "Requires both in-domain and cross-domain results.",
  },
];

export interface PerformanceSeriesPoint {
  dataset: string;
  accuracy: number | null;
  f1: number | null;
}

/** Known-vs-unseen series, all null until runs exist. */
export const DEMO_PERFORMANCE_SERIES: PerformanceSeriesPoint[] = [
  { dataset: "Known (in-domain)", accuracy: null, f1: null },
  { dataset: "Unseen (cross-domain)", accuracy: null, f1: null },
];
