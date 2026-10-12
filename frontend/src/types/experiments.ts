export type ExperimentFamily =
  | "known_dataset"
  | "cross_dataset"
  | "ablation"
  | "multimodal";

export type ExperimentStatus =
  | "planned"
  | "running"
  | "completed"
  | "aborted";

export interface ExperimentRecord {
  id: string;
  name: string;
  family: ExperimentFamily;
  /** Dataset or source->target pair description. */
  dataset: string;
  status: ExperimentStatus;
  seed: number | null;
  /** Short config hash; null until the run is pinned to a config. */
  configHash: string | null;
  notes: string;
  demo: true;
}

export interface DatasetComparisonRow {
  metric: string;
  label: string;
  /** Known (in-domain) split. Null when not measured. */
  known: number | null;
  /** Unseen (cross-domain) split. Null when not measured. */
  unseen: number | null;
}

export interface CalibrationPoint {
  bucket: string;
  predicted: number;
  observed: number;
}
