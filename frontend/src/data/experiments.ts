import type {
  CalibrationPoint,
  DatasetComparisonRow,
  ExperimentRecord,
} from "../types";

/**
 * Illustrative experiment records. No experiment has been executed; every
 * status here is "planned" and every metric is null pending a real run.
 */
export const DEMO_EXPERIMENTS: ExperimentRecord[] = [
  {
    id: "exp-image-known",
    name: "Image deepfake — known dataset baseline",
    family: "known_dataset",
    dataset: "faceforensics_plus_plus (not acquired)",
    status: "planned",
    seed: null,
    configHash: null,
    notes: "Awaiting licence verification and acquisition; no run scheduled.",
    demo: true,
  },
  {
    id: "exp-video-known",
    name: "Video deepfake — known dataset baseline",
    family: "known_dataset",
    dataset: "faceforensics_plus_plus (not acquired)",
    status: "planned",
    seed: null,
    configHash: null,
    notes: "Depends on frozen split; none exists.",
    demo: true,
  },
  {
    id: "exp-audio-cross",
    name: "Audio cross-dataset generalization",
    family: "cross_dataset",
    dataset: "asvspoof_2021_df → fakeavceleb (planned)",
    status: "planned",
    seed: null,
    configHash: null,
    notes: "Both datasets unverified; claim cannot be earned yet.",
    demo: true,
  },
  {
    id: "exp-multimodal",
    name: "Multimodal evidence fusion",
    family: "multimodal",
    dataset: "image + audio + video",
    status: "planned",
    seed: null,
    configHash: null,
    notes: "Ablation required before any headline number is reported.",
    demo: true,
  },
  {
    id: "exp-ablation-image",
    name: "Image detector component ablation",
    family: "ablation",
    dataset: "faceforensics_plus_plus (not acquired)",
    status: "planned",
    seed: null,
    configHash: null,
    notes: "Ablation plan drafted; no model exists to ablate.",
    demo: true,
  },
];

/** Known-vs-unseen comparison. All values null: nothing has been measured. */
export const DEMO_DATASET_COMPARISON: DatasetComparisonRow[] = [
  { metric: "accuracy", label: "Accuracy", known: null, unseen: null },
  { metric: "precision", label: "Precision", known: null, unseen: null },
  { metric: "recall", label: "Recall", known: null, unseen: null },
  { metric: "f1", label: "F1", known: null, unseen: null },
  { metric: "auc", label: "AUC", known: null, unseen: null },
  { metric: "eer", label: "EER", known: null, unseen: null },
];

/** Reliability-diagram placeholders; no calibration has been performed. */
export const DEMO_CALIBRATION: CalibrationPoint[] = [
  { bucket: "0.0–0.2", predicted: 0.1, observed: 0.1 },
  { bucket: "0.2–0.4", predicted: 0.3, observed: 0.3 },
  { bucket: "0.4–0.6", predicted: 0.5, observed: 0.5 },
  { bucket: "0.6–0.8", predicted: 0.7, observed: 0.7 },
  { bucket: "0.8–1.0", predicted: 0.9, observed: 0.9 },
];
