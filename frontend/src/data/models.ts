import type { ModelAvailability } from "../types";

/**
 * Declared model slots. None is trained or available; the UI must never imply
 * otherwise. States are limited to the repository's honest vocabulary.
 */
export const DEMO_MODELS: ModelAvailability[] = [
  {
    id: "image",
    name: "Image detector",
    modality: "Image",
    state: "not_trained",
    detail: "ml/image is a declared placeholder; no checkpoint exists.",
  },
  {
    id: "video",
    name: "Video detector",
    modality: "Video",
    state: "not_trained",
    detail: "ml/video depends on the image detector; not implemented.",
  },
  {
    id: "audio",
    name: "Audio detector",
    modality: "Audio",
    state: "not_trained",
    detail: "ml/audio is a declared placeholder; no checkpoint exists.",
  },
  {
    id: "fusion",
    name: "Multimodal fusion",
    modality: "Cross-modal",
    state: "unavailable",
    detail: "ml/fusion requires constituent modality models.",
  },
  {
    id: "risk",
    name: "Risk & uncertainty",
    modality: "Cross-modal",
    state: "declared",
    detail: "ml/risk scaffolding declared; calibration not implemented.",
  },
];
