import type { Modality, ModalityMeta } from "../types";

export const MEGABYTE = 1024 * 1024;

export const MODALITY_META: Record<Modality, ModalityMeta> = {
  image: {
    id: "image",
    label: "Image",
    accept: ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"],
    extensions: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"],
    maxBytes: 25 * MEGABYTE,
    description:
      "Still images, press-release graphics, or extracted frames. Detector: ml/image (not trained).",
  },
  video: {
    id: "video",
    label: "Video",
    accept: ["video/mp4", "video/quicktime", "video/x-matroska", "video/webm"],
    extensions: [".mp4", ".mov", ".mkv", ".webm", ".avi"],
    maxBytes: 250 * MEGABYTE,
    description:
      "Conference or broadcast clips. Detector: ml/video (not trained).",
  },
  audio: {
    id: "audio",
    label: "Audio",
    accept: ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "audio/mp4"],
    extensions: [".mp3", ".wav", ".flac", ".m4a", ".ogg"],
    maxBytes: 100 * MEGABYTE,
    description:
      "Voice notes and call recordings. Detector: ml/audio (not trained).",
  },
  qr: {
    id: "qr",
    label: "QR code",
    accept: ["image/png", "image/jpeg", "image/webp"],
    extensions: [".png", ".jpg", ".jpeg", ".webp"],
    maxBytes: 10 * MEGABYTE,
    description:
      "Payment or onboarding QR images. Decoding is not implemented; demo only.",
  },
  url: {
    id: "url",
    label: "URL / phishing",
    accept: [],
    extensions: [],
    maxBytes: 0,
    description:
      "Suspicious links from email or chat. No network lookup is performed.",
  },
};

export function getModalityMeta(modality: Modality): ModalityMeta {
  return MODALITY_META[modality];
}

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf(".");
  if (dot < 0) return "";
  return filename.slice(dot).toLowerCase();
}

/**
 * Validates a selected file against the chosen modality. Declared MIME type is
 * advisory; the extension is authoritative for the client-side gate. This is an
 * input check only — it says nothing about the file's real content.
 */
export function validateFile(file: File, modality: Modality): ValidationResult {
  const meta = MODALITY_META[modality];
  if (meta.maxBytes === 0) {
    return { valid: false, error: `${meta.label} analysis does not accept files.` };
  }
  if (file.size === 0) {
    return { valid: false, error: "The selected file is empty." };
  }
  if (file.size > meta.maxBytes) {
    const limitMb = Math.round(meta.maxBytes / MEGABYTE);
    return {
      valid: false,
      error: `File is ${(file.size / MEGABYTE).toFixed(
        1,
      )} MB; the ${limitMb} MB limit for ${meta.label.toLowerCase()} was exceeded.`,
    };
  }
  const ext = extensionOf(file.name);
  if (!ext || !meta.extensions.includes(ext)) {
    return {
      valid: false,
      error: `Unsupported extension "${ext || "none"}" for ${meta.label.toLowerCase()}. Allowed: ${meta.extensions.join(
        ", ",
      )}.`,
    };
  }
  return { valid: true };
}

/**
 * Validates a URL string. Only http/https are accepted and no request is made,
 * so a syntactically valid URL here is not asserted to be reachable or benign.
 */
export function validateUrl(raw: string): ValidationResult {
  const value = raw.trim();
  if (!value) {
    return { valid: false, error: "Enter a URL to analyse." };
  }
  if (value.length > 2048) {
    return { valid: false, error: "URL exceeds 2048 characters." };
  }
  if (/\s/.test(value)) {
    return { valid: false, error: "URL must not contain spaces." };
  }
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    return {
      valid: false,
      error: "Not a valid URL. Include a scheme, e.g. https://example.com/path.",
    };
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return {
      valid: false,
      error: `Unsupported scheme "${parsed.protocol}". Only http and https are accepted.`,
    };
  }
  if (!parsed.hostname || !parsed.hostname.includes(".")) {
    return { valid: false, error: "URL hostname is missing a domain." };
  }
  return { valid: true };
}
