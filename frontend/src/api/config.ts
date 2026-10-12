/**
 * API configuration.
 *
 * Values mirror `configs/project.yaml`:
 *   services.api.host: 127.0.0.1
 *   services.api.port: 8000
 *   services.frontend.dev_port: 5173
 *
 * The backend is a standard-library (`http.server`) service in `backend/`. It is
 * disabled by default in `configs/project.yaml` and started locally with
 * `python -m backend`. The default below matches its loopback bind and can be
 * overridden with VITE_API_BASE_URL.
 */
const DEFAULT_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_TIMEOUT_MS = 2500;

export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_BASE_URL
).replace(/\/+$/, "");

function parseTimeout(raw: string | undefined): number {
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMEOUT_MS;
}

export const API_TIMEOUT_MS: number = parseTimeout(
  import.meta.env.VITE_API_TIMEOUT_MS,
);

/**
 * Endpoint paths for the local inference service in `backend/`.
 *
 * `/health` and `/analyses` are served by the standard-library HTTP service;
 * `/reports/multimodal` is a provisional contract with no server route yet. The
 * typed client in `api/modalities.ts` targets these paths and surfaces failures
 * honestly rather than inventing a result.
 */
export const API_PATHS = {
  health: "/health",
  analyses: "/analyses",
  multimodalReports: "/reports/multimodal",
} as const;

/** Path for one stored analysis record. */
export function analysisPath(reportId: string): string {
  return `${API_PATHS.analyses}/${encodeURIComponent(reportId)}`;
}
