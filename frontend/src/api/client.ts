import type { BackendStatus } from "../types";
import { API_BASE_URL, API_PATHS, API_TIMEOUT_MS } from "./config";

export type ApiErrorKind = "timeout" | "network" | "http" | "parse";

export type ApiResult<T> =
  | { ok: true; data: T }
  | { ok: false; error: string; kind: ApiErrorKind; status?: number };

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  const url = `${API_BASE_URL}${path}`;

  try {
    const response = await fetch(url, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        // Only set a JSON content type for string bodies. FormData must be left
        // alone so the browser adds its own multipart boundary.
        ...(typeof init.body === "string"
          ? { "Content-Type": "application/json" }
          : {}),
        ...init.headers,
      },
    });

    if (!response.ok) {
      return {
        ok: false,
        kind: "http",
        status: response.status,
        error: `Backend responded with HTTP ${response.status}.`,
      };
    }

    try {
      const data = (await response.json()) as T;
      return { ok: true, data };
    } catch {
      return { ok: false, kind: "parse", error: "Response was not valid JSON." };
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return {
        ok: false,
        kind: "timeout",
        error: `No response within ${API_TIMEOUT_MS} ms.`,
      };
    }
    return {
      ok: false,
      kind: "network",
      error: "Could not reach the backend service.",
    };
  } finally {
    clearTimeout(timer);
  }
}

interface HealthPayload {
  status?: string;
  version?: string;
}

/**
 * Best-effort backend health probe. Returns a status object the UI can render
 * even when the service does not exist yet; this never throws.
 */
export async function checkBackendStatus(): Promise<BackendStatus> {
  const checkedAt = new Date().toISOString();
  const result = await request<HealthPayload>(API_PATHS.health, {
    method: "GET",
  });

  if (result.ok) {
    const healthy = (result.data.status ?? "ok").toLowerCase() === "ok";
    return {
      state: healthy ? "connected" : "unknown",
      baseUrl: API_BASE_URL,
      version: result.data.version ?? null,
      detail: healthy
        ? "Backend health endpoint reachable."
        : "Backend reachable but reported a non-ok status.",
      checkedAt,
    };
  }

  return {
    state: "disconnected",
    baseUrl: API_BASE_URL,
    version: null,
    detail: `${result.error} The dashboard continues in DEMO MODE.`,
    checkedAt,
  };
}

/** Generic typed helpers for future endpoint wiring. */
export const apiClient = {
  get: <T>(path: string) => request<T>(path, { method: "GET" }),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
  /** POST multipart/form-data. The browser sets the boundary from the FormData. */
  postForm: <T>(path: string, form: FormData) =>
    request<T>(path, { method: "POST", body: form }),
  baseUrl: API_BASE_URL,
};
