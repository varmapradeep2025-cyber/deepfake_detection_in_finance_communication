export type ConnectionState =
  | "connected"
  | "disconnected"
  | "checking"
  | "unknown";

export interface ModelAvailability {
  id: string;
  name: string;
  modality: string;
  /** Never a fabricated "ready"; reflects declared state only. */
  state: "not_trained" | "unavailable" | "declared";
  detail: string;
}

export interface BackendStatus {
  state: ConnectionState;
  baseUrl: string;
  version: string | null;
  detail: string;
  checkedAt: string | null;
}
