export type AuditSeverity = "info" | "warning" | "critical";

export interface AuditEvent {
  id: string;
  timestamp: string;
  actor: string;
  action: string;
  target: string;
  severity: AuditSeverity;
  detail: string;
  /** Null when the event is not part of the hash chain in this demo. */
  hash: string | null;
}
