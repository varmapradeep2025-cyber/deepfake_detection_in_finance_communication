import { RefreshCw } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge, type BadgeTone } from "../components/ui/Badge";
import { DemoBadge } from "../components/ui/DemoBadge";
import { ConnectionStatePill } from "../components/ui/ConnectionStatePill";
import { useBackendStatus } from "../context/backendStatus";
import { API_BASE_URL, API_PATHS, API_TIMEOUT_MS } from "../api/config";
import { DEMO_MODELS } from "../data";
import { formatDateTime } from "../lib/format";
import type { ModelAvailability } from "../types";

const MODEL_TONE: Record<
  ModelAvailability["state"],
  BadgeTone
> = {
  not_trained: "warning",
  unavailable: "high",
  declared: "neutral",
};

const CONFIG_ROWS = [
  { key: "Backend base URL", value: API_BASE_URL },
  { key: "Health endpoint", value: API_PATHS.health },
  { key: "Request timeout", value: `${API_TIMEOUT_MS} ms` },
  { key: "Frontend dev port", value: "5173 (configs/project.yaml)" },
  { key: "Project stage", value: "foundation" },
  { key: "App version", value: "0.1.0" },
];

export function Settings() {
  const { status, refresh } = useBackendStatus();

  return (
    <div className="page">
      <PageHeader
        eyebrow="Configuration"
        title="Settings"
        description="Application configuration and model-connection status."
        actions={<DemoBadge />}
      />

      <div className="grid grid--2">
        <Card title="Application configuration" subtitle="Read-only in this build">
          <dl className="kv">
            {CONFIG_ROWS.map((row) => (
              <div className="kv__row" key={row.key}>
                <dt className="kv__key">{row.key}</dt>
                <dd className="kv__value mono">{row.value}</dd>
              </div>
            ))}
          </dl>
          <p className="table-note">
            Values reflect repository configuration. No secret or credential is
            stored in the frontend.
          </p>
        </Card>

        <Card
          title="Backend connection"
          subtitle="Local Python service (stdlib http.server), probed via GET /health"
          actions={
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={refresh}
            >
              <RefreshCw size={14} aria-hidden /> Re-check
            </button>
          }
        >
          <ConnectionStatePill state={status.state} />
          <dl className="kv">
            <div className="kv__row">
              <dt className="kv__key">State</dt>
              <dd className="kv__value capitalize">{status.state}</dd>
            </div>
            <div className="kv__row">
              <dt className="kv__key">Version</dt>
              <dd className="kv__value mono">{status.version ?? "—"}</dd>
            </div>
            <div className="kv__row">
              <dt className="kv__key">Last checked</dt>
              <dd className="kv__value">
                {status.checkedAt ? formatDateTime(status.checkedAt) : "—"}
              </dd>
            </div>
          </dl>
          <div className="state-banner state-banner--info" role="status">
            {status.detail}
          </div>
          <p className="table-note">
            The connection check calls the real{" "}
            <code className="mono">GET /health</code> endpoint; nothing is shown
            as connected unless that probe succeeds. The service is implemented
            in <code className="mono">backend/</code> with Python's
            standard-library HTTP server and is disabled by default in
            configs/project.yaml. Start it locally with{" "}
            <code className="mono">python -m backend</code>.
          </p>
        </Card>
      </div>

      <Card
        title="Model connection status"
        subtitle="Declared slots; none is trained or reachable"
        actions={<DemoBadge label="NO MODELS" />}
      >
        <ul className="model-list">
          {DEMO_MODELS.map((model) => (
            <li key={model.id} className="model-list__item">
              <div>
                <span className="model-list__name">{model.name}</span>
                <span className="model-list__detail">{model.detail}</span>
              </div>
              <div className="model-list__meta">
                <span className="muted">{model.modality}</span>
                <Badge tone={MODEL_TONE[model.state]}>
                  {model.state.replace("_", " ")}
                </Badge>
              </div>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
