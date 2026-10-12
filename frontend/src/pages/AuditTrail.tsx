import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge, type BadgeTone } from "../components/ui/Badge";
import { DemoBadge } from "../components/ui/DemoBadge";
import { DEMO_AUDIT_EVENTS } from "../data";
import { formatDateTime } from "../lib/format";
import type { AuditSeverity } from "../types";

const SEVERITY_TONE: Record<AuditSeverity, BadgeTone> = {
  info: "neutral",
  warning: "elevated",
  critical: "high",
};

export function AuditTrail() {
  return (
    <div className="page">
      <PageHeader
        eyebrow="Governance"
        title="Audit Trail"
        description="Timestamped record of interface events. All entries are demo data."
        actions={<DemoBadge label="DEMO DATA" />}
      />

      <Card
        title="Events"
        subtitle="Append-only style list, newest context first"
      >
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Timestamp</th>
                <th scope="col">Actor</th>
                <th scope="col">Action</th>
                <th scope="col">Target</th>
                <th scope="col">Severity</th>
                <th scope="col">Detail</th>
                <th scope="col">Hash</th>
              </tr>
            </thead>
            <tbody>
              {DEMO_AUDIT_EVENTS.map((event) => (
                <tr key={event.id}>
                  <td className="muted">{formatDateTime(event.timestamp)}</td>
                  <td>{event.actor}</td>
                  <td className="mono">{event.action}</td>
                  <td className="table__subject" title={event.target}>
                    {event.target}
                  </td>
                  <td>
                    <Badge tone={SEVERITY_TONE[event.severity]}>
                      {event.severity}
                    </Badge>
                  </td>
                  <td className="muted">{event.detail}</td>
                  <td className="mono muted">{event.hash ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-note">
          <DemoBadge label="DEMO DATA" /> No real session, user, or case activity
          is recorded in this build.
        </p>
      </Card>
    </div>
  );
}
