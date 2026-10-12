import { Info, Link2 } from "lucide-react";
import { PageHeader } from "../components/ui/PageHeader";
import { Card } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { DemoBadge } from "../components/ui/DemoBadge";
import { DEMO_CAMPAIGNS, DEMO_INDICATORS } from "../data";
import { formatDate, formatPercent } from "../lib/format";

export function ThreatIntelligence() {
  const indicatorById = new Map(DEMO_INDICATORS.map((i) => [i.id, i]));

  return (
    <div className="page">
      <PageHeader
        eyebrow="Correlation"
        title="Threat Intelligence"
        description="Illustrative indicators and campaign correlations for layout demonstration."
        actions={<DemoBadge label="EXAMPLE DATA" />}
      />

      <div className="state-banner state-banner--info" role="note">
        <Info size={15} aria-hidden />
        <span>
          No live lookup is performed. Threat-intelligence collection is disabled
          by default in this repository, and the indicators below are fabricated
          examples — not results from any feed.
        </span>
      </div>

      <Card
        title="Indicators (examples)"
        subtitle="Fabricated values for interface demonstration only"
        actions={<DemoBadge label="EXAMPLE DATA" />}
      >
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Kind</th>
                <th scope="col">Value</th>
                <th scope="col">Confidence</th>
                <th scope="col">First seen</th>
                <th scope="col">Source</th>
                <th scope="col">Tags</th>
              </tr>
            </thead>
            <tbody>
              {DEMO_INDICATORS.map((indicator) => (
                <tr key={indicator.id}>
                  <td className="capitalize">{indicator.kind.replace("_", " ")}</td>
                  <td className="mono table__subject" title={indicator.value}>
                    {indicator.value}
                  </td>
                  <td>{formatPercent(indicator.confidence, 0)}</td>
                  <td className="muted">{formatDate(indicator.firstSeen)}</td>
                  <td className="muted">{indicator.source}</td>
                  <td>
                    <div className="tag-row">
                      {indicator.tags.map((tag) => (
                        <Badge key={tag} tone="neutral">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card
        title="Campaign correlations (examples)"
        subtitle="Illustrative groupings, not verified campaigns"
        actions={<DemoBadge label="EXAMPLE DATA" />}
      >
        <div className="campaign-grid">
          {DEMO_CAMPAIGNS.map((campaign) => (
            <article key={campaign.id} className="campaign">
              <div className="campaign__head">
                <h3 className="campaign__name">{campaign.name}</h3>
                <Badge tone="elevated">{formatPercent(campaign.confidence, 0)}</Badge>
              </div>
              <p className="campaign__narrative">{campaign.narrative}</p>
              <dl className="kv kv--inline">
                <div className="kv__row">
                  <dt className="kv__key">Samples</dt>
                  <dd className="kv__value">{campaign.sampleCount}</dd>
                </div>
                <div className="kv__row">
                  <dt className="kv__key">First seen</dt>
                  <dd className="kv__value">{formatDate(campaign.firstSeen)}</dd>
                </div>
              </dl>
              <ul className="campaign__links">
                {campaign.indicatorIds.map((id) => {
                  const indicator = indicatorById.get(id);
                  if (!indicator) return null;
                  return (
                    <li key={id}>
                      <Link2 size={12} aria-hidden />
                      <span className="mono">{indicator.value}</span>
                    </li>
                  );
                })}
              </ul>
            </article>
          ))}
        </div>
      </Card>
    </div>
  );
}
