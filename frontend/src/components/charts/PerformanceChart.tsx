import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DatasetComparisonRow } from "../../types";

interface Props {
  data: DatasetComparisonRow[];
}

/**
 * Known-vs-unseen metric comparison. Values are null until a real run exists,
 * so the chart renders with empty bars and the page states that explicitly.
 */
export function PerformanceChart({ data }: Props) {
  const chartData = data.map((row) => ({
    metric: row.label,
    Known: row.known,
    Unseen: row.unseen,
  }));

  return (
    <div className="chart" style={{ height: 280 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="metric"
            stroke="var(--text-muted)"
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            fontSize={12}
          />
          <YAxis
            domain={[0, 1]}
            stroke="var(--text-muted)"
            tickLine={false}
            axisLine={false}
            fontSize={12}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              color: "var(--text)",
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12, color: "var(--text-muted)" }} />
          <Bar dataKey="Known" fill="var(--accent)" radius={[4, 4, 0, 0]} />
          <Bar
            dataKey="Unseen"
            fill="var(--risk-elevated)"
            radius={[4, 4, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
