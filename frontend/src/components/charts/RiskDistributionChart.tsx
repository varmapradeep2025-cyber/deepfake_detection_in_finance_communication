import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RiskDistributionPoint } from "../../types";

interface Props {
  data: RiskDistributionPoint[];
}

export function RiskDistributionChart({ data }: Props) {
  return (
    <div className="chart" style={{ height: 240 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <XAxis
            dataKey="label"
            stroke="var(--text-muted)"
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            fontSize={12}
          />
          <YAxis
            stroke="var(--text-muted)"
            tickLine={false}
            axisLine={false}
            allowDecimals={false}
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
            formatter={(value: number) => [`${value} analyses`, "Count"]}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {data.map((entry) => (
              <Cell key={entry.risk} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
