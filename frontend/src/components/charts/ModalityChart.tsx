import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import type { ModalityBreakdownPoint } from "../../types";

const COLORS = [
  "var(--accent)",
  "#7c9cf5",
  "#5ec8c0",
  "var(--risk-elevated)",
  "var(--risk-high)",
];

interface Props {
  data: ModalityBreakdownPoint[];
}

export function ModalityChart({ data }: Props) {
  return (
    <div className="chart" style={{ height: 240 }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="label"
            innerRadius={52}
            outerRadius={80}
            paddingAngle={2}
            stroke="var(--surface-1)"
          >
            {data.map((entry, index) => (
              <Cell key={entry.modality} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              color: "var(--text)",
            }}
            formatter={(value: number, name: string) => [
              `${value} analyses`,
              name,
            ]}
          />
        </PieChart>
      </ResponsiveContainer>
      <ul className="chart-legend">
        {data.map((entry, index) => (
          <li key={entry.modality}>
            <span
              className="chart-legend__dot"
              style={{ background: COLORS[index % COLORS.length] }}
            />
            {entry.label}
          </li>
        ))}
      </ul>
    </div>
  );
}
