import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CalibrationPoint } from "../../types";

interface Props {
  data: CalibrationPoint[];
}

export function CalibrationChart({ data }: Props) {
  return (
    <div className="chart" style={{ height: 260 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="bucket"
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
            contentStyle={{
              background: "var(--surface-2)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              color: "var(--text)",
            }}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            name="Predicted"
            stroke="var(--accent)"
            strokeDasharray="4 4"
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="observed"
            name="Observed (ideal)"
            stroke="var(--risk-low)"
            dot={{ r: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
