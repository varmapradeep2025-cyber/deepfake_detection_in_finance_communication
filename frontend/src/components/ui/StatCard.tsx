import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: ReactNode;
  meta?: ReactNode;
  tone?: "default" | "low" | "elevated" | "high";
}

export function StatCard({ label, value, meta, tone = "default" }: StatCardProps) {
  return (
    <div className={`stat stat--${tone}`}>
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
      {meta && <span className="stat__meta">{meta}</span>}
    </div>
  );
}
