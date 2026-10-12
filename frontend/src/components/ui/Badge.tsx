import type { ReactNode } from "react";

export type BadgeTone =
  | "neutral"
  | "low"
  | "elevated"
  | "high"
  | "info"
  | "warning"
  | "success"
  | "critical";

interface BadgeProps {
  tone?: BadgeTone;
  children: ReactNode;
}

export function Badge({ tone = "neutral", children }: BadgeProps) {
  return <span className={`badge badge--${tone}`}>{children}</span>;
}
