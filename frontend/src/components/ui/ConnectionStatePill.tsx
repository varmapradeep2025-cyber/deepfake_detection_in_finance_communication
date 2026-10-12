import { CircleAlert, CircleCheck, LoaderCircle, Plug } from "lucide-react";
import type { ConnectionState } from "../../types";

const CONFIG: Record<
  ConnectionState,
  { label: string; className: string; Icon: typeof Plug }
> = {
  connected: { label: "Backend connected", className: "ok", Icon: CircleCheck },
  disconnected: {
    label: "Backend unavailable — demo mode",
    className: "warn",
    Icon: CircleAlert,
  },
  checking: { label: "Checking backend…", className: "muted", Icon: LoaderCircle },
  unknown: { label: "Backend status unknown", className: "muted", Icon: Plug },
};

export function ConnectionStatePill({ state }: { state: ConnectionState }) {
  const { label, className, Icon } = CONFIG[state];
  return (
    <span className={`status-pill status-pill--${className}`}>
      <Icon size={14} aria-hidden />
      {label}
    </span>
  );
}
