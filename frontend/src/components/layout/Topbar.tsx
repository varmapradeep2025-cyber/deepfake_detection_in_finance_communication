import { Menu, RotateCw } from "lucide-react";
import { useLocation } from "react-router-dom";
import { NAV_ITEMS } from "../../navigation";
import { useBackendStatus } from "../../context/backendStatus";
import { ConnectionStatePill } from "../ui/ConnectionStatePill";

interface TopbarProps {
  onMenu: () => void;
}

export function Topbar({ onMenu }: TopbarProps) {
  const location = useLocation();
  const { status, refresh } = useBackendStatus();

  const active =
    NAV_ITEMS.find((item) =>
      item.end
        ? location.pathname === item.path
        : location.pathname.startsWith(item.path),
    ) ?? NAV_ITEMS[0];

  return (
    <header className="topbar">
      <button
        type="button"
        className="topbar__menu btn btn--ghost"
        onClick={onMenu}
        aria-label="Toggle navigation"
      >
        <Menu size={18} aria-hidden />
      </button>

      <div className="topbar__heading">
        <span className="topbar__eyebrow">Console</span>
        <span className="topbar__title">{active.label}</span>
      </div>

      <div className="topbar__actions">
        <ConnectionStatePill state={status.state} />
        <button
          type="button"
          className="btn btn--ghost"
          onClick={refresh}
          aria-label="Re-check backend status"
        >
          <RotateCw size={15} aria-hidden />
          <span className="btn__text">Re-check</span>
        </button>
      </div>
    </header>
  );
}
