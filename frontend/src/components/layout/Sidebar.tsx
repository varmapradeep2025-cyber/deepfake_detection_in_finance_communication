import { NavLink } from "react-router-dom";
import { Fingerprint } from "lucide-react";
import { NAV_ITEMS } from "../../navigation";
import { DemoBadge } from "../ui/DemoBadge";

interface SidebarProps {
  open: boolean;
  onNavigate: () => void;
}

export function Sidebar({ open, onNavigate }: SidebarProps) {
  return (
    <aside className={`sidebar${open ? " sidebar--open" : ""}`}>
      <div className="sidebar__brand">
        <span className="sidebar__logo" aria-hidden>
          <Fingerprint size={20} />
        </span>
        <div>
          <span className="sidebar__title">Deepfake Forensics</span>
          <span className="sidebar__subtitle">Financial Communication</span>
        </div>
      </div>

      <nav className="sidebar__nav" aria-label="Primary">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.end}
              onClick={onNavigate}
              className={({ isActive }) =>
                `nav-item${isActive ? " nav-item--active" : ""}`
              }
            >
              <span className="nav-item__icon" aria-hidden>
                <Icon size={16} />
              </span>
              <span className="nav-item__label">{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      <div className="sidebar__footer">
        <DemoBadge />
        <p className="sidebar__note">
          This build ships sample data only. No model is trained and no live
          service is queried.
        </p>
      </div>
    </aside>
  );
}
