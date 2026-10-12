import type { LucideIcon } from "lucide-react";
import {
  Activity,
  BarChart3,
  ClipboardCheck,
  FileText,
  FlaskConical,
  LayoutDashboard,
  Microscope,
  ScanSearch,
  Settings as SettingsIcon,
  ShieldAlert,
} from "lucide-react";

export interface NavItem {
  path: string;
  label: string;
  description: string;
  icon: LucideIcon;
  /** End matching for the index route. */
  end?: boolean;
}

/** Single source of truth for sidebar navigation and the router. */
export const NAV_ITEMS: NavItem[] = [
  {
    path: "/",
    label: "Overview",
    description: "Analysis overview, modality mix, and recent activity.",
    icon: LayoutDashboard,
    end: true,
  },
  {
    path: "/image-analysis",
    label: "Image Analysis",
    description: "Image forensics analysis workflow.",
    icon: FileText,
  },
  {
    path: "/video-analysis",
    label: "Video Analysis",
    description: "Video forensics analysis workflow.",
    icon: FileText,
  },
  {
    path: "/audio-analysis",
    label: "Audio Analysis",
    description: "Audio forensics analysis workflow.",
    icon: FileText,
  },
  {
    path: "/qr-url-analysis",
    label: "QR & URL Analysis",
    description: "QR code decoding and URL intelligence.",
    icon: ShieldAlert,
  },
  {
    path: "/analysis",
    label: "Multimodal Analysis",
    description: "Image, video, audio, QR, and URL analysis workflows.",
    icon: ScanSearch,
  },
  {
    path: "/modalities",
    label: "Modality Analysis",
    description: "Run each modality through the inference library and build a report.",
    icon: Microscope,
  },
  {
    path: "/evidence",
    label: "Evidence & Provenance",
    description: "Hashes, metadata, and chain-of-custody timeline.",
    icon: ClipboardCheck,
  },
  {
    path: "/threat-intelligence",
    label: "Threat Intelligence",
    description: "Illustrative indicators and campaign correlations.",
    icon: ShieldAlert,
  },
  {
    path: "/reports",
    label: "Reports",
    description: "Forensic report preview and demo export.",
    icon: FileText,
  },
  {
    path: "/experiments",
    label: "Experiments",
    description: "Experiment records and known-vs-unseen comparison.",
    icon: FlaskConical,
  },
  {
    path: "/dataset-performance",
    label: "Dataset Performance",
    description: "Accuracy, precision, recall, F1, calibration, and gap.",
    icon: BarChart3,
  },
  {
    path: "/audit-trail",
    label: "Audit Trail",
    description: "Timestamped demo events and severity.",
    icon: Activity,
  },
  {
    path: "/settings",
    label: "Settings",
    description: "Application configuration and model-connection status.",
    icon: SettingsIcon,
  },
];
