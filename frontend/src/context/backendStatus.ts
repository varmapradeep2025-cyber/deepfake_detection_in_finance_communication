import { createContext, useContext } from "react";
import type { BackendStatus } from "../types";

export interface BackendStatusContextValue {
  status: BackendStatus;
  refresh: () => void;
  demoMode: boolean;
}

export const BackendStatusContext =
  createContext<BackendStatusContextValue | null>(null);

export function useBackendStatus(): BackendStatusContextValue {
  const context = useContext(BackendStatusContext);
  if (!context) {
    throw new Error(
      "useBackendStatus must be used within a BackendStatusProvider",
    );
  }
  return context;
}
