import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { checkBackendStatus } from "../api/client";
import { API_BASE_URL } from "../api/config";
import type { BackendStatus } from "../types";
import {
  BackendStatusContext,
  type BackendStatusContextValue,
} from "./backendStatus";

const INITIAL: BackendStatus = {
  state: "checking",
  baseUrl: API_BASE_URL,
  version: null,
  detail: "Checking backend availability…",
  checkedAt: null,
};

export function BackendStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<BackendStatus>(INITIAL);

  const refresh = useCallback(() => {
    setStatus((current) => ({ ...current, state: "checking" }));
    void checkBackendStatus().then(setStatus);
  }, []);

  useEffect(() => {
    // Probe once on mount. Failure is expected and handled, never thrown.
    refresh();
  }, [refresh]);

  const value = useMemo<BackendStatusContextValue>(
    () => ({
      status,
      refresh,
      demoMode: status.state === "disconnected",
    }),
    [status, refresh],
  );

  return (
    <BackendStatusContext.Provider value={value}>
      {children}
    </BackendStatusContext.Provider>
  );
}
