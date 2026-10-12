import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach, vi } from "vitest";

// jsdom does not provide ResizeObserver, which Recharts' ResponsiveContainer
// needs. Installed once and never unstubbed.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver =
  ResizeObserverStub;

beforeEach(() => {
  // No network in tests: the backend probe must resolve to "disconnected".
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.reject(new TypeError("Network disabled in tests"))),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});
