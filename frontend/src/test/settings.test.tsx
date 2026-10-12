import { describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { Settings } from "../pages/Settings";
import { renderWithProviders } from "./render";

function jsonResponse(payload: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => payload,
  } as Response;
}

describe("settings backend connection", () => {
  it("describes the implemented standard-library service, not a FastAPI stub", () => {
    renderWithProviders(<Settings />);

    expect(screen.queryByText(/FastAPI/i)).not.toBeInTheDocument();
    expect(screen.getByText(/http\.server/i)).toBeInTheDocument();
    expect(screen.getByText(/python -m backend/i)).toBeInTheDocument();
  });

  it("reports the service as unavailable when the health probe fails", async () => {
    renderWithProviders(<Settings />);

    expect(await screen.findByText(/Backend unavailable/i)).toBeInTheDocument();
    expect(screen.queryByText(/Backend connected/i)).not.toBeInTheDocument();
  });

  it("reports connected only when the real /health endpoint responds ok", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(jsonResponse({ status: "ok", version: "0.1.0" })),
    );
    vi.stubGlobal("fetch", fetchMock);

    renderWithProviders(<Settings />);

    expect(await screen.findByText(/Backend connected/i)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalled();
  });
});
