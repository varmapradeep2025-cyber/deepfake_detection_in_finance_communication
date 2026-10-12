import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { App } from "../App";
import { DatasetPerformance } from "../pages/DatasetPerformance";
import { renderWithProviders } from "./render";

describe("demo labelling", () => {
  it("shows demo badges on the dashboard", () => {
    renderWithProviders(<App />);
    expect(screen.getAllByText(/DEMO MODE/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/DEMO DATA/i).length).toBeGreaterThan(0);
  });

  it("shows unmeasured metrics as em dashes, never zero", () => {
    renderWithProviders(<DatasetPerformance />);
    const dashes = screen.getAllByText("—");
    expect(dashes.length).toBeGreaterThan(0);
    expect(screen.getByText(/No model has been trained/i)).toBeInTheDocument();
  });
});
