import { describe, expect, it } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { NAV_ITEMS } from "../navigation";
import { renderWithProviders } from "./render";

const INDEX_TITLE = "Deepfake Forensics Dashboard";

describe("application navigation", () => {
  it("renders the dashboard by default", () => {
    renderWithProviders(<App />);
    expect(
      screen.getByRole("heading", { level: 1, name: INDEX_TITLE }),
    ).toBeInTheDocument();
  });

  it("opens every navigation item's page", async () => {
    const user = userEvent.setup();
    renderWithProviders(<App />);
    const sidebar = screen.getByRole("navigation", { name: "Primary" });

    for (const item of NAV_ITEMS) {
      const link = within(sidebar).getByRole("link", { name: item.label });
      await user.click(link);
      const expected = item.end ? INDEX_TITLE : item.label;
      expect(
        await screen.findByRole("heading", { level: 1, name: expected }),
      ).toBeInTheDocument();
    }
  });
});
