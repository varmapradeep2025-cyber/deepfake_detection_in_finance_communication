import { describe, expect, it } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MultimodalAnalysis } from "../pages/MultimodalAnalysis";
import { DEMO_LABEL } from "../lib/mockAnalysis";
import { renderWithProviders } from "./render";

function imageFile(name = "photo.png", type = "image/png", size = 2048): File {
  return new File([new Uint8Array(size)], name, { type });
}

describe("multimodal analysis workflow", () => {
  it("starts in the idle state", () => {
    renderWithProviders(<MultimodalAnalysis />);
    expect(screen.getByTestId("analysis-panel")).toHaveAttribute(
      "data-state",
      "idle",
    );
  });

  it("shows an error for an invalid file", () => {
    renderWithProviders(<MultimodalAnalysis />);

    // Bypass the input's accept filter so an unsupported file reaches the handler.
    const input = screen.getByLabelText("Select image file");
    const invalid = imageFile("notes.txt", "text/plain");
    Object.defineProperty(input, "files", {
      value: [invalid],
      configurable: true,
    });
    fireEvent.change(input);

    expect(screen.getByTestId("analysis-panel")).toHaveAttribute(
      "data-state",
      "error",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/Unsupported extension/);
  });

  it("moves to ready, shows file details, and removes the file", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MultimodalAnalysis />);

    const input = screen.getByLabelText("Select image file");
    await user.upload(input, imageFile("portrait.png"));
    expect(screen.getByTestId("analysis-panel")).toHaveAttribute(
      "data-state",
      "ready",
    );
    expect(screen.getByTestId("selected-file")).toHaveTextContent("portrait.png");

    await user.click(screen.getByRole("button", { name: /Remove/ }));
    expect(screen.getByTestId("analysis-panel")).toHaveAttribute(
      "data-state",
      "idle",
    );
    expect(screen.queryByTestId("selected-file")).not.toBeInTheDocument();
  });

  it("runs a demo analysis and labels the result as demo data", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MultimodalAnalysis />);

    await user.upload(
      screen.getByLabelText("Select image file"),
      imageFile("portrait.png"),
    );
    await user.click(screen.getByRole("button", { name: /Run demo analysis/ }));

    expect(screen.getByTestId("analysis-panel")).toHaveAttribute(
      "data-state",
      "analyzing",
    );

    const result = await screen.findByTestId("analysis-result", undefined, {
      timeout: 4000,
    });
    expect(result).toBeInTheDocument();
    expect(screen.getByText(DEMO_LABEL)).toBeInTheDocument();
    expect(screen.getByTestId("analysis-panel")).toHaveAttribute(
      "data-state",
      "completed",
    );
  });

  it("switches to the URL workflow and validates input", async () => {
    const user = userEvent.setup();
    renderWithProviders(<MultimodalAnalysis />);

    await user.click(screen.getByRole("tab", { name: "URL / phishing" }));

    const field = screen.getByLabelText("URL to inspect");
    await user.type(field, "ftp://example.com");
    await user.click(screen.getByRole("button", { name: /Run demo analysis/ }));

    expect(screen.getByTestId("analysis-panel")).toHaveAttribute(
      "data-state",
      "error",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/scheme/);

    await user.clear(field);
    await user.type(field, "https://valid.example/path");
    await user.click(screen.getByRole("button", { name: /Run demo analysis/ }));

    await screen.findByTestId("analysis-result", undefined, { timeout: 4000 });
    expect(screen.getByText(DEMO_LABEL)).toBeInTheDocument();
  });
});
