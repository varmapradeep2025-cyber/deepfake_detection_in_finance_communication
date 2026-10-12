import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ModalityAnalysis } from "../pages/ModalityAnalysis";
import type {
  ApiAnalysisRecord,
  ApiMultimodalReport,
} from "../types";
import { renderWithProviders } from "./render";

vi.mock("../api/modalities", () => ({
  analyzeModality: vi.fn(),
  listAnalysisRecords: vi.fn(),
  buildMultimodalReport: vi.fn(),
  getAnalysisRecord: vi.fn(),
}));

import {
  analyzeModality,
  buildMultimodalReport,
  listAnalysisRecords,
} from "../api/modalities";

const networkFailure = {
  ok: false as const,
  kind: "network" as const,
  error: "Could not reach the backend service.",
};

const mockedAnalyze = vi.mocked(analyzeModality);
const mockedList = vi.mocked(listAnalysisRecords);
const mockedBuild = vi.mocked(buildMultimodalReport);

function makeRecord(reportId: string, sourceName: string): ApiAnalysisRecord {
  return {
    schema_version: 1,
    report_id: reportId,
    created_at: "2026-10-11T00:00:00Z",
    modality: "qr",
    status: "analysis_complete",
    source_name: sourceName,
    report: {
      request_id: reportId,
      status: "analysis_complete",
      modality: "qr",
      path: sourceName,
      model: null,
      output: null,
      risk: null,
      evidence: [
        {
          id: "ev-001",
          type: "hash",
          label: "File SHA-256",
          value: "abc123",
          source: sourceName,
          method: "sha256(file bytes)",
          timestamp: "2026-10-11T00:00:00Z",
          confidence: null,
          claim_kind: "observed",
        },
      ],
      warnings: [],
      errors: [],
      recommendation: "Treat this output as a supporting signal, not proof.",
      started_at: "2026-10-11T00:00:00Z",
      completed_at: "2026-10-11T00:00:01Z",
      preprocessing: null,
      result: {
        code_count: 0,
        url_count: 0,
        codes: [],
      },
    },
  };
}

function imageFile(name = "photo.png", type = "image/png"): File {
  return new File([new Uint8Array(1024)], name, { type });
}

beforeEach(() => {
  mockedAnalyze.mockReset();
  mockedList.mockReset();
  mockedBuild.mockReset();
  mockedAnalyze.mockResolvedValue(networkFailure);
  mockedList.mockResolvedValue(networkFailure);
  mockedBuild.mockResolvedValue(networkFailure);
});

describe("modality analysis page", () => {
  it("renders the heading, four modality cards, and the offline notice", async () => {
    renderWithProviders(<ModalityAnalysis />);

    expect(
      screen.getByRole("heading", { level: 1, name: "Modality Analysis" }),
    ).toBeInTheDocument();

    for (const modality of ["image", "video", "audio", "qr"]) {
      expect(screen.getByTestId(`modality-card-${modality}`)).toBeInTheDocument();
    }

    expect(
      screen.getByText(/inference service is not reachable/i),
    ).toBeInTheDocument();
  });

  it("reports a network failure honestly instead of inventing a result", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModalityAnalysis />);

    const card = screen.getByTestId("modality-card-image");
    await user.upload(
      within(card).getByLabelText("Select image file"),
      imageFile(),
    );
    await user.click(within(card).getByRole("button", { name: "Analyze" }));

    const alert = await within(card).findByRole("alert");
    expect(alert).toHaveTextContent(/not running/i);
    expect(within(card).queryByTestId("modality-outcome")).not.toBeInTheDocument();
    expect(mockedAnalyze).toHaveBeenCalledTimes(1);
  });

  it("renders a returned report, including the no-score case", async () => {
    const user = userEvent.setup();
    mockedAnalyze.mockResolvedValue({
      ok: true,
      data: makeRecord("req-qr-1", "payment-qr.png"),
    });

    renderWithProviders(<ModalityAnalysis />);

    const card = screen.getByTestId("modality-card-image");
    await user.upload(
      within(card).getByLabelText("Select image file"),
      imageFile(),
    );
    await user.click(within(card).getByRole("button", { name: "Analyze" }));

    const outcome = await within(card).findByTestId("modality-outcome");
    expect(outcome).toHaveTextContent(/No detector score produced/i);
    expect(outcome).toHaveTextContent("payment-qr.png");
  });

  it("states that stored analyses are unreachable when the service is offline", async () => {
    renderWithProviders(<ModalityAnalysis />);

    const builder = screen.getByTestId("report-builder");
    expect(
      await within(builder).findByText(/could not be listed/i),
    ).toBeInTheDocument();
  });

  it("assembles a report from the selected stored analyses", async () => {
    const user = userEvent.setup();
    const records = [
      makeRecord("req-qr-1", "payment-qr.png"),
      makeRecord("req-qr-2", "onboarding-qr.png"),
    ];
    mockedList.mockResolvedValue({ ok: true, data: records });

    const report: ApiMultimodalReport = {
      report_id: "report-1",
      title: "Multimodal analysis report",
      generated_at: "2026-10-11T00:00:02Z",
      sections: [
        {
          report_id: "req-qr-1",
          source_name: "payment-qr.png",
          modality: "qr",
          status: "analysis_complete",
          has_score: false,
          risk_level: null,
          headline: "Completed without a detector score.",
          evidence_count: 1,
          warning_count: 0,
          error_count: 0,
          notable: [],
        },
      ],
      modality_counts: { qr: 1 },
      status_counts: { analysis_complete: 1 },
      scored_count: 0,
      unscored_count: 1,
      notable_findings: [],
      disclaimer: "Scores are not combined, averaged, or summed.",
    };
    mockedBuild.mockResolvedValue({ ok: true, data: report });

    renderWithProviders(<ModalityAnalysis />);

    const builder = screen.getByTestId("report-builder");
    const checkbox = await within(builder).findByLabelText(/payment-qr\.png/);
    await user.click(checkbox);
    await user.click(within(builder).getByRole("button", { name: /Build report/ }));

    const assembled = await within(builder).findByTestId("multimodal-report");
    expect(assembled).toHaveTextContent("Multimodal analysis report");
    expect(assembled).toHaveTextContent(/not combined, averaged, or summed/i);
    await waitFor(() =>
      expect(mockedBuild).toHaveBeenCalledWith(["req-qr-1"]),
    );
  });
});
