/**
 * Typed client for the inference library's analyses, records, and report builder.
 *
 * These functions describe the intended HTTP surface over the Python layer
 * (`ml/inference`): upload one artifact for one modality, list the stored records,
 * and assemble a multimodal report from a chosen subset of them. No server
 * implements them yet — `backend/` contains only a .gitkeep — so every call is
 * expected to resolve to a failure result (tripped by the request timeout or a
 * refused connection). Callers must render that outcome honestly; this module never
 * fabricates a report.
 *
 * Results use the shared `ApiResult` envelope so a failure carries its kind
 * (`network`, `timeout`, `http`, `parse`) and the UI can say precisely what happened.
 */
import type {
  AnalysisModality,
  ApiAnalysisRecord,
  ApiMultimodalReport,
} from "../types";
import { apiClient, type ApiResult } from "./client";
import { API_PATHS, analysisPath } from "./config";

/**
 * Analyze one artifact locally for a modality.
 *
 * The file is sent as multipart/form-data (a real backend must read the uploaded
 * bytes, since the browser cannot hand it a server-side path). The response body is
 * the stored `AnalysisRecord`: the indexed header plus the full serialized report.
 */
export function analyzeModality(
  modality: AnalysisModality,
  file: File,
): Promise<ApiResult<ApiAnalysisRecord>> {
  const form = new FormData();
  form.append("modality", modality);
  form.append("file", file, file.name);
  return apiClient.postForm<ApiAnalysisRecord>(API_PATHS.analyses, form);
}

/** List every stored analysis record, ordered by the library's store. */
export function listAnalysisRecords(): Promise<ApiResult<ApiAnalysisRecord[]>> {
  return apiClient.get<ApiAnalysisRecord[]>(API_PATHS.analyses);
}

/** Fetch one stored analysis record by its report identifier. */
export function getAnalysisRecord(
  reportId: string,
): Promise<ApiResult<ApiAnalysisRecord>> {
  return apiClient.get<ApiAnalysisRecord>(analysisPath(reportId));
}

/**
 * Assemble a multimodal report from the selected records.
 *
 * The server-side builder never fuses scores: the returned report presents the
 * chosen analyses side by side with per-modality coverage and status. An empty
 * selection is rejected by the library, so callers must send at least one id.
 */
export function buildMultimodalReport(
  reportIds: string[],
): Promise<ApiResult<ApiMultimodalReport>> {
  return apiClient.post<ApiMultimodalReport>(API_PATHS.multimodalReports, {
    analysis_ids: reportIds,
  });
}
