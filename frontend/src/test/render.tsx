import type { ReactElement } from "react";
import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { BackendStatusProvider } from "../context/BackendStatusContext";

export function renderWithProviders(
  ui: ReactElement,
  { route = "/" }: { route?: string } = {},
) {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <BackendStatusProvider>{ui}</BackendStatusProvider>
    </MemoryRouter>,
  );
}
