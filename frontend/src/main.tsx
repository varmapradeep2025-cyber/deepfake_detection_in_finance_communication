import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { BackendStatusProvider } from "./context/BackendStatusContext";
import "./index.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element #root was not found.");
}

createRoot(rootElement).render(
  <StrictMode>
    <BrowserRouter>
      <BackendStatusProvider>
        <App />
      </BackendStatusProvider>
    </BrowserRouter>
  </StrictMode>,
);
