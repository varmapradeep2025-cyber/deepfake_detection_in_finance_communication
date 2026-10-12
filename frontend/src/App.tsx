import { Navigate, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/layout/AppLayout";
import { Dashboard } from "./pages/Dashboard";
import { MultimodalAnalysis } from "./pages/MultimodalAnalysis";
import { ModalityAnalysis } from "./pages/ModalityAnalysis";
import { ImageAnalysis } from "./pages/ImageAnalysis";
import { VideoAnalysis } from "./pages/VideoAnalysis";
import { AudioAnalysis } from "./pages/AudioAnalysis";
import { QRUrlAnalysis } from "./pages/QRUrlAnalysis";
import { EvidenceProvenance } from "./pages/EvidenceProvenance";
import { ThreatIntelligence } from "./pages/ThreatIntelligence";
import { Reports } from "./pages/Reports";
import { Experiments } from "./pages/Experiments";
import { DatasetPerformance } from "./pages/DatasetPerformance";
import { AuditTrail } from "./pages/AuditTrail";
import { Settings } from "./pages/Settings";

export function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="image-analysis" element={<ImageAnalysis />} />
        <Route path="video-analysis" element={<VideoAnalysis />} />
        <Route path="audio-analysis" element={<AudioAnalysis />} />
        <Route path="qr-url-analysis" element={<QRUrlAnalysis />} />
        <Route path="analysis" element={<MultimodalAnalysis />} />
        <Route path="modalities" element={<ModalityAnalysis />} />
        <Route path="evidence" element={<EvidenceProvenance />} />
        <Route path="threat-intelligence" element={<ThreatIntelligence />} />
        <Route path="reports" element={<Reports />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="dataset-performance" element={<DatasetPerformance />} />
        <Route path="audit-trail" element={<AuditTrail />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
