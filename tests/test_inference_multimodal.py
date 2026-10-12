"""Tests for the multimodal report builder: side-by-side, never fused."""

from __future__ import annotations

import json

import pytest

from ml.inference.multimodal import MultimodalReportBuilder
from ml.inference.records import AnalysisRecord


def _record(
    report_id: str,
    modality: str,
    status: str,
    *,
    output: dict | None = None,
    risk: dict | None = None,
    result: dict | None = None,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
) -> AnalysisRecord:
    report = {
        "request_id": report_id,
        "status": status,
        "modality": modality,
        "path": f"/data/{report_id}",
        "output": output,
        "risk": risk,
        "evidence": [{"id": "ev-001"}],
        "warnings": list(warnings),
        "errors": list(errors),
        "result": result,
    }
    return AnalysisRecord(
        report_id=report_id,
        created_at="2024-01-01T00:00:00+00:00",
        modality=modality,
        status=status,
        source_name=f"{report_id}.bin",
        report=report,
    )


def _clock() -> str:
    return "2024-02-02T00:00:00+00:00"


def test_empty_records_raise():
    with pytest.raises(ValueError):
        MultimodalReportBuilder(clock=_clock).build([])


def test_counts_and_sections():
    records = [
        _record(
            "img",
            "image",
            "analysis_complete",
            output={"raw_score": 0.9},
            risk={"level": "high"},
        ),
        _record("vid", "video", "model_unavailable", errors=("no video model",)),
        _record("qr", "qr", "analysis_complete"),
    ]
    report = MultimodalReportBuilder(clock=_clock).build(records)
    assert len(report.sections) == 3
    assert report.scored_count == 1
    assert report.unscored_count == 2
    assert report.modality_counts == {"image": 1, "qr": 1, "video": 1}
    assert report.status_counts["analysis_complete"] == 2
    assert report.generated_at == "2024-02-02T00:00:00+00:00"


def test_high_risk_is_notable():
    records = [
        _record(
            "img",
            "image",
            "analysis_complete",
            output={"raw_score": 0.95},
            risk={"level": "high"},
        )
    ]
    report = MultimodalReportBuilder(clock=_clock).build(records)
    assert any("risk assessed as high" in item for item in report.notable_findings)


def test_qr_high_severity_finding_is_notable():
    result = {
        "code_count": 1,
        "url_count": 1,
        "codes": [
            {
                "payload": "https://xn--x.com/",
                "kind": "url",
                "url": {
                    "findings": [
                        {"code": "punycode_host", "severity": "high"},
                        {"code": "insecure_transport", "severity": "medium"},
                    ]
                },
            }
        ],
    }
    report = MultimodalReportBuilder(clock=_clock).build(
        [_record("qr", "qr", "analysis_complete", result=result)]
    )
    assert any("punycode_host" in item for item in report.notable_findings)
    assert not any("insecure_transport" in item for item in report.notable_findings)


def test_unscored_complete_section_headline():
    report = MultimodalReportBuilder(clock=_clock).build(
        [_record("qr", "qr", "analysis_complete")]
    )
    section = report.sections[0]
    assert section.has_score is False
    assert "without a detector score" in section.headline.lower()


def test_no_fused_score_or_risk_in_output():
    report = MultimodalReportBuilder(clock=_clock).build(
        [
            _record("a", "image", "analysis_complete", output={"raw_score": 0.9},
                    risk={"level": "high"}),
            _record("b", "audio", "model_unavailable"),
        ]
    )
    payload = report.to_dict()
    json.dumps(payload)
    assert "score" not in payload
    assert "risk_level" not in payload
    assert payload["scored_count"] == 1
    assert payload["unscored_count"] == 1
    assert report.disclaimer
