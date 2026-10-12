"""Tests for persisted analysis records and the file store."""

from __future__ import annotations

import pytest

from ml.inference.modality import AnalysisModality
from ml.inference.records import AnalysisRecord, AnalysisStore
from ml.inference.report import AnalysisReport
from ml.inference.status import AnalysisStatus
from ml.risk.assess import RiskAssessment, RiskLevel


def _report(
    *,
    request_id: str = "r1",
    status: AnalysisStatus = AnalysisStatus.ANALYSIS_COMPLETE,
    modality: str = "image",
    path: str = "/data/press.png",
    risk: RiskAssessment | None = None,
) -> AnalysisReport:
    return AnalysisReport(
        request_id=request_id,
        status=status,
        modality=AnalysisModality.parse(modality),
        path=path,
        risk=risk,
        warnings=("a warning",),
        started_at="2024-01-01T00:00:00+00:00",
        completed_at="2024-01-01T00:00:01+00:00",
    )


def _record(
    *,
    request_id: str = "r1",
    modality: str = "image",
    path: str = "/data/press.png",
    status: AnalysisStatus = AnalysisStatus.ANALYSIS_COMPLETE,
) -> AnalysisRecord:
    risk = (
        RiskAssessment(level=RiskLevel.HIGH, score=0.91, score_kind="uncalibrated_score")
        if status is AnalysisStatus.ANALYSIS_COMPLETE
        else None
    )
    return AnalysisRecord.from_report(
        _report(
            request_id=request_id,
            modality=modality,
            path=path,
            status=status,
            risk=risk,
        )
    )


def test_record_header_fields():
    record = _record(request_id="r1", modality="video", path="/data/clip.mp4")
    assert record.report_id == "r1"
    assert record.modality == "video"
    assert record.status == "analysis_complete"
    assert record.source_name == "clip.mp4"
    assert record.created_at == "2024-01-01T00:00:01+00:00"


def test_record_detects_risk_and_warnings():
    record = _record()
    assert record.risk_level == "high"
    assert record.warnings == ("a warning",)
    assert record.errors == ()
    assert record.has_score is False


def test_store_save_and_load_roundtrip(tmp_path):
    store = AnalysisStore(tmp_path)
    path = store.save(_record(request_id="r1"))
    assert path.is_file()
    loaded = store.load("r1")
    assert loaded.to_dict() == _record().to_dict()
    assert loaded.report["risk"]["level"] == "high"


def test_store_refuses_overwrite_by_default(tmp_path):
    store = AnalysisStore(tmp_path)
    store.save(_record(request_id="r1"))
    with pytest.raises(FileExistsError):
        store.save(_record(request_id="r1"))


def test_store_overwrite_when_allowed(tmp_path):
    store = AnalysisStore(tmp_path)
    store.save(_record(request_id="r1"))
    store.save(_record(request_id="r1"), overwrite=True)
    assert store.count() == 1


def test_store_lists_records_in_time_order(tmp_path):
    store = AnalysisStore(tmp_path)
    store.save(_record(request_id="b"))
    store.save(_record(request_id="a"))
    ids = [record.report_id for record in store.list_records()]
    assert ids == ["a", "b"]


def test_store_exists_and_missing_load(tmp_path):
    store = AnalysisStore(tmp_path)
    assert store.exists("nope") is False
    with pytest.raises(FileNotFoundError):
        store.load("nope")


def test_store_rejects_path_traversal_identifier(tmp_path):
    store = AnalysisStore(tmp_path)
    record = AnalysisRecord(
        report_id="../escape",
        created_at="2024-01-01T00:00:00+00:00",
        modality="image",
        status="analysis_complete",
        source_name="x.png",
        report={"request_id": "../escape"},
    )
    with pytest.raises(ValueError):
        store.save(record)


def test_from_dict_rejects_unknown_schema_version():
    with pytest.raises(ValueError):
        AnalysisRecord.from_dict(
            {
                "schema_version": 999,
                "report_id": "r1",
                "created_at": "2024-01-01T00:00:00+00:00",
                "modality": "image",
                "status": "analysis_complete",
                "source_name": "x.png",
                "report": {"request_id": "r1"},
            }
        )


def test_from_dict_rejects_missing_report():
    with pytest.raises(TypeError):
        AnalysisRecord.from_dict(
            {
                "schema_version": 1,
                "report_id": "r1",
                "created_at": "2024-01-01T00:00:00+00:00",
                "modality": "image",
                "status": "analysis_complete",
                "source_name": "x.png",
            }
        )
