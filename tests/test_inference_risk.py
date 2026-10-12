"""Tests for honest risk assessment over a detector score."""

from __future__ import annotations

import pytest

from ml.risk.assess import DEFAULT_RISK_POLICY, RiskLevel, RiskPolicy, assess_risk


class TestAssessRisk:
    def test_unknown_when_no_score(self):
        assessment = assess_risk(None)
        assert assessment.level is RiskLevel.UNKNOWN
        assert assessment.is_known is False
        assert "no detector score" in assessment.rationale

    def test_unknown_carries_supplied_reason(self):
        assessment = assess_risk(None, unavailable_reason="model not installed")
        assert assessment.rationale == "model not installed"

    def test_low_band(self):
        assert assess_risk(0.1, score_kind="probability", calibrated=True).level is RiskLevel.LOW

    def test_elevated_band(self):
        assessment = assess_risk(0.6, score_kind="probability", calibrated=True)
        assert assessment.level is RiskLevel.ELEVATED

    def test_high_band(self):
        assessment = assess_risk(0.95, score_kind="probability", calibrated=True)
        assert assessment.level is RiskLevel.HIGH

    def test_band_boundaries_are_inclusive(self):
        policy = RiskPolicy(elevated_threshold=0.5, high_threshold=0.8)
        assert policy.evaluate(0.5, score_kind="s", calibrated=False).level is RiskLevel.ELEVATED
        assert policy.evaluate(0.8, score_kind="s", calibrated=False).level is RiskLevel.HIGH

    def test_requires_score_kind(self):
        with pytest.raises(ValueError):
            assess_risk(0.5)

    def test_out_of_range_score_rejected(self):
        with pytest.raises(ValueError):
            assess_risk(1.5, score_kind="s")

    def test_calibrated_flag_and_kind_are_carried(self):
        assessment = assess_risk(0.9, score_kind="probability", calibrated=True)
        payload = assessment.to_dict()
        assert assessment.calibrated is True
        assert payload["calibrated"] is True
        assert payload["score_kind"] == "probability"
        assert payload["level"] == "high"

    def test_uncalibrated_score_is_not_presented_as_probability(self):
        assessment = assess_risk(0.9, score_kind="uncalibrated_score", calibrated=False)
        assert assessment.calibrated is False
        assert assessment.score_kind == "uncalibrated_score"
        assert "not a probability" in assessment.rationale


class TestRiskPolicy:
    def test_rejects_high_below_elevated(self):
        with pytest.raises(ValueError):
            RiskPolicy(elevated_threshold=0.8, high_threshold=0.5)

    def test_rejects_out_of_range_thresholds(self):
        with pytest.raises(ValueError):
            RiskPolicy(elevated_threshold=-0.1)
        with pytest.raises(ValueError):
            RiskPolicy(high_threshold=1.1)

    def test_rejects_authenticity_orientation(self):
        with pytest.raises(ValueError):
            RiskPolicy(higher_means_more_manipulation=False)

    def test_default_policy_is_ordered(self):
        assert DEFAULT_RISK_POLICY.high_threshold > DEFAULT_RISK_POLICY.elevated_threshold
