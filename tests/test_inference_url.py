"""Tests for syntactic URL analysis: checkable signals, never a verdict."""

from __future__ import annotations

import json

import pytest

from ml.url.analyze import Severity, analyze_url, looks_like_url


def _codes(url: str) -> set[str]:
    return {finding.code for finding in analyze_url(url).findings}


def test_clean_https_url_has_no_findings():
    analysis = analyze_url("https://example.com/about")
    assert analysis.findings == ()
    assert analysis.max_severity is None
    assert analysis.is_ip_literal is False


def test_http_is_insecure_transport():
    assert "insecure_transport" in _codes("http://example.com/")


def test_ip_literal_host_is_high():
    analysis = analyze_url("https://203.0.113.5/login")
    assert analysis.is_ip_literal is True
    assert any(
        finding.code == "ip_literal_host" and finding.severity is Severity.HIGH
        for finding in analysis.findings
    )


def test_punycode_host_is_flagged():
    assert "punycode_host" in _codes("https://xn--80ak6aa92e.com/")


def test_userinfo_is_flagged_and_masked():
    analysis = analyze_url("https://user:secret@example.com/")
    assert analysis.has_userinfo is True
    finding = next(f for f in analysis.findings if f.code == "userinfo_in_url")
    assert "secret" not in (finding.detail or "")


def test_suspicious_tld_is_flagged():
    assert "suspicious_tld" in _codes("https://promo.tk/win")


def test_phishing_keyword_is_flagged():
    assert "phishing_keyword" in _codes("https://example.com/account/verify")


def test_nonstandard_port_is_flagged():
    assert "nonstandard_port" in _codes("https://example.com:8443/")


def test_at_sign_in_path_is_flagged():
    assert "at_sign_in_path" in _codes("https://example.com/a@b")


def test_rejects_non_http_scheme():
    with pytest.raises(ValueError):
        analyze_url("ftp://example.com/file")


def test_rejects_whitespace():
    with pytest.raises(ValueError):
        analyze_url("https://example.com/a b")


def test_rejects_empty():
    with pytest.raises(ValueError):
        analyze_url("   ")


def test_looks_like_url():
    assert looks_like_url("https://example.com")
    assert not looks_like_url("WIFI:S:home;T:WPA;;")
    assert not looks_like_url("example.com")


def test_to_dict_is_json_serializable():
    payload = analyze_url("http://198.51.100.9/verify").to_dict()
    json.dumps(payload)
    assert payload["max_severity"] == "high"
    assert isinstance(payload["findings"], list)
