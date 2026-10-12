"""Syntactic URL analysis for phishing signals, with no network access.

Grades a URL on the checkable properties of its text — IP-literal hosts, punycode hosts,
embedded userinfo, suspicious TLDs, phishing keywords. It is not a reputation service and
never claims a link is safe or malicious.
"""

from __future__ import annotations

from ml.url.analyze import (
    Severity,
    UrlAnalysis,
    UrlFinding,
    analyze_url,
    looks_like_url,
)

__all__ = [
    "Severity",
    "UrlAnalysis",
    "UrlFinding",
    "analyze_url",
    "looks_like_url",
]
