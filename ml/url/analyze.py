"""Syntactic URL analysis for phishing signals. No network access, ever.

A URL is graded on what its *text* reveals: an IP-literal host, a punycode host, an
embedded ``user:pass@`` userinfo section, a non-standard port, a suspicious top-level
domain, phishing keywords in the host or path, and similar patterns. These are real,
checkable properties of the string.

What this is not
----------------
It is **not** a verdict on whether a link is malicious. A clean result means "none of the
heuristic patterns this module knows about were present", not "this link is safe", and a
finding means "this pattern deserves a closer look", not "this is a phishing site".
Reputation, certificate validity, redirect chains, and page content require a network
fetch, which this module deliberately never performs — so a caller can analyze a hostile
link without the machine making contact with it.

Everything is derived from the standard library; the module has no third-party
dependencies and imports no project modules.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "Severity",
    "UrlAnalysis",
    "UrlFinding",
    "analyze_url",
    "looks_like_url",
]

#: Keyword stems that commonly appear in credential-harvesting URLs. Matched as whole
#: tokens against host labels and path segments, never as raw substrings, so "signin"
#: matches a segment named ``signin`` but "designin" does not.
_PHISHING_KEYWORDS: tuple[str, ...] = (
    "login",
    "log-in",
    "signin",
    "sign-in",
    "verify",
    "verification",
    "secure",
    "security",
    "account",
    "update",
    "confirm",
    "password",
    "wallet",
    "payment",
    "pay",
    "bank",
    "banking",
    "invoice",
    "billing",
    "recover",
    "unlock",
    "bonus",
    "prize",
    "reward",
    "gift",
    "free",
    "urgent",
    "alert",
)

#: TLDs with a high observed abuse rate in phishing campaigns. Presence is a signal, not
#: a judgement about every site under the TLD.
_SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        "zip",
        "mov",
        "top",
        "xyz",
        "gq",
        "tk",
        "ml",
        "cf",
        "ga",
        "work",
        "click",
        "country",
        "kim",
        "rest",
        "loan",
        "support",
        "live",
        "cam",
    }
)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

_MULTI_LABEL_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
#: Percent-encoded octets such as ``%2f`` used to obscure a path boundary.
_PERCENT_OCTET = re.compile(r"%[0-9a-fA-F]{2}")


class Severity(str, Enum):
    """How strongly a single URL pattern argues for scrutiny."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """Numeric rank, so severities can be compared and the maximum taken."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
}


@dataclass(frozen=True, slots=True)
class UrlFinding:
    """One heuristic observation about a URL.

    Attributes:
        code: Stable machine-readable identifier, e.g. ``"ip_literal_host"``.
        severity: Strength of the signal.
        message: Human-readable explanation.
        detail: The matched substring or value, when there is one.
    """

    code: str
    severity: Severity
    message: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class UrlAnalysis:
    """The structural facts and heuristic findings for one URL.

    Attributes:
        url: The URL as analyzed (trimmed), not normalized.
        scheme: Lowercase scheme, e.g. ``"https"``.
        hostname: Lowercase host, IP literal included, without brackets.
        port: Explicit port, or ``None`` when the default for the scheme was implied.
        path: The path component, possibly empty.
        host_suffix_hint: The last two dot-separated labels of the host, e.g.
            ``"example.com"``. A *hint* only: without a public-suffix list this is not a
            guaranteed registrable domain (``co.uk`` sites will read as ``co.uk``).
        findings: Heuristic observations, possibly empty.
        is_ip_literal: Whether the host is a bare IPv4/IPv6 address.
        is_punycode: Whether any host label begins with the ``xn--`` IDNA prefix.
        has_userinfo: Whether the authority carried a ``user:pass@`` section.
    """

    url: str
    scheme: str
    hostname: str
    port: int | None = None
    path: str = ""
    host_suffix_hint: str = ""
    findings: tuple[UrlFinding, ...] = field(default_factory=tuple)
    is_ip_literal: bool = False
    is_punycode: bool = False
    has_userinfo: bool = False

    @property
    def max_severity(self) -> Severity | None:
        """The highest severity among the findings, or ``None`` when there are none."""
        if not self.findings:
            return None
        return max(self.findings, key=lambda finding: finding.severity.rank).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "scheme": self.scheme,
            "hostname": self.hostname,
            "port": self.port,
            "path": self.path,
            "host_suffix_hint": self.host_suffix_hint,
            "is_ip_literal": self.is_ip_literal,
            "is_punycode": self.is_punycode,
            "has_userinfo": self.has_userinfo,
            "max_severity": self.max_severity.value if self.max_severity else None,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def looks_like_url(text: str) -> bool:
    """Whether ``text`` at least looks like an absolute http(s) URL.

    Used to decide whether a decoded QR payload is a link worth analyzing, without
    committing to a full validation that :func:`analyze_url` performs.
    """
    candidate = text.strip()
    if not candidate or any(char.isspace() for char in candidate):
        return False
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return False
    return parts.scheme.lower() in _ALLOWED_SCHEMES and bool(parts.hostname)


def analyze_url(raw: str) -> UrlAnalysis:
    """Analyze a URL's text for phishing signals.

    Args:
        raw: The URL, with or without surrounding whitespace.

    Returns:
        The structural facts and any heuristic findings.

    Raises:
        ValueError: If the input is not an absolute ``http``/``https`` URL with a host.
            Non-URL payloads (a QR code may encode ``WIFI:`` or plain text) are rejected
            here so the caller can record them as such rather than as a link.
    """
    candidate = raw.strip()
    if not candidate:
        raise ValueError("URL is empty")
    if any(char.isspace() for char in candidate):
        raise ValueError("URL contains whitespace")
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise ValueError(f"URL could not be parsed: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"unsupported scheme {parts.scheme!r}; only http and https are analyzed")
    hostname = parts.hostname
    if not hostname:
        raise ValueError("URL has no hostname")

    hostname = hostname.lower()
    is_ip = _is_ip_literal(hostname)
    is_punycode = any(label.startswith("xn--") for label in hostname.split("."))
    has_userinfo = parts.username is not None or parts.password is not None
    port = parts.port
    path = parts.path
    host_suffix_hint = _suffix_hint(hostname)

    findings = _collect_findings(
        parts=parts,
        raw=candidate,
        scheme=scheme,
        hostname=hostname,
        path=path,
        port=port,
        is_ip=is_ip,
        is_punycode=is_punycode,
        has_userinfo=has_userinfo,
    )

    return UrlAnalysis(
        url=candidate,
        scheme=scheme,
        hostname=hostname,
        port=port,
        path=path,
        host_suffix_hint=host_suffix_hint,
        findings=tuple(findings),
        is_ip_literal=is_ip,
        is_punycode=is_punycode,
        has_userinfo=has_userinfo,
    )


def _collect_findings(
    *,
    parts: Any,
    raw: str,
    scheme: str,
    hostname: str,
    path: str,
    port: int | None,
    is_ip: bool,
    is_punycode: bool,
    has_userinfo: bool,
) -> list[UrlFinding]:
    findings: list[UrlFinding] = []

    if scheme == "http":
        findings.append(
            UrlFinding(
                code="insecure_transport",
                severity=Severity.MEDIUM,
                message=(
                    "Link uses plain HTTP; credentials or payment data sent here "
                    "are unencrypted."
                ),
                detail="http",
            )
        )
    if is_ip:
        findings.append(
            UrlFinding(
                code="ip_literal_host",
                severity=Severity.HIGH,
                message="Host is a bare IP address rather than a domain name, "
                "a common phishing trait.",
                detail=hostname,
            )
        )
    if is_punycode:
        findings.append(
            UrlFinding(
                code="punycode_host",
                severity=Severity.HIGH,
                message="Host contains a punycode (xn--) label, often used to spoof "
                "a look-alike domain.",
                detail=hostname,
            )
        )
    if has_userinfo:
        userinfo = parts.username or ""
        if parts.password:
            userinfo = f"{userinfo}:****"
        findings.append(
            UrlFinding(
                code="userinfo_in_url",
                severity=Severity.HIGH,
                message="Authority embeds a user:pass@ section before the host, "
                "a classic way to disguise the real destination.",
                detail=f"{userinfo}@{hostname}",
            )
        )
    if port is not None:
        default_port = 443 if scheme == "https" else 80
        if port != default_port:
            findings.append(
                UrlFinding(
                    code="nonstandard_port",
                    severity=Severity.MEDIUM,
                    message=f"Link specifies port {port} instead of the default for {scheme}.",
                    detail=str(port),
                )
            )
    if "@" in path or "@" in (parts.query or ""):
        findings.append(
            UrlFinding(
                code="at_sign_in_path",
                severity=Severity.MEDIUM,
                message="An '@' appears in the path or query, which can mislead "
                "readers about the host.",
            )
        )
    keyword = _first_keyword(hostname, path)
    if keyword is not None:
        findings.append(
            UrlFinding(
                code="phishing_keyword",
                severity=Severity.MEDIUM,
                message=f"Host or path contains the phishing-associated token {keyword!r}.",
                detail=keyword,
            )
        )
    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else hostname
    if tld in _SUSPICIOUS_TLDS:
        findings.append(
            UrlFinding(
                code="suspicious_tld",
                severity=Severity.MEDIUM,
                message=f"Top-level domain {'.' + tld!r} has a high observed abuse rate.",
                detail=tld,
            )
        )
    if hostname.count(".") >= 4 and not is_ip:
        findings.append(
            UrlFinding(
                code="excessive_subdomains",
                severity=Severity.LOW,
                message="Host has an unusually deep chain of subdomains, "
                "which can hide the real domain.",
                detail=hostname,
            )
        )
    if _PERCENT_OCTET.search(path) or _PERCENT_OCTET.search(parts.query or ""):
        findings.append(
            UrlFinding(
                code="percent_encoded_components",
                severity=Severity.LOW,
                message="URL contains percent-encoded octets, which can obscure structure.",
            )
        )
    if "\\" in parts.path or "//" in path.lstrip("/"):
        findings.append(
            UrlFinding(
                code="obfuscated_path_separators",
                severity=Severity.LOW,
                message="Path contains backslashes or doubled slashes that may "
                "misrepresent the location.",
            )
        )
    if len(raw) >= 200 or len(hostname) >= 40:
        findings.append(
            UrlFinding(
                code="unusually_long",
                severity=Severity.INFO,
                message="URL or hostname is unusually long, sometimes used to push "
                "the domain out of view.",
            )
        )

    return findings


def _first_keyword(hostname: str, path: str) -> str | None:
    tokens = [label for label in hostname.split(".") if label]
    tokens.extend(segment for segment in path.split("/") if segment)
    for token in tokens:
        lowered = token.lower()
        for keyword in _PHISHING_KEYWORDS:
            if lowered == keyword or keyword in lowered.split("-"):
                return keyword
    return None


def _is_ip_literal(hostname: str) -> bool:
    if _MULTI_LABEL_IPV4.match(hostname):
        return True
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def _suffix_hint(hostname: str) -> str:
    labels = hostname.split(".")
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return hostname
