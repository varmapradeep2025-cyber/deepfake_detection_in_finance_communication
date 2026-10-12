path = 'ml/qr/analyzer.py'
with open(path, encoding='utf-8') as f:
    s = f.read()
old_start = s.find('    def _record_codes')
old_end = s.find('        return codes', old_start) + len('        return codes')
new = '''    def _record_codes(
        self,
        payloads: tuple[str, ...],
        path: str,
        counter: itertools.count[int],
        evidence: list[EvidenceItem],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """Turn decoded payloads into result entries, recording evidence as it goes."""
        codes: list[dict[str, Any]] = []
        method = f"decoder:{self._decoder.name if self._decoder else 'none'}"
        seen: set[str] = set()
        unique_payloads: list[str] = []
        for p in payloads:
            if p not in seen:
                seen.add(p)
                unique_payloads.append(p)
        payloads_list = unique_payloads
        if len(payloads_list) > MAX_QR_CODES:
            warnings.append(
                f"excessive QR code count detected ({len(payloads_list)}); truncating to {MAX_QR_CODES}"
            )
            payloads_list = payloads_list[:MAX_QR_CODES]
        for index, payload in enumerate(payloads_list, start=1):
            value = payload
            if len(value) > MAX_PAYLOAD_LENGTH:
                value = value[:MAX_PAYLOAD_LENGTH] + "... (truncated)"
            evidence.append(
                self._item(
                    counter,
                    evidence_type=EvidenceType.METADATA,
                    label=f"QR payload #{index}",
                    value=value,
                    source=path,
                    method=method,
                    claim_kind=ClaimKind.OBSERVED,
                )
            )
            if not looks_like_url(payload):
                warnings.append(
                    f"QR payload #{index} is not an http(s) URL; recorded as text and not graded"
                )
                codes.append({"payload": payload, "kind": "text", "url": None})
                continue
            try:
                analysis = analyze_url(payload)
            except ValueError as exc:
                warnings.append(
                    f"QR payload #{index} looked like a URL but could not be parsed: {exc}"
                )
                codes.append({"payload": payload, "kind": "text", "url": None})
                continue
            for finding in analysis.findings:
                evidence.append(
                    self._item(
                        counter,
                        evidence_type=EvidenceType.SIGNAL,
                        label=f"URL signal: {finding.code}",
                        value=finding.message,
                        source=payload,
                        method="ml.url.analyze_url",
                        claim_kind=ClaimKind.INFERRED,
                    )
                )
            codes.append(
                {"payload": payload, "kind": "url", "url": analysis.to_dict()}
            )
        return codes
'''
s2 = s[:old_start] + new + s[old_end:]
with open(path,'w',encoding='utf-8') as f:
    f.write(s2)
print('ok')
