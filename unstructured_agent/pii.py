"""
PII (personally identifiable information) detection for extracted document
text.

Runs after named entity recognition, once every earlier step has finished.
Structured PII types (email, phone, SSN, credit card, IP address) are
detected deterministically via regex — these have a strict enough format
that pattern matching is more reliable than an LLM call. Detected values are
masked before being stored/returned so the UI never displays raw PII.
"""
from __future__ import annotations

import re

PII_TYPES = ("EMAIL", "PHONE", "SSN", "CREDIT_CARD", "IP_ADDRESS")

_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Checked in this order so a credit-card-shaped match doesn't also fire as a
# phone/SSN false positive on the same span.
_TYPE_ORDER = ["EMAIL", "SSN", "CREDIT_CARD", "PHONE", "IP_ADDRESS"]


def _luhn_valid(digits: str) -> bool:
    """Cheap false-positive filter for CREDIT_CARD — most 13-16 digit runs
    in a business document (account numbers, phone-like strings) are not
    actual card numbers; require a valid Luhn checksum."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask(value: str, kind: str) -> str:
    digits_only = re.sub(r"\D", "", value)
    if kind == "EMAIL":
        user, _, domain = value.partition("@")
        return f"{user[:1]}***@{domain}"
    if kind in ("SSN", "CREDIT_CARD", "PHONE"):
        return f"{'*' * max(len(digits_only) - 4, 0)}{digits_only[-4:]}"
    if kind == "IP_ADDRESS":
        parts = value.split(".")
        return ".".join(parts[:1] + ["*"] * (len(parts) - 1))
    return "*" * len(value)


def detect_pii(text: str, top_n: int = 20) -> list:
    """Returns a list of {type, masked} dicts. Never returns raw PII values."""
    findings = []
    seen_spans = set()

    for kind in _TYPE_ORDER:
        pattern = _PATTERNS[kind]
        for m in pattern.finditer(text):
            span = m.span()
            if any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans):
                continue  # already claimed by a higher-priority type
            value = m.group(0)
            if kind == "CREDIT_CARD" and not _luhn_valid(re.sub(r"\D", "", value)):
                continue
            seen_spans.add(span)
            findings.append({"type": kind, "masked": _mask(value, kind)})

    return findings[:top_n]
