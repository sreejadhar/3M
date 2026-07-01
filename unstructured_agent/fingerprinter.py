"""
Semantic extraction engine for the unstructured data intelligence agent.

Applies a quality gate, then calls the LLM to produce a semantic fingerprint
(summary, topics, entities, doc_type, etc.) from an ephemeral text window.
Raw text is never stored — only the fingerprint is persisted.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .parsers import ParseResult

logger = logging.getLogger(__name__)

_MODEL_FAST   = "claude-haiku-4-5"
_MODEL_STRONG = "claude-sonnet-4-6"

_JUNK_NAME_RE = re.compile(r"(~\$|\.tmp$|_backup|draft_)", re.IGNORECASE)

_QUALITY_MIN_SIZE_BYTES  = 1024       # 1 KB
_QUALITY_MIN_TEXT_CHARS  = 800        # ~200 tokens
_QUALITY_MAX_PAGES_FAST  = 50         # pages beyond which we use stronger model


@dataclass
class QualityResult:
    passes: bool
    reason: str = ""


def quality_gate(manifest_size: int, parse_result: ParseResult,
                 file_name: str) -> QualityResult:
    if manifest_size < _QUALITY_MIN_SIZE_BYTES:
        return QualityResult(False, "file too small")
    if _JUNK_NAME_RE.search(file_name):
        return QualityResult(False, "junk filename pattern")
    if parse_result.parse_error:
        return QualityResult(False, f"parse error: {parse_result.parse_error}")
    if len(parse_result.text_window.strip()) < _QUALITY_MIN_TEXT_CHARS:
        return QualityResult(False, "insufficient extractable text")
    return QualityResult(True)


def _pick_model(parse_result: ParseResult) -> str:
    if parse_result.ocr_used and (parse_result.ocr_confidence or 1.0) < 0.6:
        return _MODEL_STRONG
    if (parse_result.page_count or 0) > _QUALITY_MAX_PAGES_FAST:
        return _MODEL_STRONG
    return _MODEL_FAST


# ── Regex-based PII scanner ───────────────────────────────────────────────────

_PII_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    # (pii_type, compiled_pattern, confidence)
    ("SSN",         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),                              0.97),
    ("CREDIT_CARD", re.compile(r'\b(?:\d{4}[\s\-]){3}\d{4}\b'),                        0.95),
    ("EMAIL",       re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), 0.99),
    ("PHONE",       re.compile(r'\b(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b'), 0.90),
    ("IP_ADDRESS",  re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),                        0.85),
    ("PASSPORT",    re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),                              0.75),
    ("IBAN",        re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,26}\b'),              0.88),
    ("DATE_OF_BIRTH", re.compile(
        r'\b(?:DOB|Date of Birth|Born)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
        re.IGNORECASE,
    ), 0.82),
    ("TAX_ID",      re.compile(r'\b\d{2}-\d{7}\b'),                                    0.78),
    ("BANK_ACCOUNT",re.compile(r'\b(?:Acct\.?|Account)\s*#?\s*\d{8,17}\b', re.IGNORECASE), 0.80),
]

# Private IP ranges are not real PII — filter them out
_PRIVATE_IP_RE = re.compile(
    r'^(?:127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.).*'
)


def _mask(pii_type: str, value: str) -> str:
    """Return a partially redacted version that confirms detection without exposing the value."""
    v = value.strip()
    if pii_type == "SSN":
        return f"***-**-{v[-4:]}"
    if pii_type == "EMAIL":
        local, _, domain = v.partition("@")
        return f"{local[0]}***@{domain}"
    if pii_type == "PHONE":
        digits = re.sub(r'\D', '', v)
        return f"(***) ***-{digits[-4:]}"
    if pii_type == "CREDIT_CARD":
        digits = re.sub(r'\D', '', v)
        return f"**** **** **** {digits[-4:]}"
    if pii_type == "IP_ADDRESS":
        parts = v.split(".")
        return ".".join(parts[:3]) + ".***"
    if pii_type == "IBAN":
        return v[:4] + "****" + v[-4:]
    if pii_type == "BANK_ACCOUNT":
        return re.sub(r'\d', '*', v[:-4]) + v[-4:]
    # Generic: show first 2 chars + asterisks
    return v[:2] + "*" * max(len(v) - 2, 3)


def _regex_pii_scan(text: str) -> List[Dict]:
    """
    Fast regex pass over the text window. Returns deduplicated PII hits with
    masked values. Never stores the raw matched text.
    """
    hits: List[Dict] = []
    seen: set = set()

    for pii_type, pattern, confidence in _PII_PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0)

            # Skip private IPs — not PII
            if pii_type == "IP_ADDRESS" and _PRIVATE_IP_RE.match(raw):
                continue

            masked = _mask(pii_type, raw)
            key = (pii_type, masked)
            if key in seen:
                continue
            seen.add(key)

            hits.append({
                "type":         pii_type,
                "masked_value": masked,
                "page":         None,   # text_window is not page-segmented
                "confidence":   confidence,
                "source":       "regex",
            })

    return hits


# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a semantic metadata extraction engine. Your task is to extract a structured
semantic fingerprint from the provided document excerpt. You must return ONLY valid JSON —
no markdown fences, no commentary.

Domain context is provided to help you disambiguate industry-specific terminology.
For example:
  CPG/RGM: "yield" means category volume/sell-out, "RSV" means Retail Sales Value
  Banking: "NII" means Net Interest Income, "LCR" means Liquidity Coverage Ratio
  Aviation: "station" means airport station code, not a physical station
  Life Sciences: "yield" means batch yield, "NME" means new molecular entity

Return a JSON object with exactly these fields:
{
  "title":           string,
  "summary":         string (1-2 sentences, what this document is about),
  "domain":          string (e.g. "CPG/RGM", "Banking/FP&A", or empty if unclear),
  "doc_type":        string (one of: report|policy|contract|presentation|research|manual|correspondence|other),
  "topics":          array of strings (5-10 specific topics, not generic terms),
  "named_entities": {
    "organizations": array of strings,
    "products":      array of strings,
    "geographies":   array of strings,
    "people":        array of strings,
    "kpis":          array of strings (metric names, KPI names, formula names)
  },
  "time_references": array of strings (e.g. "Q3 2025", "FY2025", "YoY"),
  "language":        string (ISO 639-1 code, e.g. "en"),
  "sensitivity":     string (one of: public|internal|confidential|restricted),
  "pii_entities":    array of objects — each object MUST have exactly:
                       "type":         one of SSN|EMAIL|PHONE|CREDIT_CARD|DATE_OF_BIRTH|
                                       NAME|ADDRESS|PASSPORT|BANK_ACCOUNT|TAX_ID|
                                       MEDICAL|IP_ADDRESS|OTHER
                       "masked_value": string — partially masked value so the TYPE is
                                       verifiable but the actual data is NOT exposed
                                       (e.g. "***-**-1234" for SSN, "j***@acme.com" for email)
                       "page":         integer (1-based) or null if page unknown
                       "confidence":   float 0.0-1.0
                     Only include PII that is genuinely present in the excerpt.
                     Return an empty array [] if no PII is found.
}
"""


def extract_fingerprint(parse_result: ParseResult, domain_hint: str = "",
                        analyst_role: str = "", file_name: str = "") -> Dict:
    """
    Call the LLM to produce a semantic fingerprint from the text window.
    Returns a dict with the fingerprint fields, or a minimal fallback on failure.
    """
    try:
        from llm_client import get_client
        client = get_client()
    except ImportError:
        return _structural_fingerprint(parse_result, file_name)

    headers_section = ""
    if parse_result.section_headers:
        headers_section = "\nSection headers: " + " | ".join(parse_result.section_headers[:15])

    ocr_note = ""
    if parse_result.ocr_used:
        conf = parse_result.ocr_confidence
        if conf is not None and conf < 0.6:
            ocr_note = "\n[Note: this text was OCR-extracted with low confidence — be conservative about specific named entities.]\n"

    user_content = (
        f"File: {file_name}\n"
        f"Domain context: {domain_hint or 'unknown'}\n"
        f"Analyst role: {analyst_role or 'general analyst'}"
        f"{headers_section}"
        f"{ocr_note}"
        "\n\n--- Document excerpt ---\n"
        + parse_result.text_window
    )

    model = _pick_model(parse_result)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()
        # Strip any accidental code fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        fp = json.loads(raw)
        # Ensure required keys exist
        fp.setdefault("title", _infer_title(parse_result, file_name))
        fp.setdefault("topics", [])
        fp.setdefault("named_entities", {})
        fp.setdefault("time_references", [])
        fp.setdefault("sensitivity", "internal")
        fp.setdefault("pii_entities", [])

        # Merge regex hits (run on full text_window, complement LLM output)
        fp = _merge_pii(fp, parse_result.text_window)
        return fp
    except Exception as exc:
        logger.warning("LLM fingerprint extraction failed for %s: %s", file_name, exc)
        return _structural_fingerprint(parse_result, file_name)


def _merge_pii(fp: Dict, text_window: str) -> Dict:
    """
    Merge regex-detected PII into the LLM fingerprint.
    - Deduplicates by (type, masked_value).
    - Derives pii_risk boolean from whether any entities remain.
    - Validates that each LLM entity has required fields; drops malformed ones.
    """
    # Validate / normalise LLM-returned entities
    valid_llm: List[Dict] = []
    for e in fp.get("pii_entities", []):
        if not isinstance(e, dict):
            continue
        if not e.get("type") or not e.get("masked_value"):
            continue
        e.setdefault("page", None)
        e.setdefault("confidence", 0.70)
        e["source"] = "llm"
        valid_llm.append(e)

    # Regex scan
    regex_hits = _regex_pii_scan(text_window)

    # Merge: regex hits take precedence for types it is confident about;
    # LLM fills in contextual PII (NAME, ADDRESS, MEDICAL, DATE_OF_BIRTH, etc.)
    regex_keys = {(h["type"], h["masked_value"]) for h in regex_hits}
    merged: List[Dict] = list(regex_hits)
    for e in valid_llm:
        key = (e["type"], e["masked_value"])
        if key not in regex_keys:
            merged.append(e)

    # Sort by confidence desc, then type
    merged.sort(key=lambda x: (-x.get("confidence", 0), x.get("type", "")))

    fp["pii_entities"] = merged
    fp["pii_risk"] = len(merged) > 0
    # Remove old boolean if LLM returned it as a field
    fp.pop("pii_risk_old", None)
    return fp


def _structural_fingerprint(parse_result: ParseResult, file_name: str) -> Dict:
    fp = {
        "title":           _infer_title(parse_result, file_name),
        "summary":         "",
        "domain":          "",
        "doc_type":        "other",
        "topics":          parse_result.section_headers[:8],
        "named_entities":  {"organizations": [], "products": [], "geographies": [],
                             "people": [], "kpis": []},
        "time_references": [],
        "language":        "en",
        "sensitivity":     "internal",
        "pii_entities":    [],
    }
    return _merge_pii(fp, parse_result.text_window)


def _infer_title(parse_result: ParseResult, file_name: str) -> str:
    if parse_result.title:
        return parse_result.title
    # Strip extension and clean up
    stem = file_name.rsplit(".", 1)[0]
    return re.sub(r"[_\-]+", " ", stem).strip().title()
