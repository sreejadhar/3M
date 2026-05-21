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
from typing import Any, Dict, List, Optional

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
  "pii_risk":        boolean
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
        fp.setdefault("pii_risk", False)
        return fp
    except Exception as exc:
        logger.warning("LLM fingerprint extraction failed for %s: %s", file_name, exc)
        return _structural_fingerprint(parse_result, file_name)


def _structural_fingerprint(parse_result: ParseResult, file_name: str) -> Dict:
    return {
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
        "pii_risk":        False,
    }


def _infer_title(parse_result: ParseResult, file_name: str) -> str:
    if parse_result.title:
        return parse_result.title
    # Strip extension and clean up
    stem = file_name.rsplit(".", 1)[0]
    return re.sub(r"[_\-]+", " ", stem).strip().title()
