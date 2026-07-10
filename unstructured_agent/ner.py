"""
Named entity recognition for extracted document text.

Runs after topic tagging, once text extraction and embeddings are already
done. Best-effort: tries an LLM call (haiku) for high-quality typed entities,
and falls back to a regex-based extractor — capitalized phrases, money
amounts, dates, percentages — which has no external dependency and always
works if the LLM is unavailable or errors.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

ENTITY_TYPES = ("ORG", "PERSON", "LOCATION", "MONEY", "DATE", "PERCENT", "PRODUCT", "OTHER")

_STOPWORD_STARTS = {
    "The", "This", "That", "These", "Those", "It", "A", "An", "In", "On", "For",
    "With", "As", "By", "At", "From", "See", "Note", "Step",
}


def _regex_fallback(text: str, top_n: int) -> list:
    entities = []
    seen = set()

    def add(name: str, etype: str):
        key = (name.lower(), etype)
        if name and key not in seen:
            seen.add(key)
            entities.append({"text": name, "type": etype})

    for m in re.finditer(r"\$[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|M|B|K))?", text):
        add(m.group(0), "MONEY")
    for m in re.finditer(r"\b\d+(?:\.\d+)?\s?%", text):
        add(m.group(0), "PERCENT")
    for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        add(m.group(0), "DATE")
    for m in re.finditer(r"\b(?:19|20)\d{2}\b", text):
        add(m.group(0), "DATE")
    for m in re.finditer(r"\b(?:[A-Z][a-zA-Z&]+(?:\s+[A-Z][a-zA-Z&]+){0,3})\b", text):
        phrase = m.group(0).strip()
        first_word = phrase.split()[0]
        if first_word in _STOPWORD_STARTS or len(phrase) < 3:
            continue
        add(phrase, "OTHER")

    return entities[:top_n]


def _llm_entities(text: str, top_n: int) -> list:
    from llm_client import get_client
    client = get_client()
    snippet = text[:6000]
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Extract up to {top_n} named entities from this document. Use entity "
                f"types from this set: {', '.join(ENTITY_TYPES)}. Return ONLY a JSON "
                f"array of objects like {{\"text\": \"...\", \"type\": \"...\"}}, "
                f"nothing else.\n\n---\n{snippet}\n---"
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    entities = json.loads(raw)
    if not isinstance(entities, list):
        raise ValueError("LLM did not return a JSON array")
    out = []
    for e in entities:
        if isinstance(e, dict) and e.get("text"):
            out.append({"text": str(e["text"]).strip(), "type": str(e.get("type") or "OTHER").upper()})
    return out[:top_n]


def extract_entities(text: str, top_n: int = 15) -> list:
    try:
        entities = _llm_entities(text, top_n)
        if entities:
            return entities
    except Exception as exc:
        logger.debug("ner: LLM extraction skipped — %s", exc)
    return _regex_fallback(text, top_n)
