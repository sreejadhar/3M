"""
Topic tagging for extracted document text.

Best-effort: tries an LLM call (haiku, cheap + fast) for high-quality topic
tags, and falls back to frequency-based keyword extraction — which has no
external dependency and always works — if the LLM is unavailable or errors.
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "for", "with", "as", "by", "at", "from", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "into", "such", "not", "no", "can", "will", "shall", "may",
    "should", "would", "could", "have", "has", "had", "do", "does", "did",
    "you", "your", "we", "our", "they", "their", "he", "she", "his", "her",
    "which", "who", "whom", "what", "when", "where", "how", "all", "any",
    "each", "other", "some", "than", "so", "also", "about", "over", "under",
}


def _keyword_fallback(text: str, top_n: int) -> list:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower())
    freq = {}
    for w in words:
        if w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    return [w for w, _ in ranked[:top_n]]


def _llm_topics(text: str, top_n: int) -> list:
    from llm_client import get_client
    client = get_client()
    snippet = text[:6000]
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"Extract up to {top_n} short topic tags (1-3 words each) that best "
                f"describe this document. Return ONLY a JSON array of strings, "
                f"nothing else.\n\n---\n{snippet}\n---"
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    tags = json.loads(raw)
    if not isinstance(tags, list):
        raise ValueError("LLM did not return a JSON array")
    return [str(t).strip() for t in tags if str(t).strip()][:top_n]


def extract_topics(text: str, top_n: int = 6) -> list:
    try:
        tags = _llm_topics(text, top_n)
        if tags:
            return tags
    except Exception as exc:
        logger.debug("topics: LLM tagging skipped — %s", exc)
    return _keyword_fallback(text, top_n)
