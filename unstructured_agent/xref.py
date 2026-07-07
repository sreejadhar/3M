"""
Cross-modal linking: matches a document's mentions (named entities + topics)
against the schema (table/column names) of a linked structured data source,
so a document's content can be traced to the database entities it discusses.

Runs after PII detection, once every earlier step has finished. Best-effort:
if the document's source has no linked structured source, or the metadata
service is unreachable, linking is skipped (not an error) — the rest of the
pipeline already succeeded.

Matching is LLM-based (same pattern as topics.py/ner.py): the model sees the
document's mentions plus the full table/column list and proposes matches —
catching synonyms and paraphrases ("reinsurers" ~ REINSURER_NAME, "policy
tenure" ~ POLICY_START_DATE) that no literal string match ever could. Every
LLM-proposed link is validated against the real schema (table/column must
actually exist) to guard against hallucination. Falls back to lexical token
overlap — exact substring/word matches only — if the LLM call is unavailable
or fails.
"""
from __future__ import annotations

import json
import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8005")

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "per",
    "such", "which", "each", "any", "all", "are", "was", "were",
}

_MIN_SCORE = 0.4


class XrefUnavailable(RuntimeError):
    """Raised when linking can't run — treated as a skip, not an error."""


def _tokenize(name: str) -> set:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)      # camelCase -> camel Case
    s = re.sub(r"[_\-]+", " ", s)                        # snake_case -> snake case
    return {w.lower() for w in re.findall(r"[A-Za-z]+", s)
            if len(w) > 2 and w.lower() not in _STOPWORDS}


def fetch_schema(source_id: str) -> list:
    """Returns [{table, metadata_id, columns: [...]}] for a structured source,
    via the orchestrator's metadata catalog endpoints."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{ORCHESTRATOR_URL}/metadata/entities", params={"source_id": source_id})
            resp.raise_for_status()
            tables = resp.json()

            schema = []
            for t in tables:
                mid = t.get("metadata_id")
                columns = []
                if mid:
                    r2 = client.get(f"{ORCHESTRATOR_URL}/metadata/entities/{mid}")
                    if r2.status_code == 200:
                        columns = [a.get("column_name") for a in r2.json().get("attributes", [])
                                   if a.get("column_name")]
                schema.append({"table": t.get("table_name"), "metadata_id": mid, "columns": columns})
            return schema
    except httpx.HTTPError as exc:
        raise XrefUnavailable(f"metadata service unreachable: {exc}") from exc


def _overlap(mention_tokens: set, target_tokens: set) -> float:
    if not mention_tokens or not target_tokens:
        return 0.0
    return len(mention_tokens & target_tokens) / len(mention_tokens)


def _best_lexical_match(mention_tokens: set, schema: list):
    best = None
    for tbl in schema:
        table_name = tbl.get("table")
        if not table_name:
            continue
        score = _overlap(mention_tokens, _tokenize(table_name))
        if score > 0 and (best is None or score > best["confidence"]):
            best = {"matched_table": table_name, "matched_column": None, "confidence": score,
                    "basis": "lexical:table_name"}
        for col in tbl.get("columns") or []:
            cscore = _overlap(mention_tokens, _tokenize(col))
            if cscore > 0 and (best is None or cscore > best["confidence"]):
                best = {"matched_table": table_name, "matched_column": col, "confidence": cscore,
                        "basis": "lexical:column_name"}
    return best


def _lexical_link(mentions: list, schema: list, top_n: int) -> list:
    links = []
    seen = set()
    for text, mtype in mentions:
        tokens = _tokenize(text)
        if not tokens:
            continue
        match = _best_lexical_match(tokens, schema)
        if not match or match["confidence"] < _MIN_SCORE:
            continue
        key = (text.lower(), match["matched_table"], match["matched_column"])
        if key in seen:
            continue
        seen.add(key)
        links.append({
            "mention": text, "mention_type": mtype,
            "matched_table": match["matched_table"], "matched_column": match["matched_column"],
            "confidence": round(match["confidence"], 2), "basis": match["basis"],
        })
    links.sort(key=lambda x: -x["confidence"])
    return links[:top_n]


def _llm_link(mentions: list, schema: list, top_n: int) -> list:
    from llm_client import get_client

    table_cols = {t["table"]: set(t.get("columns") or []) for t in schema if t.get("table")}
    if not table_cols:
        return []

    schema_text = "\n".join(f"{table}: {', '.join(sorted(cols))}" for table, cols in table_cols.items())
    mentions_text = "\n".join(f'- "{text}" ({mtype})' for text, mtype in mentions)
    mention_types = dict(mentions)

    client = get_client()
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1500,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                "A document's mentions (named entities and topics) are listed below, along with a "
                "database schema. Identify which mentions correspond to specific database tables or "
                "columns — including synonyms and paraphrases, not just literal string matches "
                "(e.g. \"reinsurers\" corresponds to a REINSURER_NAME column).\n\n"
                f"Schema:\n{schema_text}\n\n"
                f"Mentions:\n{mentions_text}\n\n"
                "Return ONLY a JSON array of objects like "
                '{"mention": "...", "matched_table": "...", "matched_column": "..." or null, '
                '"confidence": 0.0-1.0}. matched_table must be one of the exact table names above; '
                "matched_column (if given) must be one of that table's exact column names above. "
                "Skip mentions with no genuine, specific correspondence — do not force a match."
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    proposed = json.loads(raw)
    if not isinstance(proposed, list):
        raise ValueError("LLM did not return a JSON array")

    mention_set = {text for text, _ in mentions}
    links = []
    seen = set()
    for item in proposed:
        if not isinstance(item, dict):
            continue
        mention = str(item.get("mention", "")).strip()
        table = item.get("matched_table")
        column = item.get("matched_column") or None
        if mention not in mention_set or table not in table_cols:
            continue  # not one of our mentions, or a hallucinated table name
        if column is not None and column not in table_cols[table]:
            continue  # hallucinated column name
        key = (mention.lower(), table, column)
        if key in seen:
            continue
        seen.add(key)
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6
        links.append({
            "mention": mention, "mention_type": mention_types.get(mention, "ENTITY"),
            "matched_table": table, "matched_column": column,
            "confidence": round(confidence, 2), "basis": "llm",
        })

    links.sort(key=lambda x: -x["confidence"])
    return links[:top_n]


def link_mentions(entities: list, topics: list, schema: list, top_n: int = 20) -> list:
    """Returns [{mention, mention_type, matched_table, matched_column, confidence, basis}],
    sorted by confidence descending, deduplicated by (mention, table, column).

    Tries LLM-based linking first (catches synonyms/paraphrases); falls back
    to lexical token overlap (exact matches only) if the LLM is unavailable
    or its response fails validation against the real schema."""
    mentions = [(e.get("text", ""), e.get("type", "ENTITY")) for e in entities if e.get("text")]
    mentions += [(t, "TOPIC") for t in topics if t]
    if not mentions or not schema:
        return []

    try:
        links = _llm_link(mentions, schema, top_n)
        if links:
            return links
    except Exception as exc:
        logger.debug("xref: LLM linking skipped — %s", exc)

    return _lexical_link(mentions, schema, top_n)


def push_to_knowledge_graph(structured_source_id: str, asset_id: str, file_name: str, links: list) -> None:
    """Adds this document as a node in the structured source's Knowledge
    Graph, with edges to every table it was linked to, so it shows up in
    Graph Explorer. Best-effort — logs and returns on any failure; a KG push
    problem should never affect the xref step's own success/failure."""
    if not links:
        return
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{ORCHESTRATOR_URL}/sources/{structured_source_id}/graph/link-document",
                json={"asset_id": asset_id, "file_name": file_name, "links": links},
            )
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("xref: could not push document %s into the knowledge graph — %s", file_name, exc)
