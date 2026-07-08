"""
Cross-modal linking: matches a document's mentions (named entities + topics)
against the schema (table/column names) of the structured data source it
discusses.

Unlike the earlier version of this module, the datasource itself is not
picked manually — it's inferred automatically from the document's meaning:

  1. Shortlist: ask the LLM which candidate datasources (by name/description/
     domain/table names) this document is genuinely relevant to, based on
     subject matter. Falls back to cosine similarity between the document's
     and each candidate's cached schema-profile embeddings if the LLM call
     is unavailable — note that fallback is only as semantic as the active
     embedding backend (see embedder.py); the offline hashing fallback is
     closer to keyword overlap than real semantic similarity.
  2. Confirm: for each shortlisted candidate, run LLM-based mention matching
     (same pattern as topics.py/ner.py) against its *real* schema, catching
     synonyms and paraphrases ("reinsurers" ~ REINSURER_NAME) that no literal
     string match ever could. Every LLM-proposed link is validated against
     the real schema to guard against hallucination. Falls back to lexical
     token overlap if the LLM call is unavailable or fails.

Only matches that clear a confidence floor make it into the Knowledge Graph
— see XREF_LINK_MIN_CONFIDENCE below. Best-effort throughout: an unreachable
datasource or a schema-fetch failure is skipped, not an error.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

import httpx

from .embedder import cosine_similarity, embed_text

logger = logging.getLogger(__name__)

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8005")

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "per",
    "such", "which", "each", "any", "all", "are", "was", "were",
}

_MIN_SCORE = 0.4

# How many datasources the embedding shortlist keeps, and how similar a
# datasource's schema profile must be to the document to be considered.
SHORTLIST_TOP_K = int(os.environ.get("XREF_SHORTLIST_TOP_K", "3"))
SHORTLIST_MIN_SCORE = float(os.environ.get("XREF_SHORTLIST_MIN_SCORE", "0.15"))

# A shortlisted candidate's LLM/lexical-confirmed link confidence must clear
# this floor before it's written into the Knowledge Graph — auto-linking
# has no human in the loop, so this guards against noisy false positives.
LINK_MIN_CONFIDENCE = float(os.environ.get("XREF_LINK_MIN_CONFIDENCE", "0.55"))


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


def fetch_all_sources() -> list:
    """Returns every structured datasource the orchestrator knows about —
    [{id, name, description, domain, table_names, ...}] — the candidate pool
    for datasource inference."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{ORCHESTRATOR_URL}/sources")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        raise XrefUnavailable(f"orchestrator unreachable: {exc}") from exc


def build_schema_profile_text(meta: dict, schema: list) -> str:
    """Flattens a datasource's identity and schema into one string for
    embedding: name/description/domain carry human-written meaning; table
    and column names carry the literal structure."""
    parts = []
    for key in ("name", "description", "domain"):
        v = meta.get(key)
        if v:
            parts.append(str(v))
    for t in schema:
        table = t.get("table")
        if not table:
            continue
        cols = t.get("columns") or []
        parts.append(f"{table}: {', '.join(cols)}" if cols else table)
    return "\n".join(parts)


def get_or_build_schema_embedding(source_id: str, meta: dict, store) -> tuple:
    """Returns (vector, model_label) for a datasource's schema profile.
    Cached in the DocStore — if a cached embedding already exists, it's
    used as-is with NO schema fetch at all.

    This deliberately skips staleness detection (which would require
    calling fetch_schema() on every call just to compare hashes, and did
    in an earlier version of this function) — shortlist_datasources() calls
    this once per candidate datasource for every document processed, so
    fetching each datasource's full schema over HTTP on every call floods
    the orchestrator's metadata endpoints during a bulk reindex. A stale
    shortlist ranking is a minor accuracy cost (it only affects which
    datasources get considered, not the final links — link_mentions()
    always fetches each shortlisted candidate's live schema before
    confirming a match). Call store.save_schema_embedding() again directly
    (e.g. from an admin action) to force a refresh for a specific source."""
    cached = store.get_schema_embedding(source_id)
    if cached:
        return cached["embedding"], cached["model"]

    schema = fetch_schema(source_id)
    profile_text = build_schema_profile_text(meta, schema)
    profile_hash = hashlib.sha1(profile_text.encode("utf-8")).hexdigest()
    vector, model_label = embed_text(profile_text)
    store.save_schema_embedding(source_id, vector, model_label, profile_hash)
    return vector, model_label


# Bonus added to a candidate's cosine score when its name literally appears
# in the document's own text — a safety net for the offline hashing embedder
# (no stemming/TF-IDF weighting), which can bury an exact-name match under
# noise from a large schema profile. Harmless when sentence-transformers is
# available too: a true name match should rank near the top regardless.
_NAME_MATCH_BOOST = 0.3


def _name_matches(name: str, doc_tokens: set) -> bool:
    name_tokens = _tokenize(name or "")
    return bool(name_tokens) and name_tokens <= doc_tokens


def _llm_shortlist(doc_text: str, candidates: list, top_k: int) -> list:
    """Uses the LLM to judge which candidate datasources this document is
    genuinely relevant to, based on subject matter — not literal keyword
    overlap. Given only each candidate's name/description/domain/table
    names (no full column-level schema — that's the confirm step's job),
    this is a single cheap call per document, and it's the only path in
    this module that's actually semantic when no local embedding model is
    available (see shortlist_datasources)."""
    from llm_client import get_client

    cand_by_id = {c.get("id"): c for c in candidates if c.get("id")}
    if not cand_by_id:
        return []

    cand_text = "\n".join(
        f"- id={cid}, name={c.get('name') or 'n/a'}, domain={c.get('domain') or 'n/a'}, "
        f"description={c.get('description') or 'n/a'}, "
        f"tables={', '.join((c.get('table_names') or [])[:20]) or 'n/a'}"
        for cid, c in cand_by_id.items()
    )

    client = get_client()
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        temperature=0,
        timeout=20.0,
        messages=[{
            "role": "user",
            "content": (
                "A document's content is summarized below, followed by a list of candidate "
                "databases. Identify which database(s), if any, this document is genuinely "
                "relevant to — based on the actual subject matter it discusses, not just "
                "literal keyword overlap with the database name.\n\n"
                f"Document content:\n{doc_text[:3000]}\n\n"
                f"Candidate databases:\n{cand_text}\n\n"
                'Return ONLY a JSON array of objects like {"id": "...", "score": 0.0-1.0}, '
                "sorted by relevance descending. id must be one of the exact ids above. Only "
                "include databases with genuine, specific relevance — do not force a match. "
                "Return an empty array [] if none are relevant."
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    proposed = json.loads(raw)
    if not isinstance(proposed, list):
        raise ValueError("LLM did not return a JSON array")

    scored = []
    seen = set()
    for item in proposed:
        if not isinstance(item, dict):
            continue
        sid = item.get("id")
        if sid not in cand_by_id or sid in seen:
            continue  # not one of our candidates, or a hallucinated/duplicate id
        seen.add(sid)
        try:
            score = max(0.0, min(1.0, float(item.get("score", 0.6))))
        except (TypeError, ValueError):
            score = 0.6
        scored.append({"source_id": sid, "name": cand_by_id[sid].get("name"), "score": round(score, 3)})

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def _embedding_shortlist(doc_vector: list, candidates: list, store, doc_text: str,
                          top_k: int, min_score: float) -> list:
    """Cosine-ranks candidate datasources against the document's embedding,
    with a name-match boost. Only genuinely semantic when a real embedding
    model (e.g. sentence-transformers) is available — with the offline
    hashing fallback this degrades to near-keyword-overlap, so it's used
    only as a fallback when the LLM shortlist is unavailable."""
    doc_tokens = _tokenize(doc_text) if doc_text else set()
    if not doc_vector:
        return []

    scored = []
    for meta in candidates:
        source_id = meta.get("id")
        if not source_id:
            continue
        try:
            vector, _ = get_or_build_schema_embedding(source_id, meta, store)
        except XrefUnavailable as exc:
            logger.warning("xref: skipping unreachable source %s during shortlist — %s", source_id, exc)
            continue
        cosine = cosine_similarity(doc_vector, vector)
        name_matched = _name_matches(meta.get("name"), doc_tokens)
        score = min(1.0, cosine + (_NAME_MATCH_BOOST if name_matched else 0.0))
        if score >= min_score:
            scored.append({"source_id": source_id, "name": meta.get("name"),
                            "score": round(score, 3), "name_matched": name_matched})

    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def shortlist_datasources(doc_vector: list, store, doc_text: str = "",
                           top_k: int = None, min_score: float = None) -> list:
    """Returns [{source_id, name, score}], ranked by relevance to the
    document — the automatic "which database is this document about"
    inference, no manual selection involved.

    Tries LLM-based judgment first (genuinely semantic — reasons over each
    candidate's name/description/domain/tables against the document's
    actual content). Falls back to cosine similarity between embeddings
    (only as semantic as the active embedding backend — see embedder.py)
    if the LLM call is unavailable or fails."""
    top_k = SHORTLIST_TOP_K if top_k is None else top_k
    min_score = SHORTLIST_MIN_SCORE if min_score is None else min_score

    try:
        candidates = fetch_all_sources()
    except XrefUnavailable as exc:
        logger.warning("xref: could not list datasources for shortlisting — %s", exc)
        return []

    if doc_text:
        try:
            scored = _llm_shortlist(doc_text, candidates, top_k)
            return [c for c in scored if c["score"] >= min_score]
        except Exception as exc:
            logger.debug("xref: LLM shortlist skipped, falling back to embeddings — %s", exc)

    return _embedding_shortlist(doc_vector, candidates, store, doc_text, top_k, min_score)


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
        timeout=20.0,
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
    problem should never affect the caller's own success/failure."""
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
