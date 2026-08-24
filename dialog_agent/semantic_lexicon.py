"""
semantic_lexicon — persistent concept -> real-column bindings, scoped per
source, for business concepts that must be DERIVED from raw data rather than
looked up as a single existing column (e.g. "promotion count", "top
performer").

Companion to `verified_queries.py` (whole-question -> whole-SQL few-shots).
This module operates at a finer grain: one concept -> a structured, machine-
checkable set of column bindings + aggregation + grain + filter literals,
so a resolution can be enforced by a validator, not just pasted into a
prompt as advice (unlike `glossary_terms.sql_hint`, which is free text).

Populated by `dialog_agent/nodes/dissect_node.py` on lexicon miss, plus a
one-time `bootstrap()` seed from data that already exists (KPIs, glossary
terms, profiled semantic roles, and mining `verified_queries` for real
{table, column} references from SQL that already ran successfully).

Persists to PostgreSQL when APP_ENV=production, else SQLite — same backend
selection as verified_queries.py (see dialog_agent/pg_store.py and
pg_secrets.py).

Design: docs/Semantic_Lexicon_And_Evaluation_Loop_Design.md
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from . import pg_store
from .verified_queries import _rank_by_keyword_similarity, _stem

logger = logging.getLogger(__name__)

# ── DDL — backend-specific because of BOOLEAN/INTEGER flag storage ───────────

_DDL_PG = """
CREATE TABLE IF NOT EXISTS semantic_lexicon (
  entry_id          TEXT PRIMARY KEY,
  source_id         TEXT NOT NULL,
  term              TEXT NOT NULL,
  display_term      TEXT NOT NULL DEFAULT '',
  aliases_json      TEXT NOT NULL DEFAULT '[]',
  kind              TEXT NOT NULL,

  bindings_json     TEXT NOT NULL,
  aggregation       TEXT NOT NULL DEFAULT '',
  grain             TEXT NOT NULL DEFAULT '',
  filter_json       TEXT NOT NULL DEFAULT '[]',
  time_window_json  TEXT NOT NULL DEFAULT '',
  sql_template      TEXT NOT NULL DEFAULT '',

  rationale         TEXT NOT NULL DEFAULT '',
  confidence        REAL NOT NULL DEFAULT 0,
  provenance        TEXT NOT NULL,
  probe_ok          INTEGER NOT NULL DEFAULT 0,
  approved          INTEGER NOT NULL DEFAULT 0,
  hit_count         INTEGER NOT NULL DEFAULT 0,
  fail_count        INTEGER NOT NULL DEFAULT 0,
  schema_fingerprint TEXT NOT NULL DEFAULT '',
  created_at        REAL NOT NULL DEFAULT 0,
  verified_at       REAL NOT NULL DEFAULT 0,
  UNIQUE(source_id, term)
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS semantic_lexicon (
  entry_id          TEXT PRIMARY KEY,
  source_id         TEXT NOT NULL,
  term              TEXT NOT NULL,
  display_term      TEXT NOT NULL DEFAULT '',
  aliases_json      TEXT NOT NULL DEFAULT '[]',
  kind              TEXT NOT NULL,

  bindings_json     TEXT NOT NULL,
  aggregation       TEXT NOT NULL DEFAULT '',
  grain             TEXT NOT NULL DEFAULT '',
  filter_json       TEXT NOT NULL DEFAULT '[]',
  time_window_json  TEXT NOT NULL DEFAULT '',
  sql_template      TEXT NOT NULL DEFAULT '',

  rationale         TEXT NOT NULL DEFAULT '',
  confidence        REAL NOT NULL DEFAULT 0,
  provenance        TEXT NOT NULL,
  probe_ok          INTEGER NOT NULL DEFAULT 0,
  approved          INTEGER NOT NULL DEFAULT 0,
  hit_count         INTEGER NOT NULL DEFAULT 0,
  fail_count        INTEGER NOT NULL DEFAULT 0,
  schema_fingerprint TEXT NOT NULL DEFAULT '',
  created_at        REAL NOT NULL DEFAULT 0,
  verified_at       REAL NOT NULL DEFAULT 0,
  UNIQUE(source_id, term)
)
"""


def _ensure(cur) -> None:
    cur.ddl(_DDL_PG if pg_store.is_postgres() else _DDL_SQLITE)
    # Migration: persisted embedding columns (added after initial release —
    # see save()/lookup()). Previously Tier 3 of lookup() re-encoded EVERY
    # candidate's term on EVERY call; these columns let a term's embedding be
    # computed once at save() time and reused on every future lookup.
    for stmt in (
        "ALTER TABLE semantic_lexicon ADD COLUMN embedding_json TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE semantic_lexicon ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''",
    ):
        try:
            cur.execute(stmt)
        except Exception:
            pass


# ── Data model ─────────────────────────────────────────────────────────────

@dataclass
class LexiconEntry:
    source_id: str
    term: str
    kind: str                                       # direct_column | derived_metric | entity | filter
    bindings: List[Dict[str, str]]                   # [{"table": "...", "column": "..."}]
    display_term: str = ""
    aliases: List[str] = field(default_factory=list)
    aggregation: str = ""
    grain: str = ""
    filter_predicates: List[Dict[str, Any]] = field(default_factory=list)
    time_window: Optional[Dict[str, Any]] = None
    sql_template: str = ""
    rationale: str = ""
    confidence: float = 0.0
    provenance: str = "llm_dissector"                # llm_dissector | execution_verified | human | bootstrap
    probe_ok: bool = False
    approved: bool = False
    hit_count: int = 0
    fail_count: int = 0
    schema_fingerprint: str = ""
    entry_id: Optional[str] = None
    created_at: float = 0.0
    verified_at: float = 0.0
    embedding: List[float] = field(default_factory=list)   # persisted, see save()
    embedding_model: str = ""                               # tag — see embedding_cache.CURRENT_EMBEDDING_TAG

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id, "source_id": self.source_id, "term": self.term,
            "kind": self.kind, "bindings": self.bindings, "display_term": self.display_term,
            "aliases": self.aliases, "aggregation": self.aggregation, "grain": self.grain,
            "filter_predicates": self.filter_predicates, "time_window": self.time_window,
            "sql_template": self.sql_template, "rationale": self.rationale,
            "confidence": self.confidence, "provenance": self.provenance,
            "probe_ok": self.probe_ok, "approved": self.approved,
            "hit_count": self.hit_count, "fail_count": self.fail_count,
            "schema_fingerprint": self.schema_fingerprint,
        }


def _row(r: dict) -> LexiconEntry:
    def _j(key, default):
        try:
            return json.loads(r.get(key) or "") if r.get(key) else default
        except Exception:
            return default

    return LexiconEntry(
        entry_id=r["entry_id"], source_id=r["source_id"], term=r["term"],
        kind=r["kind"], bindings=_j("bindings_json", []),
        display_term=r.get("display_term") or "", aliases=_j("aliases_json", []),
        aggregation=r.get("aggregation") or "", grain=r.get("grain") or "",
        filter_predicates=_j("filter_json", []), time_window=_j("time_window_json", None),
        sql_template=r.get("sql_template") or "", rationale=r.get("rationale") or "",
        confidence=float(r.get("confidence") or 0.0), provenance=r["provenance"],
        probe_ok=bool(r.get("probe_ok")), approved=bool(r.get("approved")),
        hit_count=int(r.get("hit_count") or 0), fail_count=int(r.get("fail_count") or 0),
        schema_fingerprint=r.get("schema_fingerprint") or "",
        created_at=float(r.get("created_at") or 0.0),
        verified_at=float(r.get("verified_at") or 0.0),
        embedding=_j("embedding_json", []),
        embedding_model=r.get("embedding_model") or "",
    )


# ── Term normalization (§4.2) ─────────────────────────────────────────────
# The single largest correctness risk is key drift — two phrasings of the
# same concept landing on different keys, causing a miss and a fresh
# (possibly different) derivation. This pipeline collapses simple
# morphological variants; semantically distinct phrasings still miss and
# are caught by the embedding tier, then learned as an alias (see lookup()).

_STOPWORDS = {"the", "of", "for", "in", "a", "an"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_term(text: str) -> str:
    text = (text or "").lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    tokens = [_stem(t) for t in text.split() if t]
    tokens = [t for t in tokens if t not in _STOPWORDS]
    return _WS_RE.sub(" ", " ".join(tokens)).strip()


# ── Schema fingerprint (§4.6) ──────────────────────────────────────────────

def schema_fingerprint(table_columns_map: Dict[str, Set[str]], tables: List[str]) -> str:
    """Hash of the bound tables' column sets, so a binding surviving a schema
    migration (renamed/dropped column) is detected and re-dissected rather
    than silently served stale."""
    parts = []
    for t in sorted(set(tables)):
        cols = sorted((table_columns_map.get(t) or table_columns_map.get(t.upper()) or set()))
        parts.append(f"{t.lower()}:{','.join(c.lower() for c in cols)}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


# ── CRUD ───────────────────────────────────────────────────────────────────

def save(entry: LexiconEntry) -> str:
    """Upsert on (source_id, term). Returns the entry_id.

    Computes and persists the term's embedding here, once, so lookup()'s
    Tier 3 never needs to re-encode this row again — see
    dialog_agent/embedding_cache.py. Best-effort: if the embedding backend
    is unavailable, embedding/embedding_model are left empty and lookup()
    falls back to keyword similarity for this row, exactly as before this
    feature existed.
    """
    now = time.time()
    entry.entry_id = entry.entry_id or str(uuid.uuid4())
    entry.term = normalize_term(entry.display_term or entry.term) or entry.term

    from .embedding_cache import encode_text, CURRENT_EMBEDDING_TAG
    vec = encode_text(entry.term)
    if vec is not None:
        entry.embedding = vec
        entry.embedding_model = CURRENT_EMBEDDING_TAG
    else:
        entry.embedding = []
        entry.embedding_model = ""

    params = (
        entry.entry_id, entry.source_id, entry.term, entry.display_term,
        json.dumps(entry.aliases), entry.kind, json.dumps(entry.bindings),
        entry.aggregation, entry.grain, json.dumps(entry.filter_predicates),
        json.dumps(entry.time_window) if entry.time_window else "",
        entry.sql_template, entry.rationale, entry.confidence, entry.provenance,
        int(entry.probe_ok), int(entry.approved), entry.hit_count, entry.fail_count,
        entry.schema_fingerprint, now, now,
        json.dumps(entry.embedding), entry.embedding_model,
    )
    cols = ("entry_id, source_id, term, display_term, aliases_json, kind, "
            "bindings_json, aggregation, grain, filter_json, time_window_json, "
            "sql_template, rationale, confidence, provenance, probe_ok, approved, "
            "hit_count, fail_count, schema_fingerprint, created_at, verified_at, "
            "embedding_json, embedding_model")
    placeholders = ",".join(["?"] * 24)

    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        existing = cur.execute(
            "SELECT entry_id FROM semantic_lexicon WHERE source_id=? AND term=?",
            (entry.source_id, entry.term),
        ).fetchone()
        if existing:
            entry.entry_id = existing["entry_id"]
            # fail_count resets to 0 on every re-save: a re-save means dissect_node
            # re-ran the evaluation loop for this term (either the old binding was
            # auto-suppressed by lookup()'s fail_count gate — see _MAX_ALLOWED_FAILS
            # below — or the schema changed) and produced a fresh proposal that
            # passed validation + probe again. Carrying the OLD binding's fail
            # count forward would leave the new, different binding permanently
            # suppressed before it ever gets a chance to prove itself.
            cur.execute(
                "UPDATE semantic_lexicon SET display_term=?, aliases_json=?, kind=?, "
                "bindings_json=?, aggregation=?, grain=?, filter_json=?, time_window_json=?, "
                "sql_template=?, rationale=?, confidence=?, provenance=?, probe_ok=?, "
                "approved=?, fail_count=0, schema_fingerprint=?, verified_at=?, "
                "embedding_json=?, embedding_model=? WHERE entry_id=?",
                (entry.display_term, json.dumps(entry.aliases), entry.kind,
                 json.dumps(entry.bindings), entry.aggregation, entry.grain,
                 json.dumps(entry.filter_predicates),
                 json.dumps(entry.time_window) if entry.time_window else "",
                 entry.sql_template, entry.rationale, entry.confidence, entry.provenance,
                 int(entry.probe_ok), int(entry.approved), entry.schema_fingerprint,
                 now, json.dumps(entry.embedding), entry.embedding_model, entry.entry_id),
            )
        else:
            cur.execute(f"INSERT INTO semantic_lexicon ({cols}) VALUES ({placeholders})", params)
    return entry.entry_id


def list_all(source_id: Optional[str] = None) -> List[LexiconEntry]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        if source_id:
            rows = cur.execute(
                "SELECT * FROM semantic_lexicon WHERE source_id=? ORDER BY term",
                (source_id,),
            ).fetchall()
        else:
            rows = cur.execute("SELECT * FROM semantic_lexicon ORDER BY source_id, term").fetchall()
    return [_row(r) for r in rows]


def get_by_id(entry_id: str) -> Optional[LexiconEntry]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        r = cur.execute("SELECT * FROM semantic_lexicon WHERE entry_id=?", (entry_id,)).fetchone()
    return _row(r) if r else None


def delete(entry_id: str) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM semantic_lexicon WHERE entry_id=?", (entry_id,))


def add_alias(entry_id: str, surface_form: str) -> None:
    """Learn a new surface form for an existing entry (§4.2 — the lexicon
    learns its own synonyms so a second phrasing becomes an exact hit)."""
    entry = get_by_id(entry_id)
    if not entry:
        return
    norm = normalize_term(surface_form)
    if norm and norm not in entry.aliases and norm != entry.term:
        entry.aliases.append(norm)
        with pg_store.cursor_ctx() as cur:
            _ensure(cur)
            cur.execute(
                "UPDATE semantic_lexicon SET aliases_json=? WHERE entry_id=?",
                (json.dumps(entry.aliases), entry_id),
            )


def bump_hit(entry_id: str) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "UPDATE semantic_lexicon SET hit_count = hit_count + 1 WHERE entry_id=?",
            (entry_id,),
        )


def bump_fail(entry_id: str) -> None:
    """§4.5 demotion — repeated downstream failure marks an entry for review;
    this does not auto-delete or auto-unapprove, only counts."""
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "UPDATE semantic_lexicon SET fail_count = fail_count + 1 WHERE entry_id=?",
            (entry_id,),
        )


def approve(entry_id: str) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("UPDATE semantic_lexicon SET approved=1 WHERE entry_id=?", (entry_id,))


def _backfill_embedding(entry_id: str, embedding: List[float], model_tag: str) -> None:
    """Best-effort: persist an embedding computed for a row at lookup time
    (see lookup()'s Tier 3) so future lookups reuse it instead of
    recomputing. Never raises — a failed backfill just means this one row
    keeps paying the recompute cost until it succeeds another time."""
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "UPDATE semantic_lexicon SET embedding_json=?, embedding_model=? WHERE entry_id=?",
            (json.dumps(embedding), model_tag, entry_id),
        )


# ── Lookup ladder (§4.3) ────────────────────────────────────────────────────

# Lightweight correctness guard (cheaper than a full async reflection job):
# an entry that has caused this many downstream execution failures (see
# bump_fail, called from execute_node when a query built from this entry's
# binding fails) is suppressed from every lookup tier — exact/alias/embedding
# alike — until it is re-resolved. This is a synchronous circuit breaker, not
# a proactive audit: it only catches an entry AFTER it has caused real
# failures, and dissect_node's next MISS on that term re-runs the evaluation
# loop, which (if it succeeds) re-saves the entry and resets fail_count to 0
# — see the comment on the UPDATE in save(). A term whose evaluation loop
# keeps failing simply keeps getting re-attempted per-question rather than
# permanently blocking on one bad cached binding.
_MAX_ALLOWED_FAILS = 2

# Second lightweight correctness guard: Tiers 1 (exact) and 2 (alias) are
# strong signals — the question genuinely used this term's normalized form
# or a previously-confirmed alias, so they're served regardless of
# confidence. Tier 3 (embedding similarity) is the fuzziest match and the
# one most likely to confidently apply a WRONG binding to a differently
# phrased question. An unapproved, LLM-derived entry hasn't been confirmed
# by anything except its own self-reported confidence — require that to
# clear a bar before it's trusted on a fuzzy match. Human-authored and
# execution_verified entries are already trusted by a stronger signal than
# self-reported confidence, so they bypass this gate.
_MIN_CONFIDENCE_FOR_FUZZY_MATCH = 0.6


def lookup(
    source_id: str,
    term: str,
    *,
    min_similarity: float = 0.62,
    current_fingerprint: str = "",
) -> Optional[LexiconEntry]:
    """
    Ordered, first hit wins: exact -> alias -> embedding.
    Human-authored glossary/KPI definitions (checked by the caller BEFORE
    calling this — see dissect_node._identify_concepts) always outrank an
    inferred lexicon entry, per §4.3 tier 4.
    Best-effort: returns None on any failure so a broken backend never
    blocks query planning (same discipline as verified_queries.get_similar).
    """
    norm = normalize_term(term)
    if not norm:
        return None
    try:
        candidates = list_all(source_id)
    except Exception as exc:
        logger.warning("semantic_lexicon: lookup failed for source_id=%s — %s", source_id, exc)
        return None
    if not candidates:
        return None

    def _valid(e: LexiconEntry) -> bool:
        if e.fail_count >= _MAX_ALLOWED_FAILS:
            return False
        if not current_fingerprint or not e.schema_fingerprint:
            return True
        return e.schema_fingerprint == current_fingerprint

    # Tier 1 — exact
    for e in candidates:
        if e.term == norm and _valid(e):
            return e

    # Tier 2 — alias
    for e in candidates:
        if norm in (e.aliases or []) and _valid(e):
            return e

    # Tier 3 — embedding (accept above threshold; higher than verified_queries'
    # 0.35 default because a wrong binding here is worse than a missed few-shot)
    def _fuzzy_eligible(e: LexiconEntry) -> bool:
        if not _valid(e):
            return False
        if e.provenance == "llm_dissector" and not e.approved:
            return e.confidence >= _MIN_CONFIDENCE_FOR_FUZZY_MATCH
        return True

    valid_candidates = [e for e in candidates if _fuzzy_eligible(e)]
    if not valid_candidates:
        return None

    from .embedding_cache import encode_text, CURRENT_EMBEDDING_TAG
    query_vec = encode_text(norm)

    scores: Optional[np.ndarray] = None
    if query_vec is not None:
        try:
            q_arr = np.asarray(query_vec, dtype=np.float32)
            corpus_rows = []
            for e in valid_candidates:
                if e.embedding and e.embedding_model == CURRENT_EMBEDDING_TAG:
                    corpus_rows.append(e.embedding)
                    continue
                # Legacy row (saved before this feature, or a model-tag
                # mismatch) — compute once here and persist so every future
                # lookup reuses it instead of recomputing again.
                row_vec = encode_text(e.term)
                if row_vec is None:
                    row_vec = [0.0] * len(query_vec)
                else:
                    try:
                        _backfill_embedding(e.entry_id, row_vec, CURRENT_EMBEDDING_TAG)
                    except Exception as exc:
                        logger.debug("semantic_lexicon: embedding backfill failed for %s — %s",
                                     e.entry_id, exc)
                corpus_rows.append(row_vec)
            scores = np.asarray(corpus_rows, dtype=np.float32) @ q_arr
        except Exception as exc:
            logger.warning("semantic_lexicon: embedding similarity scoring failed — %s", exc)
            scores = None

    if scores is None:
        texts = [e.term for e in valid_candidates]
        try:
            scores = _rank_by_keyword_similarity(norm, texts)
        except Exception as exc:
            logger.warning("semantic_lexicon: similarity ranking failed — %s", exc)
            return None

    best_idx = int(np.argmax(scores)) if len(scores) else -1
    if best_idx >= 0 and float(scores[best_idx]) >= min_similarity:
        best = valid_candidates[best_idx]
        try:
            add_alias(best.entry_id, norm)
        except Exception:
            pass
        return best

    return None


# ── Bootstrap (§4.4) ────────────────────────────────────────────────────────

def bootstrap(
    source_id: str,
    *,
    kg_id: str = "",
    glossary_terms: Optional[List[Dict]] = None,
    kpis: Optional[List[Dict]] = None,
    mine_verified_queries: bool = True,
) -> int:
    """
    Seed the lexicon for *source_id* from data that already exists, requiring
    no new human authoring. Returns the number of entries written.

    Sources (per design §4.4):
      - glossary_terms / kpis            -> provenance="human"
      - verified_queries mining          -> provenance="execution_verified"

    NOTE: `_annotate_column_concepts()` output (ontology_agent/nodes/build_node.py)
    is intentionally NOT mined here — it is only persisted as free-text
    RDFS.comment strings on the ontology graph, not as a structured, independently
    queryable artifact. Bootstrapping from it would require a new RDF-parsing
    adapter for uncertain payoff; left as a follow-up.
    """
    written = 0

    for kpi in (kpis or []):
        try:
            name = kpi.get("name") or ""
            if not name:
                continue
            entry = LexiconEntry(
                source_id=source_id, term=normalize_term(name), display_term=name,
                kind="derived_metric", bindings=[], aggregation="",
                rationale=kpi.get("nl_formula") or kpi.get("description") or "",
                sql_template=kpi.get("sql_expression") or "",
                provenance="human", approved=True, confidence=1.0,
            )
            save(entry)
            written += 1
        except Exception as exc:
            logger.warning("semantic_lexicon.bootstrap: skipping kpi %r — %s", kpi, exc)

    for term in (glossary_terms or []):
        try:
            name = term.get("name") or ""
            if not name:
                continue
            entry = LexiconEntry(
                source_id=source_id, term=normalize_term(name), display_term=name,
                kind="derived_metric", bindings=[], aggregation="",
                rationale=term.get("definition") or "",
                sql_template=term.get("sql_hint") or term.get("formula") or "",
                aliases=[normalize_term(s) for s in (term.get("synonyms") or []) if s],
                provenance="human", approved=bool(term.get("approved")), confidence=1.0,
            )
            save(entry)
            written += 1
        except Exception as exc:
            logger.warning("semantic_lexicon.bootstrap: skipping glossary term %r — %s", term, exc)

    if mine_verified_queries and kg_id:
        try:
            written += _mine_verified_queries(source_id, kg_id)
        except Exception as exc:
            logger.warning("semantic_lexicon.bootstrap: verified_queries mining failed — %s", exc)

    return written


def _mine_verified_queries(source_id: str, kg_id: str) -> int:
    """Parse SQL that already ran successfully (verified_queries) and extract
    its real {table, column} references as bootstrap bindings for the
    question it answers. Reuses the sqlglot AST walker already built for
    hallucination detection instead of writing a second SQL parser."""
    from . import verified_queries
    from .nodes import sql_identifier_resolver as _ast

    written = 0
    for vq in verified_queries.list_all(kg_id):
        try:
            tree = _ast.sqlglot.parse_one(vq.sql, dialect=None)
        except Exception:
            continue
        tables = {str(t.name) for t in tree.find_all(_ast.exp.Table) if t.name}
        columns = {str(c.name) for c in tree.find_all(_ast.exp.Column) if c.name}
        if not tables or not columns:
            continue
        bindings = [{"table": t, "column": c} for t in sorted(tables) for c in sorted(columns)][:10]
        try:
            entry = LexiconEntry(
                source_id=source_id, term=normalize_term(vq.question),
                display_term=vq.question, kind="derived_metric", bindings=bindings,
                sql_template=vq.sql, rationale=vq.note or "mined from verified_queries",
                provenance="execution_verified", probe_ok=True, approved=False, confidence=0.8,
            )
            save(entry)
            written += 1
        except Exception as exc:
            logger.warning("semantic_lexicon.bootstrap: skipping verified query %r — %s", vq.question, exc)
    return written
