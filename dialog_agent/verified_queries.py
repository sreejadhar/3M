"""
verified_queries — persistent store of human-corrected NL -> SQL examples.

Generic prompt rules (dialect syntax, column-selection heuristics — see
plan_node.py rule 19 and _build_dialect_rules) generalise across every data
source. They cannot, however, encode a fact that only exists in ONE schema —
e.g. "meal_opt_in is the meal-participation flag in this view" or
"id_hcp_classification is a near-empty column and not a reliable HCP
identity signal here". Those are source-specific facts a human has to teach
the system once.

This module is where that teaching persists. When a user corrects a wrong
SQL plan, the corrected {question, sql} pair is saved here, scoped to the
KG (data source) it applies to. On every future plan_node call for the same
KG, the most similar past corrections are retrieved (by embedding / keyword
similarity against the new question) and injected into the planning prompt
as verified few-shot examples — so a mistake fixed once does not have to be
rediscovered on every future phrasing of a similar question.

Persists to PostgreSQL when APP_ENV=production + KG_POSTGRES_DSN is set,
else SQLite — same backend selection as kg_bridges.py / kg_registry.py
(see dialog_agent/pg_store.py).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import pg_store

logger = logging.getLogger(__name__)

# ── DDL — backend-specific because of SERIAL vs AUTOINCREMENT ─────────────────

_DDL_PG = """
CREATE TABLE IF NOT EXISTS verified_queries (
    id          SERIAL PRIMARY KEY,
    kg_id       TEXT NOT NULL,
    question    TEXT NOT NULL,
    sql         TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    db_type     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(kg_id, question)
)
"""

_DDL_SQLITE = """
CREATE TABLE IF NOT EXISTS verified_queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kg_id       TEXT NOT NULL,
    question    TEXT NOT NULL,
    sql         TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    db_type     TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL DEFAULT 0,
    UNIQUE(kg_id, question)
)
"""


def _ensure(cur) -> None:
    cur.ddl(_DDL_PG if pg_store.is_postgres() else _DDL_SQLITE)


@dataclass
class VerifiedQuery:
    kg_id:      str
    question:   str
    sql:        str
    note:       str = ""
    db_type:    str = ""
    id:         Optional[int] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


def _row(r: dict) -> VerifiedQuery:
    return VerifiedQuery(
        id=r["id"], kg_id=r["kg_id"], question=r["question"], sql=r["sql"],
        note=r["note"], db_type=r["db_type"],
        created_at=float(r["created_at"]), updated_at=float(r["updated_at"]),
    )


# ── Public CRUD API ────────────────────────────────────────────────────────────

def save(kg_id: str, question: str, sql: str, note: str = "", db_type: str = "") -> int:
    """
    Save a verified {question -> sql} correction for *kg_id*.
    Re-saving the same (kg_id, question) pair updates the SQL/note in place —
    a later correction of the same question always wins. Returns the row id.
    """
    now = time.time()
    question = question.strip()
    sql = sql.strip()
    if not kg_id or not question or not sql:
        raise ValueError("kg_id, question, and sql are all required")
    params = (kg_id, question, sql, note, db_type, now, now)

    if pg_store.is_postgres():
        stmt = """
        INSERT INTO verified_queries (kg_id, question, sql, note, db_type, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(kg_id, question) DO UPDATE SET
            sql=EXCLUDED.sql, note=EXCLUDED.note, db_type=EXCLUDED.db_type,
            updated_at=EXCLUDED.updated_at
        RETURNING id
        """
        with pg_store.cursor_ctx() as cur:
            _ensure(cur)
            return cur.insert_returning_id(stmt, params) or -1
    else:
        stmt = """
        INSERT INTO verified_queries (kg_id, question, sql, note, db_type, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(kg_id, question) DO UPDATE SET
            sql=excluded.sql, note=excluded.note, db_type=excluded.db_type,
            updated_at=excluded.updated_at
        """
        with pg_store.cursor_ctx() as cur:
            _ensure(cur)
            cur.execute(stmt, params)
            row = cur.execute(
                "SELECT id FROM verified_queries WHERE kg_id=? AND question=?",
                (kg_id, question),
            ).fetchone()
            return row["id"] if row else -1


def list_all(kg_id: Optional[str] = None) -> List[VerifiedQuery]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        if kg_id:
            rows = cur.execute(
                "SELECT * FROM verified_queries WHERE kg_id=? ORDER BY updated_at DESC",
                (kg_id,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM verified_queries ORDER BY updated_at DESC"
            ).fetchall()
    return [_row(r) for r in rows]


def get_by_id(vq_id: int) -> Optional[VerifiedQuery]:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        r = cur.execute("SELECT * FROM verified_queries WHERE id=?", (vq_id,)).fetchone()
    return _row(r) if r else None


def delete(vq_id: int) -> None:
    with pg_store.cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM verified_queries WHERE id=?", (vq_id,))


# ── Similarity retrieval ─────────────────────────────────────────────────────
# Dependency-light on purpose: this module must never block query planning.
# Tries sentence-transformers for quality; falls back to a stemmed
# bag-of-words cosine similarity (no model download, no API key) on any
# failure — mirrors the fallback ladder in retrieve_node.py's GraphRAG path.

_TOKENIZE_RE = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and tok[-3] not in "aeiou":
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKENIZE_RE.findall(text.lower())]


def _keyword_vec(text: str, vocab: Dict[str, int]) -> np.ndarray:
    v = np.zeros(max(len(vocab), 1), dtype=np.float32)
    for tok in _tokenize(text):
        if tok in vocab:
            v[vocab[tok]] = 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rank_by_keyword_similarity(question: str, texts: List[str]) -> np.ndarray:
    vocab: Dict[str, int] = {}
    for t in texts + [question]:
        for tok in _tokenize(t):
            vocab.setdefault(tok, len(vocab))
    corpus_vecs = np.stack([_keyword_vec(t, vocab) for t in texts])
    q_vec = _keyword_vec(question, vocab)
    return corpus_vecs @ q_vec


def _rank_by_embedding_similarity(question: str, texts: List[str]) -> Optional[np.ndarray]:
    try:
        from .embedding_cache import get_sentence_transformer
        model = get_sentence_transformer("all-MiniLM-L6-v2")
        corpus_vecs = model.encode(texts, normalize_embeddings=True).astype(np.float32)
        q_vec = model.encode([question], normalize_embeddings=True)[0].astype(np.float32)
        return corpus_vecs @ q_vec
    except Exception as exc:
        logger.debug("verified_queries: embedding backend unavailable (%s) — using keyword fallback", exc)
        return None


def get_similar(
    kg_id: str,
    question: str,
    top_k: int = 3,
    min_similarity: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Return up to top_k verified examples for *kg_id* most similar to
    *question*, each as {"question", "sql", "note", "similarity"}.
    Best-effort: returns [] on any failure so a broken embedding backend or
    empty store never blocks query planning.
    """
    try:
        candidates = list_all(kg_id)
    except Exception as exc:
        logger.warning("verified_queries: lookup failed for kg_id=%s — %s", kg_id, exc)
        return []
    if not candidates:
        return []

    texts = [c.question for c in candidates]
    try:
        scores = _rank_by_embedding_similarity(question, texts)
        if scores is None:
            scores = _rank_by_keyword_similarity(question, texts)
    except Exception as exc:
        logger.warning("verified_queries: similarity ranking failed — %s", exc)
        return []

    ranked = sorted(zip(candidates, scores), key=lambda t: -t[1])
    results: List[Dict[str, Any]] = []
    for cand, score in ranked[:top_k]:
        if score < min_similarity:
            continue
        results.append({
            "question":   cand.question,
            "sql":        cand.sql,
            "note":       cand.note,
            "similarity": float(score),
        })
    return results
