"""
Governed abbreviation glossary term registry — same DAMA-DMBOK-style term
lifecycle (draft/candidate/approved/deprecated), steward, many-to-many
term<->asset (column/table) links across sources, and an append-only audit
log of every edit/approval as glossary_registry.py's Business Glossary, but
for ABBREVIATION -> FULL FORM mappings discovered from column names/values
rather than business terms discovered from column purpose.

Standalone module (same SQLite/PostgreSQL backend pattern as
glossary_registry.py/metadata_catalog.py, same data/metadata.db file so
abbrev_glossary_term_assets rows can reference metadata_catalog's
metadata_id/attr_id) — importable from orchestrator_api.py without pulling
in dialog_agent. Deliberately NOT importing glossary_registry.py: the two
registries are independent tables with independent lifecycles, kept as
parallel modules the same way glossary_registry.py itself is standalone.

This is the GOVERNANCE layer only: term storage, lifecycle, links, audit.
Discovery (abbreviation detection + LLM-assisted full-form resolution) lives
in abbrev_glossary_generate.py and calls into this module to persist results.

Public API
----------
create_term(**fields)                          -> dict
get_term(term_id)                              -> dict | None
find_term_by_abbreviation(abbreviation, domain=None, status=None) -> dict | None
list_terms(source_id=None, status=None, domain=None)   -> List[dict]
list_terms_with_assets(source_id, status=None) -> List[dict]
update_term(term_id, changed_by="", **fields)  -> bool   (writes audit rows)
approve_term(term_id, changed_by="")           -> bool
reject_term(term_id, changed_by="")            -> bool
link_asset(term_id, source_id, metadata_id, attr_id, confidence, match_method, domain="") -> dict
bulk_link_assets(links)                        -> None
delete_stale_links(pairs)                      -> None
delete_assets_and_jobs_for_source(source_id)   -> None
list_assets_for_term(term_id)                  -> List[dict]
list_assets_for_source(source_id)              -> List[dict]
list_glossary_sources()                        -> List[dict]
get_asset_link(metadata_id, attr_id="")        -> dict | None
list_audit(term_id)                            -> List[dict]
create_job(source_id, force=False)             -> dict
update_job(job_id, **fields)                   -> dict | None
get_job(job_id)                                -> dict | None
list_jobs(source_id=None)                      -> List[dict]
get_active_job(source_id)                      -> dict | None
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

VALID_STATUSES = {"draft", "candidate", "approved", "deprecated"}
JOB_STATUSES = {"queued", "running", "completed", "failed"}
ACTIVE_JOB_STATUSES = {"queued", "running"}


# ── Environment helpers (mirrors glossary_registry.py/metadata_catalog.py) ──

def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def _is_postgres() -> bool:
    return _is_production()


def _sqlite_path() -> str:
    return os.environ.get("METADATA_DB", "data/metadata.db")


# ── Cursor abstraction (mirrors glossary_registry.py) ──────────────────────

class _PGCur:
    def __init__(self, conn: Any, cur: Any) -> None:
        self._conn, self._cur = conn, cur

    def ddl(self, *stmts: str) -> None:
        for s in stmts:
            self._cur.execute(s)

    def execute(self, sql: str, params: tuple = ()) -> "_PGCur":
        self._cur.execute(sql.replace("?", "%s"), params)
        return self

    def fetchall(self) -> List[Dict]:
        return [dict(r) for r in (self._cur.fetchall() or [])]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone()
        return dict(r) if r else None


class _SQLiteCur:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn, self._cur = conn, None

    def ddl(self, *stmts: str) -> None:
        for s in stmts:
            self._conn.execute(s)

    def execute(self, sql: str, params: tuple = ()) -> "_SQLiteCur":
        self._cur = self._conn.execute(sql, params)
        return self

    def fetchall(self) -> List[Dict]:
        rows = self._cur.fetchall() if self._cur else []
        return [dict(r) for r in rows]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone() if self._cur else None
        return dict(r) if r else None


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    if _is_postgres():
        import psycopg2.extras
        import pg_secrets
        conn = pg_secrets.connect(cursor_factory=psycopg2.extras.RealDictCursor)
        cur = conn.cursor()
        try:
            yield _PGCur(conn, cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        path = _sqlite_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield _SQLiteCur(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── DDL ──────────────────────────────────────────────────────────────────────

_DDL_TERMS = """
CREATE TABLE IF NOT EXISTS abbrev_glossary_terms (
    term_id        TEXT PRIMARY KEY,
    abbreviation   TEXT NOT NULL,
    full_form      TEXT NOT NULL DEFAULT '',
    definition     TEXT NOT NULL DEFAULT '',
    domain         TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'draft',
    steward        TEXT NOT NULL DEFAULT '',
    confidence     REAL NOT NULL DEFAULT 0.0,
    match_method   TEXT NOT NULL DEFAULT '',
    version        INTEGER NOT NULL DEFAULT 1,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    approved_by    TEXT NOT NULL DEFAULT '',
    approved_at    TEXT NOT NULL DEFAULT '',
    sample_values_hint TEXT NOT NULL DEFAULT ''
)
"""

_DDL_ASSETS = """
CREATE TABLE IF NOT EXISTS abbrev_glossary_term_assets (
    link_id      TEXT PRIMARY KEY,
    term_id      TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    metadata_id  TEXT NOT NULL,
    attr_id      TEXT NOT NULL DEFAULT '',
    confidence   REAL NOT NULL DEFAULT 0.0,
    match_method TEXT NOT NULL DEFAULT '',
    domain       TEXT NOT NULL DEFAULT '',
    linked_at    TEXT NOT NULL,
    UNIQUE(term_id, metadata_id, attr_id)
)
"""

_DDL_AUDIT = """
CREATE TABLE IF NOT EXISTS abbrev_glossary_term_audit (
    audit_id    TEXT PRIMARY KEY,
    term_id     TEXT NOT NULL,
    changed_by  TEXT NOT NULL DEFAULT '',
    changed_at  TEXT NOT NULL,
    field       TEXT NOT NULL DEFAULT '',
    old_value   TEXT NOT NULL DEFAULT '',
    new_value   TEXT NOT NULL DEFAULT ''
)
"""

_DDL_JOBS = """
CREATE TABLE IF NOT EXISTS abbrev_glossary_jobs (
    job_id      TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    stage       TEXT NOT NULL DEFAULT '',
    message     TEXT NOT NULL DEFAULT '',
    stats_json  TEXT NOT NULL DEFAULT '',
    force       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    finished_at TEXT NOT NULL DEFAULT ''
)
"""

_schema_ensured = False


def _ensure(cur: Any) -> None:
    global _schema_ensured
    cur.ddl(_DDL_TERMS, _DDL_ASSETS, _DDL_AUDIT, _DDL_JOBS)
    _schema_ensured = True


# ── Term CRUD ────────────────────────────────────────────────────────────────

_ALLOWED_TERM_FIELDS = {
    "abbreviation", "full_form", "definition", "domain", "status",
    "steward", "confidence", "match_method",
}


def create_term(
    abbreviation: str, full_form: str = "", definition: str = "",
    domain: str = "", status: str = "draft", confidence: float = 0.0,
    match_method: str = "", steward: str = "", sample_values_hint: str = "",
    changed_by: str = "",
) -> Dict:
    """sample_values_hint: a small JSON-encoded array of the real sibling
    sample values (if any) from the column this term was first grounded on
    actually contained — lets a later abbreviation match on a DIFFERENT
    column sanity-check itself against real data instead of trusting the
    abbreviation string alone (mirrors glossary_registry.create_term).

    Writes a "created" audit row (old_value="") so every term's history —
    not just its subsequent edits — is visible via list_audit(), the same
    way update_term()/approve_term()/reject_term() already record theirs."""
    if status not in VALID_STATUSES:
        status = "draft"
    now = _now()
    term_id = str(uuid.uuid4())
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "INSERT INTO abbrev_glossary_terms "
            "(term_id, abbreviation, full_form, definition, domain, status, "
            "steward, confidence, match_method, version, created_at, updated_at, sample_values_hint) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
            (term_id, abbreviation, full_form, definition, domain, status,
             steward, float(confidence), match_method, now, now, sample_values_hint),
        )
        _record_audit(cur, term_id, changed_by, "created",
                       "", f"abbreviation={abbreviation!r} full_form={full_form!r} status={status!r}")
    return {
        "term_id": term_id, "abbreviation": abbreviation, "full_form": full_form,
        "definition": definition, "domain": domain, "status": status, "steward": steward,
        "confidence": float(confidence), "match_method": match_method, "version": 1,
        "created_at": now, "updated_at": now, "approved_by": "", "approved_at": "",
        "sample_values_hint": sample_values_hint,
    }


def get_term(term_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute("SELECT * FROM abbrev_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
    return dict(row) if row else None


def find_term_by_abbreviation(abbreviation: str, domain: Optional[str] = None,
                               status: Optional[str] = None) -> Optional[Dict]:
    """Exact/normalized abbreviation lookup — the cheap first tier of
    discovery, and what lets a second source's matching column resolve to
    the SAME term_id instead of creating a duplicate. status=None (default)
    matches any non-deprecated term; pass status="approved" to only trust
    steward-vetted/high-confidence terms as reusable ground truth (see
    abbrev_glossary_generate.py) — an unreviewed (draft/candidate) LLM guess
    from one source must never be silently adopted as "the" expansion for a
    matching abbreviation in another source, especially since abbreviations
    are far more prone to cross-domain ambiguity than table/column names."""
    if not abbreviation:
        return None
    with _cursor_ctx() as cur:
        _ensure(cur)
        status_clause = "status = ?" if status else "status != 'deprecated'"
        status_param = (status,) if status else ()
        if domain:
            row = cur.execute(
                f"SELECT * FROM abbrev_glossary_terms WHERE lower(abbreviation)=lower(?) "
                f"AND (domain=? OR domain='') AND {status_clause} "
                f"ORDER BY (domain=?) DESC LIMIT 1",
                (abbreviation, domain, *status_param, domain),
            ).fetchone()
        else:
            row = cur.execute(
                f"SELECT * FROM abbrev_glossary_terms WHERE lower(abbreviation)=lower(?) "
                f"AND {status_clause} LIMIT 1",
                (abbreviation, *status_param),
            ).fetchone()
    return dict(row) if row else None


def list_terms_with_assets(source_id: str, status: Optional[str] = None) -> List[Dict]:
    """Same term rows as list_terms(source_id=...), but each dict also
    carries an "assets" list of {metadata_id, attr_id} for every column this
    term is linked to in *source_id*."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        sql = (
            "SELECT t.*, a.metadata_id AS a_metadata_id, a.attr_id AS a_attr_id "
            "FROM abbrev_glossary_terms t "
            "JOIN abbrev_glossary_term_assets a ON a.term_id = t.term_id "
            "WHERE a.source_id = ?"
        )
        params: List[Any] = [source_id]
        if status:
            sql += " AND t.status=?"
            params.append(status)
        sql += " ORDER BY t.abbreviation"
        rows = [dict(r) for r in cur.execute(sql, tuple(params)).fetchall()]

    terms: Dict[str, Dict] = {}
    for r in rows:
        term_id = r["term_id"]
        metadata_id = r.pop("a_metadata_id")
        attr_id = r.pop("a_attr_id")
        term = terms.setdefault(term_id, {**r, "assets": []})
        term["assets"].append({"metadata_id": metadata_id, "attr_id": attr_id})
    return list(terms.values())


def list_terms(source_id: Optional[str] = None, status: Optional[str] = None,
                domain: Optional[str] = None) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        if source_id:
            sql = (
                "SELECT DISTINCT t.* FROM abbrev_glossary_terms t "
                "JOIN abbrev_glossary_term_assets a ON a.term_id = t.term_id "
                "WHERE a.source_id = ?"
            )
            params: List[Any] = [source_id]
        else:
            sql = "SELECT * FROM abbrev_glossary_terms t WHERE 1=1"
            params = []
        if status:
            sql += " AND t.status=?"
            params.append(status)
        if domain:
            sql += " AND t.domain=?"
            params.append(domain)
        sql += " ORDER BY t.abbreviation"
        rows = cur.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def _record_audit(cur: Any, term_id: str, changed_by: str, field: str, old_value: Any, new_value: Any) -> None:
    cur.execute(
        "INSERT INTO abbrev_glossary_term_audit (audit_id, term_id, changed_by, changed_at, field, old_value, new_value) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), term_id, changed_by, _now(), field, str(old_value), str(new_value)),
    )


def update_term(term_id: str, changed_by: str = "", **fields: Any) -> bool:
    """Update a term's fields, writing one audit row per changed field. A
    manual edit (match_method='manual') is the DMBOK 'declared beats
    inferred' trust boundary — callers (the API layer) are responsible for
    setting match_method='manual', status='approved', confidence=1.0 when
    the edit originates from a human, mirroring glossary_registry.update_term."""
    updates = {k: v for k, v in fields.items() if k in _ALLOWED_TERM_FIELDS}
    if not updates:
        return False
    if "status" in updates and updates["status"] not in VALID_STATUSES:
        return False
    with _cursor_ctx() as cur:
        _ensure(cur)
        current = cur.execute("SELECT * FROM abbrev_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
        if not current:
            return False
        for field, new_value in updates.items():
            old_value = current[field]
            if str(old_value) != str(new_value):
                _record_audit(cur, term_id, changed_by, field, old_value, new_value)
        updates["updated_at"] = _now()
        updates["version"] = int(current["version"]) + 1
        set_clause = ", ".join(f"{k}=?" for k in updates)
        cur.execute(
            f"UPDATE abbrev_glossary_terms SET {set_clause} WHERE term_id=?",
            tuple(updates.values()) + (term_id,),
        )
    return True


def approve_term(term_id: str, changed_by: str = "") -> bool:
    now = _now()
    with _cursor_ctx() as cur:
        _ensure(cur)
        current = cur.execute("SELECT status FROM abbrev_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
        if not current:
            return False
        _record_audit(cur, term_id, changed_by, "status", current["status"], "approved")
        cur.execute(
            "UPDATE abbrev_glossary_terms SET status='approved', approved_by=?, approved_at=?, "
            "updated_at=?, version=version+1 WHERE term_id=?",
            (changed_by, now, now, term_id),
        )
    return True


def reject_term(term_id: str, changed_by: str = "") -> bool:
    with _cursor_ctx() as cur:
        _ensure(cur)
        current = cur.execute("SELECT status FROM abbrev_glossary_terms WHERE term_id=?", (term_id,)).fetchone()
        if not current:
            return False
        _record_audit(cur, term_id, changed_by, "status", current["status"], "deprecated")
        cur.execute(
            "UPDATE abbrev_glossary_terms SET status='deprecated', updated_at=?, version=version+1 WHERE term_id=?",
            (_now(), term_id),
        )
    return True


# ── Term <-> asset links ─────────────────────────────────────────────────────

def link_asset(
    term_id: str, source_id: str, metadata_id: str, attr_id: str = "",
    confidence: float = 0.0, match_method: str = "", domain: str = "",
) -> Dict:
    now = _now()
    link_id = str(uuid.uuid4())
    with _cursor_ctx() as cur:
        _ensure(cur)
        existing = cur.execute(
            "SELECT link_id FROM abbrev_glossary_term_assets WHERE term_id=? AND metadata_id=? AND attr_id=?",
            (term_id, metadata_id, attr_id),
        ).fetchone()
        if existing:
            link_id = existing["link_id"]
            cur.execute(
                "UPDATE abbrev_glossary_term_assets SET confidence=?, match_method=?, domain=?, linked_at=? WHERE link_id=?",
                (float(confidence), match_method, domain, now, link_id),
            )
        else:
            cur.execute(
                "INSERT INTO abbrev_glossary_term_assets "
                "(link_id, term_id, source_id, metadata_id, attr_id, confidence, match_method, domain, linked_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (link_id, term_id, source_id, metadata_id, attr_id, float(confidence), match_method, domain, now),
            )
    return {
        "link_id": link_id, "term_id": term_id, "source_id": source_id,
        "metadata_id": metadata_id, "attr_id": attr_id, "confidence": float(confidence),
        "match_method": match_method, "domain": domain, "linked_at": now,
    }


def bulk_link_assets(links: List[Dict]) -> None:
    """Same upsert as link_asset(), for many rows in ONE connection/
    transaction — mirrors glossary_registry.bulk_link_assets. Each link
    dict: {term_id, source_id, metadata_id, attr_id, confidence, match_method, domain}."""
    if not links:
        return
    now = _now()
    with _cursor_ctx() as cur:
        _ensure(cur)
        for l in links:
            term_id, metadata_id, attr_id = l["term_id"], l["metadata_id"], l.get("attr_id", "")
            existing = cur.execute(
                "SELECT link_id FROM abbrev_glossary_term_assets WHERE term_id=? AND metadata_id=? AND attr_id=?",
                (term_id, metadata_id, attr_id),
            ).fetchone()
            if existing:
                cur.execute(
                    "UPDATE abbrev_glossary_term_assets SET confidence=?, match_method=?, domain=?, linked_at=? WHERE link_id=?",
                    (float(l.get("confidence", 0.0)), l.get("match_method", ""), l.get("domain", ""), now, existing["link_id"]),
                )
            else:
                cur.execute(
                    "INSERT INTO abbrev_glossary_term_assets "
                    "(link_id, term_id, source_id, metadata_id, attr_id, confidence, match_method, domain, linked_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), term_id, l["source_id"], metadata_id, attr_id,
                     float(l.get("confidence", 0.0)), l.get("match_method", ""), l.get("domain", ""), now),
                )


def delete_stale_links(pairs: List[Any]) -> None:
    """Remove existing column links for the given (metadata_id, attr_id)
    pairs, in one connection, so abbrev_glossary_generate.py can re-evaluate
    them on the next pass. Mirrors glossary_registry.delete_stale_links —
    never pass a manually edited (match_method='manual') pair here."""
    if not pairs:
        return
    with _cursor_ctx() as cur:
        _ensure(cur)
        values_sql = ",".join(["(?,?)"] * len(pairs))
        params: List[Any] = []
        for metadata_id, attr_id in pairs:
            params.extend([metadata_id, attr_id])
        cur.execute(
            f"DELETE FROM abbrev_glossary_term_assets "
            f"WHERE (metadata_id, attr_id) IN (VALUES {values_sql}) "
            f"AND match_method != 'manual'",
            tuple(params),
        )


def delete_assets_and_jobs_for_source(source_id: str) -> None:
    """Remove every abbreviation term<->column link and job row scoped to
    this source — called when the source itself is deleted."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM abbrev_glossary_term_assets WHERE source_id=?", (source_id,))
        cur.execute("DELETE FROM abbrev_glossary_jobs WHERE source_id=?", (source_id,))


def list_assets_for_term(term_id: str) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM abbrev_glossary_term_assets WHERE term_id=? ORDER BY linked_at", (term_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def list_assets_for_source(source_id: str) -> List[Dict]:
    """Every (metadata_id, attr_id) pair already linked to a term for this
    source — used by the delta filter to skip columns already discovered."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT a.*, a.domain AS link_domain, t.status AS term_status, "
            "t.confidence AS term_confidence, t.match_method AS term_match_method "
            "FROM abbrev_glossary_term_assets a JOIN abbrev_glossary_terms t ON t.term_id = a.term_id "
            "WHERE a.source_id=?",
            (source_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_glossary_sources() -> List[Dict]:
    """Every source_id that has at least one non-deprecated term linked via
    abbrev_glossary_term_assets, i.e. sources whose Abbreviation Glossary has
    actually been generated."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT a.source_id, COUNT(DISTINCT a.term_id) AS term_count "
            "FROM abbrev_glossary_term_assets a "
            "JOIN abbrev_glossary_terms t ON t.term_id = a.term_id "
            "WHERE t.status != 'deprecated' "
            "GROUP BY a.source_id"
        ).fetchall()
    return [dict(r) for r in rows]


def get_asset_link(metadata_id: str, attr_id: str = "") -> Optional[Dict]:
    """The abbreviation term (if any) currently linked to a specific
    entity/attribute — used to check whether a column already has a
    governed abbreviation mapping before re-running discovery on it."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT a.*, t.abbreviation, t.full_form, t.definition, t.status AS term_status, "
            "t.confidence AS term_confidence, t.match_method AS term_match_method "
            "FROM abbrev_glossary_term_assets a JOIN abbrev_glossary_terms t ON t.term_id = a.term_id "
            "WHERE a.metadata_id=? AND a.attr_id=?",
            (metadata_id, attr_id),
        ).fetchone()
    return dict(row) if row else None


def list_audit(term_id: str) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM abbrev_glossary_term_audit WHERE term_id=? ORDER BY changed_at", (term_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── Abbreviation glossary generation jobs (durable job record) ──────────────

_ALLOWED_JOB_FIELDS = {"status", "stage", "message", "stats_json"}


def create_job(source_id: str, force: bool = False) -> Dict:
    now = _now()
    job_id = str(uuid.uuid4())
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "INSERT INTO abbrev_glossary_jobs "
            "(job_id, source_id, status, stage, message, stats_json, force, created_at, updated_at) "
            "VALUES (?, ?, 'queued', '', '', '', ?, ?, ?)",
            (job_id, source_id, 1 if force else 0, now, now),
        )
    return {
        "job_id": job_id, "source_id": source_id, "status": "queued", "stage": "",
        "message": "", "stats_json": "", "force": force, "created_at": now,
        "updated_at": now, "finished_at": "",
    }


def update_job(job_id: str, **fields: Any) -> Optional[Dict]:
    updates = {k: v for k, v in fields.items() if k in _ALLOWED_JOB_FIELDS}
    if not updates:
        return get_job(job_id)
    if "status" in updates and updates["status"] not in JOB_STATUSES:
        raise ValueError(f"invalid job status: {updates['status']!r}")
    now = _now()
    set_clause = ", ".join(f"{k}=?" for k in updates) + ", updated_at=?"
    params = list(updates.values()) + [now]
    finished = updates.get("status") in ("completed", "failed")
    if finished:
        set_clause += ", finished_at=?"
        params.append(now)
    params.append(job_id)
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(f"UPDATE abbrev_glossary_jobs SET {set_clause} WHERE job_id=?", tuple(params))
    return get_job(job_id)


def get_job(job_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute("SELECT * FROM abbrev_glossary_jobs WHERE job_id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(source_id: Optional[str] = None) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        if source_id:
            rows = cur.execute(
                "SELECT * FROM abbrev_glossary_jobs WHERE source_id=? ORDER BY created_at DESC", (source_id,)
            ).fetchall()
        else:
            rows = cur.execute("SELECT * FROM abbrev_glossary_jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_active_job(source_id: str) -> Optional[Dict]:
    """The queued/running job for this source, if any — backs the
    concurrency guard on the generate-abbreviation-glossary endpoint, and
    (unlike an in-memory flag) survives a process restart."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT * FROM abbrev_glossary_jobs WHERE source_id=? AND status IN ('queued','running') "
            "ORDER BY created_at DESC LIMIT 1",
            (source_id,),
        ).fetchone()
    return dict(row) if row else None
