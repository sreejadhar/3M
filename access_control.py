"""
RBAC / ABAC access control for DataChat.

Role-Based Access Control (RBAC)
---------------------------------
Four built-in roles:
  viewer   — read-only access to sources visible in their domain
  analyst  — viewer + can run GraphRAG queries and view ontology/KG
  manager  — analyst + can reindex, edit metadata catalog, manage bridges
  admin    — full access, including user management and RBAC configuration

Attribute-Based Access Control (ABAC)
--------------------------------------
Sources carry a ``persona_access`` list (e.g. ["analyst", "admin"]).
Users carry a ``domain`` attribute; a source is accessible only when
  source.domain == user.domain  OR  user.domain == "*"  OR  role == "admin"

Enforcement
-----------
  - ``is_enforced()`` returns True ONLY in production (APP_ENV=production).
  - In dev/test every check returns True (pass-through).
  - Produces no side-effects — the calling layer raises HTTP 403 if needed.

Persistence
-----------
  SQLite (AC_DB, default data/access_control.db) in dev/test.
  PostgreSQL (KG_POSTGRES_DSN) in production.

Public API
----------
bootstrap()                              — create tables, seed default admin
add_user(user_id, name, email, role, domain="*") → dict
get_user(user_id)                        → dict | None
list_users()                             → List[dict]
update_user(user_id, **fields)           → bool
delete_user(user_id)                     → bool
set_role(user_id, role)                  → bool
can_access_source(user_id, source)       → bool   (always True in dev)
can_perform(user_id, action)             → bool   (always True in dev)
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ── Roles & permissions ────────────────────────────────────────────────────────

ROLES = ("viewer", "analyst", "manager", "admin")

# Actions each role may perform (cumulative up the hierarchy)
_ROLE_PERMISSIONS: Dict[str, set] = {
    "viewer":  {"list_sources", "view_source", "view_kg", "graphrag_query"},
    "analyst": {"list_sources", "view_source", "view_kg", "graphrag_query",
                "view_ontology", "view_metadata"},
    "manager": {"list_sources", "view_source", "view_kg", "graphrag_query",
                "view_ontology", "view_metadata",
                "edit_metadata", "reindex_source", "manage_bridges",
                "manage_sources"},
    "admin":   set(),   # populated below — all permissions
}
_ALL_PERMISSIONS = (
    _ROLE_PERMISSIONS["viewer"]
    | _ROLE_PERMISSIONS["analyst"]
    | _ROLE_PERMISSIONS["manager"]
    | {"manage_users", "manage_rbac", "delete_source",
       "edit_glossary_term", "approve_glossary_term"}
)
_ROLE_PERMISSIONS["admin"] = _ALL_PERMISSIONS


# ── Backend selection ──────────────────────────────────────────────────────────

def is_enforced() -> bool:
    """
    Returns True only in production.
    In dev/test all access checks are no-ops (pass-through).
    """
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def _use_postgres() -> bool:
    if not is_enforced():
        return False
    dsn = os.environ.get("KG_POSTGRES_DSN", "")
    if dsn:
        return True
    logger.warning("APP_ENV=production but KG_POSTGRES_DSN is not set — access control uses SQLite.")
    return False


def _sqlite_path() -> str:
    return os.environ.get("AC_DB", "data/access_control.db")

def _pg_dsn() -> str:
    return os.environ.get("KG_POSTGRES_DSN", "")


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    if _use_postgres():
        import psycopg2, psycopg2.extras
        conn = psycopg2.connect(_pg_dsn(), cursor_factory=psycopg2.extras.RealDictCursor)
        cur  = conn.cursor()
        try:
            yield _PGCur(conn, cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        import sqlite3
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


class _PGCur:
    def __init__(self, conn, cur): self._conn = conn; self._cur = cur
    def ddl(self, *stmts):
        for s in stmts: self._cur.execute(s)
    def execute(self, sql, params=()):
        self._cur.execute(sql.replace("?", "%s"), params); return self
    def fetchall(self): return [dict(r) for r in (self._cur.fetchall() or [])]
    def fetchone(self):
        r = self._cur.fetchone(); return dict(r) if r else None


class _SQLiteCur:
    def __init__(self, conn): self._conn = conn; self._cur = None
    def ddl(self, *stmts):
        for s in stmts: self._conn.execute(s)
    def execute(self, sql, params=()):
        self._cur = self._conn.execute(sql, params); return self
    def fetchall(self): return [dict(r) for r in (self._cur.fetchall() if self._cur else [])]
    def fetchone(self):
        r = self._cur.fetchone() if self._cur else None; return dict(r) if r else None


# ── DDL ────────────────────────────────────────────────────────────────────────

_DDL_USERS = """
CREATE TABLE IF NOT EXISTS ac_users (
    user_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    email       TEXT NOT NULL DEFAULT '',
    role        TEXT NOT NULL DEFAULT 'viewer',
    domain      TEXT NOT NULL DEFAULT '*',
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
)
"""

_DDL_SOURCE_ACL = """
CREATE TABLE IF NOT EXISTS ac_source_acl (
    acl_id      TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    user_id     TEXT,
    role        TEXT,
    domain      TEXT,
    created_at  REAL NOT NULL
)
"""


def _ensure(cur: Any) -> None:
    cur.ddl(_DDL_USERS, _DDL_SOURCE_ACL)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap() -> None:
    """Create tables and seed a default admin user if none exist."""
    with _cursor_ctx() as cur:
        _ensure(cur)
        existing = cur.execute("SELECT COUNT(*) as cnt FROM ac_users WHERE role='admin'").fetchone()
        cnt = existing.get("cnt", 0) if existing else 0
        if cnt == 0:
            now = time.time()
            cur.execute(
                "INSERT INTO ac_users (user_id, name, email, role, domain, is_active, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), "Default Admin", "admin@datachat.local",
                 "admin", "*", 1, now, now),
            )
            logger.info("access_control: seeded default admin user")


# ── Public API ─────────────────────────────────────────────────────────────────

def add_user(
    user_id: Optional[str],
    name: str,
    email: str,
    role: str = "viewer",
    domain: str = "*",
) -> Dict:
    """Create a new user. Returns the user dict."""
    if role not in ROLES:
        raise ValueError(f"Invalid role '{role}'. Must be one of {ROLES}")
    uid  = user_id or str(uuid.uuid4())
    now  = time.time()
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "INSERT INTO ac_users (user_id, name, email, role, domain, is_active, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (uid, name, email, role, domain, 1, now, now),
        )
    return get_user(uid) or {}


def get_user(user_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute("SELECT * FROM ac_users WHERE user_id=?", (user_id,)).fetchone()
    return _coerce_user(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute("SELECT * FROM ac_users WHERE lower(email)=lower(?)", (email,)).fetchone()
    return _coerce_user(row) if row else None


def list_users() -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute("SELECT * FROM ac_users ORDER BY name").fetchall()
    return [_coerce_user(r) for r in rows]


def update_user(user_id: str, **fields: Any) -> bool:
    allowed = {"name", "email", "role", "domain", "is_active"}
    updates: Dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    if "role" in updates and updates["role"] not in ROLES:
        raise ValueError(f"Invalid role '{updates['role']}'. Must be one of {ROLES}")
    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k}=?" for k in updates)
    params = tuple(updates.values()) + (user_id,)
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(f"UPDATE ac_users SET {set_clause} WHERE user_id=?", params)
    return True


def delete_user(user_id: str) -> bool:
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM ac_users WHERE user_id=?", (user_id,))
    return True


def set_role(user_id: str, role: str) -> bool:
    return update_user(user_id, role=role)


# ── Access checks ─────────────────────────────────────────────────────────────

def can_access_source(user_id: str, source: Dict) -> bool:
    """
    Returns True if the user may access the given source.
    Always True in dev/test (is_enforced() == False).

    Rules (production):
      - admin role → always True
      - user.domain == "*" → True for all sources
      - source.domain empty → visible to all authenticated users
      - user.domain == source.domain → True
    """
    if not is_enforced():
        return True

    user = get_user(user_id)
    if not user or not user.get("is_active"):
        return False
    if user["role"] == "admin":
        return True

    # Check persona_access list (original per-source access control)
    persona_access = source.get("persona_access") or []
    if isinstance(persona_access, str):
        try:
            persona_access = json.loads(persona_access)
        except Exception:
            persona_access = []
    role_to_persona = {
        "viewer":  "business_user",
        "analyst": "analyst",
        "manager": "admin",
        "admin":   "admin",
    }
    persona = role_to_persona.get(user["role"], "business_user")
    if persona_access and persona not in persona_access:
        return False

    # Domain ABAC check
    user_domain   = user.get("domain", "*")
    source_domain = (source.get("domain") or "").strip()
    if user_domain == "*":
        return True
    if not source_domain:
        return True
    return user_domain == source_domain


def can_perform(user_id: str, action: str) -> bool:
    """
    Returns True if the user's role permits the given action.
    Always True in dev/test.
    """
    if not is_enforced():
        return True

    user = get_user(user_id)
    if not user or not user.get("is_active"):
        return False
    role = user.get("role", "viewer")
    return action in _ROLE_PERMISSIONS.get(role, set())


def filter_sources(user_id: str, sources: List[Dict]) -> List[Dict]:
    """Filter a list of source dicts to only those the user can access."""
    if not is_enforced():
        return sources
    return [s for s in sources if can_access_source(user_id, s)]


# ── Coerce helpers ─────────────────────────────────────────────────────────────

def _coerce_user(row: Dict) -> Dict:
    r = dict(row)
    r["is_active"] = bool(r.get("is_active", 1))
    r["permissions"] = sorted(_ROLE_PERMISSIONS.get(r.get("role", "viewer"), set()))
    return r
