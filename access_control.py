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

Generic source/item ACL (provider-agnostic — SQL and unstructured sources
alike; M365/Entra ID is just the first identity provider and connector
family that populates this model, not a special case inside it)
-----------------------------------------------------------------
  - ``ac_external_identities`` maps a DataNanite user to an external IdP
    principal (provider="entra" today, any future SSO/IdP tomorrow). This
    never changes how a user logs into DataNanite — local JWT/auth.db login
    is untouched; it only records identity for item-level ACL matching.
  - ``ac_source_acl`` holds source-level grants (by user_id, role, and/or
    domain), keyed generically by source_id — works for any source type.
    A source with no ac_source_acl rows falls back to its legacy
    persona_access list so existing sources don't lose access abruptly.
  - ``filter_items()`` is the generic result-level enforcement hook: source-
    level gating first, then item-level filtering by an ``allowed_principals``
    field on the item (e.g. doc_assets.allowed_principals) when present.

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
filter_sources(user_id, sources)         → List[dict]
filter_items(user_id, source, items)     → List[dict]  (Phase 4 enforcement)
link_external_identity(user_id, provider, external_id)   → dict
get_external_identity(user_id, provider) → dict | None
list_external_identities(provider=None)  → List[dict]
unlink_external_identity(user_id, provider) → bool
sync_external_identities(provider, pairs) → int   (bulk upsert, by email)
grant_source_access(source_id, user_id=None, role=None, domain=None) → acl_id
revoke_source_access(acl_id)             → bool
list_source_acl(source_id)               → List[dict]
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
       "edit_business_ontology", "approve_business_ontology_term",
       "manage_business_ontology_versions"}
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

# Provider-agnostic mapping of a DataNanite user to an external identity
# provider's principal (e.g. provider="entra", external_id=Entra object ID).
# One user may have at most one identity per provider; a future IdP just
# adds rows with a new `provider` value — no schema change needed.
_DDL_EXTERNAL_IDENTITIES = """
CREATE TABLE IF NOT EXISTS ac_external_identities (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    provider      TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    UNIQUE(user_id, provider)
)
"""


def _ensure(cur: Any) -> None:
    cur.ddl(_DDL_USERS, _DDL_SOURCE_ACL, _DDL_EXTERNAL_IDENTITIES)


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


# ── External identities (Phase 1: generic identity/principal model) ───────────
# Provider-agnostic — 'entra' today, any future SSO/IdP tomorrow. This never
# changes how a user logs into DataNanite (that stays local JWT/auth.db); it
# only records which external principal a DataNanite user corresponds to, so
# item-level ACLs populated from that provider (e.g. M365 doc permissions)
# can be matched back to a DataNanite user.

def link_external_identity(user_id: str, provider: str, external_id: str) -> Dict:
    """Create or update the (user_id, provider) -> external_id mapping."""
    now = time.time()
    with _cursor_ctx() as cur:
        _ensure(cur)
        existing = cur.execute(
            "SELECT id FROM ac_external_identities WHERE user_id=? AND provider=?",
            (user_id, provider),
        ).fetchone()
        if existing:
            cur.execute(
                "UPDATE ac_external_identities SET external_id=?, updated_at=? "
                "WHERE user_id=? AND provider=?",
                (external_id, now, user_id, provider),
            )
        else:
            cur.execute(
                "INSERT INTO ac_external_identities "
                "(id, user_id, provider, external_id, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)",
                (str(uuid.uuid4()), user_id, provider, external_id, now, now),
            )
    return get_external_identity(user_id, provider) or {}


def get_external_identity(user_id: str, provider: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT * FROM ac_external_identities WHERE user_id=? AND provider=?",
            (user_id, provider),
        ).fetchone()
    return dict(row) if row else None


def list_external_identities(provider: Optional[str] = None) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        if provider:
            rows = cur.execute(
                "SELECT * FROM ac_external_identities WHERE provider=? ORDER BY updated_at DESC",
                (provider,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM ac_external_identities ORDER BY updated_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def unlink_external_identity(user_id: str, provider: str) -> bool:
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "DELETE FROM ac_external_identities WHERE user_id=? AND provider=?",
            (user_id, provider),
        )
    return True


def sync_external_identities(provider: str, pairs: List[Dict]) -> int:
    """
    Bulk-upsert external identities for one provider.

    ``pairs`` — [{"email": ..., "external_id": ...}, ...], resolved by the
    caller (e.g. entra_sync.py maps Entra users/groups to their email first).
    This function itself is provider-agnostic: it only matches by email
    against ac_users and writes rows — no Graph API / M365 code here.
    Returns the number of identities linked (unmatched emails are skipped).
    """
    linked = 0
    for pair in pairs:
        email = (pair.get("email") or "").strip()
        external_id = (pair.get("external_id") or "").strip()
        if not email or not external_id:
            continue
        user = get_user_by_email(email)
        if not user:
            continue
        link_external_identity(user["user_id"], provider, external_id)
        linked += 1
    return linked


# ── Source-level ACL (Phase 2/3: generic ac_source_acl grants) ────────────────
# Works for any source_id regardless of source type (SQL or unstructured).

def grant_source_access(
    source_id: str,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    domain: Optional[str] = None,
) -> str:
    """Add a source-level grant. At least one of user_id/role/domain should
    be set — e.g. (user_id=X) grants that one user, (role='analyst') grants
    everyone with that role, (domain='Finance') grants that ABAC domain."""
    acl_id = str(uuid.uuid4())
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute(
            "INSERT INTO ac_source_acl (acl_id, source_id, user_id, role, domain, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (acl_id, source_id, user_id, role, domain, time.time()),
        )
    return acl_id


def revoke_source_access(acl_id: str) -> bool:
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("DELETE FROM ac_source_acl WHERE acl_id=?", (acl_id,))
    return True


def list_source_acl(source_id: str) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT * FROM ac_source_acl WHERE source_id=? ORDER BY created_at", (source_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def _acl_grants_access(user: Dict, acl_rows: List[Dict]) -> bool:
    """True if any ac_source_acl row for this source grants `user` access."""
    for row in acl_rows:
        if row.get("user_id") and row["user_id"] == user["user_id"]:
            return True
        if row.get("role") and row["role"] == user.get("role"):
            return True
        if row.get("domain") and row["domain"] == user.get("domain"):
            return True
    return False


# ── Access checks ─────────────────────────────────────────────────────────────

def can_access_source(user_id: str, source: Dict) -> bool:
    """
    Returns True if the user may access the given source.
    Always True in dev/test (is_enforced() == False).

    Rules (production):
      - admin role → always True
      - explicit ac_source_acl grant for this user/role/domain → True
      - no ac_source_acl rows exist for this source → fall back to the
        legacy persona_access + domain check (non-breaking during migration)
      - ac_source_acl rows exist but none match → False (explicit grants
        replace the legacy default once a source has any)
    """
    if not is_enforced():
        return True

    user = get_user(user_id)
    if not user or not user.get("is_active"):
        return False
    if user["role"] == "admin":
        return True

    source_id = source.get("id") or source.get("source_id") or ""
    acl_rows = list_source_acl(source_id) if source_id else []
    if acl_rows:
        return _acl_grants_access(user, acl_rows)

    # No generic ACL rows yet for this source — fall back to the legacy
    # persona_access list (original per-source access control) so no
    # existing source loses access abruptly during migration.
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


def _user_principal_ids(user: Dict) -> set:
    """All principal IDs `user` matches: their own user_id, plus every
    external identity linked to them (any provider — 'entra' or otherwise)."""
    ids = {user["user_id"]}
    for identity in list_external_identities():
        if identity["user_id"] == user["user_id"]:
            ids.add(identity["external_id"])
    return ids


def filter_items(
    user_id: str,
    source: Dict,
    items: List[Dict],
    principal_field: str = "allowed_principals",
) -> List[Dict]:
    """
    Generic result-level enforcement (Phase 4) — usable uniformly for
    unstructured doc-search results and structured query rows alike.

    First gates at the source level via can_access_source (same as today).
    Then, for item-level ACLs: an item whose `principal_field` is empty/
    absent is treated as open (inherits source-level gating only — this is
    the SQL row/column case for the first release, since connectors don't
    yet inject WHERE-clause predicates). An item that DOES carry a non-empty
    principal_field (e.g. an M365 doc's Graph permissions, mapped in via
    Phase 1's external identities) is only kept if the user matches one of
    those principals.

    Always a no-op (returns items unchanged) in dev/test.
    """
    if not is_enforced():
        return items
    if not can_access_source(user_id, source):
        return []

    user = get_user(user_id)
    if not user or not user.get("is_active"):
        return []
    if user["role"] == "admin":
        return items

    principal_ids = _user_principal_ids(user)

    def _visible(item: Dict) -> bool:
        allowed = item.get(principal_field)
        if isinstance(allowed, str):
            try:
                allowed = json.loads(allowed)
            except Exception:
                allowed = None
        if not allowed:
            return True
        return bool(principal_ids & set(allowed))

    return [i for i in items if _visible(i)]


# ── Coerce helpers ─────────────────────────────────────────────────────────────

def _coerce_user(row: Dict) -> Dict:
    r = dict(row)
    r["is_active"] = bool(r.get("is_active", 1))
    r["permissions"] = sorted(_ROLE_PERMISSIONS.get(r.get("role", "viewer"), set()))
    return r
