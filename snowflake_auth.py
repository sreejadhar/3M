"""Shared Snowflake connection helper — supports BOTH auth modes.

Used by the metadata-extraction connector (``connectors/snowflake.py``) and the
chat SQL runner (``dialog_agent/nodes/execute_node.py``).

Two ways to authenticate — selected by whether ``extra["connection_name"]`` is set:

1. PASSWORD (default; what the Register-Data-Source UI form uses): connect with
   account (derived from the host) + user + password, using EXACTLY the
   credentials supplied. There is **no fallback** to connections.toml — a blank
   password is sent as-is and Snowflake rejects it. Warehouse/role are left to
   the user's Snowflake defaults unless explicitly provided via ``extra``.

2. KEY-PAIR (only when ``extra["connection_name"]`` is explicitly set): use that
   named connection in ``~/.snowflake/connections.toml`` (account, user,
   private_key_file, role, warehouse) plus the encrypted-key passphrase, read at
   connect time and never persisted — from the ``SNOWFLAKE_PRIVATE_KEY_FILE_PWD``
   env var, then Windows Credential Manager (service ``synthgen`` / user
   ``snowflake_pk_passphrase``) via ``keyring``, then a raw Win32 ``CredRead``.

Overridable through ``extra``: ``connection_name``, ``account``, ``user``,
``warehouse``, ``role``, ``private_key_file``, ``authenticator``,
``cred_service``, ``cred_user``.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_NAME = "synthgen"
_CRED_SERVICE = "synthgen"
_CRED_USER = "snowflake_pk_passphrase"
_DEFAULT_WAREHOUSE = "COMPUTE_WH"
_DEFAULT_ROLE = "ACCOUNTADMIN"

_PASSTHROUGH_KEYS = (
    "account", "user", "warehouse", "role",
    "private_key_file", "authenticator", "host", "region",
)


def account_from_host(host: str) -> str:
    """kk70490.ap-south-1.aws.snowflakecomputing.com -> kk70490.ap-south-1.aws"""
    h = (host or "").strip().lower()
    suffix = ".snowflakecomputing.com"
    if h.endswith(suffix):
        h = h[: -len(suffix)]
    return h


def read_passphrase(cred_service: str = _CRED_SERVICE,
                    cred_user: str = _CRED_USER) -> Optional[str]:
    """Return the private-key passphrase, or None if not found anywhere."""
    pw = os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD")
    if pw:
        return pw
    try:
        import keyring
        pw = keyring.get_password(cred_service, cred_user)
        if pw:
            return pw
    except Exception as exc:  # pragma: no cover
        logger.debug("keyring passphrase read failed: %s", exc)
    try:
        import ctypes
        from ctypes import wintypes

        CRED_TYPE_GENERIC = 1
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)

        class _CRED(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD),
                ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR),
                ("Comment", wintypes.LPWSTR),
                ("LastWritten", wintypes.FILETIME),
                ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_char)),
                ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD),
                ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR),
                ("UserName", wintypes.LPWSTR),
            ]

        ptr = ctypes.POINTER(_CRED)()
        if advapi.CredReadW(cred_service, CRED_TYPE_GENERIC, 0, ctypes.byref(ptr)):
            cred = ptr.contents
            blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
            for enc in ("utf-16-le", "utf-8"):
                try:
                    return blob.decode(enc)
                except Exception:
                    continue
    except Exception as exc:  # pragma: no cover
        logger.debug("CredRead passphrase read failed: %s", exc)
    return None


def build_connect_params(*, database: str = "", schema: str = "",
                         username: str = "", password: str = "",
                         host: str = "", port: int = 0,
                         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble kwargs for ``snowflake.connector.connect``."""
    extra = dict(extra or {})
    params: Dict[str, Any] = {}

    password = password or extra.get("password", "")
    explicit_conn = extra.get("connection_name")

    if explicit_conn:
        # ── Named-connection / key-pair auth — ONLY when a connection name is
        #    explicitly requested via extra. The Register-Data-Source form never
        #    sets this, so it never silently falls back to connections.toml. ──
        params["connection_name"] = explicit_conn
        for key in _PASSTHROUGH_KEYS:
            if extra.get(key):
                params[key] = extra[key]
        if username and "user" not in params:
            params["user"] = username
        pw = read_passphrase(
            extra.get("cred_service", _CRED_SERVICE),
            extra.get("cred_user", _CRED_USER),
        )
        if pw:
            params.setdefault("private_key_file_pwd", pw)
    else:
        # ── Password auth (form path) — uses EXACTLY the credentials supplied.
        #    No connections.toml / key-pair fallback. A blank password is sent
        #    as-is and Snowflake rejects it (no silent key-file login). ──
        account = extra.get("account") or account_from_host(host or extra.get("host", ""))
        if account:
            params["account"] = account
        if username:
            params["user"] = username
        if password:
            params["password"] = password
        # warehouse/role only if explicitly provided; otherwise Snowflake uses
        # the user's DEFAULT_WAREHOUSE / DEFAULT_ROLE.
        for key in ("warehouse", "role", "authenticator"):
            if extra.get(key):
                params[key] = extra[key]

    if database:
        params["database"] = database
    if schema:
        params["schema"] = schema
    return params


def connect_snowflake(*, database: str = "", schema: str = "",
                      username: str = "", password: str = "",
                      host: str = "", port: int = 0,
                      extra: Optional[Dict[str, Any]] = None):
    """Open and return a live ``snowflake.connector`` connection."""
    import snowflake.connector
    params = build_connect_params(
        database=database, schema=schema, username=username, password=password,
        host=host, port=port, extra=extra,
    )
    return snowflake.connector.connect(**params)


# ── Process-wide connection pool ────────────────────────────────────────────
# Opening a Snowflake connection (auth handshake + warehouse resume-from-
# suspended) routinely takes 15-20s — far more than Postgres/MySQL's near-
# instant TCP connect. dialog_agent's execute_node used to open a brand-new
# connection per SQL statement (often 3-4 per chat turn: probes, the real
# query, self-heal retries), so that overhead was being paid over and over
# within a single request. This pool hands out an idle, still-fresh
# connection when one exists for the same (account/db/schema/warehouse/role)
# key, and only pays the handshake cost when the pool is empty.
#
# Liveness is NOT checked with a round-trip probe at checkout (that would
# just move the latency problem, not remove it) — instead connections older
# than _POOL_MAX_AGE_SECONDS are discarded on checkout, and the caller MUST
# report checkout failures via release_pooled_connection(..., healthy=False)
# so a connection that errored (session expired, warehouse suspended again,
# network blip) is closed and never handed to the next caller.
_POOL_LOCK = threading.Lock()
_POOL: Dict[Tuple, list] = {}
_POOL_MAX_IDLE_PER_KEY = 4
_POOL_MAX_AGE_SECONDS = 1500  # 25 min — comfortably under typical session/token TTLs


def _pool_key(database: str, schema: str, username: str, host: str, port: int,
              extra: Optional[Dict[str, Any]]) -> Tuple:
    extra = extra or {}
    return (
        database, schema, username, host, port,
        extra.get("account", ""), extra.get("warehouse", ""), extra.get("role", ""),
        extra.get("connection_name", ""),
    )


def _safe_close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def get_pooled_connection(*, database: str = "", schema: str = "",
                           username: str = "", password: str = "",
                           host: str = "", port: int = 0,
                           extra: Optional[Dict[str, Any]] = None):
    """Borrow a connection for exclusive use — MUST be paired with
    release_pooled_connection(key, conn, healthy=...) when done. Returns
    (conn, key)."""
    key = _pool_key(database, schema, username, host, port, extra)
    now = time.monotonic()
    with _POOL_LOCK:
        bucket = _POOL.get(key) or []
        while bucket:
            conn, opened_at = bucket.pop()
            if now - opened_at > _POOL_MAX_AGE_SECONDS:
                _safe_close(conn)
                continue
            logger.debug("snowflake_auth: reusing pooled connection for key=%s", key)
            return conn, key
    conn = connect_snowflake(
        database=database, schema=schema, username=username, password=password,
        host=host, port=port, extra=extra,
    )
    return conn, key


def release_pooled_connection(key: Tuple, conn, healthy: bool = True) -> None:
    """Return a borrowed connection to the pool, or close it if it errored
    or the pool for this key is already at capacity."""
    if not healthy:
        _safe_close(conn)
        return
    with _POOL_LOCK:
        bucket = _POOL.setdefault(key, [])
        if len(bucket) >= _POOL_MAX_IDLE_PER_KEY:
            _safe_close(conn)
        else:
            bucket.append((conn, time.monotonic()))
