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
from typing import Any, Dict, Optional

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
