"""
Shared AWS RDS PostgreSQL connection info for every DataNanite store
(kg_store.py, metadata_catalog.py, kpi_store.py, glossary_registry.py,
abbrev_glossary_registry.py, business_ontology.py, access_control.py,
session_store.py, dialog_agent/pg_store.py).

Production fetches credentials from AWS Secrets Manager at runtime, using the
same assumed cross-account role as Bedrock/Neptune (see aws_auth.py). Secrets
Manager handles automatic password rotation; connect() below retries once
with a forced refresh on an auth failure, so a mid-session rotation doesn't
require restarting the app.

Dev is different: the dev RDS instance is a private, PinAD-secured instance
reachable only from the app's own EC2 host, so there's no cross-account role
to assume it through. Dev credentials instead live in the .env file on that
EC2 instance (PG_USER/PG_PASSWORD) and are read directly — no Secrets
Manager call is made when they're set.

Config:
  PG_USER, PG_PASSWORD — dev-only credentials read straight from the EC2
                    .env file; when both are set, Secrets Manager is
                    bypassed entirely and a DSN is built from these plus
                    PG_HOST/PG_PORT/PG_DATABASE below.
  PG_SECRET_ID    — Secrets Manager secret id/ARN holding {"username",
                    "password"} for the app login (production; default: see
                    below). Ignored when PG_USER/PG_PASSWORD are set.
  PG_HOST         — RDS endpoint (default: pg-rds-datananite-dev-001a...)
  PG_PORT         — default 5432
  PG_DATABASE     — default "datananite"
  KG_POSTGRES_DSN — explicit DSN override (local dev only) — when set, this
                    takes priority over everything else below.
"""
from __future__ import annotations

import json
import os
import threading
import time

from aws_auth import get_session

_SECRET_ID = os.environ.get("PG_SECRET_ID", "datananite/rds/app-user")
_HOST = os.environ.get("PG_HOST", "pg-rds-datananite-dev-001a.cuepp5apko9u.us-east-1.rds.amazonaws.com")
_PORT = os.environ.get("PG_PORT", "5432")
_DATABASE = os.environ.get("PG_DATABASE", "datananite")

# Re-fetched periodically (not just on auth failure) so a rotated password is
# picked up promptly even for long-lived connection pools that reuse a cached
# DSN across many short-lived connections.
_TTL_SECONDS = 300

_lock = threading.Lock()
_cached_dsn: "str | None" = None
_cached_at = 0.0


def _fetch_secret() -> dict:
    client = get_session().client("secretsmanager")
    resp = client.get_secret_value(SecretId=_SECRET_ID)
    return json.loads(resp["SecretString"])


def get_pg_dsn(force_refresh: bool = False) -> str:
    """Return a psycopg2 DSN for the shared RDS instance."""
    override = os.environ.get("KG_POSTGRES_DSN", "").strip()
    if override:
        return override

    # Dev: private, PinAD-secured RDS instance reachable only from this EC2
    # host — credentials come from the box's own .env, not Secrets Manager.
    dev_user = os.environ.get("PG_USER", "").strip()
    dev_password = os.environ.get("PG_PASSWORD", "").strip()
    if dev_user and dev_password:
        return f"postgresql://{dev_user}:{dev_password}@{_HOST}:{_PORT}/{_DATABASE}"

    global _cached_dsn, _cached_at
    with _lock:
        if not force_refresh and _cached_dsn is not None and time.time() - _cached_at < _TTL_SECONDS:
            return _cached_dsn

        secret = _fetch_secret()
        user = secret["username"]
        password = secret["password"]
        host = secret.get("host", _HOST)
        port = secret.get("port", _PORT)
        dbname = secret.get("dbname", _DATABASE)
        _cached_dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        _cached_at = time.time()
        return _cached_dsn


def connect(**kwargs):
    """psycopg2.connect() using the current secret, transparently retrying
    once with a forced credential refresh if the cached password was just
    rotated out from under us."""
    import psycopg2

    try:
        return psycopg2.connect(get_pg_dsn(), **kwargs)
    except psycopg2.OperationalError as exc:
        if "password authentication failed" in str(exc).lower():
            return psycopg2.connect(get_pg_dsn(force_refresh=True), **kwargs)
        raise
