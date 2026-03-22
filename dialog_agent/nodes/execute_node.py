"""
execute_node — Run each planned SQL query against the target database.

Uses a minimal inline connector (psycopg2/pyodbc/etc.) driven by the
DialogConfig.  Errors on individual queries are captured and do not abort
the whole pipeline.

File-based sources (CSV / Excel)
---------------------------------
The first time a file path is seen, the data is loaded into a temporary
SQLite database file on disk (not :memory:, so SQLite can spill aggregation
buffers and never hits "database or disk is full").  That temp file is
**cached** keyed by the original file path so every subsequent request
reuses the already-loaded DB — no re-parsing on each query.

The cached DB is kept alive until the caller explicitly invokes
``purge_file_db(fpath)`` or ``purge_all_file_dbs()``.  The dialog_api
exposes a ``DELETE /file-cache`` endpoint that triggers the purge when the
user navigates away from the Dialog with Data section.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DialogConfig
from ..state import DialogState, QueryResult

logger = logging.getLogger(__name__)


# ── File-DB cache ─────────────────────────────────────────────────────────────
# Maps original file path → path of the pre-loaded temp SQLite DB file.
# Protected by _FILE_DB_LOCK for thread safety.

_FILE_DB_CACHE: Dict[str, str] = {}
_FILE_DB_LOCK  = threading.Lock()

_FILE_BASED_TYPES = {"sqlite", "csv", "excel"}


def purge_file_db(fpath: str) -> bool:
    """
    Delete the cached SQLite temp file for *fpath* and remove it from cache.
    Returns True if an entry was found and removed, False otherwise.
    Safe to call even if the entry does not exist.
    """
    with _FILE_DB_LOCK:
        tmp = _FILE_DB_CACHE.pop(fpath, None)
    if tmp:
        try:
            os.unlink(tmp)
            logger.info("file_db_cache: purged %s → %s", fpath, tmp)
        except Exception as exc:
            logger.warning("file_db_cache: could not delete %s — %s", tmp, exc)
        return True
    return False


def purge_all_file_dbs() -> int:
    """
    Delete all cached temp SQLite files.  Returns the number of entries removed.
    Call this when the user navigates away from the Dialog with Data section.
    """
    with _FILE_DB_LOCK:
        entries = dict(_FILE_DB_CACHE)
        _FILE_DB_CACHE.clear()
    count = 0
    for fpath, tmp in entries.items():
        try:
            os.unlink(tmp)
            logger.info("file_db_cache: purged %s → %s", fpath, tmp)
            count += 1
        except Exception as exc:
            logger.warning("file_db_cache: could not delete %s — %s", tmp, exc)
    return count


def list_file_dbs() -> Dict[str, str]:
    """Return a snapshot of {original_path: temp_db_path} for all cached DBs."""
    with _FILE_DB_LOCK:
        return dict(_FILE_DB_CACHE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_col(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return ("col_" + s if s and s[0].isdigit() else s) or "col"


def _safe_table(name: str) -> str:
    """Must match understand_node._to_sql_table so table names are consistent."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return ("t_" + s if s and s[0].isdigit() else s) or "tbl"


def _dedup_cols(df) -> None:
    """Deduplicate DataFrame column names in-place after sanitisation."""
    used: dict = {}
    new_cols = []
    for c in df.columns:
        sc = _safe_col(str(c))
        if sc in used:
            used[sc] += 1
            sc = f"{sc}_{used[sc]}"
        else:
            used[sc] = 1
        new_cols.append(sc)
    df.columns = new_cols


def _load_file_to_db(db_type: str, fpath: str, tmp_path: str) -> None:
    """
    Load a CSV directory or Excel file into the SQLite database at *tmp_path*.
    Raises on unrecoverable errors so the caller can fall back gracefully.
    """
    import pandas as pd

    conn = sqlite3.connect(tmp_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        if db_type == "csv":
            for f in sorted(Path(fpath).glob("*.csv")):
                try:
                    df = pd.read_csv(f)
                    _dedup_cols(df)
                    tname = _safe_table(f.stem)
                    df.to_sql(tname, conn, if_exists="replace", index=False)
                    logger.info("file_db: loaded CSV %s → %s (%d rows, %d cols)",
                                f.name, tname, len(df), len(df.columns))
                except Exception as exc:
                    logger.warning("file_db: skipping CSV %s — %s", f.name, exc)
        else:  # excel
            xl = pd.ExcelFile(fpath)
            used_sheets: dict = {}
            for sheet in xl.sheet_names:
                base = _safe_table(sheet)
                if base in used_sheets:
                    used_sheets[base] += 1
                    safe = f"{base}_{used_sheets[base]}"
                else:
                    used_sheets[base] = 1
                    safe = base
                try:
                    df = xl.parse(sheet)
                    _dedup_cols(df)
                    df.to_sql(safe, conn, if_exists="replace", index=False)
                    logger.info("file_db: loaded sheet %r → %r (%d rows, %d cols)",
                                sheet, safe, len(df), len(df.columns))
                except Exception as exc:
                    logger.warning("file_db: skipping sheet %r → %r — %s",
                                   sheet, safe, exc)

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        loaded = [r[0] for r in cur.fetchall()]
        cur.close()
        logger.info("file_db: tables in %s: %s", tmp_path, loaded)
    finally:
        conn.close()


def _get_cached_file_db(cfg: DialogConfig) -> str:
    """
    Return the path of the pre-loaded temp SQLite DB for *cfg.db_file_path*.
    Builds and caches on first call; returns instantly on subsequent calls.
    """
    fpath = cfg.db_file_path
    db    = cfg.db_type.lower()

    with _FILE_DB_LOCK:
        if fpath in _FILE_DB_CACHE:
            tmp = _FILE_DB_CACHE[fpath]
            if os.path.exists(tmp):
                logger.info("file_db_cache: HIT %s → %s", fpath, tmp)
                return tmp
            # Stale entry (file was deleted externally) — rebuild
            logger.warning("file_db_cache: stale entry for %s, rebuilding", fpath)
            _FILE_DB_CACHE.pop(fpath, None)

        # Build the temp DB (lock held so only one thread loads it)
        tmp_f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_f.close()
        tmp_path = tmp_f.name

    try:
        logger.info("file_db_cache: MISS %s — loading into %s", fpath, tmp_path)
        _load_file_to_db(db, fpath, tmp_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise

    with _FILE_DB_LOCK:
        _FILE_DB_CACHE[fpath] = tmp_path

    return tmp_path


def _open_file_conn(cfg: DialogConfig) -> sqlite3.Connection:
    """Open (or reuse) a connection to the cached temp SQLite DB."""
    if cfg.db_type.lower() == "sqlite":
        conn = sqlite3.connect(cfg.db_file_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    tmp_path = _get_cached_file_db(cfg)
    conn = sqlite3.connect(tmp_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ── Thin DB runner ────────────────────────────────────────────────────────────

def _run_sql(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    """
    Execute *sql* and return {columns, rows, error}.
    Supports postgres/redshift, oracle, sqlserver, bigquery, sqlite, csv, excel.
    """
    db = cfg.db_type.lower()
    try:
        if db in ("postgres", "redshift"):
            return _run_postgres(cfg, sql)
        elif db == "oracle":
            return _run_oracle(cfg, sql)
        elif db == "sqlserver":
            return _run_sqlserver(cfg, sql)
        elif db == "bigquery":
            return _run_bigquery(cfg, sql)
        else:
            return {"columns": [], "rows": [], "error": f"Unsupported db_type: {db}"}
    except Exception as exc:
        logger.exception("execute_node: SQL failed")
        return {"columns": [], "rows": [], "error": str(exc)}


def _cursor_to_result(cursor) -> Dict[str, Any]:
    columns = [desc[0] for desc in (cursor.description or [])]
    rows    = [list(r) for r in (cursor.fetchall() or [])]
    return {"columns": columns, "rows": rows, "error": None}


def _run_postgres(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    import psycopg2
    import psycopg2.extras

    if cfg.db_connection_string:
        conn = psycopg2.connect(cfg.db_connection_string)
    else:
        conn = psycopg2.connect(
            host=cfg.db_host, port=cfg.db_port or 5432,
            dbname=cfg.db_name, user=cfg.db_user, password=cfg.db_password,
            **cfg.db_extra,
        )
    conn.autocommit = True
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if cfg.db_schema:
                cur.execute(f"SET search_path TO {cfg.db_schema}, public")
            cur.execute(sql)
            columns = [desc[0] for desc in (cur.description or [])]
            rows    = [[row[c] for c in columns] for row in (cur.fetchall() or [])]
            return {"columns": columns, "rows": rows, "error": None}
    finally:
        conn.close()


def _run_oracle(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    import cx_Oracle

    if cfg.db_connection_string:
        conn = cx_Oracle.connect(cfg.db_connection_string)
    else:
        dsn  = cx_Oracle.makedsn(cfg.db_host, cfg.db_port or 1521,
                                  service_name=cfg.db_name)
        conn = cx_Oracle.connect(cfg.db_user, cfg.db_password, dsn)
    try:
        with conn.cursor() as cur:
            if cfg.db_schema:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {cfg.db_schema}")
            cur.execute(sql)
            return _cursor_to_result(cur)
    finally:
        conn.close()


def _run_sqlserver(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    import pyodbc

    if cfg.db_connection_string:
        conn = pyodbc.connect(cfg.db_connection_string)
    else:
        driver = cfg.db_extra.get("driver", "ODBC Driver 18 for SQL Server")
        conn = pyodbc.connect(
            f"DRIVER={{{driver}}};SERVER={cfg.db_host},{cfg.db_port or 1433};"
            f"DATABASE={cfg.db_name};UID={cfg.db_user};PWD={cfg.db_password}"
        )
    try:
        with conn.cursor() as cur:
            if cfg.db_schema:
                cur.execute(f"USE [{cfg.db_name}]")
                cur.execute(f"SET SCHEMA [{cfg.db_schema}]")
            cur.execute(sql)
            return _cursor_to_result(cur)
    finally:
        conn.close()


def _run_bigquery(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    from google.cloud import bigquery

    kwargs: Dict[str, Any] = {}
    if cfg.db_extra.get("credentials_path"):
        from google.oauth2 import service_account
        kwargs["credentials"] = service_account.Credentials.from_service_account_file(
            cfg.db_extra["credentials_path"]
        )
    client  = bigquery.Client(project=cfg.db_name or cfg.db_extra.get("project"), **kwargs)
    job     = client.query(sql)
    results = job.result()
    columns = [f.name for f in results.schema]
    rows    = [[r[c] for c in columns] for r in results]
    return {"columns": columns, "rows": rows, "error": None}


def _exec_on_conn(conn, sql: str) -> Dict[str, Any]:
    """Run one SQL statement on an already-open sqlite3 connection."""
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in (cur.description or [])]
        rows = [list(r) for r in (cur.fetchall() or [])]
        cur.close()
        return {"columns": cols, "rows": rows, "error": None}
    except sqlite3.Error as exc:
        msg = str(exc)
        if "no such table" in msg.lower():
            try:
                c2 = conn.cursor()
                c2.execute("SELECT name FROM sqlite_master WHERE type='table'")
                available = [r[0] for r in c2.fetchall()]
                c2.close()
                msg += f" (available tables: {available})"
            except Exception:
                pass
        return {"columns": [], "rows": [], "error": msg}
    except Exception as exc:
        return {"columns": [], "rows": [], "error": str(exc)}


# ── node ──────────────────────────────────────────────────────────────────────

def execute_node(state: DialogState) -> DialogState:
    """Execute all planned SQL queries and store results."""
    logger.info("=== execute_node ===")

    config      = state["config"]
    sql_queries = state.get("sql_queries") or []

    if not sql_queries:
        logger.info("execute_node: no queries to run")
        state["query_results"] = []
        state["phase"] = "execute"
        return state

    # For file-based sources open a connection to the (cached) temp SQLite DB.
    # The DB file persists until explicitly purged via purge_file_db() /
    # purge_all_file_dbs() — it is NOT deleted after each request.
    file_conn = None
    if config.db_type.lower() in _FILE_BASED_TYPES:
        try:
            file_conn = _open_file_conn(config)
        except Exception as exc:
            logger.exception("execute_node: failed to open file connection")
            state["errors"].append(f"execute_node: could not load file — {exc}")
            state["query_results"] = []
            state["phase"] = "execute"
            return state

    results: List[QueryResult] = []

    try:
        for q in sql_queries:
            logger.info("  Running %s: %s", q["query_id"], q["description"])
            if file_conn is not None:
                outcome = _exec_on_conn(file_conn, q["sql"])
            else:
                outcome = _run_sql(config, q["sql"])

            rows      = outcome.get("rows") or []
            columns   = outcome.get("columns") or []
            error_msg: Optional[str] = outcome.get("error")

            # For aggregation queries, return all rows — truncating would silently
            # drop groups and produce wrong totals.  For raw-row queries the SQL
            # already has a LIMIT clause added by plan_node, so Python truncation
            # is redundant; we keep it only as a safety cap for very large raw dumps.
            _is_agg = bool(re.search(
                r'\b(GROUP\s+BY|COUNT\s*\(|SUM\s*\(|AVG\s*\(|MIN\s*\(|MAX\s*\()\b',
                q.get("sql", ""), re.IGNORECASE,
            ))
            returned_rows = rows if _is_agg else rows[: config.row_limit]

            results.append(
                QueryResult(
                    query_id    = q["query_id"],
                    description = q["description"],
                    sql         = q["sql"],
                    columns     = columns,
                    rows        = returned_rows,
                    row_count   = len(rows),
                    error       = error_msg,
                )
            )

            if error_msg:
                logger.warning("  %s FAILED: %s", q["query_id"], error_msg)
                state["errors"].append(f"execute_node [{q['query_id']}]: {error_msg}")
            else:
                logger.info("  %s OK — %d rows returned", q["query_id"], len(rows))

    finally:
        # Close the connection to the cached DB — the DB file itself stays on disk
        if file_conn is not None:
            try:
                file_conn.close()
            except Exception:
                pass

    state["query_results"] = results
    state["phase"] = "execute"
    return state
