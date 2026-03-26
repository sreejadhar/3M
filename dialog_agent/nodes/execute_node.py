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


# ── Self-healing SQL retry ────────────────────────────────────────────────────
#
# When a query fails, we check whether the error looks like a fixable type-
# coercion / SQL-dialect problem.  If so, we ask the LLM to patch the SQL and
# retry — up to _MAX_RETRIES times per query.
#
# We never retry errors rooted in the *data* or *schema* (missing tables /
# columns, division-by-zero, constraint violations, auth/connection failures)
# because the LLM cannot invent rows or tables that don't exist.

_MAX_RETRIES = 3

# Errors the LLM can plausibly fix by adjusting casts, dialect syntax, or
# column qualification.
_RETRYABLE_RE = re.compile(
    r"invalid input syntax for type"           # PostgreSQL: string → int cast
    r"|cannot cast type"                        # PostgreSQL: explicit cast fail
    r"|operator does not exist"                 # PostgreSQL: type mismatch in comparison
    r"|could not convert"
    r"|invalid text representation"             # PostgreSQL
    r"|value .{0,40} out of range for type"     # PostgreSQL: numeric overflow
    r"|value too long for type"                 # PostgreSQL: string truncation
    r"|conversion failed when converting"       # SQL Server: CAST / CONVERT fail
    r"|operand type clash"                      # SQL Server: incompatible types
    r"|error converting data type"              # SQL Server
    r"|ORA-01722"                               # Oracle: invalid number
    r"|ORA-06502"                               # Oracle: value / program error
    r"|ORA-01858"                               # Oracle: non-numeric char
    r"|ORA-01843"                               # Oracle: not a valid month
    r"|datatype mismatch"                       # SQLite
    r"|type mismatch"
    r"|incompatible data type"
    r"|invalid cast"
    r"|cannot be cast"
    r"|ambiguous column"                        # qualify with table name
    r"|syntax error"                            # generic / PostgreSQL / SQLite
    r"|incorrect syntax"                        # SQL Server: "Incorrect syntax near ..."
    r"|no such function"                        # wrong function for this DB
    r"|is not a recognized .{0,30} function"    # SQL Server: wrong function name
    r"|is not a recognized built-in"            # SQL Server
    r"|numeric value out of range"
    r"|out of range"
    r"|truncated incorrect"                     # MySQL/MariaDB compat
    r"|not supported"                           # feature not available in this dialect
    r"|does not support"
    r"|invalid use of"                          # SQL Server: e.g. window fn in WHERE
    r"|windowed functions can only"             # SQL Server: window fn placement error
    , re.IGNORECASE,
)

# Errors rooted in actual data / schema / infrastructure — the LLM cannot fix
# these by changing SQL syntax.
_NOT_RETRYABLE_RE = re.compile(
    r"no such table"
    r"|relation .{0,80} does not exist"
    r"|table .{0,80} doesn't exist"
    r"|invalid object name"                     # SQL Server: table not found
    r"|no such column"
    r"|column .{0,80} does not exist"
    r"|unknown column"
    r"|invalid column name"                     # SQL Server: column not found
    r"|division by zero"
    r"|divide by zero"
    r"|unique constraint"
    r"|primary key constraint"
    r"|foreign key constraint"
    r"|not null constraint"
    r"|check constraint"
    r"|permission denied"
    r"|access denied"
    r"|login fail"
    r"|authentication"
    r"|connect"                                 # connection / cannot connect
    r"|timeout"
    r"|server closed"
    r"|out of memory"
    , re.IGNORECASE,
)


def _is_retryable(error: str) -> bool:
    """Return True only if the error looks like a fixable type / dialect issue."""
    if _NOT_RETRYABLE_RE.search(error):
        return False
    return bool(_RETRYABLE_RE.search(error))


def _llm_fix_sql(
    sql: str,
    error: str,
    db_type: str,
    model: str,
    temperature: float,
) -> str:
    """
    Ask the LLM to repair *sql* based on the reported *error*.
    Returns the corrected SQL, or the original if the LLM call fails or returns
    the same SQL.
    """
    import os
    import anthropic

    system = (
        "You are a SQL expert. Fix the type-conversion, type-casting, or SQL-dialect "
        "error in the query below so it executes without error.\n"
        "Rules:\n"
        "1. Change ONLY what is necessary to fix the reported error — preserve all logic.\n"
        "2. Use correct syntax for the target database type.\n"
        "3. If an integer/numeric value is compared to a text column, add an explicit CAST.\n"
        "4. If the dialect is wrong (e.g. LIMIT used in SQL Server), rewrite to the correct form.\n"
        "5. Return ONLY the corrected SQL — no explanation, no markdown fences."
    )
    user = (
        f"Target database: {db_type}\n"
        f"Error message:\n{error}\n\n"
        f"Original SQL:\n{sql}\n\n"
        "Return the corrected SQL only."
    )
    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        fixed = (msg.content[0].text if msg.content else "").strip()
        # Strip accidental markdown fences
        fixed = re.sub(r"^```[a-z]*\s*", "", fixed, flags=re.IGNORECASE)
        fixed = re.sub(r"\s*```$", "", fixed).strip()
        return fixed if fixed else sql
    except Exception as exc:
        logger.warning("self-heal: LLM fix call failed — %s", exc)
        return sql


def _run_sql_with_retry(cfg: "DialogConfig", sql: str, state: Optional[Dict] = None, kg_id: str = "") -> Dict[str, Any]:
    """
    Execute *sql* against a live DB, retrying up to _MAX_RETRIES times when
    the failure looks like a fixable type-coercion / dialect error.
    """
    current_sql = sql
    last_result: Dict[str, Any] = {}

    for attempt in range(_MAX_RETRIES + 1):   # attempt 0 = original
        result = _run_sql(cfg, current_sql, state=state, kg_id=kg_id)
        if not result.get("error"):
            if attempt > 0:
                logger.info("self-heal: query fixed after %d attempt(s)", attempt)
            return result

        error_msg: str = result["error"] or ""
        last_result = result

        if attempt == _MAX_RETRIES:
            logger.info("self-heal: max retries (%d) exhausted", _MAX_RETRIES)
            break

        if not _is_retryable(error_msg):
            logger.info("self-heal: non-retryable error — %s", error_msg[:120])
            break

        logger.info(
            "self-heal: attempt %d/%d — type/dialect error, asking LLM to fix SQL\n"
            "  error: %s",
            attempt + 1, _MAX_RETRIES, error_msg[:200],
        )
        fixed = _llm_fix_sql(
            current_sql, error_msg, cfg.db_type,
            cfg.plan_llm_model, cfg.llm_temperature,
        )
        if fixed == current_sql:
            logger.info("self-heal: LLM returned unchanged SQL — stopping")
            break
        current_sql = fixed

    return last_result


def _exec_on_conn_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    cfg: "DialogConfig",
) -> Dict[str, Any]:
    """
    Like _exec_on_conn but with LLM-based self-healing for type / dialect errors.
    """
    current_sql = sql
    last_result: Dict[str, Any] = {}

    for attempt in range(_MAX_RETRIES + 1):
        result = _exec_on_conn(conn, current_sql)
        if not result.get("error"):
            if attempt > 0:
                logger.info("self-heal (file-db): query fixed after %d attempt(s)", attempt)
            return result

        error_msg: str = result["error"] or ""
        last_result = result

        if attempt == _MAX_RETRIES:
            break

        if not _is_retryable(error_msg):
            logger.info("self-heal (file-db): non-retryable — %s", error_msg[:120])
            break

        logger.info(
            "self-heal (file-db): attempt %d/%d — asking LLM to fix SQL\n"
            "  error: %s",
            attempt + 1, _MAX_RETRIES, error_msg[:200],
        )
        fixed = _llm_fix_sql(
            current_sql, error_msg, cfg.db_type,
            cfg.plan_llm_model, cfg.llm_temperature,
        )
        if fixed == current_sql:
            break
        current_sql = fixed

    return last_result


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

def _run_sql(cfg: DialogConfig, sql: str, state: Optional[Dict] = None, kg_id: str = "") -> Dict[str, Any]:
    """
    Execute *sql* and return {columns, rows, error}.
    Supports postgres/redshift, oracle, sqlserver, bigquery, sqlite, csv, excel.

    Multi-KG: if kg_id is set and cfg.db_type is empty, look up the right config
    from state["multi_kg_configs"] matching that kg_id.
    """
    # Multi-KG config routing
    if kg_id and cfg.db_type == "" and state:
        multi_kg_configs = state.get("multi_kg_configs") or []
        for entry in multi_kg_configs:
            if entry.get("kg_id") == kg_id:
                cfg = entry["config"]
                break

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
        elif db == "teradata":
            return _run_teradata(cfg, sql)
        elif db in ("delta_lake", "databricks"):
            return _run_databricks(cfg, sql)
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
    # Try oracledb first (modern thin-mode driver, no Instant Client required),
    # fall back to cx_Oracle (legacy).
    try:
        import oracledb as cx  # type: ignore[import-untyped]
    except ImportError:
        try:
            import cx_Oracle as cx  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "No Oracle driver found. "
                "Run: pip install oracledb  (recommended, no Instant Client needed for thin mode)\n"
                "or:  pip install cx_Oracle  (legacy, requires Oracle Instant Client)"
            )

    if cfg.db_connection_string:
        conn = cx.connect(cfg.db_connection_string)
    else:
        dsn = f"{cfg.db_host}:{cfg.db_port or 1521}/{cfg.db_name}"
        conn = cx.connect(user=cfg.db_user, password=cfg.db_password, dsn=dsn)
    try:
        with conn.cursor() as cur:
            if cfg.db_schema:
                cur.execute(f"ALTER SESSION SET CURRENT_SCHEMA = {cfg.db_schema}")
            cur.execute(sql)
            return _cursor_to_result(cur)
    finally:
        conn.close()


def _run_sqlserver(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    # pymssql is tried first: pure-Python wheel, no ODBC driver installation needed.
    # pyodbc is the fallback for environments that have the Microsoft ODBC Driver installed.
    try:
        import pymssql as _mssql  # type: ignore[import-untyped]
        _driver_name = "pymssql"
    except ImportError:
        _mssql = None  # type: ignore[assignment]
        _driver_name = "pyodbc"

    if _driver_name == "pymssql":
        if cfg.db_connection_string:
            # pymssql doesn't accept a raw ODBC connection string; fall through to pyodbc
            _driver_name = "pyodbc"
        else:
            conn = _mssql.connect(
                server=cfg.db_host,
                port=int(cfg.db_port or 1433),
                user=cfg.db_user,
                password=cfg.db_password,
                database=cfg.db_name,
            )

    if _driver_name == "pyodbc":
        try:
            import pyodbc  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "No SQL Server driver found. "
                "Easiest fix — run:  pip install pymssql  (pure-Python, no system setup needed)\n"
                "Or for enterprise ODBC:  pip install pyodbc  "
                "(then install 'ODBC Driver 18 for SQL Server' at the OS level)"
            )
        if cfg.db_connection_string:
            conn = pyodbc.connect(cfg.db_connection_string)
        else:
            odbc_driver = cfg.db_extra.get("driver", "ODBC Driver 18 for SQL Server")
            conn = pyodbc.connect(
                f"DRIVER={{{odbc_driver}}};SERVER={cfg.db_host},{cfg.db_port or 1433};"
                f"DATABASE={cfg.db_name};UID={cfg.db_user};PWD={cfg.db_password}"
            )
    try:
        with conn.cursor() as cur:
            # SQL Server has no session-level SET SCHEMA; schema is embedded in
            # table names by the planner (e.g. [dbo].[orders]).
            # USE switches the active database — harmless if already connected to it.
            if cfg.db_name:
                cur.execute(f"USE [{cfg.db_name}]")
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


def _run_teradata(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    try:
        import teradatasql  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "teradatasql is not installed. Run: pip install teradatasql"
        )

    params: Dict[str, Any] = {
        "host": cfg.db_host,
        "user": cfg.db_user,
        "password": cfg.db_password,
        "logmech": cfg.db_extra.get("logmech", "TD2"),
    }
    if cfg.db_name:
        params["database"] = cfg.db_name
    params.update({k: v for k, v in cfg.db_extra.items() if k != "logmech"})

    conn = teradatasql.connect(**params)
    cur = conn.cursor()
    try:
        # Teradata: switch default database so unqualified table names resolve
        if cfg.db_schema and cfg.db_schema != cfg.db_name:
            cur.execute(f"DATABASE {cfg.db_schema}")
        cur.execute(sql)
        return _cursor_to_result(cur)
    finally:
        cur.close()
        conn.close()


def _run_databricks(cfg: DialogConfig, sql: str) -> Dict[str, Any]:
    if cfg.db_host:
        # Databricks SQL via REST / HTTP transport
        try:
            from databricks import sql as dbsql  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "databricks-sql-connector is not installed. "
                "Run: pip install databricks-sql-connector"
            )
        token = cfg.db_password or cfg.db_extra.get("token", "")
        conn = dbsql.connect(
            server_hostname=cfg.db_host,
            http_path=cfg.db_extra.get("http_path", ""),
            access_token=token,
        )
        try:
            with conn.cursor() as cur:
                if cfg.db_schema:
                    cur.execute(f"USE {cfg.db_schema}")
                cur.execute(sql)
                return _cursor_to_result(cur)
        finally:
            conn.close()
    else:
        # Local PySpark + Delta Lake
        try:
            from pyspark.sql import SparkSession  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pyspark is not installed. Run: pip install pyspark delta-spark"
            )
        master = cfg.db_extra.get("spark_master", "local[*]")
        spark = (
            SparkSession.builder
            .master(master)
            .appName("DialogAgent")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                    "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate()
        )
        # Do NOT call spark.stop() — SparkSession is a process-level singleton
        if cfg.db_schema:
            spark.sql(f"USE {cfg.db_schema}")
        df = spark.sql(sql)
        columns = df.columns
        rows = [[r[c] for c in columns] for r in df.collect()]
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


# ── Federation merge ──────────────────────────────────────────────────────────

def _federation_merge(results: List[QueryResult], bridges: List[Dict]) -> List[QueryResult]:
    """
    Try to join QueryResult pairs that have a matching bridge key.
    Uses pandas if available, otherwise returns results unchanged.
    Falls back gracefully — never raises.
    """
    if not bridges or len(results) < 2:
        return results
    try:
        import pandas as pd

        # Index results by query_id for quick lookup
        result_map: Dict[str, QueryResult] = {r["query_id"]: r for r in results}
        merged_ids: set = set()
        merged_results: List[QueryResult] = []

        for bridge in bridges:
            from_col = bridge.get("from_column", "")
            to_col   = bridge.get("to_column", "")
            if not from_col or not to_col:
                continue

            # Find result pairs that share the bridge column
            left_result: Optional[QueryResult] = None
            right_result: Optional[QueryResult] = None
            for r in results:
                if r["query_id"] in merged_ids:
                    continue
                cols_lower = [c.lower() for c in (r.get("columns") or [])]
                if from_col.lower() in cols_lower and left_result is None:
                    left_result = r
                elif to_col.lower() in cols_lower and right_result is None:
                    right_result = r
                if left_result and right_result:
                    break

            if not left_result or not right_result:
                continue

            try:
                df_left  = pd.DataFrame(left_result["rows"],  columns=left_result["columns"])
                df_right = pd.DataFrame(right_result["rows"], columns=right_result["columns"])

                # Rename to_col to from_col if they differ (for merge key alignment)
                if from_col != to_col and to_col in df_right.columns:
                    df_right = df_right.rename(columns={to_col: from_col})

                merged = pd.merge(df_left, df_right, on=from_col, how="outer", suffixes=("_l", "_r"))
                merged_qr = QueryResult(
                    query_id    = f"{left_result['query_id']}+{right_result['query_id']}",
                    description = f"Federated merge: {left_result['description']} + {right_result['description']}",
                    sql         = f"-- federation merge on {from_col}",
                    columns     = list(merged.columns),
                    rows        = merged.where(pd.notnull(merged), None).values.tolist(),
                    row_count   = len(merged),
                    error       = None,
                )
                merged_results.append(merged_qr)
                merged_ids.add(left_result["query_id"])
                merged_ids.add(right_result["query_id"])
                logger.info(
                    "federation_merge: merged %s + %s on %s → %d rows",
                    left_result["query_id"], right_result["query_id"], from_col, len(merged),
                )
            except Exception as exc:
                logger.warning("federation_merge: merge failed for bridge %s → %s: %s", from_col, to_col, exc)

        # Keep unmerged results + append merged results
        final = [r for r in results if r["query_id"] not in merged_ids] + merged_results
        return final if final else results

    except Exception as exc:
        logger.warning("federation_merge: overall failure — %s", exc)
        return results


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
            q_kg_id = q.get("kg_id", "")
            if file_conn is not None:
                outcome = _exec_on_conn_with_retry(file_conn, q["sql"], config)
            else:
                outcome = _run_sql_with_retry(config, q["sql"], state=state, kg_id=q_kg_id)

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

    # Multi-KG federation merge
    if len(state.get("active_kg_ids") or []) > 1:
        bridges = state.get("kg_bridges_active") or []
        results = _federation_merge(results, bridges)

    state["query_results"] = results
    state["phase"] = "execute"
    return state
