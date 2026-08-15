"""Snowflake connector (password or key-pair auth).

Used by the metadata-extraction pipeline.  Chat SQL execution has its own
runner in ``dialog_agent/nodes/execute_node.py``; both share ``snowflake_auth``.

Auth mode is chosen by ``snowflake_auth``: a password (e.g. from the
Register-Data-Source form) → password auth; otherwise the named connection in
``~/.snowflake/connections.toml`` + key-pair.

Identifier note: Snowflake folds unquoted identifiers to upper case, but some
tables are stored as quoted lower-case names (e.g. ``fact_gl``).
INFORMATION_SCHEMA returns the exact stored case, so discovery matches
case-insensitively while the analysis helpers quote the exact stored names.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig

logger = logging.getLogger(__name__)


class SnowflakeConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn = None
        self._cur = None

    def connect(self) -> None:
        from ..snowflake_auth import connect_snowflake
        self._conn = connect_snowflake(
            database=self._config.database or "",
            schema=self._config.schema or "",
            username=self._config.username or "",
            password=self._config.password or "",
            host=self._config.host or "",
            port=self._config.port or 0,
            extra=self._config.extra or {},
        )
        self._cur = self._conn.cursor()

    def close(self) -> None:
        try:
            if self._cur:
                self._cur.close()
        finally:
            if self._conn:
                self._conn.close()

    def execute(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        self._cur.execute(sql, params)
        if self._cur.description is None:
            return []
        cols = [d[0].lower() for d in self._cur.description]
        try:
            rows = self._cur.fetchall()
        except Exception:
            return []
        return [dict(zip(cols, r)) for r in rows]

    # ── helpers ──────────────────────────────────────────────────────────────
    def _db_prefix(self) -> str:
        db = self._config.database
        return f'"{db}".' if db else ""

    def _info(self, view: str) -> str:
        db = self._config.database
        return f'"{db}".INFORMATION_SCHEMA.{view}' if db else f"INFORMATION_SCHEMA.{view}"

    # ── schema discovery ──────────────────────────────────────────────────────
    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        if schema:
            rows = self.execute(
                f"SELECT table_schema, table_name FROM {self._info('TABLES')} "
                "WHERE UPPER(table_schema) = UPPER(%s) "
                "  AND table_type IN ('BASE TABLE', 'VIEW') "
                "ORDER BY table_name",
                (schema,),
            )
        else:
            rows = self.execute(
                f"SELECT table_schema, table_name FROM {self._info('TABLES')} "
                "WHERE table_schema <> 'INFORMATION_SCHEMA' "
                "  AND table_type IN ('BASE TABLE', 'VIEW') "
                "ORDER BY table_schema, table_name"
            )
        return [(r["table_schema"], r["table_name"]) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        return self.execute(
            "SELECT column_name AS name, data_type, is_nullable AS nullable, "
            "column_default, character_maximum_length, "
            "numeric_precision, numeric_scale "
            f"FROM {self._info('COLUMNS')} "
            "WHERE UPPER(table_schema) = UPPER(%s) AND UPPER(table_name) = UPPER(%s) "
            "ORDER BY ordinal_position",
            (schema, table),
        )

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        try:
            self._cur.execute(
                f'SHOW PRIMARY KEYS IN TABLE {self._db_prefix()}"{schema}"."{table}"'
            )
            rows = self._cur.fetchall()
            cols = [d[0].lower() for d in (self._cur.description or [])]
            recs = [dict(zip(cols, r)) for r in rows]
            recs.sort(key=lambda r: r.get("key_sequence") or 0)
            return [r["column_name"] for r in recs if r.get("column_name")]
        except Exception as exc:
            logger.debug("get_primary_keys(%s.%s) failed: %s", schema, table, exc)
            return []

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        try:
            self._cur.execute(
                f'SHOW IMPORTED KEYS IN TABLE {self._db_prefix()}"{schema}"."{table}"'
            )
            rows = self._cur.fetchall()
            cols = [d[0].lower() for d in (self._cur.description or [])]
            out: List[Dict[str, str]] = []
            for r in rows:
                rec = dict(zip(cols, r))
                out.append({
                    "column":            rec.get("fk_column_name", ""),
                    "referenced_table":  rec.get("pk_table_name", ""),
                    "referenced_column": rec.get("pk_column_name", ""),
                    "constraint_name":   rec.get("fk_name", ""),
                })
            return out
        except Exception as exc:
            logger.debug("get_foreign_keys(%s.%s) failed: %s", schema, table, exc)
            return []

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        return []

    def get_row_count(self, schema: str, table: str) -> int:
        val = self.execute_scalar(
            f"SELECT row_count FROM {self._info('TABLES')} "
            "WHERE UPPER(table_schema) = UPPER(%s) AND UPPER(table_name) = UPPER(%s)",
            (schema, table),
        )
        if val is not None:
            return int(val)
        return int(self.execute_scalar(
            f"SELECT COUNT(*) FROM {self._fqn(schema, table)}") or 0)

    def get_table_comment(self, schema: str, table: str) -> Optional[str]:
        return self.execute_scalar(
            f"SELECT comment FROM {self._info('TABLES')} "
            "WHERE UPPER(table_schema) = UPPER(%s) AND UPPER(table_name) = UPPER(%s)",
            (schema, table),
        )

    # ── dialect helpers ──────────────────────────────────────────────────────
    def _fqn(self, schema: str, table: str) -> str:
        if schema:
            return f'{self._db_prefix()}"{schema}"."{table}"'
        return f'{self._db_prefix()}"{table}"'

    def _sample_clause(self, n: int) -> str:
        return f"LIMIT {n}"
