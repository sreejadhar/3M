"""PostgreSQL / Amazon Redshift connector (uses psycopg2)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class PostgresConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn = None
        self._cur = None

    def connect(self) -> None:
        import psycopg2
        import psycopg2.extras

        if self._config.connection_string:
            self._conn = psycopg2.connect(self._config.connection_string)
        else:
            self._conn = psycopg2.connect(
                host=self._config.host,
                port=self._config.port or 5432,
                dbname=self._config.database,
                user=self._config.username,
                password=self._config.password,
                **self._config.extra,
            )
        self._conn.autocommit = True
        self._cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def close(self) -> None:
        if self._cur:
            self._cur.close()
        if self._conn:
            self._conn.close()

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        self._cur.execute(sql, params)
        try:
            return [dict(r) for r in self._cur.fetchall()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        schema = schema or "public"
        rows = self.execute(
            "SELECT table_schema, table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
            "ORDER BY table_name",
            (schema,),
        )
        return [(r["table_schema"], r["table_name"]) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        return self.execute(
            "SELECT column_name AS name, data_type, is_nullable AS nullable, "
            "column_default, character_maximum_length, numeric_precision, numeric_scale "
            "FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s "
            "ORDER BY ordinal_position",
            (schema or "public", table),
        )

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        rows = self.execute(
            "SELECT kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema = kcu.table_schema "
            "WHERE tc.constraint_type = 'PRIMARY KEY' "
            "  AND tc.table_schema = %s AND tc.table_name = %s "
            "ORDER BY kcu.ordinal_position",
            (schema or "public", table),
        )
        return [r["column_name"] for r in rows]

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        rows = self.execute(
            "SELECT kcu.column_name AS column, "
            "ccu.table_name AS referenced_table, ccu.column_name AS referenced_column, "
            "tc.constraint_name "
            "FROM information_schema.table_constraints AS tc "
            "JOIN information_schema.key_column_usage AS kcu "
            "  ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage AS ccu "
            "  ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema "
            "WHERE tc.constraint_type = 'FOREIGN KEY' "
            "  AND tc.table_schema = %s AND tc.table_name = %s",
            (schema or "public", table),
        )
        return [dict(r) for r in rows]

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        rows = self.execute(
            "SELECT i.relname AS index_name, "
            "array_agg(a.attname ORDER BY x.n) AS columns, "
            "ix.indisunique AS is_unique, ix.indisprimary AS is_primary "
            "FROM pg_class t "
            "JOIN pg_index ix ON t.oid = ix.indrelid "
            "JOIN pg_class i ON i.oid = ix.indexrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS x(attnum, n) ON TRUE "
            "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = x.attnum "
            "WHERE n.nspname = %s AND t.relname = %s "
            "GROUP BY i.relname, ix.indisunique, ix.indisprimary",
            (schema or "public", table),
        )
        return [dict(r) for r in rows]

    def get_row_count(self, schema: str, table: str) -> int:
        # Use pg_class for fast estimate, fall back to COUNT(*)
        fqn = f"'{schema or 'public'}.{table}'"
        est = self.execute_scalar(
            f"SELECT reltuples::BIGINT FROM pg_class c "
            f"JOIN pg_namespace n ON n.oid = c.relnamespace "
            f"WHERE n.nspname = '{schema or 'public'}' AND c.relname = '{table}'"
        )
        if est and est > 0:
            return int(est)
        return int(self.execute_scalar(f"SELECT COUNT(*) FROM {self._fqn(schema, table)}") or 0)

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        val = self.execute_scalar(
            f"SELECT pg_total_relation_size('{schema or 'public'}.{table}')"
        )
        return int(val) if val else None

    def get_table_comment(self, schema: str, table: str) -> Optional[str]:
        val = self.execute_scalar(
            "SELECT obj_description(c.oid) FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            f"WHERE n.nspname = '{schema or 'public'}' AND c.relname = '{table}'"
        )
        return val

    def _sample_clause(self, n: int) -> str:
        return f"TABLESAMPLE SYSTEM(5) LIMIT {n}"
