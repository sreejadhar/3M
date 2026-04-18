"""Amazon Redshift connector – inherits Postgres with Redshift-specific overrides."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .postgres import PostgresConnector
from ..config import DBConfig


class RedshiftConnector(PostgresConnector):
    """
    Redshift uses the PostgreSQL wire protocol so most logic is inherited.
    We override size/stats queries to use Redshift-specific system tables.
    """

    def connect(self) -> None:
        # Try redshift_connector first, fall back to psycopg2
        try:
            import redshift_connector
            if self._config.connection_string:
                raise ValueError("Use host/port/db for Redshift native connector")
            self._conn = redshift_connector.connect(
                host=self._config.host,
                port=self._config.port or 5439,
                database=self._config.database,
                user=self._config.username,
                password=self._config.password,
                **self._config.extra,
            )
            self._conn.autocommit = True
            self._cur = self._conn.cursor()
            # Monkey-patch fetchall to return dicts
            _orig = self._cur.fetchall

            def _dict_fetchall():
                cols = [d[0] for d in self._cur.description]
                return [dict(zip(cols, r)) for r in _orig()]

            self._cur.fetchall = _dict_fetchall  # type: ignore
        except ImportError:
            super().connect()

    def get_row_count(self, schema: str, table: str) -> int:
        schema = schema or "public"
        val = self.execute_scalar(
            "SELECT tbl_rows FROM svv_table_info "
            f"WHERE schema = '{schema}' AND \"table\" = '{table}'"
        )
        if val:
            return int(val)
        return int(self.execute_scalar(f"SELECT COUNT(*) FROM {self._fqn(schema, table)}") or 0)

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        schema = schema or "public"
        val = self.execute_scalar(
            "SELECT size * 1024 * 1024 FROM svv_table_info "
            f"WHERE schema = '{schema}' AND \"table\" = '{table}'"
        )
        return int(val) if val else None

    def get_partition_columns(self, schema: str, table: str) -> List[str]:
        schema = schema or "public"
        rows = self.execute(
            "SELECT \"column\" FROM svv_table_info "
            f"WHERE schema = '{schema}' AND \"table\" = '{table}'"
        )
        # Redshift doesn't expose sort/dist keys the same way; return dist key
        dist = self.execute(
            "SELECT column_name FROM svv_columns "
            "WHERE table_schema = %s AND table_name = %s "
            "AND distkey = 'true'",
            (schema, table),
        )
        return [r["column_name"] for r in dist]

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        # Redshift doesn't support array_agg(ORDER BY) or LATERAL/WITH ORDINALITY.
        # Use pg_index + pg_attribute with a subquery join instead.
        schema = schema or "public"
        rows = self.execute(
            "SELECT i.relname AS index_name, "
            "  a.attname AS column_name, "
            "  ix.indisunique AS is_unique, "
            "  ix.indisprimary AS is_primary "
            "FROM pg_class t "
            "JOIN pg_index ix ON t.oid = ix.indrelid "
            "JOIN pg_class i  ON i.oid  = ix.indexrelid "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "JOIN pg_attribute a ON a.attrelid = t.oid "
            "  AND a.attnum = ANY(ix.indkey) "
            "WHERE n.nspname = %s AND t.relname = %s "
            "ORDER BY i.relname, a.attname",
            (schema, table),
        )
        # Collapse multiple rows per index into one record
        seen: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            name = r["index_name"]
            if name not in seen:
                seen[name] = {
                    "index_name": name,
                    "columns":    [r["column_name"]],
                    "is_unique":  r["is_unique"],
                    "is_primary": r["is_primary"],
                }
            else:
                seen[name]["columns"].append(r["column_name"])
        return list(seen.values())

    def _sample_clause(self, n: int) -> str:
        # Redshift supports LIMIT but not TABLESAMPLE
        return f"LIMIT {n}"
