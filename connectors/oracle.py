"""Oracle connector (uses oracledb / cx_Oracle)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class OracleConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn = None

    def connect(self) -> None:
        try:
            import oracledb as cx
        except ImportError:
            import cx_Oracle as cx  # type: ignore

        if self._config.connection_string:
            self._conn = cx.connect(self._config.connection_string)
        else:
            dsn = cx.makedsn(
                self._config.host,
                self._config.port or 1521,
                service_name=self._config.database,
            )
            self._conn = cx.connect(
                user=self._config.username,
                password=self._config.password,
                dsn=dsn,
            )

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(sql, params or [])
        cols = [d[0].lower() for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        return rows

    # ------------------------------------------------------------------
    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        owner = (schema or self._config.username).upper()
        rows = self.execute(
            "SELECT owner, table_name FROM all_tables "
            "WHERE owner = :1 ORDER BY table_name",
            (owner,),
        )
        return [(r["owner"], r["table_name"]) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        owner = (schema or self._config.username).upper()
        return self.execute(
            "SELECT column_name AS name, data_type, nullable, data_default AS column_default, "
            "char_length AS character_maximum_length, data_precision AS numeric_precision, "
            "data_scale AS numeric_scale "
            "FROM all_tab_columns "
            "WHERE owner = :1 AND table_name = :2 "
            "ORDER BY column_id",
            (owner, table.upper()),
        )

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        owner = (schema or self._config.username).upper()
        rows = self.execute(
            "SELECT acc.column_name FROM all_constraints ac "
            "JOIN all_cons_columns acc ON ac.constraint_name = acc.constraint_name "
            "  AND ac.owner = acc.owner "
            "WHERE ac.constraint_type = 'P' AND ac.owner = :1 AND ac.table_name = :2 "
            "ORDER BY acc.position",
            (owner, table.upper()),
        )
        return [r["column_name"] for r in rows]

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        owner = (schema or self._config.username).upper()
        rows = self.execute(
            "SELECT acc.column_name AS column, "
            "r_ac.table_name AS referenced_table, r_acc.column_name AS referenced_column, "
            "ac.constraint_name "
            "FROM all_constraints ac "
            "JOIN all_cons_columns acc ON ac.constraint_name = acc.constraint_name AND ac.owner = acc.owner "
            "JOIN all_constraints r_ac ON ac.r_constraint_name = r_ac.constraint_name AND ac.r_owner = r_ac.owner "
            "JOIN all_cons_columns r_acc ON r_ac.constraint_name = r_acc.constraint_name AND r_ac.owner = r_acc.owner "
            "WHERE ac.constraint_type = 'R' AND ac.owner = :1 AND ac.table_name = :2",
            (owner, table.upper()),
        )
        return [dict(r) for r in rows]

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        owner = (schema or self._config.username).upper()
        rows = self.execute(
            "SELECT ai.index_name, aic.column_name, ai.uniqueness "
            "FROM all_indexes ai "
            "JOIN all_ind_columns aic ON ai.index_name = aic.index_name AND ai.owner = aic.index_owner "
            "WHERE ai.owner = :1 AND ai.table_name = :2 "
            "ORDER BY ai.index_name, aic.column_position",
            (owner, table.upper()),
        )
        indexes: Dict[str, Dict] = {}
        for r in rows:
            nm = r["index_name"]
            if nm not in indexes:
                indexes[nm] = {"index_name": nm, "columns": [], "is_unique": r["uniqueness"] == "UNIQUE", "is_primary": False}
            indexes[nm]["columns"].append(r["column_name"])
        return list(indexes.values())

    def get_row_count(self, schema: str, table: str) -> int:
        owner = (schema or self._config.username).upper()
        est = self.execute_scalar(
            "SELECT num_rows FROM all_tables WHERE owner = :1 AND table_name = :2",
            (owner, table.upper()),
        )
        if est:
            return int(est)
        return int(self.execute_scalar(f"SELECT COUNT(*) FROM {self._fqn(schema, table)}") or 0)

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        owner = (schema or self._config.username).upper()
        val = self.execute_scalar(
            "SELECT bytes FROM dba_segments WHERE owner = :1 AND segment_name = :2",
            (owner, table.upper()),
        )
        return int(val) if val else None

    def get_table_comment(self, schema: str, table: str) -> Optional[str]:
        owner = (schema or self._config.username).upper()
        return self.execute_scalar(
            "SELECT comments FROM all_tab_comments WHERE owner = :1 AND table_name = :2",
            (owner, table.upper()),
        )

    def _quote(self, name: str) -> str:
        return f'"{name.upper()}"'

    def _sample_clause(self, n: int) -> str:
        return f"SAMPLE(5) FETCH FIRST {n} ROWS ONLY"

    def _concat(self, cols: List[str]) -> str:
        parts = " || '|' || ".join(
            f"NVL(TO_CHAR({self._quote(c)}), '__NULL__')" for c in cols
        )
        return parts
