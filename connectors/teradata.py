"""Teradata connector (uses teradatasql)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class TeradataConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn = None

    def connect(self) -> None:
        import teradatasql

        params: Dict[str, Any] = {
            "host": self._config.host,
            "user": self._config.username,
            "password": self._config.password,
            "logmech": self._config.extra.get("logmech", "TD2"),
        }
        if self._config.database:
            params["database"] = self._config.database
        params.update({k: v for k, v in self._config.extra.items() if k != "logmech"})
        self._conn = teradatasql.connect(**params)

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(sql)
        try:
            cols = [d[0].lower() for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            cur.close()

    # ------------------------------------------------------------------
    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        db = schema or self._config.database
        rows = self.execute(
            f"SELECT DatabaseName, TableName FROM DBC.TablesV "
            f"WHERE DatabaseName = '{db}' AND TableKind = 'T' "
            f"ORDER BY TableName"
        )
        return [(r["databasename"], r["tablename"]) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        db = schema or self._config.database
        rows = self.execute(
            f"SELECT ColumnName AS name, ColumnType AS data_type, "
            f"Nullable AS nullable, DefaultValue AS column_default, "
            f"ColumnLength AS character_maximum_length, "
            f"DecimalTotalDigits AS numeric_precision, DecimalFractionalDigits AS numeric_scale "
            f"FROM DBC.ColumnsV "
            f"WHERE DatabaseName = '{db}' AND TableName = '{table}' "
            f"ORDER BY ColumnId"
        )
        return [dict(r) for r in rows]

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        db = schema or self._config.database
        rows = self.execute(
            f"SELECT ColumnName FROM DBC.IndicesV "
            f"WHERE DatabaseName = '{db}' AND TableName = '{table}' "
            f"AND IndexType = 'P' ORDER BY ColumnPosition"
        )
        return [r["columnname"] for r in rows]

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        # Teradata doesn't enforce FK constraints; return empty
        return []

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        db = schema or self._config.database
        rows = self.execute(
            f"SELECT IndexName, ColumnName, UniqueFlag, IndexType "
            f"FROM DBC.IndicesV "
            f"WHERE DatabaseName = '{db}' AND TableName = '{table}' "
            f"ORDER BY IndexName, ColumnPosition"
        )
        indexes: Dict[str, Dict] = {}
        for r in rows:
            nm = r["indexname"] or r["indextype"]
            if nm not in indexes:
                indexes[nm] = {"index_name": nm, "columns": [],
                               "is_unique": r["uniqueflag"] == "Y",
                               "is_primary": r["indextype"] == "P"}
            indexes[nm]["columns"].append(r["columnname"])
        return list(indexes.values())

    def get_row_count(self, schema: str, table: str) -> int:
        db = schema or self._config.database
        est = self.execute_scalar(
            f"SELECT CAST(SUM(CurrentPerm) AS BIGINT) FROM DBC.TableSize "
            f"WHERE DatabaseName = '{db}' AND TableName = '{table}'"
        )
        # fall back to exact count
        return int(self.execute_scalar(f"SELECT COUNT(*) FROM {db}.{table}") or 0)

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        db = schema or self._config.database
        val = self.execute_scalar(
            f"SELECT CAST(SUM(CurrentPerm) AS BIGINT) FROM DBC.TableSize "
            f"WHERE DatabaseName = '{db}' AND TableName = '{table}'"
        )
        return int(val) if val else None

    def _fqn(self, schema: str, table: str) -> str:
        db = schema or self._config.database
        return f"{db}.{table}"

    def _quote(self, name: str) -> str:
        return f'"{name}"'

    def _sample_clause(self, n: int) -> str:
        return f"SAMPLE {n}"
