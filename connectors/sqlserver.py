"""SQL Server connector (uses pyodbc or pymssql)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class SQLServerConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn = None

    def connect(self) -> None:
        # pymssql first: pure-Python wheel, no ODBC driver installation needed.
        try:
            import pymssql  # type: ignore[import-untyped]
            if not self._config.connection_string:
                self._conn = pymssql.connect(
                    server=self._config.host,
                    port=int(self._config.port or 1433),
                    database=self._config.database,
                    user=self._config.username,
                    password=self._config.password,
                    login_timeout=30,
                )
                return
            # pymssql doesn't accept raw ODBC connection strings; fall through to pyodbc
        except ImportError:
            pass

        # pyodbc fallback: requires OS-level ODBC Driver 18 for SQL Server
        try:
            import pyodbc  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "No SQL Server driver found. "
                "Easiest fix — run:  pip install pymssql  (pure-Python, no system setup needed)\n"
                "Or for enterprise ODBC:  pip install pyodbc  "
                "(then install 'ODBC Driver 18 for SQL Server' at the OS level)"
            )

        if self._config.connection_string:
            self._conn = pyodbc.connect(self._config.connection_string, timeout=30)
        else:
            driver  = self._config.extra.get("driver", "ODBC Driver 18 for SQL Server")
            timeout = int(self._config.extra.get("login_timeout", 30))
            cs = (
                f"DRIVER={{{driver}}};"
                f"SERVER={self._config.host},{self._config.port or 1433};"
                f"DATABASE={self._config.database};"
                f"UID={self._config.username};"
                f"PWD={self._config.password};"
                "TrustServerCertificate=yes;"
                f"LoginTimeout={timeout};"
                "Encrypt=yes;"
            )
            try:
                self._conn = pyodbc.connect(cs, timeout=timeout)
            except pyodbc.Error as exc:
                state = exc.args[0] if exc.args else ""
                msg   = exc.args[1] if len(exc.args) > 1 else str(exc)
                if "HYT00" in str(state) or "timeout" in msg.lower():
                    raise ConnectionError(
                        f"Login timeout connecting to SQL Server at "
                        f"{self._config.host}:{self._config.port or 1433}. "
                        "Check that: (1) the host/IP is correct, "
                        "(2) port 1433 is open and reachable, "
                        "(3) SQL Server is running and allows remote connections."
                    ) from exc
                raise

    def close(self) -> None:
        if self._conn:
            self._conn.close()

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(sql, params or ())
        try:
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            cur.close()

    # ------------------------------------------------------------------
    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        schema = schema or "dbo"
        rows = self.execute(
            "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE' "
            "ORDER BY TABLE_NAME",
            (schema,),
        )
        return [(r["TABLE_SCHEMA"], r["TABLE_NAME"]) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        return self.execute(
            "SELECT COLUMN_NAME AS name, DATA_TYPE AS data_type, IS_NULLABLE AS nullable, "
            "COLUMN_DEFAULT AS column_default, CHARACTER_MAXIMUM_LENGTH AS character_maximum_length, "
            "NUMERIC_PRECISION AS numeric_precision, NUMERIC_SCALE AS numeric_scale "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
            "ORDER BY ORDINAL_POSITION",
            (schema or "dbo", table),
        )

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        rows = self.execute(
            "SELECT kcu.COLUMN_NAME FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc "
            "JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu "
            "  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME "
            " AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA "
            "WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY' "
            "  AND tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ? "
            "ORDER BY kcu.ORDINAL_POSITION",
            (schema or "dbo", table),
        )
        return [r["COLUMN_NAME"] for r in rows]

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        rows = self.execute(
            "SELECT kcu.COLUMN_NAME AS [column], "
            "ccu.TABLE_NAME AS referenced_table, ccu.COLUMN_NAME AS referenced_column, "
            "tc.CONSTRAINT_NAME AS constraint_name "
            "FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc "
            "JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu "
            "  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME AND tc.TABLE_SCHEMA = kcu.TABLE_SCHEMA "
            "JOIN INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc "
            "  ON tc.CONSTRAINT_NAME = rc.CONSTRAINT_NAME "
            "JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu "
            "  ON rc.UNIQUE_CONSTRAINT_NAME = ccu.CONSTRAINT_NAME "
            "WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY' "
            "  AND tc.TABLE_SCHEMA = ? AND tc.TABLE_NAME = ?",
            (schema or "dbo", table),
        )
        return [dict(r) for r in rows]

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        rows = self.execute(
            "SELECT i.name AS index_name, c.name AS column_name, "
            "i.is_unique, i.is_primary_key "
            "FROM sys.indexes i "
            "JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id "
            "JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id "
            "JOIN sys.tables t ON i.object_id = t.object_id "
            "JOIN sys.schemas s ON t.schema_id = s.schema_id "
            "WHERE s.name = ? AND t.name = ? "
            "ORDER BY i.name, ic.key_ordinal",
            (schema or "dbo", table),
        )
        indexes: Dict[str, Dict] = {}
        for r in rows:
            nm = r["index_name"]
            if nm not in indexes:
                indexes[nm] = {"index_name": nm, "columns": [], "is_unique": bool(r["is_unique"]), "is_primary": bool(r["is_primary_key"])}
            indexes[nm]["columns"].append(r["column_name"])
        return list(indexes.values())

    def get_row_count(self, schema: str, table: str) -> int:
        est = self.execute_scalar(
            "SELECT SUM(p.rows) FROM sys.tables t "
            "JOIN sys.schemas s ON t.schema_id = s.schema_id "
            "JOIN sys.partitions p ON t.object_id = p.object_id "
            "WHERE s.name = ? AND t.name = ? AND p.index_id IN (0, 1)",
            (schema or "dbo", table),
        )
        return int(est) if est else 0

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        rows = self.execute(
            f"EXEC sp_spaceused '{schema or 'dbo'}.{table}'"
        )
        if rows and "data" in rows[0]:
            raw = rows[0]["data"].strip().replace(" KB", "")
            try:
                return int(raw) * 1024
            except Exception:
                return None
        return None

    def _quote(self, name: str) -> str:
        return f"[{name}]"

    def _sample_clause(self, n: int) -> str:
        return f"TABLESAMPLE (5 PERCENT)"

    def _concat(self, cols: List[str]) -> str:
        parts = " + '|' + ".join(
            f"ISNULL(CAST({self._quote(c)} AS NVARCHAR(MAX)), '__NULL__')" for c in cols
        )
        return parts
