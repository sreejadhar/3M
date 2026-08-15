"""SQLite connector — uses Python's built-in sqlite3 (no extra dependencies)."""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class SQLiteConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None

    def _db_path(self) -> str:
        return self._config.file_path or self._config.database or ":memory:"

    def connect(self) -> None:
        path = self._db_path()
        if path != ":memory:" and not os.path.exists(path):
            raise FileNotFoundError(f"SQLite database file not found: {path}")
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(sql, params or ())
        try:
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []
        finally:
            cur.close()

    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        rows = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [("main", r["name"]) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        rows = self.execute(f'PRAGMA table_info("{table}")')
        return [
            {
                "name":                     r["name"],
                "data_type":                r["type"] or "TEXT",
                "nullable":                 not r["notnull"],
                "column_default":           r["dflt_value"],
                "character_maximum_length": None,
                "numeric_precision":        None,
                "numeric_scale":            None,
            }
            for r in rows
        ]

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        rows = self.execute(f'PRAGMA table_info("{table}")')
        return [r["name"] for r in rows if r["pk"]]

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        rows = self.execute(f'PRAGMA foreign_key_list("{table}")')
        return [
            {
                "column":            r["from"],
                "referenced_table":  r["table"],
                "referenced_column": r["to"],
                "constraint_name":   f"fk_{table}_{r['from']}",
            }
            for r in rows
        ]

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        idx_list = self.execute(f'PRAGMA index_list("{table}")')
        result = []
        for idx in idx_list:
            idx_info = self.execute(f'PRAGMA index_info("{idx["name"]}")')
            result.append({
                "index_name": idx["name"],
                "columns":    [i["name"] for i in idx_info],
                "is_unique":  bool(idx["unique"]),
                "is_primary": idx["origin"] == "pk",
            })
        return result

    def get_row_count(self, schema: str, table: str) -> int:
        n = self.execute_scalar(f'SELECT COUNT(*) FROM "{table}"')
        return int(n) if n else 0

    # ── dialect helpers ────────────────────────────────────────────────────────
    def _quote(self, name: str) -> str:
        return f'"{name}"'

    def _sample_clause(self, n: int) -> str:
        return f"LIMIT {n}"

    def _concat(self, cols: List[str]) -> str:
        return " || '|' || ".join(
            f"COALESCE(CAST(\"{c}\" AS TEXT), '__NULL__')" for c in cols
        )
