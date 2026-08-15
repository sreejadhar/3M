"""CSV connector — loads all *.csv files from a directory into in-memory SQLite."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class CSVConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._tables: List[str] = []

    def _dir_path(self) -> Path:
        return Path(self._config.file_path or self._config.database or ".")

    def connect(self) -> None:
        import pandas as pd

        dir_path = self._dir_path()
        if not dir_path.exists():
            raise FileNotFoundError(f"CSV directory not found: {dir_path}")

        csv_files = sorted(dir_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in: {dir_path}")

        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._tables = []

        for f in csv_files:
            table_name = f.stem  # filename without extension
            try:
                df = pd.read_csv(f)
                # Sanitise column names (spaces → underscores)
                df.columns = [c.replace(" ", "_").replace("-", "_") for c in df.columns]
                df.to_sql(table_name, self._conn, if_exists="replace", index=False)
                self._tables.append(table_name)
            except Exception:
                pass  # skip unreadable files silently

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self._tables = []

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
        return [("", t) for t in self._tables]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        rows = self.execute(f'PRAGMA table_info("{table}")')
        return [
            {
                "name":                     r["name"],
                "data_type":                r["type"] or "TEXT",
                "nullable":                 True,
                "column_default":           None,
                "character_maximum_length": None,
                "numeric_precision":        None,
                "numeric_scale":            None,
            }
            for r in rows
        ]

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        return []

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        return []

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        return []

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
