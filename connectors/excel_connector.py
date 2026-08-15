"""Excel connector — loads each sheet of an .xlsx/.xls file into in-memory SQLite."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig

_VALID_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}


def _safe_name(name: str) -> str:
    """Convert a sheet name to a valid SQL table name."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "sheet_" + name
    return name or "sheet"


def _detect_header_row(xl, sheet: str, scan_rows: int = 10) -> int:
    """
    Find the first row that looks like a real header.

    Many Excel files have a title/subtitle in rows 0-1 before the actual
    column names.  pandas' default header=0 then names all but the first
    column 'Unnamed: N'.

    Strategy: read the first `scan_rows` rows without a header, count how
    many non-null, non-empty cells each row has, and return the index of the
    first row whose count equals the maximum across all scanned rows.
    That row is almost always the real header.
    """
    import pandas as pd
    preview = xl.parse(sheet, header=None, nrows=scan_rows)
    if preview.empty:
        return 0

    def _nonempty(row):
        return sum(1 for v in row if str(v).strip() not in ("", "nan", "None"))

    counts = [_nonempty(preview.iloc[i]) for i in range(len(preview))]
    max_count = max(counts) if counts else 0
    if max_count == 0:
        return 0

    # Return the FIRST row that reaches the maximum density
    for idx, cnt in enumerate(counts):
        if cnt == max_count:
            return idx
    return 0


class ExcelConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._sheets: List[str] = []

    def _file_path(self) -> Path:
        return Path(self._config.file_path or self._config.database or "")

    def connect(self) -> None:
        import pandas as pd

        path = self._file_path()
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found: {path}")
        if path.suffix.lower() not in _VALID_EXTENSIONS:
            raise ValueError(
                f"Not a supported Excel file (got '{path.suffix}'). "
                f"Expected one of: {', '.join(sorted(_VALID_EXTENSIONS))}"
            )

        xl = pd.ExcelFile(path)
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._sheets = []

        import logging as _logging
        _log = _logging.getLogger(__name__)

        used_sheets: dict = {}
        sheet_errors: list = []
        for sheet in xl.sheet_names:
            base = _safe_name(sheet)
            # Deduplicate sheet→table names
            if base in used_sheets:
                used_sheets[base] += 1
                safe = f"{base}_{used_sheets[base]}"
            else:
                used_sheets[base] = 1
                safe = base
            try:
                header_row = _detect_header_row(xl, sheet)
                if header_row != 0:
                    _log.info(
                        "ExcelConnector: sheet %r — header detected at row %d (skipping title rows)",
                        sheet, header_row,
                    )
                df = xl.parse(sheet, header=header_row)

                # Deduplicate column names after sanitisation
                # (two columns that map to the same safe name crash df.to_sql)
                used_cols: dict = {}
                new_cols = []
                for c in df.columns:
                    sc = _safe_name(str(c))
                    if sc in used_cols:
                        used_cols[sc] += 1
                        sc = f"{sc}_{used_cols[sc]}"
                    else:
                        used_cols[sc] = 1
                    new_cols.append(sc)
                df.columns = new_cols

                df.to_sql(safe, self._conn, if_exists="replace", index=False)
                self._sheets.append(safe)
                _log.info("ExcelConnector: loaded sheet %r → table %r (%d rows, %d cols)",
                          sheet, safe, len(df), len(df.columns))
            except Exception as exc:
                _log.warning(
                    "ExcelConnector: skipping sheet %r (→ %r) — %s: %s",
                    sheet, safe, type(exc).__name__, exc
                )
                sheet_errors.append(f"Sheet '{sheet}': {type(exc).__name__}: {exc}")

        if not self._sheets:
            detail = "; ".join(sheet_errors) if sheet_errors else "no sheets found"
            raise RuntimeError(
                f"Excel file '{path.name}' could not be loaded — {detail}"
            )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        self._sheets = []

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
        return [("", s) for s in self._sheets]

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
