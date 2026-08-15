"""
Abstract base class that every DB connector must implement.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple


class BaseConnector(abc.ABC):
    """
    Thin abstraction layer that lets the agent work identically
    against all supported databases.  Every method executes a
    *single* SQL/API call and returns plain Python structures.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Open / initialise the connection."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the connection / session."""

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Core query primitive
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def execute(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Run *sql* and return a list of dicts (column_name -> value).
        Never raise on empty result – return [].
        """

    def execute_scalar(self, sql: str, params: Optional[tuple] = None) -> Any:
        rows = self.execute(sql, params)
        if not rows:
            return None
        return next(iter(rows[0].values()))

    # ------------------------------------------------------------------
    # Schema discovery
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        """
        Return [(schema_name, table_name), ...] for every user table
        in *schema*.  Pass schema=None to use the default.
        """

    @abc.abstractmethod
    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """
        Return a list of dicts, one per column:
            {name, data_type, nullable, column_default, character_maximum_length, ...}
        """

    @abc.abstractmethod
    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        """Return list of PK column names."""

    @abc.abstractmethod
    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        """
        Return [
            {column, referenced_table, referenced_column, constraint_name}, ...
        ]
        """

    @abc.abstractmethod
    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        """
        Return [
            {index_name, columns: [str], is_unique, is_primary}, ...
        ]
        """

    # ------------------------------------------------------------------
    # Table-level metadata
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_row_count(self, schema: str, table: str) -> int:
        """Exact or estimated row count."""

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        """Override in connectors that expose table size."""
        return None

    def get_table_comment(self, schema: str, table: str) -> Optional[str]:
        return None

    def get_table_timestamps(self, schema: str, table: str) -> Dict[str, Optional[str]]:
        """Return {'create_time': ..., 'last_modified': ...}."""
        return {"create_time": None, "last_modified": None}

    def get_partition_columns(self, schema: str, table: str) -> List[str]:
        return []

    # ------------------------------------------------------------------
    # Column-level statistics
    # ------------------------------------------------------------------

    def get_column_stats(self, schema: str, table: str, column: str,
                         sample_size: int = 10_000) -> Dict[str, Any]:
        """
        Return statistics for a single column.  The default implementation
        issues generic SQL; connectors can override for DB-specific queries.
        """
        fqn = self._fqn(schema, table)
        sample_clause = self._sample_clause(sample_size)

        # null count + unique count
        row = self.execute(
            f"SELECT COUNT(*) AS total, COUNT({self._quote(column)}) AS non_null, "
            f"COUNT(DISTINCT {self._quote(column)}) AS uniq "
            f"FROM {fqn} {sample_clause}"
        )
        total = row[0]["total"] if row else 0
        non_null = row[0]["non_null"] if row else 0
        uniq = row[0]["uniq"] if row else 0

        stats: Dict[str, Any] = {
            "null_count": total - non_null,
            "unique_count": uniq,
            "row_count": total,
        }

        # numeric stats (best-effort)
        try:
            num_row = self.execute(
                f"SELECT MIN({self._quote(column)}) AS mn, "
                f"MAX({self._quote(column)}) AS mx, "
                f"AVG(CAST({self._quote(column)} AS FLOAT)) AS avg_val, "
                f"STDDEV(CAST({self._quote(column)} AS FLOAT)) AS std_val "
                f"FROM {fqn} {sample_clause}"
            )
            if num_row:
                stats["min_value"] = num_row[0].get("mn")
                stats["max_value"] = num_row[0].get("mx")
                stats["avg_value"] = num_row[0].get("avg_val")
                stats["stddev_value"] = num_row[0].get("std_val")
        except Exception:
            pass

        # top 10 frequent values. sample_clause (TABLESAMPLE/LIMIT/...) is a
        # row-sampling clause that must sit directly after FROM <table>, not
        # before a GROUP BY on the same query — every dialect's aggregate
        # clauses (GROUP BY/ORDER BY/LIMIT) can only follow it once there is
        # no further FROM-clause syntax in between, so the sampled rows are
        # pulled into a subquery first and the aggregation runs on top of
        # that, keeping every dialect's sample_clause syntactically valid.
        try:
            top = self.execute(
                f"SELECT val, COUNT(*) AS cnt FROM ("
                f"SELECT {self._quote(column)} AS val FROM {fqn} {sample_clause}"
                f") sampled_rows "
                f"GROUP BY val "
                f"ORDER BY cnt DESC "
                f"LIMIT 10"
            )
            stats["top_values"] = [r["val"] for r in top]
        except Exception:
            pass

        return stats

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def check_functional_dependency(
        self, schema: str, table: str,
        determinant_cols: List[str], dependent_cols: List[str],
        sample_size: int = 10_000,
    ) -> Tuple[float, int]:
        """
        Returns (confidence, num_violations).
        confidence = 1.0 means perfect FD.

        Uses: for each distinct value of determinant, count distinct values of dependent.
        If max(count) == 1 → perfect FD.

        Sampling: rows are filtered to exclude NULLs in determinant columns, then
        limited to sample_size.  The GROUP BY then runs over the sampled rows.
        For tables larger than sample_size this is an approximation — increase
        sample_size in AgentConfig for higher accuracy at the cost of more CPU.
        """
        fqn = self._fqn(schema, table)
        det_cols = ", ".join(self._quote(c) for c in determinant_cols)
        dep_cols = ", ".join(self._quote(c) for c in dependent_cols)

        # Build NOT NULL filter for all determinant columns to avoid grouping NULLs
        null_filters = " AND ".join(
            f"{self._quote(c)} IS NOT NULL" for c in determinant_cols
        )
        where_clause = f"WHERE {null_filters}" if null_filters else ""

        sql = (
            f"SELECT MAX(dep_cnt) AS max_dep, "
            f"SUM(CASE WHEN dep_cnt > 1 THEN 1 ELSE 0 END) AS violations, "
            f"COUNT(*) AS total_groups "
            f"FROM ("
            f"  SELECT {det_cols}, COUNT(DISTINCT {dep_cols}) AS dep_cnt "
            f"  FROM ("
            f"    SELECT {det_cols}, {dep_cols} FROM {fqn} {where_clause} LIMIT {sample_size}"
            f"  ) sampled "
            f"  GROUP BY {det_cols}"
            f") sub"
        )
        try:
            row = self.execute(sql)
            if not row or row[0]["total_groups"] is None:
                return 0.0, 0
            violations = row[0]["violations"] or 0
            total = row[0]["total_groups"] or 1
            confidence = 1.0 - (violations / total)
            return confidence, int(violations)
        except Exception:
            return 0.0, 0

    def check_inclusion_dependency(
        self, schema: str,
        left_table: str, left_cols: List[str],
        right_table: str, right_cols: List[str],
        sample_size: int = 10_000,
    ) -> float:
        """
        Returns the fraction of left_table[left_cols] distinct values found in
        right_table[right_cols].  1.0 = full IND.

        Sampling strategy:
        - Left (referencing) side: DISTINCT values extracted first, then limited to
          sample_size.  This ensures the sample represents the actual value distribution
          rather than just the first N rows of the table (which could miss values that
          only appear later in the file).
        - Right (referenced) side: full distinct scan — no sampling.  We need ALL
          reference values to correctly measure how many left-side values are covered.
        """
        left_fqn  = self._fqn(schema, left_table)
        right_fqn = self._fqn(schema, right_table)

        if len(left_cols) == 1:
            lc = self._quote(left_cols[0])
            rc = self._quote(right_cols[0])
            # Get all distinct left values first, then cap at sample_size distinct values
            sql = (
                f"SELECT "
                f"COUNT(*) AS total, "
                f"SUM(CASE WHEN r.{rc} IS NOT NULL THEN 1 ELSE 0 END) AS matched "
                f"FROM ("
                f"  SELECT DISTINCT {lc} FROM {left_fqn} WHERE {lc} IS NOT NULL"
                f"  LIMIT {sample_size}"
                f") l "
                f"LEFT JOIN (SELECT DISTINCT {rc} FROM {right_fqn} WHERE {rc} IS NOT NULL) r "
                f"ON l.{lc} = r.{rc}"
            )
        else:
            # composite — build CONCAT or ROW compare
            l_concat = self._concat(left_cols)
            r_concat = self._concat(right_cols)
            sql = (
                f"SELECT COUNT(*) AS total, "
                f"SUM(CASE WHEN r.rk IS NOT NULL THEN 1 ELSE 0 END) AS matched "
                f"FROM ("
                f"  SELECT DISTINCT {l_concat} AS lk FROM {left_fqn} LIMIT {sample_size}"
                f") l "
                f"LEFT JOIN (SELECT DISTINCT {r_concat} AS rk FROM {right_fqn}) r "
                f"ON l.lk = r.rk"
            )
        try:
            row = self.execute(sql)
            if not row or not row[0]["total"]:
                return 0.0
            return float(row[0]["matched"]) / float(row[0]["total"])
        except Exception:
            return 0.0

    def get_join_cardinality(
        self, schema: str,
        left_table: str, right_table: str,
        join_columns: List[str],
        sample_size: int = 10_000,
    ) -> Dict[str, Any]:
        """Determine 1:1 / 1:N / M:N between two tables using sampled data."""
        left_fqn      = self._fqn(schema, left_table)
        right_fqn     = self._fqn(schema, right_table)
        jc            = ", ".join(self._quote(c) for c in join_columns)
        sample_clause = self._sample_clause(sample_size)

        left_uniq   = self.execute_scalar(
            f"SELECT COUNT(DISTINCT {jc}) FROM {left_fqn} {sample_clause}")
        right_uniq  = self.execute_scalar(
            f"SELECT COUNT(DISTINCT {jc}) FROM {right_fqn} {sample_clause}")
        left_total  = self.execute_scalar(
            f"SELECT COUNT(*) FROM {left_fqn} {sample_clause}")
        right_total = self.execute_scalar(
            f"SELECT COUNT(*) FROM {right_fqn} {sample_clause}")

        left_unique  = left_uniq  == left_total
        right_unique = right_uniq == right_total

        if left_unique and right_unique:
            rel = "1:1"
        elif left_unique and not right_unique:
            rel = "1:N"
        elif not left_unique and right_unique:
            rel = "N:1"
        else:
            rel = "M:N"

        return {
            "relationship_type": rel,
            "left_unique": int(left_uniq or 0),
            "right_unique": int(right_uniq or 0),
        }

    # ------------------------------------------------------------------
    # Dialect helpers (override in subclasses as needed)
    # ------------------------------------------------------------------

    def _fqn(self, schema: str, table: str) -> str:
        if schema:
            return f"{self._quote(schema)}.{self._quote(table)}"
        return self._quote(table)

    def _quote(self, name: str) -> str:
        return f'"{name}"'

    def _sample_clause(self, n: int) -> str:
        """TABLESAMPLE / LIMIT clause for row sampling."""
        return f"LIMIT {n}"

    def _concat(self, cols: List[str]) -> str:
        parts = " || '|' || ".join(
            f"COALESCE(CAST({self._quote(c)} AS VARCHAR), '__NULL__')" for c in cols
        )
        return parts
