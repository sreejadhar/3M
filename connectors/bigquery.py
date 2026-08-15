"""BigQuery connector (uses google-cloud-bigquery)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class BigQueryConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._client = None

    def connect(self) -> None:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        if self._config.credentials_path:
            creds = service_account.Credentials.from_service_account_file(
                self._config.credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            self._client = bigquery.Client(
                project=self._config.project, credentials=creds
            )
        else:
            self._client = bigquery.Client(project=self._config.project)

    def close(self) -> None:
        if self._client:
            self._client.close()

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        job = self._client.query(sql)
        results = job.result()
        return [dict(row) for row in results]

    # ------------------------------------------------------------------
    def _dataset(self, schema: str) -> str:
        return schema or self._config.schema or self._config.database or ""

    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        ds = self._dataset(schema)
        project = self._config.project
        rows = self.execute(
            f"SELECT table_schema, table_name FROM `{project}.{ds}.INFORMATION_SCHEMA.TABLES` "
            f"WHERE table_type = 'BASE TABLE' ORDER BY table_name"
        )
        return [(r["table_schema"], r["table_name"]) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        ds = self._dataset(schema)
        project = self._config.project
        rows = self.execute(
            f"SELECT column_name AS name, data_type, is_nullable AS nullable, "
            f"column_default, character_maximum_length, numeric_precision, numeric_scale "
            f"FROM `{project}.{ds}.INFORMATION_SCHEMA.COLUMNS` "
            f"WHERE table_name = '{table}' ORDER BY ordinal_position"
        )
        return [dict(r) for r in rows]

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        ds = self._dataset(schema)
        project = self._config.project
        try:
            rows = self.execute(
                f"SELECT column_name FROM `{project}.{ds}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS` tc "
                f"JOIN `{project}.{ds}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE` kcu "
                f"  USING (constraint_name, table_name) "
                f"WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = '{table}' "
                f"ORDER BY kcu.ordinal_position"
            )
            return [r["column_name"] for r in rows]
        except Exception:
            return []

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        # BigQuery doesn't enforce FK constraints
        return []

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        # BigQuery has no traditional indexes
        return []

    def get_row_count(self, schema: str, table: str) -> int:
        ds = self._dataset(schema)
        project = self._config.project
        val = self.execute_scalar(
            f"SELECT row_count FROM `{project}.{ds}.__TABLES__` WHERE table_id = '{table}'"
        )
        return int(val) if val is not None else 0

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        ds = self._dataset(schema)
        project = self._config.project
        val = self.execute_scalar(
            f"SELECT size_bytes FROM `{project}.{ds}.__TABLES__` WHERE table_id = '{table}'"
        )
        return int(val) if val is not None else None

    def get_table_timestamps(self, schema: str, table: str) -> Dict[str, Optional[str]]:
        ds = self._dataset(schema)
        project = self._config.project
        rows = self.execute(
            f"SELECT TIMESTAMP_MILLIS(creation_time) AS create_time, "
            f"TIMESTAMP_MILLIS(last_modified_time) AS last_modified "
            f"FROM `{project}.{ds}.__TABLES__` WHERE table_id = '{table}'"
        )
        if rows:
            return {"create_time": str(rows[0].get("create_time")), "last_modified": str(rows[0].get("last_modified"))}
        return {"create_time": None, "last_modified": None}

    def get_partition_columns(self, schema: str, table: str) -> List[str]:
        ds = self._dataset(schema)
        project = self._config.project
        try:
            rows = self.execute(
                f"SELECT column_name FROM `{project}.{ds}.INFORMATION_SCHEMA.COLUMNS_FIELD_PATHS` "
                f"WHERE table_name = '{table}' AND data_type LIKE '%DATE%'"
                # BQ partitioning info available via DDL or table metadata API
            )
            return []  # TODO: use BQ client.get_table() for partitioning metadata
        except Exception:
            return []

    def _fqn(self, schema: str, table: str) -> str:
        ds = self._dataset(schema)
        project = self._config.project
        return f"`{project}.{ds}.{table}`"

    def _quote(self, name: str) -> str:
        return f"`{name}`"

    def _sample_clause(self, n: int) -> str:
        return f"TABLESAMPLE SYSTEM (5 PERCENT) LIMIT {n}"

    def _concat(self, cols: List[str]) -> str:
        parts = " || '|' || ".join(
            f"COALESCE(CAST({self._quote(c)} AS STRING), '__NULL__')" for c in cols
        )
        return parts

    def check_functional_dependency(self, schema, table, determinant_cols, dependent_cols, sample_size=10000):
        # BQ uses backtick quoting already handled in _fqn/_quote
        return super().check_functional_dependency(schema, table, determinant_cols, dependent_cols, sample_size)
