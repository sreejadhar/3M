"""
Delta Lake / Databricks connector.
Uses PySpark or the Databricks SQL connector depending on config.
- If spark_master is set: uses local PySpark with Delta table support.
- If host is set: uses databricks-sql-connector (REST/HTTP).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseConnector
from config import DBConfig


class DeltaLakeConnector(BaseConnector):
    def __init__(self, config: DBConfig):
        self._config = config
        self._conn = None        # databricks-sql cursor path
        self._spark = None       # PySpark path

    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self._config.host:
            self._connect_databricks_sql()
        else:
            self._connect_pyspark()

    def _connect_databricks_sql(self) -> None:
        from databricks import sql

        self._conn = sql.connect(
            server_hostname=self._config.host,
            http_path=self._config.extra.get("http_path", ""),
            access_token=self._config.password or self._config.extra.get("token"),
        )

    def _connect_pyspark(self) -> None:
        from pyspark.sql import SparkSession

        master = self._config.spark_master or "local[*]"
        builder = (
            SparkSession.builder.master(master)
            .appName("MetadataAgent")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        catalog = self._config.catalog
        if catalog:
            builder = builder.config("spark.sql.defaultCatalog", catalog)
        self._spark = builder.getOrCreate()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
        if self._spark:
            self._spark.stop()

    def execute(self, sql: str, params=None) -> List[Dict[str, Any]]:
        if self._spark:
            df = self._spark.sql(sql)
            return [row.asDict() for row in df.collect()]
        else:
            cur = self._conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            cur.close()
            return rows

    # ------------------------------------------------------------------
    def _db(self, schema: str) -> str:
        return schema or self._config.database or self._config.schema or "default"

    def list_tables(self, schema: str) -> List[Tuple[str, str]]:
        db = self._db(schema)
        rows = self.execute(f"SHOW TABLES IN {db}")
        return [(r.get("database", db), r.get("tableName", r.get("table_name", ""))) for r in rows]

    def get_columns(self, schema: str, table: str) -> List[Dict[str, Any]]:
        db = self._db(schema)
        rows = self.execute(f"DESCRIBE {db}.{table}")
        cols = []
        for r in rows:
            col_name = r.get("col_name", r.get("column_name", ""))
            if not col_name or col_name.startswith("#"):
                continue
            cols.append({
                "name": col_name,
                "data_type": r.get("data_type", r.get("col_type", "")),
                "nullable": "YES",
                "column_default": r.get("comment", None),
                "character_maximum_length": None,
                "numeric_precision": None,
                "numeric_scale": None,
            })
        return cols

    def get_primary_keys(self, schema: str, table: str) -> List[str]:
        # Delta Lake / Databricks supports PKs in Unity Catalog
        db = self._db(schema)
        try:
            rows = self.execute(
                f"SELECT column_name FROM information_schema.key_column_usage kcu "
                f"JOIN information_schema.table_constraints tc USING (constraint_name) "
                f"WHERE tc.constraint_type = 'PRIMARY KEY' "
                f"AND tc.table_schema = '{db}' AND tc.table_name = '{table}' "
                f"ORDER BY ordinal_position"
            )
            return [r["column_name"] for r in rows]
        except Exception:
            return []

    def get_foreign_keys(self, schema: str, table: str) -> List[Dict[str, str]]:
        return []

    def get_indexes(self, schema: str, table: str) -> List[Dict[str, Any]]:
        return []

    def get_row_count(self, schema: str, table: str) -> int:
        db = self._db(schema)
        # Try Delta table stats first
        try:
            rows = self.execute(f"DESCRIBE DETAIL {db}.{table}")
            if rows and rows[0].get("numFiles") is not None:
                # numRows available in Databricks >= 10.4
                nr = rows[0].get("numRows")
                if nr is not None:
                    return int(nr)
        except Exception:
            pass
        return int(self.execute_scalar(f"SELECT COUNT(*) FROM {db}.{table}") or 0)

    def get_table_size_bytes(self, schema: str, table: str) -> Optional[int]:
        db = self._db(schema)
        try:
            rows = self.execute(f"DESCRIBE DETAIL {db}.{table}")
            if rows:
                return rows[0].get("sizeInBytes")
        except Exception:
            pass
        return None

    def get_partition_columns(self, schema: str, table: str) -> List[str]:
        db = self._db(schema)
        try:
            rows = self.execute(f"DESCRIBE DETAIL {db}.{table}")
            if rows and rows[0].get("partitionColumns"):
                return list(rows[0]["partitionColumns"])
        except Exception:
            pass
        return []

    def get_table_timestamps(self, schema: str, table: str) -> Dict[str, Optional[str]]:
        db = self._db(schema)
        try:
            rows = self.execute(f"DESCRIBE DETAIL {db}.{table}")
            if rows:
                return {
                    "create_time": str(rows[0].get("createdAt")),
                    "last_modified": str(rows[0].get("lastModified")),
                }
        except Exception:
            pass
        return {"create_time": None, "last_modified": None}

    def _fqn(self, schema: str, table: str) -> str:
        db = self._db(schema)
        return f"{db}.{table}"

    def _quote(self, name: str) -> str:
        return f"`{name}`"

    def _sample_clause(self, n: int) -> str:
        return f"LIMIT {n}"
