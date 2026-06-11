"""
Configuration dataclasses for the metadata extraction agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class DBType(str, Enum):
    POSTGRES   = "postgres"
    ORACLE     = "oracle"
    TERADATA   = "teradata"
    DELTA_LAKE = "delta_lake"
    REDSHIFT   = "redshift"
    SQLSERVER  = "sqlserver"
    SNOWFLAKE  = "snowflake"
    BIGQUERY   = "bigquery"
    SQLITE     = "sqlite"
    CSV        = "csv"
    EXCEL      = "excel"


@dataclass
class DBConfig:
    db_type: DBType
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    schema: Optional[str] = None          # target schema / dataset
    username: Optional[str] = None
    password: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    # BigQuery / Delta Lake specifics
    project: Optional[str] = None         # GCP project
    credentials_path: Optional[str] = None
    catalog: Optional[str] = None         # Spark catalog (Delta)
    spark_master: Optional[str] = None    # e.g. "local[*]"
    # Connection string override (optional – takes priority)
    connection_string: Optional[str] = None
    # File-based sources (SQLite, CSV directory, Excel)
    file_path: Optional[str] = None


@dataclass
class AgentConfig:
    db_config: DBConfig
    target_tables: Optional[List[str]] = None   # None = all tables
    sample_size: int = 10_000               # rows sampled for FD/ID analysis
    fd_threshold: float = 1.0               # 1.0 = exact FDs only; <1.0 = approximate
    id_threshold: float = 0.95              # fraction of values that must match for IND
    max_fd_column_pairs: int = 200          # cap combinatorial explosion
    max_id_column_pairs: int = 500
    # Used only by agent.ask() for Q&A on the finished report — not by any
    # extraction pipeline node (all extraction is rule-based, no LLM calls).
    # Haiku is fully capable of factual Q&A on structured JSON.
    llm_model: str = "claude-haiku-4-5"
    llm_temperature: float = 0.0
    output_path: Optional[str] = None       # where to write JSON report
    # Optional callback(step, status, message, detail) for fine-grained progress
    progress_callback: Optional[Callable[[str, str, str, str], None]] = None
