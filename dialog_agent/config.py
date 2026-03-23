"""
Configuration for the Dialog with Data Agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class DialogConfig:
    # ── Target database (SQL execution) ───────────────────────────────────────
    db_type: str = "postgres"          # "postgres" | "oracle" | "sqlserver" | "sqlite" | "csv" | "excel"
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_schema: str = "public"
    db_user: str = ""
    db_password: str = ""
    db_connection_string: str = ""     # overrides individual fields when set
    db_extra: Dict[str, Any] = field(default_factory=dict)
    db_file_path: str = ""             # for SQLite / CSV / Excel sources

    # ── LLM settings ──────────────────────────────────────────────────────────
    # plan_llm_model: used by plan_node to generate SQL.
    #   Haiku is ~10-15× cheaper than Sonnet and fully capable of structured
    #   JSON output for SQL generation — the biggest per-question cost driver.
    plan_llm_model: str = "claude-haiku-4-5-20251001"
    # llm_model: used by synthesize_node to write the final user-facing insight.
    #   Kept at Sonnet — this is what the business user reads.
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0

    # ── Query behaviour ───────────────────────────────────────────────────────
    max_sql_queries: int = 10          # max SQL queries the planner may emit
    row_limit: int = 500               # LIMIT applied to each query
    max_insight_rows: int = 2000       # rows passed to the synthesizer LLM

    # ── User context ──────────────────────────────────────────────────────────
    analyst_role: str = ""             # e.g. "Financial Analyst" — personalises insights
