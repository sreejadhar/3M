"""Factory function: returns the correct connector for a DBConfig."""
from __future__ import annotations

from ..config import DBConfig, DBType
from .base import BaseConnector


def get_connector(config: DBConfig) -> BaseConnector:
    db_type = config.db_type

    if db_type == DBType.POSTGRES:
        from .postgres import PostgresConnector
        return PostgresConnector(config)

    if db_type == DBType.ORACLE:
        from .oracle import OracleConnector
        return OracleConnector(config)

    if db_type == DBType.TERADATA:
        from .teradata import TeradataConnector
        return TeradataConnector(config)

    if db_type == DBType.DELTA_LAKE:
        from .delta_lake import DeltaLakeConnector
        return DeltaLakeConnector(config)

    if db_type == DBType.REDSHIFT:
        from .redshift import RedshiftConnector
        return RedshiftConnector(config)

    if db_type == DBType.SQLSERVER:
        from .sqlserver import SQLServerConnector
        return SQLServerConnector(config)

    if db_type == DBType.SNOWFLAKE:
        from .snowflake import SnowflakeConnector
        return SnowflakeConnector(config)

    if db_type == DBType.BIGQUERY:
        from .bigquery import BigQueryConnector
        return BigQueryConnector(config)

    if db_type == DBType.SQLITE:
        from .sqlite import SQLiteConnector
        return SQLiteConnector(config)

    if db_type == DBType.CSV:
        from .csv_connector import CSVConnector
        return CSVConnector(config)

    if db_type == DBType.EXCEL:
        from .excel_connector import ExcelConnector
        return ExcelConnector(config)

    raise ValueError(f"Unsupported DB type: {db_type}")
