"""
Runs an EvalDataset's questions against a built KG via the existing dialog_api.py
/query + /jobs/{id} polling pattern (same mechanics as run_lifesciences_doc_questions.py),
pointed at the trial's kg_id/nodes/edges instead of a fixed source.

Known limitation: dialog_api.py's QueryRequest has no field to override
graphrag_top_k per-request (checked dialog_api.py:127) — that gene is tracked in
the genome for when such an override is added, but has no effect on eval results
through this HTTP path today.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from kg_optimizer.build_runner import KGBuildResult
from kg_optimizer.config import OptimizerConfig
from kg_optimizer.datasets import EvalDataset, load_connection_from_kg_store

logger = logging.getLogger(__name__)


@dataclass
class QuestionRunResult:
    question: str
    answer: Optional[str] = None
    sql_count: int = 0
    error: Optional[str] = None
    wall_time_s: float = 0.0
    gold_answer: Optional[str] = None


@dataclass
class DatasetRunResult:
    question_results: List[QuestionRunResult] = field(default_factory=list)
    total_time_s: float = 0.0


def _resolve_connection(dataset: EvalDataset) -> Dict[str, Any]:
    if dataset.connection is not None:
        return dataset.connection
    if dataset.kg_store_source_id:
        loaded = load_connection_from_kg_store(dataset.kg_store_source_id, dataset.kg_store_path)
        cfg = loaded["cfg"]
        return {
            "db_type": "snowflake",
            "db_host": cfg.get("host", ""),
            "db_port": cfg.get("port", 0),
            "db_name": cfg.get("database", ""),
            "db_schema": cfg.get("schema_") or cfg.get("schema", ""),
            "db_user": cfg.get("username", ""),
            "db_password": cfg.get("password", ""),
            "db_extra": cfg.get("extra", {}),
        }
    raise ValueError("EvalDataset has neither `connection` nor `kg_store_source_id` set")


def _submit(question: str, kg_build: KGBuildResult, connection: Dict[str, Any],
           source_id: str, cfg: OptimizerConfig) -> str:
    payload = {
        "natural_query": question,
        "kg_nodes": kg_build.nodes,
        "kg_edges": kg_build.edges,
        "source_id": source_id,
        "skip_cache": True,
        "row_limit": 500,
        **connection,
    }
    r = requests.post(f"{cfg.kg_query_api_base}/query", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["job_id"]


def _poll(job_id: str, cfg: OptimizerConfig) -> Dict[str, Any]:
    start = time.time()
    while time.time() - start < cfg.query_timeout_s:
        r = requests.get(f"{cfg.kg_query_api_base}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        status_data = r.json()
        if status_data["status"] == "done":
            results = requests.get(f"{cfg.kg_query_api_base}/jobs/{job_id}/results", timeout=30).json()
            return {
                "status": "done",
                "query_count": status_data.get("query_count", 0),
                "insights": results.get("insights"),
            }
        if status_data["status"] == "error":
            return {"status": "error", "error": status_data.get("error", "unknown error")}
        time.sleep(2)
    return {"status": "timeout"}


def run_dataset(dataset: EvalDataset, kg_build: KGBuildResult, cfg: OptimizerConfig) -> DatasetRunResult:
    connection = _resolve_connection(dataset)
    t0 = time.time()
    results: List[QuestionRunResult] = []
    for q in dataset.questions:
        q_t0 = time.time()
        try:
            job_id = _submit(q.question, kg_build, connection, dataset.source_id, cfg)
            data = _poll(job_id, cfg)
            if data["status"] == "done":
                results.append(QuestionRunResult(
                    question=q.question, answer=data.get("insights"),
                    sql_count=data.get("query_count", 0),
                    wall_time_s=time.time() - q_t0, gold_answer=q.gold_answer,
                ))
            else:
                results.append(QuestionRunResult(
                    question=q.question, error=data.get("error", data["status"]),
                    wall_time_s=time.time() - q_t0, gold_answer=q.gold_answer,
                ))
        except Exception as exc:
            logger.warning("Question failed: %r — %s", q.question[:60], exc)
            results.append(QuestionRunResult(
                question=q.question, error=str(exc),
                wall_time_s=time.time() - q_t0, gold_answer=q.gold_answer,
            ))
    return DatasetRunResult(question_results=results, total_time_s=time.time() - t0)
