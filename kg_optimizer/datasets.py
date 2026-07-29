"""
Generic, pluggable eval-dataset schema for the KG optimizer.

The optimizer core never hardcodes a dataset — `load_dataset(path)` reads any JSON
file matching this schema. `from_lifesciences_assets()` is one adapter, built from
existing repo assets (lifesciences_questions.py + data/kg_store.db), shipped purely
as a working example fixture — not a baked-in target.

Gold answers/bridges are optional. Without them, eval_harness/judge fall back to
a heuristic scorer (see kg_optimizer/judge.py). Curating real gold answers/bridges
from the existing manually-verified reports (e.g. LIFESCIENCE_Compliance_Questions_
Verification_Report.md) is a follow-up task, not automated here.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalQuestion:
    question: str
    gold_answer: Optional[str] = None
    gold_sql: Optional[str] = None


@dataclass
class GoldBridge:
    table_a: str
    col_a: str
    table_b: str
    col_b: str
    expected: bool = True


@dataclass
class EvalDataset:
    source_id: str                                  # logical id for cache namespacing
    questions: List[EvalQuestion] = field(default_factory=list)
    gold_bridges: List[GoldBridge] = field(default_factory=list)
    # How to reach the target DB for query execution. Either point at an existing
    # registered source in data/kg_store.db (kg_store_source_id), or supply an
    # explicit connection dict matching the /query payload fields.
    kg_store_source_id: Optional[str] = None
    connection: Optional[Dict[str, Any]] = None
    kg_store_path: str = "data/kg_store.db"


def load_dataset(path: str) -> EvalDataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = [EvalQuestion(**q) for q in raw.pop("questions", [])]
    gold_bridges = [GoldBridge(**b) for b in raw.pop("gold_bridges", [])]
    return EvalDataset(questions=questions, gold_bridges=gold_bridges, **raw)


def save_dataset(dataset: EvalDataset, path: str) -> None:
    raw = asdict(dataset)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


def load_connection_from_kg_store(kg_store_source_id: str, kg_store_path: str = "data/kg_store.db") -> Dict[str, Any]:
    """Generalized version of the connection-loading pattern used by
    run_lifesciences_doc_questions.py — reads db creds + the latest KG
    node/edge snapshot for a registered source."""
    conn = sqlite3.connect(kg_store_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    row = cur.execute(
        "SELECT name, connection_json FROM kg_sources WHERE source_id=?", (kg_store_source_id,)
    ).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"no kg_sources row for source_id={kg_store_source_id!r}")
    cfg = json.loads(row["connection_json"])
    snap = cur.execute(
        "SELECT nodes_json, edges_json FROM kg_snapshots WHERE source_id=?", (kg_store_source_id,)
    ).fetchone()
    conn.close()
    nodes = json.loads(snap["nodes_json"]) if snap else []
    edges = json.loads(snap["edges_json"]) if snap else []
    return {"name": row["name"], "cfg": cfg, "nodes": nodes, "edges": edges}


def gold_bridges_from_metadata_report(
    report: Dict[str, Any],
    include_cardinality_types: Optional[List[str]] = None,
) -> List[GoldBridge]:
    """Derive real gold bridges from a metadata extraction report (the same
    report shape OntologyAgent.run() / dialog_agent.kg_inference_engine.
    _enrich_from_report() consume — see nodes/report_node.py for the producer).

    Three sources, in priority order (all emitted; duplicates naturally
    collapse when scored since GoldBridge identity is table+column):

      1. Declared FKs — report["tables"][t]["foreign_keys"]: explicit,
         highest-confidence ground truth.
      2. fk_candidates — report["fk_candidates"]: inclusion-dependency-based
         FK detection (left_table/left_columns -> right_table/right_columns).
      3. cardinality_relationships — report["cardinality_relationships"]:
         value-overlap-based join detection (left_table/right_table/
         join_columns/type). Weaker signal than 1-2 — a "M:N" relationship
         can reflect a shared categorical dimension (e.g. a repeated
         "region_name") rather than a true join key, not just a missed FK.
         Pass include_cardinality_types (e.g. ["1:1", "1:N", "N:1"]) to
         exclude M:N noise; None includes all types.

    For a source with no declared FKs and no fk_candidates (as observed for
    the LifeSciences test source — both empty), cardinality_relationships is
    the only available signal, so it is included by default.
    """
    gold: List[GoldBridge] = []

    tables: Dict[str, Any] = report.get("tables") or {}
    for tname, tdata in tables.items():
        if not isinstance(tdata, dict):
            continue
        for fk_def in (tdata.get("foreign_keys") or []):
            col = fk_def.get("column", "")
            ref_table = fk_def.get("references_table", "")
            ref_col = fk_def.get("references_column", "")
            if col and ref_table and ref_col:
                gold.append(GoldBridge(table_a=tname, col_a=col, table_b=ref_table, col_b=ref_col, expected=True))

    for fk in report.get("fk_candidates") or []:
        left_table = fk.get("left_table", "")
        right_table = fk.get("right_table", "")
        left_cols = fk.get("left_columns") or []
        right_cols = fk.get("right_columns") or []
        for lc, rc in zip(left_cols, right_cols):
            if left_table and right_table and lc and rc:
                gold.append(GoldBridge(table_a=left_table, col_a=lc, table_b=right_table, col_b=rc, expected=True))

    for card in report.get("cardinality_relationships") or []:
        if include_cardinality_types is not None and card.get("type") not in include_cardinality_types:
            continue
        left_table = card.get("left_table", "")
        right_table = card.get("right_table", "")
        for col in card.get("join_columns") or []:
            if left_table and right_table and col:
                gold.append(GoldBridge(table_a=left_table, col_a=col, table_b=right_table, col_b=col, expected=True))

    return gold


def from_lifesciences_assets(limit: Optional[int] = 10) -> EvalDataset:
    """Example adapter — wraps the existing 136-question LifeSciences set.
    gold_answer is left unset; the judge falls back to its no-gold heuristic
    until someone freezes verified answers from the existing verification
    report into per-question gold_answer values."""
    from lifesciences_questions import QUESTIONS

    qs = QUESTIONS[:limit] if limit else QUESTIONS
    return EvalDataset(
        source_id="lifesciences_testusecase",
        questions=[EvalQuestion(question=q) for q in qs],
        gold_bridges=[],
        kg_store_source_id="ec94dc92-2c1c-43bd-9f0f-d73b64b2b159",
    )
