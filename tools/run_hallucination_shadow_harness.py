"""
Phase 2 evidence-gathering harness: run the AST-based hallucination resolver
(dialog_agent/nodes/sql_identifier_resolver.py) in shadow mode against real
SQL from the PNC and LIFESCIENCESTESTUSECASE verification reports, using the
REAL schema pulled from data/kg_store.db (not synthetic fixtures), and report
every disagreement with the existing regex-based checks in plan_node.py.

This does not touch plan_node.py's control flow or any live pipeline — it is
a standalone comparison script whose only output is a report for triage
before deciding whether to flip PLAN_NODE_AST_HALLUCINATION_SHADOW / cut over
in Phase 3.

Usage:
    python tools/run_hallucination_shadow_harness.py [--verbose]
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the OLD regex checks directly from plan_node.py without triggering
# its heavy module-level imports (langgraph/anthropic/etc. are already
# satisfied in this env, but keep this resilient regardless).
from dialog_agent.nodes.plan_node import (  # noqa: E402
    _find_hallucinated_tables,
    _find_hallucinated_columns,
    _dotted_literal_parts,
)
from dialog_agent.nodes.sql_identifier_resolver import (  # noqa: E402
    SchemaGraph,
    find_hallucinated_identifiers,
)

KG_STORE = ROOT / "data" / "kg_store.db"

SOURCES = {
    "PNC": {
        "kind": "markdown_report",
        "source_id": "3662d229-3b5c-4807-a05c-34d967196b37",
        "report_md": ROOT / "PNC_DataChat_Verification_Report.md",
        "dialect": "snowflake",
    },
    "LIFESCIENCESTESTUSECASE": {
        "kind": "markdown_report",
        "source_id": "ec94dc92-2c1c-43bd-9f0f-d73b64b2b159",
        "report_md": ROOT / "LIFESCIENCESTESTUSECASE_DataChat_Verification_Report.md",
        "dialect": "snowflake",
    },
    "Claim Underwriting": {
        "kind": "job_results_json",
        # Two source_ids exist for this name in kg_store.db with identical
        # report_json (same schema snapshot); either resolves the same tables.
        "source_id": "4fc2df3a-d41f-477e-8bc9-de99ba7bb5dd",
        "results_json": ROOT / "claim_underwriting_doc_run_results.json",
        "dialect": "snowflake",
    },
}

SQL_BLOCK_RE = re.compile(r"```sql\s*\n(.*?)```", re.DOTALL)

# Distinguishes ground-truth SQL pulled from the source .docx (never executed
# by plan_node.py — not production evidence) from SQL DataChat actually
# generated (the real signal for whether the resolver would change live
# behavior). A block's provenance is whichever of these two headers most
# recently preceded it in the report.
_REFERENCE_HEADER_RE = re.compile(r"\*\*Reference query")
_GENERATED_HEADER_RE = re.compile(r"\*\*DataChat-generated")


def load_schema(source_id: str) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Return ({TABLE_NAME: {columns}}, [table names]) from kg_store.db's
    stored report_json for this source — the real, previously-indexed schema,
    not a hand-written fixture."""
    conn = sqlite3.connect(str(KG_STORE))
    try:
        cur = conn.cursor()
        cur.execute("SELECT report_json FROM kg_sources WHERE source_id = ?", (source_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError(f"no report_json for source {source_id}")
        report = json.loads(row[0])
    finally:
        conn.close()

    tables = report.get("tables", {})
    table_columns_map: Dict[str, Set[str]] = {}

    if isinstance(tables, dict):
        items = tables.items()
    else:
        items = ((t.get("table_name"), t) for t in tables)

    for name, tinfo in items:
        if not name:
            continue
        cols = tinfo.get("columns", [])
        if isinstance(cols, dict):
            col_names = set(cols.keys())
        else:
            col_names = {c.get("name") for c in cols if c.get("name")}
        table_columns_map[name.upper()] = {c.lower() for c in col_names}

    table_labels = sorted(table_columns_map.keys())
    return table_columns_map, table_labels


def extract_sql_blocks(report_md: Path) -> List[Tuple[str, str]]:
    """Return [(sql, provenance)] where provenance is 'reference' or
    'generated', determined by the nearest preceding header."""
    text = report_md.read_text(encoding="utf-8")

    # Build a sorted list of (position, label) for both header kinds.
    markers = [(m.start(), "reference") for m in _REFERENCE_HEADER_RE.finditer(text)]
    markers += [(m.start(), "generated") for m in _GENERATED_HEADER_RE.finditer(text)]
    markers.sort()

    results: List[Tuple[str, str]] = []
    for m in SQL_BLOCK_RE.finditer(text):
        sql = m.group(1).strip()
        if not sql or sql.upper().startswith("--"):
            continue
        # Provenance = label of the nearest marker before this SQL block.
        provenance = "unknown"
        for pos, label in markers:
            if pos < m.start():
                provenance = label
            else:
                break
        results.append((sql, provenance))
    return results


def extract_sql_from_job_results(results_json: Path) -> List[Tuple[str, str]]:
    """
    Return [(sql, provenance)] from a DataChat job-results JSON dump
    (claim_underwriting_doc_run_results.json shape: a list of
    {question, result: {query_debug: [{sql, ...}, ...], errors: [...]}}).

    Every SQL statement here already executed against the live warehouse
    (or was attempted and errored) — all of it is 'generated' provenance,
    there is no separate reference-query concept in this format.
    """
    data = json.loads(results_json.read_text(encoding="utf-8"))
    results: List[Tuple[str, str]] = []
    for entry in data:
        for qd in entry.get("result", {}).get("query_debug", []) or []:
            sql = (qd.get("sql") or "").strip()
            if sql:
                results.append((sql, "generated"))
    return results


def known_columns_for_regex(table_columns_map: Dict[str, Set[str]], table_labels: List[str]) -> Set[str]:
    """Reproduce plan_node.py's `known_columns` construction (flat set across
    all tables + table labels + dotted-literal-part augmentation) so the old
    regex check runs under the same conditions it does in production."""
    known: Set[str] = set()
    for cols in table_columns_map.values():
        known.update(cols)
    known.update(t.lower() for t in table_labels)
    known.update(_dotted_literal_parts(known))
    return known


def run_for_source(name: str, cfg: dict, verbose: bool) -> dict:
    table_columns_map, table_labels = load_schema(cfg["source_id"])
    known_columns = known_columns_for_regex(table_columns_map, table_labels)
    schema = SchemaGraph(table_columns_map)

    if cfg["kind"] == "job_results_json":
        sql_blocks = extract_sql_from_job_results(cfg["results_json"])
    else:
        sql_blocks = extract_sql_blocks(cfg["report_md"])

    stats = {
        "total": len(sql_blocks),
        "generated_total": sum(1 for _, p in sql_blocks if p == "generated"),
        "reference_total": sum(1 for _, p in sql_blocks if p == "reference"),
        "agreements": 0,
        "disagreements": [],
    }

    for i, (sql, provenance) in enumerate(sql_blocks):
        old_bad_tables = sorted({t.lower() for t in _find_hallucinated_tables(sql, table_labels)})
        old_bad_cols = sorted({c.lower() for c in _find_hallucinated_columns(sql, known_columns)})

        new_bad = find_hallucinated_identifiers(sql, schema, dialect=cfg["dialect"])
        new_bad_tables = sorted({b.name.lower() for b in new_bad if b.kind == "table"})
        new_bad_cols = sorted({b.name.lower() for b in new_bad if b.kind == "column"})

        if old_bad_tables == new_bad_tables and old_bad_cols == new_bad_cols:
            stats["agreements"] += 1
            continue

        stats["disagreements"].append({
            "index": i,
            "provenance": provenance,
            "sql": sql,
            "regex_tables": old_bad_tables,
            "regex_cols": old_bad_cols,
            "ast_tables": new_bad_tables,
            "ast_cols": new_bad_cols,
        })

    return stats


def main() -> None:
    verbose = "--verbose" in sys.argv
    overall_lines = []

    for name, cfg in SOURCES.items():
        stats = run_for_source(name, cfg, verbose)
        generated_disagreements = [d for d in stats["disagreements"] if d["provenance"] == "generated"]
        reference_disagreements = [d for d in stats["disagreements"] if d["provenance"] != "generated"]

        overall_lines.append(f"\n{'=' * 70}\n{name}  ({cfg['source_id']})\n{'=' * 70}")
        overall_lines.append(
            f"SQL blocks scanned: {stats['total']} "
            f"(generated={stats['generated_total']}, reference={stats['reference_total']})  |  "
            f"agreements: {stats['agreements']}  |  "
            f"disagreements: {len(stats['disagreements'])} "
            f"(generated={len(generated_disagreements)}, reference/other={len(reference_disagreements)})"
        )
        overall_lines.append(
            "\n*** DataChat-GENERATED disagreements (real production evidence) ***"
            if generated_disagreements else
            "\n*** No disagreements on DataChat-generated SQL — reference-only differences below ***"
        )
        for d in generated_disagreements:
            overall_lines.append(
                f"\n--- disagreement #{d['index']} [{d['provenance']}] ---\n"
                f"regex: tables={d['regex_tables']} cols={d['regex_cols']}\n"
                f"ast  : tables={d['ast_tables']} cols={d['ast_cols']}\n"
                f"sql  : {d['sql'][:300]}"
            )
        overall_lines.append(f"\n--- reference/other disagreements ({len(reference_disagreements)}), collapsed ---")
        for d in reference_disagreements:
            overall_lines.append(
                f"#{d['index']} [{d['provenance']}] regex: t={d['regex_tables']} c={d['regex_cols']} "
                f"| ast: t={d['ast_tables']} c={d['ast_cols']}"
            )

    report_text = "\n".join(overall_lines)
    print(report_text)

    out_path = ROOT / "hallucination_shadow_report.txt"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
