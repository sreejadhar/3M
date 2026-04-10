"""
LangGraph node: extract schema + statistics for every discovered table.

For each table this node:
  1. Calls SchemaExtractorTool  → columns, PKs, FKs, indexes
  2. Calls MetadataCollectorTool → row count, size, per-column stats
  3. Infers plain-English descriptions and domain labels for each column and table
     using rule-based logic (no LLM — no hallucination risk).
  4. Stores a TableMeta object in state['table_metadata']
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..state import AgentState, ColumnMeta, TableMeta
from ..tools import MetadataCollectorTool, SchemaExtractorTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based domain and description inference (no LLM — grounded in metadata)
# ---------------------------------------------------------------------------

_DOMAIN_RULES: List[tuple] = [
    # (domain_label, regex_pattern_on_lower_column_name)
    ("identifier",      re.compile(r'\b(id|key|code|no|num|number|ref|pk|fk|seq|uuid|guid)\b')),
    ("monetary",        re.compile(r'\b(revenue|cost|price|amount|salary|wage|budget|expense|value|profit|loss|margin|fee|rate|charge)\b')),
    ("percentage",      re.compile(r'(pct|percent|_pct$|_percent$|_ratio$|_rate$|share|proportion)')),
    ("date_time",       re.compile(r'\b(date|time|year|month|day|period|quarter|week|hour|created|updated|modified|timestamp|dt|tm)\b')),
    ("count_quantity",  re.compile(r'\b(count|qty|quantity|volume|units|headcount|fte|staff|employees?|workers?|resources?)\b')),
    ("status_flag",     re.compile(r'\b(status|flag|indicator|active|enabled|deleted|is_|has_|type|category|class|tier|level|grade|rank)\b')),
    ("geographic",      re.compile(r'\b(country|region|city|state|zip|postal|address|location|site|territory|area|zone|district)\b')),
    ("descriptive_text",re.compile(r'\b(name|label|title|description|desc|text|comment|note|remark|summary|detail|narrative)\b')),
    ("weight_measure",  re.compile(r'\b(weight|mass|length|width|height|size|dimension|capacity|volume_unit)\b')),
]


def _infer_column_domain(col_name: str, data_type: str, stats: Dict[str, Any]) -> str:
    """
    Infer a semantic domain label from column name, type, and statistics.
    Returns one of the domain labels in _DOMAIN_RULES or a dtype-based fallback.
    Only uses facts present in the metadata — no guessing.
    """
    name_lower = col_name.lower()

    # Name-based domain matching (highest confidence)
    for domain, pattern in _DOMAIN_RULES:
        if pattern.search(name_lower):
            return domain

    # Data-type fallback
    dtype = (data_type or "").lower()
    if any(t in dtype for t in ("int", "numeric", "decimal", "float", "double", "real", "number")):
        # Distinguish identifiers from measures by cardinality
        unique_count = stats.get("unique_count") or 0
        row_count = stats.get("row_count") or 1
        ratio = unique_count / row_count if row_count else 0
        if ratio >= 0.95:
            return "identifier"
        return "numeric_measure"
    if any(t in dtype for t in ("date", "time", "timestamp")):
        return "date_time"
    if any(t in dtype for t in ("bool", "bit")):
        return "status_flag"
    if any(t in dtype for t in ("char", "varchar", "text", "string", "clob")):
        return "descriptive_text"

    return "unknown"


def _infer_column_description(
    col_name: str,
    data_type: str,
    stats: Dict[str, Any],
    is_pk: bool,
    is_fk: bool,
    fk_ref: Optional[str],
    domain: str,
    pattern_hints: Optional[List[str]],
) -> str:
    """
    Build a plain-English column description from factual metadata only.
    No LLM, no domain knowledge beyond what is present in the data.
    """
    parts: List[str] = []

    # Role context
    roles = []
    if is_pk:
        roles.append("primary key")
    if is_fk and fk_ref:
        roles.append(f"foreign key → {fk_ref}")
    elif is_fk:
        roles.append("foreign key")
    if roles:
        parts.append(f"[{', '.join(roles)}]")

    # Domain + type
    domain_str = domain.replace("_", " ") if domain and domain != "unknown" else ""
    if domain_str:
        parts.append(f"{data_type} {domain_str} column")
    else:
        parts.append(f"{data_type} column")

    # Value pattern hints (from MetadataCollector)
    if pattern_hints:
        parts.append(f"— values match pattern(s): {', '.join(pattern_hints)}")

    # Cardinality context
    unique_count = stats.get("unique_count")
    row_count = stats.get("row_count") or 1
    null_rate = stats.get("null_rate", 0) or 0

    if unique_count is not None:
        ratio = unique_count / row_count if row_count else 0
        top_vals = stats.get("top_values", [])

        if ratio >= 0.99:
            parts.append("(high cardinality — near-unique values)")
        elif unique_count <= 2 and top_vals:
            vals = ", ".join(f"'{v}'" for v in top_vals[:2] if v is not None)
            parts.append(f"(binary: {vals})")
        elif unique_count <= 10 and top_vals:
            vals = ", ".join(f"'{v}'" for v in top_vals[:5] if v is not None)
            more = f" + {unique_count - 5} more" if unique_count > 5 else ""
            parts.append(f"(categorical — {unique_count} distinct values: {vals}{more})")

    # Null rate note
    if null_rate > 0.0:
        parts.append(f"({null_rate * 100:.1f}% null)")

    # Numeric range
    mn = stats.get("min_value")
    mx = stats.get("max_value")
    if mn is not None and mx is not None and domain in ("monetary", "numeric_measure", "count_quantity", "percentage"):
        parts.append(f"[range: {mn} – {mx}]")

    return " ".join(parts)


def _infer_table_description(
    table_name: str,
    columns: List[Any],   # List[ColumnMeta]
    row_count: Optional[int],
    primary_keys: List[str],
    foreign_keys: List[Dict],
) -> str:
    """
    Infer a plain-English table description from its name and structure.
    Grounded in structural facts only — no domain-knowledge guessing.
    """
    name_lower = table_name.lower()

    # Determine likely entity type from common table-name patterns.
    # Checked in priority order — first match wins.
    entity_hint = ""
    if re.search(r'\b(fact|fct|measure|metric|kpi)\b', name_lower):
        entity_hint = "fact/measure table"
    elif re.search(r'\b(dim|dimension)\b', name_lower):
        entity_hint = "dimension table"
    elif re.search(r'\b(lookup|ref|reference|master|mst|code_table|codelist)\b', name_lower):
        entity_hint = "lookup/reference table"
    elif re.search(r'\b(bridge|xref|cross_ref|mapping|map|link|assoc|junction|rel)\b', name_lower):
        entity_hint = "bridge/mapping table"
    # ── OLTP-specific patterns ────────────────────────────────────────────
    elif re.search(r'\b(order|invoice|transaction|payment|shipment|receipt|'
                   r'booking|reservation|claim|ticket|request|case|incident)\b', name_lower):
        entity_hint = "OLTP transaction table"
    elif re.search(r'\b(order_item|invoice_line|line_item|detail|position|entry|'
                   r'component|item|row)\b', name_lower):
        entity_hint = "OLTP transaction line/detail table"
    elif re.search(r'\b(customer|client|account|party|person|employee|emp|staff|'
                   r'vendor|supplier|partner|contact|user|member)\b', name_lower):
        entity_hint = "OLTP entity/master table"
    elif re.search(r'\b(product|item|sku|article|service|catalogue|catalog|'
                   r'asset|equipment|device)\b', name_lower):
        entity_hint = "OLTP product/item master table"
    elif re.search(r'\b(address|location|site|warehouse|store|branch|outlet|'
                   r'territory|region)\b', name_lower):
        entity_hint = "OLTP location/address table"
    elif re.search(r'\b(category|type|subtype|class|group|segment|tier|'
                   r'hierarchy|level)\b', name_lower):
        entity_hint = "OLTP classification/hierarchy table"
    elif re.search(r'\b(status|state|workflow|stage|phase|queue)\b', name_lower):
        entity_hint = "OLTP workflow/status table"
    # ── Structural patterns ───────────────────────────────────────────────
    elif re.search(r'\b(log|audit|history|hist|event|trail|changelog|change_log)\b', name_lower):
        entity_hint = "audit/history table"
    elif re.search(r'\b(staging|stg|stage|temp|tmp|raw|landing|inbound)\b', name_lower):
        entity_hint = "staging/temporary table"
    elif re.search(r'\b(agg|aggregat|summary|summ|rollup|snapshot)\b', name_lower):
        entity_hint = "aggregate/summary table"

    col_count = len(columns)
    pk_str = f"PK: ({', '.join(primary_keys)})" if primary_keys else "no defined primary key"
    fk_count = len(foreign_keys)

    parts = [f"Table '{table_name}'"]
    if entity_hint:
        parts.append(f"— {entity_hint}")
    parts.append(f"with {col_count} column{'s' if col_count != 1 else ''}")
    if row_count is not None:
        parts.append(f"and {row_count:,} row{'s' if row_count != 1 else ''}")
    parts.append(f"({pk_str}")
    if fk_count:
        parts.append(f", {fk_count} FK reference{'s' if fk_count != 1 else ''}")
    parts.append(")")

    # Summarise dominant domains
    domain_counts: Dict[str, int] = {}
    for col in columns:
        d = getattr(col, "domain", None) or "unknown"
        domain_counts[d] = domain_counts.get(d, 0) + 1
    top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:3]
    if top_domains:
        dom_str = ", ".join(f"{d.replace('_', ' ')} ({n})" for d, n in top_domains if d != "unknown")
        if dom_str:
            parts.append(f"Dominant attribute types: {dom_str}.")

    return " ".join(parts)


def extraction_node(state: AgentState) -> AgentState:
    config = state["agent_config"]
    connector = state["connector"]
    cb = getattr(config, "progress_callback", None)

    schema_tool = SchemaExtractorTool(connector=connector)
    meta_tool = MetadataCollectorTool(connector=connector)

    total = len(state["all_tables"])
    if cb:
        cb("extract", "running",
           f"Extracting schema + column statistics for {total} table{'s' if total != 1 else ''}…",
           "Algorithm: column type inference, cardinality sampling, null-rate scan, top-value frequency")

    for idx, (schema_name, table_name) in enumerate(state["all_tables"], 1):
        if table_name in state["tables_done"]:
            continue

        logger.info("Extracting metadata for %s.%s …", schema_name, table_name)
        if cb:
            cb("extract:table", "running",
               f"[{idx}/{total}] {schema_name}.{table_name}",
               "Fetching columns, PKs, FKs, indexes; sampling row stats")
        try:
            # --- Schema extraction ---
            schema_json = schema_tool._run(schema_name, table_name)
            schema_data: Dict[str, Any] = json.loads(schema_json)
            if "error" in schema_data:
                raise RuntimeError(schema_data["error"])

            col_names = [c["name"] for c in schema_data.get("columns", [])]

            # --- Statistics ---
            stats_json = meta_tool._run(
                schema_name, table_name, col_names, config.sample_size
            )
            stats_data: Dict[str, Any] = json.loads(stats_json)

            col_stats: Dict[str, Any] = stats_data.get("column_stats", {})

            # Build ColumnMeta objects with domain + description inference
            col_metas: List[ColumnMeta] = []
            for col in schema_data.get("columns", []):
                cname = col["name"]
                dtype = col.get("data_type", "")
                cs = col_stats.get(cname, {})
                is_pk = col.get("is_primary_key", False)
                is_fk = col.get("is_foreign_key", False)
                fk_ref = col.get("fk_references")
                pattern_hints = cs.get("pattern_hints") or []

                domain = _infer_column_domain(cname, dtype, cs)
                description = _infer_column_description(
                    cname, dtype, cs, is_pk, is_fk, fk_ref, domain, pattern_hints
                )

                col_metas.append(
                    ColumnMeta(
                        name=cname,
                        data_type=dtype,
                        nullable=col.get("nullable", True),
                        is_primary_key=is_pk,
                        is_foreign_key=is_fk,
                        fk_references=fk_ref,
                        unique_count=cs.get("unique_count"),
                        null_count=cs.get("null_count"),
                        row_count=cs.get("row_count"),
                        min_value=cs.get("min_value"),
                        max_value=cs.get("max_value"),
                        avg_value=cs.get("avg_value"),
                        stddev_value=cs.get("stddev_value"),
                        top_values=cs.get("top_values"),
                        description=description,
                        domain=domain,
                        pattern_hints=pattern_hints or None,
                    )
                )

            pks = schema_data.get("primary_keys", [])
            fks = schema_data.get("foreign_keys", [])
            row_count = stats_data.get("row_count")

            table_description = _infer_table_description(
                table_name, col_metas, row_count, pks, fks
            )

            table_meta = TableMeta(
                schema_name=schema_name,
                table_name=table_name,
                row_count=row_count,
                size_bytes=stats_data.get("size_bytes"),
                columns=col_metas,
                primary_keys=pks,
                foreign_keys=fks,
                indexes=schema_data.get("indexes", []),
                table_comment=schema_data.get("table_comment"),
                create_time=schema_data.get("create_time"),
                last_modified=schema_data.get("last_modified"),
                partitioned_by=schema_data.get("partitioned_by", []),
                description=table_description,
            )

            state["table_metadata"][table_name] = table_meta
            state["tables_done"].add(table_name)
            row_str = f"{table_meta.row_count:,}" if table_meta.row_count else "?"
            logger.info("  → %d columns, %s rows", len(col_metas), row_str)
            if cb:
                pk_str = f"PK: {', '.join(pks)}" if pks else "no PK"
                fk_str = f"{len(fks)} FK{'s' if len(fks) != 1 else ''}" if fks else ""
                detail = f"{len(col_metas)} cols · {row_str} rows · {pk_str}" + (f" · {fk_str}" if fk_str else "")
                cb("extract:table", "done",
                   f"[{idx}/{total}] {schema_name}.{table_name}",
                   detail)

        except Exception as exc:
            err = f"Extraction failed for {schema_name}.{table_name}: {exc}"
            logger.error(err)
            if cb:
                cb("extract:table", "error", f"[{idx}/{total}] {schema_name}.{table_name}", str(exc))
            state["errors"].append(err)

    done_count = len(state["tables_done"])
    if cb:
        cb("extract", "done",
           f"Schema extraction complete — {done_count} table{'s' if done_count != 1 else ''} profiled",
           "Domain labels assigned (identifier, monetary, date_time, status_flag, …)")

    state["phase"] = "extracted"
    return state
