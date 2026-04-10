"""
understand_node — Build a structured schema context from KG nodes/edges.

No LLM call here; this node transforms the raw graph data (nodes = classes /
tables, edges = relationships) into a concise, human-readable schema description
that downstream nodes feed to the LLM.

Key design:
- The KG node `label` = original table name (via rdfs:label in the ontology).
- The KG node `title` tooltip contains column names and types.
- We extract these and present them with the correct schema-qualified name
  (e.g. `public.orders`) so the LLM generates runnable SQL on the first try.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..config import DialogConfig
from ..state import DialogState
from .execute_node import _get_cached_file_db

logger = logging.getLogger(__name__)

_FILE_BASED_TYPES = {"sqlite", "csv", "excel"}

# Max distinct sample values for high-cardinality columns (IDs, free text)
_SAMPLE_LIMIT = 3
# Categorical columns (low distinct count) get more samples so the LLM can
# resolve synonym mismatches between user terminology and data labels.
_SAMPLE_LIMIT_CATEGORICAL = 5
# Threshold: columns with <= this many distinct values are treated as categorical
_CATEGORICAL_DISTINCT_MAX = 50
# Only sample columns whose distinct count is <= this (to skip IDs/timestamps)
_SAMPLE_DISTINCT_MAX = 200


def _to_sql_col(name: str) -> str:
    """
    Sanitize a column name to a valid SQLite identifier.
    Must match execute_node._safe_col exactly so the schema context
    uses the same column names that are actually stored in SQLite.
    """
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return ("col_" + s if s and s[0].isdigit() else s) or "col"


def _to_sql_table(name: str) -> str:
    """Sanitize a table/sheet name to a valid SQLite identifier."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return ("t_" + s if s and s[0].isdigit() else s) or "tbl"


def _sample_file_data(config: DialogConfig) -> tuple:
    """
    Collect distinct non-null values per column from the cached temp SQLite DB,
    and build taxonomy hierarchies for parent-child categorical column pairs.

    For categorical columns (distinct count <= _CATEGORICAL_DISTINCT_MAX) we
    retrieve ALL distinct values (up to _SAMPLE_LIMIT_CATEGORICAL) and mark
    the column as categorical.  Parent-child pairs (e.g. category → sub_category)
    are detected by naming convention and the full cross-tab hierarchy is queried.

    Returns:
        samples   : { sql_table_name: { sql_col_name: {"values": [...], "categorical": bool} } }
        hierarchy : { sql_table_name: { parent_col: { parent_val: [child_vals] } } }
    """
    import sqlite3

    db    = config.db_type.lower()
    fpath = config.db_file_path
    if not fpath:
        return {}

    try:
        if db == "sqlite":
            conn = sqlite3.connect(fpath, check_same_thread=False)
        else:
            tmp_path = _get_cached_file_db(config)
            conn = sqlite3.connect(tmp_path, check_same_thread=False)

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]

        samples: Dict[str, Dict[str, Any]] = {}
        for tbl in tables:
            try:
                cur.execute(f'PRAGMA table_info("{tbl}")')
                col_rows = cur.fetchall()   # (cid, name, type, notnull, dflt, pk)
                col_samples: Dict[str, Any] = {}
                for row in col_rows:
                    col      = row[1]
                    col_type = (row[2] or "").lower()
                    is_text  = not any(t in col_type for t in ("int", "real", "float", "double", "numeric", "decimal", "bool"))
                    try:
                        cur.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{tbl}"')
                        distinct_count = (cur.fetchone() or [0])[0]
                        if distinct_count > _SAMPLE_DISTINCT_MAX:
                            continue
                        is_categorical = is_text and distinct_count <= _CATEGORICAL_DISTINCT_MAX
                        limit = _SAMPLE_LIMIT_CATEGORICAL if is_categorical else _SAMPLE_LIMIT
                        cur.execute(
                            f'SELECT DISTINCT "{col}" FROM "{tbl}" '
                            f'WHERE "{col}" IS NOT NULL ORDER BY "{col}" LIMIT {limit}'
                        )
                        vals = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                        if vals:
                            col_samples[col] = {"values": vals, "categorical": is_categorical}
                    except Exception:
                        pass
                if col_samples:
                    samples[tbl] = col_samples
            except Exception:
                pass

        # Build taxonomy hierarchies for parent-child categorical column pairs
        full_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for tbl, col_map in samples.items():
            pairs = _detect_parent_child_pairs(col_map)
            if pairs:
                tbl_hier = _build_taxonomy_hierarchy(conn, tbl, pairs)
                if tbl_hier:
                    full_hierarchy[tbl] = tbl_hier
                    logger.info(
                        "understand_node: built taxonomy hierarchy for table %r: %s",
                        tbl, {p: list(m.keys()) for p, m in tbl_hier.items()},
                    )

        conn.close()
        return samples, full_hierarchy

    except Exception as exc:
        logger.warning("understand_node: sample_file_data failed — %s", exc)
        return {}, {}


def _detect_parent_child_pairs(col_samples: Dict[str, Any]) -> List[tuple]:
    """
    Detect parent→child column pairs from categorical columns in a single table.

    Two heuristics (both must agree before a pair is returned):

    1. Naming pattern — "sub_X" is a child of "X":
         sub_category → category
         sub_region   → region
         sub_segment  → segment

    2. Cardinality — the parent has FEWER distinct values than the child:
         parent distinct count  <  child distinct count

    Returns a list of (parent_col, child_col) tuples.
    """
    cat_cols = {
        col for col, info in col_samples.items()
        if isinstance(info, dict) and info.get("categorical")
    }
    pairs: List[tuple] = []
    for col in sorted(cat_cols):
        if col.startswith("sub_"):
            parent_candidate = col[4:]          # strip "sub_" prefix
            if parent_candidate in cat_cols:
                p_count = len(col_samples[parent_candidate].get("values", []))
                c_count = len(col_samples[col].get("values", []))
                if p_count <= c_count:          # parent must have ≤ distinct vals
                    pairs.append((parent_candidate, col))
    return pairs


def _load_samples_from_catalog(source_id: str) -> tuple:
    """
    For non-file sources (PostgreSQL, etc.) load categorical column values and
    taxonomy hierarchies from the metadata catalog (md_attributes) that was
    populated during indexing.

    Returns the same (samples, hierarchy) shape as _sample_file_data so the
    rest of understand_node can treat them identically.

    samples   : { table_name: { col_name: {"values": [...], "categorical": bool} } }
    hierarchy : { table_name: { parent_col: { parent_val: [child_vals] } } }

    Hierarchy note: without a live DB we cannot build a true cross-tab
    (which parent values own which child values).  We detect parent-child column
    pairs by naming convention (sub_X → X) and expose the parent column values
    in the context so the LLM filters at the correct level.
    """
    try:
        import metadata_catalog as _mc

        entities = _mc.list_entities(source_id)
        if not entities:
            return {}, {}

        samples: Dict[str, Dict[str, Any]] = {}

        for ent in entities:
            full = _mc.get_entity(ent["metadata_id"])
            if not full:
                continue
            tbl = ent["table_name"]
            for attr in full.get("attributes", []):
                stat_type = (attr.get("statistical_type") or "").strip()
                top_vals = attr.get("top_values") or []
                col = attr["column_name"]

                unique_count = attr.get("unique_count")

                # Expose columns that have known categorical/ordinal type.
                # - If top_values are stored: show them as sample values.
                # - If top_values are empty but unique_count is low (≤ 50):
                #   still mark the column as categorical so the LLM knows to
                #   use LIKE rather than exact equality — even without samples.
                if stat_type in ("categorical", "ordinal"):
                    if top_vals:
                        samples.setdefault(tbl, {})[col] = {
                            "values": top_vals,
                            "categorical": True,
                        }
                    elif unique_count is not None and unique_count <= 50:
                        samples.setdefault(tbl, {})[col] = {
                            "values": [],
                            "categorical": True,
                            "values_unknown": True,
                        }

        # Detect parent-child pairs by naming convention (sub_X → X).
        # Without a live DB cross-tab, represent the hierarchy as a flat
        # {parent_col: {"(all children)": child_vals}} mapping so the LLM
        # knows to filter at parent level and sees the child value list.
        hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for tbl, col_map in samples.items():
            pairs = _detect_parent_child_pairs(col_map)
            if not pairs:
                continue
            tbl_hier: Dict[str, Dict[str, List[str]]] = {}
            for parent_col, child_col in pairs:
                c_vals = col_map.get(child_col, {}).get("values", [])
                # Use the child values as a flat group under a descriptive key
                tbl_hier[parent_col] = {"(all sub-values)": c_vals}
            if tbl_hier:
                hierarchy[tbl] = tbl_hier

        if samples:
            logger.info(
                "understand_node: loaded catalog samples for %d tables (source %s)",
                len(samples), source_id[:8],
            )
        return samples, hierarchy

    except Exception as exc:
        logger.warning(
            "understand_node: _load_samples_from_catalog failed for %s — %s",
            source_id[:8], exc,
        )
        return {}, {}


def _build_taxonomy_hierarchy(
    conn: Any,
    tbl: str,
    pairs: List[tuple],
) -> Dict[str, Dict[str, List[str]]]:
    """
    For each (parent_col, child_col) pair, query the cross-tab to get
    the mapping from every parent value → list of child values.

    Returns:
        { parent_col: { parent_value: [child_value, ...] } }
    """
    import sqlite3

    hierarchy: Dict[str, Dict[str, List[str]]] = {}
    cur = conn.cursor()
    for parent_col, child_col in pairs:
        try:
            cur.execute(
                f'SELECT DISTINCT "{parent_col}", "{child_col}" '
                f'FROM "{tbl}" '
                f'WHERE "{parent_col}" IS NOT NULL AND "{child_col}" IS NOT NULL '
                f'ORDER BY "{parent_col}", "{child_col}"'
            )
            mapping: Dict[str, List[str]] = {}
            for pval, cval in cur.fetchall():
                mapping.setdefault(str(pval), []).append(str(cval))
            if mapping:
                hierarchy[parent_col] = mapping
        except Exception:
            pass
    return hierarchy


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_columns_from_title(title: str, table_label: str) -> List[str]:
    """
    Parse the KG node title tooltip to get a list of column definitions.

    KG translate_node formats the title as:
        Class: <name>
        <rdfs:comments...>

        Properties:
          col1: xsd_type1
          col2: xsd_type2
    """
    cols: List[str] = []
    in_props = False
    for line in title.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("properties:"):
            in_props = True
            continue
        if in_props:
            # Lines look like "  col_name: integer"
            # Stop if we hit something that looks like a new section
            if stripped.startswith("Class:") or stripped.startswith("["):
                in_props = False
            elif ":" in stripped:
                cols.append(stripped)
    return cols


def _extract_comments_from_title(title: str) -> List[str]:
    """
    Return the rdfs:comment lines from the node title (row_count, table description etc.)
    Skips the first 'Class: X' line, the 'Properties:' block, and ALL FD annotation lines.

    FD annotations are excluded because they reference column names from the
    determinant/dependent sets (e.g. "'Segment_ID' determines 'Segment_Flag'") which
    the LLM misreads as valid SELECT-able columns, causing "column does not exist"
    errors at runtime.

    build_node marks each FD comment block with a leading "FD-ANNOTATION:" sentinel line
    and a trailing "FD type: ... | confidence=N" line.  We use a stateful flag to skip
    the ENTIRE block including the description line between them, which also contains
    column names and would otherwise leak through.
    """
    comments: List[str] = []
    in_props    = False
    in_fd_block = False
    for line in title.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("class:"):
            continue
        if stripped.lower().startswith("properties:"):
            in_props = True
            continue
        if in_props:
            break

        # Detect the start of an FD annotation block (sentinel from build_node)
        if stripped.startswith("FD-ANNOTATION:"):
            in_fd_block = True
            continue

        # Detect the closing line of an FD block and exit the block
        if "FD type:" in stripped or "| confidence=" in stripped:
            in_fd_block = False   # reset — next line is a new comment
            continue

        # While inside an FD block, drop every line (the plain-English description
        # also contains column names like "'Segment_ID' determines 'Segment_Flag'")
        if in_fd_block:
            continue

        comments.append(stripped)
    return comments


def _qualified(schema: str, table: str) -> str:
    """Return `schema.table` if schema is set, else just `table`."""
    return f"{schema}.{table}" if schema else table


def _summarise_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    db_schema: str,
    samples: Optional[Dict[str, Dict[str, List]]] = None,
    hierarchy: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None,
) -> str:
    """
    Convert KG nodes/edges to a schema description the SQL planner can use.

    Produces two sections:
      1. QUICK REFERENCE — a compact table list with exact qualified names
      2. DETAILED SCHEMA — column names, types, and relationships per table
    """
    if not nodes:
        return (
            "(No schema context available from the knowledge graph. "
            "Generate SQL based on the natural language question alone.)"
        )

    # Index edges by source node id
    edges_by_src: Dict[str, List[Dict]] = {}
    for e in edges:
        src = e.get("from", "")
        edges_by_src.setdefault(src, []).append(e)

    # Build an index of node id → label for edge target lookup
    node_label_by_id: Dict[str, str] = {
        n.get("id", ""): n.get("label", "") for n in nodes
    }

    lines: List[str] = []

    # ── Section 1: Quick reference ────────────────────────────────────────────
    lines.append("DATABASE SCHEMA CONTEXT")
    lines.append("=" * 60)
    if db_schema:
        lines.append(f"Schema: {db_schema}")
        lines.append(
            "IMPORTANT: Always write table names as "
            f"`{db_schema}.tablename` in your SQL queries."
        )
    lines.append("")

    lines.append("AVAILABLE TABLES (use these exact qualified names in SQL):")
    for node in nodes:
        label = node.get("label", "")
        if label:
            # For file-based sources, show the sanitized SQLite table name so
            # the LLM generates SQL that actually matches what is in SQLite.
            sql_name = _to_sql_table(label) if samples is not None else label
            lines.append(f"  - {_qualified(db_schema, sql_name)}")
    lines.append("")

    # ── Section 1b: Join keys ──────────────────────────────────────────────────
    # 1. Same-name columns appearing in 2+ tables (implicit join candidates).
    col_to_tables: Dict[str, List[str]] = {}
    for node in nodes:
        lbl   = node.get("label", "")
        title = node.get("title", "")
        sql_tbl = _to_sql_table(lbl) if samples is not None else lbl
        for col_def in _extract_columns_from_title(title, lbl):
            orig = col_def.split(":")[0].strip()
            sql_col = _to_sql_col(orig) if samples is not None else orig
            col_to_tables.setdefault(sql_col.lower(), []).append(
                _qualified(db_schema, sql_tbl)
            )

    shared_cols = {
        col: tbls
        for col, tbls in col_to_tables.items()
        if len(tbls) >= 2
    }

    # 2. Explicit FK join pairs parsed from ObjectProperty rdfs:comments.
    #    These capture cross-name FKs (e.g. orders.customer_id → customers.id).
    explicit_fk_lines: List[str] = []
    for e in edges:
        src_lbl = node_label_by_id.get(e.get("from", ""), "")
        tgt_lbl = node_label_by_id.get(e.get("to", ""), "")
        if not src_lbl or not tgt_lbl:
            continue
        src_tbl = _to_sql_table(src_lbl) if samples is not None else src_lbl
        tgt_tbl = _to_sql_table(tgt_lbl) if samples is not None else tgt_lbl
        src_q   = _qualified(db_schema, src_tbl)
        tgt_q   = _qualified(db_schema, tgt_tbl)
        for pair in e.get("join_columns", []):
            if len(pair) == 2:
                sc = _to_sql_col(pair[0]) if samples is not None else pair[0]
                tc = _to_sql_col(pair[1]) if samples is not None else pair[1]
                explicit_fk_lines.append(f"  - JOIN ON {src_q}.{sc} = {tgt_q}.{tc}")

    if shared_cols or explicit_fk_lines:
        lines.append(
            "POSSIBLE JOIN KEYS — use ONLY these in JOIN ON conditions:"
        )
        for col, tbls in sorted(shared_cols.items()):
            lines.append(f"  - {col}: shared by {', '.join(tbls)}")
        for fk_line in explicit_fk_lines:
            lines.append(fk_line)
        lines.append(
            "WARNING: Do NOT use any column in a JOIN ON clause unless it "
            "appears in this list or is shown on a FK line below."
        )
    else:
        lines.append(
            "JOIN KEYS: No columns are shared between tables according to the "
            "schema. Do NOT join these tables — query each one separately."
        )
    lines.append("")

    # ── Section 2: Detailed schema per table ──────────────────────────────────
    lines.append("DETAILED SCHEMA:")
    lines.append("-" * 60)

    for node in nodes:
        node_id   = node.get("id", "")
        label     = node.get("label", node_id)
        title     = node.get("title", "")

        # For file-based sources, display the sanitized table name (the actual
        # SQLite table name) so the LLM writes correct FROM/JOIN clauses.
        display_table = _to_sql_table(label) if samples is not None else label
        qualified_name = _qualified(db_schema, display_table)
        lines.append(f"\nTable: {qualified_name}")

        # Metadata comments (row_count, FD hints)
        for comment in _extract_comments_from_title(title):
            lines.append(f"  -- {comment}")

        # Columns
        cols = _extract_columns_from_title(title, label)
        if cols:
            lines.append("  Columns:")
            # For file-based sources, look up sample values using the sanitized
            # table name, since that is what is actually stored in SQLite.
            sql_table  = _to_sql_table(label) if samples is not None else label
            tbl_samples   = (samples or {}).get(sql_table, {})
            tbl_hierarchy = (hierarchy or {}).get(sql_table, {})
            for col in cols:
                original_col = col.split(":")[0].strip()
                col_type     = col[len(original_col):].strip()  # e.g. ": integer"
                # Translate original KG column name → SQL-safe name used in SQLite
                sql_col  = _to_sql_col(original_col) if samples is not None else original_col
                col_info = tbl_samples.get(sql_col)
                display  = f"{sql_col}{col_type}" if samples is not None else col
                if col_info:
                    # col_info is {"values": [...], "categorical": bool, "values_unknown"?: bool}
                    vals           = col_info.get("values", []) if isinstance(col_info, dict) else col_info
                    categorical    = col_info.get("categorical", False) if isinstance(col_info, dict) else False
                    values_unknown = col_info.get("values_unknown", False) if isinstance(col_info, dict) else False
                    sample_str     = ", ".join(repr(v) for v in vals)
                    if categorical and values_unknown:
                        lines.append(
                            f"    {display}  [categorical — stored values not sampled;"
                            f" use LIKE '%keyword%' filter, NOT exact equality]"
                        )
                    elif categorical:
                        lines.append(
                            f"    {display}  [categorical — ALL stored values: {sample_str}]"
                            f"  ← your filter term may differ from these labels"
                        )
                        # Show taxonomy hierarchy when this column is a parent
                        if sql_col in tbl_hierarchy:
                            child_col = next(
                                (c for c in tbl_samples if f"sub_{sql_col}" == c or c.startswith(f"sub_{sql_col}")),
                                None,
                            )
                            # Find the child column name by matching tbl_hierarchy key
                            # (hierarchy is keyed by parent_col, values are {parent_val: [child_vals]})
                            mapping = tbl_hierarchy[sql_col]
                            # Detect child column name from detect_parent_child_pairs output
                            pairs = _detect_parent_child_pairs(tbl_samples)
                            child_col_name = next((c for p, c in pairs if p == sql_col), None)
                            if child_col_name:
                                lines.append(
                                    f"      TAXONOMY HIERARCHY ({sql_col} → {child_col_name}):"
                                )
                                for pval, cvals in sorted(mapping.items()):
                                    children = " | ".join(cvals)
                                    lines.append(f"        '{pval}' → [{children}]")
                                lines.append(
                                    f"      ← Filter at {sql_col} level to include ALL {child_col_name} values:"
                                    f" WHERE {sql_col} = '<parent>' captures every child automatically"
                                )
                    else:
                        lines.append(f"    {display}  [sample values: {sample_str}]")
                else:
                    lines.append(f"    {display}")

        # Foreign-key relationships (edges)
        for edge in edges_by_src.get(node_id, []):
            tgt_label   = node_label_by_id.get(edge.get("to", ""), edge.get("to", "?"))
            tgt_display  = _to_sql_table(tgt_label) if samples is not None else tgt_label
            tgt_qualified = _qualified(db_schema, tgt_display)
            rel_lbl      = edge.get("label", "")
            join_cols    = edge.get("join_columns", [])
            if join_cols:
                for pair in join_cols:
                    if len(pair) == 2:
                        sc = _to_sql_col(pair[0]) if samples is not None else pair[0]
                        tc = _to_sql_col(pair[1]) if samples is not None else pair[1]
                        lines.append(
                            f"  FK ({rel_lbl}): "
                            f"JOIN ON {qualified_name}.{sc} = {tgt_qualified}.{tc}"
                        )
            else:
                edge_title = edge.get("title", "")
                lines.append(f"  FK: {rel_lbl} -> {tgt_qualified}  ({edge_title})")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ── node ──────────────────────────────────────────────────────────────────────

def _build_nodes_from_file_db(config: DialogConfig) -> List[Dict[str, Any]]:
    """
    Fallback: when kg_nodes is empty (KG pipeline didn't run or failed), build
    synthetic KG nodes by reading the schema directly from the cached SQLite DB.
    This lets the LLM generate SQL even without a KG graph.
    """
    import sqlite3

    db    = config.db_type.lower()
    fpath = config.db_file_path
    if not fpath:
        return []

    try:
        if db == "sqlite":
            conn = sqlite3.connect(fpath, check_same_thread=False)
        else:
            from .execute_node import _get_cached_file_db
            tmp_path = _get_cached_file_db(config)
            conn = sqlite3.connect(tmp_path, check_same_thread=False)

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]

        nodes: List[Dict[str, Any]] = []
        for tbl in tables:
            cur.execute(f'PRAGMA table_info("{tbl}")')
            pragma_rows = cur.fetchall()   # (cid, name, type, notnull, dflt, pk)
            col_lines: List[str] = []
            for row in pragma_rows:
                col_name = row[1]
                raw_type = (row[2] or "text").lower()
                if "int" in raw_type:
                    simple_type = "integer"
                elif any(t in raw_type for t in ("real", "float", "double", "numeric", "decimal")):
                    simple_type = "float"
                else:
                    simple_type = "text"
                col_lines.append(f"  {col_name}: {simple_type}")

            title = "Class: {}\nProperties:\n{}".format(tbl, "\n".join(col_lines))
            nodes.append({"id": tbl, "label": tbl, "title": title})

        conn.close()
        logger.info(
            "understand_node: built %d synthetic KG nodes from SQLite schema (KG fallback)",
            len(nodes),
        )
        return nodes

    except Exception as exc:
        logger.warning("understand_node: fallback DB schema read failed — %s", exc)
        return []


def understand_node(state: DialogState) -> DialogState:
    """Build schema_context from KG graph data, qualified with the target schema."""
    logger.info("=== understand_node ===")

    nodes     = state.get("kg_nodes") or []
    edges     = state.get("kg_edges") or []
    config: DialogConfig = state["config"]
    db_schema = config.db_schema or ""

    # ── Multi-KG path ──────────────────────────────────────────────────────────
    active_kg_ids = state.get("active_kg_ids") or []
    if len(active_kg_ids) > 1:
        # Group nodes by kg_id property (nodes tagged by dialog_api with their source kg_id)
        kg_node_groups: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            kid = n.get("kg_id", "")
            kg_node_groups.setdefault(kid, []).append(n)

        sections: List[str] = []
        for kid in active_kg_ids:
            kg_nodes_group = kg_node_groups.get(kid, [])
            section_text = _summarise_graph(kg_nodes_group, edges, db_schema, None)
            sections.append(f"=== KG: {kid} ===\n{section_text}")

        # Append cross-KG bridge section
        bridges = state.get("kg_bridges_active") or []
        if bridges:
            bridge_lines: List[str] = []
            for b in bridges:
                bridge_lines.append(
                    f"  {b.get('from_kg','')}.{b.get('from_column','')}  →  "
                    f"{b.get('to_kg','')}.{b.get('to_column','')}  "
                    f"[{b.get('join_type','FK')}]  "
                    f"(use this as the join key when combining results)"
                )
            sections.append("=== CROSS-KG BRIDGES ===\n" + "\n".join(bridge_lines))

        schema_context = "\n\n".join(sections)
        logger.info(
            "Multi-KG schema context built: %d chars, %d KGs, %d bridges",
            len(schema_context), len(active_kg_ids), len(bridges),
        )
        state["schema_context"] = schema_context
        state["active_kpis"]   = []
        state["phase"] = "understand"
        return state

    # ── Single-KG path (unchanged) ─────────────────────────────────────────────
    # If the KG pipeline didn't run or failed, fall back to reading the schema
    # directly from the SQLite DB so queries still work.
    if not nodes and config.db_type.lower() in _FILE_BASED_TYPES and config.db_file_path:
        nodes = _build_nodes_from_file_db(config)
        if nodes:
            logger.info(
                "understand_node: kg_nodes was empty — using direct DB schema fallback (%d tables)",
                len(nodes),
            )
        else:
            state["errors"] = state.get("errors") or []
            state["errors"].append(
                "understand_node: kg_nodes is empty and fallback DB schema read also failed. "
                "The KG pipeline may not have completed for this source."
            )

    # Inject sample values so the LLM sees actual data values and generates
    # accurate WHERE clauses.  File-based sources are sampled live; DB-backed
    # sources (PostgreSQL etc.) use top_values / taxonomy_tree from the catalog.
    samples:   Optional[Dict[str, Dict[str, List]]] = None
    hierarchy: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None
    source_id_for_samples = getattr(config, "source_id", "") or ""
    if config.db_type.lower() in _FILE_BASED_TYPES and config.db_file_path:
        samples, hierarchy = _sample_file_data(config)
        if samples:
            logger.info("understand_node: sampled %d tables for value hints", len(samples))
        if hierarchy:
            logger.info(
                "understand_node: taxonomy hierarchy built for %d table(s)", len(hierarchy)
            )
    elif source_id_for_samples:
        # Non-file source: load categorical values and hierarchy from md_attributes
        # (populated by the indexer via infer_taxonomy / enrich_taxonomy).
        samples, hierarchy = _load_samples_from_catalog(source_id_for_samples)
        if hierarchy:
            logger.info(
                "understand_node: catalog taxonomy hierarchy loaded for %d table(s)",
                len(hierarchy),
            )

    schema_context = _summarise_graph(nodes, edges, db_schema, samples, hierarchy)

    logger.info(
        "Schema context built: %d chars, %d tables, %d relationships, schema=%r",
        len(schema_context), len(nodes), len(edges), db_schema,
    )

    # Populate categorical_columns for resolve_node so it can map user terms
    # to exact stored data values without re-parsing the schema context string.
    categorical_columns: Dict[str, Dict[str, list]] = {}
    if samples:
        for tbl, col_map in samples.items():
            for col, info in col_map.items():
                if isinstance(info, dict) and info.get("categorical"):
                    categorical_columns.setdefault(tbl, {})[col] = info.get("values", [])

    # Load active KPI definitions for this source (if source_id is set)
    active_kpis: List[Dict] = []
    source_id = source_id_for_samples  # already extracted above
    if source_id:
        try:
            import kpi_store as _kpi_store
            active_kpis = _kpi_store.list_kpis(source_id=source_id, status="active")
            if active_kpis:
                logger.info(
                    "understand_node: loaded %d active KPI(s) for source %s",
                    len(active_kpis), source_id[:8],
                )
                kpi_lines = [
                    "",
                    "=" * 60,
                    "DEFINED KPIs — PRE-CONFIGURED BUSINESS METRICS",
                    "=" * 60,
                    "The following KPIs have been defined by the BI Manager for this data source.",
                    "When a user asks about one of these metrics, use the sql_expression directly.",
                    "",
                ]
                for kpi in active_kpis:
                    kpi_lines.append(f"KPI: {kpi['name']}")
                    if kpi.get("description"):
                        kpi_lines.append(f"  Description : {kpi['description']}")
                    if kpi.get("category"):
                        kpi_lines.append(f"  Category    : {kpi['category']}")
                    if kpi.get("unit"):
                        kpi_lines.append(f"  Unit        : {kpi['unit']}")
                    if kpi.get("sql_expression"):
                        kpi_lines.append(f"  SQL formula : {kpi['sql_expression']}")
                    elif kpi.get("nl_formula"):
                        kpi_lines.append(f"  NL formula  : {kpi['nl_formula']} (not yet compiled to SQL)")
                    kpi_lines.append("")
                kpi_lines.append("=" * 60)
                schema_context = schema_context + "\n" + "\n".join(kpi_lines)
        except Exception as exc:
            logger.warning("understand_node: failed to load KPIs for source %s — %s", source_id[:8], exc)

    state["schema_context"]      = schema_context
    state["categorical_columns"] = categorical_columns
    state["column_hierarchy"]    = hierarchy or {}
    state["active_kpis"]         = active_kpis
    state["phase"]               = "understand"
    return state
