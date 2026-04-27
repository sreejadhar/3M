"""
Ontology Enricher — links business semantics (glossary terms, KPIs) to the
structural knowledge graph (entity/attribute nodes in the metadata catalog and
KG snapshots).

Two enrichment passes
---------------------
1. Attribute annotation
   - Approved glossary terms → md_attributes.semantic_role / description
   - Active KPI sql_expressions → KPI→attribute link table (kpi_attr_links)

2. KG node injection
   - GLOSSARY_CONCEPT nodes (purple) added to every KG snapshot
   - KPI_METRIC nodes (amber) added to every KG snapshot
   - GLOSSARY_MATCHES edges connecting concept nodes → matching entity nodes
   - MEASURES edges connecting metric nodes → entity nodes whose properties
     appear in the KPI's sql_expression

All link records are stored in a lightweight SQLite sidecar DB
(data/ontology_enrichment.db) so enrichment state survives restarts and is
queryable without touching metadata_catalog or kg_store schemas.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── DB path ───────────────────────────────────────────────────────────────────

def _db_path() -> str:
    return os.environ.get(
        "ONTOLOGY_ENRICHMENT_DB",
        os.path.join(os.environ.get("DATA_DIR", "data"), "ontology_enrichment.db"),
    )


@contextmanager
def _cur() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS glossary_attr_links (
    link_id     TEXT PRIMARY KEY,
    term_id     TEXT NOT NULL,
    term_name   TEXT NOT NULL,
    term_def    TEXT NOT NULL DEFAULT '',
    attr_id     TEXT NOT NULL,
    column_name TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    table_name  TEXT NOT NULL DEFAULT '',
    match_type  TEXT NOT NULL DEFAULT 'exact_name',
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL,
    UNIQUE(term_id, attr_id)
);

CREATE TABLE IF NOT EXISTS kpi_attr_links (
    link_id     TEXT PRIMARY KEY,
    kpi_id      TEXT NOT NULL,
    kpi_name    TEXT NOT NULL,
    unit        TEXT NOT NULL DEFAULT '',
    direction   TEXT NOT NULL DEFAULT 'up',
    attr_id     TEXT NOT NULL,
    column_name TEXT NOT NULL,
    entity_id   TEXT NOT NULL,
    table_name  TEXT NOT NULL DEFAULT '',
    match_type  TEXT NOT NULL DEFAULT 'column_ref',
    created_at  TEXT NOT NULL,
    UNIQUE(kpi_id, attr_id)
);

CREATE TABLE IF NOT EXISTS enrichment_runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    stats_json  TEXT NOT NULL DEFAULT '{}'
);
"""

_schema_done = False


def _ensure_schema() -> None:
    global _schema_done
    if _schema_done:
        return
    with _cur() as conn:
        for stmt in _SCHEMA.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
    _schema_done = True


# ── Text normalisation helpers ────────────────────────────────────────────────

_SQL_KEYWORDS: Set[str] = {
    "select", "from", "where", "and", "or", "not", "in", "is", "null",
    "as", "on", "join", "left", "right", "inner", "outer", "full", "cross",
    "group", "by", "order", "having", "distinct", "with", "union", "all",
    "sum", "avg", "count", "max", "min", "case", "when", "then", "else", "end",
    "nullif", "coalesce", "iif", "if", "cast", "convert", "trim", "lower",
    "upper", "substr", "length", "round", "floor", "ceil", "abs",
    "over", "partition", "lag", "lead", "row_number", "rank",
    "date", "year", "month", "day", "current", "prior", "now", "today",
    "true", "false", "between", "like", "exists", "any", "some",
}

_SUFFIX_RE = re.compile(r"(_(id|key|code|num|no|ref|fk|pk))?$", re.IGNORECASE)

# Matches "  col_name: xsd_type" lines in a KG node title (Properties: section)
_TITLE_PROP_RE = re.compile(r'^\s{2}(.+?):\s+(\S+)', re.MULTILINE)


def _norm(s: str) -> str:
    """Lowercase, strip spaces/underscores/hyphens, remove common FK suffixes."""
    s = re.sub(r"[\s_\-]+", "", s.lower())
    s = _SUFFIX_RE.sub("", s)
    return s


def _extract_sql_columns(sql: str) -> List[str]:
    """
    Extract likely column name tokens from a SQL expression.
    Strips SQL keywords, numeric literals, quoted strings, and single-char tokens.
    """
    # Remove quoted strings and numeric literals
    sql = re.sub(r"'[^']*'", " ", sql)
    sql = re.sub(r"\b\d+(\.\d+)?\b", " ", sql)
    tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", sql)
    seen: Set[str] = set()
    result = []
    for t in tokens:
        tl = t.lower()
        if tl in _SQL_KEYWORDS or len(t) <= 1:
            continue
        if tl not in seen:
            seen.add(tl)
            result.append(t)
    return result


# ── Pass 1a: Glossary → md_attributes ────────────────────────────────────────

def _enrich_from_glossary() -> Dict[str, int]:
    """
    Match approved glossary terms (and their synonyms) against metadata catalog
    attributes. On match, annotate md_attributes.semantic_role and description.
    Returns {"terms_scanned": N, "attrs_annotated": M, "links_created": K}
    """
    _ensure_schema()

    try:
        import glossary_store as _gl
    except ImportError:
        logger.warning("ontology_enricher: glossary_store unavailable — skipping glossary pass")
        return {"terms_scanned": 0, "attrs_annotated": 0, "links_created": 0}

    try:
        import metadata_catalog as _mc
    except ImportError:
        logger.warning("ontology_enricher: metadata_catalog unavailable — skipping glossary pass")
        return {"terms_scanned": 0, "attrs_annotated": 0, "links_created": 0}

    terms = _gl.list_terms(approved_only=True)
    if not terms:
        return {"terms_scanned": 0, "attrs_annotated": 0, "links_created": 0}

    # Build normalised lookup: norm_name → (term_id, term_name, definition, match_type)
    norm_to_term: Dict[str, Tuple[str, str, str, str]] = {}
    for t in terms:
        key = _norm(t["name"])
        if key:
            norm_to_term[key] = (t["term_id"], t["name"], t.get("definition", ""), "exact_name")
        for syn in t.get("synonyms") or []:
            sk = _norm(syn.get("synonym", "") or "")
            if sk and sk not in norm_to_term:
                norm_to_term[sk] = (t["term_id"], t["name"], t.get("definition", ""), "synonym")

    entities = _mc.list_entities()
    attrs_annotated = 0
    links_created = 0

    for ent in entities:
        full = _mc.get_entity(ent["metadata_id"])
        if not full:
            continue
        for attr in full.get("attributes") or []:
            col_norm = _norm(attr.get("column_name", ""))
            if not col_norm or col_norm not in norm_to_term:
                continue

            term_id, term_name, term_def, match_type = norm_to_term[col_norm]

            # Write annotation link
            with _cur() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO glossary_attr_links
                       (link_id,term_id,term_name,term_def,attr_id,column_name,
                        entity_id,table_name,match_type,confidence,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (str(uuid.uuid4()), term_id, term_name, term_def,
                     attr["attr_id"], attr["column_name"],
                     ent["metadata_id"], ent.get("table_name", ""),
                     match_type, 1.0, _now()),
                )

            # Annotate md_attributes — only fill gaps, don't overwrite existing
            updates: Dict[str, Any] = {}
            if not (attr.get("semantic_role") or "").strip():
                updates["semantic_role"] = term_name
            if not (attr.get("description") or "").strip() and term_def:
                updates["description"] = term_def
            if updates:
                _mc.update_attribute(attr["attr_id"], **updates)
                attrs_annotated += 1

            links_created += 1

    logger.info(
        "ontology_enricher: glossary pass — %d terms, %d links, %d attrs annotated",
        len(terms), links_created, attrs_annotated,
    )
    return {"terms_scanned": len(terms), "attrs_annotated": attrs_annotated,
            "links_created": links_created}


# ── Pass 1b: KPIs → md_attributes ────────────────────────────────────────────

def _enrich_from_kpis() -> Dict[str, int]:
    """
    Parse each active KPI's sql_expression to extract column references, then
    find matching attributes in the metadata catalog and record kpi_attr_links.
    Returns {"kpis_scanned": N, "links_created": M}
    """
    _ensure_schema()

    try:
        import kpi_store as _kpi
    except ImportError:
        logger.warning("ontology_enricher: kpi_store unavailable — skipping KPI pass")
        return {"kpis_scanned": 0, "links_created": 0}

    try:
        import metadata_catalog as _mc
    except ImportError:
        logger.warning("ontology_enricher: metadata_catalog unavailable — skipping KPI pass")
        return {"kpis_scanned": 0, "links_created": 0}

    kpis = _kpi.list_kpis(status="active")
    if not kpis:
        return {"kpis_scanned": 0, "links_created": 0}

    # Build flat attribute lookup: norm_col → [(attr_id, column_name, entity_id, table_name)]
    attr_lookup: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for ent in _mc.list_entities():
        full = _mc.get_entity(ent["metadata_id"])
        if not full:
            continue
        for attr in full.get("attributes") or []:
            col_norm = _norm(attr.get("column_name", ""))
            if col_norm:
                attr_lookup.setdefault(col_norm, []).append((
                    attr["attr_id"], attr["column_name"],
                    ent["metadata_id"], ent.get("table_name", ""),
                ))

    links_created = 0

    for kpi in kpis:
        sql = (kpi.get("sql_expression") or "").strip()
        if not sql:
            continue

        col_tokens = _extract_sql_columns(sql)
        matched_attrs: Set[str] = set()

        for token in col_tokens:
            tn = _norm(token)
            for attr_id, col_name, entity_id, table_name in attr_lookup.get(tn, []):
                if attr_id in matched_attrs:
                    continue
                matched_attrs.add(attr_id)
                with _cur() as conn:
                    conn.execute(
                        """INSERT OR IGNORE INTO kpi_attr_links
                           (link_id,kpi_id,kpi_name,unit,direction,
                            attr_id,column_name,entity_id,table_name,match_type,created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (str(uuid.uuid4()), kpi["kpi_id"], kpi["name"],
                         kpi.get("unit", ""), kpi.get("direction", "up"),
                         attr_id, col_name, entity_id, table_name,
                         "column_ref", _now()),
                    )
                    links_created += 1

    logger.info("ontology_enricher: KPI pass — %d KPIs, %d links", len(kpis), links_created)
    return {"kpis_scanned": len(kpis), "links_created": links_created}


# ── Pass 2: Inject enrichment nodes into KG snapshots ────────────────────────

def _inject_kg_nodes() -> Dict[str, int]:
    """
    For every source in kg_store:
    - Strip previously injected GLOSSARY_CONCEPT / KPI_METRIC nodes+edges
    - Add fresh GLOSSARY_CONCEPT nodes for all approved glossary terms
    - Add fresh KPI_METRIC nodes for all active KPIs
    - Add GLOSSARY_MATCHES edges where term name/synonym matches a KG entity property
    - Add MEASURES edges where KPI sql columns match KG entity properties
    - Save updated snapshot via kg_store.save()
    """
    try:
        import glossary_store as _gl
    except ImportError:
        _gl = None  # type: ignore

    try:
        import kpi_store as _kpi
    except ImportError:
        _kpi = None  # type: ignore

    try:
        import kg_store as _kg
    except ImportError:
        logger.warning("ontology_enricher: kg_store unavailable — skipping KG injection")
        return {"sources_updated": 0, "nodes_added": 0, "edges_added": 0}

    terms = _gl.list_terms(approved_only=True) if _gl else []
    kpis  = _kpi.list_kpis(status="active")   if _kpi else []

    # Build term lookup structures for matching
    term_norms: List[Tuple[str, Set[str], Dict]] = []  # (term_id, {norm_variants}, term)
    for t in terms:
        variants = {_norm(t["name"])}
        for syn in t.get("synonyms") or []:
            sv = _norm(syn.get("synonym", "") or "")
            if sv:
                variants.add(sv)
        variants.discard("")
        if variants:
            term_norms.append((t["term_id"], variants, t))

    # KPI column sets for matching
    kpi_cols: List[Tuple[str, Set[str], Dict]] = []  # (kpi_id, {norm_col_tokens}, kpi)
    for k in kpis:
        sql = k.get("sql_expression", "")
        cols = {_norm(c) for c in _extract_sql_columns(sql)} - {""}
        if cols:
            kpi_cols.append((k["kpi_id"], cols, k))

    sources = _kg.load_all()
    sources_updated = total_nodes = total_edges = 0

    for src in sources:
        existing_nodes = [
            n for n in (src.get("kg_nodes") or [])
            if not str(n.get("id", "")).startswith(("glossary:", "kpi:"))
        ]
        existing_edges = [
            e for e in (src.get("kg_edges") or [])
            if e.get("label") not in {"GLOSSARY_MATCHES", "MEASURES"}
        ]

        # Build normalised property index for this source's entity nodes.
        # Strategy 1: structured properties list [{name, type}, ...]
        # Strategy 2: parse column names from the title text (legacy nodes built
        #   before the structured properties field was added — format is
        #   "Properties:\n  col_name: type\n  ...")
        prop_to_nodes: Dict[str, List[str]] = {}
        for node in existing_nodes:
            node_id = node.get("id", "")
            if not node_id:
                continue
            props = node.get("properties") or []
            if props:
                for prop in props:
                    pn = _norm(prop.get("name", "") or prop.get("column", ""))
                    if pn:
                        prop_to_nodes.setdefault(pn, []).append(node_id)
            else:
                title = node.get("title") or ""
                pi = title.find("Properties:")
                if pi != -1:
                    for m in _TITLE_PROP_RE.finditer(title[pi + len("Properties:"):]):
                        pn = _norm(m.group(1).strip())
                        if pn:
                            prop_to_nodes.setdefault(pn, []).append(node_id)

        new_nodes: List[Dict] = []
        new_edges: List[Dict] = []

        # ── Glossary concept nodes ────────────────────────────────────────────
        for term_id, variants, t in term_norms:
            node_id = f"glossary:{term_id}"
            synonyms_str = ", ".join(
                s.get("synonym", "") for s in (t.get("synonyms") or []) if s.get("synonym")
            )
            title_parts = [f"GLOSSARY CONCEPT: {t['name']}"]
            if t.get("domain"):
                title_parts.append(f"Domain: {t['domain']}")
            if t.get("definition"):
                title_parts.append(t["definition"])
            if synonyms_str:
                title_parts.append(f"Synonyms: {synonyms_str}")
            if t.get("formula"):
                title_parts.append(f"Formula: {t['formula']}")

            new_nodes.append({
                "id":    node_id,
                "label": t["name"],
                "title": "\n".join(title_parts),
                "color": "#a78bfa",   # purple — business concept
                "size":  16,
                "properties": [
                    {"name": "definition", "type": "text"},
                    {"name": "domain",     "type": "text"},
                ],
            })

            # Edges to every entity node that has a matching property
            matched_entity_nodes: Set[str] = set()
            for variant in variants:
                for target_node_id in prop_to_nodes.get(variant, []):
                    matched_entity_nodes.add(target_node_id)

            for target_id in matched_entity_nodes:
                new_edges.append({
                    "from":  node_id,
                    "to":    target_id,
                    "label": "GLOSSARY_MATCHES",
                    "title": f"{t['name']} → {target_id.split('/')[-1]}",
                    "join_columns": [],
                })

        # ── KPI metric nodes ──────────────────────────────────────────────────
        for kpi_id, col_norms, k in kpi_cols:
            node_id = f"kpi:{kpi_id}"
            direction_arrow = "↑" if k.get("direction") == "up" else "↓"
            title_parts = [
                f"KPI METRIC: {k['name']}",
                f"Unit: {k.get('unit','—')}  Direction: {direction_arrow}",
            ]
            if k.get("nl_formula"):
                title_parts.append(f"Formula: {k['nl_formula']}")
            if k.get("sql_expression"):
                title_parts.append(f"SQL: {k['sql_expression'][:160]}")

            new_nodes.append({
                "id":    node_id,
                "label": k["name"],
                "title": "\n".join(title_parts),
                "color": "#10b981",   # green — business metric
                "size":  16,
                "properties": [
                    {"name": "unit",           "type": "text"},
                    {"name": "direction",      "type": "text"},
                    {"name": "sql_expression", "type": "text"},
                ],
            })

            # Edges to entity nodes that contain any of the referenced columns
            matched_entity_nodes_kpi: Set[str] = set()
            for col_norm in col_norms:
                for target_node_id in prop_to_nodes.get(col_norm, []):
                    matched_entity_nodes_kpi.add(target_node_id)

            for target_id in matched_entity_nodes_kpi:
                new_edges.append({
                    "from":  node_id,
                    "to":    target_id,
                    "label": "MEASURES",
                    "title": f"{k['name']} measures {target_id.split('/')[-1]}",
                    "join_columns": [],
                })

        if not new_nodes and not new_edges:
            continue

        merged_nodes = existing_nodes + new_nodes
        merged_edges = existing_edges + new_edges

        updated_src = {**src, "kg_nodes": merged_nodes, "kg_edges": merged_edges}
        _kg.save(updated_src)

        sources_updated += 1
        total_nodes += len(new_nodes)
        total_edges += len(new_edges)

    logger.info(
        "ontology_enricher: KG injection — %d sources, %d nodes, %d edges",
        sources_updated, total_nodes, total_edges,
    )
    return {"sources_updated": sources_updated, "nodes_added": total_nodes,
            "edges_added": total_edges}


# ── Public entry points ───────────────────────────────────────────────────────

def run_enrichment() -> Dict[str, Any]:
    """
    Run all three passes and record the result in enrichment_runs.
    Returns a stats dict suitable for returning from an API endpoint.
    """
    _ensure_schema()
    run_id = str(uuid.uuid4())
    started = _now()

    with _cur() as conn:
        conn.execute(
            "INSERT INTO enrichment_runs (run_id,started_at,status,stats_json) VALUES (?,?,?,?)",
            (run_id, started, "running", "{}"),
        )

    try:
        g = _enrich_from_glossary()
        k = _enrich_from_kpis()
        kg = _inject_kg_nodes()

        stats = {
            "glossary": g,
            "kpis":     k,
            "kg":       kg,
        }
        with _cur() as conn:
            conn.execute(
                "UPDATE enrichment_runs SET finished_at=?,status=?,stats_json=? WHERE run_id=?",
                (_now(), "done", json.dumps(stats), run_id),
            )
        return {"run_id": run_id, "status": "done", "stats": stats}

    except Exception as exc:
        logger.exception("ontology_enricher: run_enrichment failed")
        with _cur() as conn:
            conn.execute(
                "UPDATE enrichment_runs SET finished_at=?,status=?,stats_json=? WHERE run_id=?",
                (_now(), "error", json.dumps({"error": str(exc)}), run_id),
            )
        raise


def get_status() -> Optional[Dict[str, Any]]:
    """Return the most recent enrichment run record, or None if never run."""
    _ensure_schema()
    with _cur() as conn:
        row = conn.execute(
            "SELECT * FROM enrichment_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["stats"] = json.loads(d.pop("stats_json", "{}") or "{}")
    except Exception:
        d["stats"] = {}
    return d


def get_glossary_attr_links(term_id: Optional[str] = None) -> List[Dict]:
    """Return glossary→attribute links, optionally filtered by term."""
    _ensure_schema()
    with _cur() as conn:
        if term_id:
            rows = conn.execute(
                "SELECT * FROM glossary_attr_links WHERE term_id=? ORDER BY created_at",
                (term_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM glossary_attr_links ORDER BY term_name, table_name"
            ).fetchall()
    return [dict(r) for r in rows]


def get_kpi_attr_links(kpi_id: Optional[str] = None) -> List[Dict]:
    """Return KPI→attribute links, optionally filtered by KPI."""
    _ensure_schema()
    with _cur() as conn:
        if kpi_id:
            rows = conn.execute(
                "SELECT * FROM kpi_attr_links WHERE kpi_id=? ORDER BY created_at",
                (kpi_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM kpi_attr_links ORDER BY kpi_name, table_name"
            ).fetchall()
    return [dict(r) for r in rows]
