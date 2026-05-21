"""
Ontology pipeline node: build an rdflib OWL graph from the metadata report.

Mapping strategy
----------------
Table           -> owl:Class
  PK column     -> owl:DatatypeProperty + owl:FunctionalProperty + owl:InverseFunctionalProperty
  NOT NULL col  -> owl:DatatypeProperty + rdfs:subClassOf owl:Restriction (minCardinality 1)
  nullable col  -> owl:DatatypeProperty
FK candidate    -> owl:ObjectProperty (domain = left table, range = right table)
Explicit FK     -> owl:ObjectProperty
Cardinality 1:1 -> owl:FunctionalProperty + owl:InverseFunctionalProperty on the object property
Cardinality 1:N -> owl:FunctionalProperty on the object property
FD              -> rdfs:comment annotation on the owning class
Statistics      -> rdfs:comment annotations on datatype properties (if include_statistics=True)

Descriptions (NEW)
------------------
Every class and property now carries one or more structured rdfs:comment annotations:
  - owl:Class      : table description (entity type, row count, dominant domains)
  - DatatypeProperty : column description (role, semantic domain, value patterns, cardinality, range)
  - ObjectProperty   : relationship description (ind_type / cardinality, coverage)
  - FD annotation    : fd_type, plain-English description, confidence
All descriptions are derived from metadata facts — no LLM, no hallucination.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from rdflib import BNode, Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD

from ..state import OntologyState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM-based concept annotation (one batched call per table at index time)
# ---------------------------------------------------------------------------

_CONCEPT_SYSTEM_TEMPLATE = """\
You are a database metadata expert specialising in {domain_context}.
Given a table name, column names, SQL data types, and OBSERVED DATA EVIDENCE
(sample values, numeric range, cardinality, rule-based description and domain),
identify the standard business concept each column represents.

SOURCE CONTEXT
==============
{source_context_block}

You MUST ground every label in the observed data evidence AND the source context
above.  Do NOT guess from column names alone — abbreviations like arpu, gmv,
nii, tts, nrv, gsv, fte, oee, cac, otif are common across industries but mean
different things in different domains.  Use the source context to disambiguate.

Return ONLY a JSON object mapping each column name to a short concept label
(kebab-case, lowercase), or null when:
  - The column is an identifier, primary/foreign key, or date/timestamp.
  - The column name is already self-explanatory (e.g. "revenue", "headcount").
  - The evidence contradicts the apparent meaning (e.g. a "value" column with
    only 0/1 values is a status flag, not a monetary measure).

GROUNDING RULES — read evidence in this priority order:
  1. top_values  — strongest signal: numeric → measure; strings → dimension;
                   0/1 → flag; currency-scale integers → monetary.
  2. description — already rule-grounded by the extraction pipeline; trust it.
  3. min/max/avg — range confirms type: large decimals → monetary;
                   0–100 range → percentage; small integers → count or index.
  4. cardinality — near 1.0 (high) → continuous measure or identifier;
                   few distinct values → categorical dimension.
  5. data_type   — decimal/float → measure; integer → count/id; varchar → text/category.
  6. column name — last resort only; resolve against source domain context.

DOMAIN-AWARE LABEL CONVENTIONS
Use the exact terminology standard for the source domain:
  CPG / RGM    : gross-rsv, net-realized-value, trade-spend, pricing-impact,
                 price-index, market-share, volume-offtake
  Telecom      : avg-revenue-per-user, minutes-of-use, data-usage, churn-rate,
                 net-promoter-score, revenue-per-gb
  Banking / FS : net-interest-income, net-interest-margin, gross-npa,
                 casa-ratio, return-on-assets, cost-to-income-ratio
  E-commerce   : gross-merchandise-value, avg-order-value, customer-acquisition-cost,
                 lifetime-value, return-on-ad-spend, conversion-rate
  Manufacturing: overall-equipment-effectiveness, first-pass-yield, scrap-rate,
                 mean-time-between-failures, cycle-time, capacity-utilisation
  Healthcare   : length-of-stay, bed-occupancy, claims-paid, denial-rate,
                 cost-per-patient, readmission-rate
  HR / People  : headcount, attrition-rate, time-to-hire, cost-per-hire,
                 offer-acceptance-rate, engagement-score
  Supply Chain : fill-rate, on-time-in-full, inventory-days, lead-time,
                 perfect-order-rate, supplier-on-time-delivery

Keep labels concise: 1–4 words, kebab-case, no spaces.
Return ONLY valid JSON — no prose, no markdown fences.

EXAMPLE
-------
Source context: Telecom operator | postgresql | Subscriber analytics system
Table: fact_subscriber_kpis
Columns:
  arpu    | decimal | min=45.2 max=312.0 avg=118.5 | top_values=[120.5,98.3,145.2] | high-cardinality | domain=numeric_measure
  mou     | integer | min=0 max=1200 avg=342 | top_values=[250,480,120] | high-cardinality | domain=count/volume
  subtype | varchar | top_values=['Prepaid','Postpaid','Enterprise'] | 3-distinct-values | domain=status_flag
  churn   | integer | top_values=[0,1] | 2-distinct-values | domain=status_flag
  rev     | decimal | min=120.0 max=98000.0 avg=4200.0 | high-cardinality | description="decimal monetary column" | domain=monetary

Output:
  {{"arpu": "avg-revenue-per-user", "mou": "minutes-of-use", "subtype": null, "churn": null, "rev": "revenue"}}
"""


def _build_col_evidence(col: Dict) -> str:
    """
    Build a compact evidence string for one column to include in the annotation
    prompt.  Covers all signals that help the LLM ground its concept label in
    observed data rather than guessing from the name alone.
    """
    parts: List[str] = []

    dtype = col.get("data_type", "unknown")
    parts.append(dtype)

    # Numeric range — confirms monetary, percentage, count, index
    mn, mx, avg = col.get("min_value"), col.get("max_value"), col.get("avg_value")
    if mn is not None and mx is not None:
        avg_str = f" avg={avg:.2f}" if avg is not None else ""
        parts.append(f"min={mn} max={mx}{avg_str}")

    # Top values — the strongest grounding signal
    top = col.get("top_values") or []
    if top:
        sample = top[:6]
        parts.append(f"top_values={sample}")

    # Cardinality signal
    unique = col.get("unique_count")
    rows   = col.get("row_count") or col.get("unique_count")
    if unique is not None and rows:
        ratio = unique / rows
        if ratio >= 0.95:
            parts.append("high-cardinality")
        elif unique <= 2:
            parts.append(f"{unique}-distinct-values")
        elif unique <= 20:
            parts.append(f"{unique}-distinct-values")

    # Null rate
    null_rate = col.get("null_rate")
    if null_rate and null_rate > 0.01:
        parts.append(f"null_rate={null_rate:.0%}")

    # Rule-based description from extraction_node — already grounded in metadata.
    # Trim to first clause (split on ' — ' or first parenthetical) to avoid
    # truncating mid-word on decimal points (e.g. "range: 0.0 – 500000.0").
    desc = col.get("description", "")
    if desc:
        # Prefer splitting at ' — ' (used by extraction_node as clause separator)
        # rather than '.' which appears in numeric range values.
        short = desc.split(" — ")[0].strip()
        if len(short) < 20:          # too short, take more
            short = desc.split("(")[0].strip() or desc[:120]
        short = short[:120]          # hard cap to keep prompt size predictable
        if short:
            parts.append(f'description="{short}"')

    # Rule-based domain from extraction_node
    domain = col.get("domain", "")
    if domain and domain != "unknown":
        parts.append(f"domain={domain}")

    return " | ".join(parts)


def _annotate_column_concepts(
    table_name: str,
    columns: List[Dict],
    model: str,
    domain_hint: str = "",
) -> Dict[str, str]:
    """
    Call an LLM to map each column in `columns` to a standard business concept,
    grounded in observed data evidence (sample values, numeric range, cardinality,
    rule-based description and domain already computed by extraction_node).

    Returns {column_name: concept_label} for columns that received a non-null
    label.  Fails gracefully — returns empty dict on any error.
    """
    if not columns:
        return {}

    try:
        from llm_client import get_client
        client = get_client()
    except Exception:
        return {}

    # Render the system prompt with domain context baked in so the LLM
    # resolves abbreviations against the correct industry vocabulary.
    if domain_hint:
        domain_context    = f"the {domain_hint} domain"
        source_ctx_block  = f"Domain / industry : {domain_hint}"
    else:
        domain_context    = "enterprise data systems across multiple industries"
        source_ctx_block  = "(No domain context provided — use evidence signals only)"

    system = _CONCEPT_SYSTEM_TEMPLATE.format(
        domain_context       = domain_context,
        source_context_block = source_ctx_block,
    )

    # Build one evidence line per column
    col_lines = []
    for c in columns:
        name     = c.get("name", "")
        evidence = _build_col_evidence(c)
        col_lines.append(f"  {name} | {evidence}")

    user_msg = (
        f"Table: {table_name}\n"
        f"Columns (name | data_type | range | top_values | cardinality | description | domain):\n"
        + "\n".join(col_lines)
        + "\n\nReturn a JSON object mapping each column name to its concept label "
          "(grounded in both the source context and the evidence above), "
          "or null if no concept label is needed."
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text if resp.content else "{}"
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
        result = json.loads(raw)
        # Filter nulls and non-string values; lowercase keys for lookup
        return {
            str(k).lower(): str(v).lower().strip()
            for k, v in result.items()
            if v and isinstance(v, str) and v.strip()
        }
    except Exception as exc:
        logger.warning(
            "build_node: concept annotation failed for table %s: %s",
            table_name, exc,
        )
        return {}

# ---------------------------------------------------------------------------
# XSD type mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: Dict[str, URIRef] = {
    "varchar":            XSD.string,
    "character varying":  XSD.string,
    "char":               XSD.string,
    "text":               XSD.string,
    "string":             XSD.string,
    "nvarchar":           XSD.string,
    "nchar":              XSD.string,
    "clob":               XSD.string,
    "json":               XSD.string,
    "jsonb":              XSD.string,
    "uuid":               XSD.string,
    "int":                XSD.integer,
    "integer":            XSD.integer,
    "int4":               XSD.integer,
    "int8":               XSD.integer,
    "bigint":             XSD.integer,
    "smallint":           XSD.integer,
    "tinyint":            XSD.integer,
    "serial":             XSD.integer,
    "bigserial":          XSD.integer,
    "number":             XSD.decimal,
    "numeric":            XSD.decimal,
    "decimal":            XSD.decimal,
    "float":              XSD.double,
    "float4":             XSD.double,
    "float8":             XSD.double,
    "double":             XSD.double,
    "double precision":   XSD.double,
    "real":               XSD.float,
    "boolean":            XSD.boolean,
    "bool":               XSD.boolean,
    "date":               XSD.date,
    "timestamp":          XSD.dateTime,
    "datetime":           XSD.dateTime,
    "timestamp with time zone":    XSD.dateTime,
    "timestamp without time zone": XSD.dateTime,
    "time":               XSD.time,
    "interval":           XSD.duration,
    "bytea":              XSD.hexBinary,
    "blob":               XSD.hexBinary,
}


def _xsd_type(db_type: str) -> URIRef:
    """Map a DB data type string to the nearest XSD type."""
    dt = db_type.lower().split("(")[0].strip()
    return _TYPE_MAP.get(dt, XSD.string)


def _safe(name: str) -> str:
    """Convert a table/column name to a safe URI fragment."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if s and s[0].isdigit():
        s = "_" + s
    return s or "unknown"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def build_node(state: OntologyState) -> OntologyState:
    config = state["config"]
    report = state["report"]

    base_uri = config.base_uri.rstrip("/") + "/" + _safe(config.ontology_name) + "/"
    ns       = Namespace(base_uri)
    onto_uri = URIRef(base_uri.rstrip("/"))

    g = Graph()
    g.bind("",    ns)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # Ontology header
    g.add((onto_uri, RDF.type,    OWL.Ontology))
    g.add((onto_uri, RDFS.label,  Literal(config.ontology_name)))
    g.add((onto_uri, RDFS.comment, Literal(
        f"Generated from database: {report.get('schema','unknown')} "
        f"({report.get('database_type','unknown')})"
    )))

    tables:      Dict = report.get("tables") or {}
    class_map:   Dict = {}
    property_map: Dict = {}
    added_obj_props: Set[URIRef] = set()

    # ------------------------------------------------------------------
    # 1. OWL Classes from tables + DatatypeProperties from columns
    # ------------------------------------------------------------------
    for table_name, table_meta in tables.items():
        cls_uri = ns[_safe(table_name)]
        g.add((cls_uri, RDF.type,    OWL.Class))
        g.add((cls_uri, RDFS.label,  Literal(table_name)))

        if isinstance(table_meta, dict):
            # ── Rich class description ──────────────────────────────────
            desc_parts = []

            # Primary description from extraction_node (entity type, row count, domain summary)
            table_desc = table_meta.get("description")
            if table_desc:
                desc_parts.append(table_desc)

            # Table comment from the database itself (if any)
            db_comment = table_meta.get("table_comment")
            if db_comment:
                desc_parts.append(f"Database comment: {db_comment}")

            # Row count
            row_count = table_meta.get("row_count")
            if row_count is not None:
                desc_parts.append(f"Approximate row count: {row_count:,}")

            # Size
            size_bytes = table_meta.get("size_bytes")
            if size_bytes:
                desc_parts.append(f"Storage size: {size_bytes:,} bytes")

            # Partitioning
            partitioned_by = table_meta.get("partitioned_by") or []
            if partitioned_by:
                desc_parts.append(f"Partitioned by: {', '.join(partitioned_by)}")

            if desc_parts:
                g.add((cls_uri, RDFS.comment, Literal("\n".join(desc_parts))))

        class_map[table_name] = cls_uri
        logger.debug("  Class: %s", table_name)

        if not isinstance(table_meta, dict):
            continue

        # ── LLM concept annotation (one batched call per table) ──────────
        # Maps opaque column names (arpu, gsv, tts, nii, gmv…) to standard
        # business concept labels regardless of domain.  Skipped when
        # annotate_concepts=False or when ANTHROPIC_API_KEY is not set.
        col_concepts: Dict[str, str] = {}
        if getattr(config, "annotate_concepts", True) and os.environ.get("ANTHROPIC_API_KEY"):
            cols_for_annotation = table_meta.get("columns") or []
            col_concepts = _annotate_column_concepts(
                table_name,
                [c for c in cols_for_annotation if isinstance(c, dict)],
                model=getattr(config, "llm_model", "claude-haiku-4-5"),
                domain_hint=getattr(config, "source_domain", ""),
            )
            if col_concepts:
                logger.debug(
                    "build_node: concept annotations for %s: %s",
                    table_name, col_concepts,
                )

        for col in table_meta.get("columns") or []:
            if not isinstance(col, dict):
                continue

            col_name  = col.get("name", "")
            prop_uri  = ns[f"{_safe(table_name)}_{_safe(col_name)}"]
            xsd_range = _xsd_type(col.get("data_type", ""))

            g.add((prop_uri, RDF.type,    OWL.DatatypeProperty))
            g.add((prop_uri, RDFS.label,  Literal(col_name)))
            g.add((prop_uri, RDFS.domain, cls_uri))
            g.add((prop_uri, RDFS.range,  xsd_range))

            is_pk = col.get("is_primary_key", False)
            if is_pk:
                g.add((prop_uri, RDF.type, OWL.FunctionalProperty))
                g.add((prop_uri, RDF.type, OWL.InverseFunctionalProperty))

            if not col.get("nullable", True):
                restriction = BNode()
                g.add((restriction, RDF.type,           OWL.Restriction))
                g.add((restriction, OWL.onProperty,     prop_uri))
                g.add((restriction, OWL.minCardinality,
                       Literal(1, datatype=XSD.nonNegativeInteger)))
                g.add((cls_uri, RDFS.subClassOf, restriction))

            # ── Rich property description ───────────────────────────────
            prop_desc_parts = []

            # Plain-English description from extraction_node
            col_desc = col.get("description")
            if col_desc:
                prop_desc_parts.append(col_desc)

            # Business concept label (from LLM annotation or rule-based fallback)
            # Stored as "Business concept: <label>" so understand_node can read it.
            concept_label = col_concepts.get(col_name.lower(), "")
            if concept_label:
                prop_desc_parts.append(f"Business concept: {concept_label}")

            # Semantic domain label
            domain = col.get("domain")
            if domain and domain != "unknown":
                prop_desc_parts.append(f"Semantic domain: {domain.replace('_', ' ')}")

            # Value pattern hints
            pattern_hints = col.get("pattern_hints") or []
            if pattern_hints:
                prop_desc_parts.append(f"Value patterns detected: {', '.join(pattern_hints)}")

            # FK reference
            fk_ref = col.get("fk_references")
            if col.get("is_foreign_key") and fk_ref:
                prop_desc_parts.append(f"References: {fk_ref}")

            # Statistics (only if include_statistics is enabled)
            if config.include_statistics:
                stat_parts = []
                for key, label in (
                    ("unique_count", "unique values"),
                    ("null_count",   "null count"),
                    ("min_value",    "min"),
                    ("max_value",    "max"),
                    ("avg_value",    "avg"),
                ):
                    val = col.get(key)
                    if val is not None:
                        stat_parts.append(f"{label}={val}")
                null_rate = col.get("null_rate")
                if null_rate is not None:
                    stat_parts.append(f"null_rate={null_rate:.1%}")
                if stat_parts:
                    prop_desc_parts.append(f"Statistics: {', '.join(stat_parts)}")

                # Top values for categorical/low-cardinality columns
                top_values = col.get("top_values") or []
                unique_count = col.get("unique_count") or 0
                if top_values and unique_count <= 20:
                    vals = ", ".join(f"'{v}'" for v in top_values[:10] if v is not None)
                    prop_desc_parts.append(f"Sample values: {vals}")

            if prop_desc_parts:
                g.add((prop_uri, RDFS.comment, Literal("\n".join(prop_desc_parts))))

            property_map[(table_name, col_name)] = prop_uri

    # ------------------------------------------------------------------
    # 2. ObjectProperties from explicit FK definitions
    # ------------------------------------------------------------------
    for table_name, table_meta in tables.items():
        if not isinstance(table_meta, dict):
            continue
        for fk in table_meta.get("foreign_keys") or []:
            if not isinstance(fk, dict):
                continue
            ref_table = (
                fk.get("ref_table")
                or fk.get("referenced_table")
                or fk.get("foreign_table")
                or ""
            )
            if not ref_table or ref_table not in class_map:
                continue
            fk_col     = fk.get("column", "")
            ref_col    = fk.get("referenced_column", "")
            constraint = fk.get("constraint_name", "")
            prop_uri   = ns[f"{_safe(table_name)}_fk_{_safe(ref_table)}"]
            if prop_uri in added_obj_props:
                continue
            g.add((prop_uri, RDF.type,    OWL.ObjectProperty))
            g.add((prop_uri, RDFS.label,  Literal(f"{table_name} → {ref_table} (explicit FK)")))
            g.add((prop_uri, RDFS.domain, class_map[table_name]))
            g.add((prop_uri, RDFS.range,  class_map[ref_table]))
            # Description
            fk_desc_parts = [
                f"Explicit foreign key: '{table_name}'.{fk_col} → '{ref_table}'.{ref_col}",
                "Relationship type: explicit referential integrity constraint (DDL-defined)",
            ]
            if constraint:
                fk_desc_parts.append(f"Constraint name: {constraint}")
            g.add((prop_uri, RDFS.comment, Literal("\n".join(fk_desc_parts))))
            added_obj_props.add(prop_uri)

    # ------------------------------------------------------------------
    # 3. ObjectProperties from FK candidates (inclusion dependencies)
    # ------------------------------------------------------------------
    for fk in report.get("fk_candidates") or []:
        left_t  = fk.get("left_table",  "")
        right_t = fk.get("right_table", "")
        if left_t not in class_map or right_t not in class_map:
            continue
        prop_uri = ns[f"{_safe(left_t)}_references_{_safe(right_t)}"]
        if prop_uri in added_obj_props:
            continue
        g.add((prop_uri, RDF.type,    OWL.ObjectProperty))

        ind_type = fk.get("ind_type", "value_subset")
        left_cols  = fk.get("left_columns", [])
        right_cols = fk.get("right_columns", [])
        coverage   = fk.get("coverage", 0)

        label = f"{left_t} references {right_t}"
        if ind_type == "exact_foreign_key":
            label = f"{left_t} → {right_t} (inferred FK)"
        elif ind_type == "strong_fk_candidate":
            label = f"{left_t} → {right_t} (strong FK candidate)"
        g.add((prop_uri, RDFS.label,  Literal(label)))
        g.add((prop_uri, RDFS.domain, class_map[left_t]))
        g.add((prop_uri, RDFS.range,  class_map[right_t]))

        # Rich description
        fk_cand_desc_parts = []
        ind_description = fk.get("description")
        if ind_description:
            fk_cand_desc_parts.append(ind_description)
        fk_cand_desc_parts.append(
            f"Relationship type: {ind_type.replace('_', ' ')}"
        )
        fk_cand_desc_parts.append(
            f"Join columns: {', '.join(left_cols)} → {', '.join(right_cols)}"
        )
        fk_cand_desc_parts.append(f"Coverage: {coverage:.1%} of left-table values matched")
        if coverage >= 0.99:
            fk_cand_desc_parts.append(
                "Confidence: high — suitable for use as a JOIN key"
            )
        elif coverage >= 0.95:
            fk_cand_desc_parts.append(
                "Confidence: moderate — orphan records exist; verify before using as FK"
            )
        g.add((prop_uri, RDFS.comment, Literal("\n".join(fk_cand_desc_parts))))
        added_obj_props.add(prop_uri)

    # ------------------------------------------------------------------
    # 4. Cardinality annotations — enrich existing FK edges; only create a
    #    new edge when no FK-derived edge exists for this table pair.
    #
    #    This prevents duplicate edges (one from IND→FK, one from cardinality)
    #    which caused the cluttered bidirectional ↔ labels in the graph UI.
    #
    #    Fact-table detection: match the same naming conventions used by
    #    extraction_node so that fact↔fact cardinality relationships (which
    #    share ID columns but have no meaningful FK relationship) are not
    #    rendered as graph edges.
    # ------------------------------------------------------------------
    _FACT_NAME_RE = re.compile(r'\b(fact|fct|measure|metric|kpi)\b', re.IGNORECASE)

    def _is_fact_table(name: str) -> bool:
        """Return True if the table is a fact/measure table."""
        if _FACT_NAME_RE.search(name):
            return True
        # Also check the stored description in case the table was labelled by extraction_node
        desc = (tables.get(name) or {}).get("description") or ""
        return "fact/measure table" in desc

    for cr in report.get("cardinality_relationships") or []:
        left_t    = cr.get("left_table",  "")
        right_t   = cr.get("right_table", "")
        rel_type  = cr.get("type", "M:N")
        join_cols = cr.get("join_columns", [])
        left_uniq = cr.get("left_unique_values", 0)
        right_uniq= cr.get("right_unique_values", 0)
        if left_t not in class_map or right_t not in class_map:
            continue

        # Prefer annotating the existing IND-derived or explicit FK edge for
        # this pair rather than creating a separate cardinality edge.
        fk_uri    = ns[f"{_safe(left_t)}_fk_{_safe(right_t)}"]
        ref_uri   = ns[f"{_safe(left_t)}_references_{_safe(right_t)}"]
        rel_uri   = ns[f"{_safe(left_t)}_relates_{_safe(right_t)}"]

        if fk_uri in added_obj_props:
            prop_uri = fk_uri
        elif ref_uri in added_obj_props:
            prop_uri = ref_uri
        else:
            # No FK edge exists — skip creating a spurious edge between two
            # fact tables (they share ID columns but are not meaningfully joined).
            if _is_fact_table(left_t) and _is_fact_table(right_t):
                logger.debug(
                    "Skipping fact↔fact cardinality edge: %s ↔ %s", left_t, right_t
                )
                continue
            # Create a new directed edge (not bidirectional)
            prop_uri = rel_uri
            if prop_uri not in added_obj_props:
                g.add((prop_uri, RDF.type,    OWL.ObjectProperty))
                # Clean directed label — cardinality goes in comments/OWL only
                g.add((prop_uri, RDFS.label,  Literal(f"{left_t} → {right_t}")))
                g.add((prop_uri, RDFS.domain, class_map[left_t]))
                g.add((prop_uri, RDFS.range,  class_map[right_t]))
                added_obj_props.add(prop_uri)

        # OWL cardinality semantics on whichever edge we resolved
        if rel_type == "1:1":
            g.add((prop_uri, RDF.type, OWL.FunctionalProperty))
            g.add((prop_uri, RDF.type, OWL.InverseFunctionalProperty))
        elif rel_type in ("1:N", "N:1"):
            g.add((prop_uri, RDF.type, OWL.FunctionalProperty))

        # Rich cardinality description (appended as additional rdfs:comment)
        card_desc_parts = [
            f"Cardinality: {rel_type} relationship between '{left_t}' and '{right_t}'",
        ]
        if join_cols:
            card_desc_parts.append(f"Determined via join columns: {', '.join(join_cols)}")
        if left_uniq and right_uniq:
            card_desc_parts.append(
                f"Distinct values: {left_uniq:,} in '{left_t}', {right_uniq:,} in '{right_t}'"
            )
        if rel_type == "1:1":
            card_desc_parts.append(
                "Each instance of the left class corresponds to exactly one instance "
                "of the right class, and vice versa."
            )
        elif rel_type == "1:N":
            card_desc_parts.append(
                f"One instance of '{left_t}' corresponds to many instances of '{right_t}'."
            )
        elif rel_type == "N:1":
            card_desc_parts.append(
                f"Many instances of '{left_t}' correspond to one instance of '{right_t}'."
            )
        else:
            card_desc_parts.append(
                f"Many-to-many relationship between '{left_t}' and '{right_t}'."
            )
        g.add((prop_uri, RDFS.comment, Literal("\n".join(card_desc_parts))))

    # ------------------------------------------------------------------
    # 5. Annotate classes with FD summaries (rich descriptions)
    # ------------------------------------------------------------------
    for fd in report.get("functional_dependencies") or []:
        tbl  = fd.get("table", "")
        if tbl not in class_map:
            continue
        det     = ", ".join(fd.get("determinant") or [])
        dep     = ", ".join(fd.get("dependent")   or [])
        conf    = fd.get("confidence", 0)
        fd_type = fd.get("fd_type", "non_key")
        fd_desc = fd.get("description")

        # Build a structured FD annotation.
        # The sentinel prefix "FD-ANNOTATION:" is used by understand_node to
        # strip these comments from the SQL planning schema context, preventing
        # the LLM from treating determinant/dependent column names as real columns.
        fd_parts = [f"FD-ANNOTATION:"]
        if fd_desc:
            fd_parts.append(fd_desc)
        fd_parts.append(
            f"FD type: {fd_type.replace('_', ' ')} | "
            f"[{det}] → [{dep}] | confidence={conf:.3f}"
        )
        g.add((class_map[tbl], RDFS.comment, Literal("\n".join(fd_parts))))

    state["ontology_graph"]  = g
    state["class_map"]       = class_map
    state["property_map"]    = property_map
    state["class_count"]     = len(class_map)
    state["property_count"]  = len(property_map) + len(added_obj_props)
    state["triple_count"]    = len(g)
    state["phase"]           = "built"

    logger.info(
        "Ontology built: %d classes, %d datatype props, %d object props, %d triples",
        len(class_map), len(property_map), len(added_obj_props), len(g),
    )
    return state
