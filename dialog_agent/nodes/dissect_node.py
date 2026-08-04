"""
dissect_node — resolve business concepts ("promotion count", "top performer")
to real, profiled columns before plan_node generates SQL.

Runs between resolve_node and plan_node. Two-part mechanism:

  1. Semantic Lexicon lookup (dialog_agent/semantic_lexicon.py) — a persistent,
     source-scoped cache of concept -> validated column bindings. A hit is
     deterministic: same question, same bindings, every time.
  2. Data-Driven Evaluation Loop (this module) — runs ONLY on lexicon miss.
     Identifies unresolved concepts, assembles evidence already available in
     `state` (categorical values, FK join columns) plus one indexed profiling
     lookup against metadata_catalog, asks a constrained LLM to propose
     bindings, mechanically validates every column/literal against the real
     schema, probes the proposal with a real (harmless) SELECT, and — only if
     all of that succeeds — writes the result back to the lexicon.

Failure isolation (§6.5 of the design doc): every step is best-effort. Any
exception anywhere in this module is caught, logged, and the pipeline
continues exactly as it would have without this node — mirrors the documented
discipline of `verified_queries.get_similar` ("returns [] on any failure so a
broken embedding backend never blocks query planning").

Master off-switch: `config.lexicon_enabled` defaults to False, so this node
returns `state` completely untouched unless a caller explicitly opts in.

Design: docs/Semantic_Lexicon_And_Evaluation_Loop_Design.md
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import semantic_lexicon
from ..state import DialogState
from . import sql_identifier_resolver as _ast_resolver
from .plan_node import _extract_table_columns

logger = logging.getLogger(__name__)

# Cache of (source_id, normalized_question) -> identified concept list, so a
# repeated question costs zero extra LLM calls once concepts are known.
_IDENTIFY_CACHE: Dict[Tuple[str, str], List[str]] = {}

_MAX_TOP_VALUES = 20

# Best-effort calls (identify_concepts, dissect) must fail fast: a stalled
# request here should skip one concept, not hang the whole question for the
# SDK's much longer default timeout/retry budget.
_DISSECT_LLM_TIMEOUT_S = 25.0


def dissect_node(state: DialogState) -> DialogState:
    """Entry point. No-op (state returned unchanged) unless
    config.lexicon_enabled is True."""
    config = state.get("config")
    if not getattr(config, "lexicon_enabled", False):
        return state

    try:
        return _dissect_node_impl(state)
    except Exception as exc:
        logger.warning("dissect_node: unhandled failure, leaving state untouched — %s", exc)
        return state


def _dissect_node_impl(state: DialogState) -> DialogState:
    config = state["config"]
    natural_query = state.get("natural_query", "").strip()
    source_id = getattr(config, "source_id", "") or ""
    kg_id = getattr(config, "graphrag_kg_id", "") or "default"
    if not natural_query:
        return state

    schema_context = state.get("schema_context", "") or ""
    table_columns_map = _extract_table_columns(schema_context)
    schema = _ast_resolver.SchemaGraph(table_columns_map)
    fingerprint = semantic_lexicon.schema_fingerprint(table_columns_map, list(table_columns_map.keys()))

    # ── Step 1 — identify candidate concepts ────────────────────────────────
    try:
        concepts = _identify_concepts(natural_query, table_columns_map, config)
    except Exception as exc:
        logger.warning("dissect_node: concept identification failed — %s", exc)
        concepts = []

    max_terms = getattr(config, "dissect_max_terms", 4)
    terms = concepts[:max_terms]

    # Each term is resolved fully independently (its own lexicon lookup +,
    # on miss, its own LLM/DB evaluation loop) — nothing shares mutable state
    # across terms (semantic_lexicon opens its own DB connection per call,
    # see pg_store.cursor_ctx; _probe opens its own SQL connection per call),
    # so terms run concurrently instead of one at a time. Previously a
    # question with N lexicon-miss concepts paid N sequential evaluation
    # loops (each with its own LLM calls + a live probe query) end to end.
    if len(terms) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(terms)) as pool:
            term_outcomes = list(pool.map(
                lambda term: _resolve_one_term(term, source_id, table_columns_map, schema, state, config, fingerprint),
                terms,
            ))
    else:
        term_outcomes = [
            _resolve_one_term(term, source_id, table_columns_map, schema, state, config, fingerprint)
            for term in terms
        ]

    diagnostics: List[str] = []
    derived_metrics: List[Dict[str, Any]] = []
    unresolved_terms: List[str] = []
    for term_diagnostics, derived_metric, unresolved_term in term_outcomes:
        diagnostics.extend(term_diagnostics)
        if derived_metric is not None:
            derived_metrics.append(derived_metric)
        if unresolved_term is not None:
            unresolved_terms.append(unresolved_term)

    shadow_mode = getattr(config, "lexicon_shadow_mode", True)
    if shadow_mode:
        # Log-only: record what WOULD have been injected, inject nothing.
        for dm in derived_metrics:
            diagnostics.append(f"dissect_node: [shadow] would inject binding for {dm.get('term')!r}")
        state["lexicon_diagnostics"] = (state.get("lexicon_diagnostics") or []) + diagnostics
        state["unresolved_terms"] = (state.get("unresolved_terms") or []) + unresolved_terms
        return state

    state["derived_metrics"] = (state.get("derived_metrics") or []) + derived_metrics
    state["unresolved_terms"] = (state.get("unresolved_terms") or []) + unresolved_terms
    state["lexicon_diagnostics"] = (state.get("lexicon_diagnostics") or []) + diagnostics
    return state


def _resolve_one_term(
    term: str,
    source_id: str,
    table_columns_map: Dict[str, Set[str]],
    schema: "_ast_resolver.SchemaGraph",
    state: DialogState,
    config: Any,
    fingerprint: str,
) -> Tuple[List[str], Optional[Dict[str, Any]], Optional[str]]:
    """Resolve a single concept: lexicon lookup, then (on miss) the
    evaluation loop. Returns (diagnostics, derived_metric_or_None,
    unresolved_term_or_None). Safe to call concurrently for different terms —
    see the call site in _dissect_node_impl for why."""
    diagnostics: List[str] = []
    try:
        entry = semantic_lexicon.lookup(
            source_id, term,
            min_similarity=getattr(config, "lexicon_min_similarity", 0.62),
            current_fingerprint=fingerprint,
        )
    except Exception as exc:
        logger.warning("dissect_node: lexicon lookup failed for %r — %s", term, exc)
        entry = None

    if entry is not None:
        try:
            semantic_lexicon.bump_hit(entry.entry_id)
        except Exception:
            pass
        diagnostics.append(f"dissect_node: lexicon HIT for {term!r} -> entry {entry.entry_id}")
        derived_metric = {
            "term": entry.term, "kind": entry.kind, "bindings": entry.bindings,
            "aggregation": entry.aggregation, "grain": entry.grain,
            "filter_predicates": entry.filter_predicates, "time_window": entry.time_window,
            "provenance": entry.provenance, "approved": int(entry.approved),
        }
        return diagnostics, derived_metric, None

    diagnostics.append(f"dissect_node: lexicon MISS for {term!r} — running evaluation loop")

    if not getattr(config, "dissect_enabled", False):
        return diagnostics, None, term

    try:
        resolved = _run_evaluation_loop(term, table_columns_map, schema, state, config, fingerprint)
    except Exception as exc:
        logger.warning("dissect_node: evaluation loop failed for %r — %s", term, exc)
        resolved = None

    if resolved is None:
        return diagnostics, None, term

    diagnostics.append(f"dissect_node: resolved {term!r} via evaluation loop")
    return diagnostics, resolved, None


# ── Step 1 — identify candidate concepts ────────────────────────────────────

def _identify_concepts(
    natural_query: str,
    table_columns_map: Dict[str, Set[str]],
    config: Any,
) -> List[str]:
    """One constrained LLM call: which concepts does the question require that
    are NOT directly available as a column? Cached per (source, question)."""
    source_id = getattr(config, "source_id", "") or ""
    cache_key = (source_id, semantic_lexicon.normalize_term(natural_query))
    if cache_key in _IDENTIFY_CACHE:
        return _IDENTIFY_CACHE[cache_key]

    known_columns = sorted({c for cols in table_columns_map.values() for c in cols})
    system = (
        "You identify business concepts in a question that are NOT directly "
        "available as an existing column name, and therefore must be computed "
        "from raw data. Respond with ONLY a JSON array of short concept phrases "
        "(e.g. [\"promotion count\", \"top performer\"]). If every concept in the "
        "question maps directly to an existing column, respond with []. Do not "
        "explain, do not include markdown fences."
    )
    user = (
        f"Question: {natural_query}\n\n"
        f"Existing columns across the schema (a concept matching one of these "
        f"names closely is NOT a candidate): {', '.join(known_columns[:200])}"
    )

    try:
        from llm_client import get_client as _get_client
        client = _get_client()
        model = getattr(config, "dissect_llm_model", "claude-haiku-4-5")
        resp = client.messages.create(
            model=model, max_tokens=256, system=system,
            messages=[{"role": "user", "content": user}],
            # This is a best-effort side-path (failure just skips the concept,
            # see the except below) — it must not inherit the SDK's ~10-minute
            # default timeout/retry budget, or one stalled call blocks the
            # whole question for many minutes. Fail fast instead.
            timeout=_DISSECT_LLM_TIMEOUT_S,
        )
        raw = resp.content[0].text.strip()
        concepts = _extract_json_array(raw)
    except Exception as exc:
        logger.warning("dissect_node: _identify_concepts LLM call failed — %s", exc)
        concepts = []

    concepts = [c.strip() for c in concepts if isinstance(c, str) and c.strip()]
    _IDENTIFY_CACHE[cache_key] = concepts
    return concepts


def _extract_json_array(raw: str) -> List[Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except Exception:
        return []


# ── Step 2 — assemble evidence ───────────────────────────────────────────────

def _assemble_evidence(
    tables: List[str],
    state: DialogState,
    config: Any,
) -> Dict[str, Any]:
    """Read-only. No live source-DB query.
    - categorical_columns (state): literal values already sampled.
    - kg_edges (state): join_columns already parsed onto each edge.
    - metadata_catalog: one indexed lookup for profiling (unique_count,
      null_count, min/max, semantic_role, statistical_type) — this is the one
      piece genuinely NOT already in DialogState (see plan doc's correction
      to the design's §2.1/§5.2 assumption)."""
    categorical_columns = state.get("categorical_columns") or {}
    kg_edges = state.get("kg_edges") or []
    source_id = getattr(config, "source_id", "") or ""

    joins: List[Dict[str, Any]] = []
    table_set_lower = {t.lower() for t in tables}
    for edge in kg_edges:
        join_cols = edge.get("join_columns") or []
        if not join_cols:
            continue
        joins.append({
            "from": edge.get("from", ""), "to": edge.get("to", ""),
            "label": edge.get("label", ""), "join_columns": join_cols,
        })

    profiling: Dict[str, List[Dict[str, Any]]] = {}
    table_row_counts: Dict[str, int] = {}
    if source_id:
        try:
            profiling, table_row_counts = _load_profiling(source_id, tables)
        except Exception as exc:
            logger.warning("dissect_node: metadata_catalog profiling lookup failed — %s", exc)
            profiling, table_row_counts = {}, {}

    literal_values: Dict[str, Dict[str, List[str]]] = {}
    for table, cols in categorical_columns.items():
        if table.lower() in table_set_lower or not tables:
            literal_values[table] = {c: vals[:_MAX_TOP_VALUES] for c, vals in cols.items()}

    return {
        "joins": joins, "profiling": profiling, "literal_values": literal_values,
        # Row counts let the dissector distinguish an event-log-shaped table
        # (rows >> distinct entities, e.g. a history table) from a
        # one-row-per-entity snapshot table — the signal that catches picking
        # a "current status" table for a concept that needs history. Purely
        # structural, not tied to any one schema's naming.
        "table_row_counts": table_row_counts,
    }


def _load_profiling(source_id: str, tables: List[str]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    """Single indexed read against metadata_catalog's md_entities/md_attributes
    — same cost class as verified_queries.list_all, not a live schema scan of
    the source database."""
    import metadata_catalog

    wanted = {t.lower() for t in tables}
    result: Dict[str, List[Dict[str, Any]]] = {}
    row_counts: Dict[str, int] = {}
    for entity in metadata_catalog.list_entities(source_id=source_id):
        table_name = entity.get("table_name") or ""
        if wanted and table_name.lower() not in wanted:
            continue
        full = metadata_catalog.get_entity(entity.get("metadata_id"))
        if not full:
            continue
        attrs = []
        for a in full.get("attributes") or []:
            attrs.append({
                "column_name": a.get("column_name"),
                "data_type": a.get("data_type"),
                "semantic_role": a.get("semantic_role"),
                "statistical_type": a.get("statistical_type"),
                "unique_count": a.get("unique_count"),
                "null_count": a.get("null_count"),
                "min_value": a.get("min_value"),
                "max_value": a.get("max_value"),
                "top_values": a.get("top_values"),
                "is_primary_key": a.get("is_primary_key"),
                # Already profiled for every source (not HR-specific) — lets
                # the validator prefer an identifier/FK column over a
                # free-text one for establishing a relationship.
                "is_foreign_key": a.get("is_foreign_key"),
                "fk_references": a.get("fk_references"),
            })
        result[table_name] = attrs
        if full.get("row_count") is not None:
            row_counts[table_name] = full["row_count"]
    return result, row_counts


# ── Step 3 — Data Dissector LLM ──────────────────────────────────────────────

def _dissect(term: str, evidence: Dict[str, Any], config: Any) -> Optional[Dict[str, Any]]:
    system = (
        "You are a Data Dissector. Given a business concept and REAL profiled "
        "evidence about a database schema (actual columns, actual literal "
        "values, actual join columns), map the concept onto real columns. You "
        "may NOT invent a column name that is not listed in the evidence. You "
        "may NOT invent a filter literal that does not appear in that column's "
        "top_values. If the concept cannot be resolved from the evidence given, "
        "set \"resolvable\": false.\n\n"
        "Five rules that apply to ANY schema, not just this one:\n"
        "0. COMPLETE THE JOIN: if resolving this concept requires reading "
        "from two tables, \"bindings\" MUST include the column on BOTH sides "
        "of that join (e.g. table_a.fk_id AND table_b.id) — not just the "
        "value/aggregate columns. A rationale that mentions joining to "
        "another table's column without that column appearing in "
        "\"bindings\" is incomplete and will be rejected.\n"
        "1. RELATIONSHIPS: to connect two tables, prefer a column flagged "
        "is_foreign_key or semantic_role=identifier over a free-text/name "
        "column (e.g. a last name) — text matches collide when values repeat.\n"
        "2. NEVER join two columns that are each their OWN table's "
        "is_primary_key=true column. A table's own primary key identifies "
        "its own rows; it is not automatically a foreign key into another "
        "table just because the column is also named 'id' there.\n"
        "3. EVENT LOG vs SNAPSHOT: table_row_counts tells you, per table, how "
        "many rows exist. If a concept needs a COUNT or HISTORY of "
        "occurrences over time, prefer a table whose row count is much "
        "larger than the number of distinct entities (an event/history log) "
        "over a table with roughly one row per entity (a current-state "
        "snapshot) — the snapshot table cannot hold a count of past events.\n"
        "4. DURATION: if the concept is a duration or elapsed time (tenure, "
        "age, 'how long', years of service), it is the difference between an "
        "anchor date and a reference point (now, or another date column) — "
        "NEVER a bare MIN/MAX of one date column alone (that only finds the "
        "earliest/latest EVENT, not an elapsed duration). Say so explicitly: "
        "set \"aggregation\" to something like \"reference_date - "
        "<anchor column>\", not a bare \"MIN\"/\"MAX\".\n\n"
        "Respond with ONLY a JSON object of this exact shape, no markdown "
        "fences, no explanation outside the JSON:\n"
        '{"term": "...", "resolvable": true, "rationale": "...", '
        '"bindings": [{"table": "...", "column": "..."}], '
        '"aggregation": "...", "grain": "...", '
        '"filter_predicates": [{"column": "...", "op": "=", "value": "...", '
        '"value_source": "top_values"}], '
        '"time_window": null, "confidence": 0.5}'
    )
    user = f"Concept: {term}\n\nEvidence (JSON):\n{json.dumps(evidence)[:12000]}"

    try:
        from llm_client import get_client as _get_client
        client = _get_client()
        model = getattr(config, "dissect_llm_model", "claude-haiku-4-5")
        resp = client.messages.create(
            model=model, max_tokens=1024, system=system,
            messages=[{"role": "user", "content": user}],
            # See _identify_concepts: fail fast rather than inherit the SDK's
            # long default timeout/retry budget on this best-effort call.
            timeout=_DISSECT_LLM_TIMEOUT_S,
        )
        raw = resp.content[0].text.strip()
    except Exception as exc:
        logger.warning("dissect_node: dissector LLM call failed for %r — %s", term, exc)
        return None

    proposal = _extract_json_object(raw)
    if not proposal or not proposal.get("resolvable"):
        return None
    return proposal


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


# ── Step 4 — mechanical validation ──────────────────────────────────────────

# Generic English cue for "elapsed time since an anchor point" concepts.
# Deliberately schema/domain-agnostic — matches on the CONCEPT phrase the
# question produced, not on any column or table name.
_DURATION_TERM_RE = re.compile(
    r"\b(tenure|duration|how\s+long|years?\s+of\s+(service|experience|employment)|"
    r"time\s+since|age)\b",
    re.IGNORECASE,
)
# A bare MIN/MAX (optionally qualified, e.g. "MAX(col)") with no arithmetic —
# the shape that answers "most/least recent event", not "elapsed duration".
_BARE_MINMAX_RE = re.compile(r"^\s*(MIN|MAX)\s*(\([^)]*\))?\s*$", re.IGNORECASE)


def _column_info(evidence: Dict[str, Any], table: str, column: str) -> Optional[Dict[str, Any]]:
    for t, attrs in (evidence.get("profiling") or {}).items():
        if t.lower() != table.lower():
            continue
        for a in attrs:
            if (a.get("column_name") or "").lower() == column.lower():
                return a
    return None


def _identifier_columns(evidence: Dict[str, Any], table: str) -> List[Dict[str, Any]]:
    for t, attrs in (evidence.get("profiling") or {}).items():
        if t.lower() == table.lower():
            return [a for a in attrs if a.get("semantic_role") == "identifier" or a.get("is_foreign_key")]
    return []


def _validate_proposal(
    proposal: Dict[str, Any],
    schema: "_ast_resolver.SchemaGraph",
    evidence: Dict[str, Any],
    term: str = "",
) -> Tuple[bool, str]:
    bindings = proposal.get("bindings") or []
    if not bindings:
        return False, "no bindings proposed"

    for b in bindings:
        table, column = b.get("table", ""), b.get("column", "")
        if not table or not column:
            return False, f"incomplete binding {b!r}"
        if not schema.has_table(table):
            return False, f"table {table!r} does not exist in schema"
        if not schema.has_column(table, column):
            return False, f"column {table}.{column} does not exist in schema"

    literal_values = evidence.get("literal_values") or {}
    for f in (proposal.get("filter_predicates") or []):
        col, val = f.get("column", ""), f.get("value")
        if val is None:
            continue
        found = False
        for table_cols in literal_values.values():
            vals = table_cols.get(col) or []
            if str(val) in [str(v) for v in vals]:
                found = True
                break
        if not found:
            return False, f"filter literal {val!r} for column {col!r} not present in profiled top_values"

    # ── Generic relationship-key checks (no domain/schema knowledge) ───────
    tables_used = {b["table"] for b in bindings}
    if len(tables_used) > 1 and evidence.get("profiling"):
        for i, b1 in enumerate(bindings):
            for b2 in bindings[i + 1:]:
                if b1["table"] == b2["table"]:
                    continue
                info1 = _column_info(evidence, b1["table"], b1["column"])
                info2 = _column_info(evidence, b2["table"], b2["column"])
                if not info1 or not info2:
                    continue
                # Rule: a column that is its OWN table's primary key is an
                # identity for that table's rows, not automatically a
                # foreign key into a different table sharing the same
                # column name (e.g. two unrelated tables each having an
                # autoincrement "id"). Joining two independent primary keys
                # together is virtually never a valid relationship.
                if info1.get("is_primary_key") and info2.get("is_primary_key"):
                    return False, (
                        f"{b1['table']}.{b1['column']} and {b2['table']}.{b2['column']} "
                        f"are each their own table's primary key — joining two "
                        f"independent primary keys is not a valid relationship; "
                        f"a real join needs a column flagged as a foreign key "
                        f"referencing the other table"
                    )

        # Rule: prefer identifier/FK columns over free-text/categorical
        # columns for establishing a relationship, when the SAME table also
        # has an identifier/FK column available in evidence.
        for b in bindings:
            info = _column_info(evidence, b["table"], b["column"])
            if not info:
                continue
            is_id_like = info.get("semantic_role") == "identifier" or info.get("is_foreign_key")
            if is_id_like or info.get("statistical_type") != "categorical":
                continue
            better = [
                a["column_name"] for a in _identifier_columns(evidence, b["table"])
                if a.get("column_name") != b["column"]
            ]
            if better:
                return False, (
                    f"{b['table']}.{b['column']} is a free-text/categorical column being "
                    f"used to relate two tables, but {b['table']} also has identifier/FK "
                    f"column(s) {better} available — prefer an identifier/FK column over "
                    f"free-text matching for relationships (text values can collide)"
                )

    # ── Generic duration-concept check ──────────────────────────────────────
    # Schema-agnostic: keyed only on the English concept phrase, not on any
    # column or table name, so it applies to any domain's "tenure"/"age"/
    # "duration" style concept.
    if term and _DURATION_TERM_RE.search(term):
        agg = (proposal.get("aggregation") or "").strip()
        if _BARE_MINMAX_RE.match(agg):
            return False, (
                f"concept {term!r} describes a duration/elapsed time, but the proposed "
                f"aggregation {agg!r} is a bare MIN/MAX of a single date column — that "
                f"finds the most/least recent EVENT, not an elapsed duration. Express it "
                f"as the difference between an anchor date and a reference point."
            )

    return True, ""


# ── Step 5 — probe execution ─────────────────────────────────────────────────

def _probe(proposal: Dict[str, Any], config: Any, state: DialogState) -> Tuple[bool, str]:
    """Compile the minimum statement proving the bindings are selectable and
    run it via execute_node._run_sql, which never raises. Probes column
    selectability and filter validity only — not the full aggregate/window
    expression (dialect-fragile; the generator remains responsible for that,
    still gated by the existing AST hallucination check)."""
    if not getattr(config, "dissect_probe_enabled", True):
        return True, "probe disabled"

    # NOTE: `from . import execute_node` would resolve to the FUNCTION
    # `execute_node.execute_node` here, not the submodule — dialog_agent/
    # nodes/__init__.py's `from .execute_node import execute_node` rebinds
    # the package attribute of the same name once the package finishes
    # importing. Import the private helper directly to avoid that trap.
    from .execute_node import _run_sql

    bindings = proposal.get("bindings") or []
    if not bindings:
        return False, "no bindings to probe"

    table = bindings[0]["table"]
    cols = ", ".join(sorted({_quote_ident(b["column"], config.db_type) for b in bindings if b["table"] == table}))
    if not cols:
        cols = _quote_ident(bindings[0]["column"], config.db_type)

    where_parts = []
    for f in (proposal.get("filter_predicates") or []):
        if f.get("column") and f.get("value") is not None:
            where_parts.append(f"{_quote_ident(f['column'], config.db_type)} = '{f['value']}'")
    where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

    sql = f"SELECT {cols} FROM {table}{where_clause} LIMIT 1"
    result = _run_sql(config, sql, state=state)
    if result.get("error"):
        return False, str(result["error"])
    return True, ""


def _quote_ident(name: str, db_type: str) -> str:
    if (db_type or "").lower() in ("snowflake", "sqlserver"):
        return f'"{name}"'
    return name


# ── Orchestration of steps 2-6 for one term ─────────────────────────────────

def _run_evaluation_loop(
    term: str,
    table_columns_map: Dict[str, Set[str]],
    schema: "_ast_resolver.SchemaGraph",
    state: DialogState,
    config: Any,
    fingerprint: str,
) -> Optional[Dict[str, Any]]:
    tables = list(table_columns_map.keys())
    evidence = _assemble_evidence(tables, state, config)

    proposal = _dissect(term, evidence, config)
    if proposal is None:
        return None

    ok, reason = _validate_proposal(proposal, schema, evidence, term)
    if not ok:
        logger.info("dissect_node: proposal for %r rejected — %s", term, reason)
        return None

    probe_ok, probe_reason = _probe(proposal, config, state)
    if not probe_ok:
        logger.info("dissect_node: probe failed for %r — %s", term, probe_reason)
        return None

    source_id = getattr(config, "source_id", "") or ""
    entry = semantic_lexicon.LexiconEntry(
        source_id=source_id,
        term=semantic_lexicon.normalize_term(term),
        display_term=term,
        kind="derived_metric",
        bindings=proposal.get("bindings") or [],
        aggregation=proposal.get("aggregation") or "",
        grain=proposal.get("grain") or "",
        filter_predicates=proposal.get("filter_predicates") or [],
        time_window=proposal.get("time_window"),
        rationale=proposal.get("rationale") or "",
        confidence=float(proposal.get("confidence") or 0.0),
        provenance="llm_dissector",
        probe_ok=True,
        approved=False,
        schema_fingerprint=fingerprint,
    )
    try:
        semantic_lexicon.save(entry)
    except Exception as exc:
        logger.warning("dissect_node: failed to persist resolved concept %r — %s", term, exc)

    return {
        "term": entry.term, "kind": entry.kind, "bindings": entry.bindings,
        "aggregation": entry.aggregation, "grain": entry.grain,
        "filter_predicates": entry.filter_predicates, "time_window": entry.time_window,
        "provenance": entry.provenance, "approved": 0,
    }
