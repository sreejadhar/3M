"""
LangGraph state for the Dialog with Data Agent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class SQLQuery(TypedDict):
    query_id: str
    description: str
    sql: str
    table_refs: List[str]
    kg_id: str  # which KG this query targets (empty = use default config)


class Source(TypedDict):
    type: str          # "table" | "document"
    name: str          # table name (schema.table) or file name


class QueryResult(TypedDict):
    query_id: str
    description: str
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    error: Optional[str]


class ConversationTurn(TypedDict, total=False):
    """One completed Q&A exchange stored in the session history."""
    turn: int           # 1-based turn number
    question: str       # the user's original question
    insights: str       # synthesized answer (first 600 chars to keep prompt size manageable)
    tables_queried: List[str]   # table names referenced in the SQL queries

    # ── Execution diagnostics (populated from query_results + errors) ─────
    # Used by plan_node on the NEXT turn to avoid repeating failed approaches.
    query_diagnostics: List[Dict[str, Any]]
    # Each entry: {
    #   "query_id":   str,
    #   "sql":        str,          # exact SQL that was run
    #   "row_count":  int,          # rows returned (0 if error)
    #   "columns":    List[str],    # column names returned
    #   "error":      str | None,   # DB error message if failed
    #   "preflight_gaps": List[str] # pre-flight check gaps (empty = none)
    # }


class DialogState(TypedDict, total=False):
    # Inputs
    config: Any                        # DialogConfig
    natural_query: str                 # the user's NQL string
    schema_context: str                # graph/ontology summary fed to LLM
    kg_nodes: List[Dict[str, Any]]     # knowledge graph nodes (from KG agent)
    kg_edges: List[Dict[str, Any]]     # knowledge graph edges

    # Document Intelligence "mentions" edges (doc:<asset_id> -> table node),
    # set aside by retrieve_node before it filters kg_nodes/kg_edges down to
    # the selected table subgraph, so document_context_node can still find
    # which documents are linked to whichever tables got selected.
    doc_mention_edges: List[Dict[str, Any]]

    # Per-table query-relevance scores for the tables retrieve_node selected
    # (table node id -> score), reused by document_context_node to rank
    # documents by their best-linked table's relevance rather than an
    # arbitrary tie-break when several documents touch the same number of
    # selected tables.
    table_relevance_scores: Dict[str, float]

    # Document excerpts linked to the selected tables, set by
    # document_context_node — [{file_name, excerpt, topics, matched_tables}].
    # Folded into synthesize_node's prompt alongside the SQL results.
    document_context: List[Dict[str, Any]]

    # Conversation context (last N turns from the session)
    conversation_history: List[ConversationTurn]

    # Intermediate
    sql_queries: List[SQLQuery]        # planner output
    query_results: List[QueryResult]   # executor output

    # Output
    insights: str                      # LLM-derived narrative
    plan_explanation: str              # prose from plan LLM when it returns [] (unanswerable)

    # Where the answer's data came from — tables referenced by sql_queries
    # plus document_context file names, deduped. Built by synthesize_node.
    sources: List[Source]

    errors: List[str]
    phase: str                         # understand | plan | execute | synthesize | done | error

    # Categorical column values extracted from schema (used by resolve_node)
    # { sql_table_name: { sql_col_name: [val1, val2, ...] } }
    categorical_columns: Dict[str, Dict[str, List[str]]]

    # Taxonomy hierarchy between related categorical columns (parent → children)
    # { sql_table_name: { parent_col: { parent_value: [child_values] } } }
    # e.g. { "fact_market_share": { "category": { "Snacks & Foods": ["Potato Chips & Crisps", ...] } } }
    column_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]]

    # Resolved term → data-value mappings produced by resolve_node
    # [{"user_term": "savoury snacks", "column": "category",
    #   "matched_values": ["Snacks & Foods"], "sql_fragment": "LOWER(category) = 'snacks & foods'"}]
    term_resolution: List[Dict[str, Any]]

    # Active KPI definitions for this source (loaded by understand_node from kpi_store)
    # [{"kpi_id": "...", "name": "RSV Growth", "nl_formula": "...", "sql_expression": "...", ...}]
    active_kpis: List[Dict[str, Any]]

    # Business glossary terms (loaded by understand_node from glossary_store)
    # [{"term_id": "...", "name": "Gross Margin", "definition": "...", "synonyms": [...], ...}]
    glossary_terms: List[Dict[str, Any]]

    # Multi-KG federation (empty = single-KG mode)
    active_kg_ids: List[str]           # KG ids selected by router
    kg_bridges_active: List[Dict[str, Any]]  # bridges between active KGs
    multi_kg_configs: List[Any]        # one DialogConfig per active KG (for execute_node routing)

    # ── Semantic Lexicon / Evaluation Loop (dissect_node) ─────────────────
    # Business concepts resolved to real columns, either from the lexicon
    # (cache hit) or the evaluation loop (cache miss). Consumed by plan_node
    # for prompt injection. Empty = behave exactly as before this feature
    # existed.
    # [{"term": "promotion count", "kind": "derived_metric",
    #   "bindings": [{"table": "...", "column": "..."}],
    #   "aggregation": "...", "grain": "...", "filter_predicates": [...],
    #   "time_window": {...}, "provenance": "...", "approved": 0}]
    derived_metrics: List[Dict[str, Any]]

    # Concepts identified in the question that could NOT be resolved. Planning
    # proceeds exactly as today for these — presence here is never fatal.
    unresolved_terms: List[str]

    # Ambiguous proper-noun terms found by resolve_node with 2+ comparably-
    # scored candidate stored values (e.g. "Smith" could be "John Smith" or
    # "Jane Smith") — when non-empty, plan_node short-circuits to ask the
    # user to disambiguate instead of guessing and running SQL against the
    # wrong entity. Empty (the default) = behave exactly as before this
    # feature existed.
    # [{"term": "Smith", "candidates": ["John Smith", "Jane Smith"]}]
    clarification_needed: List[Dict[str, Any]]

    # Shadow-mode / diagnostic observations from dissect_node, surfaced the
    # same way state["errors"] is, so they're visible without changing
    # planner behaviour.
    lexicon_diagnostics: List[str]
