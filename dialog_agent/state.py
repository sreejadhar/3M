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


class QueryResult(TypedDict):
    query_id: str
    description: str
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    error: Optional[str]


class ConversationTurn(TypedDict):
    """One completed Q&A exchange stored in the session history."""
    turn: int           # 1-based turn number
    question: str       # the user's original question
    insights: str       # synthesized answer (first 600 chars to keep prompt size manageable)
    tables_queried: List[str]   # table names referenced in the SQL queries


class DialogState(TypedDict, total=False):
    # Inputs
    config: Any                        # DialogConfig
    natural_query: str                 # the user's NQL string
    schema_context: str                # graph/ontology summary fed to LLM
    kg_nodes: List[Dict[str, Any]]     # knowledge graph nodes (from KG agent)
    kg_edges: List[Dict[str, Any]]     # knowledge graph edges

    # Conversation context (last N turns from the session)
    conversation_history: List[ConversationTurn]

    # Intermediate
    sql_queries: List[SQLQuery]        # planner output
    query_results: List[QueryResult]   # executor output

    # Output
    insights: str                      # LLM-derived narrative
    plan_explanation: str              # prose from plan LLM when it returns [] (unanswerable)

    errors: List[str]
    phase: str                         # understand | plan | execute | synthesize | done | error

    # Categorical column values extracted from schema (used by resolve_node)
    # { sql_table_name: { sql_col_name: [val1, val2, ...] } }
    categorical_columns: Dict[str, Dict[str, List[str]]]

    # Resolved term → data-value mappings produced by resolve_node
    # [{"user_term": "savoury snacks", "column": "category",
    #   "matched_values": ["Snacks & Foods"], "sql_fragment": "LOWER(category) = 'snacks & foods'"}]
    term_resolution: List[Dict[str, Any]]

    # Multi-KG federation (empty = single-KG mode)
    active_kg_ids: List[str]           # KG ids selected by router
    kg_bridges_active: List[Dict[str, Any]]  # bridges between active KGs
    multi_kg_configs: List[Any]        # one DialogConfig per active KG (for execute_node routing)
