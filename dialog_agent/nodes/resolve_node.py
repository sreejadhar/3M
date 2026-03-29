"""
resolve_node — Pre-resolution of user query terms to exact database values.

This node runs BETWEEN understand_node and plan_node.  It makes a lightweight
LLM call that explicitly maps each filter concept in the user's question to the
exact values stored in categorical columns.

The resolved mappings are stored in state["term_resolution"] and injected into
the plan_node prompt as mandatory WHERE clause bindings.  This prevents the LLM
SQL planner from fabricating category values that do not exist in the data
(e.g. writing WHERE category = 'Savory Snacks' when the data uses 'Snacks & Foods').
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from ..state import DialogState

logger = logging.getLogger(__name__)

_RESOLVE_SYSTEM = """\
You are a data analyst whose only job is to map terminology used in a natural-language
question to the exact values that exist in a database.

You will be given:
1. The user's question
2. A list of categorical columns with ALL their stored values

Your task:
- Identify every filter concept in the question (e.g. a product category, a country,
  a time period, a brand, a segment name)
- For each filter concept, find the best-matching stored value(s) from the categorical
  columns listed
- Return ONLY a JSON object — no prose, no markdown fences

MATCHING RULES:

1. EXACT MATCH (case-insensitive): use that exact stored value.
   sql_fragment: LOWER(col) = 'stored value'

2. YEAR / PERIOD FORMAT: The user writes a plain year (e.g. "2024") but the column
   stores it with a prefix or suffix (e.g. 'FY2024', 'CY2024', '2024-Q1').
   Always scan the stored values for the format pattern and match accordingly.
   Examples:
     user "2024"   → stored 'FY2024'  → sql_fragment: LOWER(fiscal_year) = 'fy2024'
     user "2023"   → stored 'CY2023'  → sql_fragment: LOWER(calendar_year) = 'cy2023'
     user "Q1"     → stored 'FY2024-Q1', 'FY2023-Q1'  → use LIKE '%q1%'
   NEVER write WHERE fiscal_year = '2024' if the column stores 'FY2024'.

3. PARENT-CATEGORY MATCH WITH TAXONOMY HIERARCHY: If a TAXONOMY HIERARCHY is shown for
   a column, filtering at the PARENT level (e.g. category = 'Snacks & Foods') automatically
   captures EVERY child value listed under that parent.  Always prefer this over LIKE or
   IN on child columns when a clear parent match exists.
   Examples:
     user "savoury snacks" → taxonomy shows: 'Snacks & Foods' → [Potato Chips & Crisps, ...]
       → sql_fragment: LOWER(category) = 'snacks & foods'
       NOT: LOWER(sub_category) LIKE '%snack%'  ← misses 'Potato Chips & Crisps'
     user "beverages" → taxonomy shows: 'Beverages' → [Carbonated Drinks, Juices, ...]
       → sql_fragment: LOWER(category) = 'beverages'
   When INCLUDING ALL CHILDREN IS EXPLICITLY REQUIRED (e.g. user asks for
   "breakdown by sub-category"), return BOTH:
     • parent filter: LOWER(category) = 'snacks & foods'
     • child filter:  LOWER(sub_category) IN ('potato chips & crisps', 'tortilla chips & corn snacks', ...)
   Use the child IN list from the taxonomy hierarchy — never invent or guess child values.

4. SEMANTIC MATCH (synonyms, shorthand, no exact or parent match):
     user "EMEA"    → stored "Europe", "Middle East", "Africa"
       → sql_fragment: LOWER(region) IN ('europe', 'middle east', 'africa')
   Include ALL semantically related stored values.

5. NO MATCH: set matched_values to [] and sql_fragment to null.
   Never invent values not in the stored list.

OUTPUT FORMAT (return exactly this JSON, no other text):
{
  "resolved_filters": [
    {
      "user_term": "<the term from the question>",
      "column": "<column name>",
      "table": "<table name>",
      "reasoning": "<one sentence explaining the match>",
      "matched_values": ["<stored value 1>", "<stored value 2>"],
      "sql_fragment": "<LOWER(column) = 'stored value' or LOWER(column) IN (...) or LOWER(column) LIKE '%...'>"
    }
  ]
}

If no categorical filters are needed (e.g. purely numeric aggregation), return:
{"resolved_filters": []}
"""


def _build_categorical_context(
    categorical_columns: Dict[str, Dict[str, List[str]]],
    column_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = None,
) -> str:
    """
    Format categorical columns (with taxonomy hierarchy when available) for the resolve prompt.

    For parent columns that have a hierarchy (e.g. category → sub_category), the context
    shows the full parent→children mapping so the LLM can:
      1. Match the user term to the correct parent value
      2. Understand that filtering at parent level captures all children automatically
    """
    if not categorical_columns:
        return "(No categorical columns found in schema)"

    hierarchy = column_hierarchy or {}
    lines: List[str] = []

    for tbl, col_map in categorical_columns.items():
        tbl_hier = hierarchy.get(tbl, {})
        for col, vals in col_map.items():
            quoted = ", ".join(repr(v) for v in vals)
            if col in tbl_hier:
                # Parent column: show the hierarchy so the LLM can filter at parent level
                lines.append(
                    f"  table={tbl!r}  column={col!r}  [PARENT — stored values: [{quoted}]]"
                )
                lines.append(
                    f"    TAXONOMY HIERARCHY (filtering at {col!r} level captures ALL children automatically):"
                )
                for pval, cvals in sorted(tbl_hier[col].items()):
                    children_str = " | ".join(cvals)
                    lines.append(f"      '{pval}' → [{children_str}]")
            else:
                lines.append(f"  table={tbl!r}  column={col!r}  stored values: [{quoted}]")

    return "\n".join(lines)


def _call_resolve_llm(system: str, user: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,   # deterministic — we want the most confident match
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text if msg.content else ""


def _parse_resolution(text: str) -> List[Dict[str, Any]]:
    """Extract the resolved_filters list from the LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj.get("resolved_filters", [])
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        start = cleaned.find("{")
        end   = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(cleaned[start : end + 1])
                return obj.get("resolved_filters", [])
            except json.JSONDecodeError:
                pass
    logger.warning("resolve_node: could not parse LLM response as JSON")
    return []


def resolve_node(state: DialogState) -> DialogState:
    """
    Map user query terms to exact categorical column values before SQL planning.
    Populates state['term_resolution'] with a list of resolved filter mappings.
    """
    logger.info("=== resolve_node ===")

    categorical_columns: Dict[str, Dict[str, List[str]]] = state.get("categorical_columns") or {}
    column_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = state.get("column_hierarchy") or {}
    natural_query: str = state.get("natural_query", "")

    # Skip if no categorical data was found (DB-backed sources without samples,
    # or non-file sources where sampling is not available)
    if not categorical_columns:
        logger.info("resolve_node: no categorical columns — skipping resolution")
        state["term_resolution"] = []
        return state

    config = state["config"]
    model  = getattr(config, "llm_model", None) or os.environ.get(
        "DIALOG_LLM_MODEL", "claude-haiku-4-5-20251001"
    )

    categorical_context = _build_categorical_context(categorical_columns, column_hierarchy)

    user_prompt = (
        f"USER QUESTION:\n{natural_query}\n\n"
        f"CATEGORICAL COLUMNS AND THEIR STORED VALUES:\n{categorical_context}\n\n"
        "Resolve each filter concept in the question to exact stored values. "
        "Return the JSON object as specified."
    )

    logger.info(
        "resolve_node: resolving terms for question=%r using model=%s "
        "(%d categorical columns across %d tables)",
        natural_query[:80], model,
        sum(len(cols) for cols in categorical_columns.values()),
        len(categorical_columns),
    )

    try:
        raw = _call_resolve_llm(_RESOLVE_SYSTEM, user_prompt, model)
        logger.debug("resolve_node raw LLM response: %s", raw[:500])
        term_resolution = _parse_resolution(raw)
        logger.info(
            "resolve_node: resolved %d filter term(s): %s",
            len(term_resolution),
            [(r.get("user_term"), r.get("matched_values")) for r in term_resolution],
        )
    except Exception as exc:
        logger.warning("resolve_node: LLM call failed — %s; continuing without resolution", exc)
        term_resolution = []

    state["term_resolution"] = term_resolution
    state["phase"] = "resolve"
    return state
