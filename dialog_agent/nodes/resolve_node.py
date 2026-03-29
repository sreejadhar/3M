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
- Exact match (case-insensitive): use that exact value
- Semantic match (synonym, subcategory, broader/narrower term):
    e.g. user "savoury snacks" → stored "Snacks & Foods" (and related sub-categories)
    e.g. user "beverages" → stored "Drinks" or "Beverages"
    e.g. user "EMEA" → stored "Europe", "Middle East", "Africa"
  Include ALL semantically related stored values
- No match: set matched_values to [] and sql_fragment to null
- Never invent values that are not in the stored list

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


def _build_categorical_context(categorical_columns: Dict[str, Dict[str, List[str]]]) -> str:
    """Format categorical columns for the resolve prompt."""
    if not categorical_columns:
        return "(No categorical columns found in schema)"
    lines: List[str] = []
    for tbl, col_map in categorical_columns.items():
        for col, vals in col_map.items():
            quoted = ", ".join(repr(v) for v in vals)
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

    categorical_context = _build_categorical_context(categorical_columns)

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
