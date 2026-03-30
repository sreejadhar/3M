"""
resolve_node — Pre-resolution of user query terms to exact database values.

This node runs BETWEEN understand_node and plan_node.  It makes a lightweight
LLM call that explicitly maps each filter concept in the user's question to the
exact values stored in categorical columns.

The resolved mappings are stored in state["term_resolution"] and injected into
the plan_node prompt as mandatory WHERE clause bindings.  This prevents the LLM
SQL planner from fabricating category values that do not exist in the data
(e.g. writing WHERE category = 'Savory Snacks' when the data uses 'Snacks & Foods').

Fuzzy pre-matching
------------------
Before calling the LLM, _fuzzy_match_candidates() does a Python-level token-overlap
scan: it tokenises the user question, removes stopwords, and finds every stored
categorical value that shares at least one significant word.  These candidates are
injected into the LLM prompt as "KEYWORD MATCH HINTS".  This converts the LLM's
task from "discover the match from scratch" to "confirm / rank pre-computed hints",
which is far more reliable for domain-specific terminology like:
  "savoury snacks" → 'Snacks & Foods'
  "snacks"         → 'Snacks & Foods'
  "cola"           → 'Carbonated Drinks'  (semantic, no keyword overlap)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..state import DialogState

logger = logging.getLogger(__name__)

# ── Stop-words stripped before token overlap scoring ────────────────────────
_STOP_WORDS = {
    "which", "what", "who", "where", "when", "why", "how", "the", "a", "an",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "and", "or", "but", "if", "of", "to",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "we",
    "i", "me", "my", "his", "her", "their", "our", "us", "you", "your",
    "showed", "show", "highest", "lowest", "top", "bottom", "best", "worst",
    "most", "least", "more", "less", "all", "each", "every", "any", "some",
    "no", "not", "just", "only", "also", "than", "so", "such", "had", "has",
    "last", "first", "second", "third", "full", "period", "growth", "change",
    "value", "share", "total", "average", "count", "number", "percent",
    "across", "within", "across", "among", "per", "vs", "versus",
}

# Single-character tokens and pure-numeric tokens are never useful for matching
_MIN_TOKEN_LEN = 2


_RESOLVE_SYSTEM = """\
You are a data analyst whose only job is to map terminology used in a natural-language
question to the exact values that exist in a database.

You will be given:
1. The user's question
2. A list of categorical columns with ALL their stored values
3. KEYWORD MATCH HINTS — Python-computed token overlaps that prove certain stored
   values share words with the user's question  ← use these as primary evidence

Your task:
- Identify every filter concept in the question (e.g. a product category, a country,
  a time period, a brand, a segment name)
- For each filter concept, find the best-matching stored value(s) using the rules below
- Return ONLY a JSON object — no prose, no markdown fences

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATCHING RULES — try in order, stop at the first rule that gives a match
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. EXACT MATCH (case-insensitive): user term equals a stored value verbatim.
   sql_fragment: LOWER(col) = 'stored value'

2. YEAR / PERIOD FORMAT: user writes a plain year / quarter but the column
   stores it with a prefix or suffix (e.g. 'FY2024', 'CY2023', '2024-Q1').
   Always scan stored values for the format pattern and match accordingly.
   Examples:
     user "2024"   → stored 'FY2024'  → sql_fragment: LOWER(fiscal_year) = 'fy2024'
     user "2023"   → stored 'CY2023'  → sql_fragment: LOWER(calendar_year) = 'cy2023'
     user "Q1"     → stored 'FY2024-Q1', 'FY2023-Q1' → LIKE '%q1%'
   NEVER write WHERE fiscal_year = '2024' if the column stores 'FY2024'.

3. PARENT-CATEGORY MATCH WITH TAXONOMY HIERARCHY: If a TAXONOMY HIERARCHY is shown
   for a column, filtering at the PARENT level captures ALL child values automatically.
   Always prefer this over LIKE or IN on child columns.
   Examples:
     user "savoury snacks" → taxonomy shows: 'Snacks & Foods' → [Potato Chips & Crisps, ...]
       → sql_fragment: LOWER(category) = 'snacks & foods'
     user "beverages" → taxonomy shows: 'Beverages' → [Carbonated Drinks, Juices, ...]
       → sql_fragment: LOWER(category) = 'beverages'
   When a breakdown by sub-category is explicitly required, return BOTH parent AND
   child filter using the IN list from the taxonomy — never invent child values.

4. KEYWORD-WITHIN-VALUE MATCH: A significant word from the user's term appears
   inside a stored value (case-insensitive). This is the most common case for
   industry shorthand, abbreviated labels, and CPG/FMCG category names.
   Examples:
     user "snacks"         → stored 'Snacks & Foods'        → sql: LOWER(category) = 'snacks & foods'
     user "savoury snacks" → stored 'Snacks & Foods'        → sql: LOWER(category) = 'snacks & foods'
       (keyword "snacks" appears in 'Snacks & Foods')
     user "chips"          → stored 'Potato Chips & Crisps' → sql: LOWER(sub_category) = 'potato chips & crisps'
     user "dairy"          → stored 'Dairy & Eggs'          → sql: LOWER(category) = 'dairy & eggs'
     user "personal care"  → stored 'Beauty & Personal Care'→ sql: LOWER(category) = 'beauty & personal care'
   If KEYWORD MATCH HINTS are shown below, a hint for a given stored value is PROOF
   that a keyword overlap exists — you MUST use that stored value as the match.
   When multiple columns have keyword overlap for the same user term, prefer the
   PARENT (e.g. category > sub_category) unless the user specifically asks for
   a sub-category breakdown.

5. SEMANTIC MATCH (synonyms, domain knowledge — no keyword overlap at all):
   Use this when you know from FMCG / business domain knowledge that the user's
   term maps to a stored value even without a shared word.
   Examples:
     user "EMEA"         → stored "Europe", "Middle East", "Africa"
     user "savoury"      → stored "Snacks & Foods"   (in FMCG: savoury = salty snack foods)
     user "soft drinks"  → stored "Beverages" or "Carbonated Drinks"
     user "HPC"          → stored "Home & Personal Care"
     user "confectionery"→ stored "Candy & Chocolate" or "Sweets"
   Include ALL semantically equivalent stored values in an IN() list.

6. LAST RESORT — NO MATCH: ONLY use this when rules 1-5 all fail AND there are
   no KEYWORD MATCH HINTS for this term.
   Set matched_values to [] and sql_fragment to null.
   ⚠ If KEYWORD MATCH HINTS shows overlap for a user term, you MUST NOT use
     NO MATCH for that term — the hint is Python-computed proof of overlap.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


def _fuzzy_match_candidates(
    natural_query: str,
    categorical_columns: Dict[str, Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
    """
    Python-level token-overlap scan between user question words and stored values.

    Tokenises the query, strips stopwords and short tokens, then for every
    stored categorical value counts how many significant query tokens appear
    inside the value string (case-insensitive).

    Returns a list of candidate dicts sorted by score descending:
        [{"table", "column", "stored_value", "overlap_tokens", "score"}, ...]
    """
    # Extract significant query tokens
    raw_tokens = re.findall(r'\b[a-zA-Z]\w+\b', natural_query.lower())
    query_tokens = {
        t for t in raw_tokens
        if t not in _STOP_WORDS and len(t) >= _MIN_TOKEN_LEN
    }
    if not query_tokens:
        return []

    candidates: List[Dict[str, Any]] = []
    seen: set = set()

    for tbl, col_map in categorical_columns.items():
        for col, vals in col_map.items():
            for val in vals:
                val_lower = val.lower()
                # Which query tokens appear anywhere inside this stored value?
                overlap = {t for t in query_tokens if t in val_lower}
                if not overlap:
                    continue
                key = (col, val)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "table":          tbl,
                    "column":         col,
                    "stored_value":   val,
                    "overlap_tokens": sorted(overlap),
                    "score":          len(overlap),
                })

    # Sort: more overlapping tokens first; tie-break by shorter value (more specific)
    return sorted(candidates, key=lambda x: (-x["score"], len(x["stored_value"])))


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


def _build_hint_section(candidates: List[Dict[str, Any]]) -> str:
    """
    Format fuzzy candidates as a KEYWORD MATCH HINTS block for the LLM prompt.
    Limited to the top 25 candidates to avoid bloating the context.
    """
    if not candidates:
        return ""
    lines = [
        "KEYWORD MATCH HINTS (Python-computed — token overlap between query words and stored values):",
        "These are PROVEN matches. You MUST use them unless a better/higher-priority match exists.",
        "",
    ]
    for c in candidates[:25]:
        tokens_str = ", ".join(f'"{t}"' for t in c["overlap_tokens"])
        lines.append(
            f"  Query token(s) [{tokens_str}] overlap with "
            f"table={c['table']!r}  column={c['column']!r}  value={c['stored_value']!r}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


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


def _apply_fuzzy_fallback(
    term_resolution: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    categorical_columns: Dict[str, Dict[str, List[str]]],
    column_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]],
) -> List[Dict[str, Any]]:
    """
    Post-LLM safety net: for any resolved filter whose sql_fragment is null/empty
    but which has a fuzzy candidate with score >= 1, inject the best candidate
    as the resolution automatically (without another LLM call).

    This catches cases where the LLM returned NO MATCH despite clear keyword
    overlap — e.g. "savoury snacks" returned null but hints showed 'Snacks & Foods'.
    """
    # Index candidates by column for quick lookup
    best_by_col: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        col = c["column"]
        if col not in best_by_col or c["score"] > best_by_col[col]["score"]:
            best_by_col[col] = c

    patched = []
    for r in term_resolution:
        if r.get("sql_fragment"):
            patched.append(r)
            continue
        # Find the best fuzzy candidate whose column has the highest overlap
        # and whose stored value shares tokens with the user_term
        user_term_lower = r.get("user_term", "").lower()
        term_tokens = {
            t for t in re.findall(r'\b[a-zA-Z]\w+\b', user_term_lower)
            if t not in _STOP_WORDS and len(t) >= _MIN_TOKEN_LEN
        }
        best: Optional[Dict] = None
        best_score = 0
        for c in candidates:
            val_lower = c["stored_value"].lower()
            overlap = sum(1 for t in term_tokens if t in val_lower)
            if overlap > best_score:
                best_score = overlap
                best = c

        if best and best_score > 0:
            col  = best["column"]
            val  = best["stored_value"]
            tbl  = best["table"]
            # Prefer filtering at parent column when hierarchy exists
            hier = column_hierarchy.get(tbl, {})
            frag = f"LOWER({col}) = '{val.lower()}'"
            patched_r = dict(r)
            patched_r["column"]         = col
            patched_r["table"]          = tbl
            patched_r["matched_values"] = [val]
            patched_r["sql_fragment"]   = frag
            patched_r["reasoning"]      = (
                f"Fuzzy fallback: query token(s) {best['overlap_tokens']} "
                f"matched stored value '{val}' in column '{col}'."
            )
            patched.append(patched_r)
            logger.info(
                "resolve_node: fuzzy fallback matched %r → %r in %r.%r",
                r.get("user_term"), val, tbl, col,
            )
        else:
            patched.append(r)

    return patched


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

    # ── Fuzzy pre-match: Python token overlap ─────────────────────────────────
    fuzzy_candidates = _fuzzy_match_candidates(natural_query, categorical_columns)
    hint_section = _build_hint_section(fuzzy_candidates)
    if fuzzy_candidates:
        logger.info(
            "resolve_node: fuzzy pre-match found %d candidate(s): %s",
            len(fuzzy_candidates),
            [(c["column"], c["stored_value"], c["overlap_tokens"]) for c in fuzzy_candidates[:5]],
        )

    categorical_context = _build_categorical_context(categorical_columns, column_hierarchy)

    user_prompt = (
        f"USER QUESTION:\n{natural_query}\n\n"
        f"CATEGORICAL COLUMNS AND THEIR STORED VALUES:\n{categorical_context}\n\n"
        + (hint_section if hint_section else "")
        + "Resolve each filter concept in the question to exact stored values. "
        "Return the JSON object as specified."
    )

    logger.info(
        "resolve_node: resolving terms for question=%r using model=%s "
        "(%d categorical columns across %d tables, %d fuzzy hints)",
        natural_query[:80], model,
        sum(len(cols) for cols in categorical_columns.values()),
        len(categorical_columns),
        len(fuzzy_candidates),
    )

    try:
        raw = _call_resolve_llm(_RESOLVE_SYSTEM, user_prompt, model)
        logger.debug("resolve_node raw LLM response: %s", raw[:500])
        term_resolution = _parse_resolution(raw)

        # Safety net: fill in any null sql_fragments that have fuzzy evidence
        if fuzzy_candidates:
            term_resolution = _apply_fuzzy_fallback(
                term_resolution, fuzzy_candidates, categorical_columns, column_hierarchy
            )

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
