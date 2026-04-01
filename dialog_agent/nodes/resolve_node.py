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


_STEM_SUFFIXES = ("ies", "ing", "ness", "ment", "tion", "sion", "er", "est", "ed", "es", "s")


def _stem(token: str) -> str:
    """Strip common English suffixes so 'snacks'/'snack', 'savoury'/'savour' share a root."""
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — fast early-exit when gap > 2."""
    if abs(len(a) - len(b)) > 2:
        return 99
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _token_matches_text(token: str, text: str) -> bool:
    """
    True when token fuzzy-matches anywhere inside text via three strategies:
    1. Exact substring  (savory  ∈ 'Savory Snacks')
    2. Stemmed match    (snack   ~ 'Snacks & Foods' after stripping -s)
    3. Edit distance ≤ 1 for tokens ≥ 5 chars  (savory ≈ savoury)
    """
    if token in text:
        return True
    stem_t = _stem(token)
    text_words = re.findall(r'\b\w+\b', text)
    # Stemmed substring: stem(token) inside text
    if stem_t != token and stem_t in text:
        return True
    # Stem each word in the stored value and compare stems
    if len(stem_t) >= 3:
        for word in text_words:
            if _stem(word) == stem_t:
                return True
    # Edit distance for longer tokens (spelling variants: savory/savoury)
    if len(token) >= 5:
        for word in text_words:
            if abs(len(word) - len(token)) <= 2 and _edit_distance(token, word) <= 1:
                return True
    return False


def _fuzzy_match_candidates(
    natural_query: str,
    categorical_columns: Dict[str, Dict[str, List[str]]],
    column_hierarchy: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Python-level fuzzy scan between user question tokens and stored categorical values.

    Matching strategy (each adds to a token's match score):
    1. Exact substring      — token appears inside the stored value string
    2. Stemmed match        — stem(token) matches stem of any word in the stored value
    3. Edit distance ≤ 1   — handles spelling variants (savory/savoury, colour/color)

    Hierarchy promotion:
    When a token matches a value in a CHILD column (sub_category) and a parent column
    exists (category), an additional candidate is added at the PARENT level so the LLM
    is guided to filter at the correct hierarchy level.

    Returns candidates sorted by score descending:
        [{"table", "column", "stored_value", "overlap_tokens", "score",
          "match_type", "promoted_from"}, ...]
    """
    raw_tokens = re.findall(r'\b[a-zA-Z]\w+\b', natural_query.lower())
    query_tokens = {
        t for t in raw_tokens
        if t not in _STOP_WORDS and len(t) >= _MIN_TOKEN_LEN
    }
    if not query_tokens:
        return []

    # Build reverse map: child_col → parent_col per table (from hierarchy + naming)
    child_to_parent: Dict[str, Dict[str, str]] = {}  # tbl → {child_col: parent_col}
    hier = column_hierarchy or {}
    for tbl, tbl_hier in hier.items():
        for parent_col in tbl_hier:
            child_col = "sub_" + parent_col
            child_to_parent.setdefault(tbl, {})[child_col] = parent_col
    # Also derive from naming convention regardless of stored hierarchy
    for tbl, col_map in categorical_columns.items():
        for col in col_map:
            if col.startswith("sub_"):
                parent_candidate = col[4:]
                if parent_candidate in col_map:
                    child_to_parent.setdefault(tbl, {})[col] = parent_candidate

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    # Track promoted parent candidates separately to avoid duplicates
    promoted_parents: set = set()

    for tbl, col_map in categorical_columns.items():
        tbl_child_map = child_to_parent.get(tbl, {})
        for col, vals in col_map.items():
            parent_col = tbl_child_map.get(col)
            parent_vals = col_map.get(parent_col, []) if parent_col else []

            for val in vals:
                val_lower = val.lower()
                matched_tokens = {t for t in query_tokens if _token_matches_text(t, val_lower)}
                if not matched_tokens:
                    continue

                key = (tbl, col, val)
                if key not in seen:
                    seen.add(key)
                    candidates.append({
                        "table":          tbl,
                        "column":         col,
                        "stored_value":   val,
                        "overlap_tokens": sorted(matched_tokens),
                        "score":          len(matched_tokens),
                        "match_type":     "direct",
                        "promoted_from":  None,
                    })

                # Hierarchy promotion: child match → add each parent value as candidate
                # so the LLM filters at the correct (parent) level.
                if parent_col and parent_vals:
                    for pval in parent_vals:
                        pkey = (tbl, parent_col, pval)
                        if pkey in promoted_parents:
                            continue
                        promoted_parents.add(pkey)
                        # Score the parent candidate by how well the query tokens
                        # match the parent value itself
                        pval_lower = pval.lower()
                        p_matched = {t for t in query_tokens if _token_matches_text(t, pval_lower)}
                        # Only promote parents with at least some token affinity
                        # OR where the child match is strong (score >= 1)
                        if p_matched or len(matched_tokens) >= 1:
                            # Bonus: +len(matched_tokens) from the child match that drove promotion.
                            # The parent value that also matches query tokens gets an extra boost,
                            # so the correct parent ranks above siblings that share no tokens.
                            parent_bonus = len(matched_tokens) if p_matched else 0
                            candidates.append({
                                "table":          tbl,
                                "column":         parent_col,
                                "stored_value":   pval,
                                "overlap_tokens": sorted(matched_tokens | p_matched),
                                "score":          len(p_matched) + parent_bonus + 1,
                                "match_type":     "promoted_parent",
                                "promoted_from":  f"{col}={val!r}",
                            })

    # Sort: promoted parents first (higher score), then direct matches; tie-break shorter value
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
    Promoted parent candidates are labelled so the LLM knows to prefer them.
    Limited to the top 25 candidates to avoid bloating the context.
    """
    if not candidates:
        return ""
    lines = [
        "KEYWORD MATCH HINTS (Python-computed — fuzzy token overlap with spelling/stem tolerance):",
        "These are PROVEN matches. You MUST use them unless a better/higher-priority match exists.",
        "Hints labelled [PARENT LEVEL] mean: filter at the parent column — do NOT use LIKE on a child column.",
        "",
    ]
    for c in candidates[:25]:
        tokens_str = ", ".join(f'"{t}"' for t in c["overlap_tokens"])
        match_type = c.get("match_type", "direct")
        label = "  [PARENT LEVEL — via child match]  " if match_type == "promoted_parent" else "  "
        promoted = f"  (child: {c['promoted_from']})" if c.get("promoted_from") else ""
        lines.append(
            f"{label}token(s) [{tokens_str}] → "
            f"table={c['table']!r}  column={c['column']!r}  value={c['stored_value']!r}{promoted}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _call_resolve_llm(system: str, user: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("CLAUDE_API_KEY", ""),
        base_url=os.environ.get("CLAUDE_BASE_URL", "https://api.anthropic.com"),
    )
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
            overlap = sum(1 for t in term_tokens if _token_matches_text(t, val_lower))
            # Promoted parent candidates get a bonus so they beat child-level matches
            if c.get("match_type") == "promoted_parent":
                overlap += 1
            if overlap > best_score:
                best_score = overlap
                best = c

        if best and best_score > 0:
            col  = best["column"]
            val  = best["stored_value"]
            tbl  = best["table"]
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

    # ── Fuzzy pre-match: stemmed/edit-distance token overlap + hierarchy promotion ──
    fuzzy_candidates = _fuzzy_match_candidates(natural_query, categorical_columns, column_hierarchy)
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
