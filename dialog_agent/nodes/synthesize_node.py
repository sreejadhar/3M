"""
synthesize_node — Stitch query results and derive insights with LLM.

The LLM receives:
  - The original natural language question
  - Each executed query (description + SQL + tabular results as markdown)

It returns a narrative insight in plain markdown.

Token management:
  - Dynamic row limit: reduces rows per result table (20 → 10 → 5) when total
    result content is large.
  - History summarization: older turns (beyond MAX_VERBATIM_TURNS) are
    summarized with a cheap Haiku call to keep the prompt compact.
  - Token guard: hard cap at 180k input tokens via guard_synth_sections().
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from ..state import DialogState, QueryResult
from ..token_guard import estimate_tokens, guard_synth_sections

logger = logging.getLogger(__name__)

# ── History compression settings ─────────────────────────────────────────────
# Show this many recent turns verbatim; summarize anything older.
_MAX_VERBATIM_TURNS = 5

# ── Dynamic row limits (chars of rendered result markdown) ───────────────────
_RESULTS_LARGE_CHARS  = 80_000   # ~20k tokens  → reduce to 10 rows/query
_RESULTS_XLARGE_CHARS = 200_000  # ~50k tokens  → reduce to  5 rows/query

_SYSTEM_PROMPT = """\
You are an expert data analyst.  You have been given the results of one or more
SQL queries that were executed to answer a user's question about their database.

CRITICAL RULES — follow these without exception:
1. ONLY report numbers, percentages, and values that appear in the query results.
   Do NOT invent or hallucinate figures not present in the data.
2. For financial metrics (Revenue, GM, GM%, OM, OM%, margins, costs, etc.) you
   MUST quote the exact values from the result tables.  If a metric is not in the
   results, say "not available in the data."
3. Do NOT use domain knowledge to fill in or adjust numbers.  If a number looks
   wrong or inconsistent, report it as-is and note the discrepancy.
4. If a query returned zero rows, say so explicitly — do not substitute estimates.
5. If any queries failed, acknowledge the gap; do not invent replacement figures.
6. ANALYTICAL DEPTH — this is mandatory, not optional:
   a. Look for PATTERNS across groups: are values surprisingly similar or different?
   b. Surface NULL FINDINGS: if a metric is nearly identical across all groups
      (e.g. average scores within a tight range like 0.82–0.84 across all categories),
      that flat distribution IS the insight — it means the metric does not predict
      the outcome.  State this explicitly and explain what it implies.
   c. Look for OUTLIERS: which group is highest/lowest, and by how much?
   d. Look for UNEXPECTED results: findings that contradict the obvious assumption
      are often the most valuable insight.  Flag them clearly.
   e. If results span multiple queries, CONNECT the findings — do the results
      corroborate or contradict each other?
7. Format your response as readable Markdown:
   a. "## Summary" — 2-3 sentences covering the main answer.
   b. "## Key Findings" — bullet points with the most important data points,
      including any null findings or surprising uniformity.
   c. Highlight the single most important or surprising finding:
        > 💡 **Key Insight:** <one sentence — prefer counter-intuitive or null findings
        >    over obvious ones>
   d. If the data suggests a concrete business action:
        > ✅ **Recommendation:** <specific, actionable next step>
   e. Optional "## Details" for supporting breakdowns or caveats.
8. Do NOT reproduce the raw SQL.  You MAY include a compact result table if it
   helps clarity, using only values from the query output.
9. Keep the response concise but analytically complete — business users need
   both the numbers AND what those numbers mean.
"""

_USER_PROMPT = """\
ORIGINAL QUESTION:
{question}
{history_section}
QUERY RESULTS:
{results_text}

Instructions:
- Use ONLY values present in the query results above — do not invent figures.
- Before writing your response, scan every numeric column for these patterns:
    * Are values across groups surprisingly SIMILAR (tight range)? → null finding
    * Are values across groups DIVERGENT? → identify the outlier and the gap
    * Does the pattern CONTRADICT the question's assumption? → flag it prominently
- If a metric is not in the results, say "not available in the data."
- If the question refers to a previous answer, use the CONVERSATION HISTORY for
  context but report only current query result values unless explicitly comparing.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_result_chars(query_results: List[QueryResult]) -> int:
    """Quick estimate of total rendered markdown chars without building the strings."""
    total = 0
    for qr in query_results:
        total += len(qr.get("description", "")) + 50  # header overhead
        if qr.get("error"):
            total += len(qr.get("error", ""))
            continue
        cols = qr.get("columns") or []
        rows = qr.get("rows") or []
        col_chars = sum(len(str(c)) for c in cols) + len(cols) * 3
        total += col_chars * 2  # header + separator
        for row in rows[:20]:
            total += sum(len(str(v)) for v in row) + len(row) * 3
    return total


def _dynamic_row_limit(query_results: List[QueryResult]) -> int:
    """
    Choose max rows per result table based on estimated total content size.
    Reduces rows to stay under token thresholds.
    """
    estimated = _estimate_result_chars(query_results)
    if estimated > _RESULTS_XLARGE_CHARS:
        logger.info(
            "synthesize_node: large results (~%d chars) — capping at 5 rows/query",
            estimated,
        )
        return 5
    if estimated > _RESULTS_LARGE_CHARS:
        logger.info(
            "synthesize_node: moderate results (~%d chars) — capping at 10 rows/query",
            estimated,
        )
        return 10
    return 20


def _result_to_markdown(qr: QueryResult, max_rows: int = 20) -> str:
    lines = [
        f"### {qr['query_id']}: {qr['description']}",
    ]
    if qr.get("error"):
        lines.append(f"**Error:** {qr['error']}")
        return "\n".join(lines)

    cols = qr["columns"]
    rows = qr["rows"]

    if not cols:
        lines.append("*(no data returned)*")
        return "\n".join(lines)

    lines.append(f"*Rows returned: {qr['row_count']}*")

    # Markdown table capped at max_rows
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    lines += [header, sep]

    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    if len(rows) > max_rows:
        lines.append(f"*... and {len(rows) - max_rows} more rows (truncated)*")

    return "\n".join(lines)


def _summarize_old_turns(old_turns: list, model: str) -> str:
    """
    Summarize older conversation turns into a compact bullet-point paragraph
    using a cheap Haiku call.  Returns empty string on failure.
    """
    lines = [
        "Summarize these past Q&A exchanges into 3-5 bullet points. "
        "Be very concise. Focus on key findings, dimensions/filters used, and metrics:\n"
    ]
    for turn in old_turns:
        lines.append(f"Q{turn['turn']}: {turn['question']}")
        if turn.get("insights"):
            lines.append(f"A: {turn['insights'][:200]}")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            system="You summarize data analysis Q&A history into concise bullet points.",
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        return msg.content[0].text if msg.content else ""
    except Exception as exc:
        logger.warning("synthesize_node: history summarization failed — %s", exc)
        return ""


def _build_history_section(history: list, model: str) -> str:
    """
    Build the CONVERSATION HISTORY section for the synthesize prompt.
    If history is long, older turns are summarized with a Haiku call.
    """
    if not history:
        return ""

    lines = ["CONVERSATION HISTORY (for context on follow-up questions):"]

    if len(history) > _MAX_VERBATIM_TURNS:
        old_turns = history[:-3]
        recent    = history[-3:]
        summary   = _summarize_old_turns(old_turns, model)
        if summary:
            lines.append(
                f"[Summary of {len(old_turns)} earlier turn(s) — use for background context:]"
            )
            lines.append(summary)
            lines.append("")
    else:
        recent = history

    for turn in recent:
        lines.append(f"Q{turn['turn']}: {turn['question']}")
        if turn.get("insights"):
            lines.append(f"  Previous answer: {turn['insights'][:400]}")

    return "\n".join(lines) + "\n"


def _call_llm(system: str, user: str, model: str, temperature: float) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text if msg.content else ""


def synthesize_node(state: DialogState) -> DialogState:
    """Combine query results into LLM-generated insights."""
    logger.info("=== synthesize_node ===")

    config         = state["config"]
    natural_query  = state.get("natural_query", "")
    query_results: List[QueryResult] = state.get("query_results") or []

    if not query_results:
        # If plan_node captured an explanation from the LLM (returned [] with prose),
        # surface that directly — it's a meaningful "can't answer" reason from the model.
        plan_explanation = state.get("plan_explanation", "").strip()
        if plan_explanation:
            state["insights"] = plan_explanation
            state["phase"] = "synthesize"
            return state

        # Otherwise fall back to a generic message with any logged errors
        errors = state.get("errors") or []
        if errors:
            error_lines = "\n".join(f"- {e}" for e in errors)
            detail = f"\n\n**Why this happened:**\n{error_lines}"
        else:
            detail = ""
        state["insights"] = (
            "No query results were produced. "
            "This may be because the schema did not contain relevant tables, "
            "the question could not be answered from the available data, "
            "or all generated queries were rejected during validation."
            + detail
        )
        state["phase"] = "synthesize"
        return state

    # ── Dynamic row limit ─────────────────────────────────────────────────────
    max_rows    = _dynamic_row_limit(query_results)
    results_text = "\n\n".join(_result_to_markdown(qr, max_rows) for qr in query_results)

    # Legacy char-based safety truncation (config.max_insight_rows is in rows,
    # but the old guard used * 4 as a char budget — keep as a backstop).
    char_budget = config.max_insight_rows * 4
    if len(results_text) > char_budget:
        results_text = results_text[:char_budget] + "\n\n*(truncated)*"

    # ── History section with summarization ───────────────────────────────────
    history          = state.get("conversation_history") or []
    haiku_model      = getattr(config, "plan_llm_model", "claude-haiku-4-5-20251001")
    history_section  = _build_history_section(history, haiku_model)

    # ── Token guard ───────────────────────────────────────────────────────────
    # Prepend analyst-role context so the LLM tailors terminology and framing
    analyst_role = getattr(config, "analyst_role", "").strip()
    if analyst_role:
        role_prefix = (
            f"You are assisting a **{analyst_role}**. "
            f"Tailor your insights, terminology, and recommendations to be most "
            f"relevant and actionable for this specific role.\n\n"
        )
        system = role_prefix + _SYSTEM_PROMPT
    else:
        system = _SYSTEM_PROMPT

    # base_chars: system + static template text + question (everything except
    # the two variable sections that the guard may trim)
    base_chars = len(system) + len(natural_query) + len(_USER_PROMPT) + 50
    results_text, history_section = guard_synth_sections(
        system, results_text, history_section, base_chars
    )

    user_prompt = _USER_PROMPT.format(
        question=natural_query,
        history_section=history_section,
        results_text=results_text,
    )

    try:
        insights = _call_llm(
            system, user_prompt,
            config.llm_model, config.llm_temperature,
        )
        logger.info("synthesize_node: insights generated (%d chars)", len(insights))
    except Exception as exc:
        logger.exception("synthesize_node: LLM call failed")
        insights = f"*Insight generation failed: {exc}*"
        state["errors"].append(f"synthesize_node: {exc}")

    state["insights"] = insights
    state["phase"] = "synthesize"
    return state
