"""
synthesize_node — Stitch query results and derive insights with LLM.

The LLM receives:
  - The original natural language question
  - Each executed query (description + SQL + tabular results as markdown)

It returns a narrative insight in plain markdown.
"""
from __future__ import annotations

import logging
import os
from typing import List

from ..state import DialogState, QueryResult

logger = logging.getLogger(__name__)

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


def _result_to_markdown(qr: QueryResult) -> str:
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

    # Markdown table (max 20 rows to keep prompt size manageable)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    lines += [header, sep]

    for row in rows[:20]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")

    if len(rows) > 20:
        lines.append(f"*... and {len(rows)-20} more rows*")

    return "\n".join(lines)


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

    results_text = "\n\n".join(_result_to_markdown(qr) for qr in query_results)

    # Trim if too long (rough token budget ~4k chars ≈ 1k tokens)
    if len(results_text) > config.max_insight_rows * 4:
        results_text = results_text[: config.max_insight_rows * 4] + "\n\n*(truncated)*"

    # Build conversation history context
    history = state.get("conversation_history") or []
    if history:
        lines = ["CONVERSATION HISTORY (for context on follow-up questions):"]
        for turn in history:
            lines.append(f"Q{turn['turn']}: {turn['question']}")
            if turn.get("insights"):
                lines.append(f"  Previous answer: {turn['insights'][:400]}")
        history_section = "\n".join(lines) + "\n"
    else:
        history_section = ""

    user_prompt = _USER_PROMPT.format(
        question=natural_query,
        history_section=history_section,
        results_text=results_text,
    )

    try:
        insights = _call_llm(
            _SYSTEM_PROMPT, user_prompt,
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
