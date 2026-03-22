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
1. ONLY report numbers, percentages, and values that appear VERBATIM in the query
   results below.  Do NOT calculate, estimate, derive, round, or approximate any
   figure that is not explicitly present in the data.
2. For financial metrics (Revenue, GM, GM%, OM, OM%, margins, costs, etc.) you
   MUST quote the exact values from the result tables.  Never compute a metric
   yourself — if it is not in the results, say "not available in the data."
3. Do NOT use domain knowledge to fill in or adjust numbers.  If a number looks
   wrong or inconsistent, report it as-is and note the discrepancy — do not
   "correct" it.
4. If a query returned zero rows, say so explicitly — do not substitute estimates.
5. If any queries failed, acknowledge the gap; do not invent replacement figures.
6. Format your response as readable Markdown following this structure:
   a. Start with a brief "## Summary" section (2-3 sentences max).
   b. Use "## Key Findings" for the main data points, with bullet points.
   c. Highlight the single most important finding as a blockquote callout:
        > 💡 **Key Insight:** <one sentence stating the most important finding>
   d. If the data suggests an action the business should take, add:
        > ✅ **Recommendation:** <one concrete, actionable recommendation>
   e. You may add a "## Details" section for supporting breakdowns.
7. Do NOT reproduce the raw SQL.  You MAY include a compact result table if it
   helps clarity, but only with values taken directly from the query output.
8. Keep the response concise — business users want insight, not raw data dumps.
"""

_USER_PROMPT = """\
ORIGINAL QUESTION:
{question}
{history_section}
QUERY RESULTS (use ONLY these values — do not calculate or invent any figures):
{results_text}

Answer the question using ONLY the exact values present in the query results above.
If a metric is not in the results, say "not available in the data."
If the question refers to a previous answer (e.g. "compared to before", "same metric"),
use the CONVERSATION HISTORY above as reference — but still only report numbers
from the current query results unless explicitly comparing.
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
        max_tokens=1024,
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
        state["insights"] = (
            "No query results were produced. "
            "This may be because the schema did not contain relevant tables "
            "or all queries failed. Please check the error log."
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
