"""
synthesize_node — Stitch query results and derive insights with LLM.

Output structure
----------------
The final ``insights`` string always follows this layout:

    ## Summary
    2-3 sentence direct answer.

    ## Key Findings
    - **Metric / Group**: value — one-line interpretation
    ...

    > 💡 **Key Insight:** <single most important finding>
    > ✅ **Recommendation:** <actionable next step>  ← only when applicable

    ## Analysis
    Deeper narrative: patterns, outliers, comparisons across queries.

    ---
    ## Data
    ### Q1: <description>
    | col | col | … |
    |-----|-----|---|
    | …   | …   | … |
    *N rows returned*

The ``## Data`` section is generated programmatically from query_results so it
is always present and consistently formatted.  The LLM focuses on interpretation.

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
import re
from typing import List, Optional

from ..state import DialogState, QueryResult
from ..token_guard import estimate_tokens, guard_synth_sections

logger = logging.getLogger(__name__)

# ── History compression settings ─────────────────────────────────────────────
_MAX_VERBATIM_TURNS = 5

# ── Dynamic row limits (chars of rendered result markdown) ───────────────────
_RESULTS_LARGE_CHARS  = 80_000   # ~20k tokens  → reduce to 10 rows/query
_RESULTS_XLARGE_CHARS = 200_000  # ~50k tokens  → reduce to  5 rows/query

# ── Max rows in the programmatic ## Data section appended to output ───────────
_DATA_SECTION_MAX_ROWS = 50

_SYSTEM_PROMPT = """\
You are an expert data analyst.  You have been given the results of one or more
SQL queries that were executed to answer a user's question about their database.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL ACCURACY RULES — follow without exception
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ONLY report numbers, percentages, and values that appear in the query results.
   Do NOT invent or hallucinate figures not present in the data.
2. For financial metrics (Revenue, GM, GM%, OM, OM%, margins, costs, etc.) you
   MUST quote the exact values from the result tables.  If a metric is not in the
   results, say "not available in the data."
3. Do NOT use domain knowledge to fill in or adjust numbers.
4. If a query returned zero rows, say so explicitly — do not substitute estimates.
5. If any queries failed, acknowledge the gap; do not invent replacement figures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANALYTICAL DEPTH — mandatory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Before writing, scan every numeric column for:
   a. PATTERNS — are values across groups surprisingly similar or different?
   b. NULL FINDINGS — if a metric is nearly identical across all groups (e.g.
      scores in a tight 0.82–0.84 range), that flat distribution IS the insight.
      State it explicitly and explain what it implies for the business question.
   c. OUTLIERS — which group is highest/lowest, and by how much (absolute + %)?
   d. SURPRISES — findings that contradict the obvious assumption are most
      valuable; flag them clearly.
   e. CROSS-QUERY CONNECTIONS — if results span multiple queries, do they
      corroborate or contradict each other?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — use this exact section structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Produce EXACTLY these sections in this order:

## Summary
2-3 sentences that directly answer the question.  Lead with the most important
number or finding.  Do not restate the question.

## Key Findings
Bullet list of the 3-8 most important data points.  Format each as:
  - **<Group or Metric>**: <value with units> — <one-line interpretation>
Include any null/flat findings here — e.g.
  - **Score distribution**: nearly uniform (0.82–0.84) — score does not
    differentiate customer segments

Then place one or both callouts (omit ✅ if no actionable step is obvious):
> 💡 **Key Insight:** <single most counter-intuitive or impactful finding>
> ✅ **Recommendation:** <specific, actionable next step>

## Analysis
2-5 paragraphs of deeper commentary.  Cover:
  - Patterns, outliers, and their magnitude
  - Comparisons across groups or time periods
  - Any caveats, data quality notes, or gaps (failed queries, zero-row results)
  - Cross-query connections when multiple queries were run

Do NOT reproduce raw SQL.
Do NOT include data tables here — they will be appended automatically.
Numbers in prose: use comma-separated thousands (e.g. 1,234,567) and
include units (e.g. $4.2M, 37.5%, 2,300 employees).
"""

_USER_PROMPT = """\
ORIGINAL QUESTION:
{question}
{history_section}
QUERY RESULTS:
{results_text}

Instructions:
- Use ONLY values present in the query results above.
- Follow the exact output format from your instructions (Summary → Key Findings
  → callouts → Analysis).
- Do NOT include a "## Data" section — it will be appended automatically.
- Format numbers with thousand separators and units in your prose and bullet points.
- If the question refers to a previous answer, use CONVERSATION HISTORY for
  context but report only current query result values unless explicitly comparing.
"""


# ── Data section (programmatic) ───────────────────────────────────────────────

def _fmt_value(v: object) -> str:
    """
    Format a cell value for the markdown table.
    - Integers >= 1000 get thousand separators.
    - Floats are rounded to 2 decimal places and get thousand separators.
    - Everything else is stringified.
    """
    if isinstance(v, float):
        if v != v:          # NaN
            return "—"
        if abs(v) >= 1_000:
            return f"{v:,.2f}"
        return f"{v:.2f}"
    if isinstance(v, int) and abs(v) >= 1_000:
        return f"{v:,}"
    if v is None:
        return "—"
    return str(v)


def _result_to_data_table(qr: QueryResult, max_rows: int = _DATA_SECTION_MAX_ROWS) -> str:
    """
    Render one QueryResult as a clean markdown block for the ## Data section.
    """
    lines = [f"### {qr['query_id']}: {qr['description']}"]

    if qr.get("error"):
        lines.append(f"> ⚠️ **Query failed:** {qr['error']}")
        return "\n".join(lines)

    cols = qr.get("columns") or []
    rows = qr.get("rows") or []
    row_count = qr.get("row_count", len(rows))

    if not cols:
        lines.append("*No data returned.*")
        return "\n".join(lines)

    lines.append(f"*{row_count:,} row{'s' if row_count != 1 else ''} returned*")
    lines.append("")

    # Header
    col_strs = [str(c).replace("|", "\\|") for c in cols]
    lines.append("| " + " | ".join(col_strs) + " |")
    lines.append("| " + " | ".join(
        # Right-align numeric columns (detected by first non-null value)
        ("--:" if _is_numeric_col(rows, i) else "---")
        for i in range(len(cols))
    ) + " |")

    display_rows = rows[:max_rows]
    for row in display_rows:
        cells = [_fmt_value(v).replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(cells) + " |")

    if len(rows) > max_rows:
        lines.append(f"*… {len(rows) - max_rows:,} more rows not shown*")

    return "\n".join(lines)


def _is_numeric_col(rows: list, col_idx: int) -> bool:
    """Return True if the first non-null value in this column is numeric."""
    for row in rows[:10]:
        v = row[col_idx] if col_idx < len(row) else None
        if v is not None:
            return isinstance(v, (int, float))
    return False


def _build_data_section(query_results: List[QueryResult]) -> str:
    """Build the full ## Data section appended after the LLM output."""
    if not query_results:
        return ""
    parts = ["\n\n---\n## Data\n"]
    for qr in query_results:
        parts.append(_result_to_data_table(qr))
        parts.append("")
    return "\n".join(parts)


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _estimate_result_chars(query_results: List[QueryResult]) -> int:
    """Quick estimate of total rendered markdown chars without building strings."""
    total = 0
    for qr in query_results:
        total += len(qr.get("description", "")) + 50
        if qr.get("error"):
            total += len(qr.get("error", ""))
            continue
        cols = qr.get("columns") or []
        rows = qr.get("rows") or []
        col_chars = sum(len(str(c)) for c in cols) + len(cols) * 3
        total += col_chars * 2
        for row in rows[:20]:
            total += sum(len(str(v)) for v in row) + len(row) * 3
    return total


def _dynamic_row_limit(query_results: List[QueryResult]) -> int:
    """Choose max rows in the prompt context based on estimated content size."""
    estimated = _estimate_result_chars(query_results)
    if estimated > _RESULTS_XLARGE_CHARS:
        logger.info("synthesize_node: large results (~%d chars) — capping prompt at 5 rows/query", estimated)
        return 5
    if estimated > _RESULTS_LARGE_CHARS:
        logger.info("synthesize_node: moderate results (~%d chars) — capping prompt at 10 rows/query", estimated)
        return 10
    return 20


def _result_to_prompt_markdown(qr: QueryResult, max_rows: int = 20) -> str:
    """Render a QueryResult as markdown for the LLM prompt (compact, capped rows)."""
    lines = [f"### {qr['query_id']}: {qr['description']}"]
    if qr.get("error"):
        lines.append(f"**Error:** {qr['error']}")
        return "\n".join(lines)

    cols = qr.get("columns") or []
    rows = qr.get("rows") or []

    if not cols:
        lines.append("*(no data returned)*")
        return "\n".join(lines)

    lines.append(f"*Rows returned: {qr['row_count']}*")
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    lines += [header, sep]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    if len(rows) > max_rows:
        lines.append(f"*... and {len(rows) - max_rows} more rows*")
    return "\n".join(lines)


# ── History helpers ───────────────────────────────────────────────────────────

def _summarize_old_turns(old_turns: list, model: str) -> str:
    """Summarize older turns into compact bullets via Haiku. Returns '' on failure."""
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
    """Build CONVERSATION HISTORY section, summarizing old turns when history is long."""
    if not history:
        return ""

    lines = ["CONVERSATION HISTORY (for context on follow-up questions):"]

    if len(history) > _MAX_VERBATIM_TURNS:
        old_turns = history[:-3]
        recent    = history[-3:]
        summary   = _summarize_old_turns(old_turns, model)
        if summary:
            lines.append(f"[Summary of {len(old_turns)} earlier turn(s):]")
            lines.append(summary)
            lines.append("")
    else:
        recent = history

    for turn in recent:
        lines.append(f"Q{turn['turn']}: {turn['question']}")
        if turn.get("insights"):
            lines.append(f"  Previous answer: {turn['insights'][:400]}")

    return "\n".join(lines) + "\n"


# ── LLM call ──────────────────────────────────────────────────────────────────

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


# ── Node ──────────────────────────────────────────────────────────────────────

def synthesize_node(state: DialogState) -> DialogState:
    """Combine query results into LLM-generated insights + programmatic data tables."""
    logger.info("=== synthesize_node ===")

    config         = state["config"]
    natural_query  = state.get("natural_query", "")
    query_results: List[QueryResult] = state.get("query_results") or []

    if not query_results:
        plan_explanation = state.get("plan_explanation", "").strip()
        if plan_explanation:
            state["insights"] = plan_explanation
            state["phase"] = "synthesize"
            return state

        errors = state.get("errors") or []
        detail = (
            "\n\n**Why this happened:**\n" + "\n".join(f"- {e}" for e in errors)
            if errors else ""
        )
        state["insights"] = (
            "No query results were produced. "
            "This may be because the schema did not contain relevant tables, "
            "the question could not be answered from the available data, "
            "or all generated queries were rejected during validation."
            + detail
        )
        state["phase"] = "synthesize"
        return state

    # ── Build prompt-side results (compact, token-capped) ─────────────────────
    max_rows    = _dynamic_row_limit(query_results)
    results_text = "\n\n".join(
        _result_to_prompt_markdown(qr, max_rows) for qr in query_results
    )

    # Legacy char-based backstop
    char_budget = config.max_insight_rows * 4
    if len(results_text) > char_budget:
        results_text = results_text[:char_budget] + "\n\n*(truncated)*"

    # ── History section ───────────────────────────────────────────────────────
    history         = state.get("conversation_history") or []
    haiku_model     = getattr(config, "plan_llm_model", "claude-haiku-4-5-20251001")
    history_section = _build_history_section(history, haiku_model)

    # ── Build system prompt (with optional analyst-role prefix) ───────────────
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

    # ── Token guard ───────────────────────────────────────────────────────────
    base_chars = len(system) + len(natural_query) + len(_USER_PROMPT) + 50
    results_text, history_section = guard_synth_sections(
        system, results_text, history_section, base_chars
    )

    user_prompt = _USER_PROMPT.format(
        question=natural_query,
        history_section=history_section,
        results_text=results_text,
    )

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        llm_output = _call_llm(
            system, user_prompt,
            config.llm_model, config.llm_temperature,
        )
        logger.info("synthesize_node: LLM output generated (%d chars)", len(llm_output))
    except Exception as exc:
        logger.exception("synthesize_node: LLM call failed")
        llm_output = f"*Insight generation failed: {exc}*"
        state["errors"].append(f"synthesize_node: {exc}")

    # ── Append programmatic ## Data section ───────────────────────────────────
    # Ensures data tables are always present, consistently formatted, and
    # independently of what the LLM chose to include in its prose.
    data_section = _build_data_section(query_results)
    insights = llm_output.rstrip() + data_section

    state["insights"] = insights
    state["phase"] = "synthesize"
    return state
