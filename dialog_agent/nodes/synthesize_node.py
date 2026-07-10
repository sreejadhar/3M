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

import httpx

from ..state import DialogState, QueryResult
from ..token_guard import estimate_tokens, guard_synth_sections

logger = logging.getLogger(__name__)

# ── History compression settings ─────────────────────────────────────────────
_MAX_VERBATIM_TURNS = 5

# ── Dynamic row limits (chars of rendered result markdown) ───────────────────
_RESULTS_LARGE_CHARS  = 20_000   # ~5k tokens   → reduce to 10 rows/query
_RESULTS_XLARGE_CHARS = 60_000   # ~15k tokens  → reduce to  5 rows/query

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
6. SAMPLED DATA — when a query_id ends in "_count", it shows the TOTAL matching
   rows in the full dataset.  Its companion query (same id without "_count") is a
   sampled page of those rows.  Always state both numbers:
     "Showing 20 of 12,483 matching rows (full dataset count from <id>_count)."
   Never imply the sample is the complete result set.

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

7. STRUCTURAL ANOMALIES — LEAD WITH THEM:
   If a finding fundamentally undermines the premise of the original question,
   that finding MUST be the first sentence of the Summary — not an appendix note.
   Examples of premise-collapsing anomalies:
     • A metric that is supposed to differentiate groups shows near-zero variance
       across all groups (e.g. pricing impact % capped at 9.99–10.00% for every SKU).
       → The original ranking question cannot be answered with this metric.
         Lead: "The [metric] is effectively capped at [value] across all groups —
         it cannot be used to rank or differentiate them.  The [alternative metric]
         is the operative differentiator."
     • A "deliberate vs windfall" question is answered with an absolute index (e.g.
       price_index = 120) rather than a year-over-year movement (price_index_vs_yago).
       → Answer with what was returned, then ask naturally whether the user wants
         year-on-year price change included.  Do NOT mention column names, query
         plans, or ask the user to "re-run" anything.
     • A query returned 0 rows or the count is implausibly small.
       → Surface this as a data quality issue before drawing conclusions.

8. CO-OCCURRENCE CAVEAT — independent queries do NOT prove period-level co-occurrence:
   If the queries were run SEPARATELY (e.g. q1 retrieves pricing impact, q2 retrieves
   price index independently) and the question asks whether the same brand-pack showed
   BOTH conditions in the SAME period, your answer is INFERRED, not proven.
   You MUST include a caveat in the Analysis section:
     "Note: the two queries were not joined on period — the conclusion that [A] and [B]
      co-occurred in the same periods is based on the same brand-packs appearing in both
      result sets, not on a period-matched JOIN.  A joined query on (brand_pack_id,
      period_id) would give a definitive period-level answer."
   Omit this caveat only if one of the queries already joined on the period dimension.

8a. DUPLICATE DETECTION IN JOIN RESULTS — before citing any COUNT or SUM from a JOIN
    query, scan the raw rows for duplicate (entity, period) combinations.
    Signs of duplication: the same entity+period pair appears multiple times, OR
    the same entity appears with wildly different metric values for the same period
    (e.g. Pepsi 500ml appears 28 times with different gross_rsv — this is NOT a data
    feature; it is granular transaction/channel rows multiplying through the join).
    You MUST flag this explicitly:
      "⚠️ Duplicate rows detected: [entity] appears [N] times for period [P].
       The JOIN pulled granular rows — row count of [X] is inflated."
    Do NOT cite the inflated row count as a headline number.
    Do NOT recommend "re-run with a pre-aggregated join" — that recommendation has
    already been made and is not the analyst's job to defer again.  If a q_summary
    query is present, USE IT as the answer.  If q_summary is absent, note the
    duplication and work with the DISTINCT entity set you can derive.

9. EXECUTIVE SUMMARY TABLE — when the data spans many rows across multiple queries,
   produce a concise 3–8 row summary table in the Key Findings section that captures
   the top performers or most important segments.  Format:
     | Entity | Metric A | Metric B | Direction | Interpretation |
     |--------|---------|---------|-----------|----------------|
     | …      | …       | …       | …         | …              |
   This replaces a verbal bullet-point list when the data is tabular in nature.
   The full raw data will appear in the ## Data section; do NOT reproduce all rows
   in the narrative.  A director wants a 5-row exec table, not a scrollable dump.

9a. q_summary IS THE ANSWER — when a query named "q_summary" or ending in "_summary"
    is present in the results, it IS the primary answer.  Treat it as such:
    • Lead the Key Findings section with ALL rows from q_summary (or top 8 if > 8),
      formatted as a table with every column the query returned.
    • If q_summary has a direction / classification column (e.g. 'positive' / 'negative',
      'deliberate' / 'windfall', 'growing' / 'declining'), include that column in the
      table and use it to group or classify entities in your narrative.
    • Raw join queries (q3, etc.) that show granular period rows are supporting detail
      only — reference their row count with duplicate caveat if applicable, but do not
      use them as the primary answer.
    • NEVER defer analysis with "re-run with better aggregation" when q_summary is
      present — q_summary IS that re-run.  The analysis is complete; just report it.

9b. INCOMPLETE PLAN vs DATA UNAVAILABILITY — handle silently:
    Before writing "this cannot be answered from the data returned", ask:
    "Is the column absent from the schema, or was it simply not queried?"

    If the schema context contains a column that would answer an outstanding
    sub-question but no query was run for it, that is an INCOMPLETE QUERY PLAN.
    DO NOT expose this to the user with technical language like "the column was
    not included in the query plan" or "re-run with a query that pulls X".
    Users do not know what a query plan is and should never be asked to re-run.

    Instead, handle it in one of two ways:
      (a) If the data you DO have is sufficient to give a substantive answer,
          answer with what you have and ask a natural follow-up question:
            "Would you like me to also look at how year-on-year price changes
             compare across these brand-packs?"
          This invites the user to ask, which will trigger the correct query
          automatically on the next turn.
      (b) If the missing data is essential and no useful answer can be given
          without it, ask a plain-language clarifying question:
            "To classify these as deliberate or windfall price increases I also
             need to look at how prices changed year-on-year.  Should I include
             that in the analysis?"

    NEVER write:
      - "the X column was not included in this query plan"
      - "re-run with a query that pulls X"
      - "the classification cannot be completed from this query alone"
      - any reference to query plans, columns, schemas, or SQL to the user

    This matters because users reading "cannot be answered" conclude the data
    does not exist, which is misleading when the column is in the schema but
    simply was not fetched in this turn.

9c. SUPPORTING DOCUMENTS — when a SUPPORTING DOCUMENTS section is present below,
    those excerpts come from documents linked (via Document Intelligence) to the
    tables used in the query results. Use them to add qualitative context the
    SQL results can't provide — definitions, policy/treaty language, narrative
    explanations of why a number looks the way it does. Quote or paraphrase them
    naturally in the Analysis section when they help answer the question.
    This does NOT relax rule 1: every number, percentage, or metric you state
    must still come from the query results, never from a document. If a document
    and the query results appear to disagree on a specific figure, trust the
    query results and note the discrepancy rather than reporting the document's
    number as fact.

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
{document_section}
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
    COUNT companion queries (query_id ending in _count) get a callout banner.
    """
    is_count_companion = qr["query_id"].endswith("_count")
    title = (
        f"### {qr['query_id']}: {qr['description']}"
        if not is_count_companion
        else f"### {qr['query_id']}: Total rows in full dataset (no row limit)"
    )
    lines = [title]

    if is_count_companion and not qr.get("error"):
        rows = qr.get("rows") or []
        if rows and rows[0]:
            total = rows[0][0]
            lines.append(f"> 📊 **{_fmt_value(total)} total matching rows** in the full dataset.")
            lines.append(
                f"> The sample query shows up to {_DATA_SECTION_MAX_ROWS} rows from this result."
            )
            return "\n".join(lines)

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
        from llm_client import get_client
        client = get_client()
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


# ── Document context (from Document Intelligence cross-modal linking) ─────────

def _build_document_section(document_context: list) -> str:
    """Build SUPPORTING DOCUMENTS section from documents linked (via the KG's
    "mentions" edges) to the tables used in this query. Empty when the source
    has no linked documents or none were relevant — a complete no-op in that
    case, so behavior is unchanged for sources without Document Intelligence."""
    if not document_context:
        return ""
    lines = ["\nSUPPORTING DOCUMENTS (linked to the tables above via Document "
              "Intelligence — see rule 9c for how to use these):"]
    for doc in document_context:
        lines.append(f"\n### {doc.get('file_name', 'document')}")
        if doc.get("topics"):
            lines.append(f"Topics: {', '.join(doc['topics'])}")
        excerpt = doc.get("excerpt", "").strip()
        if excerpt:
            lines.append(excerpt)
    lines.append(
        "\nNot every document above is necessarily relevant to this question — "
        "they are simply all documents linked to the tables involved. After "
        "your Analysis section, on its own line, list ONLY the filenames (exactly "
        "as shown in the ### headers above) whose content you actually drew on "
        "to write this answer, e.g.:\n"
        "DOCS_USED: Underwriting_Guidelines.txt\n"
        "Use DOCS_USED: none if you didn't end up using any of them. "
        "This line is required and will be removed before the user sees it."
    )
    return "\n".join(lines) + "\n"


_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+((?:\"[^\"]+\"|\w+)(?:\.(?:\"[^\"]+\"|\w+))*)",
    re.IGNORECASE,
)


def _extract_table_refs_from_sql(sql: str) -> List[str]:
    """Best-effort fallback when the planner LLM's JSON omits table_refs for a
    query that nonetheless ran and returned real rows — pull table names
    straight out of FROM/JOIN clauses so a source is never silently dropped."""
    return [m.group(1).replace('"', "") for m in _TABLE_REF_RE.finditer(sql or "")]


def _build_sources(state: DialogState, used_documents: Optional[set] = None) -> list:
    """Dedup list of {type, name} for the tables and documents that fed this
    answer — table_refs from sql_queries (falling back to parsing the SQL
    itself when the LLM omitted table_refs), plus file_name from
    document_context.

    document_context holds every document linked (via KG "mentions" edges) to
    the tables retrieve_node selected — not necessarily documents the answer
    actually drew on. used_documents, when provided, is the set of file names
    the LLM self-reported actually using (see DOCS_USED parsing below) and is
    used to filter document_context down to just those. None means "no
    filtering info available" (e.g. no document_section was ever shown to the
    LLM for this branch) — falls back to including everything attached.

    Tables get the same "actually contributed" standard: sql_queries lists
    every query the planner produced, but a query that errored out never fed
    any data into the answer, so its table(s) are excluded — only queries with
    a matching, error-free query_result count as a real source."""
    succeeded_query_ids = {
        qr["query_id"] for qr in (state.get("query_results") or [])
        if not qr.get("error")
    }
    sources = []
    seen = set()
    for q in state.get("sql_queries") or []:
        if q.get("query_id") not in succeeded_query_ids:
            continue
        table_refs = q.get("table_refs") or _extract_table_refs_from_sql(q.get("sql", ""))
        for table in table_refs:
            key = ("table", table)
            if table and key not in seen:
                seen.add(key)
                sources.append({"type": "table", "name": table})
    for doc in state.get("document_context") or []:
        file_name = doc.get("file_name")
        if not file_name or (used_documents is not None and file_name not in used_documents):
            continue
        key = ("document", file_name)
        if key not in seen:
            seen.add(key)
            sources.append({"type": "document", "name": file_name})
    return sources


_DOCS_USED_RE = re.compile(r"^[ \t]*DOCS_USED:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)


def _extract_docs_used(llm_output: str) -> "tuple[str, Optional[set]]":
    """Strip the trailing DOCS_USED: line the LLM was instructed to append and
    return (cleaned_output, set_of_file_names_actually_used). Returns
    (llm_output, None) if the LLM didn't comply — callers should treat None as
    'unknown' and fall back to the permissive (include-everything) behavior
    rather than silently dropping legitimate sources."""
    m = _DOCS_USED_RE.search(llm_output)
    if not m:
        return llm_output, None
    cleaned = _DOCS_USED_RE.sub("", llm_output).rstrip()
    raw = m.group(1).strip()
    if not raw or raw.lower() in ("none", "n/a"):
        return cleaned, set()
    return cleaned, {n.strip() for n in re.split(r"[,|]", raw) if n.strip()}


# ── LLM call ──────────────────────────────────────────────────────────────────

_COST_PER_M = {
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-6":           (15.00, 75.00),
}


def _log_cost(node: str, model: str, usage) -> None:
    inp, out = usage.input_tokens, usage.output_tokens
    in_price, out_price = _COST_PER_M.get(model, (3.00, 15.00))
    cost = (inp * in_price + out * out_price) / 1_000_000
    logger.info(
        "COST %s [%s] in=%d out=%d  $%.5f",
        node, model, inp, out, cost,
    )


def _call_llm(system: str, user: str, model: str, temperature: float) -> str:
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    _log_cost("synthesize_node", model, msg.usage)
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
        # Safety net: discard plan_explanation if it still looks like raw JSON
        # (starts with '[' or '{', or contains "query_id":).
        if plan_explanation and (
            plan_explanation.lstrip()[:1] in ('[', '{')
            or '"query_id"' in plan_explanation[:300]
        ):
            logger.warning(
                "synthesize_node: plan_explanation looks like raw JSON — discarding"
            )
            plan_explanation = ""
        if plan_explanation:
            state["insights"] = plan_explanation
            # No document_section was ever shown to the LLM here — exclude
            # documents entirely rather than falling back to "include all".
            state["sources"] = _build_sources(state, used_documents=set())
            state["phase"] = "synthesize"
            return state

        # No SQL results — but if documents are linked to this datasource,
        # try answering from those alone rather than immediately giving up.
        document_context = state.get("document_context") or []
        if document_context:
            document_section = _build_document_section(document_context)
            try:
                answer = _call_llm(
                    _SYSTEM_PROMPT,
                    f"ORIGINAL QUESTION:\n{natural_query}\n{document_section}\n"
                    "No structured query results were available for this question. "
                    "Answer using ONLY the supporting documents above — do not "
                    "invent figures. If they don't contain enough information to "
                    "answer, say so plainly rather than guessing.",
                    getattr(config, "llm_model", "claude-haiku-4-5"),
                    getattr(config, "llm_temperature", 0.0),
                )
                if answer.strip():
                    cleaned, used_documents = _extract_docs_used(answer)
                    state["insights"] = cleaned
                    state["sources"] = _build_sources(state, used_documents=used_documents)
                    state["phase"] = "synthesize"
                    return state
            except Exception as exc:
                logger.warning("synthesize_node: document-only synthesis failed — %s", exc)

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
        state["sources"] = _build_sources(state, used_documents=set())
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
    haiku_model     = getattr(config, "plan_llm_model", "claude-haiku-4-5")
    history_section = _build_history_section(history, haiku_model)

    # ── Document context (Document Intelligence cross-modal linking) ──────────
    document_section = _build_document_section(state.get("document_context") or [])

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
    # document_section isn't trimmed by guard_synth_sections (it's small —
    # capped at a few documents x ~1200 chars each), but it IS counted in
    # base_chars so results_text/history_section still get trimmed correctly
    # against the true remaining budget.
    base_chars = len(system) + len(natural_query) + len(_USER_PROMPT) + len(document_section) + 50
    results_text, history_section = guard_synth_sections(
        system, results_text, history_section, base_chars
    )

    user_prompt = _USER_PROMPT.format(
        question=natural_query,
        history_section=history_section,
        results_text=results_text,
        document_section=document_section,
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

    llm_output, used_documents = _extract_docs_used(llm_output)

    # ── Append programmatic ## Data section ───────────────────────────────────
    data_section = _build_data_section(query_results)
    insights = llm_output.rstrip() + data_section

    # ── Glossary threshold context ────────────────────────────────────────────
    # Append a threshold/benchmark callout for any glossary term whose name
    # appears in the answer. Non-blocking — skipped silently if unavailable.
    try:
        import glossary_store as _gl_store
        glossary_terms = state.get("glossary_terms") or []
        threshold_lines = []
        answer_lower = insights.lower()
        for term in glossary_terms:
            thr = term.get("threshold")
            if not thr:
                continue
            term_name = term.get("name", "")
            # Check if this term or any synonym appears in the answer
            all_names = [term_name] + [s["synonym"] for s in (term.get("synonyms") or [])]
            if not any(n.lower() in answer_lower for n in all_names if n):
                continue
            parts = []
            unit = thr.get("unit", "")
            direction = thr.get("direction", "higher_is_better")
            benchmark = thr.get("benchmark_value")
            red = thr.get("threshold_red")
            amber = thr.get("threshold_amber")
            if red is not None:
                parts.append(f"red < {red}{unit}")
            if amber is not None:
                parts.append(f"amber < {amber}{unit}")
            if benchmark is not None:
                src = thr.get("benchmark_source", "industry")
                parts.append(f"{src} benchmark {benchmark}{unit}")
            if parts:
                arrow = "↑ higher is better" if direction == "higher_is_better" else "↓ lower is better"
                threshold_lines.append(f"**{term_name}** thresholds: {' · '.join(parts)} ({arrow})")
        if threshold_lines:
            insights = insights.rstrip() + "\n\n> **Thresholds & Benchmarks**  \n> " + "  \n> ".join(threshold_lines)
    except Exception:
        pass

    # ── KPI context callout ───────────────────────────────────────────────────
    # For each active KPI whose name appears in the answer, append its unit,
    # direction, and formula so the user understands what the number represents.
    try:
        active_kpis = state.get("active_kpis") or []
        kpi_lines = []
        answer_lower = insights.lower()
        for kpi in active_kpis:
            kpi_name = kpi.get("name", "")
            if not kpi_name or kpi_name.lower() not in answer_lower:
                continue
            parts = []
            unit      = kpi.get("unit", "")
            direction = kpi.get("direction", "up")
            nl        = kpi.get("nl_formula", "")
            sql       = kpi.get("sql_expression", "")
            arrow = "↑ higher is better" if direction == "up" else "↓ lower is better"
            if unit:
                parts.append(f"unit: **{unit}**")
            parts.append(arrow)
            if nl:
                parts.append(f"formula: _{nl}_")
            if sql:
                parts.append(f"`{sql}`")
            if parts:
                kpi_lines.append(f"**{kpi_name}** — " + " · ".join(parts))
        if kpi_lines:
            insights = insights.rstrip() + "\n\n> **KPI Definitions**  \n> " + "  \n> ".join(kpi_lines)
    except Exception:
        pass

    state["insights"] = insights
    state["sources"] = _build_sources(state, used_documents=used_documents)
    state["phase"] = "synthesize"
    return state
