"""
Conformity Agent node: call Claude to generate human-readable recommendations
based on the detected conformity candidates.
"""
from __future__ import annotations

import json
import logging
from typing import List

from ..config import ConformityConfig
from ..state import Conformity, ConformityState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior data governance and data architecture expert reviewing the results
of an automated cross-knowledge-graph conformity analysis.

Your job is to read the conformity candidates detected between two or more knowledge
graphs and produce clear, actionable recommendations for how the graphs can be
stitched into a unified "super knowledge graph".

Guidelines:
- Group recommendations by match type (exact, fuzzy, property-jaccard).
- For fuzzy matches, point out likely reasons (naming conventions, pluralisation, etc.).
- For property-jaccard matches with different labels, suggest a canonical alias.
- Be concise but specific — name the actual nodes.
- End with a short "Recommended Actions" numbered list.
- Use markdown formatting with headings and tables.
"""

_USER_TEMPLATE = """\
## Knowledge Graph Summary
{kg_summary}

## Conformity Candidates ({total} total)
{candidate_table}

Please analyse these results and generate your recommendations.
"""


def recommend_node(state: ConformityState) -> ConformityState:
    logger.info("=== recommend_node ===")
    conformities: List[Conformity] = state.get("conformities") or []
    snapshots = state.get("kg_snapshots") or []
    config: ConformityConfig = state["config"]

    if not conformities:
        state["recommendations"] = (
            "No conformity candidates were found between the selected knowledge graphs. "
            "The graphs appear to have entirely distinct schemas — no stitching is recommended."
        )
        state["phase"] = "recommended"
        return state

    # Build KG summary
    kg_lines = []
    for s in snapshots:
        kg_lines.append(
            f"- **{s['kg_id']}**: {len(s['nodes'])} nodes, {len(s['edges'])} edges"
        )
    kg_summary = "\n".join(kg_lines)

    # Build candidate table (cap at max_conformities_in_prompt)
    cap = config.max_conformities_in_prompt
    subset = conformities[:cap]
    rows = ["| # | Type | KG-A | Node-A | KG-B | Node-B | Score | Jaccard |",
            "|---|------|------|--------|------|--------|-------|---------|"]
    for c in subset:
        rows.append(
            f"| {c['index']} | {c['match_type']} "
            f"| {c['kg_a_id'][:16]} | {c['node_a_label']} "
            f"| {c['kg_b_id'][:16]} | {c['node_b_label']} "
            f"| {c['score']:.2f} | {c['jaccard']:.2f} |"
        )
    if len(conformities) > cap:
        rows.append(f"| … | *(+{len(conformities) - cap} more not shown)* | | | | | | |")
    candidate_table = "\n".join(rows)

    user_msg = _USER_TEMPLATE.format(
        kg_summary=kg_summary,
        total=len(conformities),
        candidate_table=candidate_table,
    )

    try:
        import anthropic  # noqa: PLC0415
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.llm_model,
            max_tokens=2048,
            temperature=config.llm_temperature,
            messages=[
                {"role": "user", "content": _SYSTEM_PROMPT + "\n\n" + user_msg}
            ],
        )
        recommendations = response.content[0].text.strip()
    except Exception as exc:
        logger.exception("recommend_node: LLM call failed")
        state["errors"] = state.get("errors") or []
        state["errors"].append(f"recommend_node: LLM error — {exc}")
        # Provide a fallback summary
        exact   = state.get("exact_count", 0)
        fuzzy   = state.get("fuzzy_count", 0)
        jaccard = state.get("jaccard_count", 0)
        recommendations = (
            f"## Conformity Summary\n\n"
            f"Found **{len(conformities)}** conformity candidate(s): "
            f"{exact} exact, {fuzzy} fuzzy, {jaccard} property-structure matches.\n\n"
            f"*(LLM recommendation unavailable: {exc})*"
        )

    state["recommendations"] = recommendations
    state["phase"]            = "recommended"
    logger.info("recommend_node: recommendations generated (%d chars)", len(recommendations))
    return state
