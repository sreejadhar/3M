"""
token_guard — Estimate token counts and enforce budget limits before LLM calls.

Approximation: 1 token ≈ 4 characters (conservative for English/SQL text).
Hard budget:   180,000 input tokens (20k buffer under Claude's 200k limit).

Usage:
    from .token_guard import guard_plan_prompt, guard_synth_prompt, estimate_tokens

    # In plan_node:
    system, user = guard_plan_prompt(system, user, schema_context)

    # In synthesize_node:
    results_text, history_section = guard_synth_sections(
        system, results_text, history_section, base_chars
    )
"""
from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN: int = 4
_MAX_INPUT_TOKENS: int = 180_000
_MAX_INPUT_CHARS: int = _MAX_INPUT_TOKENS * _CHARS_PER_TOKEN  # 720,000 chars


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token per 4 characters of text."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _trim_section(text: str, budget_chars: int, label: str) -> str:
    """Trim text to budget_chars with a truncation notice appended."""
    if len(text) <= budget_chars:
        return text
    trimmed = text[:budget_chars]
    saved_tokens = (len(text) - budget_chars) // _CHARS_PER_TOKEN
    logger.warning(
        "token_guard: %s trimmed %d → %d chars (~%d tokens saved)",
        label, len(text), budget_chars, saved_tokens,
    )
    return trimmed + "\n\n... [content truncated to stay within token budget]"


def guard_plan_prompt(
    system: str,
    user: str,
    schema_context: str,
) -> Tuple[str, str]:
    """
    Enforce token budget for plan_node prompts.

    If total chars exceed budget, trim schema_context in the user prompt.
    schema_context must be the exact substring that appears in user (it is
    formatted in via _USER_PROMPT.format(schema_context=...) so this is safe).

    Returns (system, user) with schema trimmed if needed.
    """
    total = len(system) + len(user)
    if total <= _MAX_INPUT_CHARS:
        return system, user

    # Budget left after all other content is preserved
    other_chars = total - len(schema_context)
    schema_budget = max(_MAX_INPUT_CHARS - other_chars, 10_000)
    trimmed_schema = _trim_section(schema_context, schema_budget, "schema_context")
    user = user.replace(schema_context, trimmed_schema, 1)

    logger.warning(
        "token_guard: plan prompt over budget — was %d chars (~%dk tokens), schema trimmed",
        total, total // _CHARS_PER_TOKEN // 1000,
    )
    return system, user


def guard_synth_sections(
    system: str,
    results_text: str,
    history_section: str,
    base_chars: int,
) -> Tuple[str, str]:
    """
    Enforce token budget for synthesize_node by trimming results_text and
    history_section *before* they are formatted into the user prompt.

    base_chars: chars already committed (system + question + static template text).
    Returns (results_text, history_section) possibly trimmed.
    """
    total = base_chars + len(results_text) + len(history_section)
    if total <= _MAX_INPUT_CHARS:
        return results_text, history_section

    # Step 1: trim results first (most content)
    results_budget = max(_MAX_INPUT_CHARS - base_chars - len(history_section), 5_000)
    results_text = _trim_section(results_text, results_budget, "results_text")

    # Re-check
    total = base_chars + len(results_text) + len(history_section)
    if total <= _MAX_INPUT_CHARS:
        logger.warning(
            "token_guard: synth prompt trimmed results only (~%dk tokens total)",
            total // _CHARS_PER_TOKEN // 1000,
        )
        return results_text, history_section

    # Step 2: also trim history
    history_budget = max(_MAX_INPUT_CHARS - base_chars - len(results_text), 1_000)
    history_section = _trim_section(history_section, history_budget, "history_section")

    logger.warning(
        "token_guard: synth prompt trimmed results + history (~%dk tokens total)",
        (base_chars + len(results_text) + len(history_section)) // _CHARS_PER_TOKEN // 1000,
    )
    return results_text, history_section
