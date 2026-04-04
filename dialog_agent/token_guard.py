"""
token_guard — Exact token counting and budget enforcement before LLM calls.

Primary:  Anthropic count_tokens API (exact, no LLM cost).
Fallback: Conservative char-based estimate at 2.5 chars/token (schema/SQL
          content is denser than prose — the old 4 chars/token was too loose
          and allowed prompts over 200k through).

Hard budget: 185,000 input tokens (15k buffer under Claude's 200k limit).

Usage:
    # In plan_node:
    system, user = guard_plan_prompt(system, user, schema_context, model)

    # In synthesize_node:
    results_text, history_section = guard_synth_sections(
        system, results_text, history_section, base_chars
    )
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Conservative chars-per-token for schema/SQL content (prose is ~4, SQL ~2-3).
_CHARS_PER_TOKEN: float = 2.5

# Token budget: 15k below the hard 200k limit.
_MAX_INPUT_TOKENS: int = 185_000

# Char-based threshold (used when count_tokens API is unavailable).
_MAX_INPUT_CHARS: int = int(_MAX_INPUT_TOKENS * _CHARS_PER_TOKEN)  # ≈ 462,500


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: 1 token per 2.5 characters."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _count_tokens_exact(
    system: str,
    user: str,
    model: str,
) -> Optional[int]:
    """
    Call Anthropic's count_tokens API to get the exact input token count.
    Returns None on any failure (network error, old SDK version, etc.) so
    the caller can fall back to the char-based estimate.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        resp = client.messages.count_tokens(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.input_tokens
    except Exception as exc:
        logger.debug("token_guard: count_tokens API unavailable (%s) — using char estimate", exc)
        return None


def _trim_section(text: str, budget_chars: int, label: str) -> str:
    """Trim text to budget_chars with a truncation notice appended."""
    if len(text) <= budget_chars:
        return text
    trimmed = text[:budget_chars]
    logger.warning(
        "token_guard: %s trimmed %d → %d chars (~%d tokens removed)",
        label, len(text), budget_chars,
        int((len(text) - budget_chars) / _CHARS_PER_TOKEN),
    )
    return trimmed + "\n\n... [content truncated to stay within token budget]"


def guard_plan_prompt(
    system: str,
    user: str,
    schema_context: str,
    model: str = "claude-haiku-4-5-20251001",
) -> Tuple[str, str]:
    """
    Enforce the token budget for plan_node prompts using char-based estimation.

    We intentionally avoid the count_tokens API here: calling it sends the full
    schema prompt to Anthropic just to measure it, effectively doubling the
    billed input tokens for every plan_node call.  The conservative char estimate
    (2.5 chars/token for schema/SQL content) is accurate enough and costs nothing.

    schema_context must be the exact substring present in user — it is
    formatted in via _USER_PROMPT.format(schema_context=...) so str.replace
    finds it unambiguously.

    Returns (system, user) with schema trimmed if needed.
    """
    total_chars = len(system) + len(user)
    if total_chars <= _MAX_INPUT_CHARS:
        logger.debug(
            "token_guard: plan prompt OK — ~%d tokens (char estimate)",
            int(total_chars / _CHARS_PER_TOKEN),
        )
        return system, user

    other_chars   = total_chars - len(schema_context)
    schema_budget = max(_MAX_INPUT_CHARS - other_chars, 10_000)
    trimmed_schema = _trim_section(schema_context, schema_budget, "schema_context")
    user = user.replace(schema_context, trimmed_schema, 1)

    logger.warning(
        "token_guard: plan prompt over budget — was ~%dk tokens, schema trimmed %d → %d chars",
        int(total_chars / _CHARS_PER_TOKEN / 1000), len(schema_context), schema_budget,
    )
    return system, user


def guard_synth_sections(
    system: str,
    results_text: str,
    history_section: str,
    base_chars: int,
) -> Tuple[str, str]:
    """
    Enforce the token budget for synthesize_node by trimming results_text and
    history_section *before* they are formatted into the user prompt.

    base_chars: chars already committed (system + question + static template).
    Uses char-based estimate (count_tokens requires the fully assembled prompt).
    Returns (results_text, history_section) possibly trimmed.
    """
    total = base_chars + len(results_text) + len(history_section)
    if total <= _MAX_INPUT_CHARS:
        return results_text, history_section

    # Step 1: trim results first (largest section)
    results_budget = max(_MAX_INPUT_CHARS - base_chars - len(history_section), 5_000)
    results_text = _trim_section(results_text, results_budget, "results_text")

    total = base_chars + len(results_text) + len(history_section)
    if total <= _MAX_INPUT_CHARS:
        logger.warning(
            "token_guard: synth prompt trimmed results (~%dk tokens estimated)",
            int(total / _CHARS_PER_TOKEN / 1000),
        )
        return results_text, history_section

    # Step 2: also trim history
    history_budget = max(_MAX_INPUT_CHARS - base_chars - len(results_text), 1_000)
    history_section = _trim_section(history_section, history_budget, "history_section")

    logger.warning(
        "token_guard: synth prompt trimmed results + history (~%dk tokens estimated)",
        int((base_chars + len(results_text) + len(history_section)) / _CHARS_PER_TOKEN / 1000),
    )
    return results_text, history_section
