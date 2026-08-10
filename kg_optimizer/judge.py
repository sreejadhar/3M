"""
LLM-as-judge answer scoring. No such scoring code exists elsewhere in the repo
(confirmed by exploration) — this is new. Uses llm_client.get_client(), the same
Anthropic/Bedrock-dispatching factory used by dialog_agent and kg_inference_engine,
so it works unmodified against either local dev (ANTHROPIC_API_KEY) or prod (Bedrock).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = (
    "You are grading a data-chat answer against a known-correct reference answer. "
    "Score correctness and completeness on a 1-5 scale:\n"
    "  5 = fully correct and complete\n"
    "  3 = partially correct or missing detail\n"
    "  1 = wrong or contradicts the reference\n"
    "Return ONLY a JSON object: {\"score\": <1-5 int>, \"reasoning\": \"<brief>\"}."
)


def score_with_gold(question: str, answer: str, gold_answer: str, model: str) -> float:
    from llm_client import get_client

    client = get_client()
    user_msg = (
        f"Question: {question}\n\n"
        f"Reference (correct) answer: {gold_answer}\n\n"
        f"Answer to grade: {answer}"
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            temperature=0.0,
            system=_JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text if resp.content else "{}"
        raw = re.sub(r'^```[a-z]*\s*', '', raw.strip())
        raw = re.sub(r'\s*```$', '', raw.strip())
        parsed = json.loads(raw)
        return max(1.0, min(5.0, float(parsed.get("score", 3))))
    except Exception as exc:
        logger.warning("Judge scoring failed for question %r: %s", question[:60], exc)
        return 3.0  # neutral fallback — don't let judge failures tank/inflate fitness


def score_no_gold(answer: Optional[str], had_error: bool, sql_count: int) -> float:
    """Heuristic fallback when no gold answer exists for a question: reward a
    non-empty, error-free answer backed by at least one executed query."""
    if had_error or not answer or not answer.strip():
        return 1.0
    if sql_count <= 0:
        return 2.0
    return 3.5


def score_answer(question: str, answer: Optional[str], gold_answer: Optional[str],
                 had_error: bool, sql_count: int, model: str) -> float:
    if gold_answer:
        return score_with_gold(question, answer or "", gold_answer, model)
    return score_no_gold(answer, had_error, sql_count)
