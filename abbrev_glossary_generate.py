"""
Orchestrates on-demand, per-source abbreviation glossary generation, scoped
to the selected source only: abbreviation-candidate detection (column names
+ stored categorical values) -> abbreviation match (cheap, exact lookup
against abbreviations already governed for ANY source within the same
domain) -> LLM fallback for the full form -> governed persistence
(abbrev_glossary_registry).

Deliberately narrower than glossary_generate.py's business-glossary
discovery: instead of asking the LLM to name every column, this only fires
for tokens that actually LOOK like abbreviations (short, all-caps column
name fragments, or short all-caps values inside a small categorical set) —
most columns/values are not abbreviations and must never be sent to the LLM
on that basis alone.

NOT part of the indexing pipeline — only invoked by
POST /metadata/sources/{source_id}/generate-abbreviation-glossary
(orchestrator_api.py).

Confidence -> governance status mapping (same as glossary_generate.py):
  >= 0.9  -> approved   (very strong/unambiguous LLM hit)
  0.6-0.9 -> candidate  (needs steward review before being treated as authoritative)
  <  0.6  -> draft      (weak/ambiguous signal, flagged for review)
  manual edit / steward approval -> approved, confidence=1.0 (handled by the API layer)
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import metadata_catalog as _mc
import abbrev_glossary_registry as _reg

logger = logging.getLogger(__name__)

_LLM_MAX_TOKENS = 4096
_LLM_CONCURRENCY = 6
_MAX_ABBREVS_PER_LLM_CALL = 20

# ── Abbreviation-candidate detection ────────────────────────────────────────

_MIN_ABBREV_LEN = 2
_MAX_ABBREV_LEN = 6
_CATEGORICAL_MAX_UNIQUE = 15
_CATEGORICAL_MAX_LEN = 40

# Common non-abbreviation short all-caps tokens that show up as literal
# stored values/column-name fragments (booleans, currency/unit codes we
# don't want to send to an LLM as if they were ambiguous business jargon).
_ABBREV_STOPLIST = {
    "id", "ids", "no", "yes", "n", "y", "na", "nil", "null", "tbd", "usd",
    "eur", "gbp", "inr", "url", "uri", "api", "sql", "csv", "json", "xml",
}

_WORD_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _split_identifier_fragments(name: str) -> List[str]:
    """Split a column/table identifier on underscores/camelCase boundaries,
    e.g. 'TTS_amount' -> ['TTS', 'amount'], 'ttsAmount' -> ['tts', 'Amount']."""
    parts: List[str] = []
    for chunk in re.split(r"[_\-\s]+", name):
        if not chunk:
            continue
        parts.extend(m.group(0) for m in _WORD_SPLIT_RE.finditer(chunk))
    return parts


def _looks_like_abbreviation(token: str) -> bool:
    """True when *token* itself (not the whole identifier/value) looks like
    a genuine abbreviation candidate: short, alphabetic, ALL CAPS (the
    strongest signal an author/system actually intended it as an
    abbreviation rather than a short common word)."""
    if not (_MIN_ABBREV_LEN <= len(token) <= _MAX_ABBREV_LEN):
        return False
    if not token.isalpha():
        return False
    if not token.isupper():
        return False
    if token.lower() in _ABBREV_STOPLIST:
        return False
    return True


def _column_name_abbrev_candidates(column_name: str) -> List[str]:
    """Abbreviation-looking fragments of a column's own name, e.g.
    'TTS_amount' -> ['TTS']. Deduplicated, order-preserving."""
    seen: List[str] = []
    for frag in _split_identifier_fragments(column_name):
        if _looks_like_abbreviation(frag) and frag not in seen:
            seen.append(frag)
    return seen


def _looks_categorical(values: List[Any]) -> bool:
    """Mirrors glossary_generate.py's _looks_categorical: True when *values*
    looks like a small fixed set of short tokens rather than free text/IDs."""
    vals = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not vals:
        return False
    if len(set(vals)) > _CATEGORICAL_MAX_UNIQUE:
        return False
    return all(len(v) <= _CATEGORICAL_MAX_LEN for v in vals)


def _value_abbrev_candidates(values: List[Any]) -> List[str]:
    """Abbreviation-looking DISTINCT stored values from a small categorical
    set, e.g. a department column whose values are ['IT', 'HR', 'Finance']
    yields ['IT', 'HR'] (Finance is not all-caps/short so it's left alone —
    it's presumably already a readable business term)."""
    if not _looks_categorical(values):
        return []
    seen: List[str] = []
    for v in values:
        s = str(v).strip() if v is not None else ""
        if _looks_like_abbreviation(s) and s not in seen:
            seen.append(s)
    return seen


# ── LLM full-form resolution ─────────────────────────────────────────────────

_ABBREV_SYSTEM = """\
You are a data analyst annotating a data catalog. You will be given a list of
short ABBREVIATION CANDIDATES found either in column names or as stored
values inside a database column, each with the table/column it came from and
a few sibling sample values from that same column for context.

For each candidate, decide whether it is genuinely an abbreviation/acronym
that should be expanded for a business user, and if so propose its FULL FORM
and a one-sentence DEFINITION.

You will usually be given a BUSINESS and/or DOMAIN context line describing
what industry/function this data actually belongs to. Ground every
interpretation in that context. Many short codes are ambiguous in general
English (e.g. "PR" can mean "Public Relations", "Performance Rating", or
"Puerto Rico" depending on the table it's in). When a candidate could
plausibly mean something else outside the given business/domain, prefer the
interpretation consistent with the column's own name/purpose and the stated
business/domain, and LOWER your confidence score instead of guessing. If the
candidate is not actually an abbreviation (e.g. a real word, a currency code
you should skip, an ID), set "is_abbreviation": false and omit full_form.

Return ONLY a JSON object — no prose, no markdown fences.

For each item return a "confidence" float between 0.0 and 1.0 reflecting how
confident you are the full form is correct given the available evidence
(column name/purpose + sibling sample values + business/domain context). Use
lower confidence when the code is ambiguous or you had to guess without
business/domain context to anchor the meaning.

OUTPUT FORMAT:
{
  "candidates": [
    {"abbreviation": "<as given>", "is_abbreviation": true,
     "full_form": "<expanded form>", "definition": "<one sentence>", "confidence": 0.0}
  ]
}
"""


def _call_abbrev_llm(items: List[Dict], model: str, domain: str = "", business: str = "") -> Dict[str, Any]:
    """Same call+parse pattern as glossary_generate.py's _call_glossary_llm."""
    lines = []
    for it in items:
        sv = it.get("sibling_values") or []
        sv_str = ", ".join(repr(v) for v in sv[:15]) if sv else "(no sibling samples)"
        lines.append(
            f"  abbreviation={it['abbreviation']!r} table={it['table_name']!r} "
            f"column={it['column_name']!r} sibling_values=[{sv_str}]"
        )

    context_lines = []
    if business:
        context_lines.append(f"Business: {business}")
    if domain:
        context_lines.append(f"Domain: {domain}")
    context_prefix = ("\n".join(context_lines) + "\n") if context_lines else ""
    user_msg = (
        f"{context_prefix}Abbreviation candidates:\n" + "\n".join(lines)
        + "\n\nFor each, decide if it's a genuine abbreviation and propose its full form and definition."
    )

    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model, max_tokens=_LLM_MAX_TOKENS, temperature=0.0,
            system=_ABBREV_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text if msg.content else ""
        if getattr(msg, "stop_reason", None) == "max_tokens":
            logger.warning("generate_abbreviations: LLM response truncated (max_tokens) for %d candidate(s)",
                            len(items))
            return {}
    except Exception as exc:
        logger.warning("generate_abbreviations: LLM call failed — %s", exc)
        return {}

    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s != -1 and e != -1:
            try:
                return json.loads(cleaned[s:e + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("generate_abbreviations: could not parse LLM JSON")
    return {}


def _status_for_confidence(confidence: float) -> str:
    if confidence >= 0.9:
        return "approved"
    if confidence >= 0.6:
        return "candidate"
    return "draft"


def _build_abbrev_index() -> Dict[str, List[Dict]]:
    """One-time bulk load of every approved abbreviation term, keyed by
    lower(abbreviation) — mirrors glossary_generate.py's _build_canonical_index."""
    idx: Dict[str, List[Dict]] = {}
    for t in _reg.list_terms(status="approved"):
        key = (t.get("abbreviation") or "").strip().lower()
        if key:
            idx.setdefault(key, []).append(t)
    return idx


def _lookup_abbrev(idx: Dict[str, List[Dict]], abbreviation: str, domain: str) -> Optional[Dict]:
    candidates = idx.get(abbreviation.strip().lower())
    if not candidates:
        return None
    if domain:
        matches = [t for t in candidates if t.get("domain") == domain or not t.get("domain")]
        if not matches:
            return None
        matches.sort(key=lambda t: t.get("domain") == domain, reverse=True)
        return matches[0]
    return candidates[0]


def _sample_column_live(connector: Any, schema_name: str, table_name: str,
                         column_name: str) -> List[Any]:
    """Mirrors glossary_generate.py's _sample_column_live — on-demand
    profiling for a column that was never sampled at index time."""
    try:
        stats = connector.get_column_stats(schema_name, table_name, column_name, 10_000)
        return [v for v in (stats.get("top_values") or []) if v is not None]
    except Exception as exc:
        logger.warning("generate_abbreviations: live sampling failed for %s.%s.%s — %s",
                        schema_name, table_name, column_name, exc)
        return []


def generate_abbreviations_for_source(
    source_id: str, model: Optional[str] = None, domain: str = "", business: str = "",
    progress_cb: Optional[Callable[[str, str], None]] = None,
    connector_factory: Optional[Callable[[], Any]] = None,
) -> Dict[str, int]:
    """
    Runs the full abbreviation-discovery pipeline for every entity in
    source_id. progress_cb(stage, message) is called at each stage boundary
    for SSE progress reporting — optional, no-op if omitted.

    Returns {"candidates_found", "terms_created", "terms_linked", "needs_review",
             "skipped_existing"}.
    """
    def progress(stage: str, message: str) -> None:
        if progress_cb:
            try:
                progress_cb(stage, message)
            except Exception:
                pass

    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5")
    stats = {"candidates_found": 0, "terms_created": 0, "terms_linked": 0,
              "needs_review": 0, "skipped_existing": 0}

    _connector_state: Dict[str, Any] = {"built": False, "connector": None}

    def get_connector() -> Optional[Any]:
        if not _connector_state["built"]:
            _connector_state["built"] = True
            if connector_factory:
                try:
                    _connector_state["connector"] = connector_factory()
                except Exception as exc:
                    logger.warning("generate_abbreviations: could not build live connector for %s — %s",
                                    source_id, exc)
        return _connector_state["connector"]

    entities = _mc.list_entities(source_id)
    progress("scan", f"Scanning {len(entities)} table(s) for abbreviation candidates…")

    source_assets = _reg.list_assets_for_source(source_id)
    already_governed = {
        (a["metadata_id"], a.get("attr_id", "")) for a in source_assets
        if a.get("term_match_method") == "manual"
        or (a.get("term_status") == "approved" and a.get("match_method") != "canonical"
            and (a.get("link_domain") or "") == (domain or ""))
    }
    stale_links = [
        (a["metadata_id"], a.get("attr_id", "")) for a in source_assets
        if (a["metadata_id"], a.get("attr_id", "")) not in already_governed
    ]
    if stale_links:
        _reg.delete_stale_links(stale_links)
        progress("scan", f"Re-evaluating {len(stale_links)} stale/unreviewed link(s)…")
    abbrev_idx = _build_abbrev_index()

    # ── Pass 1: local candidate detection + cheap canonical matching ───────
    pending_links: List[Dict[str, Any]] = []
    need_llm: List[Dict[str, Any]] = []

    for ent_summary in entities:
        full = _mc.get_entity(ent_summary["metadata_id"])
        if not full:
            continue
        metadata_id = full["metadata_id"]
        table_name = full["table_name"]
        schema_name = full.get("schema_name", "")
        attrs = full.get("attributes", [])

        for attr in attrs:
            attr_id = attr["attr_id"]
            column_name = attr["column_name"]
            if (metadata_id, attr_id) in already_governed:
                continue

            samples = attr.get("top_values") or attr.get("sample_values") or []
            if not samples:
                connector = get_connector()
                if connector:
                    samples = _sample_column_live(connector, schema_name, table_name, column_name)

            candidates: List[str] = list(_column_name_abbrev_candidates(column_name))
            candidates += [c for c in _value_abbrev_candidates(samples) if c not in candidates]
            if not candidates:
                continue

            for abbrev in candidates:
                stats["candidates_found"] += 1
                canon = _lookup_abbrev(abbrev_idx, abbrev, domain)
                if canon:
                    pending_links.append({
                        "term_id": canon["term_id"], "source_id": source_id,
                        "metadata_id": metadata_id, "attr_id": attr_id,
                        "confidence": 0.95, "match_method": "canonical", "domain": domain,
                    })
                    stats["terms_linked"] += 1
                    continue
                need_llm.append({
                    "abbreviation": abbrev, "table_name": table_name, "column_name": column_name,
                    "metadata_id": metadata_id, "attr_id": attr_id,
                    "sibling_values": [v for v in samples if str(v).strip() != abbrev][:15],
                })

    if pending_links:
        _reg.bulk_link_assets(pending_links)
        progress("canonical", f"Linked {len(pending_links)} column(s) to existing abbreviation terms…")

    # ── Pass 2: LLM full-form resolution, chunked + concurrent ─────────────
    created_this_run: Dict[str, Dict] = {}

    if need_llm:
        chunks = [need_llm[i:i + _MAX_ABBREVS_PER_LLM_CALL]
                  for i in range(0, len(need_llm), _MAX_ABBREVS_PER_LLM_CALL)]
        progress("llm_generate",
                 f"Resolving {len(need_llm)} abbreviation candidate(s) via LLM "
                 f"(up to {_LLM_CONCURRENCY} concurrently)…")
        with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool_exec:
            future_to_chunk = {
                pool_exec.submit(_call_abbrev_llm, chunk, model, domain, business): chunk
                for chunk in chunks
            }
            for fut in as_completed(future_to_chunk):
                chunk = future_to_chunk[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.warning("generate_abbreviations: LLM task failed — %s", exc)
                    result = {}

                ann_by_abbrev: Dict[str, Dict] = {}
                for ann in result.get("candidates") or []:
                    if ann.get("abbreviation"):
                        ann_by_abbrev[ann["abbreviation"]] = ann

                for item in chunk:
                    ann = ann_by_abbrev.get(item["abbreviation"])
                    if not ann or not ann.get("is_abbreviation", True) or not ann.get("full_form"):
                        continue
                    key = item["abbreviation"].strip().lower()
                    dup = created_this_run.get(key) or _reg.find_term_by_abbreviation(
                        item["abbreviation"], domain=domain, status="approved")
                    if dup:
                        _reg.link_asset(dup["term_id"], source_id, item["metadata_id"], item["attr_id"],
                                         confidence=0.95, match_method="canonical", domain=domain)
                        stats["terms_linked"] += 1
                        continue
                    conf = min(max(float(ann.get("confidence", 0.5) or 0.5), 0.0), 0.9)
                    status = _status_for_confidence(conf)
                    term = _reg.create_term(
                        item["abbreviation"], full_form=ann.get("full_form", ""),
                        definition=ann.get("definition", ""), domain=domain,
                        status=status, confidence=conf, match_method="llm_generated",
                        sample_values_hint=json.dumps(item["sibling_values"][:10], default=str),
                        changed_by="abbrev_glossary_generate",
                    )
                    _reg.link_asset(term["term_id"], source_id, item["metadata_id"], item["attr_id"],
                                     confidence=conf, match_method="llm_generated", domain=domain)
                    created_this_run[key] = term
                    stats["terms_created"] += 1
                    if status != "approved":
                        stats["needs_review"] += 1
                progress("llm_generate", f"Processed {len(chunk)} candidate(s)")

    progress("done", f"{stats['terms_created']} term(s) created, {stats['terms_linked']} column(s) linked "
                      f"to existing terms, {stats['needs_review']} need review")
    logger.info("generate_abbreviations_for_source: source=%s %s", source_id, stats)
    return stats
