"""
Orchestrates on-demand, per-source business glossary generation, scoped to
the selected source only: NLP normalization -> canonical-term match (cheap,
exact-name lookup against terms already governed for ANY source, so two
sources with an identically-named column still share one term) -> LLM
fallback -> governed persistence (glossary_registry).

Deliberately does NOT do structural/semantic cross-source matching (comparing
this source's columns against every OTHER source's already-linked columns) —
that pool grows unboundedly as more sources get glossaried (thousands of rows
after normal use) and turned generation into a multi-minute-or-worse operation
dominated by pool hydration, for a source that should only take as long as its
own column count warrants. See glossary_matcher.py — still available, just no
longer called from here.

NOT part of the indexing pipeline — only invoked by
POST /metadata/sources/{source_id}/generate-glossary (orchestrator_api.py).

Confidence -> governance status mapping (per the approved plan):
  >= 0.9  -> approved   (canonical match, or a very strong LLM hit)
  0.6-0.9 -> candidate  (needs steward review before being treated as authoritative)
  <  0.6  -> draft      (weak signal, flagged for review)
  manual edit / steward approval -> approved, confidence=1.0 (handled by the API layer,
  not here — this module only ever CREATES candidate/draft/canonical terms)
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import metadata_catalog as _mc
import glossary_nlp as _nlp
import glossary_registry as _reg

logger = logging.getLogger(__name__)

# A wide table (40+ columns) asking for a term+definition+confidence per
# column in one response reliably blows past a small max_tokens budget and
# comes back truncated mid-JSON (stop_reason="max_tokens"), so the whole
# table's terms — entity included — got silently dropped. Chunking keeps
# every single call well under budget regardless of table width, and the
# per-table parallelism below (LLM calls are network-bound, not CPU-bound)
# is what actually keeps a multi-table source fast instead of paying each
# call's latency serially.
_MAX_COLS_PER_LLM_CALL = 20
_LLM_MAX_TOKENS = 4096
_LLM_CONCURRENCY = 6

_GLOSSARY_SYSTEM = """\
You are a business analyst annotating a data catalog. Given a database table name
and its columns (with data types and sample values), propose a business-friendly
GLOSSARY TERM and a one-sentence DEFINITION for the table itself and for each
column — the plain-English name and meaning a business user (not an engineer)
would recognize, not a restatement of the technical name or data type.

You will usually be given a BUSINESS and/or DOMAIN context line describing what
industry/function this data actually belongs to. Ground every interpretation in
that context. Many column/table name fragments and abbreviations are ambiguous
in general English (e.g. "CBI" can mean "Citizenship by Investment" on the open
web, or something entirely different inside a pharma/life-sciences dataset).
When a fragment or abbreviation could plausibly mean something else outside the
given business/domain, DO NOT default to the generic/most-common web meaning —
prefer the interpretation consistent with the stated business/domain, and lower
your confidence score instead of guessing. If no business/domain context is
given, treat any ambiguous abbreviation as low-confidence rather than picking
the first meaning that comes to mind.

Return ONLY a JSON object — no prose, no markdown fences.

For each item also return a "confidence" float between 0.0 and 1.0 reflecting how
confident you are that the term/definition is correct given the available
evidence (column name + sample values + business/domain context). Use lower
confidence when the column name is abbreviated, ambiguous, the sample values
don't clearly confirm your interpretation, or you had to guess without
business/domain context to anchor the meaning.

OUTPUT FORMAT:
{
  "entity": {"term": "<business name for the table>", "definition": "<one sentence>", "confidence": 0.0},
  "columns": [
    {"name": "<column_name>", "term": "<business term>", "definition": "<one sentence>", "confidence": 0.0}
  ]
}
"""


def _call_glossary_llm(table_name: str, col_specs: List[Dict], model: str,
                        domain: str = "", business: str = "") -> Dict[str, Any]:
    """Same call+parse pattern already proven in metadata_catalog.py's
    _call_enrich_llm: from llm_client import get_client, markdown-fence strip,
    brace-extraction JSON fallback, fail-soft on any error."""
    lines = []
    for c in col_specs:
        sv = c.get("sample_values") or []
        sv_str = ", ".join(repr(v) for v in sv[:20]) if sv else "(no samples)"
        lines.append(f"  {c['name']} ({c.get('data_type', '')}): samples=[{sv_str}]")

    context_lines = []
    if business:
        context_lines.append(f"Business: {business}")
    if domain:
        context_lines.append(f"Domain: {domain}")
    context_prefix = ("\n".join(context_lines) + "\n") if context_lines else ""
    user_msg = (
        f"{context_prefix}Table: {table_name}\nColumns:\n"
        + "\n".join(lines)
        + "\n\nPropose a business glossary term and definition for the table and each column."
    )

    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(
            model=model, max_tokens=_LLM_MAX_TOKENS, temperature=0.0,
            system=_GLOSSARY_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = msg.content[0].text if msg.content else ""
        if getattr(msg, "stop_reason", None) == "max_tokens":
            # Even after chunking to _MAX_COLS_PER_LLM_CALL this call still ran
            # out of budget (unusually long sample values/descriptions) — the
            # JSON is truncated and unparseable, so fail loud here rather than
            # silently dropping the whole chunk via a JSONDecodeError below.
            logger.warning(
                "generate_glossary: LLM response truncated (max_tokens) for table %r (%d cols)",
                table_name, len(col_specs),
            )
            return {}
    except Exception as exc:
        logger.warning("generate_glossary: LLM call failed for table %r — %s", table_name, exc)
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
    logger.warning("generate_glossary: could not parse LLM JSON for table %r", table_name)
    return {}


def _call_glossary_llm_for_table(
    table_name: str, col_specs_for_llm: List[Dict], entity_needs_llm: bool,
    model: str, domain: str = "", business: str = "",
) -> Dict[str, Any]:
    """Wraps _call_glossary_llm with column chunking so a wide table never
    exceeds the per-call token budget. Only the first chunk's "entity"
    annotation is kept (subsequent chunks still get asked for one — that's
    harmless, just discarded — simpler than varying the prompt per chunk)."""
    if not col_specs_for_llm and not entity_needs_llm:
        return {}
    if not col_specs_for_llm:
        return _call_glossary_llm(table_name, [], model, domain=domain, business=business)

    chunks = [col_specs_for_llm[i:i + _MAX_COLS_PER_LLM_CALL]
              for i in range(0, len(col_specs_for_llm), _MAX_COLS_PER_LLM_CALL)]
    merged: Dict[str, Any] = {"columns": []}
    for i, chunk in enumerate(chunks):
        result = _call_glossary_llm(table_name, chunk, model, domain=domain, business=business)
        if i == 0 and result.get("entity"):
            merged["entity"] = result["entity"]
        merged["columns"].extend(result.get("columns") or [])
    return merged


def _status_for_confidence(confidence: float) -> str:
    if confidence >= 0.9:
        return "approved"
    if confidence >= 0.6:
        return "candidate"
    return "draft"


def _build_canonical_index() -> Dict[str, List[Dict]]:
    """One-time bulk load mirroring find_term_by_canonical_key(status="approved")'s
    matching logic, keyed by canonical_key so pass 1 does zero per-column SQL
    round-trips for this — glossary_registry opens a fresh sqlite3 connection
    per call, and at hundreds of columns that connection overhead was the
    dominant cost of a run, not the actual matching work.

    Only approved terms are indexed — draft/candidate terms are unreviewed
    LLM guesses (possibly generated before business/domain context existed,
    or just wrong), and must never be silently adopted as "the" definition
    for a matching column in another source just because they happen to
    share a normalized name. Each source keeps generating its own
    context-grounded guess via the LLM until a term earns approval (auto, at
    >=0.9 confidence, or by a steward) — only then does it become reusable
    ground truth for everyone else."""
    idx: Dict[str, List[Dict]] = {}
    for t in _reg.list_terms(status="approved"):
        key = (t.get("canonical_key") or "").strip().lower()
        if key:
            idx.setdefault(key, []).append(t)
    return idx


def _lookup_canonical(idx: Dict[str, List[Dict]], canonical_key: str, domain: str) -> Optional[Dict]:
    if not canonical_key:
        return None
    candidates = idx.get(canonical_key.strip().lower())
    if not candidates:
        return None
    if domain:
        matches = [t for t in candidates if t.get("domain") == domain or not t.get("domain")]
        if not matches:
            return None
        matches.sort(key=lambda t: t.get("domain") == domain, reverse=True)
        return matches[0]
    return candidates[0]


def generate_glossary_for_source(
    source_id: str, model: Optional[str] = None, domain: str = "", business: str = "",
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, int]:
    """
    Runs the full discovery pipeline for every entity in source_id.
    progress_cb(stage, message) is called at each stage boundary for SSE
    progress reporting — optional, no-op if omitted.

    Returns {"columns_processed", "terms_created", "terms_linked", "needs_review"}.
    """
    def progress(stage: str, message: str) -> None:
        if progress_cb:
            try:
                progress_cb(stage, message)
            except Exception:
                pass

    model = model or os.environ.get("DIALOG_LLM_MODEL", "claude-haiku-4-5")
    stats = {"columns_processed": 0, "terms_created": 0, "terms_linked": 0, "needs_review": 0,
              "skipped_existing": 0}

    entities = _mc.list_entities(source_id)
    progress("normalize", f"Normalizing columns for {len(entities)} table(s)…")

    # Bulk-preload what pass 1 needs instead of one SQL round-trip per column:
    # glossary_registry opens a fresh sqlite3 connection on every call, so
    # get_asset_link()/find_term_by_canonical_key() called per-column (as this
    # used to do) meant ~2 connections x every column in the source — for a
    # ~1000-column source that's the dominant cost of a run, dwarfing the
    # actual LLM time. Both of these are read-only snapshots taken once,
    # which is safe because pass 1 never creates new canonical terms itself
    # (only pass 2 does, and it re-checks the live registry before creating).
    # Only approved (or manually-edited — update_term always forces
    # status='approved' for a manual edit, see glossary_registry.update_term)
    # links are permanently skipped on re-run. A link to a draft/candidate
    # term is an unreviewed guess — possibly generated before business/domain
    # context existed — and is treated as still eligible for regeneration, so
    # "regenerate" can actually replace a bad guess instead of being a no-op
    # forever once anything, right or wrong, has been linked once.
    source_assets = _reg.list_assets_for_source(source_id)
    already_governed = {
        (a["metadata_id"], a.get("attr_id", "")) for a in source_assets
        if a.get("term_match_method") == "manual"
        or (a.get("term_status") == "approved" and (a.get("link_domain") or "") == (domain or ""))
    }
    stale_links = [
        (a["metadata_id"], a.get("attr_id", "")) for a in source_assets
        if (a["metadata_id"], a.get("attr_id", "")) not in already_governed
    ]
    if stale_links:
        _reg.delete_stale_links(stale_links)
        progress("normalize", f"Re-evaluating {len(stale_links)} stale/unreviewed link(s)…")
    canonical_idx = _build_canonical_index()
    progress("normalize", f"{len(already_governed)} column(s) already governed, "
                          f"{sum(len(v) for v in canonical_idx.values())} approved canonical term(s) loaded…")

    # ── Pass 1: cheap, local, sequential — canonical-key matching (exact
    # normalized-name match against terms already governed for ANY source)
    # for every column. Deliberately no structural/semantic cross-source
    # matching here — that step compared against every other source's linked
    # columns, an unbounded and ever-growing pool, and dominated run time as
    # more sources got glossaried; generation is scoped to this source's own
    # columns. No network calls in this pass, so there's nothing to gain from
    # parallelizing it; what's left over per table (need_llm)
    # is queued for pass 2 instead of being sent to the LLM immediately, so
    # that pass 2 can run every table's LLM call concurrently. Matches found
    # here are batched into one write at the end instead of one link_asset()
    # call (= one connection) per match.
    table_jobs: List[Dict[str, Any]] = []
    pending_links: List[Dict[str, Any]] = []
    for ent_summary in entities:
        full = _mc.get_entity(ent_summary["metadata_id"])
        if not full:
            continue
        metadata_id = full["metadata_id"]
        table_name = full["table_name"]
        attrs = full.get("attributes", [])

        items: List[Dict] = [{
            "attr_id": "", "column_name": table_name, "data_type": "",
            "description": full.get("description", ""), "sample_values": [],
        }]
        items += [{
            "attr_id": a["attr_id"], "column_name": a["column_name"],
            "data_type": a.get("data_type", ""), "description": a.get("description", ""),
            "sample_values": a.get("top_values") or a.get("sample_values") or [],
        } for a in attrs]

        need_llm: List[Dict] = []
        for item in items:
            if (metadata_id, item["attr_id"]) in already_governed:
                # Already governed by a prior run (or a manual edit) — re-running discovery
                # must never re-score/relink it. Without this guard, a column that got an
                # llm_generated term on run 1 becomes a match for that same term's
                # canonical_key on run 2, and link_asset() unconditionally overwrites
                # confidence/match_method — silently laundering a low-confidence guess into
                # a 0.95 "canonical" match with no review.
                stats["skipped_existing"] += 1
                continue

            stats["columns_processed"] += 1
            normalized = _nlp.normalize_identifier(item["column_name"])

            canon = _lookup_canonical(canonical_idx, normalized, domain)
            if canon:
                pending_links.append({"term_id": canon["term_id"], "source_id": source_id,
                                       "metadata_id": metadata_id, "attr_id": item["attr_id"],
                                       "confidence": 0.95, "match_method": "canonical", "domain": domain})
                stats["terms_linked"] += 1
                continue

            need_llm.append({**item, "normalized_phrase": normalized})

        if need_llm:
            table_jobs.append({"metadata_id": metadata_id, "table_name": table_name, "need_llm": need_llm})

    if pending_links:
        _reg.bulk_link_assets(pending_links)
        progress("canonical", f"Linked {len(pending_links)} column(s) to existing terms by name…")

    # ── Pass 2: the network-bound part, run concurrently across tables.
    # This is what previously made a multi-table source take minutes — one
    # LLM round-trip at a time, serially, table after table. Running up to
    # _LLM_CONCURRENCY of them at once turns "sum of every call's latency"
    # into "roughly the slowest call's latency x (table_count / concurrency)".
    # Registry writes (SQLite) still happen back on this thread as each
    # future completes, so there's no concurrent-write contention.
    # Same-run cache for the write-time dedup re-check below: lets two tables
    # sharing a normalized name (e.g. both have "created_at") within THIS run
    # consolidate onto one term without querying the registry for it — and,
    # critically, is checked BEFORE falling back to an approved-only registry
    # lookup, so a freshly-grounded term from earlier in this same run is
    # preferred over reaching for an old approved term that might predate
    # today's business/domain context.
    created_this_run: Dict[str, Dict] = {}

    if table_jobs:
        progress("llm_generate",
                 f"Generating terms via LLM for {len(table_jobs)} table(s) "
                 f"(up to {_LLM_CONCURRENCY} concurrently)…")
        with ThreadPoolExecutor(max_workers=_LLM_CONCURRENCY) as pool_exec:
            future_to_job = {}
            for job in table_jobs:
                need_llm = job["need_llm"]
                col_specs_for_llm = [
                    {"name": i["column_name"], "data_type": i["data_type"], "sample_values": i["sample_values"]}
                    for i in need_llm if i["attr_id"]
                ]
                entity_needs_llm = any(i["attr_id"] == "" for i in need_llm)
                fut = pool_exec.submit(
                    _call_glossary_llm_for_table, job["table_name"], col_specs_for_llm,
                    entity_needs_llm, model, domain, business,
                )
                future_to_job[fut] = job

            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                metadata_id = job["metadata_id"]
                table_name = job["table_name"]
                need_llm = job["need_llm"]
                try:
                    llm_result = fut.result()
                except Exception as exc:
                    logger.warning("generate_glossary: LLM task failed for table %r — %s", table_name, exc)
                    llm_result = {}

                entity_needs_llm = any(i["attr_id"] == "" for i in need_llm)
                if entity_needs_llm:
                    e_ann = llm_result.get("entity") or {}
                    if e_ann.get("term"):
                        entity_item = next(i for i in need_llm if i["attr_id"] == "")
                        # Pass 1 ran canonical-key matching for every table BEFORE any
                        # LLM/term-creation happened, so two tables sharing a column/
                        # table-name fragment (e.g. both have a "created_at") both
                        # landed in need_llm instead of the second one matching the
                        # first's freshly-created term the way the old sequential
                        # one-table-at-a-time flow allowed. Re-check here, at write
                        # time — writes are already serialized to this thread as
                        # futures complete, so whichever table's result lands first
                        # wins the create, and every later duplicate links to it
                        # instead of spawning a near-duplicate term. Same-run cache
                        # first (today's context, already known-good), THEN an
                        # approved-only registry lookup — never an unreviewed
                        # draft/candidate term from a past run, which is exactly the
                        # stale-hallucination bug this is fixing.
                        key = entity_item["normalized_phrase"].strip().lower()
                        dup = created_this_run.get(key) or _reg.find_term_by_canonical_key(
                            entity_item["normalized_phrase"], domain=domain, status="approved")
                        if dup:
                            _reg.link_asset(dup["term_id"], source_id, metadata_id, "",
                                             confidence=0.95, match_method="canonical", domain=domain)
                            stats["terms_linked"] += 1
                        else:
                            conf = min(max(float(e_ann.get("confidence", 0.5) or 0.5), 0.0), 0.9)
                            status = _status_for_confidence(conf)
                            term = _reg.create_term(
                                e_ann["term"], canonical_key=entity_item["normalized_phrase"],
                                definition=e_ann.get("definition", ""), domain=domain,
                                status=status, confidence=conf, match_method="llm_generated",
                            )
                            _reg.link_asset(term["term_id"], source_id, metadata_id, "",
                                             confidence=conf, match_method="llm_generated", domain=domain)
                            created_this_run[key] = term
                            stats["terms_created"] += 1
                            if status != "approved":
                                stats["needs_review"] += 1

                col_ann_by_name = {c["name"]: c for c in (llm_result.get("columns") or []) if "name" in c}
                for item in need_llm:
                    if not item["attr_id"]:
                        continue
                    ann = col_ann_by_name.get(item["column_name"])
                    if not ann or not ann.get("term"):
                        continue
                    key = item["normalized_phrase"].strip().lower()
                    dup = created_this_run.get(key) or _reg.find_term_by_canonical_key(
                        item["normalized_phrase"], domain=domain, status="approved")
                    if dup:
                        _reg.link_asset(dup["term_id"], source_id, metadata_id, item["attr_id"],
                                         confidence=0.95, match_method="canonical", domain=domain)
                        stats["terms_linked"] += 1
                        continue
                    conf = min(max(float(ann.get("confidence", 0.5) or 0.5), 0.0), 0.9)
                    status = _status_for_confidence(conf)
                    term = _reg.create_term(
                        ann["term"], canonical_key=item["normalized_phrase"],
                        definition=ann.get("definition", ""), domain=domain,
                        status=status, confidence=conf, match_method="llm_generated",
                    )
                    _reg.link_asset(term["term_id"], source_id, metadata_id, item["attr_id"],
                                     confidence=conf, match_method="llm_generated", domain=domain)
                    created_this_run[key] = term
                    stats["terms_created"] += 1
                    if status != "approved":
                        stats["needs_review"] += 1
                progress("llm_generate", f"Processed {table_name}")

    progress("done", f"{stats['terms_created']} term(s) created, {stats['terms_linked']} column(s) linked "
                      f"to existing terms, {stats['needs_review']} need review, "
                      f"{stats['skipped_existing']} already governed (skipped)")
    logger.info("generate_glossary_for_source: source=%s %s", source_id, stats)
    return stats
