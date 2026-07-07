"""Indexing pipeline: enumerate a source's files via its connector and persist them as assets."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import httpx

from .connectors import make_connector
from .embedder import embed_text
from .extractor import SUPPORTED_EXTENSIONS, ExtractionError, extract_text
from .ner import extract_entities
from .pii import detect_pii
from .store import DocStore
from .topics import extract_topics
from .xref import ORCHESTRATOR_URL, XrefUnavailable, fetch_schema, link_mentions, push_to_knowledge_graph

logger = logging.getLogger(__name__)

PROCESSING_STEPS = [
    ("extract_text", "Text Extraction"),
    ("embed", "Semantic Embeddings"),
    ("tag_topics", "Topic Tagging"),
    ("ner", "Named Entity Recognition"),
    ("pii", "PII Detection"),
    ("xref", "Cross-Modal Linking"),
    ("kg_bridge", "KG Bridge Mapping"),
]

# Files larger than this are indexed (metadata only) but not auto-processed
# during a bulk reindex — process them individually via Upload Document instead.
_AUTO_PROCESS_MAX_BYTES = 15 * 1024 * 1024


def _init_steps() -> list:
    return [{"key": k, "label": label, "status": "pending", "detail": None} for k, label in PROCESSING_STEPS]


def _set_step(steps: list, key: str, status: str, detail: str = None) -> list:
    for s in steps:
        if s["key"] == key:
            s["status"] = status
            s["detail"] = detail
    return steps


def run_index_job(job_id: str, source_id: str, store: DocStore) -> None:
    """Runs synchronously in a background thread; updates job + source status as it goes."""
    source = store.get_source(source_id)
    if not source:
        store.update_job(job_id, status="error", finished_at=store.now())
        return

    store.set_source_status(source_id, "indexing")
    is_local = source["connector_type"] == "local"
    processed = 0
    errors = 0
    try:
        connector = make_connector(source["connector_type"], source["config"])
        for manifest in connector.list_files():
            # Snapshot the prior checksum/status BEFORE upsert_asset overwrites
            # it, so we can tell whether this file actually changed since the
            # last time it was indexed.
            prior = store.get_prior_checksum(source_id, manifest.remote_id)
            try:
                asset_id = store.upsert_asset(
                    source_id=source_id,
                    remote_id=manifest.remote_id,
                    file_name=manifest.file_name,
                    size_bytes=manifest.size_bytes,
                    mime_type=manifest.mime_type,
                    checksum=manifest.checksum,
                    modified_at=manifest.modified_at,
                    local_path=manifest.remote_id if is_local else None,
                )
                processed += 1
            except Exception as exc:
                logger.warning("Failed to index %s: %s", manifest.file_name, exc)
                errors += 1
                store.update_job(job_id, processed=processed, errors=errors, total_files=processed + errors)
                continue
            store.update_job(job_id, processed=processed, errors=errors, total_files=processed + errors)

            # Skip reprocessing a file that already finished successfully and
            # whose checksum hasn't changed since last indexed — avoids
            # re-running extraction/embedding/NER/PII/xref/kg_bridge (and
            # re-downloading cloud files) on every reindex for unchanged
            # content. A missing checksum can't be compared reliably, so
            # those always reprocess; a prior 'error'/'running' status also
            # always retries.
            prior_checksum, prior_status = prior or (None, None)
            unchanged = (
                prior_checksum is not None
                and manifest.checksum is not None
                and prior_checksum == manifest.checksum
                and prior_status == "done"
            )
            if unchanged:
                continue

            # Run every connector type through the same extract/embed/tag
            # pipeline as an explicit upload, right away, so a reindex ends
            # with every file fully processed (not just enumerated). Local
            # files are already on disk; cloud files are downloaded to a
            # temp file first since extract_text needs a filesystem path.
            _maybe_autoprocess(asset_id, manifest, store, connector, is_local)

        store.update_job(job_id, status="done", total_files=processed + errors,
                          processed=processed, errors=errors, finished_at=store.now())
        store.set_source_status(source_id, "ready")
    except Exception as exc:
        logger.exception("Index job failed for source %s", source_id)
        store.update_job(job_id, status="error", finished_at=store.now())
        store.set_source_status(source_id, "error", error_message=str(exc))


def _maybe_autoprocess(asset_id: str, manifest, store: DocStore, connector, is_local: bool) -> None:
    """Runs process_uploaded_asset for an indexed file, unless it's a type
    extract_text doesn't support or too large to process inline during a
    bulk reindex — those are marked 'skipped' with a clear reason instead
    of silently doing nothing. Cloud files are downloaded to a temp path
    first, since extract_text requires a filesystem path; the temp file is
    removed afterwards regardless of success or failure."""
    ext = Path(manifest.file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        store.update_asset_processing(asset_id, status="skipped", steps=[
            {"key": "skip", "label": "Processing", "status": "error",
             "detail": f"unsupported file type for text extraction: {ext or '(none)'}"},
        ])
        return
    if manifest.size_bytes > _AUTO_PROCESS_MAX_BYTES:
        store.update_asset_processing(asset_id, status="skipped", steps=[
            {"key": "skip", "label": "Processing", "status": "error",
             "detail": f"file too large to auto-process ({manifest.size_bytes / 1_048_576:.1f} MB) "
                       f"— re-upload it via Upload Document to process it individually"},
        ])
        return

    if is_local:
        process_uploaded_asset(asset_id, manifest.remote_id, store)
        return

    try:
        data = connector.read_bytes(manifest.remote_id)
    except Exception as exc:
        logger.warning("Failed to download %s for auto-processing: %s", manifest.file_name, exc)
        store.update_asset_processing(asset_id, status="skipped", steps=[
            {"key": "skip", "label": "Processing", "status": "error",
             "detail": f"could not download file from source: {exc}"},
        ])
        return

    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        process_uploaded_asset(asset_id, tmp_path, store)
    finally:
        os.remove(tmp_path)


def process_uploaded_asset(asset_id: str, local_path: str, store: DocStore) -> None:
    """Runs the full processing pipeline for a single asset: text extraction
    → semantic embeddings → topic tagging → named entity recognition → PII
    detection → cross-modal linking. Updates processing_steps on the asset
    after every step so the frontend can poll and show live progress. Never
    raises — errors are recorded on the failing step instead."""
    steps = _init_steps()
    store.update_asset_processing(asset_id, status="running", steps=steps)

    try:
        _set_step(steps, "extract_text", "running")
        store.update_asset_processing(asset_id, steps=steps)
        text = extract_text(local_path)
        _set_step(steps, "extract_text", "done", f"{len(text):,} characters extracted")
        store.update_asset_processing(asset_id, steps=steps, extracted_text=text)
    except ExtractionError as exc:
        _set_step(steps, "extract_text", "error", str(exc))
        store.update_asset_processing(asset_id, status="error", steps=steps)
        return
    except Exception as exc:
        logger.exception("Text extraction failed for asset %s", asset_id)
        _set_step(steps, "extract_text", "error", str(exc))
        store.update_asset_processing(asset_id, status="error", steps=steps)
        return

    try:
        _set_step(steps, "embed", "running")
        store.update_asset_processing(asset_id, steps=steps)
        vector, model_label = embed_text(text)
        _set_step(steps, "embed", "done", f"{len(vector)}-dim vector ({model_label})")
        store.update_asset_processing(asset_id, steps=steps, embedding_model=model_label,
                                       embedding_dims=len(vector))
    except Exception as exc:
        logger.exception("Embedding failed for asset %s", asset_id)
        _set_step(steps, "embed", "error", str(exc))
        store.update_asset_processing(asset_id, status="error", steps=steps)
        return

    try:
        _set_step(steps, "tag_topics", "running")
        store.update_asset_processing(asset_id, steps=steps)
        topics = extract_topics(text)
        _set_step(steps, "tag_topics", "done", f"{len(topics)} topics")
        store.update_asset_processing(asset_id, steps=steps, topics=topics)
    except Exception as exc:
        logger.exception("Topic tagging failed for asset %s", asset_id)
        _set_step(steps, "tag_topics", "error", str(exc))
        store.update_asset_processing(asset_id, status="error", steps=steps)
        return

    # Named entity recognition only starts once topic tagging has finished —
    # it never runs ahead of or in place of the earlier steps.
    try:
        _set_step(steps, "ner", "running")
        store.update_asset_processing(asset_id, steps=steps)
        entities = extract_entities(text)
        _set_step(steps, "ner", "done", f"{len(entities)} entities")
        store.update_asset_processing(asset_id, steps=steps, entities=entities)
    except Exception as exc:
        logger.exception("Named entity recognition failed for asset %s", asset_id)
        _set_step(steps, "ner", "error", str(exc))
        store.update_asset_processing(asset_id, status="error", steps=steps)
        return

    # PII detection only starts once named entity recognition has finished.
    try:
        _set_step(steps, "pii", "running")
        store.update_asset_processing(asset_id, steps=steps)
        pii_findings = detect_pii(text)
        _set_step(steps, "pii", "done", f"{len(pii_findings)} PII items found" if pii_findings else "no PII found")
        store.update_asset_processing(asset_id, steps=steps, pii_findings=pii_findings)
    except Exception as exc:
        logger.exception("PII detection failed for asset %s", asset_id)
        _set_step(steps, "pii", "error", str(exc))
        store.update_asset_processing(asset_id, status="error", steps=steps)
        return

    # Cross-modal linking only starts once PII detection has finished. It
    # needs at least one structured source linked to this document source —
    # if none is configured, that's a clean skip, not an error (everything
    # before it already succeeded). Each linked source is matched against
    # independently — a document can genuinely relate to more than one
    # database, and a table name collision between two linked sources must
    # not cause a match to land against the wrong one.
    linked_source_ids = []
    try:
        asset = store.get_asset(asset_id)
        source = store.get_source(asset["source_id"]) if asset else None
        linked_source_ids = (source.get("linked_source_ids") or []) if source else []

        if not linked_source_ids:
            _set_step(steps, "xref", "skipped", "no linked database source configured for this document source")
            store.update_asset_processing(asset_id, status="done", steps=steps)
        else:
            _set_step(steps, "xref", "running")
            store.update_asset_processing(asset_id, steps=steps)

            all_links = []
            per_source_counts = []
            for sid in linked_source_ids:
                try:
                    schema = fetch_schema(sid)
                    links = link_mentions(entities, topics, schema)
                except XrefUnavailable as exc:
                    logger.warning("xref: skipping unreachable source %s for asset %s — %s", sid, asset_id, exc)
                    continue
                for link in links:
                    all_links.append({**link, "source_id": sid})
                per_source_counts.append(f"{len(links)} in {sid[:8]}")
                if links:
                    push_to_knowledge_graph(sid, asset_id, asset["file_name"], links)

            detail = f"{len(all_links)} links across {len(linked_source_ids)} database(s)"
            if per_source_counts:
                detail += " (" + ", ".join(per_source_counts) + ")"
            _set_step(steps, "xref", "done", detail)
            store.update_asset_processing(asset_id, status="done", steps=steps, xref_links=all_links)
    except Exception as exc:
        logger.exception("Cross-modal linking failed for asset %s", asset_id)
        _set_step(steps, "xref", "error", str(exc))
        store.update_asset_processing(asset_id, status="error", steps=steps)
        return

    # KG Bridge Mapping only starts once cross-modal linking has finished. It
    # needs at least 2 linked structured sources to have anything to bridge —
    # a single linked source (or none) is a clean skip, not an error.
    try:
        if len(linked_source_ids) < 2:
            _set_step(steps, "kg_bridge", "skipped",
                      "fewer than 2 linked structured sources - nothing to cross-map")
            store.update_asset_processing(asset_id, status="done", steps=steps)
        else:
            _set_step(steps, "kg_bridge", "running")
            store.update_asset_processing(asset_id, steps=steps)
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    f"{ORCHESTRATOR_URL}/kg-bridges/infer",
                    json={"kg_ids": linked_source_ids},
                )
                resp.raise_for_status()
                result = resp.json()
            if result.get("status") == "skipped":
                _set_step(steps, "kg_bridge", "skipped", result.get("reason", "skipped"))
            else:
                _set_step(steps, "kg_bridge", "done",
                          f"inference queued across {result.get('kg_count', len(linked_source_ids))} "
                          f"linked KGs ({result.get('pairs', 0)} pairs)")
            store.update_asset_processing(asset_id, status="done", steps=steps)
    except Exception as exc:
        # Best-effort - a bridge-mapping failure never rolls back the
        # successful extract/embed/ner/pii/xref work already recorded above.
        logger.warning("kg_bridge: inference request failed for asset %s - %s", asset_id, exc)
        _set_step(steps, "kg_bridge", "error", str(exc))
        store.update_asset_processing(asset_id, status="done", steps=steps)
