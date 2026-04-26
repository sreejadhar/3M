"""
End-to-end indexing pipeline for a document source.

Steps:
  1. Enumerate files via connector
  2. Compare checksums (skip unchanged)
  3. Parse each file (structural metadata + text window)
  4. Quality gate
  5. LLM semantic extraction (fingerprint)
  6. Persist fingerprint
  7. Cross-modal linking
  8. Update job progress
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .connectors import make_connector, file_type_from_extension
from .fingerprinter import quality_gate, extract_fingerprint
from .linker import run_linking
from .parsers import parse_file
from .store import UnstructuredStore

logger = logging.getLogger(__name__)

_WORKER_THREADS = int(os.environ.get("UNSTRUCTURED_WORKERS", "4"))


def run_index_job(job_id: str, source_id: str, store: UnstructuredStore,
                  metadata_api: str, force: bool = False) -> None:
    """
    Execute an indexing run for a source. Runs in a background thread.
    Updates job progress in the store throughout.
    When force=True, all existing assets are cleared first so every file
    is re-processed regardless of checksum.
    """
    source = store.get_source(source_id)
    if not source:
        store.update_job(job_id, status="error",
                         error_log=[f"Source {source_id} not found"],
                         finished_at=_now())
        return

    if force:
        cleared = store.clear_source_assets(source_id)
        logger.info("Job %s: force reindex — cleared %d existing assets", job_id, cleared)

    try:
        connection = _load_connection(source)
    except Exception as exc:
        store.update_job(job_id, status="error",
                         error_log=[str(exc)], finished_at=_now())
        return

    domain = source.get("domain", "")

    try:
        connector = make_connector(source["source_type"], connection)
    except ValueError as exc:
        store.update_job(job_id, status="error",
                         error_log=[str(exc)], finished_at=_now())
        return

    manifests = list(connector.enumerate())
    store.update_job(job_id, total_files=len(manifests))
    logger.info("Job %s: discovered %d files", job_id, len(manifests))

    error_log: List[str] = []
    processed = enriched = errors = 0

    def _process_one(manifest):
        nonlocal processed, enriched, errors
        try:
            file_type = file_type_from_extension(manifest.file_name)
            structural = {
                "size_bytes":       manifest.size_bytes,
                "created_at_file":  manifest.created_at,
                "modified_at_file": manifest.modified_at,
            }

            # Snapshot old version (if any) BEFORE upsert so we can diff later
            old_version_snapshot = _get_old_snapshot(manifest, source_id, store)

            asset_id = store.upsert_asset(
                source_id=source_id,
                file_path=manifest.path,
                file_name=manifest.file_name,
                file_type=file_type,
                checksum=manifest.checksum,
                structural=structural,
                remote_path=manifest.remote_path,
            )

            # Parse file
            parse_result = parse_file(manifest.path)
            structural["title"]  = parse_result.title
            structural["author"] = parse_result.author
            structural["page_count"] = parse_result.page_count

            # Quality gate
            qr = quality_gate(manifest.size_bytes, parse_result, manifest.file_name)
            if not qr.passes:
                logger.debug("Skip LLM for %s: %s", manifest.file_name, qr.reason)
                processed += 1
                store.update_job(job_id, processed=processed, enriched=enriched,
                                 errors=errors)
                return

            # LLM fingerprint
            fp = extract_fingerprint(
                parse_result=parse_result,
                domain_hint=domain,
                file_name=manifest.file_name,
            )
            fp.setdefault("title", parse_result.title or manifest.file_name)
            store.save_fingerprint(asset_id, fp)

            # If this was a version update, compute and persist the change summary
            if old_version_snapshot is not None:
                asset = store.get_asset(asset_id)
                old_ver = (asset.get("version_num") or 1) - 1
                if old_ver >= 1:
                    change_summary = _compute_change_summary(old_version_snapshot, fp)
                    store.save_version_change_summary(asset_id, old_ver, change_summary)

            # Cross-modal linking (non-blocking, best-effort)
            try:
                run_linking(asset_id, fp, store, metadata_api)
            except Exception as link_exc:
                logger.warning("Linking failed for %s: %s", asset_id, link_exc)

            processed += 1
            enriched  += 1
            store.update_job(job_id, processed=processed, enriched=enriched,
                             errors=errors)

        except Exception as exc:
            errors += 1
            error_log.append(f"{manifest.file_name}: {exc}")
            logger.warning("Error processing %s: %s", manifest.file_name, exc)
            store.update_job(job_id, processed=processed, enriched=enriched,
                             errors=errors, error_log=error_log)

    with ThreadPoolExecutor(max_workers=_WORKER_THREADS) as pool:
        futures = [pool.submit(_process_one, m) for m in manifests]
        for _ in as_completed(futures):
            pass  # progress tracked inside _process_one

    # Clean up temp files created by cloud connectors (e.g. gdrive downloads)
    if hasattr(connector, "cleanup"):
        connector.cleanup()

    store.touch_source_indexed(source_id)
    store.update_job(
        job_id,
        status="done" if not errors else "done_with_errors",
        processed=processed,
        enriched=enriched,
        errors=errors,
        error_log=error_log,
        finished_at=_now(),
    )
    logger.info("Job %s complete: %d processed, %d enriched, %d errors",
                job_id, processed, enriched, errors)


def _load_connection(source: dict) -> dict:
    import json
    raw = source.get("connection_json", "{}")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _get_old_snapshot(manifest, source_id: str,
                      store: UnstructuredStore) -> "dict | None":
    """
    Return the current enriched state of an asset before it is overwritten,
    or None if this is a new file or the file hasn't changed.
    Only called when we expect a potential version bump.
    """
    try:
        if manifest.remote_path:
            row = store._conn().execute(
                "SELECT * FROM unstructured_assets WHERE source_id=? AND remote_path=?",
                (source_id, manifest.remote_path),
            ).fetchone()
        else:
            row = store._conn().execute(
                "SELECT * FROM unstructured_assets WHERE source_id=? AND file_path=?",
                (source_id, manifest.path),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        if d.get("checksum") == manifest.checksum:
            return None  # file unchanged — no version bump
        return d
    except Exception:
        return None


def _compute_change_summary(old: dict, new_fp: dict) -> str:
    """
    Produce a human-readable change summary by diffing the old asset state
    against the newly computed fingerprint. No LLM call — pure structural diff.
    """
    parts: List[str] = []

    # Title change
    old_title = (old.get("title") or "").strip()
    new_title = (new_fp.get("title") or "").strip()
    if old_title and new_title and old_title != new_title:
        parts.append(f'Title changed from "{old_title}" to "{new_title}"')

    # Doc type change
    old_type = (old.get("doc_type") or "").strip()
    new_type = (new_fp.get("doc_type") or "").strip()
    if old_type and new_type and old_type != new_type:
        parts.append(f"Document type changed from {old_type} to {new_type}")

    # Topic changes
    import json as _json
    try:
        old_topics = set(_json.loads(old.get("topics_json") or "[]"))
    except Exception:
        old_topics = set()
    new_topics = set(new_fp.get("topics") or [])

    added   = new_topics - old_topics
    removed = old_topics - new_topics
    if added:
        parts.append("New topics: " + ", ".join(sorted(added)))
    if removed:
        parts.append("Removed topics: " + ", ".join(sorted(removed)))

    # Size change (significant = >10%)
    old_size = old.get("size_bytes") or 0
    new_size = new_fp.get("size_bytes") or 0
    if old_size and new_size:
        pct = abs(new_size - old_size) / old_size * 100
        if pct >= 10:
            direction = "grew" if new_size > old_size else "shrank"
            parts.append(f"File {direction} by {pct:.0f}%")

    if not parts:
        parts.append("Content updated (no structural changes detected)")

    return "; ".join(parts)
