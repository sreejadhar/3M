"""
Expensive-tier build cache — avoids rebuilding the ontology + KG for every GA
individual. Keyed by a stable hash of the expensive-tier genes (see genome.py)
plus the source id.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from kg_optimizer.genome import Genome, expensive_key


def expensive_hash(source_id: str, genome: Genome) -> str:
    payload = {"source_id": source_id, **expensive_key(genome)}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class CacheEntry:
    kg_id: str
    hash: str
    source_id: str
    meta: Dict[str, Any]


class BuildCache:
    """JSON-file-backed manifest: expensive_hash -> built kg_id."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.manifest_path = os.path.join(cache_dir, "manifest.json")
        os.makedirs(cache_dir, exist_ok=True)
        self._manifest: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, indent=2)

    def get(self, source_id: str, genome: Genome) -> Optional[CacheEntry]:
        h = expensive_hash(source_id, genome)
        entry = self._manifest.get(h)
        if entry is None:
            return None
        return CacheEntry(kg_id=entry["kg_id"], hash=h, source_id=source_id, meta=entry.get("meta", {}))

    def put(self, source_id: str, genome: Genome, kg_id: str, meta: Optional[Dict[str, Any]] = None) -> CacheEntry:
        h = expensive_hash(source_id, genome)
        self._manifest[h] = {"kg_id": kg_id, "source_id": source_id, "meta": meta or {}}
        self._save()
        return CacheEntry(kg_id=kg_id, hash=h, source_id=source_id, meta=meta or {})

    def stats(self) -> Dict[str, int]:
        return {"cached_builds": len(self._manifest)}
