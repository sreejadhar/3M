"""
embedding_cache — process-wide, thread-safe cache for the local
sentence-transformers embedding model.

Shared by dialog_agent.nodes.retrieve_node (GraphRAG schema retrieval) and
dialog_agent.verified_queries (few-shot similarity ranking) / semantic_lexicon
(concept lookup) — every call site that needs "all-MiniLM-L6-v2" embeddings
goes through get_sentence_transformer() so the (slow) model load happens at
most once per process, not once per call or once per concurrent caller.

Two problems this fixes, both observed in production timing logs:

1. Every call site previously did `SentenceTransformer("all-MiniLM-L6-v2")`
   independently, reloading the model from disk on every single call.
2. dissect_node resolves multiple concepts concurrently (see
   nodes/dissect_node.py), so without a lock, several threads could all miss
   an empty cache at once and each construct their own redundant model
   instance — wasted work instead of one shared load.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict

# sentence-transformers lazily hits huggingface.co on first load to check for
# newer model revisions, even when the model is already cached on disk. In
# environments where huggingface.co is network-blocked (e.g. corporate
# firewalls returning HTTP 403), this turns into a multi-minute retry storm
# on every request (~4 min observed in production logs) before falling back
# to the already-cached local weights (or failing outright if never cached).
# Defaulting to offline mode skips that network round-trip entirely: if the
# model is cached, load is instant; if not, it fails immediately instead of
# retrying, and callers' existing except blocks fall back to tfidf/keyword
# similarity. `setdefault` so an operator who explicitly sets these (e.g. to
# force a fresh download) is never overridden.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """
    Return a cached SentenceTransformer instance for *model_name*, loading it
    at most once per process. Thread-safe: concurrent callers racing on a
    cold cache block on the lock instead of each constructing their own
    redundant model instance.
    """
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _MODEL_CACHE_LOCK:
        # Re-check inside the lock — another thread may have already built it
        # while this one was waiting.
        cached = _MODEL_CACHE.get(model_name)
        if cached is None:
            from sentence_transformers import SentenceTransformer
            cached = SentenceTransformer(model_name)
            _MODEL_CACHE[model_name] = cached
        return cached
