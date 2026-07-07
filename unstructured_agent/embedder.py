"""
Semantic embedding generation for the unstructured data intelligence agent.

Embeds a short text representation of each document's fingerprint (title +
summary + topics) so doc-to-doc similarity can use real semantic distance
instead of exact topic-string overlap (Jaccard). Mirrors the backend
detection pattern used by knowledge_graph_agent/nodes/embed_node.py so the
two agents share the same optional-dependency story.

Supported backends:
  sentence-transformers  — all-MiniLM-L6-v2, 384 dims (default, local/offline)
  openai                 — text-embedding-3-small, 1536 dims (needs OPENAI_API_KEY)

Backend is best-effort: if neither package is installed, embed_text() returns
None and callers skip the embedding step without failing the pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
_OPENAI_MODEL = "text-embedding-3-small"

# Lazily-loaded sentence-transformers model (loading it per-call is expensive)
_st_model = None


def _detect_backend(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    for candidate, pkg in [
        ("sentence-transformers", "sentence_transformers"),
        ("openai", "openai"),
    ]:
        try:
            __import__(pkg)
            return candidate
        except ImportError:
            continue
    return "unsupported"


def _embed_sentence_transformers(text: str) -> List[float]:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(_SENTENCE_TRANSFORMERS_MODEL)
    vec = _st_model.encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in vec]


def _embed_openai(text: str) -> List[float]:
    from openai import OpenAI
    resp = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "")).embeddings.create(
        model=_OPENAI_MODEL, input=[text],
    )
    return [float(x) for x in resp.data[0].embedding]


def fingerprint_text(fp: dict) -> str:
    """Build the text representation embedded for a document fingerprint."""
    parts = [
        fp.get("title", ""),
        fp.get("summary", ""),
        fp.get("doc_type", ""),
        " ".join(fp.get("topics", []) or []),
    ]
    return " ".join(p for p in parts if p).strip()


def embed_text(text: str) -> Optional[Tuple[List[float], str]]:
    """
    Embed a single text string.

    Returns (vector, model_label) where model_label identifies the backend
    and model (e.g. "sentence-transformers:all-MiniLM-L6-v2"), or None if no
    embedding backend is available or the text is empty.
    """
    if not text or not text.strip():
        return None

    backend = _detect_backend(os.environ.get("UNSTRUCTURED_EMBED_BACKEND", "auto"))
    if backend == "unsupported":
        logger.debug("embedder: no embedding backend installed — skipping")
        return None

    try:
        if backend == "sentence-transformers":
            return _embed_sentence_transformers(text), f"sentence-transformers:{_SENTENCE_TRANSFORMERS_MODEL}"
        if backend == "openai":
            return _embed_openai(text), f"openai:{_OPENAI_MODEL}"
    except Exception as exc:
        logger.warning("embedder: embedding failed (backend=%s): %s", backend, exc)
        return None

    return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity — avoids a hard numpy dependency here."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
