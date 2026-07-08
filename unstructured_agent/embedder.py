"""
Semantic embedding generation for uploaded documents.

Backends (auto-detected, best-effort):
  sentence-transformers  — all-MiniLM-L6-v2, 384 dims (used when installed)
  hashing                — deterministic feature-hashing bag-of-words, 256
                           dims (always available, no dependency — offline
                           fallback so the pipeline works out of the box)
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List, Optional, Tuple

_SENTENCE_TRANSFORMERS_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
_HASHING_DIMS = 256

_st_model = None
_st_unavailable = False


def _st_cached_locally() -> bool:
    """True if the model is already in the local HuggingFace cache. Checked
    with huggingface_hub's own cache-lookup helper, which never touches the
    network — unlike passing local_files_only=True straight to
    SentenceTransformer(...), which (at least as of sentence-transformers
    5.6.0) still attempts a network reachability check regardless, turning
    a blocked/slow network (SSL-intercepting proxy, no outbound internet)
    into a multi-minute hang on the first document processed after every
    restart before falling back. Checking the cache ourselves first avoids
    ever calling into that path when the model isn't already downloaded."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return try_to_load_from_cache(_SENTENCE_TRANSFORMERS_REPO, "config.json") is not None
    except Exception:
        return False


def _detect_backend() -> str:
    preferred = os.environ.get("UNSTRUCTURED_EMBED_BACKEND", "auto")
    if preferred != "auto":
        return preferred
    if _st_unavailable:
        return "hashing"
    # Check the cache (via the lightweight huggingface_hub package, no
    # network) BEFORE importing sentence_transformers at all — importing
    # that package pulls in transformers/torch and, at least in this
    # environment, is itself slow (~2-3 minutes) independent of whether the
    # model is cached. Skipping the import entirely when we already know
    # we'll fall back to hashing avoids paying that cost for nothing.
    if not _st_cached_locally():
        return "hashing"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return "hashing"
    return "sentence-transformers"


def _embed_sentence_transformers(text: str) -> List[float]:
    global _st_model, _st_unavailable
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        try:
            _st_model = SentenceTransformer(_SENTENCE_TRANSFORMERS_MODEL, local_files_only=True)
        except Exception:
            _st_unavailable = True
            raise
    vec = _st_model.encode([text], normalize_embeddings=True)[0]
    return [float(x) for x in vec]


def _embed_hashing(text: str, dims: int = _HASHING_DIMS) -> List[float]:
    """Deterministic bag-of-words embedding via feature hashing + TF weighting.
    No external dependency — always works, used when no ML embedding backend
    is installed."""
    vec = [0.0] * dims
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dims
        sign = 1.0 if (h // dims) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def fingerprint_text(text: str, topics: Optional[List[str]] = None,
                      entities: Optional[List[dict]] = None, title: Optional[str] = None,
                      excerpt_chars: int = 2000) -> str:
    """Builds a compact, meaning-dense string to embed for a document.
    Topics and entity names carry more semantic signal per token than raw
    prose, so they're placed ahead of a short excerpt of the actual text —
    this is what lets datasource matching key off what the document is
    *about* rather than incidental wording."""
    parts = []
    if title:
        parts.append(title)
    if topics:
        parts.append(", ".join(topics))
    if entities:
        ent_texts = [e.get("text", "") for e in entities if isinstance(e, dict) and e.get("text")]
        if ent_texts:
            parts.append(", ".join(ent_texts[:50]))
    if text:
        parts.append(text[:excerpt_chars])
    return "\n".join(p for p in parts if p)


def embed_text(text: str) -> Tuple[List[float], str]:
    """Returns (vector, model_label). Always succeeds — falls back to the
    hashing backend if no ML embedding library is installed."""
    backend = _detect_backend()
    if backend == "sentence-transformers":
        try:
            return _embed_sentence_transformers(text), f"sentence-transformers:{_SENTENCE_TRANSFORMERS_MODEL}"
        except Exception:
            pass  # fall through to hashing backend
    return _embed_hashing(text), f"hashing:{_HASHING_DIMS}d"


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity — avoids a hard numpy dependency here."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
