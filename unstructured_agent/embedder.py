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
from typing import List, Tuple

_SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
_HASHING_DIMS = 256

_st_model = None


def _detect_backend() -> str:
    preferred = os.environ.get("UNSTRUCTURED_EMBED_BACKEND", "auto")
    if preferred != "auto":
        return preferred
    try:
        import sentence_transformers  # noqa: F401
        return "sentence-transformers"
    except ImportError:
        return "hashing"


def _embed_sentence_transformers(text: str) -> List[float]:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(_SENTENCE_TRANSFORMERS_MODEL)
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
