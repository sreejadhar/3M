"""NLQ → KG Router: vector shortlist then LLM confirmation."""
from __future__ import annotations
import json, logging, os, re
from typing import Dict, List, Optional, Tuple
from .kg_registry import KGEntry, list_all as list_kgs

logger = logging.getLogger(__name__)

def route(nlq:str, threshold:float=0.30, llm_model:str="claude-haiku-4-5",
          llm_temperature:float=0.0, backend:str="auto") -> List[str]:
    """Return kg_ids relevant to *nlq*. Returns [] if no KGs registered."""
    kgs = list_kgs()
    if not kgs: return []
    if len(kgs) == 1: return [kgs[0].kg_id]
    candidates = _vector_shortlist(nlq, kgs, threshold, backend)
    if not candidates:
        logger.info("kg_router: all below threshold → returning all")
        return [k.kg_id for k in kgs]
    if len(candidates) == 1:
        return [candidates[0][0].kg_id]
    confirmed = _llm_confirm(nlq, candidates, llm_model, llm_temperature)
    return confirmed if confirmed else [c[0].kg_id for c in candidates]

def _get_st():
    # Delegate to embedding_cache's shared, process-wide singleton instead of
    # keeping a second independent SentenceTransformer instance here. That
    # module sets HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 before ever loading
    # the model — without it, sentence-transformers hits huggingface.co on
    # first load to check for revisions even when already cached, which turns
    # into a multi-minute retry storm in network-restricted environments
    # (observed: ~4 min). This function used to build its own SentenceTransformer
    # directly, bypassing that protection entirely — every dissect_node call
    # that resolves embeddings via kg_router._embed hit the unprotected path.
    from .embedding_cache import get_sentence_transformer
    return get_sentence_transformer("all-MiniLM-L6-v2")

def _embed(text:str, backend:str="auto") -> Optional[List[float]]:
    try:
        if backend in ("sentence-transformers","auto"):
            try:
                return _get_st().encode([text], normalize_embeddings=True)[0].tolist()
            except ImportError:
                if backend == "sentence-transformers": raise
        # keyword fallback
        import math
        tokens = re.findall(r"\w+", text.lower())
        vocab: Dict[str,int] = {}
        for t in tokens: vocab[t] = vocab.get(t,0)+1
        mag = math.sqrt(sum(v*v for v in vocab.values())) or 1.0
        keys = sorted(vocab)
        return [vocab[k]/mag for k in keys], keys  # type: ignore
    except Exception as e:
        logger.warning("kg_router embed failed: %s", e)
        return None

def _cosine(a,b) -> float:
    if len(a)!=len(b): return 0.0
    dot=sum(x*y for x,y in zip(a,b))
    ma=sum(x*x for x in a)**0.5; mb=sum(x*x for x in b)**0.5
    return dot/(ma*mb) if ma and mb else 0.0

def _vector_shortlist(nlq,kgs,threshold,backend):
    emb = _embed(nlq, backend)
    # handle keyword-fallback tuple (vec, keys)
    if isinstance(emb, tuple): emb = emb[0]
    if emb is None: return [(k,1.0) for k in kgs]
    results = []
    for kg in kgs:
        if not kg.embedding:
            results.append((kg, 0.5)); continue
        if len(emb) != len(kg.embedding):
            results.append((kg, 0.5)); continue
        score = _cosine(emb, kg.embedding)
        if score >= threshold: results.append((kg, score))
    results.sort(key=lambda x:x[1], reverse=True)
    return results

def _llm_confirm(nlq, candidates, model, temperature) -> List[str]:
    import anthropic
    kg_list = "\n".join(
        f'- kg_id: "{c.kg_id}"  name: "{c.display_name}"  description: "{c.description}"  entities: {c.entity_types}'
        for c,_ in candidates)
    system = ("You are a data routing assistant. Given a question and available Knowledge Graphs, "
              "return ONLY a JSON array of kg_id strings for the KGs needed to answer the question.")
    user = f"Question: {nlq}\n\nAvailable Knowledge Graphs:\n{kg_list}\n\nReturn JSON array of kg_id strings."
    try:
        from llm_client import get_client
        client = get_client()
        msg = client.messages.create(model=model, max_tokens=256, temperature=temperature,
                                     system=system, messages=[{"role":"user","content":user}])
        text = (msg.content[0].text if msg.content else "[]").strip()
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        return json.loads(m.group(0)) if m else []
    except Exception as e:
        logger.warning("kg_router llm_confirm failed: %s", e)
        return []

def embed_kg_description(entry: KGEntry, backend:str="auto") -> Optional[List[float]]:
    """Embed a KG's description+keywords for storage in registry."""
    text = f"{entry.display_name}. {entry.description}. " + " ".join(entry.domain_keywords + entry.entity_types)
    emb = _embed(text, backend)
    if isinstance(emb, tuple): emb = emb[0]
    return emb
