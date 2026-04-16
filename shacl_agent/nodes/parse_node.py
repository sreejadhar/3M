"""
SHACL pipeline node: parse the ontology and load SHACL shapes into rdflib graphs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import rdflib

from ..state import SHACLState

logger = logging.getLogger(__name__)

# Path to the built-in shapes directory (sibling of this nodes/ package)
_SHAPES_DIR = Path(__file__).parent.parent / "shapes"
_BUILTIN_SHAPES = [
    _SHAPES_DIR / "ontology_quality.ttl",
]

_FORMAT_ALIASES = {
    "turtle": "turtle",
    "ttl":    "turtle",
    "xml":    "xml",
    "owl":    "xml",
    "n3":     "n3",
}


def _detect_format(text: str) -> str:
    """Heuristic: turtle starts with @prefix or a < URI; XML has <?xml or <rdf:."""
    stripped = text.lstrip()
    if stripped.startswith("@") or stripped.startswith("<http") or stripped.startswith("PREFIX"):
        return "turtle"
    if stripped.startswith("<?xml") or stripped.startswith("<rdf:"):
        return "xml"
    return "turtle"  # safest default


def _parse_ontology(text: str, fmt_hint: str) -> rdflib.Graph:
    """Try to parse `text` as `fmt_hint`; fall back once before raising."""
    g = rdflib.Graph()
    fmt = _FORMAT_ALIASES.get(fmt_hint.lower(), "turtle")
    try:
        g.parse(data=text, format=fmt)
        return g
    except Exception:
        pass
    # Fallback: try the other main format
    alt = "xml" if fmt == "turtle" else "turtle"
    g = rdflib.Graph()
    g.parse(data=text, format=alt)
    return g


def parse_node(state: SHACLState) -> SHACLState:
    """
    1. Detect / honour ontology_format hint.
    2. Parse ontology_text → ontology_graph.
    3. Load all built-in SHACL shapes + any extra_shapes_ttl → shapes_graph.
    """
    config = state["config"]
    text   = state.get("ontology_text", "").strip()

    if not text:
        state["errors"].append("parse_node: ontology_text is empty")
        state["phase"] = "error"
        return state

    # ── Resolve format ────────────────────────────────────────────────────────
    fmt_hint = config.ontology_format
    if fmt_hint == "auto":
        fmt_hint = _detect_format(text)
    state["ontology_format"] = fmt_hint

    # ── Parse ontology ────────────────────────────────────────────────────────
    try:
        onto_graph = _parse_ontology(text, fmt_hint)
    except Exception as exc:
        msg = f"parse_node: failed to parse ontology — {exc}"
        state["errors"].append(msg)
        logger.error(msg)
        if config.abort_on_parse_error:
            state["phase"] = "error"
            return state
        state["ontology_graph"] = rdflib.Graph()

    else:
        state["ontology_graph"] = onto_graph
        logger.info("parse_node: ontology parsed — %d triples", len(onto_graph))

    # ── Load SHACL shapes ─────────────────────────────────────────────────────
    shapes_graph = rdflib.Graph()

    for shapes_path in _BUILTIN_SHAPES:
        if shapes_path.exists():
            try:
                shapes_graph.parse(str(shapes_path), format="turtle")
                logger.info("parse_node: loaded shapes from %s", shapes_path.name)
            except Exception as exc:
                logger.warning("parse_node: could not load %s — %s", shapes_path.name, exc)

    # Merge caller-supplied extra shapes
    extra = (config.extra_shapes_ttl or "").strip()
    if extra:
        try:
            shapes_graph.parse(data=extra, format="turtle")
            logger.info("parse_node: loaded extra_shapes_ttl (%d chars)", len(extra))
        except Exception as exc:
            state["errors"].append(f"parse_node: failed to parse extra_shapes_ttl — {exc}")

    state["shapes_graph"] = shapes_graph
    logger.info("parse_node: shapes graph has %d triples", len(shapes_graph))

    state["phase"] = "parsed"
    return state
