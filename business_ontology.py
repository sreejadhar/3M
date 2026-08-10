"""
Business Ontology — a SKOS+OWL graph generated from the governed Business
Glossary (biz_glossary_terms / biz_glossary_term_relations), distinct from
the per-source structural ontology produced by ontology_agent.

Scoped per data source, one artifact per source_id — mirroring the structural
ontology's per-source model — and restricted to sources that actually have a
generated Business Glossary (i.e. at least one term linked via
biz_glossary_term_assets; see glossary_registry.list_glossary_sources()).

Every non-deprecated glossary term linked to that source becomes a
skos:Concept (prefLabel, definition, altLabel from synonym relations,
broader/narrower/related from term relations). Where a term is linked to a
table/column via ontology_enricher.py's glossary_attr_links, this module
resolves the corresponding class/property URI in that source's
structural-ontology TTL (matched by rdfs:label, the same convention
ontology_agent/nodes/build_node.py uses) and adds owl:equivalentClass
(whole-table/entity match) or rdfs:seeAlso (column/attribute match) —
best-effort, skipped silently if unresolvable.

Storage model: one mutable "draft" row per source_id (the live, editable
artifact) plus an immutable "versions" history per source_id created only by
explicit save_version() calls. Editing a term/relation regenerates the draft
for the given source_id from the glossary; a raw-TTL edit overwrites that
source's draft directly without writing back into the glossary.

Same SQLite/PostgreSQL backend pattern as glossary_registry.py (same
data/metadata.db file) — importable from orchestrator_api.py without pulling
in dialog_agent.

Public API
----------
list_glossary_sources()                        -> List[dict]  (delegates to glossary_registry)
generate_business_ontology(source_id, created_by="", source_ontology="") -> dict
get_draft(source_id)                            -> dict
save_draft_ttl(source_id, ttl_content, updated_by="") -> dict
update_concept(source_id, term_id, changed_by="", **fields) -> dict
delete_concept(source_id, term_id, changed_by="") -> dict
add_relation(source_id, term_id, related_term_id, relationship_type, changed_by="") -> dict
delete_relation(source_id, relation_id, changed_by="") -> dict
save_version(source_id, label, created_by="") -> dict
list_versions(source_id)                       -> List[dict]
get_version(source_id, version_id)             -> dict | None
restore_version(source_id, version_id, restored_by="") -> dict | None
"""
from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import OWL, RDFS, SKOS

import glossary_registry as _gr

logger = logging.getLogger(__name__)

BUSINESS_NS = Namespace("https://datananite.local/business-ontology/")


# ── Environment helpers (mirrors glossary_registry.py) ──────────────────────

def _is_production() -> bool:
    return os.environ.get("APP_ENV", "").strip().lower() == "production"


def _is_postgres() -> bool:
    if not _is_production():
        return False
    return bool(os.environ.get("KG_POSTGRES_DSN", ""))


def _pg_dsn() -> str:
    return os.environ.get("KG_POSTGRES_DSN", "")


def _sqlite_path() -> str:
    return os.environ.get("METADATA_DB", "data/metadata.db")


class _SQLiteCur:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._cur = conn.cursor()

    def execute(self, sql: str, params: tuple = ()) -> "_SQLiteCur":
        self._cur.execute(sql, params)
        return self

    def ddl(self, *statements: str) -> None:
        for stmt in statements:
            self._cur.execute(stmt)

    def fetchall(self) -> List[Dict]:
        rows = self._cur.fetchall() if self._cur else []
        return [dict(r) for r in rows]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone() if self._cur else None
        return dict(r) if r else None


class _PGCur:
    def __init__(self, conn: Any, cur: Any) -> None:
        self._conn = conn
        self._cur = cur

    def execute(self, sql: str, params: tuple = ()) -> "_PGCur":
        self._cur.execute(sql.replace("?", "%s"), params)
        return self

    def ddl(self, *statements: str) -> None:
        for stmt in statements:
            self._cur.execute(stmt)

    def fetchall(self) -> List[Dict]:
        rows = self._cur.fetchall() if self._cur else []
        return [dict(r) for r in rows]

    def fetchone(self) -> Optional[Dict]:
        r = self._cur.fetchone() if self._cur else None
        return dict(r) if r else None


@contextmanager
def _cursor_ctx() -> Iterator[Any]:
    if _is_postgres():
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(_pg_dsn(), cursor_factory=psycopg2.extras.RealDictCursor)
        cur = conn.cursor()
        try:
            yield _PGCur(conn, cur)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        path = _sqlite_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield _SQLiteCur(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── DDL ──────────────────────────────────────────────────────────────────────
# `scope` holds the source_id — one draft row and N version rows per source.

_DDL_DRAFT = """
CREATE TABLE IF NOT EXISTS biz_ontology_draft (
    scope              TEXT PRIMARY KEY,
    ttl_content        TEXT NOT NULL DEFAULT '',
    triple_count       INTEGER NOT NULL DEFAULT 0,
    source_version_id  TEXT NOT NULL DEFAULT '',
    updated_by         TEXT NOT NULL DEFAULT '',
    updated_at         TEXT NOT NULL
)
"""

_DDL_VERSIONS = """
CREATE TABLE IF NOT EXISTS biz_ontology_versions (
    version_id      TEXT PRIMARY KEY,
    scope           TEXT NOT NULL DEFAULT '',
    version_number  INTEGER NOT NULL,
    label           TEXT NOT NULL DEFAULT '',
    ttl_content     TEXT NOT NULL,
    triple_count    INTEGER NOT NULL DEFAULT 0,
    created_by      TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    is_current      INTEGER NOT NULL DEFAULT 0
)
"""

_schema_ensured = False


def _ensure(cur: Any) -> None:
    global _schema_ensured
    cur.ddl(_DDL_DRAFT, _DDL_VERSIONS)
    if not _is_postgres():
        # Migration for dbs created before `scope` existed on biz_ontology_versions.
        cols = {r["name"] for r in cur.execute("PRAGMA table_info(biz_ontology_versions)").fetchall()}
        if "scope" not in cols:
            cur.execute("ALTER TABLE biz_ontology_versions ADD COLUMN scope TEXT NOT NULL DEFAULT ''")
    _schema_ensured = True


# ── Sources with a generated glossary ───────────────────────────────────────

def list_glossary_sources() -> List[Dict]:
    return _gr.list_glossary_sources()


# ── Structural-ontology cross-link resolution ───────────────────────────────

def _find_label_uri(graph: Graph, label: str) -> Optional[URIRef]:
    """Find a subject whose rdfs:label literal matches `label` — the same
    convention ontology_agent/nodes/build_node.py uses for both owl:Class
    (table) and owl:DatatypeProperty (column) subjects."""
    for subj in graph.subjects(RDFS.label, Literal(label)):
        return subj
    return None


def _resolve_structural_uri(link: Dict, struct_graph: Optional[Graph]) -> Optional[URIRef]:
    """Best-effort resolution of a glossary_attr_links row to a class/property
    URI inside this source's own structural-ontology TTL (already parsed by
    the caller). Returns None (never raises) if the table/column can't be
    matched or no structural ontology exists for this source yet."""
    if struct_graph is None:
        return None
    table_name = link.get("table_name") or ""
    column_name = link.get("column_name") or ""
    attr_id = link.get("attr_id") or ""
    if not table_name:
        return None
    if attr_id and column_name:
        return _find_label_uri(struct_graph, column_name)
    return _find_label_uri(struct_graph, table_name)


# ── Graph construction ───────────────────────────────────────────────────────

def _build_graph(source_id: str, source_ontology: str = "") -> Graph:
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("", BUSINESS_NS)

    scheme_uri = BUSINESS_NS[f"scheme/{source_id}"]
    g.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
    g.add((scheme_uri, RDFS.label, Literal(f"Business Ontology — {source_id}")))

    terms = [t for t in _gr.list_terms(source_id=source_id) if t.get("status") != "deprecated"]
    term_by_id = {t["term_id"]: t for t in terms}

    struct_graph: Optional[Graph] = None
    if source_ontology.strip():
        try:
            struct_graph = Graph()
            struct_graph.parse(data=source_ontology, format="turtle")
        except Exception as exc:
            logger.debug("business_ontology: failed to parse structural ontology for %s: %s", source_id, exc)
            struct_graph = None

    try:
        import ontology_enricher as _enricher
        all_links = _enricher.get_glossary_attr_links()
    except Exception:
        all_links = []
    term_ids_in_scope = set(term_by_id.keys())
    links_by_term: Dict[str, List[Dict]] = {}
    for link in all_links:
        tid = link.get("term_id", "")
        if tid in term_ids_in_scope:
            links_by_term.setdefault(tid, []).append(link)

    relations_by_term: Dict[str, List[Dict]] = {}
    for rel in _gr.list_all_relations():
        if rel["term_id"] not in term_ids_in_scope and rel["related_term_id"] not in term_ids_in_scope:
            continue
        relations_by_term.setdefault(rel["term_id"], []).append(rel)
        if rel["related_term_id"] != rel["term_id"]:
            relations_by_term.setdefault(rel["related_term_id"], []).append(rel)

    for term in terms:
        term_id = term["term_id"]
        concept_uri = BUSINESS_NS[f"term/{term_id}"]
        g.add((concept_uri, RDF.type, SKOS.Concept))
        g.add((concept_uri, SKOS.inScheme, scheme_uri))
        g.add((concept_uri, SKOS.prefLabel, Literal(term.get("preferred_name", ""))))
        if term.get("definition"):
            g.add((concept_uri, SKOS.definition, Literal(term["definition"])))
        if term.get("domain"):
            g.add((concept_uri, RDFS.comment, Literal(f"Domain: {term['domain']}")))

        for rel in relations_by_term.get(term_id, []):
            other_id = rel["related_term_id"] if rel["term_id"] == term_id else rel["term_id"]
            other = term_by_id.get(other_id)
            if not other:
                continue
            rel_type = rel.get("relationship_type", "related")
            if rel_type == "synonym":
                g.add((concept_uri, SKOS.altLabel, Literal(other.get("preferred_name", ""))))
                continue
            other_uri = BUSINESS_NS[f"term/{other_id}"]
            # relations are stored directionally (term_id -> related_term_id);
            # only emit the predicate in the direction the row was authored
            if rel["term_id"] != term_id:
                continue
            pred = {"broader": SKOS.broader, "narrower": SKOS.narrower,
                    "related": SKOS.related}.get(rel_type, SKOS.related)
            g.add((concept_uri, pred, other_uri))

        for link in links_by_term.get(term_id, []):
            struct_uri = _resolve_structural_uri(link, struct_graph)
            if struct_uri is None:
                continue
            if link.get("attr_id"):
                g.add((concept_uri, RDFS.seeAlso, struct_uri))
            else:
                g.add((concept_uri, OWL.equivalentClass, struct_uri))

    return g


# ── Generation ───────────────────────────────────────────────────────────────

def generate_business_ontology(source_id: str, created_by: str = "",
                                source_ontology: str = "") -> Dict:
    g = _build_graph(source_id, source_ontology)
    ttl_content = g.serialize(format="turtle")
    triple_count = len(g)
    term_count = len(list(g.subjects(RDF.type, SKOS.Concept)))
    now = _now()
    with _cursor_ctx() as cur:
        _ensure(cur)
        existing = cur.execute("SELECT scope FROM biz_ontology_draft WHERE scope=?", (source_id,)).fetchone()
        if existing:
            cur.execute(
                "UPDATE biz_ontology_draft SET ttl_content=?, triple_count=?, updated_by=?, updated_at=? "
                "WHERE scope=?",
                (ttl_content, triple_count, created_by, now, source_id),
            )
        else:
            cur.execute(
                "INSERT INTO biz_ontology_draft "
                "(scope, ttl_content, triple_count, source_version_id, updated_by, updated_at) "
                "VALUES (?, ?, ?, '', ?, ?)",
                (source_id, ttl_content, triple_count, created_by, now),
            )
    try:
        import ontology_enricher as _enricher
        _enricher.enrich_source_kg_from_business_glossary(source_id)
    except Exception:
        logger.exception("business_ontology: KG enrichment failed for source %s", source_id)

    return {"source_id": source_id, "ttl_content": ttl_content, "triple_count": triple_count,
            "term_count": term_count, "updated_at": now}


def get_draft(source_id: str) -> Dict:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute("SELECT * FROM biz_ontology_draft WHERE scope=?", (source_id,)).fetchone()
    if row:
        return dict(row)
    return generate_business_ontology(source_id)


def save_draft_ttl(source_id: str, ttl_content: str, updated_by: str = "") -> Dict:
    try:
        g = Graph()
        g.parse(data=ttl_content, format="turtle")
    except Exception as exc:
        raise ValueError(f"Invalid Turtle content: {exc}") from exc
    triple_count = len(g)
    now = _now()
    with _cursor_ctx() as cur:
        _ensure(cur)
        existing = cur.execute("SELECT scope FROM biz_ontology_draft WHERE scope=?", (source_id,)).fetchone()
        if existing:
            cur.execute(
                "UPDATE biz_ontology_draft SET ttl_content=?, triple_count=?, updated_by=?, updated_at=? "
                "WHERE scope=?",
                (ttl_content, triple_count, updated_by, now, source_id),
            )
        else:
            cur.execute(
                "INSERT INTO biz_ontology_draft "
                "(scope, ttl_content, triple_count, source_version_id, updated_by, updated_at) "
                "VALUES (?, ?, ?, '', ?, ?)",
                (source_id, ttl_content, triple_count, updated_by, now),
            )
    return get_draft(source_id)


# ── Structured term/triple editing ──────────────────────────────────────────

_ALLOWED_CONCEPT_FIELDS = {"preferred_name", "definition", "domain", "steward", "status"}


def update_concept(source_id: str, term_id: str, changed_by: str = "", **fields: Any) -> Dict:
    updates = {k: v for k, v in fields.items() if k in _ALLOWED_CONCEPT_FIELDS}
    if updates:
        _gr.update_term(term_id, changed_by=changed_by, **updates)
    return generate_business_ontology(source_id, created_by=changed_by)


def delete_concept(source_id: str, term_id: str, changed_by: str = "") -> Dict:
    _gr.reject_term(term_id, changed_by=changed_by)
    return generate_business_ontology(source_id, created_by=changed_by)


def add_relation(source_id: str, term_id: str, related_term_id: str, relationship_type: str,
                  changed_by: str = "") -> Dict:
    _gr.add_relation(term_id, related_term_id, relationship_type)
    return generate_business_ontology(source_id, created_by=changed_by)


def delete_relation(source_id: str, relation_id: str, changed_by: str = "") -> Dict:
    _gr.delete_relation(relation_id)
    return generate_business_ontology(source_id, created_by=changed_by)


# ── Versioning ───────────────────────────────────────────────────────────────

def save_version(source_id: str, label: str, created_by: str = "") -> Dict:
    draft = get_draft(source_id)
    version_id = str(uuid.uuid4())
    now = _now()
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT MAX(version_number) AS mx FROM biz_ontology_versions WHERE scope=?", (source_id,)
        ).fetchone()
        next_number = int((row or {}).get("mx") or 0) + 1
        cur.execute("UPDATE biz_ontology_versions SET is_current=0 WHERE scope=?", (source_id,))
        cur.execute(
            "INSERT INTO biz_ontology_versions "
            "(version_id, scope, version_number, label, ttl_content, triple_count, created_by, created_at, is_current) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (version_id, source_id, next_number, label or "", draft.get("ttl_content", ""),
             draft.get("triple_count", 0), created_by, now),
        )
        cur.execute(
            "UPDATE biz_ontology_draft SET source_version_id=? WHERE scope=?",
            (version_id, source_id),
        )
    return get_version(source_id, version_id) or {}


def list_versions(source_id: str) -> List[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        rows = cur.execute(
            "SELECT version_id, version_number, label, triple_count, created_by, created_at, is_current "
            "FROM biz_ontology_versions WHERE scope=? ORDER BY version_number DESC",
            (source_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_version(source_id: str, version_id: str) -> Optional[Dict]:
    with _cursor_ctx() as cur:
        _ensure(cur)
        row = cur.execute(
            "SELECT * FROM biz_ontology_versions WHERE version_id=? AND scope=?", (version_id, source_id)
        ).fetchone()
    return dict(row) if row else None


def restore_version(source_id: str, version_id: str, restored_by: str = "") -> Optional[Dict]:
    version = get_version(source_id, version_id)
    if not version:
        return None
    now = _now()
    with _cursor_ctx() as cur:
        _ensure(cur)
        cur.execute("UPDATE biz_ontology_versions SET is_current=0 WHERE scope=?", (source_id,))
        cur.execute("UPDATE biz_ontology_versions SET is_current=1 WHERE version_id=?", (version_id,))
        existing = cur.execute("SELECT scope FROM biz_ontology_draft WHERE scope=?", (source_id,)).fetchone()
        if existing:
            cur.execute(
                "UPDATE biz_ontology_draft SET ttl_content=?, triple_count=?, source_version_id=?, "
                "updated_by=?, updated_at=? WHERE scope=?",
                (version["ttl_content"], version["triple_count"], version_id, restored_by, now, source_id),
            )
        else:
            cur.execute(
                "INSERT INTO biz_ontology_draft "
                "(scope, ttl_content, triple_count, source_version_id, updated_by, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (source_id, version["ttl_content"], version["triple_count"], version_id, restored_by, now),
            )
    return get_draft(source_id)
