# DataNanite Unstructured Data Intelligence — Architecture & Novelty

> **Status:** Planned — not yet implemented  
> **Last updated:** 2026-04-24  
> **Scope:** Architectural design for extending DataNanite to index, understand, and semantically link unstructured documents to the existing structured data model — without modifying any existing service.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Design Principles](#2-design-principles)
3. [System Architecture](#3-system-architecture)
4. [Component Reference](#4-component-reference)
5. [Data Flow — End to End](#5-data-flow--end-to-end)
6. [Schema Design](#6-schema-design)
7. [Knowledge Graph Integration](#7-knowledge-graph-integration)
8. [Cross-Modal Linking Algorithm](#8-cross-modal-linking-algorithm)
9. [Isolation Contract with Existing Nanite](#9-isolation-contract-with-existing-nanite)
10. [Novelty and Differentiation](#10-novelty-and-differentiation)
11. [Known Complexities and Mitigations](#11-known-complexities-and-mitigations)

---

## 1. Problem Statement

Enterprise knowledge exists in two worlds that have never been connected:

**Structured world** — databases, data warehouses, data lakes. DataNanite already understands this world completely: it extracts schema, infers taxonomy, builds an OWL ontology, populates a knowledge graph, and lets users ask questions in plain English.

**Unstructured world** — PDF reports, Word documents, PowerPoint decks, policy files, contracts, research papers, email archives. This world contains the *context*, *decisions*, *policies*, and *definitions* that explain why the structured data looks the way it does. A pricing policy document explains the pricing columns in the fact table. A market research report explains the customer segments. An audit report explains the compliance flags.

No existing platform connects these two worlds automatically. DataNanite's unstructured intelligence layer does.

**What this is NOT:** A document search engine. A RAG (Retrieval-Augmented Generation) system. A content management system. Raw text is never stored, never indexed, never made searchable verbatim.

**What this IS:** A semantic fingerprinting and graph-linking engine that understands what a document is *about* and connects that meaning to the structured data model that already exists in DataNanite.

---

## 2. Design Principles

### P1 — Metadata only, never content
Raw document text is read once, in memory, for semantic extraction. It is never persisted, never indexed for full-text search, never stored in any database. What is stored is a compact semantic fingerprint: summary, topics, entities, domain, document type. This is a privacy-by-design decision — the system can tell you *that* a document discusses trade spend without ever revealing *what* it says about trade spend.

### P2 — Additive to the existing system, never modifying it
The unstructured service is a sidecar. It reads from Nanite's existing APIs. It writes only to its own database and to new node labels in the shared graph. It does not modify any existing table, any existing KG node, or any existing pipeline stage. Nanite can run indefinitely without the unstructured service being present.

### P3 — Cross-modal linking is the primary value
Storing document metadata alone is commodity. The differentiated value is the automatic detection of semantic relationships between documents and structured data assets — a document that *describes* a KPI, *references* a table, or *provides context* for a data domain. Every design decision prioritises the quality of these cross-modal links over the volume of documents ingested.

### P4 — Domain-aware semantic extraction
Semantic extraction uses the same domain intelligence already present in DataNanite (`_INDUSTRY_SIGNALS`, `_DOMAIN_LLM_HINTS`). The same domain that was inferred for a data source is used as context when extracting entities from its associated documents. This ensures that "NII" is extracted as "net interest income" for a Banking document, not as an ambiguous acronym.

### P5 — Quality over quantity
A document quality gate filters out junk (temp files, auto-saves, duplicates, zero-content files) before any LLM call is made. LLM extraction is reserved for documents that pass a minimum quality threshold. Low-quality documents receive structural metadata only.

---

## 3. System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                   EXISTING NANITE SYSTEM                         ║
║  (zero changes — runs independently at all times)                ║
║                                                                  ║
║  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ ║
║  │ orchestrator_api│    │  dialog_agent    │    │  metadata   │ ║
║  │   port 8005     │    │  port 8003       │    │  catalog.db │ ║
║  └────────┬────────┘    └──────────────────┘    └──────┬──────┘ ║
║           │ (serves chat UI, NLQ pipeline)             │        ║
║           │ ← ADD: 2 proxy routes only                 │ READ   ║
╚═══════════╪════════════════════════════════════════════╪════════╝
            │                                            │
            │ HTTP proxy (additive)                      │ REST API
            ↓                                            ↓
╔══════════════════════════════════════════════════════════════════╗
║              UNSTRUCTURED AGENT  (new, isolated)                 ║
║                        port 8007                                 ║
║                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │  FILE CONNECTOR LAYER                                     │   ║
║  │  Local FS │ S3/GCS/Azure │ SharePoint │ Google Drive      │   ║
║  └───────────────────────┬──────────────────────────────────┘   ║
║                           │  file manifest (path, hash, mime)   ║
║                           ↓                                      ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │  FORMAT PARSER                                            │   ║
║  │  PDF (digital + OCR) │ DOCX │ PPTX │ HTML │ MD │ TXT    │   ║
║  │  Output: text window + structural metadata               │   ║
║  │  Raw text: IN MEMORY ONLY — never persisted              │   ║
║  └───────────────────────┬──────────────────────────────────┘   ║
║                           │  text window (ephemeral)            ║
║                           ↓                                      ║
║  ┌──────────────────────────────────────────────────────────┐   ║
║  │  SEMANTIC EXTRACTION ENGINE (LLM-based)                  │   ║
║  │  Domain-aware fingerprinting                             │   ║
║  │  Output: {summary, topics, entities, doc_type, ...}      │   ║
║  └───────────────────────┬──────────────────────────────────┘   ║
║                           │  semantic fingerprint               ║
║                           ↓                                      ║
║  ┌──────────────┐   ┌──────────────────────────────────────┐   ║
║  │ unstructured │   │  CROSS-MODAL LINKER                   │   ║
║  │    .db       │←──│  doc↔doc │ doc↔table │ doc↔KPI       │   ║
║  │ (new SQLite) │   │  reads metadata API + KG              │   ║
║  └──────────────┘   └──────────────────┬─────────────────── ┘   ║
║                                         │  DocNode + edges       ║
║                                         ↓                        ║
╚═════════════════════════════════════════╪════════════════════════╝
                                          │ WRITE :DocNode only
                                          ↓
                              ╔═══════════════════════╗
                              ║  NEO4J / TINKERPOP    ║
                              ║  Existing: :KGNode    ║
                              ║  New:      :DocNode   ║
                              ║  Cross:    edges only ║
                              ╚═══════════════════════╝
```

---

## 4. Component Reference

### 4.1 File Connector Layer

**Purpose:** Enumerate documents from any enterprise file source, produce a normalised file manifest, and detect changes via content hash.

**Connectors:**

| Source | Protocol | Change Detection |
|---|---|---|
| Local filesystem / NFS | `pathlib.Path.walk()` | SHA-256 of file content |
| AWS S3 | `boto3` list_objects_v2 | ETag (S3 content hash) |
| Google Cloud Storage | `google-cloud-storage` | Object generation number |
| Azure Blob Storage | `azure-storage-blob` | Content-MD5 header |
| SharePoint / OneDrive | Microsoft Graph API | `lastModifiedDateTime` + file hash |
| Google Drive | Drive API v3 | `md5Checksum` field |

**File manifest schema:**
```
{
  "path":        "s3://bucket/reports/q3_rgm_review.pdf",
  "source_type": "s3",
  "size_bytes":  2048576,
  "mime_type":   "application/pdf",
  "checksum":    "sha256:a3f4...",
  "created_at":  "2025-09-14T10:22:00Z",
  "modified_at": "2025-10-01T14:05:00Z"
}
```

**Change detection algorithm:** On each indexing run, compare incoming checksum against the stored checksum in `unstructured.db`. Only process files where the checksum has changed or no record exists. Files deleted from source are soft-deleted (never hard-deleted — audit trail preserved).

---

### 4.2 Format Parser

**Purpose:** Extract raw text and structural metadata from each file format. Raw text is held in memory only for the duration of the extraction pipeline — it is never written to disk or database.

**Parsing strategy per format:**

| Format | Library | Structural metadata extracted |
|---|---|---|
| PDF (digital) | `pdfplumber` | Title, author, page count, section headers (H1/H2 by font size), embedded table count |
| PDF (scanned) | `pdf2image` + `pytesseract` | Same as above; OCR confidence score recorded |
| DOCX | `python-docx` | Title, author, heading hierarchy, word count, table count |
| PPTX | `python-pptx` | Title, slide count, speaker notes presence, chart count |
| XLSX | `openpyxl` | Sheet names, named range list (not cell values) |
| HTML | `BeautifulSoup` | `<title>`, `<meta>` tags, `<h1>`–`<h3>` hierarchy |
| Markdown | `mistune` | Heading hierarchy, code block count |
| Plain text | stdlib | Line count, word count estimate |

**Text windowing — the privacy-preserving extraction strategy:**

Rather than feeding the full document to the LLM, the parser produces a **text window**:
```
[First 2000 tokens of body text]
+ [All section/heading titles]
+ [Last 300 tokens of body text]
```

This captures the document's opening context (abstract, introduction, executive summary), its full structural outline (all headings), and its closing context (conclusions, recommendations). For the vast majority of enterprise documents, this window contains sufficient signal to extract an accurate semantic fingerprint without requiring the full content.

For very short documents (< 500 tokens), the full text is used.

**OCR confidence gating:** When OCR is used, if the Tesseract confidence score is below 60%, the document is flagged `ocr_low_confidence: true` and the semantic extraction LLM prompt is adjusted to be more conservative about named entity claims.

---

### 4.3 Semantic Extraction Engine

**Purpose:** Transform the ephemeral text window into a durable semantic fingerprint using a domain-aware LLM call.

**Quality gate — runs before any LLM call:**

A document passes to LLM extraction only if it meets ALL of:
- File size > 1 KB (filters empty/stub files)
- Extracted text length > 200 tokens (filters image-only or corrupted files)
- Not a known junk pattern (filename matches `~$*`, `*.tmp`, `*_backup*`, `draft_*`)
- MIME type is in the supported set
- File age < configured retention window (configurable per source)

Documents that fail the quality gate receive **structural metadata only** — no LLM call, no semantic fingerprint. They are still indexed (for completeness of the document estate) but marked `enriched: false`.

**The semantic extraction prompt:**

The LLM receives:
1. The domain context (same domain label already inferred for the associated data source, e.g. "CPG/RGM")
2. The analyst role if configured (e.g. "Revenue Growth Manager")
3. The structural metadata (filename, section headers, doc type hint)
4. The text window

It is instructed to return a structured JSON semantic fingerprint:

```json
{
  "title": "Q3 2025 RGM Performance Review — APAC",
  "summary": "Quarterly revenue growth management review covering pricing performance, trade spend efficiency, and distribution gains across APAC markets for Q3 2025.",
  "domain": "CPG/RGM",
  "doc_type": "report",
  "topics": ["pricing strategy", "trade spend", "distribution gain", "Q3 2025", "APAC", "promo ROI"],
  "named_entities": {
    "organizations": ["Unilever", "Walmart", "Carrefour"],
    "products":      ["Dove", "Surf Excel", "Lifebuoy"],
    "geographies":   ["India", "Indonesia", "Thailand"],
    "people":        [],
    "kpis":          ["RSV", "GSV", "trade spend %", "distribution gain bps", "price index"]
  },
  "time_references": ["Q3 2025", "FY2025", "YoY"],
  "language": "en",
  "sensitivity": "internal",
  "pii_risk": false
}
```

**Domain-aware disambiguation:** The same `_DOMAIN_LLM_HINTS` dictionary used in taxonomy inference is injected into the extraction prompt. This ensures:
- In CPG documents: "yield" → category yield / volume, not financial return
- In Banking documents: "NII" → net interest income, not an unknown acronym
- In Aviation documents: "station" → airport station code, not physical station

**Model selection:** `claude-haiku-4-5-20251001` for standard documents (fast, low cost). `claude-sonnet-4-6` for documents flagged as high-sensitivity, large (> 50 pages), or where the quality gate score is borderline. This tiering keeps cost proportional to document value.

---

### 4.4 Cross-Modal Linker

**Purpose:** Detect and record semantic relationships between documents, and between documents and the structured data assets already known to DataNanite.

This component runs after the semantic fingerprint is stored. It calls the existing DataNanite metadata API (read-only) to retrieve structured data assets and then computes relationships.

Four relationship types are detected:

#### Doc ↔ Doc: Topic Similarity

The `topics` array from each document is embedded using a lightweight embedding model. Cosine similarity is computed pairwise within the same domain and source. Pairs exceeding a configurable threshold (default: 0.85) receive a `SIMILAR_TOPIC` edge in the KG.

To avoid N² pairwise computation at scale, an approximate nearest-neighbour index (FAISS or `annoy`) is maintained per domain. Only candidate pairs within the ANN index are evaluated for exact cosine similarity.

Similarity score is stored on the edge: `{similarity: 0.91, basis: "topics"}`. Edges below 0.70 are pruned. Edges between 0.70 and 0.85 are stored as `WEAKLY_SIMILAR` and not surfaced in the default UI.

#### Doc ↔ Doc: Citation / Reference

During text window extraction, a regex scan looks for patterns matching:
- Other known document titles (fuzzy-matched against `unstructured.db`)
- File names referenced in the text
- Report codes or identifiers (e.g. "see Report RGM-2025-Q2")

Confirmed matches produce a `REFERENCES` edge. Confidence score derived from match specificity (exact title match: 1.0, fuzzy match > 0.9: 0.8, regex pattern only: 0.5).

#### Doc ↔ Structured Data: KPI Link

The `named_entities.kpis` list from each document is matched against:
1. `kpi_store.kpi_name` and `kpi_store.nl_formula` in the existing Nanite database
2. `md_attributes.semantic_role` for columns classified as `metric` or `measure`

Matching uses the same three-strategy fuzzy matcher already implemented in `dialog_agent/nodes/resolve_node.py` (exact → stem → edit distance ≤ 1). A confirmed match produces a `DESCRIBES_KPI` edge between the `DocNode` and the `KGNode` representing that KPI or attribute.

This is the highest-value link in the system: it connects a business analysis document directly to the data column that powers it.

#### Doc ↔ Structured Data: Table Reference

The `named_entities.organizations`, `named_entities.products`, and free-text topics are matched against `md_entities.table_name` and `md_entities.description`. A document that discusses "customer segments" is linked to a `customer_segments` table if one exists in the same domain's data source.

Match confidence is scored on:
- Exact table name mention in document text: 1.0
- Topic-level semantic match (e.g. "customer" topic → `customer_master` table): 0.7
- Domain co-location only (same domain, no specific match): 0.3

Only links above confidence 0.6 are written to the KG as `REFERENCES_TABLE`. Lower-confidence links are stored in `unstructured.db` for review but not exposed in the UI.

---

### 4.5 Document KG Writer

**Purpose:** Write `DocNode` entries and cross-modal edges to the existing Neo4j / TinkerPop instance. Never touches `:KGNode` entries.

**Neo4j node schema:**
```cypher
MERGE (d:DocNode {asset_id: $asset_id, kg_id: $source_id})
ON CREATE SET
  d.title         = $title,
  d.summary       = $summary,
  d.domain        = $domain,
  d.doc_type      = $doc_type,
  d.topics        = $topics,        // JSON string
  d.sensitivity   = $sensitivity,
  d.pii_risk      = $pii_risk,
  d.language      = $language,
  d.file_path     = $file_path,
  d.indexed_at    = $indexed_at
ON MATCH SET
  d.summary       = $summary,       // re-enrichment updates fingerprint
  d.topics        = $topics,
  d.indexed_at    = $indexed_at
```

**Edge types written by this service:**
```cypher
(DocNode)-[:SIMILAR_TOPIC    {similarity: float, basis: "topics"}]->(DocNode)
(DocNode)-[:REFERENCES       {confidence: float}                 ]->(DocNode)
(DocNode)-[:DESCRIBES_KPI    {confidence: float, kpi_name: str}  ]->(KGNode)
(DocNode)-[:REFERENCES_TABLE {confidence: float}                 ]->(KGNode)
(DocNode)-[:ABOUT_CONCEPT    {confidence: float}                 ]->(KGNode)
```

All edges carry a `confidence` property. The UI and NLQ layer can filter by confidence to control signal/noise tradeoff.

---

## 5. Data Flow — End to End

```
1. REGISTRATION
   Admin calls POST /unstructured/sources
   Body: {name, source_type:"s3", connection:{bucket, prefix}, source_id:"existing-nanite-source-id"}
   → Creates unstructured source record linked to existing Nanite data source

2. DISCOVERY
   File connector enumerates all files in source
   → Produces file manifests
   → Compares checksums against unstructured.db
   → Queues only new / changed files for processing

3. PARSING (per file, in parallel worker pool)
   Format parser reads file → extracts text window + structural metadata
   Raw text held in memory only
   Quality gate evaluation:
     PASS → forward to semantic extraction
     FAIL → store structural metadata only, mark enriched=false

4. SEMANTIC EXTRACTION (LLM, batched)
   Text window + domain context → LLM call
   Returns semantic fingerprint JSON
   Fingerprint written to unstructured.db (unstructured_assets table)
   Raw text discarded

5. CROSS-MODAL LINKING (async, after batch completes)
   For each newly indexed document:
   a. Embed topics → ANN index lookup → candidate similar docs
   b. Fuzzy-match KPIs → metadata API → DESCRIBES_KPI candidates
   c. Fuzzy-match entities → metadata API → REFERENCES_TABLE candidates
   d. Regex scan text window → known doc titles → REFERENCES candidates
   All candidates scored → confidence threshold applied
   Qualifying links written to unstructured.db and Neo4j

6. KG WRITE
   DocNode upserted in Neo4j
   Cross-modal edges written
   Existing :KGNode records untouched

7. SEARCH INDEX UPDATE
   Document title + summary + topics written to FTS5 search_index
   asset_type = "document"
   Searchable alongside existing tables and glossary terms

8. GOVERNANCE (if pii_risk=true or sensitivity=confidential)
   Governance workflow created automatically
   Same workflow engine as existing certification/access-request flows
```

---

## 6. Schema Design

### `unstructured.db` — new SQLite file, separate from `metadata_catalog.db`

```sql
CREATE TABLE doc_sources (
    source_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    source_type     TEXT NOT NULL,  -- local|s3|gcs|azure|sharepoint|gdrive
    connection_json TEXT NOT NULL DEFAULT '{}',
    nanite_source_id TEXT,          -- FK to existing md_sources (soft link, not enforced)
    domain          TEXT NOT NULL DEFAULT '',
    enabled         INTEGER NOT NULL DEFAULT 1,
    last_indexed_at TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE unstructured_assets (
    asset_id        TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES doc_sources(source_id),
    file_path       TEXT NOT NULL,
    file_name       TEXT NOT NULL,
    file_type       TEXT NOT NULL,  -- pdf|docx|pptx|xlsx|html|md|txt
    checksum        TEXT NOT NULL,
    size_bytes      INTEGER,
    page_count      INTEGER,
    -- Structural metadata (always populated)
    title           TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    created_at_file TEXT,
    modified_at_file TEXT,
    -- Semantic fingerprint (populated after LLM extraction)
    enriched        INTEGER NOT NULL DEFAULT 0,
    summary         TEXT NOT NULL DEFAULT '',
    domain          TEXT NOT NULL DEFAULT '',
    doc_type        TEXT NOT NULL DEFAULT '',  -- report|policy|contract|presentation|research|manual|correspondence
    language        TEXT NOT NULL DEFAULT 'en',
    topics_json     TEXT NOT NULL DEFAULT '[]',
    entities_json   TEXT NOT NULL DEFAULT '{}',
    time_refs_json  TEXT NOT NULL DEFAULT '[]',
    sensitivity     TEXT NOT NULL DEFAULT 'internal',  -- public|internal|confidential|restricted
    pii_risk        INTEGER NOT NULL DEFAULT 0,
    ocr_used        INTEGER NOT NULL DEFAULT 0,
    ocr_confidence  REAL,
    -- Lifecycle
    deleted         INTEGER NOT NULL DEFAULT 0,
    indexed_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE doc_relationships (
    rel_id          TEXT PRIMARY KEY,
    from_asset_id   TEXT NOT NULL,
    to_asset_id     TEXT,           -- NULL for cross-modal links (other side is in Nanite)
    to_nanite_id    TEXT,           -- entity_id or attr_id in metadata_catalog.db
    to_nanite_type  TEXT,           -- 'entity'|'attribute'|'kpi'
    rel_type        TEXT NOT NULL,  -- SIMILAR_TOPIC|REFERENCES|DESCRIBES_KPI|REFERENCES_TABLE|ABOUT_CONCEPT
    confidence      REAL NOT NULL,
    basis           TEXT NOT NULL DEFAULT '',  -- how the link was detected
    created_at      TEXT NOT NULL
);

CREATE TABLE doc_topic_embeddings (
    asset_id        TEXT PRIMARY KEY REFERENCES unstructured_assets(asset_id),
    embedding_json  TEXT NOT NULL,  -- JSON array of floats (topic embedding, not content embedding)
    model           TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- FTS5 for document search (title + summary + topics only — never full content)
CREATE VIRTUAL TABLE doc_search USING fts5(
    asset_id UNINDEXED,
    title,
    summary,
    topics,
    domain UNINDEXED,
    doc_type UNINDEXED
);
```

---

## 7. Knowledge Graph Integration

### Node label strategy

| Label | Created by | Queries that use it |
|---|---|---|
| `:KGNode` | Existing Nanite KG agent | All existing Cypher/Gremlin queries |
| `:DocNode` | New unstructured service | New document queries only |

Existing queries are written as `MATCH (n:KGNode ...)` — they will never traverse `:DocNode` entries. New document queries start from `:DocNode`. Cross-modal edges connect the two label namespaces but are only traversed by explicit cross-modal queries.

### Cross-modal traversal patterns

**"What documents are related to this table?"**
```cypher
MATCH (n:KGNode {uri: $table_uri, kg_id: $kg_id})
MATCH (d:DocNode)-[:REFERENCES_TABLE|DESCRIBES_KPI]->(n)
WHERE d.confidence >= 0.6
RETURN d.title, d.summary, d.doc_type, d.domain
ORDER BY d.indexed_at DESC
```

**"What KPIs in the data model does this document discuss?"**
```cypher
MATCH (d:DocNode {asset_id: $asset_id})
MATCH (d)-[r:DESCRIBES_KPI]->(n:KGNode)
RETURN n.uri, n.label, r.confidence, r.kpi_name
ORDER BY r.confidence DESC
```

**"What other documents are topically similar to this one?"**
```cypher
MATCH (d:DocNode {asset_id: $asset_id})
MATCH (d)-[r:SIMILAR_TOPIC]->(d2:DocNode)
WHERE r.similarity >= 0.85
RETURN d2.title, d2.summary, d2.doc_type, r.similarity
ORDER BY r.similarity DESC LIMIT 10
```

**"Show me everything — structured and unstructured — related to trade spend"**
```cypher
MATCH (n:KGNode)
WHERE toLower(n.label) CONTAINS 'trade' OR toLower(n.label) CONTAINS 'spend'
OPTIONAL MATCH (d:DocNode)-[:DESCRIBES_KPI|REFERENCES_TABLE]->(n)
RETURN
  'structured' AS asset_class, n.label AS title, n.uri AS id, null AS summary
UNION ALL
MATCH (d2:DocNode)
WHERE any(t IN d2.topics WHERE toLower(t) CONTAINS 'trade spend')
RETURN
  'document' AS asset_class, d2.title AS title, d2.asset_id AS id, d2.summary AS summary
```

This last query is the proof-of-concept for the unified structured + unstructured knowledge surface.

---

## 8. Cross-Modal Linking Algorithm

The cross-modal linker is the most novel component in the system. It solves a genuinely hard problem: given a document fingerprint (topics, named entities, KPI mentions) and a structured data model (table names, column semantic roles, KPI definitions), find meaningful semantic links without requiring any content similarity computation.

### KPI Link Detection — step by step

```
INPUT:
  document.entities.kpis = ["RSV", "GSV", "trade spend %", "distribution gain bps"]
  nanite.kpi_store = [{kpi_name: "RSV", nl_formula: "Retail Sales Value"}, ...]
  nanite.md_attributes where semantic_role IN ('metric', 'measure', 'kpi')

STEP 1 — Exact match
  For each doc_kpi in document.entities.kpis:
    if doc_kpi.lower() in {k.kpi_name.lower() for k in kpi_store}:
      → DESCRIBES_KPI edge, confidence=1.0

STEP 2 — Abbreviation expansion
  For unmatched doc_kpis that look like abbreviations (len <= 6, all uppercase):
    Call LLM with domain context to expand abbreviation
    "RSV" + domain="CPG/RGM" → "Retail Sales Value"
    Retry exact match with expansion
    → DESCRIBES_KPI edge, confidence=0.9

STEP 3 — Fuzzy match (same 3-strategy algorithm as resolve_node.py)
  For remaining unmatched doc_kpis:
    Strategy 1: token substring match against kpi_name and semantic_role
    Strategy 2: stemmed token match
    Strategy 3: edit distance ≤ 1 for tokens ≥ 5 chars
    Best match score > 0 → DESCRIBES_KPI edge, confidence = 0.5 + (score/max_score * 0.4)

STEP 4 — Confidence threshold application
  confidence >= 0.8 → written to KG immediately
  confidence 0.6–0.79 → written to KG, flagged for human review
  confidence < 0.6 → stored in doc_relationships for observability, not written to KG
```

### Why reusing `resolve_node.py`'s fuzzy matcher matters

The same algorithm that resolves user filter values in NLQ queries ("Premium Sachet" → `pack_type = 'PREMIUM SACHET'`) is reused for cross-modal linking. This is deliberate. It means the cross-modal linker has the same accuracy profile as the query resolver — a system that has been validated against real enterprise data models across multiple clients. It also means that improvements to the fuzzy matcher automatically improve cross-modal link quality.

---

## 9. Isolation Contract with Existing Nanite

This section formally defines what the unstructured service is and is not allowed to do with respect to the existing system. This contract must be maintained in all future development.

### Permitted

| Action | Rationale |
|---|---|
| `GET` requests to existing metadata API endpoints | Read-only, no side effects |
| Reading Neo4j nodes with label `:KGNode` | Read-only traversal for cross-modal detection |
| Writing Neo4j nodes with label `:DocNode` | New label namespace, invisible to existing queries |
| Writing Neo4j edges from `:DocNode` to `:KGNode` | Cross-modal edges, not traversed by existing queries |
| Writing to `unstructured.db` | Separate file, no lock contention |
| Writing to `search_index` FTS5 table with `asset_type='document'` | New asset type, existing queries filter by other types |
| Adding two proxy routes to `orchestrator_api.py` | Additive only, isolated error surface |
| Adding a new UI tab | Additive, no existing component modified |

### Prohibited

| Action | Rationale |
|---|---|
| `POST`/`PATCH`/`DELETE` to any existing metadata API | Could corrupt production data |
| Modifying any row in `md_sources`, `md_entities`, `md_attributes`, `md_changes` | Existing data is owned by the existing service |
| Altering any existing table schema | Migration failures kill the existing service on restart |
| Writing Neo4j nodes with label `:KGNode` | Would appear in existing queries and corrupt KG results |
| Modifying any existing KG node properties | Side effects invisible to existing queries |
| Importing from `dialog_agent`, `ontology_agent`, `shacl_agent` | Creates a dependency that breaks isolation |
| Calling the dialog agent's internal API | Creates circular dependency risk |
| Sharing the Anthropic API client instance | Rate limit contention — each service must use its own |

---

## 10. Novelty and Differentiation

### What exists today

**Document search platforms** (Elastic, Solr, Coveo): Index full text, enable keyword and semantic search within documents. No understanding of what the document means. No connection to structured data. Privacy-risky (full content indexed).

**Enterprise content management** (SharePoint, Confluence, Notion): Store and organise documents. Basic metadata (author, date, tags). No semantic extraction. No connection to data models. No knowledge graph.

**RAG systems** (LlamaIndex, LangChain RAG pipelines): Chunk documents, embed chunks, retrieve relevant chunks at query time. Content is indexed. Answers come from document text. No connection to structured data models. Privacy-risky.

**Data catalogs with document linking** (Collibra, Alation): Allow users to *manually* attach documents to data assets. No automatic semantic extraction. No relationship detection. The link is as good as the person who made it.

### What DataNanite's unstructured layer does differently

**1. Semantic fingerprinting without content indexing**

Every other document intelligence system indexes content. DataNanite indexes *meaning*. The semantic fingerprint (summary, topics, entities, KPIs) is computed once from the document and the source text is discarded. This is architecturally novel — it gives you the benefits of document understanding with the privacy profile of a pure metadata system.

No other enterprise data platform offers this boundary.

**2. Automatic cross-modal linking — document to structured data**

The `DESCRIBES_KPI` and `REFERENCES_TABLE` edges are detected automatically, without any user input. A pricing policy document is automatically linked to the `price_index` column in the fact table. A quarterly RGM review is automatically linked to the `rsv`, `gsv`, and `trade_spend_pct` KPIs.

This has never been done automatically at the semantic level. Collibra and Alation allow *manual* linking — a data steward must explicitly attach a document to a data asset. DataNanite detects these links from the document's own language using the same domain-aware entity resolution that powers the NLQ query resolver.

**3. Domain-aware entity extraction**

The same industry signal sets and domain disambiguation hints that make DataNanite's taxonomy inference accurate for CPG, Banking, Telecom, and Life Sciences are applied to document entity extraction. "RSV" means Retail Sales Value in a CPG document, not Respiratory Syncytial Virus (a medical abbreviation). "NII" means Net Interest Income in a Banking document. No general-purpose document intelligence system makes these distinctions without custom configuration.

**4. Unified structured + unstructured knowledge graph**

After both pipelines run, a single graph traversal can answer questions that span structured data and documents. "Show me everything DataNanite knows about trade spend" returns: fact table columns, KPI definitions, ontology concepts, *and* the 12 documents that discuss trade spend — all from one graph walk. No other platform connects these two asset classes in a single graph without manual curation.

**5. Reuse of existing NLQ and fuzzy-match infrastructure**

The cross-modal linker reuses `_fuzzy_match_candidates()` from `resolve_node.py` — the same algorithm validated against real enterprise data models for the NLQ query resolver. Most document intelligence systems build separate entity-linking pipelines. DataNanite's link quality inherits from a pipeline already proven in production.

**6. Governance-native from day one**

PII detection and sensitivity classification are first-class outputs of the semantic extraction pipeline — not post-processing add-ons. A document flagged `pii_risk: true` automatically triggers a governance workflow in the existing certification engine. A document marked `sensitivity: confidential` automatically receives an access policy. No other document intelligence platform integrates this tightly with a governance workflow engine.

---

## 11. Known Complexities and Mitigations

| Complexity | Impact | Mitigation |
|---|---|---|
| OCR quality on scanned documents | Wrong entities extracted, cross-modal links missed | OCR confidence gate; conservative extraction prompt for low-confidence OCR; flag for manual review |
| Entity disambiguation (Apple: company vs. fruit) | False cross-modal links | Domain-scoped resolution; confidence threshold keeps low-quality links out of KG |
| Topic similarity false positives | Unrelated documents linked as similar | ANN + cosine threshold; `WEAKLY_SIMILAR` edge type for borderline pairs; human review queue |
| LLM cost at scale (100K+ documents) | High API spend | Quality gate filters junk; tiered model selection (haiku vs. sonnet); incremental indexing (only changed files) |
| Neo4j write contention under bulk load | Slows existing KG queries | Batched writes with back-pressure; off-peak scheduling; separate Neo4j database option |
| Document freshness (frequent auto-saves) | Re-processing unchanged documents | Content hash (not filesystem timestamp) as change signal |
| Cross-modal link accuracy depends on structured model quality | Links miss if `semantic_role` is empty | Run taxonomy enrichment before document indexing; enrich on-demand if stale |
| Multimodal content (charts, tables in PDFs) | Rich business content not extracted | Phase 2 enhancement: vision LLM call per page for high-value document types; cost-gated |
| Non-English document content | Entity extraction quality degrades | Language detection gate; multilingual embedding model for topic similarity; English-first rollout |
| "No content" boundary is philosophically ambiguous | Design drift over time | Formal definition in isolation contract (Section 9); code review checklist item |
