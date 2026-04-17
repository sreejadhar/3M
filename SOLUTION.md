# DataNanite — Complete Solution Documentation

> **Last updated:** 2026-04-17

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Service Registry](#service-registry)
4. [Metadata Extraction Agent](#metadata-extraction-agent)
5. [Ontology Agent](#ontology-agent)
6. [Knowledge Graph Agent](#knowledge-graph-agent)
7. [Dialog with Data Agent](#dialog-with-data-agent)
8. [Conformity Agent](#conformity-agent)
9. [SHACL Validation Agent](#shacl-validation-agent)
10. [Orchestrator API](#orchestrator-api)
11. [Tech UI](#tech-ui)
12. [DataChat UI](#datachat-ui)
13. [Docker Setup](#docker-setup)
14. [Data Flow](#data-flow)
15. [SQL Post-Processing Pipeline](#sql-post-processing-pipeline)
16. [Domain Inference](#domain-inference)
17. [GraphRAG — Hybrid Semantic Retrieval](#graphrag--hybrid-semantic-retrieval)
18. [KG Isolation — Multi-KG Support](#kg-isolation--multi-kg-support)
19. [Configuration Reference](#configuration-reference)

---

## Overview

DataNanite is an AI-native metadata intelligence platform that connects to enterprise databases, automatically extracts full schema and statistical metadata, infers an OWL/RDF ontology, builds a knowledge graph, and exposes everything through a natural-language query interface — with zero manual curation and no data movement.

The system is built on **seven independently deployable microservices**, each a FastAPI + LangGraph pipeline:

| Service | Purpose | Port |
|---|---|---|
| **Metadata Extraction Agent** | Connects to a database and extracts schema, statistics, functional dependencies, inclusion dependencies, and cardinality relationships | 8000 |
| **Ontology Agent** | Reads a metadata report and generates a formal OWL/RDF ontology with LLM concept annotation | 8001 |
| **Knowledge Graph Agent** | Converts OWL/RDF to Cypher (Neo4j) or Gremlin (TinkerPop) and executes on a live graph database | 8002 |
| **Dialog with Data Agent** | Accepts natural language queries, retrieves schema context via GraphRAG, plans and executes SQL, and synthesizes insights | 8003 |
| **Conformity Agent** | Validates data quality and conformity rules against indexed sources | 8004 |
| **Orchestrator / Chat UI** | End-to-end pipeline orchestration, session management, source registry, REST proxy for all agents | 8005 |
| **Tech UI** | Engineer workbench — pipeline monitor, KG visualizer, ontology editor, SHACL validation | 8006 |
| **SHACL Validation Agent** | Validates OWL/RDF ontologies against SHACL quality shapes; read-only, opt-in | 8007 |

All services are **completely decoupled**: zero cross-package imports. They communicate only through JSON over HTTP.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Docker Network: metadata-net                          │
│                                                                                 │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐               │
│  │  Tech UI     │   │  Chat UI /        │   │  agent-api       │ :8000         │
│  │  :8006       │──►│  Orchestrator     │──►│  (metadata       │               │
│  │  (Engineer   │   │  :8005            │   │   extraction)    │               │
│  │   Workbench) │   │                   │   └──────────────────┘               │
│  └──────────────┘   │   Central proxy   │   ┌──────────────────┐               │
│                     │   for all agents  │──►│  ontology-api    │ :8001         │
│                     │                   │   └──────────────────┘               │
│                     │                   │──►┌──────────────────┐               │
│                     │                   │   │  kg-api          │ :8002         │
│                     │                   │   └──────────────────┘               │
│                     │                   │──►┌──────────────────┐               │
│                     │                   │   │  dialog-api      │ :8003         │
│                     │                   │   └──────────────────┘               │
│                     │                   │──►┌──────────────────┐               │
│                     │                   │   │  conformity-api  │ :8004         │
│                     │                   │   └──────────────────┘               │
│                     │                   │──►┌──────────────────┐               │
│                     └───────────────────┘   │  shacl-api       │ :8007         │
│                                             │  (opt-in, r/o)   │               │
│                                             └──────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────────┘
         │                          │                    │
         ▼                          ▼                    ▼
┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────────────┐
│  Source Database │  │  Source Database     │  │  Graph Database          │
│  PostgreSQL /    │  │  (Dialog SQL target) │  │  Neo4j (bolt://) or      │
│  SQL Server /    │  │  any supported DB    │  │  Gremlin (ws://)         │
│  Oracle /        │  └─────────────────────┘  └──────────────────────────┘
│  BigQuery /      │
│  Snowflake /     │
│  SQLite / CSV    │
└──────────────────┘
```

**Key design decisions:**

- The Orchestrator is the sole cross-service coordinator. Upstream services never call downstream services.
- All SQL execution is **read-only**; row limits are enforced post-LLM by the post-processing pipeline.
- The SHACL service is **opt-in and stateless** — it never modifies any source, ontology, or KG data.
- All connection credentials are encrypted at rest and never exposed to the AI or query layer.
- Microservices use Docker internal DNS (`http://service-name:port`) for inter-service communication.

---

## Service Registry

| Service | Default URL | Env var override | Dockerfile |
|---|---|---|---|
| agent-api | `http://localhost:8000` | `METADATA_API_URL` | `Dockerfile.agent` |
| ontology-api | `http://localhost:8001` | `ONTOLOGY_API_URL` | `Dockerfile.ontology` |
| kg-api | `http://localhost:8002` | `KG_API_URL` | `Dockerfile.kg` |
| dialog-api | `http://localhost:8003` | `DIALOG_API_URL` | `Dockerfile.dialog` |
| conformity-api | `http://localhost:8004` | `CONFORMITY_API_URL` | `Dockerfile.conformity` |
| chat-ui (orchestrator) | `http://localhost:8005` | — | `Dockerfile.chat` |
| tech-ui | `http://localhost:8006` | — | _(static + proxy)_ |
| shacl-api | `http://localhost:8007` | `SHACL_API_URL` | `Dockerfile.shacl` |

---

## Metadata Extraction Agent

### Package Structure

```
metadata_agent/
├── __init__.py               # Exports: MetadataExtractionAgent, AgentConfig, DBConfig, DBType
├── config.py                 # AgentConfig, DBConfig, DBType dataclasses
├── state.py                  # AgentState TypedDict
├── agent.py                  # LangGraph graph + MetadataExtractionAgent class
├── nodes/
│   ├── connection_node.py    # Open DB connection via ConnectorFactory
│   ├── discovery_node.py     # List all tables in the target schema
│   ├── extraction_node.py    # Extract schema + statistics per table
│   ├── analysis_node.py      # Detect FDs, INDs, cardinality relationships
│   └── report_node.py        # Aggregate results into final report dict
├── tools/
│   ├── schema_extractor.py   # Column metadata: name, type, nullability, PK, FK
│   ├── metadata_collector.py # Row count, null counts, distinct counts, min/max/avg
│   ├── fd_detector.py        # Functional dependency detection via value hashing
│   ├── id_detector.py        # Inclusion dependency (FK candidate) detection
│   └── cardinality_analyzer.py  # 1:1 / 1:N / M:N relationship classification
└── connectors/
    ├── base.py               # Abstract BaseConnector interface
    ├── factory.py            # ConnectorFactory — maps DBType to connector class
    ├── postgres.py           # PostgreSQL (psycopg2)
    ├── oracle.py             # Oracle (oracledb)
    ├── sqlserver.py          # SQL Server (pyodbc)
    ├── teradata.py           # Teradata (teradatasql)
    ├── redshift.py           # Amazon Redshift (psycopg2)
    ├── bigquery.py           # Google BigQuery (google-cloud-bigquery)
    ├── snowflake.py          # Snowflake (snowflake-connector-python)
    ├── sqlite.py             # SQLite / CSV / Excel (in-memory)
    └── delta_lake.py         # Delta Lake (PySpark)
```

### LangGraph Pipeline

```
START → connection_node → discovery_node → extraction_node → analysis_node → report_node → END
                ↓ error          ↓ error
            error_end ──────────────────────────────────────────────────────────► END
```

### Report Format (key fields)

```json
{
  "database_type": "sqlserver",
  "schema": "dbo",
  "tables": {
    "fact_orders": {
      "row_count": 5000000,
      "columns": [
        {
          "name": "order_id",
          "data_type": "int",
          "is_primary_key": true,
          "unique_count": 5000000,
          "null_count": 0
        }
      ],
      "foreign_keys": [...]
    }
  },
  "functional_dependencies": [...],
  "inclusion_dependencies": [...],
  "cardinality_relationships": [...]
}
```

---

## Ontology Agent

### Package Structure

```
ontology_agent/
├── __init__.py
├── config.py     # OntologyConfig dataclass
├── state.py      # OntologyState TypedDict
├── agent.py      # LangGraph pipeline
└── nodes/
    ├── load_node.py      # Validate incoming metadata report
    ├── build_node.py     # Construct rdflib OWL graph + LLM concept annotation
    └── serialize_node.py # Serialize to Turtle / RDF-XML / N3
```

### LangGraph Pipeline

```
START → load_node → build_node → serialize_node → END
              ↓ error
          error_end → END
```

### OWL Mapping

| Metadata element | OWL/RDF representation |
|---|---|
| Table | `owl:Class` with `rdfs:label`, `rdfs:comment` (entity type, row count, domain) |
| Column | `owl:DatatypeProperty` with `rdfs:domain` (table), `rdfs:range` (XSD type) |
| PK column | + `owl:FunctionalProperty` + `owl:InverseFunctionalProperty` |
| NOT NULL column | `owl:Restriction` minCardinality 1 as `rdfs:subClassOf` |
| Explicit FK | `owl:ObjectProperty` domain=child, range=parent |
| FK candidate (IND) | `owl:ObjectProperty` with coverage % in `rdfs:comment` |
| 1:N cardinality | `owl:FunctionalProperty` on ObjectProperty |
| 1:1 cardinality | `owl:FunctionalProperty` + `owl:InverseFunctionalProperty` |
| Functional dependency | `rdfs:comment` on owning class: `FD-ANNOTATION: [det] → [dep]` |
| Column statistics | `rdfs:comment` on DatatypeProperty: unique/null/min/max/avg |
| Business concept | `rdfs:comment`: `Business concept: avg-revenue-per-user` |

### LLM Concept Annotation

`build_node` calls Claude Haiku once per table with per-column evidence (data type, min/max, top values, null rate, rule-based domain) to resolve domain-specific abbreviations into standard business concept labels, grounded in observed data rather than model priors. Domain context (e.g. `"CPG/RGM"`) is injected so ambiguous abbreviations resolve correctly per industry.

---

## Knowledge Graph Agent

### Package Structure

```
knowledge_graph_agent/
├── __init__.py
├── config.py      # KGConfig dataclass
├── state.py       # KGState TypedDict
├── agent.py       # LangGraph pipeline
└── nodes/
    ├── parse_node.py     # Parse OWL/Turtle with rdflib
    ├── translate_node.py # OWL → Cypher / Gremlin + graph_data for UI
    ├── execute_node.py   # Execute on Neo4j or Gremlin
    ├── fetch_node.py     # Load existing KG snapshot from database
    ├── profile_node.py   # Column taxonomy profiling via LLM
    └── embed_node.py     # GraphRAG embedding storage (Neo4j HNSW index)
```

### LangGraph Pipeline

```
START → parse_node → translate_node → execute_node → [embed_node] → END
              ↓ error
          error_end → END
```

### OWL → Graph Mapping

| OWL element | Graph representation |
|---|---|
| `owl:Class` | Node with label = class name, `uri`, `kg_id` properties |
| `owl:DatatypeProperty` | Node property attribute (xsd type as metadata) |
| `owl:ObjectProperty` | Directed edge with semantic verb label + cardinality |
| `owl:FunctionalProperty` | Edge cardinality = `1:N` |
| + `owl:InverseFunctionalProperty` | Edge cardinality = `1:1` |

**Multi-KG isolation:** Every node and edge is stamped with `kg_id` = `source_id`. Multiple sources coexist in the same graph database with zero URI collisions. All MATCH/MERGE queries filter by `kg_id`.

**Semantic edge labels:** Relationship edges carry human-readable verb phrases (e.g. `has Customer (1:N)`) rather than raw cardinality codes, derived from the domain and direction of the ObjectProperty.

---

## Dialog with Data Agent

### Package Structure

```
dialog_agent/
├── __init__.py
├── config.py      # DialogConfig dataclass
├── state.py       # DialogState, SQLQuery, QueryResult TypedDicts
├── agent.py       # LangGraph pipeline
└── nodes/
    ├── understand_node.py  # Build schema context from KG nodes/edges
    ├── resolve_node.py     # Pre-resolve categorical values via fuzzy match
    ├── plan_node.py        # LLM decomposes NQL into SQL + post-processing pipeline
    ├── retrieve_node.py    # GraphRAG retrieval — semantic KG node search
    ├── execute_node.py     # Execute SQL against target database
    └── synthesize_node.py  # LLM synthesizes results into narrative insights
```

### LangGraph Pipeline

```
START → understand_node → retrieve_node → resolve_node → plan_node → execute_node → synthesize_node → END
```

### SQL Post-Processing Pipeline

Every LLM-generated SQL statement passes through a sequential post-processor before execution:

```python
sql = _qualify_sql(sql, db_schema, table_labels)          # prefix bare table names with schema
sql = _fix_count_vs_sum(sql, natural_query)               # COUNT vs SUM intent correction
sql = _fix_percentage(sql, natural_query, db_type)        # percentage calculation normalization
sql = _enforce_sql_limits(sql, row_limit, db_type)        # inject TOP/LIMIT/FETCH per dialect
sql = _fix_dialect_syntax(sql, db_type)                   # cross-dialect contamination fixer
sql = _fix_sqlserver_subquery_limits(sql, db_type)        # ORDER BY → OFFSET in SQL Server CTEs
sql = _fix_subquery_order_by(sql, db_type)                # strip bare ORDER BY in subqueries
sql = _fix_window_functions(sql, db_type)                 # inject missing ORDER BY in OVER()
sql = _fix_multicolumn_subquery(sql)                      # scalar subquery → single column
sql = _fix_distinct_order_by(sql)                         # add missing ORDER BY cols to DISTINCT
```

**`_fix_dialect_syntax`** — Runtime cross-dialect contamination fixer (all dialects):

| Transformation | Target dialects |
|---|---|
| `ILIKE` → `LOWER(col) LIKE LOWER(pat)` | SQL Server, Oracle, BigQuery, SQLite |
| `col::TYPE` → `CAST(col AS TYPE)` | SQL Server, Oracle, BigQuery, SQLite |
| `NOW()` → `GETDATE()` / `SYSDATE` / `CURRENT_TIMESTAMP()` | SQL Server / Oracle / BigQuery |
| `CURRENT_DATE` → dialect equivalent | all non-PostgreSQL |
| `DATE_TRUNC(unit, col)` → `DATEADD/DATEDIFF` | SQL Server |
| `DATE_TRUNC(unit, col)` → `TRUNC(col, fmt)` | Oracle |
| `LENGTH(` → `LEN(` | SQL Server |
| `LIMIT N` (statement end) → `FETCH FIRST N ROWS ONLY` | Oracle |
| `col \|\| other` → `col + other` | SQL Server |

**`_fix_subquery_order_by`** — Depth-aware OFFSET guard:
Strips bare `ORDER BY` from SQL Server subqueries/CTEs unless protected by `TOP`, `OFFSET`, or `FOR XML` at **paren-depth 0** within the block — preventing false-negatives when a nested subquery already has `OFFSET` added by `_fix_sqlserver_subquery_limits`.

### Synthesize Node

`synthesize_node` calls Claude with `max_tokens=4096` to produce a three-section narrative:

```
## Summary
## Key Findings
## Analysis
```

The `plan_explanation` field is guarded against raw JSON leakage — discarded if it starts with `[` or `{` or contains `"query_id"`.

---

## Conformity Agent

Validates data quality rules against indexed sources.

```
conformity_agent/
├── config.py
├── state.py
├── agent.py
└── nodes/
    ├── load_node.py
    ├── check_node.py
    └── report_node.py
```

**Port:** 8004 | **Dockerfile:** `Dockerfile.conformity` | **Requirements:** `requirements.conformity.txt`

---

## SHACL Validation Agent

Read-only, opt-in service that validates OWL/RDF ontologies against SHACL quality shapes. Does not modify any source, ontology, or KG data.

### Package Structure

```
shacl_agent/
├── __init__.py             # Exports: SHACLAgent, SHACLConfig
├── config.py               # SHACLConfig dataclass
├── state.py                # SHACLState TypedDict
├── agent.py                # LangGraph pipeline
├── nodes/
│   ├── parse_node.py       # Detect format, parse ontology, load shapes
│   ├── validate_node.py    # SHACL structural check + semantic checks
│   └── report_node.py      # Build quality report with suggestions
└── shapes/
    └── ontology_quality.ttl  # Built-in SHACL shapes
```

### LangGraph Pipeline

```
START → parse_node → validate_node → report_node → END
              ↓ error
          error_end → END
```

### Built-in SHACL Shapes

| Shape | What it validates |
|---|---|
| `ClassCompletenessShape` | `owl:Class` has `rdfs:label` and `rdfs:comment` |
| `DatatypePropertyShape` | Column props have `rdfs:domain`, `rdfs:range`, `rdfs:label` |
| `ObjectPropertyShape` | FK/IND edges have `rdfs:domain`, `rdfs:range`, `rdfs:comment` |
| `OntologyHeaderShape` | `owl:Ontology` has `rdfs:label` |
| `FunctionalPropertyClassification` | Functional markers typed as D or O property |

### Semantic Checks (Python)

| Check | Description |
|---|---|
| `OrphanClass` | Classes never referenced as domain/range of any property |
| `LowCoverage` | ObjectProperty edges with IND coverage < `min_coverage` threshold |
| `NamespaceDrift` | URIs outside the declared `owl:Ontology` base namespace |
| `DuplicateClassLabel` | Two+ classes sharing the same `rdfs:label` |

### Quality Labels

| Label | Meaning |
|---|---|
| `PASS` | Fully conformant — no violations or semantic issues |
| `WARN` | No violations but warnings or suggestions exist |
| `FAIL` | At least one `sh:Violation` |

### API Endpoints

```
POST /validate            submit ontology text or ontology_job_id → {job_id}
GET  /jobs/{job_id}       poll status + summary counts
GET  /jobs/{job_id}/report  full report (violations, warnings, suggestions)
GET  /list                list all jobs
GET  /health
```

**Port:** 8007 | **Dockerfile:** `Dockerfile.shacl` | **Requirements:** `requirements.shacl.txt`

---

## Orchestrator API

**File:** `orchestrator_api.py` | **Port:** 8005

Central FastAPI service that:
- Registers and indexes data sources (end-to-end pipeline: metadata → ontology → KG → dialog)
- Proxies all agent APIs with a uniform REST surface
- Manages sessions and conversation history
- Streams pipeline progress via Server-Sent Events
- Stores source metadata, ontology content, and KG snapshots in SQLite/PostgreSQL

### Key Endpoints

```
POST   /sources                           register + index a source
GET    /sources                           list all sources
PATCH  /sources/{id}                      update name/description/domain (admin)
GET    /sources/{id}/graph                KG graph data {nodes, edges}
GET    /sources/{id}/ontology             raw ontology text
POST   /sources/{id}/ontology             save edited ontology + rebuild KG
POST   /sources/{id}/validate-ontology    SHACL validation (opt-in, read-only)
POST   /sources/{id}/kg-preview           preview KG without saving
POST   /sources/{id}/reindex              trigger re-indexing

POST   /sessions                          create chat session
POST   /sessions/{id}/chat                send NL query
GET    /sessions/{id}/events              SSE pipeline progress stream
```

### Domain Inference

Two-tier signal detection at index time:

- **Industry tier** (`_INDUSTRY_SIGNALS`): CPG, Life Sciences, Healthcare, Telecom, Banking/FS, Insurance, Retail, E-commerce, Manufacturing, SaaS
- **Function tier** (`_FUNCTION_SIGNALS`): RGM, FP&A, Supply Chain, Sales, Marketing, HR/People, Operations, CX

When both tiers fire, a compound label is returned: `"CPG/Supply Chain"`, `"LS/FP&A"`, etc.

---

## Tech UI

**File:** `tech_ui/` | **Port:** 8006 (served by `tech_ui_server.py`)

Engineer workbench with five views:

| View | Functionality |
|---|---|
| Pipeline Monitor | Source registration, indexing status, re-index, domain edit |
| KG Explorer | Interactive vis.js graph of KG nodes and edges |
| Ontology Viewer | OWL/Turtle editor, SPARQL query, ontology class graph |
| SQL Workbench | Multi-query SQL runner against indexed sources |
| Catalog | Metadata catalog browser |

### Ontology Validation (SHACL)

The **Validate** button in the ontology toolbar sends the current editor content to the SHACL service and renders a full report modal:

- Traffic-light badge: `PASS` / `WARN` / `FAIL`
- Stats bar: class count, relationship count, column count, triple count, violation count, warning count
- Recommendations banner (actionable suggestions)
- Per-check result rows — every check shown with pass/fail status and description, failed checks expandable to show affected node URI and exact message
- Validates **current editor content** (unsaved edits), not the stored ontology

---

## DataChat UI

**File:** `chat_ui/` | served by Orchestrator on port 8005

Conversational interface for business users:
- Natural language query → SQL → results → narrative insights
- Multi-turn conversation with session history
- Source selector and multi-source queries
- Streaming pipeline progress (SSE)

---

## Docker Setup

```yaml
# docker-compose.yml services
agent-api        :8000   Dockerfile.agent
ontology-api     :8001   Dockerfile.ontology
kg-api           :8002   Dockerfile.kg
dialog-api       :8003   Dockerfile.dialog
conformity-api   :8004   Dockerfile.conformity
chat-ui          :8005   Dockerfile.chat
tech-ui          :8006   (static + proxy)
shacl-api        :8007   Dockerfile.shacl
```

Start all services:
```bash
docker-compose up -d
```

Start SHACL service only (standalone):
```bash
pip install -r requirements.shacl.txt
python shacl_api.py   # port 8007
```

---

## Data Flow

### Full Indexing Pipeline

```
1. User registers source (name, DB type, credentials) via Tech UI or Chat UI

2. Orchestrator:
   a. POST /agent-api/run         → metadata report JSON
   b. POST /ontology-api/generate → ontology Turtle (async job)
   c. POST /kg-api/generate       → KG graph (async job)
   d. Stores report + ontology + KG snapshot in kg_store

3. Source status → "ready"

4. (Optional) User clicks Validate in Tech UI:
   POST /orchestrator/sources/{id}/validate-ontology
   → proxied to SHACL API → returns quality report modal
```

### Natural Language Query Flow

```
1. User sends NL query in Chat UI

2. Orchestrator POST /dialog-api/query with:
   - natural_query
   - kg_nodes + kg_edges (from stored KG snapshot)
   - db connection config

3. Dialog pipeline:
   understand_node  → build schema context string from KG
   retrieve_node    → GraphRAG: embed query, cosine similarity over KG nodes
   resolve_node     → fuzzy-match categorical values from catalog
   plan_node        → LLM generates SQL queries + post-processing pipeline
   execute_node     → execute SQL against live source DB
   synthesize_node  → LLM generates ## Summary / ## Key Findings / ## Analysis

4. Response streamed back to UI
```

---

## SQL Post-Processing Pipeline

See [Dialog with Data Agent](#dialog-with-data-agent) for the full ordered pipeline. Dialect rules enforced at prompt level by `_build_dialect_rules()` and at runtime by the post-processors.

### Dialect Rules Matrix

| Feature | SQLite | PostgreSQL | SQL Server | Oracle | BigQuery |
|---|---|---|---|---|---|
| Row limit | `LIMIT N` | `LIMIT N` | `OFFSET 0 ROWS FETCH NEXT N ROWS ONLY` | `FETCH FIRST N ROWS ONLY` | `LIMIT N` |
| Current date | `DATE('now')` | `NOW()` | `CAST(GETDATE() AS DATE)` | `SYSDATE` | `CURRENT_DATE()` |
| String concat | `\|\|` | `\|\|` | `+` | `\|\|` | `\|\|` |
| Casting | `CAST()` | `CAST()` or `::` | `CAST()` | `CAST()` | `CAST()` |
| Case-insensitive LIKE | `LOWER() LIKE LOWER()` | `ILIKE` | `LOWER() LIKE LOWER()` | `LOWER() LIKE LOWER()` | `LOWER() LIKE LOWER()` |
| String length | `LENGTH()` | `LENGTH()` | `LEN()` | `LENGTH()` | `LENGTH()` |
| Window ORDER BY | Required for LAG/LEAD etc. | Required | Required | Required | Required |
| JOIN USING | Not supported | Supported | Not supported | Supported | Supported |
| Subquery ORDER BY | Allowed | Allowed | Only with TOP/OFFSET/FOR XML | Allowed | Allowed |

---

## Domain Inference

`_infer_domain_from_report()` in `orchestrator_api.py` runs automatically at index time when no domain is manually specified.

**Algorithm:**
1. Tokenise all table and column names (split on `_`, lowercase)
2. Score each industry signal set and each function signal set independently
3. Return compound `"Industry/Function"` label when both tiers fire; single label when only one fires

**Industry signals:** CPG (brand, sku, category, retailer, upc, rsv, gsv), Life Sciences, Healthcare, Telecom (arpu, mou, subscriber), Banking/FS (nii, nim, casa), Insurance, Retail, E-commerce, Manufacturing, SaaS

**Function signals:** RGM (promo, trade_spend, price_index), FP&A (gl_account, cost_centre, scenario, version, reforecast), Supply Chain (purchase_order, goods_receipt, shipment), Sales, Marketing, HR/People, Operations, CX

---

## GraphRAG — Hybrid Semantic Retrieval

`retrieve_node` uses a two-stage retrieval strategy:

1. **Embed query** — encode the natural language query using sentence-transformers (all-MiniLM-L6-v2) or OpenAI text-embedding-3-small
2. **Cosine similarity** — rank all KG node embeddings against the query embedding
3. **Top-K selection** — return the most semantically relevant nodes as additional schema context
4. **Fallback** — if embeddings are unavailable, BM25 keyword overlap over node titles

This ensures that even for large schemas (100+ tables), the planner LLM receives only the most relevant tables rather than the full schema.

---

## KG Isolation — Multi-KG Support

Every KG node, edge, and embedding is tagged with `kg_id` = `source_id`. This enables:

- **Multi-source queries** — multiple sources in one session, each with its own KG
- **Cross-source bridges** — detected by inclusion dependency overlap across `kg_id` boundaries
- **Safe re-indexing** — only the affected `kg_id` is cleared and rebuilt; other sources are untouched
- **Persona access control** — `persona_access` list on each source restricts which roles can query it

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for LLM calls |
| `METADATA_API_URL` | `http://localhost:8000` | Metadata extraction service |
| `ONTOLOGY_API_URL` | `http://localhost:8001` | Ontology generation service |
| `KG_API_URL` | `http://localhost:8002` | Knowledge graph service |
| `DIALOG_API_URL` | `http://localhost:8003` | Dialog with data service |
| `SHACL_API_URL` | `http://localhost:8007` | SHACL validation service |
| `DATA_DIR` | `./reports` | Report and ontology file storage |
| `METADATA_DB` | `/data/metadata.db` | Metadata catalog SQLite path |
| `KG_STORE_DB` | `/data/kg_store.db` | KG snapshot SQLite path |
| `KG_POSTGRES_DSN` | — | PostgreSQL DSN for production persistence |
| `NEO4J_URI` | — | Neo4j bolt URI (production only) |
| `NEO4J_USERNAME` | `neo4j` | Neo4j credentials |
| `NEO4J_PASSWORD` | — | Neo4j credentials |
| `LOG_LEVEL` | `info` | Uvicorn log level |

### Port Overrides (docker-compose)

| Variable | Default |
|---|---|
| `AGENT_PORT` | `8000` |
| `ONTOLOGY_PORT` | `8001` |
| `KG_PORT` | `8002` |
| `DIALOG_PORT` | `8003` |
| `CONFORMITY_PORT` | `8004` |
| `CHAT_PORT` | `8005` |
| `TECH_PORT` | `8006` |
| `SHACL_PORT` | `8007` |
