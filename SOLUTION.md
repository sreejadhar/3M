# Metadata Agent — Complete Solution Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Metadata Extraction Agent](#metadata-extraction-agent)
4. [Ontology Agent](#ontology-agent)
5. [Knowledge Graph Agent](#knowledge-graph-agent)
6. [Dialog with Data Agent](#dialog-with-data-agent)
7. [API Services](#api-services)
8. [Streamlit UI](#streamlit-ui)
9. [Docker Setup](#docker-setup)
10. [Deployment](#deployment)
11. [Data Flow](#data-flow)
12. [Configuration Reference](#configuration-reference)
13. [GraphRAG — Hybrid Semantic Retrieval](#graphrag--hybrid-semantic-retrieval)
14. [KG Isolation — Multi-KG Support via kg_id](#kg-isolation--multi-kg-support-via-kg_id)
15. [Production Embedding — embed_node](#production-embedding--embed_node)
16. [DataChat UI — orchestrator_api.py](#datachat-ui--orchestrator_apipy)
17. [DataChat UI — Frontend (chat_ui/)](#datachat-ui--frontend-chat_ui)

---

## Overview

The Metadata Agent is a multi-container system that connects to a database, automatically extracts its full schema and statistical metadata, discovers implicit data relationships, and generates a machine-readable OWL/RDF ontology from the results.

The system is built on four independently deployable AI agents:

| Agent | Purpose | Port |
|---|---|---|
| **Metadata Extraction Agent** | Connects to a database and extracts schema, statistics, functional dependencies, inclusion dependencies, and cardinality relationships | 8000 |
| **Ontology Agent** | Reads a metadata report and generates a formal OWL/RDF ontology in Turtle, RDF/XML, or N3 format | 8001 |
| **Knowledge Graph Agent** | Converts an OWL/RDF ontology to Cypher (Neo4j) or Gremlin (TinkerPop), optionally executes on a live graph database, and returns interactive graph data for the UI | 8002 |
| **Dialog with Data Agent** | Accepts a natural language query, traverses the knowledge graph for schema context, plans and executes multiple SQL queries, stitches results, and derives LLM insights | 8003 |
| **Streamlit UI** | Web interface for extractions, history, search, ontology generation/editing, knowledge graph creation/visualisation, and natural-language dialog | 8501 |

All four agents are **completely decoupled**: each has zero imports from the others. They communicate only through JSON contracts passed via the UI.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Docker Network: metadata-net                    │
│                                                                          │
│  ┌──────────────┐  HTTP   ┌──────────────────┐                          │
│  │  Streamlit   │ ──────► │  agent-api       │  port 8000               │
│  │  UI          │         │  (FastAPI)        │                          │
│  │  port 8501   │         │  metadata_agent/  │                          │
│  │              │  HTTP   │  api.py           │                          │
│  │              │ ──────► └──────────────────┘                          │
│  │              │                  │                                     │
│  │              │         shared Docker volume: reports_data             │
│  │              │                  │                                     │
│  │              │  HTTP   ┌──────────────────┐                          │
│  │              │ ──────► │  ontology-api    │  port 8001               │
│  │              │         │  (FastAPI)        │                          │
│  │              │         │  ontology_api.py  │                          │
│  │              │  HTTP   └──────────────────┘                          │
│  │              │ ──────► ┌──────────────────┐                          │
│  │              │         │  kg-api          │  port 8002               │
│  │              │         │  (FastAPI)        │                          │
│  │              │         │  kg_api.py        │                          │
│  │              │  HTTP   └──────────────────┘                          │
│  │              │ ──────► ┌──────────────────┐                          │
│  │              │         │  dialog-api      │  port 8003               │
│  └──────────────┘         │  (FastAPI)        │                          │
│                           │  dialog_api.py    │                          │
│                           └──────────────────┘                          │
└──────────────────────────────────────────────────────────────────────────┘
         │                          │                    │
         ▼                          ▼                    ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────────────────┐
│  Source Database     │  │  Source Database     │  │  Target Graph Database   │
│  PostgreSQL / Oracle │  │  (Dialog SQL target) │  │  Neo4j (bolt://)  or     │
│  SQL Server /        │  │  any supported DB    │  │  Gremlin Server (ws://)  │
│  Teradata / Redshift │  └─────────────────────┘  └──────────────────────────┘
│  BigQuery / Delta    │
└─────────────────────┘
```

**Key design decisions:**

- The UI is the orchestrator for all cross-service workflows. It fetches results from one service and sends them to the next. The backend services never talk to each other.
- For knowledge graph creation: the UI fetches the ontology text from `ontology-api` and sends it to `kg-api`. The `kg-api` connects directly to the user's Neo4j or Gremlin server.
- For dialog with data: the UI fetches graph data from `kg-api` and sends it along with the NQL and DB credentials to `dialog-api`. The `dialog-api` connects directly to the user's target database for SQL execution.
- Reports and ontology files are written to a named Docker volume (`reports_data`) mounted by both `agent-api` and `ontology-api`. The `kg-api` and `dialog-api` do not need the volume — they operate entirely in-memory.
- All inter-service communication uses Docker's internal DNS (service names, not `localhost`).
- All API services expose a `/health` endpoint and use FastAPI `BackgroundTasks` for non-blocking job execution.

---

## Metadata Extraction Agent

### Package Structure

```
metadata_agent/
├── __init__.py               # Exports: MetadataExtractionAgent, AgentConfig, DBConfig, DBType
├── config.py                 # AgentConfig, DBConfig, DBType dataclasses
├── state.py                  # AgentState TypedDict
├── agent.py                  # LangGraph graph definition + MetadataExtractionAgent class
├── nodes/
│   ├── connection_node.py    # Open the database connection
│   ├── discovery_node.py     # List all tables in the target schema
│   ├── extraction_node.py    # Extract schema + statistics per table
│   ├── analysis_node.py      # Detect FDs, INDs, cardinality relationships
│   └── report_node.py        # Aggregate results into the final report dict
├── tools/
│   ├── schema_extractor.py   # Column metadata: name, type, nullability, PK, FK
│   ├── metadata_collector.py # Row count, null counts, distinct counts, min/max/avg
│   ├── fd_detector.py        # Functional dependency detection via value hashing
│   ├── id_detector.py        # Inclusion dependency (FK candidate) detection
│   └── cardinality_analyzer.py  # 1:1 / 1:N / M:N relationship classification
└── connectors/
    ├── base.py               # Abstract BaseConnector interface
    ├── factory.py            # ConnectorFactory — maps DBType to connector class
    ├── postgres.py           # PostgreSQL connector (psycopg2)
    ├── oracle.py             # Oracle connector (cx_Oracle)
    ├── sqlserver.py          # SQL Server connector (pyodbc)
    ├── teradata.py           # Teradata connector (teradatasql)
    ├── redshift.py           # Amazon Redshift connector (psycopg2)
    ├── bigquery.py           # Google BigQuery connector (google-cloud-bigquery)
    └── delta_lake.py         # Delta Lake connector (PySpark)
```

### Configuration

```python
# config.py

class DBType(str, Enum):
    POSTGRES   = "postgres"
    ORACLE     = "oracle"
    TERADATA   = "teradata"
    DELTA_LAKE = "delta_lake"
    REDSHIFT   = "redshift"
    SQLSERVER  = "sqlserver"
    BIGQUERY   = "bigquery"

@dataclass
class DBConfig:
    db_type: DBType
    host: Optional[str]           # DB host
    port: Optional[int]           # DB port
    database: Optional[str]       # Database name
    schema: Optional[str]         # Target schema / dataset
    username: Optional[str]
    password: Optional[str]
    project: Optional[str]        # GCP project (BigQuery)
    credentials_path: Optional[str]  # Service account JSON (BigQuery)
    catalog: Optional[str]        # Spark catalog (Delta Lake)
    spark_master: Optional[str]   # e.g. "local[*]" (Delta Lake)
    connection_string: Optional[str]  # Overrides all individual fields if set
    extra: Dict[str, Any]

@dataclass
class AgentConfig:
    db_config: DBConfig
    target_tables: Optional[List[str]] = None  # None = all tables
    sample_size: int = 10_000           # Rows sampled for FD/IND analysis
    fd_threshold: float = 1.0           # 1.0 = exact FDs; <1.0 = approximate
    id_threshold: float = 0.95          # Coverage fraction required for IND
    max_fd_column_pairs: int = 200      # Cap combinatorial explosion
    max_id_column_pairs: int = 500
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0
    output_path: Optional[str] = None   # Where to write JSON report
```

### State

```python
# state.py

class AgentState(TypedDict, total=False):
    agent_config:   Any        # AgentConfig instance
    db_config:      Any        # DBConfig instance
    connector:      Any        # Active BaseConnector instance
    phase:          str        # "init" | "connected" | "discovered" | ... | "error"
    all_tables:     List       # All table names discovered
    tables_done:    Set        # Tables fully extracted
    table_metadata: Dict       # {table_name: {columns, row_count, ...}}
    func_deps:      List       # Detected functional dependencies
    incl_deps:      List       # Detected inclusion dependencies (FK candidates)
    cardinalities:  List       # Cardinality relationships
    messages:       List       # LangChain message history (for LLM steps)
    errors:         List       # Accumulated non-fatal errors
    final_report:   Dict       # Complete aggregated report
```

The state uses `TypedDict(total=False)` so LangGraph 0.2+ can introspect the schema and correctly merge partial state updates returned by each node.

### LangGraph Pipeline

```
START
  │
  ▼
connection_node      Opens the DB connection via ConnectorFactory.
  │                  Sets state["connector"] and state["phase"] = "connected".
  │                  On failure: phase = "error".
  │
  ├── error? ──► error_end ──► END
  │
  ▼
discovery_node       Calls connector.list_tables(schema).
  │                  Sets state["all_tables"].
  │                  On failure: phase = "error".
  │
  ├── error? ──► error_end ──► END
  │
  ▼
extraction_node      Iterates over all_tables.
  │                  For each table calls:
  │                    - SchemaExtractorTool    → columns, types, PK, FK
  │                    - MetadataCollectorTool  → row_count, null/unique counts, min/max/avg
  │                  Populates state["table_metadata"].
  │
  ▼
analysis_node        Runs three analysis passes:
  │                    - FunctionalDependencyTool  → determinant → dependent column sets
  │                    - InclusionDependencyTool   → FK candidates (col A ⊆ col B)
  │                    - CardinalityAnalyzerTool   → 1:1 / 1:N / M:N classification
  │                  Populates func_deps, incl_deps, cardinalities.
  │
  ▼
report_node          Aggregates all state into final_report dict.
  │                  Writes JSON to output_path if configured.
  │                  Sets state["final_report"].
  │
  ▼
END
```

### Tools

**SchemaExtractorTool** (`tools/schema_extractor.py`)
- Queries information_schema (or DB-specific catalog) for column definitions.
- Returns for each column: name, data_type, nullable, is_primary_key, is_foreign_key, referenced_table, referenced_column, character_maximum_length, numeric_precision.

**MetadataCollectorTool** (`tools/metadata_collector.py`)
- Executes `SELECT COUNT(*)`, `COUNT(DISTINCT col)`, `COUNT(CASE WHEN col IS NULL)`, `MIN(col)`, `MAX(col)`, `AVG(col)` per column.
- Samples up to `sample_size` rows to limit runtime on large tables.
- Returns row_count, null_count, unique_count, min_value, max_value, avg_value per column.

**FunctionalDependencyTool** (`tools/fd_detector.py`)
- Hashes column value combinations to detect where column set A always determines column B.
- Respects `fd_threshold` (1.0 = only exact FDs, lower values allow approximate FDs).
- Respects `max_fd_column_pairs` to cap the number of column pair comparisons.
- Returns: table, determinant (list of columns), dependent (list of columns), confidence.

**InclusionDependencyTool** (`tools/id_detector.py`)
- Checks whether the value set of column A is a subset of the value set of column B across different tables.
- Coverage above `id_threshold` → FK candidate (inclusion dependency).
- Returns: left_table, left_column, right_table, right_column, coverage.

**CardinalityAnalyzerTool** (`tools/cardinality_analyzer.py`)
- For pairs of tables joined by a FK/IND, counts distinct values on each side.
- Classifies the relationship: 1:1, 1:N, N:1, or M:N.
- Returns: left_table, right_table, type.

### Report Format

The final report is a JSON document with this structure:

```json
{
  "database_type": "postgres",
  "schema": "public",
  "extraction_timestamp": "2025-03-07T12:00:00",
  "summary": {
    "total_tables": 12,
    "total_columns": 87,
    "total_rows": 450000,
    "total_functional_dependencies": 23,
    "total_inclusion_dependencies": 8,
    "total_cardinality_relationships": 6
  },
  "tables": {
    "orders": {
      "row_count": 50000,
      "columns": [
        {
          "name": "order_id",
          "data_type": "integer",
          "nullable": false,
          "is_primary_key": true,
          "is_foreign_key": false,
          "unique_count": 50000,
          "null_count": 0,
          "min_value": 1,
          "max_value": 50000
        }
      ],
      "foreign_keys": [
        {
          "column": "customer_id",
          "referenced_table": "customers",
          "referenced_column": "customer_id"
        }
      ]
    }
  },
  "functional_dependencies": [
    {
      "table": "orders",
      "determinant": ["order_id"],
      "dependent": ["customer_id", "order_date"],
      "confidence": 1.0
    }
  ],
  "inclusion_dependencies": [
    {
      "left_table": "orders",
      "left_column": "customer_id",
      "right_table": "customers",
      "right_column": "customer_id",
      "coverage": 0.998
    }
  ],
  "fk_candidates": [...],
  "cardinality_relationships": [
    {
      "left_table": "orders",
      "right_table": "customers",
      "type": "1:N"
    }
  ]
}
```

### LLM Q&A

The `MetadataExtractionAgent.ask(question)` method (and the `/history/{run_id}/ask` API endpoint) answers natural-language questions about a saved report by:

1. Loading the report JSON from disk.
2. Injecting up to 40,000 characters of the report as context into a system message.
3. Calling `ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0)` with the question.
4. Returning the answer text.

This is implemented directly with the LangChain ChatAnthropic client — no full agent graph is reconstructed for Q&A, keeping the call lightweight and fast.

---

## Ontology Agent

### Package Structure

```
ontology_agent/
├── __init__.py               # Exports: OntologyAgent, OntologyConfig
├── config.py                 # OntologyConfig dataclass
├── state.py                  # OntologyState TypedDict
├── agent.py                  # LangGraph graph definition + OntologyAgent class
└── nodes/
    ├── load_node.py          # Validate the incoming metadata report
    ├── build_node.py         # Construct the rdflib OWL graph
    └── serialize_node.py     # Serialize graph to Turtle/RDF/N3, write to disk
```

This package has **zero imports** from the `metadata_agent` package. Its only dependency on the extraction agent is the JSON report format.

### Configuration

```python
# ontology_agent/config.py

@dataclass
class OntologyConfig:
    base_uri:           str  = "http://metadata-agent.io/ontology/"
    ontology_name:      str  = "DatabaseOntology"
    output_path:        Optional[str] = None   # Where to write the file
    serialize_format:   str  = "turtle"        # "turtle" | "xml" | "n3"
    include_statistics: bool = True            # Annotate properties with col stats
```

### State

```python
# ontology_agent/state.py

class OntologyState(TypedDict, total=False):
    config:          Any        # OntologyConfig instance
    report:          Dict       # Input metadata report
    ontology_graph:  Any        # rdflib Graph instance
    class_map:       Dict       # {table_name: URIRef} — OWL class URIs
    property_map:    Dict       # {(table, col): URIRef} — DatatypeProperty URIs
    ontology_turtle: str        # Serialized ontology text
    output_path:     str        # Final file path written
    triple_count:    int
    class_count:     int
    property_count:  int
    errors:          List
    phase:           str        # "init" | "loaded" | "built" | "done" | "error"
```

### LangGraph Pipeline

```
START
  │
  ▼
load_node          Validates that report is non-empty and contains at least one table.
  │                Logs table / FD / IND / cardinality counts.
  │                On failure: phase = "error".
  │
  ├── error? ──► error_end ──► END
  │
  ▼
build_node         Constructs the rdflib OWL graph (see OWL Mapping below).
  │                Sets ontology_graph, class_map, property_map, class_count,
  │                property_count, triple_count.
  │
  ▼
serialize_node     Serializes the graph to the configured format (turtle/xml/n3).
  │                Writes the file to output_path.
  │                Sets ontology_turtle, output_path, phase = "done".
  │
  ▼
END
```

### OWL Mapping Strategy

The `build_node` maps every element of the metadata report to OWL/RDF constructs:

| Metadata element | OWL/RDF representation |
|---|---|
| Table | `owl:Class` with `rdfs:label` = table name, `rdfs:comment` = row count |
| Column (general) | `owl:DatatypeProperty` with `rdfs:domain` = table class, `rdfs:range` = XSD type |
| Primary key column | Additionally: `owl:FunctionalProperty` + `owl:InverseFunctionalProperty` |
| NOT NULL column | `owl:Restriction` (minCardinality 1) as `rdfs:subClassOf` on the class |
| Column statistics | `rdfs:comment` on the DatatypeProperty (unique/null/min/max/avg) |
| Explicit FK | `owl:ObjectProperty` with domain = child table, range = parent table |
| FK candidate (IND) | `owl:ObjectProperty` with coverage fraction in `rdfs:comment` |
| 1:1 cardinality | `owl:FunctionalProperty` + `owl:InverseFunctionalProperty` on the ObjectProperty |
| 1:N cardinality | `owl:FunctionalProperty` on the ObjectProperty |
| Functional dependency | `rdfs:comment` on the owning class: `FD: [det] → [dep] conf=1.000` |
| Ontology header | `owl:Ontology` with `rdfs:label` and database source comment |

### XSD Type Mapping

The build node maps 30+ database column types to XSD equivalents:

| DB types | XSD |
|---|---|
| varchar, char, text, nvarchar, clob, json, uuid | `xsd:string` |
| int, integer, bigint, smallint, serial | `xsd:integer` |
| numeric, decimal, number | `xsd:decimal` |
| float, double, float8 | `xsd:double` |
| boolean | `xsd:boolean` |
| date | `xsd:date` |
| timestamp, datetime | `xsd:dateTime` |
| time | `xsd:time` |
| interval | `xsd:duration` |
| bytea, blob | `xsd:hexBinary` |
| (unrecognized) | `xsd:string` (safe fallback) |

URI fragments are sanitized by `_safe(name)` which replaces non-alphanumeric characters with underscores and prefixes a leading underscore if the name starts with a digit.

### Example Output (Turtle)

```turtle
@prefix :     <http://metadata-agent.io/ontology/DatabaseOntology/> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

<http://metadata-agent.io/ontology/DatabaseOntology>
    a owl:Ontology ;
    rdfs:label "DatabaseOntology" ;
    rdfs:comment "Generated from database: public (postgres)" .

:orders a owl:Class ;
    rdfs:label "orders" ;
    rdfs:comment "row_count=50000" ;
    rdfs:comment "FD: [order_id] → [customer_id, order_date]  conf=1.000" .

:orders_order_id a owl:DatatypeProperty, owl:FunctionalProperty, owl:InverseFunctionalProperty ;
    rdfs:label "order_id" ;
    rdfs:domain :orders ;
    rdfs:range xsd:integer ;
    rdfs:comment "unique_count=50000, null_count=0" .

:orders_fk_customers a owl:ObjectProperty ;
    rdfs:label "orders → customers (FK)" ;
    rdfs:domain :orders ;
    rdfs:range :customers .
```

---

## Knowledge Graph Agent

### Package Structure

```
knowledge_graph_agent/
├── __init__.py               # Exports: KGAgent, KGConfig
├── config.py                 # KGConfig dataclass
├── state.py                  # KGState TypedDict
├── agent.py                  # LangGraph graph definition + KGAgent class
└── nodes/
    ├── __init__.py
    ├── parse_node.py         # Parse OWL/Turtle with rdflib into a Graph object
    ├── translate_node.py     # Convert OWL graph to Cypher or Gremlin statements
    └── execute_node.py       # Execute statements on Neo4j or Gremlin server
```

This package has **zero imports** from `metadata_agent` or `ontology_agent`. Its only input is a raw ontology string (Turtle, RDF/XML, or N3).

### Configuration

```python
# knowledge_graph_agent/config.py

@dataclass
class KGConfig:
    graph_type: str = "neo4j"              # "neo4j" | "gremlin"

    # Neo4j — empty uri = skip execution (preview/translate-only mode)
    neo4j_uri:      str = ""               # e.g. "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    # Gremlin — empty url = skip execution
    gremlin_url:              str = ""     # e.g. "ws://localhost:8182/gremlin"
    gremlin_traversal_source: str = "g"

    clear_existing: bool = False           # Drop all vertices/edges before loading
    batch_size:     int  = 50             # Queries per batch
```

### State

```python
# knowledge_graph_agent/state.py

class KGState(TypedDict, total=False):
    config:            Any        # KGConfig instance
    ontology_text:     str        # Raw ontology string
    ontology_format:   str        # "turtle" | "xml" | "n3"
    ontology_graph:    Any        # rdflib.Graph after parsing
    queries:           List[str]  # Generated Cypher or Gremlin statements
    graph_data:        Dict       # {nodes: [...], edges: [...]} for UI visualisation
    execution_results: List[Dict] # Per-query execution summaries
    node_count:        int        # OWL classes → graph nodes
    edge_count:        int        # OWL object properties → graph edges
    executed_count:    int        # Successfully executed query count
    errors:            List[str]
    phase:             str        # "init"|"parsed"|"translated"|"executed"|"error"
```

### LangGraph Pipeline

```
START
  │
  ▼
parse_node         Parses ontology_text with rdflib into an rdflib.Graph.
  │                Validates non-empty. Sets ontology_graph.
  │
  ├── error? ──► error_end ──► END
  │
  ▼
translate_node     Extracts owl:Class, owl:DatatypeProperty, owl:ObjectProperty
  │                from the rdflib graph. Generates queries and graph_data.
  │
  ├── error? ──► error_end ──► END
  │
  ▼
execute_node       If connection URI is non-empty: connects to Neo4j or Gremlin
  │                and executes all queries. If URI is empty, skips gracefully
  │                (preview / translate-only mode — graph_data still returned).
  │
  ▼
END
```

### OWL → Graph Mapping

| OWL element | Graph representation |
|---|---|
| `owl:Class` | Vertex/node with label = class name, `uri` property = full URI |
| `owl:DatatypeProperty` | Node property attribute (xsd type stored as metadata) |
| `owl:ObjectProperty` | Directed edge between two class nodes |
| `owl:FunctionalProperty` on ObjectProperty | Edge annotated `cardinality=1:N` |
| `owl:FunctionalProperty` + `owl:InverseFunctionalProperty` | Edge annotated `cardinality=1:1` |
| `rdfs:comment` | Node/edge description / tooltip in `graph_data` |
| `rdfs:label` | Human-readable name used as vertex/edge label |

### Cypher Generation (Neo4j)

Each `owl:Class` becomes a `MERGE` statement creating a node with two labels: `KGNode` (for constraint queries) and the class name (for type-specific queries):

```cypher
CREATE CONSTRAINT kg_node_uri IF NOT EXISTS FOR (n:KGNode) REQUIRE n.uri IS UNIQUE

MERGE (n:KGNode:Orders {uri: 'http://...#orders', name: 'orders', type: 'owl:Class'})
ON CREATE SET n.order_id_xsd_type = 'integer', n.order_date_xsd_type = 'dateTime'

MATCH (a:KGNode {uri: '...#orders'}), (b:KGNode {uri: '...#customers'})
MERGE (a)-[r:FK_CUSTOMERS {name: 'orders_fk_customers', type: 'owl:ObjectProperty', cardinality: '1:N'}]->(b)
```

### Gremlin Generation (TinkerPop)

Each class becomes an upsert `coalesce` statement; each object property becomes an `addE` with upsert:

```groovy
g.V().has('uri', '...#orders').fold().coalesce(unfold(),
  addV('orders').property('uri','...#orders').property('name','orders')
  .property('order_id_xsd_type','integer')).next()

g.V().has('uri', '...#orders').as('a')
  .V().has('uri', '...#customers')
  .coalesce(
    inE('FK_CUSTOMERS').where(outV().as('a')),
    addE('FK_CUSTOMERS').from('a').property('type','owl:ObjectProperty').property('cardinality','1:N')
  ).next()
```

### Preview Mode

If `neo4j_uri` (or `gremlin_url`) is left empty in the request, the `execute_node` skips DB connection entirely. The API still returns `graph_data` and `queries` — useful for:
- Previewing the generated schema before committing to a live DB
- Downloading the query file for manual execution
- Visualising the ontology structure without a running graph server

### graph_data Format

```json
{
  "nodes": [
    {
      "id":    "http://metadata-agent.io/ontology/DatabaseOntology/orders",
      "label": "orders",
      "title": "Class: orders\nrow_count=50000\n\nProperties:\n  order_id: integer\n  order_date: dateTime",
      "color": "#63b3ed",
      "size":  28
    }
  ],
  "edges": [
    {
      "from":  "http://...#orders",
      "to":    "http://...#customers",
      "label": "orders_fk_customers (FK)",
      "title": "orders_fk_customers (1:N)\ncoverage=0.998"
    }
  ]
}
```

Node size scales with the number of datatype properties (more columns → larger node). This gives an immediate visual indication of table complexity.

---

## Dialog with Data Agent

### Package Structure

```
dialog_agent/
├── __init__.py               # Exports: DialogAgent, DialogConfig
├── config.py                 # DialogConfig dataclass
├── state.py                  # DialogState TypedDict, SQLQuery, QueryResult
├── agent.py                  # LangGraph graph definition + DialogAgent class
└── nodes/
    ├── __init__.py
    ├── understand_node.py    # Build schema context from KG nodes/edges
    ├── plan_node.py          # LLM decomposes NQL into SQL queries
    ├── execute_node.py       # Execute SQL queries against target database
    └── synthesize_node.py    # LLM stitches results into narrative insights
```

This package has **zero imports** from `metadata_agent`, `ontology_agent`, or `knowledge_graph_agent`. Its inputs are:
1. A natural language query string
2. KG graph data (nodes + edges from the KG Agent) — optional
3. DB connection credentials for SQL execution

### Configuration

```python
# dialog_agent/config.py

@dataclass
class DialogConfig:
    # Target database for SQL execution
    db_type: str = "postgres"       # "postgres"|"oracle"|"sqlserver"|"bigquery"|etc.
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_schema: str = "public"
    db_user: str = ""
    db_password: str = ""
    db_connection_string: str = "" # overrides individual fields when set
    db_extra: Dict[str, Any] = {}

    # LLM settings
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.0

    # Query behaviour
    max_sql_queries: int = 10       # max SQL queries the planner may emit
    row_limit: int = 500            # LIMIT applied to each query
    max_insight_rows: int = 2000    # rows passed to the synthesizer LLM
```

### State

```python
# dialog_agent/state.py

class SQLQuery(TypedDict):
    query_id: str
    description: str
    sql: str
    table_refs: List[str]

class QueryResult(TypedDict):
    query_id: str
    description: str
    sql: str
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    error: Optional[str]

class DialogState(TypedDict, total=False):
    config: Any                # DialogConfig
    natural_query: str         # the user's NQL string
    schema_context: str        # graph/ontology summary fed to LLM
    kg_nodes: List[Dict]       # knowledge graph nodes
    kg_edges: List[Dict]       # knowledge graph edges
    sql_queries: List[SQLQuery]     # planner output
    query_results: List[QueryResult]# executor output
    insights: str              # LLM-derived narrative
    errors: List[str]
    phase: str
```

### LangGraph Pipeline

```
START
  │
  ▼
understand_node    Converts KG nodes/edges into a structured schema description
  │                (table names, columns, relationships). No LLM call.
  │                Sets: schema_context
  │
  ▼
plan_node          Calls Claude with the NQL + schema_context.
  │                LLM returns a JSON array of {query_id, description, sql, table_refs}.
  │                Sets: sql_queries
  │
  ▼
execute_node       Connects to target DB using the appropriate driver.
  │                Executes each SQL query, captures columns + rows + errors.
  │                Sets: query_results
  │
  ▼
synthesize_node    Renders each result as a markdown table (max 20 rows each).
  │                Calls Claude to derive narrative insights from all results.
  │                Sets: insights
  │
  ▼
END
```

### Node Details

**understand_node** (`nodes/understand_node.py`)
- Parses KG node `title` tooltips to extract column/property lists
- Parses KG edges to extract inter-table relationships
- Produces a structured text schema context like:
  ```
  Table / Class: orders  (id=...#orders)
    Properties:
      - order_id: integer
      - order_date: dateTime
    [RELATION] fk_customers -> customers (1:N, coverage=0.998)
  ```
- Falls back to `"(No schema context available)"` if KG data is empty (schema-less mode)

**plan_node** (`nodes/plan_node.py`)
- System prompt enforces: return only JSON, use only known tables/columns, apply LIMIT
- Uses `re.sub` to strip markdown code fences from LLM output before JSON parsing
- Caps output at `config.max_sql_queries`
- Errors are caught and stored in `state["errors"]`; empty list returned (pipeline continues)

**execute_node** (`nodes/execute_node.py`)
- Supports: PostgreSQL/Redshift (psycopg2), Oracle (cx_Oracle), SQL Server (pyodbc), BigQuery (google-cloud-bigquery)
- Per-query error capture — a failed query does not abort the pipeline
- Respects `config.row_limit` for each result set

**synthesize_node** (`nodes/synthesize_node.py`)
- Renders each `QueryResult` as a markdown table (max 20 preview rows)
- Truncates the combined results text to `config.max_insight_rows * 4` chars to stay within LLM context
- Produces a structured markdown narrative with key findings, patterns, and recommendations

### Schema-less Mode

If no KG job is selected in the UI, `kg_nodes` and `kg_edges` are empty. The `understand_node` sets `schema_context = "(No schema context available)"`. The `plan_node` LLM must then rely solely on the natural language query and DB type to generate SQL — it may ask for table names or produce generic queries. This mode is useful for ad-hoc exploration where the user knows the schema.

---

## API Services

### Metadata Extraction API (`api.py`, port 8000)

The FastAPI service wraps the `MetadataExtractionAgent` and provides all backend functionality for the UI.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns `{"status": "ok"}` |
| `POST` | `/extract` | Start async extraction → `{"job_id": "..."}` (HTTP 202) |
| `GET` | `/jobs/{job_id}` | Poll job status — returns completed_nodes, current_node, summary |
| `GET` | `/jobs/{job_id}/report` | Retrieve the full JSON report once done |
| `GET` | `/history` | List all saved extraction runs |
| `DELETE` | `/history/{run_id}` | Delete a run and its report file from disk |
| `GET` | `/history/{run_id}/report` | Load a saved report by its history ID |
| `POST` | `/history/{run_id}/ask` | LLM Q&A — `{"question": "..."}` → `{"answer": "..."}` |
| `GET` | `/search` | Full-text search across all saved reports |

**Job lifecycle:**

1. `POST /extract` creates a job record (`status: "queued"`) in `_jobs` dict and starts `_run_extraction` as a `BackgroundTask`.
2. The background task creates a `MetadataExtractionAgent`, calls `stream_run()`, and updates `_jobs[job_id]` with `completed_nodes` and `current_node` after each pipeline node.
3. On completion, the full report is written to disk as JSON and a history entry is appended to `.history.json`.
4. `GET /jobs/{job_id}` polls the in-memory job store and returns progress.
5. `GET /jobs/{job_id}/report` or `GET /history/{run_id}/report` reads the JSON file from disk.

**History persistence:**

History is stored as a JSON array in `DATA_DIR/.history.json`. Each entry records:
- `id` (= job_id), `timestamp`, `db_type`, `host`, `database`, `schema`
- `summary` (table/column/FD/IND/cardinality counts)
- `report_path` (absolute path to the JSON report file)

**Search:**

`GET /search?q=<term>&scope=tables|fds|all&db_type=postgres|all` scans all history entries, loads each report JSON from disk, and returns matches (up to 100 results) including tables, columns, and functional dependencies.

**LLM Q&A implementation:**

```python
def _ask_llm(report: Dict, question: str) -> str:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.0)
    system = SystemMessage(content=(
        "You are a data engineering expert. ...\n\n"
        "METADATA REPORT (JSON):\n"
        + json.dumps(report, indent=2, default=str)[:40_000]
    ))
    human = HumanMessage(content=question)
    response = llm.invoke([system, human])
    return response.content
```

The LLM is called directly (not via a full agent graph) to keep the Q&A endpoint fast and avoid unnecessary overhead.

### Knowledge Graph API (`kg_api.py`, port 8002)

Completely standalone FastAPI service. Accepts a raw ontology string, runs the KG pipeline, and returns graph data for visualisation plus the generated queries.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/generate` | Start async KG creation → `{"job_id": "..."}` (HTTP 202) |
| `GET` | `/jobs/{job_id}` | Poll status — returns node_count, edge_count, executed_count |
| `GET` | `/jobs/{job_id}/graph` | Fetch `{nodes, edges}` for UI visualisation |
| `GET` | `/jobs/{job_id}/queries` | Fetch `{graph_type, queries, count}` — the raw statements |
| `GET` | `/list` | List all KG jobs |

**Generate request body:**
```json
{
  "ontology_text":             "...",
  "ontology_format":           "turtle",
  "graph_type":                "neo4j",
  "neo4j_uri":                 "bolt://localhost:7687",
  "neo4j_username":            "neo4j",
  "neo4j_password":            "secret",
  "neo4j_database":            "neo4j",
  "gremlin_url":               "",
  "gremlin_traversal_source":  "g",
  "clear_existing":            false
}
```

Leave `neo4j_uri` (and `gremlin_url`) empty to run in **preview mode** — only parse + translate, no DB connection.

**Job record (in-memory):**
The `_jobs` dict stores per-job: status, completed_nodes, current_node, node_count, edge_count, executed_count, graph_data, queries, execution_results, errors. The large `graph_data` and `queries` blobs are excluded from the `/jobs/{id}` poll response and only returned on dedicated endpoints.

---

### Dialog with Data API (`dialog_api.py`, port 8003)

Completely standalone FastAPI service. Accepts a natural language query + KG graph data + DB credentials, runs the dialog pipeline, and returns SQL queries, tabular results, and LLM insights.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/query` | Start async dialog job → `{"job_id": "..."}` (HTTP 202) |
| `GET` | `/jobs/{job_id}` | Poll status — returns query_count, result_count |
| `GET` | `/jobs/{job_id}/results` | Fetch full results: insights, sql_queries, query_results |
| `GET` | `/list` | List all dialog jobs |

**Query request body:**
```json
{
  "natural_query": "Which customers placed more than 5 orders in the last 90 days?",
  "kg_nodes": [{ "id": "...", "label": "orders", "title": "...", "size": 25 }],
  "kg_edges": [{ "from": "...", "to": "...", "label": "fk_customers", "title": "..." }],
  "db_type": "postgres",
  "db_host": "localhost",
  "db_port": 5432,
  "db_name": "mydb",
  "db_schema": "public",
  "db_user": "admin",
  "db_password": "secret",
  "max_sql_queries": 5,
  "row_limit": 500
}
```

**Results response:**
```json
{
  "natural_query": "Which customers placed more than 5 orders...",
  "insights": "## Key Findings\n\n- **147 customers** placed more than 5 orders...",
  "sql_queries": [
    { "query_id": "q1", "description": "Count orders per customer", "sql": "SELECT ...", "table_refs": ["orders"] }
  ],
  "query_results": [
    { "query_id": "q1", "description": "...", "sql": "...", "columns": ["customer_id", "order_count"], "rows": [[42, 12], ...], "row_count": 147, "error": null }
  ],
  "errors": []
}
```

---

### Ontology API (`ontology_api.py`, port 8001)

Completely standalone FastAPI service. Accepts a metadata report JSON, runs the ontology pipeline, and returns an OWL/RDF file.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/generate` | Start async ontology generation → `{"job_id": "..."}` (HTTP 202) |
| `GET` | `/jobs/{job_id}` | Poll status — returns class_count, property_count, triple_count |
| `GET` | `/jobs/{job_id}/content` | Fetch the raw ontology text (for display/edit) |
| `PUT` | `/jobs/{job_id}/content` | Save edited ontology text back to disk |
| `GET` | `/jobs/{job_id}/download` | Download the file (FileResponse with correct MIME type) |
| `GET` | `/list` | List all generated ontology jobs |

**Generate request body:**
```json
{
  "report": { ... },
  "base_uri": "http://metadata-agent.io/ontology/",
  "ontology_name": "DatabaseOntology",
  "serialize_format": "turtle",
  "include_statistics": true
}
```

**Edit and save flow:**

`PUT /jobs/{job_id}/content` writes the edited content to disk and re-parses it with rdflib to keep the `triple_count` statistic accurate even after manual edits. If the edited content cannot be parsed (syntax error), a warning is logged but the save still succeeds.

**Progress tracking:**

The background task (`_run_ontology`) uses `agent.stream_run()` to yield `(node_name, state_update)` pairs as the pipeline progresses, updating `_jobs[job_id]["completed_nodes"]` and `"current_node"` after each node. After streaming finishes, `agent.run()` is called to obtain the final result dict (output path, counts).

---

## Streamlit UI

### File: `app.py`

The UI is organized into six navigation views, reachable from the sidebar.

### API Clients

```python
# Points to the agent-api container
AGENT_API_URL    = os.environ.get("AGENT_API_URL",    "http://localhost:8000")
# Points to the ontology-api container
ONTOLOGY_API_URL = os.environ.get("ONTOLOGY_API_URL", "http://localhost:8001")

api      = APIClient(AGENT_API_URL)       # for extraction, history, search, Q&A
onto_api = OntologyAPIClient(ONTOLOGY_API_URL)  # for ontology generation and management
```

`APIClient._post()` extracts FastAPI's `detail` field from error responses so that meaningful error messages reach the user instead of generic HTTP error strings.

### Session State Keys

| Key | Purpose |
|---|---|
| `job_id` | Active extraction job UUID |
| `job_status` | Last polled job status dict |
| `run_id` | Run ID of the report currently being viewed |
| `onto_job_id` | Active ontology generation job UUID |
| `onto_result` | Last completed ontology job result dict |
| `onto_content` | Cached ontology text for the editor (avoids redundant API calls) |
| `onto_last_job_id` | Job ID whose content is cached in `onto_content` |

### API Clients

```python
api        = APIClient(AGENT_API_URL)            # extraction, history, search, Q&A
onto_api   = OntologyAPIClient(ONTOLOGY_API_URL) # ontology generation and management
kg_api     = KGAPIClient(KG_API_URL)             # knowledge graph creation and visualisation
dialog_api = DialogAPIClient(DIALOG_API_URL)     # dialog with data (NQL → SQL → insights)
```

All four clients extract FastAPI `detail` fields from error responses for meaningful error messages.

**`KGAPIClient` methods:**

| Method | HTTP call | Returns |
|---|---|---|
| `health()` | `GET /health` | `bool` |
| `generate(payload)` | `POST /generate` | `job_id: str` |
| `get_job(job_id)` | `GET /jobs/{id}` | status dict |
| `get_graph(job_id)` | `GET /jobs/{id}/graph` | `{nodes, edges}` |
| `get_queries(job_id)` | `GET /jobs/{id}/queries` | `{graph_type, queries, count}` |
| `list_jobs()` | `GET /list` | `List[Dict]` |

**`DialogAPIClient` methods:**

| Method | HTTP call | Returns |
|---|---|---|
| `health()` | `GET /health` | `bool` |
| `query(payload)` | `POST /query` | `job_id: str` |
| `get_job(job_id)` | `GET /jobs/{id}` | status dict |
| `get_results(job_id)` | `GET /jobs/{id}/results` | full results dict |
| `list_jobs()` | `GET /list` | `List[Dict]` |

### Views

**1. Extract — `_extract_view()`**

- DB type selector with icon badges for each supported database.
- Connection form: host, port, database, schema, username, password (or BigQuery project/credentials, Delta Lake catalog/Spark master).
- Advanced options: target tables (comma-separated), sample size, FD/IND thresholds.
- On submit: calls `api.start_extraction(payload)`, stores `job_id` in session state.
- Progress tracker: polls `api.get_job(job_id)` every 1.5 seconds, renders 5-node pipeline progress bar (connection → discovery → extraction → analysis → report).
- On completion: fetches `api.get_job_report(job_id)` and renders the full result panel.
- Result panel: stat cards (tables, columns, rows, FDs, INDs, cardinalities), table-by-table detail with column schema and statistics, FD list, cardinality relationship table.

**2. History — `_history_view()`**

- Fetches `api.get_history()` and renders a card for each past run.
- Each card shows: database type badge, host/database/schema, timestamp, summary stats.
- "View Report" button loads the report and renders the same result panel as the Extract view.
- "Delete" button calls `api.delete_history(run_id)`.
- Expandable LLM Q&A panel for each run: text input → `api.ask(run_id, question)` → rendered answer.

**3. Search — `_search_view()`**

- Query input with scope selector (Tables/Columns, Functional Dependencies, All) and DB type filter.
- Calls `api.search(q, scope, db_type)` and renders grouped results.
- Each result shows: match type badge, matched name, context (database/schema/date), detail.

**5. Knowledge Graph — `_kg_view()`**

- Health check: if `kg_api.health()` returns `False`, shows error banner.
- Ontology selector: dropdown of completed ontology jobs from `onto_api.list_jobs()`.
- Graph type radio: Neo4j (Cypher) or Gremlin (TinkerPop).
- Connection form: bolt URI / WebSocket URL, credentials, database name.
- Preview toggle: when unchecked, connection fields are ignored and only parse+translate runs.
- Clear existing checkbox: drops all vertices/edges before loading.
- On "Create Knowledge Graph":
  1. Fetches ontology text via `onto_api.get_content(onto_job_id)`.
  2. POSTs to `kg_api.generate({ontology_text, graph_type, connection_params})`.
  3. Polls `kg_api.get_job(job_id)` with 3-node progress tracker (parse → translate → execute).
- On completion:
  - Stat cards: Graph Nodes, Graph Edges, Queries Run, DB Type.
  - Interactive graph via `_render_kg_graph(graph_data)` using pyvis + `st.components.v1.html()`.
  - Expandable queries panel: shows raw Cypher/Gremlin with syntax highlighting + download button.
  - Previous KG jobs listed with "View" buttons.

**Graph rendering (`_render_kg_graph`):**

Uses `pyvis.network.Network` to build an in-memory interactive network graph:
- Nodes: dark blue fill (`#1e2740`), blue border (`#63b3ed`), white label, scaled by property count.
- Edges: green (`#68d391`) directed arrows, labelled with property name.
- Physics: ForceAtlas2 layout for natural graph spreading.
- Interaction: zoom, pan, hover tooltips (class metadata on nodes, cardinality on edges).
- Rendered via `st.components.v1.html(net.generate_html(), height=630)`.

**4. Ontology Generator — `_ontology_view()`**

- Health check: if `onto_api.health()` returns `False`, shows an error banner and exits early.
- Run selector: dropdown populated from history. User picks which extraction run to build an ontology from.
- Configuration: ontology name, base URI, serialization format, include statistics toggle.
- On "Generate Ontology":
  1. Fetches report JSON from `api.get_history_report(run_id)`.
  2. POSTs to `onto_api.generate({report: ..., config: ...})`.
  3. Polls `onto_api.get_job(job_id)` with a 3-node progress tracker (load → build → serialize).
- On completion:
  - Stat cards: OWL Classes, OWL Properties, RDF Triples.
  - Syntax-highlighted code view: `st.code(content, language="turtle")`.
  - Editable text area (height 400): `st.text_area` pre-filled with the ontology source.
  - "Save Changes" button: calls `onto_api.save_content(job_id, edited_text)` via PUT.
  - "Download File" button: calls `onto_api.get_bytes(job_id)` and triggers browser download via `st.download_button`.
- Previously generated ontologies section: lists all jobs from `onto_api.list_jobs()` with View/Download actions.

**6. Dialog with Data — `_dialog_view()`**

- Health check: if `dialog_api.health()` returns `False`, shows error banner.
- Natural language input: multi-line text area for the user's question.
- KG context selector: dropdown of completed KG jobs from `kg_api.list_jobs()`. Selecting one provides schema context. "(no KG — schema-less mode)" skips KG context.
- Database connection form: type selector + host/port/database/schema/credentials (same as Extract view).
- Query controls: max SQL queries (1–20), row limit per query (10–5000).
- On "Ask the Data":
  1. Fetches KG graph data via `kg_api.get_graph(selected_kg_id)` if a KG was selected.
  2. POSTs to `dialog_api.query({natural_query, kg_nodes, kg_edges, db_*, max_sql_queries, row_limit})`.
  3. Polls `dialog_api.get_job(job_id)` with a 4-node progress tracker (understand → plan → execute → synthesize).
- On completion:
  - Stat cards: Queries Planned, Queries Executed, Succeeded, Failed.
  - Insights narrative: LLM-generated markdown rendered in a styled container.
  - Query results: each SQL query result in a collapsible expander with the SQL, row count, and a `st.dataframe` table.
  - Generated SQL queries: collapsed expander showing all planned SQL statements.
  - Previous dialogs: load past sessions with "Load" buttons.
- `_render_query_results(results)`: renders each `QueryResult` as a labeled expander. Uses `pandas.DataFrame` for the tabular view.

---

## Docker Setup

### Four Dockerfiles

**`Dockerfile.kg`** — Knowledge Graph API

```
Stage 1 (builder):
  - python:3.11-slim
  - pip install -r requirements.kg.txt into /install

Stage 2 (runtime):
  - python:3.11-slim
  - Non-root user: agent:agent
  - Copies /install from builder
  - Copies knowledge_graph_agent/ package only
  - Copies kg_api.py (FastAPI entry point)
  - No shared volume — KG agent is stateless (in-memory job store)
  - Exposes port 8002
  - Healthcheck: urllib.request to /health
  - CMD: uvicorn kg_api:app --host 0.0.0.0 --port 8002
```

**`Dockerfile.dialog`** — Dialog with Data API

```
Stage 1 (builder):
  - python:3.11-slim
  - pip install -r requirements.dialog.txt into /install

Stage 2 (runtime):
  - python:3.11-slim
  - Copies /install from builder
  - Copies dialog_agent/ package only
  - Copies dialog_api.py (FastAPI entry point)
  - No shared volume — Dialog agent is stateless (in-memory job store)
  - Exposes port 8003
  - CMD: uvicorn dialog_api:app --host 0.0.0.0 --port 8003
```

**`Dockerfile.agent`** — Metadata Extraction API

```
Stage 1 (builder):
  - python:3.11-slim
  - pip install -r requirements.agent.txt into /install

Stage 2 (runtime):
  - python:3.11-slim
  - Non-root user: agent:agent
  - Copies /install from builder
  - Copies metadata_agent/ package
  - Copies ontology_agent/ package  (included so report_node can reference OntologyAgent if needed)
  - Copies api.py (FastAPI entry point)
  - Exposes port 8000
  - Healthcheck: urllib.request to /health
  - CMD: uvicorn api:app --host 0.0.0.0 --port 8000
```

**`Dockerfile.ontology`** — Ontology API

```
Stage 1 (builder):
  - python:3.11-slim
  - pip install -r requirements.ontology.txt into /install

Stage 2 (runtime):
  - python:3.11-slim
  - Non-root user: agent:agent
  - Copies /install from builder
  - Copies ontology_agent/ package only (no metadata_agent)
  - Copies ontology_api.py (FastAPI entry point)
  - Exposes port 8001
  - Healthcheck: urllib.request to /health
  - CMD: uvicorn ontology_api:app --host 0.0.0.0 --port 8001
```

**`Dockerfile.ui`** — Streamlit UI

```
Stage 1 (builder):
  - python:3.11-slim
  - pip install -r requirements.ui.txt into /install

Stage 2 (runtime):
  - python:3.11-slim
  - Copies /install from builder
  - Copies app.py
  - Exposes port 8501
  - CMD: streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

### `docker-compose.yml`

```yaml
services:

  agent-api:
    build: { dockerfile: Dockerfile.agent }
    image: metadata-agent-api:latest
    ports: ["${AGENT_PORT:-8000}:8000"]
    volumes: [reports_data:/data/reports]
    environment:
      ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
      DATA_DIR: /data/reports
      LOG_LEVEL: "${LOG_LEVEL:-info}"
    healthcheck:
      test: [python -c "urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 15s  timeout: 5s  retries: 5  start_period: 15s
    networks: [metadata-net]

  ontology-api:
    build: { dockerfile: Dockerfile.ontology }
    image: metadata-ontology-api:latest
    ports: ["${ONTOLOGY_PORT:-8001}:8001"]
    volumes: [reports_data:/data/reports]   # shared with agent-api
    environment:
      DATA_DIR: /data/reports
      LOG_LEVEL: "${LOG_LEVEL:-info}"
    healthcheck:
      test: [python -c "urllib.request.urlopen('http://localhost:8001/health')"]
      interval: 15s  timeout: 5s  retries: 5  start_period: 15s
    networks: [metadata-net]

  kg-api:
    build: { dockerfile: Dockerfile.kg }
    image: metadata-kg-api:latest
    ports: ["${KG_PORT:-8002}:8002"]
    environment:
      LOG_LEVEL: "${LOG_LEVEL:-info}"
    healthcheck:
      test: [python -c "urllib.request.urlopen('http://localhost:8002/health')"]
      interval: 15s  timeout: 5s  retries: 5  start_period: 15s
    networks: [metadata-net]
    # No volume — KGAgent is stateless; it connects directly to user's Neo4j/Gremlin server

  dialog-api:
    build: { dockerfile: Dockerfile.dialog }
    image: metadata-dialog-api:latest
    ports: ["${DIALOG_PORT:-8003}:8003"]
    environment:
      ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
      LOG_LEVEL: "${LOG_LEVEL:-info}"
    healthcheck:
      test: [python -c "urllib.request.urlopen('http://localhost:8003/health')"]
      interval: 15s  timeout: 5s  retries: 5  start_period: 15s
    networks: [metadata-net]
    # No volume — DialogAgent connects directly to user's target DB at query time

  ui:
    build: { dockerfile: Dockerfile.ui }
    image: metadata-agent-ui:latest
    ports: ["${UI_PORT:-8501}:8501"]
    environment:
      AGENT_API_URL:    "http://agent-api:8000"
      ONTOLOGY_API_URL: "http://ontology-api:8001"
      KG_API_URL:       "http://kg-api:8002"
      DIALOG_API_URL:   "http://dialog-api:8003"
    depends_on:
      agent-api:    { condition: service_healthy }
      ontology-api: { condition: service_started }
      kg-api:       { condition: service_started }
      dialog-api:   { condition: service_started }
    networks: [metadata-net]

volumes:
  reports_data:
    driver: local

networks:
  metadata-net:
    driver: bridge
```

**Key points:**
- `reports_data` is a named Docker volume mounted by `agent-api` (writes reports, reads them for Q&A/history) and `ontology-api` (writes ontology files, serves downloads). The `kg-api` does not mount any volume — it is stateless.
- The UI does not mount the volume — it accesses all files via HTTP API calls.
- `agent-api` uses `condition: service_healthy` (the UI fundamentally requires it). `ontology-api` and `kg-api` use `condition: service_started` so the UI is not blocked if those services are slow to initialize.
- Docker's internal DNS allows the UI to address all three APIs by service name: `http://agent-api:8000`, `http://ontology-api:8001`, `http://kg-api:8002`.
- The `kg-api` connects directly to the user's Neo4j or Gremlin server (outside Docker) using the URI/URL provided at request time — it is not configured at deploy time.

### Requirements Files

| File | Key packages |
|---|---|
| `requirements.agent.txt` | fastapi, uvicorn, pydantic, langchain-anthropic, langchain-core, langgraph, rdflib, psycopg2-binary, cx_Oracle, pyodbc, teradatasql, google-cloud-bigquery, pyspark |
| `requirements.ontology.txt` | fastapi, uvicorn, pydantic, rdflib, langgraph, langchain-core |
| `requirements.kg.txt` | fastapi, uvicorn, pydantic, rdflib, langgraph, langchain-core, neo4j, gremlinpython |
| `requirements.ui.txt` | streamlit, requests, pyvis |

---

## Deployment

### `deploy.sh`

The deployment script manages the full container lifecycle.

**Commands:**

| Command | Effect |
|---|---|
| `./deploy.sh` | Build all 3 images and start all containers (default) |
| `./deploy.sh --build-only` | Build images only, do not start |
| `./deploy.sh --start` | Start pre-built images without rebuilding |
| `./deploy.sh --restart` | Restart running containers |
| `./deploy.sh --stop` | Stop containers (volumes preserved) |
| `./deploy.sh --down` | Stop containers and delete volumes (all reports deleted) |
| `./deploy.sh --logs` | Tail logs from all containers |
| `./deploy.sh --status` | Show container status via `docker compose ps` |
| `./deploy.sh --help` | Show help |

**Health checking:**

After starting containers, the script polls each service's health endpoint until it responds (or times out after 90 seconds):
- `http://localhost:8000/health` — agent-api
- `http://localhost:8001/health` — ontology-api
- `http://localhost:8501/_stcore/health` — Streamlit UI

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required for LLM Q&A features |
| `AGENT_PORT` | 8000 | Host port for the metadata extraction API |
| `ONTOLOGY_PORT` | 8001 | Host port for the ontology API |
| `KG_PORT` | 8002 | Host port for the knowledge graph API |
| `UI_PORT` | 8501 | Host port for the Streamlit UI |
| `LOG_LEVEL` | info | Log verbosity: debug / info / warning |

Set these in a `.env` file at the project root or export them before running the script.

### Quick Start

```bash
# 1. Set your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 2. Build and start all containers
./deploy.sh

# 3. Open the UI
open http://localhost:8501

# 4. View API docs
open http://localhost:8000/docs    # Metadata Extraction API
open http://localhost:8001/docs    # Ontology API
open http://localhost:8002/docs    # Knowledge Graph API
```

**Note:** The Knowledge Graph API connects to your Neo4j or Gremlin server at request time. You provide the connection URI in the UI when creating a knowledge graph — no configuration is needed at deploy time. Ensure Neo4j or your Gremlin server is reachable from the Docker network (use `host.docker.internal` instead of `localhost` when the DB is on the host machine).

### Running Locally Without Docker

```bash
# Terminal 1 — Metadata Extraction API
export ANTHROPIC_API_KEY=sk-ant-...
export DATA_DIR=./reports
uvicorn api:app --port 8000 --reload

# Terminal 2 — Ontology API
export DATA_DIR=./reports
uvicorn ontology_api:app --port 8001 --reload

# Terminal 3 — Knowledge Graph API
uvicorn kg_api:app --port 8002 --reload

# Terminal 4 — Streamlit UI
export AGENT_API_URL=http://localhost:8000
export ONTOLOGY_API_URL=http://localhost:8001
export KG_API_URL=http://localhost:8002
streamlit run app.py
```

---

## Data Flow

### Extraction Flow

```
User fills form in UI
  │
  ▼
POST /extract (agent-api:8000)
  │  Creates job_id, starts background task
  │
  ▼
_run_extraction() [background thread]
  │
  ├── MetadataExtractionAgent.stream_run()
  │     ├── connection_node  → opens DB connector
  │     ├── discovery_node   → lists tables
  │     ├── extraction_node  → schema + stats per table
  │     ├── analysis_node    → FDs + INDs + cardinality
  │     └── report_node      → builds final_report dict
  │               (stream_run() captures final_report from "report" node state_update)
  │
  ├── Writes report JSON to reports_data volume
  └── Appends entry to .history.json

UI polls GET /jobs/{job_id} every 1.5s
  → renders 5-node progress tracker
  → on status=done: fetches GET /jobs/{job_id}/report
  → renders result panel
```

### Knowledge Graph Creation Flow

```
User selects a completed ontology in Knowledge Graph view
  │  Chooses Neo4j or Gremlin, enters connection details
  │
  ▼
UI: GET /jobs/{onto_job_id}/content  (ontology-api:8001)
  │  Returns the raw ontology text
  │
  ▼
UI: POST /generate  (kg-api:8002)
  │  Body: { ontology_text, graph_type, neo4j_uri/gremlin_url, ... }
  │  Returns: { job_id: "..." }
  │
  ▼
_run_kg() [background thread in kg-api]
  ├── parse_node      → rdflib.Graph
  ├── translate_node  → Cypher/Gremlin statements + graph_data
  └── execute_node    → runs queries on Neo4j/Gremlin (skipped in preview mode)

UI polls GET /jobs/{job_id} every 1.5s
  → on done: GET /jobs/{job_id}/graph → renders pyvis network + stats + queries
```

### Ontology Generation Flow

```
User selects a past run in Ontology Generator view
  │
  ▼
UI: GET /history/{run_id}/report  (agent-api:8000)
  │  Returns the metadata report JSON
  │
  ▼
UI: POST /generate  (ontology-api:8001)
  │  Sends: { report: <metadata JSON>, config: {...} }
  │  Returns: { job_id: "..." }
  │
  ▼
_run_ontology() [background thread in ontology-api]
  │
  ├── OntologyAgent.stream_run()
  │     ├── load_node      → validates report
  │     ├── build_node     → constructs rdflib OWL graph
  │     └── serialize_node → writes .ttl/.owl/.n3 to reports_data volume
  │
  └── OntologyAgent.run() → returns result dict (counts, output_path)

UI polls GET /jobs/{job_id} (ontology-api:8001) every 1.5s
  → renders 3-node progress tracker
  → on status=done:
      GET /jobs/{job_id}/content  → fetches ontology text
      renders stat cards + code view + editor
```

### Edit and Save Flow

```
User edits ontology in the text area
  │
  ▼
User clicks "Save Changes"
  │
  ▼
PUT /jobs/{job_id}/content  (ontology-api:8001)
  │  Body: { "content": "<edited turtle text>" }
  │
  ▼
ontology_api writes content to disk
  │
  └── re-parses with rdflib to update triple_count
```

---

## Configuration Reference

### Supported Database Connectors

| DBType | Driver | Notes |
|---|---|---|
| `postgres` | psycopg2-binary | Standard PostgreSQL |
| `oracle` | cx_Oracle | Requires Oracle Instant Client |
| `sqlserver` | pyodbc | Requires ODBC driver |
| `teradata` | teradatasql | Requires Teradata ODBC |
| `redshift` | psycopg2-binary | Uses PostgreSQL wire protocol |
| `bigquery` | google-cloud-bigquery | Uses service account JSON or ADC |
| `delta_lake` | pyspark | Uses Spark + Delta Lake |

### LangGraph Version Note

Both agents use LangGraph 0.2+. The `StateGraph` requires the state class to be annotated as a `TypedDict` (not a plain dict or dict subclass) so the framework can introspect field types and correctly merge partial state updates from node functions. Both `AgentState` and `OntologyState` are defined as `TypedDict(total=False)` to allow optional fields.

### Critical Implementation Detail: stream_run() and State Capture

In LangGraph, `graph.stream()` yields `{node_name: state_update}` dicts where `state_update` is the dict **returned by the node function**, not the full accumulated state. This means that after streaming completes, `self._report` is only populated if it is explicitly captured from the "report" node's update:

```python
# agent.py — MetadataExtractionAgent.stream_run()
for event in self._graph.stream(initial_state):
    for node_name, state_update in event.items():
        if node_name == "report" and isinstance(state_update, dict):
            self._report = state_update.get("final_report") or {}
        yield node_name, state_update
```

Without this capture, `api.py` would read `agent._report = None`, write `{}` to disk, and the UI would render nothing after a successful extraction.

---

## GraphRAG — Hybrid Semantic Retrieval

### What it is

GraphRAG is a **query-time retrieval layer** that sits at the start of the Dialog pipeline. Instead of sending the full schema (all KG nodes and edges) to the LLM for every question, it selects only the most relevant subset using a combination of **vector similarity** and **graph traversal**.

This matters at scale: a schema with 200 tables produces a schema context that exceeds LLM context windows. Even at 20–50 tables, including irrelevant tables wastes tokens and confuses the planner.

### Why not just vector similarity alone?

Vector similarity finds tables whose *names and descriptions* match the question semantically. But SQL queries frequently need JOIN partners — tables that are structurally connected by foreign keys but may not mention the same keywords.

Example: "What is the revenue per customer?" semantically matches `orders` and `customers`. But to compute revenue you also need `order_items` (joined to `orders` via FK). `order_items` may score low in vector similarity — its description says "line items per order", not "revenue" or "customer". Without graph expansion, the planner would miss the JOIN and produce wrong SQL.

### Two Execution Paths

| Path | When | How |
|---|---|---|
| **In-memory** | `graphrag_neo4j_uri` is empty (dev / small schema) | Embeds all node titles once per session; numpy cosine similarity; cache keyed by schema hash |
| **Production (Neo4j)** | `graphrag_neo4j_uri` is set | Uses HNSW vector index in Neo4j (`db.index.vector.queryNodes`); embeddings persisted by `embed_node` at KG build time |

Both paths produce the same output: a filtered `kg_nodes` / `kg_edges` subgraph that replaces the full schema in state.

### Step 1 — Vector Similarity (Seed Nodes)

Each KG node's `title` field (label + column names + comments + top values) is embedded into a fixed-dimension vector. At query time, the NLQ is embedded into the same space and cosine similarity is computed against all node vectors. The top-K nodes (default K=8) are selected as **seed nodes**.

```
NLQ: "total revenue by customer for last quarter"
         │
         ▼  embed (all-MiniLM-L6-v2 / TF-IDF / keyword)
     query vector [0.12, -0.34, 0.87, ...]
         │
         ▼  cosine similarity against corpus matrix [N × D]
     scores: orders=0.87  customers=0.82  products=0.61  inventory=0.23 ...
         │
         ▼  top-K=8 seeds
     seeds: [orders, customers, products, order_items, ...]
```

### Step 2 — BFS Graph Expansion (Expanded Nodes)

**Breadth-First Search (BFS)** starts from the seed nodes and explores the FK graph hop by hop, adding every reachable table within `hop_depth` hops (default 2).

```
Seeds from vector similarity:
  orders, customers

Hop 1 — follow FK edges from seeds:
  orders    ──FK──► order_items   (orders.order_id → order_items.order_id)
  customers ──FK──► addresses     (customers.id → addresses.customer_id)

Hop 2 — follow FK edges from hop-1 nodes:
  order_items ──FK──► products    (order_items.product_id → products.id)

Final subgraph: orders, customers, order_items, addresses, products
```

BFS is **bidirectional** — it follows FK edges in both directions (parent→child and child→parent).

The `hop_depth` config controls reach:
- `hop_depth=1` → direct FK neighbours of seeds only
- `hop_depth=2` → neighbours of neighbours (default; usually sufficient)
- Higher values → more tables included, less focused context for the LLM

### Step 3 — Filter to Subgraph

Nodes not in the BFS-expanded set are dropped. Only edges where both endpoints are in the subgraph are kept. The filtered `kg_nodes` and `kg_edges` replace the full schema in `DialogState` so `understand_node` builds a focused schema context.

### Embedding Backends

| Backend | Quality | Requires | Dimensions | Neo4j HNSW compatible |
|---|---|---|---|---|
| `sentence-transformers` | Best | `pip install sentence-transformers` | 384 | Yes |
| `openai` | Good | `OPENAI_API_KEY` | 1536 | Yes |
| `tfidf` | Good (no GPU) | `pip install scikit-learn` | vocabulary-size | No (variable) |
| `keyword` | Basic | None (pure Python) | vocabulary-size | No (variable) |

Auto-detection order: `sentence-transformers` → `tfidf` → `keyword`.

**Note:** TF-IDF and keyword backends produce variable-dimension vectors. They work for in-memory GraphRAG but cannot be used with the Neo4j HNSW index (which requires fixed dimensions). `embed_node` enforces this: it will error if `embed_backend` is `tfidf` or `keyword`.

### Configuration (DialogConfig)

```python
# dialog_agent/config.py

graphrag_enabled:           bool = True
graphrag_top_k:             int  = 8      # seed tables from vector search
graphrag_hop_depth:         int  = 2      # BFS hops from seeds via FK edges
graphrag_min_tables:        int  = 10     # skip retrieval if schema ≤ this many tables
graphrag_embedding_backend: str  = "auto" # "auto"|"sentence-transformers"|"openai"|"tfidf"|"keyword"

# Production Neo4j path — leave empty to use in-memory fallback
graphrag_neo4j_uri:         str  = ""     # e.g. "bolt://localhost:7687"
graphrag_neo4j_username:    str  = "neo4j"
graphrag_neo4j_password:    str  = ""
graphrag_neo4j_database:    str  = "neo4j"
graphrag_kg_id:             str  = ""     # must match KGConfig.kg_id used at build time
graphrag_neo4j_index:       str  = ""     # auto-derived as "kg-{kg_id}-embeddings" if empty
```

### Skip Conditions

The `retrieve_node` is a no-op (full schema passed through) when:
- `graphrag_enabled = False`
- `kg_nodes` is empty
- `len(kg_nodes) <= graphrag_min_tables` — small schemas fit in the prompt directly
- `natural_query` is empty

### Module-level Cache

The in-memory path maintains a module-level `_EMBED_CACHE` dict keyed by `f"{schema_hash}:{backend}"`. The schema hash is a stable MD5 over all `(node_id, node_title)` pairs. When the KG schema changes (e.g. after reindexing), the key changes and the cache is automatically rebuilt.

### Updated Dialog Pipeline

```
START
  │
  ▼
retrieve_node    GraphRAG: filter kg_nodes/kg_edges to the relevant subgraph.
  │              In-memory: cosine similarity + BFS. Production: Neo4j HNSW + Cypher BFS.
  │              No-op for small schemas (≤ graphrag_min_tables).
  │
  ▼
understand_node  Build focused schema context from the (now filtered) KG nodes/edges.
  │              Surfaces FK join columns as explicit JOIN ON clauses.
  │
  ▼
plan_node        LLM (claude-haiku-4-5) decomposes NLQ into SQL queries using schema context.
  │
  ▼
execute_node     Execute each SQL query against the target database.
  │
  ▼
synthesize_node  LLM (claude-sonnet-4-6) derives narrative insights from all results.
  │
  ▼
END
```

### Visualising GraphRAG in the KG Explorer

The DataChat UI KG explorer includes a **"Test GraphRAG"** panel. Type any natural language question and click Search:

- **Gold nodes** = seed tables returned by vector similarity. Percentage label = cosine similarity score.
- **Green nodes** = tables pulled in by BFS expansion via FK edges.
- **Grey/dimmed** = tables GraphRAG filtered out for this question.
- The legend shows the backend used and how many tables were retrieved vs. the total schema size.

This makes the retrieval layer visible and debuggable without running a full dialog query.

---

## KG Isolation — Multi-KG Support via kg_id

### Problem

Without isolation, two KGs loaded into the same Neo4j instance collide: `MERGE` on `uri` alone matches nodes from a different schema, causing graph corruption.

### Solution

Every node, edge, and vector index is stamped with a `kg_id`. The `MERGE` key is `{uri, kg_id}` — not `uri` alone. The HNSW vector index is named `kg-{kg_id}-embeddings` so each KG has its own isolated index.

### KGConfig additions

```python
# knowledge_graph_agent/config.py

kg_id: str = ""   # Unique identifier for this KG. Default "default".
                  # Use meaningful names: "sales_prod", "hr_staging", etc.

embed_enabled:    bool = False   # Set True to run embed_node after execute_node
embed_backend:    str  = "auto"  # "sentence-transformers" | "openai" (tfidf/keyword not supported)
embed_dimensions: int  = 384     # Must match the chosen model's output dimension
embed_index_name: str  = ""      # Auto-derived as "kg-{kg_id}-embeddings" if empty
```

### Cypher changes

```cypher
-- Constraint is now per (uri, kg_id) pair
CREATE CONSTRAINT IF NOT EXISTS FOR (n:KGNode) REQUIRE (n.uri, n.kg_id) IS UNIQUE

-- MERGE key includes kg_id
MERGE (n:KGNode {uri: 'http://...#orders', kg_id: 'sales_prod'})
  ON CREATE SET n.name = 'orders', n.type = 'owl:Class', ...

-- Relationships match both endpoints by kg_id
MATCH (a:KGNode {uri: '...#orders', kg_id: 'sales_prod'}),
      (b:KGNode {uri: '...#customers', kg_id: 'sales_prod'})
MERGE (a)-[r:FK_CUSTOMERS {kg_id: 'sales_prod', ...}]->(b)

-- Clearing a KG only deletes its own nodes
MATCH (n:KGNode {kg_id: 'sales_prod'}) DETACH DELETE n
```

### Gremlin changes

Vertex lookup uses `has('uri').has('kg_id')` so cross-KG nodes are never confused:

```groovy
g.V().has('kg_id', 'sales_prod').has('uri', '...#orders')
  .fold().coalesce(unfold(),
    addV('orders').property('uri','...').property('kg_id','sales_prod')
  ).next()
```

### How DialogConfig references a KG

Set `graphrag_kg_id` on `DialogConfig` to the same value used for `KGConfig.kg_id` at build time. The `retrieve_node` uses this to scope all Neo4j queries and to derive the HNSW index name:

```python
kg_id      = getattr(config, "graphrag_kg_id", "").strip() or "default"
index_name = getattr(config, "graphrag_neo4j_index", "").strip() or f"kg-{kg_id}-embeddings"
```

---

## Production Embedding — embed_node

### Purpose

`embed_node` runs after `execute_node` in the KG pipeline (when `embed_enabled=True` and `neo4j_uri` is set). It:

1. Builds a rich text representation for each node: `label + title body` (columns, types, statistics, sample values — stripped of the redundant "Class: X" prefix).
2. Embeds all texts in one batch via the configured backend (sentence-transformers or OpenAI).
3. Writes `embedding` vector + full `title` back onto the existing `KGNode` nodes in Neo4j, matched by `(uri, kg_id)`.
4. Serialises `join_columns` onto edges as JSON so `retrieve_node` can reconstruct exact `JOIN ON t1.col = t2.col` conditions.
5. Creates an HNSW cosine vector index `kg-{kg_id}-embeddings` (IF NOT EXISTS) with automatic fallback to the legacy `db.index.vector.createNodeIndex` API for Neo4j < 5.11.

### Location in the pipeline

```
START → parse_node → translate_node → execute_node → [embed_node] → END
                                                            ▲
                                              only when embed_enabled=True
                                              and neo4j_uri is non-empty
```

### Skip conditions

`embed_node` is a no-op when:
- `embed_enabled = False` (default — safe for dev/preview)
- `neo4j_uri` is empty
- `graph_data` has no nodes
- `embed_backend` resolves to `tfidf` or `keyword` (variable dimensions — incompatible with HNSW)

### Cypher written by embed_node

```cypher
-- Vector index (HNSW cosine, 384 dims for sentence-transformers)
CREATE VECTOR INDEX `kg-sales_prod-embeddings` IF NOT EXISTS
FOR (n:KGNode) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}

-- Per-node embedding write (matched by uri + kg_id for isolation)
MATCH (n:KGNode {uri: $uri, kg_id: $kg_id})
SET n.embedding = $embedding, n.title = $title

-- Per-edge join_columns serialisation
MATCH (a:KGNode {uri: $src, kg_id: $kg_id})-[r]->(b:KGNode {uri: $tgt, kg_id: $kg_id})
SET r.join_columns = $jc, r.title = $title
```

### Supported backends for production mode

| Backend | Model | Dimensions |
|---|---|---|
| `sentence-transformers` | all-MiniLM-L6-v2 | 384 |
| `openai` | text-embedding-3-small | 1536 |

---

## DataChat UI — orchestrator_api.py

### Overview

`orchestrator_api.py` (port 8005) is a FastAPI-based orchestrator that replaces the Streamlit UI for end-user access. It:

- Serves the `chat_ui/` single-page application (HTML + JS + CSS)
- Manages a **source registry** — pre-registered database connections with full indexing lifecycle
- Manages **chat sessions** — each session attaches to one source and runs the dialog pipeline
- Streams **Server-Sent Events (SSE)** for real-time progress to the browser

### Architecture

```
Browser (chat_ui/index.html)
    │
    │  HTTP / SSE (port 8005)
    ▼
orchestrator_api.py (port 8005)
    │
    ├── POST /sources              → register a data source
    ├── POST /sources/{id}/reindex → trigger indexing pipeline
    ├── GET  /sources/{id}/graph   → return KG nodes/edges
    ├── GET  /sources/{id}/ontology → return OWL/Turtle text
    ├── POST /sources/{id}/ontology → save edited ontology + rebuild KG
    ├── POST /sources/{id}/kg-preview → preview KG from ontology without saving
    ├── GET  /sources/{id}/index-events → SSE stream of indexing steps
    ├── POST /sources/{id}/graphrag-query → run in-memory GraphRAG retrieval
    │
    ├── POST /sessions             → create a chat session (optionally with source)
    ├── POST /sessions/{id}/chat   → send NLQ → runs dialog pipeline → returns insights
    └── GET  /sessions/{id}/events → SSE stream: pipeline progress + chat events
         │
         ├── METADATA_API (port 8000)  — extract metadata from source DB
         ├── ONTOLOGY_API (port 8001)  — build OWL ontology from metadata
         ├── KG_API       (port 8002)  — build KG from ontology
         └── DIALOG_API   (port 8003)  — NLQ → SQL → insights
```

### Source Registry

Sources are registered once (by an admin) and then available to all users as a named data source. They persist in memory (`_sources` dict) for the lifetime of the process.

**Source fields:**

| Field | Description |
|---|---|
| `id` | UUID |
| `name` | Display name |
| `description` | Optional description |
| `domain` | Business domain (Sales, Finance, HR, etc.) |
| `icon` | Domain emoji |
| `db_type` | postgres / oracle / sqlserver / mysql / sqlite / csv / excel |
| `connection` | DB credentials (never returned to the browser) |
| `persona_access` | Which personas can see this source |
| `status` | `idle` / `indexing` / `ready` / `error` |
| `indexed_at` | Unix timestamp of last successful index |
| `table_count` | Number of tables discovered |
| `table_names` | List of table names |
| `report` | Full metadata report (internal) |
| `ontology_content` | OWL/Turtle text (internal) |
| `kg_nodes` | KG graph nodes (internal) |
| `kg_edges` | KG graph edges (internal) |
| `error_message` | Last error (if status=error) |

### Indexing Pipeline (`_index_source`)

When `POST /sources/{id}/reindex` is called (or `auto_index=true` on creation), `_index_source` runs as an `asyncio.Task`:

```
_index_source(source_id)
  │
  ├─ push event: extract / running
  ├─ POST metadata-api/extract → poll until done → get report
  ├─ push event: extract / done  (table count + names)
  │
  ├─ push event: ontology / running
  ├─ POST ontology-api/generate → poll until done → get content
  ├─ push event: ontology / done  (line count)  [or warn on failure]
  │
  ├─ push event: kg / running
  ├─ POST kg-api/generate → poll until done → get graph
  ├─ push event: kg / done  (node + edge count)  [or warn on failure]
  │
  └─ push event: complete / done  [or complete / error]
```

Each step event has the shape:
```json
{
  "type":    "index_step",
  "step":    "extract",
  "status":  "done",
  "message": "Metadata extracted — 12 tables found",
  "detail":  "Tables: orders, customers, products, …",
  "ts":      1711123456.789
}
```

### SSE Endpoints

**`GET /sources/{id}/index-events`**

Replays the stored event log (last 100 events) immediately, then streams live events as they are pushed by `_index_source`. Clients that connect mid-indexing receive all past steps plus the live tail. The stream terminates automatically when a `step=complete` event is received.

**`GET /sessions/{id}/events`**

Streams pipeline progress (`type=progress`) and chat reply events (`type=chat_reply`) for a session. Used by the browser to update the pipeline progress bar and render streaming chat responses.

### Source Detail Endpoints

| Endpoint | Description |
|---|---|
| `GET /sources/{id}/graph` | Returns `{nodes, edges}` from the last successful index |
| `GET /sources/{id}/ontology` | Returns `{content: "...turtle..."}` |
| `POST /sources/{id}/ontology` | Body: `{content, rebuild_kg}`. Saves ontology. If `rebuild_kg=true`, triggers `_rebuild_kg` background task. |
| `POST /sources/{id}/kg-preview` | Body: `{ontology_text}`. Calls `kg-api/generate`, waits for result, returns `{nodes, edges}` without saving. Used by the ontology editor preview. |
| `POST /sources/{id}/graphrag-query` | Body: `{query, top_k, hop_depth}`. Runs in-memory GraphRAG against the source's KG nodes and returns seed nodes with cosine scores and BFS-expanded node IDs. |

### Session Flow

1. `POST /sessions` with `source_id` creates a session. If the source is ready, the dialog session is pre-configured with the source's DB credentials.
2. `POST /sessions/{id}/chat` accepts `{message}`, runs the dialog pipeline (understand → plan → execute → synthesize), and returns insights + SQL + results.
3. All pipeline events are pushed to the session SSE queue so the browser can render progress in real time.

---

## DataChat UI — Frontend (chat_ui/)

The `chat_ui/` directory contains a single-page application served by `orchestrator_api.py` at `/`.

### Files

| File | Purpose |
|---|---|
| `index.html` | Main HTML — sidebar, chat view, landing catalog, all modals |
| `app.js` | All client-side logic (~2000 lines) |
| `style.css` | All styles — CSS custom properties for theme, dark mode via `prefers-color-scheme` |

### Personas

Three personas control UI capabilities:

| Persona | Can ask questions | Can connect DBs | Admin panel | Index Log | Save ontology |
|---|---|---|---|---|---|
| Business User | Yes | No | No | No | No |
| Business Analyst | Yes | Yes | No | No | No |
| Data Admin | Yes | Yes | Yes | Yes | Yes |

### Source Catalog (Landing)

Shows registered sources as cards. Each ready source has a **KG** button that opens the KG & Ontology Explorer. Clicking the card opens a chat session for that source.

### Admin Panel

Accessible to the Data Admin persona. Shows all registered sources in a table with columns: Source, Type, Domain, Status, Tables, Actions.

Actions per source:
- **Index Log** — opens the Index Log modal
- **View KG** — opens the KG & Ontology Explorer
- **Reindex** — triggers `POST /sources/{id}/reindex`
- **Delete** — removes the source

### Connection Wizard

4-step wizard for registering new data sources:
1. **Type** — pick DB type (PostgreSQL, SQL Server, Oracle, MySQL, SQLite, CSV/Excel)
2. **Connect** — enter credentials or upload a file; "Test connection" button
3. **Name** — display name, description, domain, persona access
4. **Confirm** — summary + "Index automatically" checkbox

### Index Log Modal

Opens when "Index Log" is clicked for a source. Connects to `GET /sources/{id}/index-events` via EventSource. Shows each pipeline step as a row:

- **Spinner** (animated CSS) = `status: running`
- **✓** (green) = `status: done`
- **✗** (red) = `status: error`
- **⚠** (yellow) = `status: warn`

Clicking a row shows the `detail` field (e.g. table names discovered, line counts) in a detail panel below the steps.

### KG & Ontology Explorer Modal

Split-pane modal:

**Left pane — Knowledge Graph (vis.js)**

An interactive force-directed graph rendered with `vis-network`. Nodes are ellipses sized by column count. Edges are directed arrows labelled with the relationship name.

Interaction:
- Zoom and pan (scroll/drag)
- Hover tooltip shows full node/edge metadata
- Physics: ForceAtlas2 with BFS spring layout, auto-fits after stabilization

**GraphRAG test panel** (above the graph):

Input bar where any persona can type a natural language question to test GraphRAG retrieval:

```
[ What is the revenue per customer for last quarter? ] [ Search ] [✕]
● Seed (top-K)  ● Expanded (BFS)  ● Other nodes       backend: tfidf · 5/12 nodes retrieved
```

- **Gold nodes** = seed tables from vector similarity (label shows cosine % score)
- **Green nodes** = tables added by BFS expansion via FK edges
- **Grey/dimmed** = tables not relevant to this question
- **✕ button** resets all highlights and fits the full graph back into view

**Right pane — Ontology editor**

A monospace `<textarea>` pre-filled with the OWL/Turtle ontology text. Editable by all personas.

Behavior:
- Editing triggers a debounced (1.8s) call to `POST /sources/{id}/kg-preview`, which rebuilds the graph preview from the modified ontology
- A "modified" badge appears on the pane label when the editor content differs from the saved state
- **Admins only** see a "Save Ontology & Rebuild KG" button. Clicking it calls `POST /sources/{id}/ontology` with `rebuild_kg=true`, which persists the change and kicks off `_rebuild_kg` in the background.

**vis.js initialization detail:** The Network is created inside a `requestAnimationFrame` callback (not synchronously) to guarantee the flex container has computed pixel dimensions before vis.js measures it. After physics stabilization, `network.fit()` is called automatically.

### Chat View

Standard chat interface:
- Messages rendered with `marked.js` for markdown
- SQL results shown in collapsible result blocks with a `<table>` and optional Chart.js visualization (auto-detected: bar, horizontal bar, doughnut, stacked bar, line, KPI tiles)
- Pipeline progress bar shows current stage (understanding → planning → executing → synthesizing)

### API functions (app.js)

| Function | HTTP call | Used by |
|---|---|---|
| `apiListSources()` | `GET /sources?persona=X` | Sidebar, landing |
| `apiCreateSource(payload)` | `POST /sources` | Wizard |
| `apiDeleteSource(id)` | `DELETE /sources/{id}` | Admin panel |
| `apiReindexSource(id)` | `POST /sources/{id}/reindex` | Admin panel |
| `apiTestConnection(payload)` | `POST /sources/test-connection` | Wizard |
| `apiGetSourceGraph(id)` | `GET /sources/{id}/graph` | KG explorer |
| `apiGetSourceOntology(id)` | `GET /sources/{id}/ontology` | KG explorer |
| `apiSaveOntology(id, content, rebuild)` | `POST /sources/{id}/ontology` | KG explorer (admin) |
| `apiPreviewKG(id, text)` | `POST /sources/{id}/kg-preview` | KG explorer (debounced) |
| `apiGraphRAGQuery(id, query, k, hops)` | `POST /sources/{id}/graphrag-query` | KG GraphRAG test panel |
| `apiCreateSession(title, sourceId)` | `POST /sessions` | Source card click |
| `apiSendChat(sessionId, message)` | `POST /sessions/{id}/chat` | Send button |

### Running the DataChat UI

```bash
# Start all microservices (ports 8000–8003) then:
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn orchestrator_api:app --port 8005 --reload

# Open browser
open http://localhost:8005
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `METADATA_API_URL` | `http://localhost:8000` | Metadata extraction service |
| `ONTOLOGY_API_URL` | `http://localhost:8001` | Ontology generation service |
| `KG_API_URL` | `http://localhost:8002` | Knowledge graph service |
| `DIALOG_API_URL` | `http://localhost:8003` | Dialog with data service |
| `DATA_DIR` | `./reports` | Directory for persisted session data |
