# Algorithm Reference — Metadata Agent

This document describes every major algorithm implemented in the Metadata Agent system: how each one works, what inputs it uses, what it produces, and what design decisions were made.

---

## Table of Contents

1. [Metadata Extraction & Persistence](#1-metadata-extraction--persistence)
   - 1.1 [persist() — Upsert with CDC](#11-persist--upsert-with-cdc)
   - 1.2 [Redundancy Detection — Jaccard Similarity with Domain Gating](#12-redundancy-detection--jaccard-similarity-with-domain-gating)
2. [Taxonomy Algorithms](#2-taxonomy-algorithms)
   - 2.1 [infer_taxonomy() — Deterministic Pattern-Based Classification](#21-infer_taxonomy--deterministic-pattern-based-classification)
   - 2.2 [enrich_taxonomy() — LLM-Based Classification](#22-enrich_taxonomy--llm-based-classification)
   - 2.3 [_sync_taxonomy_from_kg_nodes() — KG Annotation Sync](#23-_sync_taxonomy_from_kg_nodes--kg-annotation-sync)
3. [Knowledge Graph & Ontology Algorithms](#3-knowledge-graph--ontology-algorithms)
   - 3.1 [_extract_ontology() — OWL Graph Parsing](#31-_extract_ontology--owl-graph-parsing)
   - 3.2 [_build_graph_data() — UI Visualisation Format](#32-_build_graph_data--ui-visualisation-format)
   - 3.3 [_generate_cypher() / _generate_gremlin() — Query Generation](#33-_generate_cypher--_generate_gremlin--query-generation)
4. [Domain Inference & Concept Annotation](#4-domain-inference--concept-annotation)
   - 4.1 [_infer_domain_from_report() — Signal-Word Voting](#41-_infer_domain_from_report--signal-word-voting)
   - 4.2 [_build_col_evidence() — Per-Column Evidence String](#42-_build_col_evidence--per-column-evidence-string)
   - 4.3 [_annotate_column_concepts() — LLM Concept Mapping](#43-_annotate_column_concepts--llm-concept-mapping)
   - 4.4 [End-to-End Pipeline & Examples](#44-end-to-end-pipeline--examples)
5. [Query Resolution Algorithms](#5-query-resolution-algorithms)
   - 5.1 [_load_samples_from_catalog() — Categorical Value Loading](#51-_load_samples_from_catalog--categorical-value-loading)
   - 5.2 [_detect_parent_child_pairs() — Taxonomy Hierarchy Detection](#52-_detect_parent_child_pairs--taxonomy-hierarchy-detection)
   - 5.3 [_build_taxonomy_hierarchy() — Cross-Tab Hierarchy Building](#53-_build_taxonomy_hierarchy--cross-tab-hierarchy-building)
   - 5.4 [_fuzzy_match_candidates() — Multi-Strategy Token Matching](#54-_fuzzy_match_candidates--multi-strategy-token-matching)
   - 5.5 [_apply_fuzzy_fallback() — Safety Net Resolution](#55-_apply_fuzzy_fallback--safety-net-resolution)

---

## 1. Metadata Extraction & Persistence

### 1.1 `persist()` — Upsert with CDC

**File:** `metadata_catalog.py`

**Purpose:** Take the raw extraction report produced by the metadata extraction service (table names, column names, statistics, sample values) and durably store it — handling new tables, updated statistics, deleted columns, and previously-deleted entities that reappear.

**Inputs:**
- `source_id` — UUID identifying the data source
- `source_name` — human-readable name
- `report` — dict of `{ tables: { table_name: { columns: [...], row_count, ... } } }`

**Output:** Count of entities persisted.

**Algorithm:**

1. **Source registration.** The source is upserted into `md_sources`. On first registration, `_infer_domain(source_name)` uses keyword matching to auto-assign a business domain (e.g., "sales", "hr").

2. **Pre-fetch existing state.** All current `md_entities` for this source are loaded into a dict keyed by `(schema_name, table_name)`. This single pre-fetch avoids N+1 queries during the upsert loop.

3. **Entity upsert loop.** For each table in the report:
   - If the entity exists and was previously soft-deleted (`deleted_from_source=True`), it is restored and a `"restored"` change event is logged.
   - If the entity exists and is active, stats (`row_count`, `size_bytes`, `primary_keys`) are overwritten. The `description` field is only overwritten if it is currently empty — preserving any human-curated descriptions.
   - If the entity does not exist, a new UUID is generated and a full INSERT is performed.

4. **Attribute upsert loop.** For each column in the table:
   - Same restore/update/insert pattern as entities.
   - On update: stats (`unique_count`, `null_count`, `min_value`, `max_value`, `avg_value`, `stddev_value`, `pattern_hints`, `top_values`) are always overwritten. Description is preserved if non-empty.
   - Data type changes are detected by comparing old vs new `data_type` strings and logged as `"type_changed"` events.
   - `statistical_type`, `semantic_role`, and `taxonomy_tree` are **not touched** in persist() — these are managed by the taxonomy algorithms and would otherwise be wiped on every re-index.

5. **Soft-delete pass.** After processing all tables/columns in the report, any entity or attribute that was active before the run but is absent from the new report is marked `deleted_from_source=True`. This is a soft delete — data is preserved, the UI shows the column as removed.

6. **CDC logging.** Every structural event (added, deleted, restored, type_changed) is appended to `md_changes` with a timestamp, source_id, and entity label.

7. **Redundancy check.** `_run_redundancy_check()` is called at the end of the same transaction — see §1.2.

**Key design decisions:**
- Never truncate and reload — always diff to preserve golden-record flags and descriptions.
- Soft deletes rather than hard deletes to maintain history and CDC audit trail.
- taxonomy fields are excluded from persist() so re-indexing doesn't wipe LLM classifications.

---

### 1.2 Redundancy Detection — Jaccard Similarity with Domain Gating

**File:** `metadata_catalog.py` → `_run_redundancy_check()`

**Purpose:** Detect tables that are likely duplicates of each other (same data stored in two places) by comparing column name sets using Jaccard similarity.

**Inputs:** All active entities for the current source and any same-domain sources.

**Output:** Upserted rows in `md_redundancies` for pairs with Jaccard ≥ 0.9; deleted rows for pairs that no longer meet the threshold.

**Algorithm:**

1. **Domain gating.** Retrieve the domain of the source being indexed. Candidate sources for comparison are:
   - Always: the source itself (within-source duplicate detection)
   - Conditionally: other sources sharing the same non-empty domain

   This prevents spurious cross-domain matches (e.g., a Sales `customers` table should not be flagged as a duplicate of an HR `employees` table just because both have a `name` column).

2. **Column set extraction.** For each candidate entity, load its active column names into a lowercase set. Case-normalisation ensures `CustomerID` and `customer_id` are treated as the same column.

3. **Jaccard computation.** For every pair `(A, B)` where A is from the current source:

   ```
   Jaccard(A, B) = |cols_A ∩ cols_B| / |cols_A ∪ cols_B|
   ```

4. **Threshold decision.** Jaccard ≥ 0.9 → upsert into `md_redundancies` with the shared column list and score. Jaccard < 0.9 → delete any existing redundancy record for this pair.

5. **Canonical pair ordering.** Pairs are stored as `(min(id_a, id_b), max(id_a, id_b))` to satisfy the UNIQUE constraint regardless of which side is A and which is B.

**Threshold rationale:** 0.9 means 90% of all columns are shared — this is intentionally strict to avoid false positives from tables that legitimately share a few common columns (like `id`, `created_at`).

---

## 2. Taxonomy Algorithms

Taxonomy classification runs in three layers, each acting as a safety net for the one above:

```
Layer 1: infer_taxonomy()      — deterministic, no dependencies, runs immediately after persist()
Layer 2: enrich_taxonomy()     — LLM-based, overwrites layer 1 with higher confidence
Layer 3: _sync_taxonomy_from_kg_nodes() — KG profile_node annotations, highest confidence
```

### 2.1 `infer_taxonomy()` — Deterministic Pattern-Based Classification

**File:** `metadata_catalog.py`

**Purpose:** Assign `statistical_type` and `semantic_role` to every column using only column name regex patterns and SQL data types — zero external dependencies.

**Inputs:** All active attributes for a source that have no existing `statistical_type`.

**Output:** Updated `statistical_type`, `semantic_role`, `taxonomy_tree` in `md_attributes`. Returns count updated.

**Key invariant:** Does NOT overwrite columns that already have a `statistical_type` set (LLM or KG annotations take precedence).

**Algorithm — Decision Sequence:**

The algorithm applies rules in a strict priority order. The priority order was designed to prevent numeric dtype from incorrectly winning over semantic column name evidence.

**Step 1 — Boolean and identifier name rules (highest priority)**

Check the first two name rules before looking at data type. This prevents a column like `is_active INTEGER` from being classified as `continuous` just because its dtype is numeric.

| Pattern | statistical_type | semantic_role |
|---|---|---|
| `\b(is\|has\|flag\|active\|enabled\|status_flag)\b` | boolean | boolean_flag |
| `(_\|^)(id\|key\|code\|pk\|sk\|seq\|num\|nr\|no)($\|_)` | identifier | identifier |

**Step 2 — Data type rules**

Only reached if step 1 found no match:

| Data type pattern | statistical_type | semantic_role |
|---|---|---|
| date / datetime / timestamp / time | date | time_dimension_key |
| int / decimal / float / numeric / money | continuous | measure |

**Step 3 — Remaining name rules**

Only reached if step 2 found no match (column is a text type with no boolean/id name):

| Pattern | statistical_type | semantic_role |
|---|---|---|
| `fiscal_year\|fy\|calendar_year\|cy\|year_period` | ordinal | time_period |
| `year\|month\|quarter\|week\|period\|qtr` | ordinal | time_period |
| `date\|datetime\|timestamp\|time` | date | time_dimension_key |
| `country\|nation\|region\|market\|geography\|territory` | categorical | geography |
| `sub_category\|subcategory\|sub_segment` | categorical | product_sub_category |
| `category\|segment\|vertical\|product_type` | categorical | product_category |
| `brand\|manufacturer\|vendor\|supplier` | categorical | product_dimension_key |
| `customer\|account\|client\|retailer\|buyer` | categorical | customer_dimension_key |
| `channel\|distribution\|outlet\|store` | categorical | org_unit |
| `division\|business_unit\|department\|org` | categorical | org_unit |

**Step 4 — Cardinality fallback**

If no rule matched and the column is a text type with `unique_count ≤ 50` (or top_values contains ≤ 50 entries), it is classified as `categorical / other`. This catches columns that don't follow standard naming conventions but are clearly low-cardinality dimension columns.

**taxonomy_tree population:**
For categorical and ordinal columns, the `top_values` JSON array (already populated by the extractor) is copied into `taxonomy_tree`. This makes the stored values immediately available to the query resolution pipeline without any additional DB query.

---

### 2.2 `enrich_taxonomy()` — LLM-Based Classification

**File:** `metadata_catalog.py`

**Purpose:** Use an LLM (Claude Haiku) to classify all columns of all tables in a source, producing higher-confidence taxonomy annotations than pattern matching alone can achieve.

**Inputs:** All active attributes per entity, with their `data_type` and `top_values` (sample values).

**Output:** Overwrites `statistical_type`, `semantic_role`, `taxonomy_tree` for all columns. Returns count updated.

**Algorithm:**

1. **Per-entity LLM call.** For each entity (table), construct a prompt listing every column with its data type and up to 20 sample values. All columns in a table are sent in a single call — this gives the LLM cross-column context (e.g., it can see that `sub_category` and `category` co-exist and infer the parent-child relationship).

2. **System prompt constraints.** The LLM is constrained to:
   - Pick exactly one `statistical_type` from a fixed vocabulary of 8 types
   - Pick exactly one `semantic_role` from a fixed vocabulary of 16 roles
   - Return `taxonomy_values` (the distinct stored values) for categorical/ordinal columns

3. **JSON extraction.** The raw LLM response is cleaned of markdown fences, then parsed as JSON. If the top-level parse fails, a fallback substring search finds the first `{...}` block and retries.

4. **Upsert.** Each classified column is written back with `statistical_type`, `semantic_role`, and `taxonomy_tree` (the LLM's `taxonomy_values` list serialised as JSON). This overwrites any `infer_taxonomy()` classifications.

**Why LLM after pattern matching:** Pattern matching handles 80% of cases deterministically. The LLM handles edge cases — renamed columns, multi-lingual names, unconventional schemas — and provides confidence the pattern rules cannot.

---

### 2.3 `_sync_taxonomy_from_kg_nodes()` — KG Annotation Sync

**File:** `orchestrator_api.py`

**Purpose:** After the KG pipeline runs its `profile_node` (which writes taxonomy annotations directly into OWL `rdfs:comment` triples on DatatypeProperty nodes), extract those annotations and write them back to `md_attributes`. This is the highest-confidence taxonomy signal because `profile_node` has access to the full ontology context.

**Inputs:** `source_id`, list of KG `kg_nodes` (each with a `title` string assembled by `translate_node`).

**Output:** Updated `statistical_type`, `semantic_role`, `taxonomy_tree` in `md_attributes`. Returns count updated.

**Algorithm:**

1. **Parse KG node titles.** Each KG node title is a multi-line string. Column definitions appear as:
   ```
     column_name: xsd:string  -- taxonomy: statistical_type=categorical | semantic_role=product_category | ...
   ```
   The regex `^\s{2}(.+?):\s+\S+.*?--\s*(taxonomy:\s*.+)$` extracts `column_name` and the taxonomy annotation string. The `(.+?)` (non-greedy) handles column names with spaces.

2. **Taxonomy annotation regex.** A second regex extracts the three fields from the annotation:
   ```
   taxonomy:\s*statistical_type=(\w+)\s*\|\s*semantic_role=(\w+)(?:\s*\|\s*format_pattern=(\S+))?
   ```

3. **Normalised matching.** Column names in the KG may use spaces (original DB column names) while `md_attributes` may use underscores. A `_norm()` function (`strip().lower().replace(" ", "_")`) normalises both sides before matching, preventing silent misses.

4. **Write-back.** For each matched column, `statistical_type` and `semantic_role` are written. For categorical/ordinal columns, the existing `top_values` in `md_attributes` is reused as `taxonomy_tree` (the KG annotation doesn't carry the value list — that comes from the extraction phase).

---

## 3. Knowledge Graph & Ontology Algorithms

### 3.1 `_extract_ontology()` — OWL Graph Parsing

**File:** `knowledge_graph_agent/nodes/translate_node.py`

**Purpose:** Parse an rdflib OWL graph and produce a structured in-memory representation of classes (tables), datatype properties (columns), and object properties (FK relationships).

**Inputs:** An rdflib `Graph` object populated from OWL/Turtle source.

**Outputs:**
- `classes` dict: `{ uri: { name, comments, datatype_props: [{name, range, comments}] } }`
- `obj_props` list: `[ { name, domain, range, is_functional, is_inv_functional, comments } ]`

**Algorithm:**

**Step 1 — Class extraction.**
Query all subjects typed as `owl:Class`. Skip:
- Blank nodes (anonymous class expressions)
- URIs in XSD, OWL, RDF, RDFS namespaces (meta-vocabulary, not domain classes)

For each class: extract `rdfs:label` as the display name, all `rdfs:comment` triples as annotation strings.

**Step 2 — DatatypeProperty attachment.**
Query all subjects typed as `owl:DatatypeProperty`. For each property:
- Find its `rdfs:domain` — if not in the known classes dict, skip (orphan property)
- Find its `rdfs:range` — extract the local name (e.g., `xsd:string` → `"string"`)
- Collect all `rdfs:comment` triples — these carry column statistics and taxonomy annotations
- Attach to the domain class's `datatype_props` list

**Step 3 — ObjectProperty extraction.**
Query all subjects typed as `owl:ObjectProperty`. For each property:
- Require both `rdfs:domain` and `rdfs:range` to be in the known classes — skip cross-namespace or dangling properties
- Detect `owl:FunctionalProperty` and `owl:InverseFunctionalProperty` annotations to determine cardinality (1:1, 1:N, M:N)
- Extract `rdfs:comment` triples — these carry FK join-column hints like `"Join columns: order_id → id"`
- Deduplicate by URI using a `seen_props` set

**Key design:** Comments on DatatypeProperties carry two distinct kinds of information that must coexist:
1. Statistics (row count, null count, min/max): written by the ontology generator
2. Taxonomy annotations: written by `profile_node`

Since rdflib returns `rdfs:comment` triples in arbitrary order, `translate_node._build_graph_data()` explicitly partitions them — non-taxonomy comments first, taxonomy comments last — so downstream regex parsers can reliably find the taxonomy annotation at the end of a column definition line.

---

### 3.2 `_build_graph_data()` — UI Visualisation Format

**File:** `knowledge_graph_agent/nodes/translate_node.py`

**Purpose:** Convert the parsed ontology classes and object properties into a JSON structure suitable for the vis.js network visualisation in the UI.

**Inputs:** `classes` dict and `obj_props` list from `_extract_ontology()`.

**Output:** `{ nodes: [...], edges: [...] }` where each node/edge has display properties.

**Algorithm:**

**Nodes (one per OWL Class):**
- `id` = class URI (used as stable identifier in edges)
- `label` = class name (display name in the graph)
- `title` = multi-line tooltip string containing:
  - Class-level `rdfs:comment` lines (row count, FD hints)
  - All column definitions: `  col_name: xsd_type  -- stats_comment  -- taxonomy_comment`
- `size` = 20 + min(len(datatype_props) × 2, 20) — tables with more columns appear larger

**Column definition construction (critical ordering):**
```
non-taxonomy comments (statistics) → taxonomy comment (always last)
```
This ordering is mandatory because `_sync_taxonomy_from_kg_nodes()` uses a regex that expects the taxonomy annotation at the end of the line. Without this partition, rdflib's arbitrary triple ordering would silently drop taxonomy annotations.

**Edges (one per OWL ObjectProperty):**
- `from` / `to` = domain / range URIs
- `label` = property name
- `join_columns` = list of `[src_col, tgt_col]` pairs parsed from `rdfs:comment` strings

Two regex patterns extract join column pairs:
1. `Join columns: col1 → col2` — simple same-name joins
2. `Explicit FK: tbl1.col1 → tbl2.col2` — cross-name FK joins

These `join_columns` are later consumed by `understand_node._summarise_graph()` to emit precise `JOIN ON t1.col = t2.col` directives to the SQL planning LLM.

**Cardinality annotation:**
- `is_functional=True, is_inv_functional=True` → `1:1`
- `is_functional=True` only → `1:N`
- Neither → `M:N`

---

### 3.3 `_generate_cypher()` / `_generate_gremlin()` — Query Generation

**File:** `knowledge_graph_agent/nodes/translate_node.py`

**Purpose:** Generate executable Cypher (Neo4j) or Gremlin (TinkerPop) statements that materialise the OWL ontology as a property graph.

**Multi-KG isolation:** Every node, edge, and vertex is stamped with a `kg_id` property equal to the `source_id`. This allows multiple data sources to coexist in the same graph database with zero URI collisions. All MATCH/MERGE clauses filter by both `uri` AND `kg_id`.

**Cypher generation:**

1. **Schema constraint:** `CREATE CONSTRAINT ... FOR (n:KGNode) REQUIRE (n.uri, n.kg_id) IS UNIQUE` — composite key prevents duplicate nodes when the same source is re-indexed.

2. **Class nodes:** `MERGE (n:KGNode:ClassName {uri: '...', kg_id: '...'})` with `ON CREATE SET` for properties. The double label (`KGNode` + class name) allows Neo4j queries to filter by either.

3. **DatatypeProperty columns** are stored as node properties: `n.column_name = 'xsd_type'`. This keeps column metadata on the node rather than creating separate property nodes.

4. **ObjectProperty edges:** `MATCH (a) ... MATCH (b) ... MERGE (a)-[r:RELATIONSHIP_TYPE ...]->(b)`. The MATCH filters by kg_id so edges only connect nodes within the same KG.

**Gremlin generation:**

1. **Vertex upsert:** `g.V().has('uri', ...).has('kg_id', ...).fold().coalesce(unfold(), addV(...))` — find-or-create pattern using both uri and kg_id.

2. **Edge upsert:** `g.V().has(...).as('a').V().has(...).coalesce(inE(...).where(outV().as('a')), addE(...).from('a'))` — prevents duplicate edges on re-index.

---

## 4. Domain Inference & Concept Annotation

These algorithms ensure that every column in an indexed data source has a human-readable business concept label — even when the column name is an opaque abbreviation like `tts`, `arpu`, or `nii`.  The pipeline runs in two stages: first a deterministic voting step identifies the industry domain from the schema; then an LLM call maps each column to a standard concept label, grounded in both the domain context and observed data evidence.

### Why this matters

Abbreviations are deeply ambiguous across industries:

| Column | Telecom meaning | CPG/RGM meaning | Banking meaning |
|--------|----------------|-----------------|-----------------|
| `arpu` | avg revenue per user | — | avg revenue per user |
| `tts` | time to serve | trade spend | — |
| `nii`  | — | — | net interest income |
| `gsv`  | — | gross sales value | — |
| `gmv`  | — | — | — (e-commerce term) |

Without knowing the domain, a generic LLM will guess or return `null`.  With domain context it resolves correctly every time.

---

### 4.1 `_infer_domain_from_report()` — Signal-Word Voting

**File:** [`orchestrator_api.py:847`](orchestrator_api.py#L847)

**Purpose:** Detect the business domain of a data source automatically from its table and column names, so that concept annotation always has a domain context even when the admin did not specify one during source registration.

**Inputs:**
- `report` — the full extraction report dict (`{ "tables": { table_name: { "columns": [...] } } }`)

**Output:** A domain label string such as `"CPG/RGM"`, `"Telecom"`, `"Banking/FS"`, or `""` if no signals fire.

**Algorithm:**

1. **Tokenise** every table name and column name: add the full name and each `_`-split part to a flat token set, all lowercased.
2. **Vote** by intersecting the token set with each domain's signal-word set (`_DOMAIN_SIGNALS`). The score for a domain is the number of tokens that matched.
3. **Select** the domain with the highest score. Ties are broken by whichever entry appears first in the ordered list. Return `""` if no domain scored > 0.

**Signal vocabulary (`_DOMAIN_SIGNALS`):**

| Domain | Example signals |
|--------|----------------|
| CPG/RGM | `gsv`, `nrv`, `tts`, `rsv`, `sku`, `rgm`, `trade_spend`, `market_share` |
| Telecom | `arpu`, `mou`, `subscriber`, `churn_rate`, `prepaid`, `roaming` |
| Banking/FS | `nii`, `nim`, `casa`, `gnpa`, `loan_book`, `provisioning` |
| Insurance | `premium`, `claim`, `loss_ratio`, `lapse_rate`, `underwriting` |
| E-commerce | `gmv`, `aov`, `cac`, `roas`, `cart`, `conversion_rate` |
| Retail | `footfall`, `same_store`, `basket`, `planogram`, `sell_through` |
| Manufacturing | `oee`, `scrap_rate`, `mtbf`, `cycle_time`, `yield_pct` |
| Supply Chain | `otif`, `fill_rate`, `lead_time`, `doh`, `safety_stock` |
| Healthcare | `length_of_stay`, `bed_occupancy`, `readmission`, `patient` |
| HR/People | `headcount`, `attrition`, `time_to_hire`, `engagement_score` |
| Marketing | `impressions`, `cpm`, `cpc`, `media_spend`, `attribution` |
| SaaS/Product | `dau`, `mau`, `mrr`, `arr`, `churn`, `retention_rate` |

**Example:**

Schema with tables `fact_rgm_kpis`, `dim_brand_pack` and columns `gsv`, `nrv`, `tts`, `price_index`, `market_share`, `sku`:

```
Token set: {fact, rgm, kpis, gsv, nrv, tts, price_index, market_share, dim, brand, pack, sku, ...}

Scores:
  CPG/RGM    → gsv ✓, nrv ✓, tts ✓, market_share ✓, sku ✓, rgm ✓  = 6
  Telecom    → 0
  Banking/FS → 0
  ...

Winner: CPG/RGM
```

**Fallback behaviour:** Called in `_index_source` only when `src["domain"]` is blank or `"Other"`.  If inference fires, the result is written back to `src["domain"]` in place and a status event is pushed to the UI:
```
Domain auto-detected: CPG/RGM
```
If inference returns `""`, the empty string is used — the annotation LLM falls back to a generic multi-industry prompt.

---

### 4.2 `_build_col_evidence()` — Per-Column Evidence String

**File:** [`ontology_agent/nodes/build_node.py:116`](ontology_agent/nodes/build_node.py#L116)

**Purpose:** Serialise all observed data signals for one column into a compact pipe-delimited string that is included verbatim in the LLM annotation prompt.

**Input:** A column dict from the extraction report (same schema as persisted to the metadata catalog).

**Output:** A single string like:
```
decimal | min=0 max=500000 avg=45200.00 | top_values=[5000, 12000, 88000] | high-cardinality | description="decimal monetary column" | domain=monetary
```

**Signals included (in order):**

| Signal | Source field | Why it matters |
|--------|-------------|---------------|
| SQL data type | `data_type` | Distinguishes measure (decimal) from flag (integer 0/1) from category (varchar) |
| Numeric range | `min_value`, `max_value`, `avg_value` | Large range → monetary; 0–100 → percentage; small integers → count |
| Top values | `top_values` | Strongest signal: currency-scale numerics → monetary; short strings → dimension |
| Cardinality | `unique_count` / `row_count` | Near-1.0 ratio → continuous measure or identifier; ≤ 20 distinct → categorical |
| Null rate | `null_rate` | High null rate → optional attribute, not a key |
| Rule-based description | `description` | Already grounded by the extraction pipeline's pattern rules |
| Rule-based domain | `domain` | `monetary`, `status_flag`, `categorical`, `numeric_measure`, etc. |

**Design note:** The `description` field is truncated at the first ` — ` clause separator (not at `.`) to avoid cutting mid-number in range strings like `"range: 0.0 – 500000.0"`.

---

### 4.3 `_annotate_column_concepts()` — LLM Concept Mapping

**File:** [`ontology_agent/nodes/build_node.py:178`](ontology_agent/nodes/build_node.py#L178)

**Purpose:** Call Claude to map every column in a table to a standard business concept label (kebab-case, 1–4 words), grounded in both the domain context and the per-column evidence.  Returns `null` for identifiers, timestamps, and self-explanatory names to avoid annotation noise.

**Inputs:**
- `table_name` — used in the user message for context
- `columns` — list of column dicts from the extraction report
- `model` — LLM model ID (default `claude-haiku-4-5-20251001`)
- `domain_hint` — the `source_domain` string from `OntologyConfig`, e.g. `"CPG/RGM | bigquery | Pricing Analytics DB"`

**Output:** `{ column_name: concept_label_or_null }`

**Prompt construction:**

The system prompt is rendered from `_CONCEPT_SYSTEM_TEMPLATE` with two placeholders:

- `{domain_context}` — e.g. `"the CPG/RGM | bigquery | Pricing Analytics DB domain"` or `"enterprise data systems across multiple industries"` when no domain is known
- `{source_context_block}` — the full domain hint displayed under `SOURCE CONTEXT`

The user message lists every column as one evidence line (output of `_build_col_evidence`):
```
Table: fact_rgm_kpis
Columns (name | data_type | range | top_values | cardinality | description | domain):
  gsv   | decimal | min=0 max=9800000 avg=420000.00 | high-cardinality | domain=monetary
  tts   | decimal | min=0 max=500000 avg=45200.00 | top_values=[5000,12000,88000] | high-cardinality | description="decimal monetary column" | domain=monetary
  ...
```

**Grounding rules baked into the system prompt (priority order):**

1. `top_values` — strongest signal (0/1 → flag; currency-scale → monetary; strings → dimension)
2. `description` — already rule-grounded by extraction pipeline; trust it
3. `min/max/avg` — range confirms type
4. `cardinality` — near 1.0 → continuous measure; few distinct → categorical
5. `data_type` — decimal/float → measure; integer → count/id; varchar → category
6. `column name` — last resort; resolve against source domain vocabulary

**Return `null` when:**
- The column is an identifier, primary/foreign key, or date/timestamp
- The name is already self-explanatory (e.g. `revenue`, `headcount`)
- Evidence contradicts the apparent meaning (e.g. a `value` column with only 0/1 values is a status flag, not a monetary measure)

**Model choice:** `claude-haiku-4-5-20251001` by default — cheapest and fast enough for column-name interpretation at indexing time.  Can be overridden per source via `OntologyConfig.llm_model`.

**Failure mode:** Any exception returns an empty dict — annotation is best-effort and never blocks ontology generation.

---

### 4.4 End-to-End Pipeline & Examples

**Flow from source registration to OWL annotation:**

```
Admin registers source
  domain = "Other" (or blank)
        │
        ▼
_index_source() — extraction completes
        │
        ├── admin_domain blank or "Other"?
        │       YES → _infer_domain_from_report()
        │               schema tokens voted against _DOMAIN_SIGNALS
        │               → src["domain"] = "CPG/RGM"   (written in place)
        │       NO  → src["domain"] unchanged
        │
        ▼
POST /generate to ontology_api
  source_domain = "CPG/RGM"
  source_name   = "Pricing Analytics DB"
  db_type       = "bigquery"
  source_description = "Revenue Growth Management data mart"
        │
        ▼
ontology_api.py — builds full_domain_context
  "CPG/RGM | bigquery | Pricing Analytics DB | Revenue Growth Management data mart"
        │
        ▼
build_node.py — for each table:
  _build_col_evidence() per column
  _annotate_column_concepts(domain_hint=full_domain_context)
        │
        ▼
LLM returns { "tts": "trade-spend", "gsv": "gross-sales-value", ... }
        │
        ▼
OWL triple written:
  :fact_rgm_kpis_tts rdfs:comment "Business concept: trade-spend"
```

**Example A — CPG/RGM schema (domain auto-detected)**

Input columns in `fact_rgm_kpis`:

| Column | Data type | Evidence | Annotation |
|--------|-----------|----------|------------|
| `gsv` | decimal | min=0 max=9.8M, high-cardinality, domain=monetary | `gross-sales-value` |
| `nrv` | decimal | min=0 max=8.5M, high-cardinality, domain=monetary | `net-realized-value` |
| `tts` | decimal | min=0 max=500K, top_values=[5000,12000,88000], domain=monetary | `trade-spend` |
| `price_index` | decimal | min=85.2 max=112.4 avg=99.8, domain=numeric_measure | `price-index` |
| `region_id` | integer | 5-distinct-values, domain=categorical | `null` — categorical key |
| `period_dt` | date | — | `null` — timestamp |

**Example B — Telecom schema**

Input columns in `fact_subscriber_kpis`:

| Column | Evidence | Without domain | With Telecom domain |
|--------|----------|---------------|---------------------|
| `arpu` | decimal, min=45 max=312 | `avg-revenue-per-user` *(guessed from name)* | `avg-revenue-per-user` *(confirmed)* |
| `mou` | integer, min=0 max=1200 | `null` *(ambiguous abbreviation)* | `minutes-of-use` |
| `subtype` | varchar, 3-distinct-values: Prepaid/Postpaid/Enterprise | `null` | `null` — self-explanatory |
| `churn` | integer, top_values=[0,1] | `null` *(flag detected)* | `null` *(flag)* |

**Example C — Same abbreviation, different domains**

| Column | Domain hint | LLM output | Correct? |
|--------|-------------|------------|---------|
| `tts` | *(none)* | `total-time-spent` | Wrong — ambiguous |
| `tts` | CPG/RGM | `trade-spend` | Correct |
| `nii` | *(none)* | `null` | Safe fallback |
| `nii` | Banking/FS | `net-interest-income` | Correct |
| `gmv` | *(none)* | `gross-merchandise-value` *(name is known)* | Correct |
| `gmv` | Banking/FS | `null` *(not a banking term)* | Correct — LLM rejects it |

**OWL output** for `tts` in the CPG/RGM case:
```turtle
:fact_rgm_kpis_tts
    a owl:DatatypeProperty ;
    rdfs:label "tts" ;
    rdfs:comment "Business concept: trade-spend" ;
    rdfs:domain :FactRgmKpis ;
    rdfs:range xsd:decimal .
```

---

## 5. Query Resolution Algorithms

These algorithms bridge user natural language to the exact values stored in the database — the critical step that prevents the SQL planning LLM from inventing filter values.

### 5.1 `_load_samples_from_catalog()` — Categorical Value Loading

**File:** `dialog_agent/nodes/understand_node.py`

**Purpose:** For non-file-based sources (PostgreSQL, Redshift, etc.) where live DB sampling is not available, load categorical column values and taxonomy hierarchy from the metadata catalog that was populated during indexing.

**Inputs:** `source_id`

**Output:**
- `samples`: `{ table_name: { col_name: {"values": [...], "categorical": bool} } }`
- `hierarchy`: `{ table_name: { parent_col: { "(all sub-values)": [child_vals] } } }`

**Algorithm:**

1. Load all active entities for the source via `_mc.list_entities(source_id)`.
2. For each entity, call `_mc.get_entity()` to get the full attribute list including `statistical_type` and `top_values`.
3. Include a column in `samples` only if it has `statistical_type ∈ {categorical, ordinal}` AND non-empty `top_values`. This filters out identifiers, measures, and dates that should never appear in WHERE clause resolution.
4. Detect parent-child column pairs via `_detect_parent_child_pairs()` (see §5.2).
5. For each detected pair, build a hierarchy entry with the child column's values grouped under the key `"(all sub-values)"`. Without a live DB cross-tab we cannot know which parent value owns which child values, so the entire child value list is exposed.

This function is the bridge between the indexing pipeline (which runs once) and the query pipeline (which runs per user query) — it ensures the query pipeline has categorical value knowledge even for database sources that cannot be sampled live.

---

### 5.2 `_detect_parent_child_pairs()` — Taxonomy Hierarchy Detection

**File:** `dialog_agent/nodes/understand_node.py`

**Purpose:** Identify parent-child categorical column pairs (e.g., `category → sub_category`) from the column set of a single table, using naming convention and cardinality heuristics.

**Input:** `col_samples` dict for one table.

**Output:** List of `(parent_col, child_col)` tuples.

**Algorithm:**

For every column that starts with `"sub_"`:
1. Derive the parent candidate by stripping the prefix: `sub_category` → `category`.
2. Check that the parent candidate exists as a column in the same table.
3. Verify cardinality: parent distinct count ≤ child distinct count. A parent category must have fewer unique values than its sub-categories.

Both conditions must hold. This dual heuristic (naming + cardinality) prevents false positives like `sub_id` (which might start with `sub_` but has higher cardinality than `id`).

---

### 5.3 `_build_taxonomy_hierarchy()` — Cross-Tab Hierarchy Building

**File:** `dialog_agent/nodes/understand_node.py`

**Purpose:** For file-based sources with live SQLite access, build a full cross-tabulation of parent → child value mappings.

**Input:** SQLite connection, table name, list of `(parent_col, child_col)` pairs.

**Output:** `{ parent_col: { parent_val: [child_vals] } }`

**Algorithm:**

For each parent-child pair, execute:
```sql
SELECT DISTINCT parent_col, child_col
FROM table
WHERE parent_col IS NOT NULL AND child_col IS NOT NULL
ORDER BY parent_col, child_col
```

Group results by parent value. Each parent value maps to the sorted list of distinct child values that co-occur with it in the data. This produces the exact hierarchy used to tell the LLM: *"filtering at parent level captures ALL these child values automatically."*

---

### 5.4 `_fuzzy_match_candidates()` — Multi-Strategy Token Matching

**File:** `dialog_agent/nodes/resolve_node.py`

**Purpose:** Find stored categorical values that are likely to match user query terms using three complementary matching strategies, then promote matched child values to their parent column so the LLM filters at the correct hierarchy level.

**Inputs:**
- `natural_query`: the user's question string
- `categorical_columns`: `{ table: { col: [stored_values] } }`
- `column_hierarchy`: `{ table: { parent_col: { ... } } }`

**Output:** List of candidate matches sorted by score descending, each with `{ table, column, stored_value, overlap_tokens, score, match_type, promoted_from }`.

**Preprocessing:**

Tokenise the query with `\b[a-zA-Z]\w+\b`, strip stop-words (a 60-word set including common English words plus FMCG-specific terms like `"growth"`, `"share"`, `"period"`), and discard tokens shorter than 2 characters. This leaves only semantically significant tokens.

**Three matching strategies (applied per token per stored value):**

**Strategy 1 — Exact substring**
`token in stored_value.lower()` — catches direct word containment. Example: `"snacks"` in `"Snacks & Foods"`.

**Strategy 2 — Stemmed match**
Strip common English suffixes (`-s`, `-es`, `-ing`, `-ed`, `-er`, `-tion`, `-ness`, `-ment`, `-ies`, `-est`) from both the query token and each word in the stored value, then compare stems. Minimum root length: 3 characters. Example: `"snack"` (stem of `"snacks"`) matches `"snack"` (stem of `"Snack"` in `"Snack Foods"`).

**Strategy 3 — Edit distance ≤ 1 (for tokens ≥ 5 chars)**
Compute Levenshtein distance between the query token and each word in the stored value using dynamic programming. Only applied to tokens ≥ 5 characters (shorter tokens have too many 1-edit neighbours). Early exit when `|len(a) - len(b)| > 2`. Example: `"savory"` ↔ `"savoury"` = 1 edit → match.

**Hierarchy promotion:**

After finding direct matches on child columns (e.g., `"savoury"` matches `sub_category = "Savoury Snacks"`), the algorithm promotes parent column candidates. For the parent column of each matching child:

1. Detect `child_col → parent_col` via stored hierarchy OR naming convention (`sub_X` → `X`).
2. For each parent value, compute how many query tokens also match the parent value directly.
3. Score the parent candidate as: `len(parent_direct_matches) + len(child_matched_tokens) + 1` — the `+1` ensures promoted parents outscore direct child matches.

This score design means: if the user says "savoury snacks", the promoted `category = 'Snacks & Foods'` candidate scores 4 while the direct `sub_category = 'Savoury Snacks'` scores 2 — the parent always wins in the hint list.

**Output labels:**
- `match_type = "direct"` — the stored value itself matched the token
- `match_type = "promoted_parent"` — a child's match caused this parent to be promoted

These labels appear in the hint block sent to the LLM, annotated as `[PARENT LEVEL — via child match]`.

---

### 5.5 `_apply_fuzzy_fallback()` — Safety Net Resolution

**File:** `dialog_agent/nodes/resolve_node.py`

**Purpose:** After the LLM has produced term resolutions, fill in any filters where the LLM returned a null `sql_fragment` despite fuzzy evidence existing. This catches cases where the LLM is overly conservative.

**Inputs:**
- `term_resolution`: LLM output list (may contain null sql_fragments)
- `candidates`: fuzzy candidates from `_fuzzy_match_candidates()`
- `categorical_columns`, `column_hierarchy`

**Output:** Patched `term_resolution` list with nulls filled in where possible.

**Algorithm:**

For each resolved filter with a null or empty `sql_fragment`:
1. Tokenise the `user_term` field using the same stop-word filter as the fuzzy pre-match.
2. Score every fuzzy candidate by how many user term tokens match its stored value (using `_token_matches_text()`, same three-strategy check).
3. Apply a `+1` bonus to candidates with `match_type = "promoted_parent"`.
4. Select the highest-scoring candidate.
5. If score > 0, inject a deterministic `sql_fragment`: `LOWER(col) = 'stored_value_lower'`.

The fallback does not make another LLM call — it constructs the fragment directly from the Python-computed best match. This guarantees the filter is exact and correct rather than relying on the LLM a second time.

---

## Algorithm Interaction Summary

```
Index time:
  extract report
       │
       ▼
  persist()              ← upsert tables/columns, CDC log, soft deletes
       │
       ▼
  infer_taxonomy()       ← pattern rules, immediate, no deps
       │
       ▼  (async, may fail gracefully)
  enrich_taxonomy()      ← LLM classification, overwrites pattern
       │
       ▼  (if KG pipeline ran)
  _sync_taxonomy_from_kg_nodes()  ← KG profile_node annotations, highest confidence
       │
       ▼  (if ontology was built)
  _extract_ontology() → _build_graph_data() → _generate_cypher/gremlin()
                         KG stored in Neo4j / in-memory graph_data

Query time (per user question):
  _load_samples_from_catalog()      ← pull top_values from md_attributes
       │
       ▼
  _detect_parent_child_pairs()      ← naming + cardinality heuristics
       │
       ▼
  _build_taxonomy_hierarchy()       ← live cross-tab (file) or flat (catalog)
       │
       ▼
  _fuzzy_match_candidates()         ← substring + stem + edit-distance + hierarchy promotion
       │
       ▼
  LLM resolve call with hints       ← LLM confirms/ranks fuzzy candidates
       │
       ▼
  _apply_fuzzy_fallback()           ← fill nulls deterministically
       │
       ▼
  plan_node SQL generation          ← pre-resolved fragments as mandatory WHERE bindings
```
