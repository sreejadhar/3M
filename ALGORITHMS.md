# Algorithm Reference — DataNanite

> **Last updated:** 2026-04-17

This document describes every major algorithm implemented in the DataNanite system: how each one works, what inputs it uses, what it produces, and what design decisions were made.

---

## Table of Contents

1. [Metadata Extraction & Persistence](#1-metadata-extraction--persistence)
   - 1.1 [persist() — Upsert with CDC](#11-persist--upsert-with-cdc)
   - 1.2 [Redundancy Detection — Jaccard Similarity](#12-redundancy-detection--jaccard-similarity-with-domain-gating)
2. [Taxonomy Algorithms](#2-taxonomy-algorithms)
   - 2.1 [infer_taxonomy() — Deterministic Pattern Classification](#21-infer_taxonomy--deterministic-pattern-based-classification)
   - 2.2 [enrich_taxonomy() — LLM Classification](#22-enrich_taxonomy--llm-based-classification)
   - 2.3 [_sync_taxonomy_from_kg_nodes() — KG Annotation Sync](#23-_sync_taxonomy_from_kg_nodes--kg-annotation-sync)
3. [Knowledge Graph & Ontology Algorithms](#3-knowledge-graph--ontology-algorithms)
   - 3.1 [_extract_ontology() — OWL Graph Parsing](#31-_extract_ontology--owl-graph-parsing)
   - 3.2 [_build_graph_data() — UI Visualisation Format](#32-_build_graph_data--ui-visualisation-format)
   - 3.3 [_generate_cypher() — Declarative Statement Preview](#33-_generate_cypher--declarative-statement-preview)
4. [Domain Inference & Concept Annotation](#4-domain-inference--concept-annotation)
   - 4.1 [_infer_domain_from_report() — Two-Tier Signal Voting](#41-_infer_domain_from_report--two-tier-signal-voting)
   - 4.2 [_build_col_evidence() — Per-Column Evidence String](#42-_build_col_evidence--per-column-evidence-string)
   - 4.3 [_annotate_column_concepts() — LLM Concept Mapping](#43-_annotate_column_concepts--llm-concept-mapping)
   - 4.4 [End-to-End Pipeline & Examples](#44-end-to-end-pipeline--examples)
5. [Query Resolution Algorithms](#5-query-resolution-algorithms)
   - 5.1 [_load_samples_from_catalog()](#51-_load_samples_from_catalog--categorical-value-loading)
   - 5.2 [_detect_parent_child_pairs()](#52-_detect_parent_child_pairs--taxonomy-hierarchy-detection)
   - 5.3 [_build_taxonomy_hierarchy()](#53-_build_taxonomy_hierarchy--cross-tab-hierarchy-building)
   - 5.4 [_fuzzy_match_candidates()](#54-_fuzzy_match_candidates--multi-strategy-token-matching)
   - 5.5 [_apply_fuzzy_fallback()](#55-_apply_fuzzy_fallback--safety-net-resolution)
6. [SQL Post-Processing Pipeline](#6-sql-post-processing-pipeline)
   - 6.1 [_fix_dialect_syntax() — Cross-Dialect Contamination Fixer](#61-_fix_dialect_syntax--cross-dialect-contamination-fixer)
   - 6.2 [_fix_subquery_order_by() — Depth-Aware ORDER BY Guard](#62-_fix_subquery_order_by--depth-aware-order-by-guard)
   - 6.3 [_fix_window_functions() — Window ORDER BY Injection](#63-_fix_window_functions--window-order-by-injection)
   - 6.4 [_fix_multicolumn_subquery() — Scalar Subquery Fix](#64-_fix_multicolumn_subquery--scalar-subquery-fix)
   - 6.5 [_fix_sqlserver_subquery_limits()](#65-_fix_sqlserver_subquery_limits)
7. [SHACL Validation Algorithms](#7-shacl-validation-algorithms)
   - 7.1 [Structural Validation — pyshacl](#71-structural-validation--pyshacl)
   - 7.2 [Semantic Checks — Python](#72-semantic-checks--python)
   - 7.3 [Quality Report Assembly](#73-quality-report-assembly)

---

## 1. Metadata Extraction & Persistence

### 1.1 `persist()` — Upsert with CDC

**File:** `metadata_catalog.py`

**Purpose:** Take the raw extraction report and durably store it — handling new tables, updated statistics, deleted columns, and previously-deleted entities that reappear.

**Inputs:**
- `source_id` — UUID identifying the data source
- `report` — dict `{ tables: { table_name: { columns: [...], row_count, ... } } }`

**Algorithm:**

1. **Source registration.** Upsert into `md_sources`. On first registration, `_infer_domain(source_name)` uses keyword matching to auto-assign a business domain.

2. **Pre-fetch existing state.** All current `md_entities` for this source are loaded into a dict keyed by `(schema_name, table_name)`. Single pre-fetch avoids N+1 queries.

3. **Entity upsert loop.** For each table in the report:
   - If previously soft-deleted (`deleted_from_source=True`) → restore + log `"restored"` event
   - If active → overwrite stats, preserve human-curated `description`
   - If new → INSERT with new UUID

4. **Attribute upsert loop.** Same restore/update/insert for each column. Stats always overwritten; `description` preserved if non-empty. `statistical_type`, `semantic_role`, `taxonomy_tree` are **never touched** here — managed by taxonomy algorithms.

5. **Soft-delete pass.** Entities/attributes absent from the new report are marked `deleted_from_source=True`. Data is preserved; the UI shows removed columns.

6. **CDC logging.** Every structural event (added, deleted, restored, type_changed) appended to `md_changes`.

7. **Redundancy check.** `_run_redundancy_check()` called at end of same transaction.

**Key design:** Never truncate-and-reload — always diff to preserve golden-record flags and descriptions.

---

### 1.2 Redundancy Detection — Jaccard Similarity with Domain Gating

**File:** `metadata_catalog.py` → `_run_redundancy_check()`

**Purpose:** Detect tables that are likely duplicates by comparing column name sets.

**Algorithm:**

1. **Domain gating.** Only compare tables from the same domain to prevent cross-domain false positives.

2. **Column set extraction.** Lowercase all column names for case-normalised comparison.

3. **Jaccard computation.**
   ```
   Jaccard(A, B) = |cols_A ∩ cols_B| / |cols_A ∪ cols_B|
   ```

4. **Threshold:** Jaccard ≥ 0.9 → upsert `md_redundancies`; < 0.9 → delete existing record.

5. **Canonical pair ordering.** Stored as `(min(id_a, id_b), max(id_a, id_b))` for UNIQUE constraint.

**Threshold rationale:** 0.9 is intentionally strict to avoid false positives from tables that share common columns like `id`, `created_at`.

---

## 2. Taxonomy Algorithms

Three layers, each acting as a safety net:

```
Layer 1: infer_taxonomy()             — deterministic, runs immediately after persist()
Layer 2: enrich_taxonomy()            — LLM-based, overwrites layer 1
Layer 3: _sync_taxonomy_from_kg_nodes() — KG annotations, highest confidence
```

### 2.1 `infer_taxonomy()` — Deterministic Pattern-Based Classification

**File:** `metadata_catalog.py`

**Purpose:** Assign `statistical_type` and `semantic_role` using column name regex and SQL data types — zero external dependencies.

**Key invariant:** Does NOT overwrite columns that already have a `statistical_type`.

**Decision sequence (strict priority):**

**Step 1 — Boolean and identifier name rules (highest priority)**

| Pattern | statistical_type | semantic_role |
|---|---|---|
| `\b(is\|has\|flag\|active\|enabled\|status_flag)\b` | boolean | boolean_flag |
| `(_\|^)(id\|key\|code\|pk\|sk\|seq\|num\|nr\|no)($\|_)` | identifier | identifier |

**Step 2 — Data type rules**

| Data type | statistical_type | semantic_role |
|---|---|---|
| date / datetime / timestamp | date | time_dimension_key |
| int / decimal / float / money | continuous | measure |

**Step 3 — Remaining name rules** (text types only)

Patterns for: fiscal year, time period, geography, sub-category, category, brand, customer, channel, division.

**Step 4 — Cardinality fallback**

`unique_count ≤ 50` and text type → `categorical / other`.

---

### 2.2 `enrich_taxonomy()` — LLM-Based Classification

**File:** `metadata_catalog.py`

**Algorithm:**

1. Per-entity LLM call with all columns + data types + up to 20 sample values.
2. Constrained to fixed vocabularies: 8 `statistical_type` values, 16 `semantic_role` values.
3. JSON response parsed; `taxonomy_tree` populated with the LLM's `taxonomy_values`.
4. Overwrites any `infer_taxonomy()` classifications.

**Why LLM after pattern matching:** Pattern matching handles ~80% of cases deterministically. LLM handles renamed columns, multi-lingual names, unconventional schemas.

---

### 2.3 `_sync_taxonomy_from_kg_nodes()` — KG Annotation Sync

**File:** `orchestrator_api.py`

**Purpose:** After `profile_node` writes taxonomy annotations into OWL `rdfs:comment` triples, extract and write them back to `md_attributes`. Highest-confidence signal.

**Algorithm:**

1. Parse KG node titles with regex: `^\s{2}(.+?):\s+\S+.*?--\s*(taxonomy:\s*.+)$`
2. Extract three fields: `statistical_type`, `semantic_role`, `format_pattern`
3. Normalised matching: `strip().lower().replace(" ", "_")` on both sides
4. Write back `statistical_type` and `semantic_role`; reuse `top_values` for `taxonomy_tree`

---

## 3. Knowledge Graph & Ontology Algorithms

### 3.1 `_extract_ontology()` — OWL Graph Parsing

**File:** `knowledge_graph_agent/nodes/translate_node.py`

**Steps:**

1. **Class extraction.** Query `owl:Class` subjects. Skip blank nodes and XSD/OWL/RDF/RDFS meta-vocabulary.

2. **DatatypeProperty attachment.** For each `owl:DatatypeProperty`: find `rdfs:domain` (skip orphans), find `rdfs:range` (extract local XSD type), collect `rdfs:comment` triples (statistics + taxonomy).

3. **ObjectProperty extraction.** For each `owl:ObjectProperty`: require both domain and range in known classes, detect `FunctionalProperty` / `InverseFunctionalProperty` for cardinality, extract join-column hints from `rdfs:comment`.

**Key design:** Comments carry two distinct information types (statistics + taxonomy) that must coexist. `translate_node` explicitly partitions them — non-taxonomy first, taxonomy last — so downstream regex parsers find annotations reliably.

---

### 3.2 `_build_graph_data()` — UI Visualisation Format

**File:** `knowledge_graph_agent/nodes/translate_node.py`

**Nodes (one per OWL Class):**
- `id` = class URI
- `label` = class name
- `title` = multi-line tooltip: row count, FD hints, all columns with types and taxonomy
- `size` = 20 + min(column_count × 2, 20)

**Column definition ordering (critical):**
```
non-taxonomy comments → taxonomy comment (always last)
```
Mandatory because `_sync_taxonomy_from_kg_nodes()` expects taxonomy annotation at end of line.

**Edges (one per OWL ObjectProperty):**
- Semantic verb label (e.g. `has Customer (1:N)`) derived from domain + cardinality
- `join_columns` = `[[src_col, tgt_col]]` pairs parsed from `rdfs:comment`
- Cardinality: `1:1` / `1:N` / `M:N` from FunctionalProperty flags

---

### 3.3 `_generate_cypher()` — Declarative Statement Preview

**File:** `knowledge_graph_agent/nodes/translate_node.py`

**Multi-KG isolation:** Every node/edge stamped with `kg_id` = `source_id`. MATCH/MERGE clauses always filter by `kg_id`.

The real output of `translate_node` is `graph_data` (`{nodes, edges}`), which
`execute_node` persists to the KG snapshot store (`kg_store.py`) — there is no
live graph database these statements run against. `_generate_cypher()` exists
purely to produce a human-readable, declarative preview of the same graph
(exposed via `GET /jobs/{id}/queries` for documentation/export):

```cypher
CREATE CONSTRAINT kg_node_uri IF NOT EXISTS FOR (n:KGNode) REQUIRE (n.uri, n.kg_id) IS UNIQUE
MERGE (n:KGNode:Orders {uri: '...', kg_id: '...'}) ON CREATE SET n.order_id = 'integer'
MATCH (a:KGNode {uri: '...'}), (b:KGNode {uri: '...'})
MERGE (a)-[r:FK_CUSTOMERS {cardinality: '1:N'}]->(b)
```

---

## 4. Domain Inference & Concept Annotation

### 4.1 `_infer_domain_from_report()` — Two-Tier Signal Voting

**File:** `orchestrator_api.py`

**Purpose:** Detect the compound business domain (Industry/Function) from table and column names automatically at index time.

**Inputs:** Full extraction report dict.

**Output:** Compound label such as `"CPG/Supply Chain"`, `"LS/FP&A"`, `"Telecom"`, or `""`.

**Algorithm:**

1. **Tokenise** every table name and column name: full name + `_`-split parts, all lowercased.

2. **Score independently** against two signal tiers:
   - `_INDUSTRY_SIGNALS`: CPG, Life Sciences, Healthcare, Telecom, Banking/FS, Insurance, Retail, E-commerce, Manufacturing, SaaS
   - `_FUNCTION_SIGNALS`: RGM, FP&A, Supply Chain, Sales, Marketing, HR/People, Operations, CX

3. **Return compound label** when both tiers fire: `f"{industry_label}/{function_label}"`. Single label when only one tier fires. Empty string if neither fires.

**Industry signals (examples):**

| Industry | Signal words |
|---|---|
| CPG | `brand`, `sku`, `category`, `retailer`, `upc`, `rsv`, `gsv`, `pack_size`, `distributor` |
| Life Sciences | `trial`, `cohort`, `adverse_event`, `molecule`, `indication`, `clinical` |
| Telecom | `arpu`, `mou`, `subscriber`, `churn_rate`, `prepaid`, `roaming`, `spectrum` |
| Banking/FS | `nii`, `nim`, `casa`, `gnpa`, `loan_book`, `provisioning`, `kyc` |

**Function signals (examples):**

| Function | Signal words |
|---|---|
| RGM | `promo`, `trade_spend`, `price_index`, `mix_effect`, `promotion_type` |
| FP&A | `gl_account`, `cost_centre`, `scenario`, `version`, `reforecast`, `profit_centre` |
| Supply Chain | `purchase_order`, `goods_receipt`, `shipment`, `vendor`, `otif`, `lead_time` |

**Fallback behaviour:** If inference returns `""`, the LLM concept annotator uses a generic multi-industry prompt rather than domain-specific vocabulary.

---

### 4.2 `_build_col_evidence()` — Per-Column Evidence String

**File:** `ontology_agent/nodes/build_node.py`

**Purpose:** Serialise all observed data signals for one column into a compact evidence string for the LLM annotation prompt.

**Output example:**
```
decimal | min=0 max=500000 avg=45200.00 | top_values=[5000, 12000, 88000] | high-cardinality | description="decimal monetary column" | domain=monetary
```

**Signals (in order):** SQL data type → numeric range → top values → cardinality ratio → null rate → rule-based description → rule-based domain.

---

### 4.3 `_annotate_column_concepts()` — LLM Concept Mapping

**File:** `ontology_agent/nodes/build_node.py`

**Purpose:** Map every column to a standard business concept label (kebab-case, 1–4 words) grounded in domain context and per-column evidence.

**Grounding rules (priority order):**
1. `top_values` — strongest signal
2. `description` — already rule-grounded by extraction pipeline
3. `min/max/avg` — range confirms type
4. `cardinality` — near 1.0 → continuous; few distinct → categorical
5. `data_type` — decimal/float → measure; integer → count/id
6. `column name` — last resort, resolved against domain vocabulary

**Returns `null` for:** identifiers, PKs, FKs, timestamps, self-explanatory names.

**Model:** `claude-haiku-4-5-20251001` — cheapest, sufficient for column-name interpretation. Any exception returns empty dict — never blocks ontology generation.

---

### 4.4 End-to-End Pipeline & Examples

```
Admin registers source → domain = "Other"
        │
        ▼
_index_source() — extraction completes
        │
        ├── domain blank / "Other"?
        │       YES → _infer_domain_from_report() → "CPG/Supply Chain"
        │       NO  → domain unchanged
        │
        ▼
POST /generate to ontology_api
  source_domain = "CPG/Supply Chain"
        │
        ▼
build_node — per table:
  _build_col_evidence() + _annotate_column_concepts(domain_hint)
        │
        ▼
OWL triple: :fact_kpis_tts rdfs:comment "Business concept: trade-spend"
```

**Disambiguation examples:**

| Column | Domain | LLM output |
|---|---|---|
| `tts` | CPG/RGM | `trade-spend` |
| `tts` | (none) | `total-time-spent` *(wrong)* |
| `nii` | Banking/FS | `net-interest-income` |
| `nii` | (none) | `null` *(safe fallback)* |
| `arpu` | Telecom | `avg-revenue-per-user` |
| `mou` | Telecom | `minutes-of-use` |
| `mou` | (none) | `null` *(ambiguous)* |

---

## 5. Query Resolution Algorithms

Bridge user natural language to exact stored values — preventing the SQL planner from inventing filter values.

### 5.1 `_load_samples_from_catalog()` — Categorical Value Loading

**File:** `dialog_agent/nodes/understand_node.py`

**Algorithm:**
1. Load all active entities + attributes via metadata catalog.
2. Include column in `samples` only if `statistical_type ∈ {categorical, ordinal}` AND non-empty `top_values`.
3. Detect parent-child pairs via `_detect_parent_child_pairs()`.
4. Group child values under `"(all sub-values)"` when live cross-tab is unavailable.

---

### 5.2 `_detect_parent_child_pairs()` — Taxonomy Hierarchy Detection

**File:** `dialog_agent/nodes/understand_node.py`

For every column starting with `"sub_"`:
1. Derive parent by stripping prefix: `sub_category` → `category`
2. Verify parent exists in same table
3. Verify cardinality: parent distinct count ≤ child distinct count

Both conditions required to prevent false positives.

---

### 5.3 `_build_taxonomy_hierarchy()` — Cross-Tab Hierarchy Building

**File:** `dialog_agent/nodes/understand_node.py`

For file-based sources with live SQLite, executes:
```sql
SELECT DISTINCT parent_col, child_col FROM table
WHERE parent_col IS NOT NULL AND child_col IS NOT NULL
ORDER BY parent_col, child_col
```
Groups by parent value → list of child values.

---

### 5.4 `_fuzzy_match_candidates()` — Multi-Strategy Token Matching

**File:** `dialog_agent/nodes/resolve_node.py`

**Preprocessing:** Tokenise with `\b[a-zA-Z]\w+\b`, strip 60-word stop-word set, discard tokens < 2 chars.

**Three strategies per token:**

1. **Exact substring** — `token in stored_value.lower()`
2. **Stemmed match** — strip common suffixes (`-s`, `-es`, `-ing`, `-ed`, `-er`, `-tion`, `-ness`, `-ment`, `-ies`, `-est`), compare stems (min length 3)
3. **Edit distance ≤ 1** (tokens ≥ 5 chars) — Levenshtein DP with early exit when `|len(a) - len(b)| > 2`

**Hierarchy promotion:** When a child column matches, promote the parent candidate with score = `parent_direct_matches + child_matched_tokens + 1`. The `+1` ensures promoted parents always outscore direct child matches so the LLM filters at the correct hierarchy level.

---

### 5.5 `_apply_fuzzy_fallback()` — Safety Net Resolution

**File:** `dialog_agent/nodes/resolve_node.py`

For each filter where the LLM returned null `sql_fragment`:
1. Score every fuzzy candidate against user term tokens (same three strategies)
2. Apply `+1` bonus to `match_type = "promoted_parent"` candidates
3. If score > 0 → inject deterministic `LOWER(col) = 'value'` fragment

No second LLM call — fragment is constructed directly from Python-computed best match.

---

## 6. SQL Post-Processing Pipeline

**File:** `dialog_agent/nodes/plan_node.py`

Every LLM-generated SQL passes through this ordered pipeline before execution:

```python
sql = _qualify_sql(sql, db_schema, table_labels)
sql = _fix_count_vs_sum(sql, natural_query)
sql = _fix_percentage(sql, natural_query, db_type)
sql = _enforce_sql_limits(sql, row_limit, db_type)
sql = _fix_dialect_syntax(sql, db_type)              # NEW — cross-dialect fixer
sql = _fix_sqlserver_subquery_limits(sql, db_type)
sql = _fix_subquery_order_by(sql, db_type)
sql = _fix_window_functions(sql, db_type)
sql = _fix_multicolumn_subquery(sql)
sql = _fix_distinct_order_by(sql)
```

**Ordering is critical:** `_fix_dialect_syntax` runs before the SQL Server-specific fixers so OFFSET is already present before `_fix_subquery_order_by` evaluates protection.

---

### 6.1 `_fix_dialect_syntax()` — Cross-Dialect Contamination Fixer

**Purpose:** Catch PostgreSQL/ANSI idioms that the LLM emits for the wrong target dialect and rewrite them at runtime.

**Algorithm:** Pattern-matched regex substitutions per dialect group:

| Transformation | Trigger | Target dialects |
|---|---|---|
| `ILIKE` → `LOWER(col) LIKE LOWER(pat)` | `\bILIKE\b` | SQL Server, Oracle, BigQuery, SQLite |
| `col::TYPE` → `CAST(col AS TYPE)` | `\w+\s*::\s*\w+` | non-PostgreSQL |
| `NOW()` → `GETDATE()` | `\bNOW\s*\(\s*\)` | SQL Server |
| `NOW()` → `SYSDATE` | `\bNOW\s*\(\s*\)` | Oracle |
| `NOW()` → `CURRENT_TIMESTAMP()` | `\bNOW\s*\(\s*\)` | BigQuery |
| `CURRENT_DATE` → `CAST(GETDATE() AS DATE)` | `\bCURRENT_DATE\b` | SQL Server |
| `CURRENT_DATE` → `SYSDATE` | `\bCURRENT_DATE\b` | Oracle |
| `DATE_TRUNC(unit, col)` → `DATEADD(unit, DATEDIFF(unit, 0, col), 0)` | `\bDATE_TRUNC\b` | SQL Server |
| `DATE_TRUNC(unit, col)` → `TRUNC(col, 'fmt')` | `\bDATE_TRUNC\b` | Oracle |
| `LENGTH(` → `LEN(` | `(?<![A-Z_])LENGTH\s*\(` | SQL Server |
| `LIMIT N` (statement end) → `FETCH FIRST N ROWS ONLY` | `\bLIMIT\s+(\d+)\s*$` | Oracle |
| `\|\|` → `+` | `['\w)]\s*\|\|(\s*['\w(])` | SQL Server |

---

### 6.2 `_fix_subquery_order_by()` — Depth-Aware ORDER BY Guard

**Purpose:** Strip bare `ORDER BY` from SQL Server subqueries/CTEs where it is illegal (error 1033) unless protected by `TOP`, `OFFSET`, or `FOR XML`.

**Root cause addressed:** `_fix_sqlserver_subquery_limits` adds `OFFSET 0 ROWS FETCH NEXT N ROWS ONLY` inside *nested* subqueries. If the outer block's OFFSET check was naive (`re.search(r'\bOFFSET\b', block_content)`), the nested OFFSET would be found and incorrectly exempt the outer bare ORDER BY from being stripped.

**Fix — `_keyword_at_d0()` depth-aware scan:**
```python
def _keyword_at_d0(text: str, pattern: str) -> bool:
    depth = 0
    for tok in re.finditer(r'[()]|' + pattern, text, re.IGNORECASE):
        if tok.group(0) == '(':  depth += 1
        elif tok.group(0) == ')': depth -= 1
        elif depth == 0: return True
    return False
```

Applied separately to `text_before_ORDER_BY` (for TOP check) and `text_after_ORDER_BY` (for OFFSET/FOR XML check). Only keywords at paren-depth 0 relative to the block are counted as protection.

**OVER() guard:** Before the depth scan, if the token immediately preceding the enclosing `(` matches `\bOVER$`, the ORDER BY is part of a window function spec and is **never** stripped.

---

### 6.3 `_fix_window_functions()` — Window ORDER BY Injection

**Purpose:** Inject missing `ORDER BY` into navigation window function `OVER()` clauses for all dialects. SQL Server, Oracle, and BigQuery all require ORDER BY inside OVER for LAG, LEAD, ROW_NUMBER, RANK, DENSE_RANK, FIRST_VALUE, LAST_VALUE, NTILE, CUME_DIST, PERCENT_RANK.

**Algorithm:**
1. Find navigation function calls with regex `_ORDER_REQUIRED`
2. Scan forward with paren-depth counter to find the matching `)`
3. Extract the OVER content; skip if ORDER BY already present
4. Inject ORDER BY: reuse first PARTITION BY column if available; fall back to `(SELECT NULL)` as a no-op sentinel

---

### 6.4 `_fix_multicolumn_subquery()` — Scalar Subquery Fix

**Purpose:** Rewrite scalar subqueries that return multiple columns — illegal in all dialects.

**Two patterns:**

1. **Scalar IN with multi-column SELECT:**
   ```sql
   WHERE id IN (SELECT id, rn FROM ranked)
   -- becomes:
   WHERE id IN (SELECT id FROM ranked)
   ```

2. **Row-value constructor `(a,b) IN (SELECT x,y FROM t)`** → rewrites to EXISTS:
   ```sql
   WHERE EXISTS (SELECT 1 FROM t WHERE t.x = a AND t.y = b)
   ```
   Only triggered for `(col, col) IN (SELECT ...)` patterns — does NOT rewrite every `NOT IN`.

---

### 6.5 `_fix_sqlserver_subquery_limits()`

**Purpose:** Convert `ORDER BY … LIMIT N` inside SQL Server CTEs/subqueries to `ORDER BY … OFFSET 0 ROWS FETCH NEXT N ROWS ONLY`.

**Algorithm:** Paren-depth scan identifies ORDER BY + LIMIT pairs inside subquery blocks; rewrites LIMIT to OFFSET/FETCH; strips bare LIMITs (no ORDER BY) entirely. SQL Server only — no-op for other dialects.

**Pipeline position:** Must run **before** `_fix_subquery_order_by` so the injected OFFSET keyword is in place when the ORDER BY protection check runs.

---

## 7. SHACL Validation Algorithms

**File:** `shacl_agent/nodes/`

The SHACL validation pipeline runs two independent passes and assembles a quality report.

### 7.1 Structural Validation — pyshacl

**File:** `shacl_agent/nodes/validate_node.py`

Calls `pyshacl.validate()` with:
- `inference="rdfs"` — run RDFS-level inference so `owl:Class` instances resolve
- `abort_on_first=False` — collect all violations before returning
- `allow_warnings=True` — distinguish `sh:Violation` from `sh:Warning`

Results graph is walked for `sh:ValidationResult` subjects; each result extracts `sh:focusNode`, `sh:resultPath`, `sh:resultMessage`, `sh:sourceShape`, `sh:resultSeverity`.

**Graceful degradation:** If `pyshacl` is not installed, records a non-fatal error and continues with semantic checks — the service never crashes due to a missing optional dependency.

---

### 7.2 Semantic Checks — Python

Four checks that are beyond SHACL expressivity:

**OrphanClass:**
```
all_classes = subjects(rdf:type, owl:Class)
referenced  = objects(rdfs:domain) ∪ objects(rdfs:range)
orphans     = all_classes − referenced − blank_nodes
```

**LowCoverage:**
Regex `[Cc]overage[:\s]+(\d+(?:\.\d+)?)\s*%` extracts coverage from `rdfs:comment` on `owl:ObjectProperty` nodes. Fires when `coverage < config.min_coverage` (default 0.5).

**NamespaceDrift:**
Detects the `owl:Ontology` subject URI as base namespace. Flags any `owl:Class` / `owl:DatatypeProperty` / `owl:ObjectProperty` whose URI does not start with that base. Skips blank nodes.

**DuplicateClassLabel:**
Builds `{ label.lower(): [uris] }` map over all `owl:Class` nodes. Flags any label with `len(uris) > 1`.

---

### 7.3 Quality Report Assembly

**File:** `shacl_agent/nodes/report_node.py`

**Quality label logic:**
```
PASS  ← conforms=True AND no violations AND no semantic Violations
WARN  ← no violations but warnings or semantic issues exist
FAIL  ← at least one sh:Violation
```

**Suggestions are auto-generated** for each issue category:
- Orphan classes → "Review IND detection threshold or add FK constraints"
- Low coverage → "Treat as candidate links, not confirmed JOIN keys"
- Namespace drift → "Normalise all URIs to the declared owl:Ontology base URI"
- Duplicate labels → "Use unique rdfs:label values — NL query planner uses labels to disambiguate tables"
- Missing rdfs:domain → "Breaks OWL-DL reasoning and KG node→edge translation"

The report is designed so the Tech UI can render a per-check results table — every check shown with pass/fail status, description, and expandable issue detail — giving the user full visibility into both what passed and what failed.

---

## Algorithm Interaction Summary

```
Index time:
  extract report
       │
       ▼
  persist()                  ← upsert, CDC log, soft deletes
       │
       ▼
  infer_taxonomy()            ← pattern rules, immediate
       │
       ▼ (async, graceful failure)
  enrich_taxonomy()           ← LLM, overwrites pattern
       │
       ▼ (after KG pipeline)
  _sync_taxonomy_from_kg_nodes() ← KG annotations, highest confidence
       │
       ▼
  _infer_domain_from_report() ← two-tier industry/function signal voting
       │
       ▼
  build_node → _annotate_column_concepts() ← domain-grounded LLM concept labels
       │
       ▼
  _extract_ontology() → _build_graph_data() → _generate_cypher() (preview only)
  KG stored as a {nodes, edges} snapshot in the KG snapshot store (kg_store.py)

  (optional, user-initiated)
  validate_ontology → SHACL API → structural + semantic checks → quality report

Query time (per user question):
  _load_samples_from_catalog()       ← pull top_values from md_attributes
       │
       ▼
  _detect_parent_child_pairs()       ← naming + cardinality heuristics
       │
       ▼
  _build_taxonomy_hierarchy()        ← live cross-tab (file) or flat (catalog)
       │
       ▼
  _fuzzy_match_candidates()          ← substring + stem + edit-distance + hierarchy promotion
       │
       ▼
  LLM resolve call with hints        ← LLM confirms/ranks fuzzy candidates
       │
       ▼
  _apply_fuzzy_fallback()            ← fill nulls deterministically
       │
       ▼
  plan_node SQL generation           ← pre-resolved fragments as mandatory WHERE bindings
       │
       ▼
  _fix_dialect_syntax()              ← cross-dialect contamination (ILIKE, ::, NOW, etc.)
  _fix_sqlserver_subquery_limits()   ← ORDER BY + LIMIT → OFFSET/FETCH in CTEs
  _fix_subquery_order_by()           ← depth-aware bare ORDER BY removal
  _fix_window_functions()            ← inject missing ORDER BY in OVER()
  _fix_multicolumn_subquery()        ← scalar subquery column count fix
  _fix_distinct_order_by()           ← add ORDER BY cols to SELECT DISTINCT
       │
       ▼
  execute_node → synthesize_node    ← SQL results → narrative insights
```
