# Design — Semantic Lexicon + Data-Driven Evaluation Loop

A two-component subsystem that resolves high-level business concepts
("promotion count", "top performer", "eligible") to **real, profiled columns
before SQL is generated**, and **persists** each resolution so the same question
returns the same answer every time.

> **How to read this doc.** §1–§2 state the problem and inventory what already
> exists, so nothing is rebuilt. §3 is the architecture at a glance. §4 designs
> the **Semantic Lexicon** (the persistent knowledge layer). §5 designs the
> **Evaluation Loop** (the runtime resolver). §6 is the integration with the
> existing LangGraph pipeline. §7 is the complete change surface — every file
> touched. §8 is an end-to-end worked example. §9–§13 cover rollout, accuracy,
> risks, and measurement. Every claim about current behaviour cites `file:line`.

**Status:** design only. **No implementation exists.** No existing file has been
modified by this design.

**Supersedes:** `docs/Derived_Metric_Resolution_Design.md`, which covered the
loop plus a thin cache but not the lexicon. That document can be deleted once
this one is accepted.

---

## 1. Problem

### 1.1 Symptom

The same question intermittently succeeds and fails. Against the `HRData`
source, *"who are eligible for next promotion"* sometimes answers and sometimes
returns:

```
No query results were produced. …
plan_node: query q1 skipped — unremovable hallucinated column(s):
  ['promotion_count_2yr', 'education_count', 'degrees',
   'institutions', 'certificate_count']
plan_node: LLM returned 2 item(s) but all were dropped or empty.
```

None of those columns exist. `HRData` holds five tables of **raw records**, not
rollups (verified from the `kg_snapshots` row for source
`807e7775-40db-4794-b9aa-a832631e10a9`): `job_history`, `job_current`,
`education`, `certificates`, `prev_employment`. Every one of those five metrics
must be **computed**, not looked up.

### 1.2 Why it is inconsistent

`llm_temperature` is already `0.0` (`dialog_agent/config.py:50`), so this is not
sampling randomness. The real causes:

1. **Nothing anchors the resolution.** The mechanism built for exactly this —
   glossary/KPI definitions with a `sql_hint` — holds **6 rows system-wide, none
   for HR** (`data/metadata.db`, table `glossary_terms`). `verified_queries`
   (`dialog_agent/verified_queries.py:223`) has no entry for these questions. So
   the derivation is re-invented on every single call.
2. Names like `promotion_count_2yr` are highly plausible — real HR warehouses
   often *do* carry such rollups — so the model weighs "derive it" against
   "assume it exists" with no tie-breaker in the prompt.
3. Serving-level non-determinism can flip a request sitting on the boundary
   between two completions even at temperature 0.

### 1.3 Two failure classes — scope boundary

| Class | Nature | Examples observed in logs | Addressed by this design? |
|---|---|---|---|
| **A — identifier naming/casing** | Column **exists**; model emitted a display-cased variant | `"Market Name"` (real: `market_name`), `E."country"`, `"Limit Amt"`, `"Meals Expense Details"`, `"Material Name"`, `"Honorarium Expense Details"` | **No** — that is `sql_identifier_resolver.repair()`'s job (§2.4) |
| **B — derived metric** | Column exists in **no** form; concept must be computed | `promotion_count_2yr`, `education_count`, `certificate_count`, `degrees`, `institutions` | **Yes** |

Class A dominated by volume in the LifeScience_V2 runs observed on 2026-07-30.
The HR failure is Class B. **This design targets Class B.** See §9.0 — Class A
should be addressed first, and is nearly free.

### 1.4 Retrieval is not the cause for `HRData`

`retrieve_node` skips GraphRAG entirely when the schema has no more tables than
`graphrag_min_tables`, which is **10** (`config.py:88`; skip logic at
`retrieve_node.py:588-593`). `HRData` has 5 tables, so the planner already
receives the **full** schema every request. The needed tables are present. The
defect is in planning, not retrieval.

---

## 2. What already exists (reuse, do not rebuild)

### 2.1 Profiling data — no runtime metadata scan needed

`profile_node` and the metadata report already capture per column: `data_type`,
`unique_count`, `null_count`, `min_value`, `max_value`, `top_values`,
`semantic_role`, `statistical_type` — plus report-level
`functional_dependencies`, `inclusion_dependencies`, `fk_candidates`, and
`cardinality_relationships`.

Additionally, `understand_node` already materialises actual literal values into
state: `categorical_columns: {table: {column: [values]}}`
(`understand_node.py:1270`, key declared at `state.py:98`).

**Consequence:** evidence assembly is a local read of data already in state. No
live interrogation query, and no Neo4j `apoc.meta.data()` equivalent is needed
(none exists — see §2.5).

Why literal values matter: to count promotions you must know the **token** in
the source column (`'Promotion'` vs `'PROMO'` vs `'Position Change'`). Column
*names* cannot supply that; profiled values can.

### 2.2 Similarity ranking infrastructure

`verified_queries` already implements a robust ranking ladder:
sentence-transformers embeddings (`verified_queries.py:211`) with a stemmed
bag-of-words cosine fallback (`:175-208`) that needs no model download and no
API key. The lexicon **reuses these helpers** rather than adding a dependency.

### 2.3 Storage abstraction

`dialog_agent/pg_store.py` selects PostgreSQL (when `APP_ENV=production` +
`KG_POSTGRES_DSN`) else SQLite, with `cursor_ctx()`, `ddl()`, `execute()`,
`insert_returning_id()`. `verified_queries.py` is the reference implementation to
mirror.

### 2.4 AST validation and repair

`dialog_agent/nodes/sql_identifier_resolver.py` provides `SchemaGraph`
(`:30`, constructed from `{table: {columns}}` at `:38`, with `has_table()` /
`has_column()` / `all_tables()` at `:45-53`), `find_hallucinated_identifiers()`,
and `repair()`. It backs the hallucinated-table, hallucinated-column,
cross-table-column, and join-pair checks in `plan_node._validate_plan_items`.

**Currently gated off:** `_AST_RESOLVER_ENABLED` defaults to `0`
(`PLAN_NODE_AST_HALLUCINATION_ENABLED`), despite `hallucination_shadow_report.txt`
reporting 363 real statements across three sources with 0 divergent outcomes.

`plan_node._extract_table_columns()` (`:2341`) already parses `schema_context`
into the exact `{TABLE: {columns}}` shape `SchemaGraph` consumes.

### 2.5 No graph database

Neo4j and Gremlin were removed (commit `1bf84eb`). KG state is `kg_store`
snapshots; queries run as SQL against the source DB. Designs borrowed from Neo4j
reference architectures must be translated: Cypher → SQL, `apoc.meta.data()` →
§2.1.

### 2.6 Probe execution primitive

`execute_node._run_sql(cfg, sql)` (`:750`) returns `{columns, rows, error}` and
**never raises** — it catches everything. This is the probe mechanism (§5.5);
nothing new is required.

### 2.7 Corrective retry

`plan_node` has a one-shot retry feeding rejection reasons back to the LLM. It
was **inert** for this failure mode until commit `0ea5a91`: the
unremovable-hallucinated-column paths appended only to `state["errors"]`, never
to `reasons`, so the `if not sql_queries and retry_reasons and plan:` guard never
fired. Now fixed, plus a generic retry rule to derive rollups via
`COUNT()/SUM()/GROUP BY`.

**That lowers the failure rate but does not remove the variance source** — the
derivation is still re-invented per call. Removing that variance is the whole
point of the lexicon.

---

## 3. Architecture at a glance

Two components with a deliberate division of labour:

| | **Semantic Lexicon** (§4) | **Evaluation Loop** (§5) |
|---|---|---|
| Nature | Persistent knowledge layer | Runtime resolver |
| Answers | "Has this concept been resolved before, for this source?" | "Given real data, how *can* this concept be computed?" |
| Cost | One indexed lookup | 1–2 LLM calls + 1 probe query |
| Frequency | Every request | Cache miss only |
| Output | Stored bindings | New bindings → written to lexicon |
| Purpose | **Stability** (same answer every time) | **Coverage** (no hand-authoring needed) |

The lexicon without the loop requires humans to author every term. The loop
without the lexicon is stateless — it would answer every time but with a
*possibly different derivation each run*, converting a loud failure into silent
drift. **Both are required**; that is why they are designed together.

```
NLQ
 │
 ▼
┌─────────────────────────────────────────────────────────┐
│ identify candidate concepts (cached per question)       │
└─────────────────────────────────────────────────────────┘
 │
 ▼
┌─────────────────────────────────────────────────────────┐
│ SEMANTIC LEXICON lookup   exact → alias → embedding     │
└─────────────────────────────────────────────────────────┘
 │                                        │
 │ HIT (deterministic)                    │ MISS
 │                                        ▼
 │                    ┌──────────────────────────────────────┐
 │                    │ EVALUATION LOOP                      │
 │                    │  1. assemble evidence (§2.1 data)    │
 │                    │  2. Data Dissector LLM (constrained) │
 │                    │  3. mechanical validation            │
 │                    │  4. probe execution                  │
 │                    │  5. write back to lexicon            │
 │                    └──────────────────────────────────────┘
 │                                        │
 │        ┌───────────────────────────────┴──── unresolvable ──┐
 ▼        ▼                                                     ▼
bindings injected into plan_node                      plan as today
(prompt + validator)                                  (no regression)
 │
 ▼
AST gate — UNCHANGED, still the enforcement boundary (§5.7)
```

---

## 4. Component A — Semantic Lexicon

### 4.1 Data model

New table. **No existing table is altered.**

```sql
CREATE TABLE semantic_lexicon (
  entry_id          TEXT PRIMARY KEY,     -- uuid4
  source_id         TEXT NOT NULL,        -- scope; '' = global
  term              TEXT NOT NULL,        -- normalized canonical form (§4.2)
  display_term      TEXT,                 -- original surface form, for UI
  aliases_json      TEXT,                 -- surface forms that mapped here
  kind              TEXT NOT NULL,        -- direct_column | derived_metric | entity | filter

  -- Resolution payload — machine-readable so the VALIDATOR can use it,
  -- not just the prompt. This is the key difference from glossary.sql_hint.
  bindings_json     TEXT NOT NULL,        -- [{"table":"...","column":"..."}]
  aggregation       TEXT,                 -- "COUNT(DISTINCT …)"
  grain             TEXT,                 -- "per employee_id"
  filter_json       TEXT,                 -- [{"column","op","value","value_source"}]
  time_window_json  TEXT,                 -- {"column","span"}
  sql_template      TEXT,                 -- optional compiled snippet

  -- Provenance & governance (§4.5)
  rationale         TEXT,
  confidence        REAL,
  provenance        TEXT NOT NULL,        -- llm_dissector | execution_verified | human | bootstrap
  probe_ok          INTEGER DEFAULT 0,
  approved          INTEGER DEFAULT 0,
  hit_count         INTEGER DEFAULT 0,
  fail_count        INTEGER DEFAULT 0,
  schema_fingerprint TEXT,                -- invalidation trigger (§4.6)
  created_at        REAL,
  verified_at       REAL,
  UNIQUE(source_id, term)
);
```

**`bindings_json` is the crux.** `glossary_terms.sql_hint` is a free-text blob —
usable in a prompt, useless to a validator. Structured bindings can be enforced
mechanically (§6.4), which is what makes the resolution binding rather than
advisory.

Scoping by `source_id` is required: "promotion" means something different in an
HR source than in a retail-promotions source.

### 4.2 Term normalization

The single largest correctness risk is key drift — "top performers" and "best
employees" landing on different keys, producing a miss and a fresh (possibly
different) derivation. Normalization pipeline:

1. lowercase, strip, collapse internal whitespace
2. strip punctuation
3. stem tokens using the existing `verified_queries._stem` (handles
   `ies→y`, `es`, trailing `s`)
4. drop a small stopword set (`the`, `of`, `for`, `in`, `a`, `an`)
5. rejoin with single spaces

So `"Top Performers"`, `"top performer"`, `"the top performers"` → `top perform`.

Semantically distinct phrasings ("best employees") will **not** normalize
together — those are caught by the embedding tier (§4.3) and then recorded as an
**alias** on the matched entry, so the second phrasing becomes an exact hit
thereafter. The lexicon learns its own synonyms.

### 4.3 Lookup ladder

Ordered, first hit wins:

1. **Exact** — `(source_id, normalized_term)`
2. **Alias** — normalized term appears in any entry's `aliases_json`
3. **Embedding** — rank all entries for `source_id` by similarity to the term,
   reusing `verified_queries._rank_by_embedding_similarity` with its keyword
   fallback. Accept above `lexicon_min_similarity` (default `0.62` — higher than
   `verified_queries`' `0.35`, because a wrong binding here is worse than a
   missed few-shot example). On accept, **append the surface form as an alias**.
4. **Existing glossary/KPI** — if `glossary_terms` or `active_kpis` (already in
   state at `state.py:112,116`) define the term, that wins over any
   LLM-derived entry. Human definitions outrank inferred ones.
5. Miss → Evaluation Loop.

### 4.4 Bootstrap — seeding without hand-authoring

The lexicon must not start empty. Four sources, none requiring new human work:

| Source | Yields | Provenance |
|---|---|---|
| `glossary_terms` + `kpis` tables | `kind='derived_metric'`, definition + formula | `human` |
| `_annotate_column_concepts()` output (`ontology_agent/nodes/build_node.py:374`) — the `≈ concept` labels | `kind='direct_column'`, 1:1 bindings | `bootstrap` |
| `semantic_role` / `statistical_type` taxonomy (`metadata_catalog.py`) | `kind='direct_column'` for role-typed columns | `bootstrap` |
| **`verified_queries` mining** — parse SQL that already worked, extract its table/column references | `kind='derived_metric'`, real bindings | `execution_verified` |

The fourth is the highest-value: those derivations already ran successfully
against real data.

### 4.5 Governance

An LLM-derived entry is **not** authoritative. Lifecycle:

```
llm_dissector (approved=0, probe_ok=1)
      │
      ├── human review ──► approved=1
      │
      └── N successful executions ──► provenance=execution_verified
```

- `approved=0` entries are usable but **must** have their derivation echoed in
  the answer (§12.1) so a human can challenge it.
- `provenance='human'` always outranks `llm_dissector` on lookup.
- Repeated downstream failure increments `fail_count`; past a threshold the
  entry is demoted and re-dissected.
- This mirrors the `approved` flag convention `glossary_terms` already
  establishes.

### 4.6 Invalidation

`schema_fingerprint` stores a hash of the bound tables' column sets at write
time. On lookup, if the current fingerprint differs, the entry is stale —
columns may have been renamed or dropped — so it is skipped and re-dissected.
This prevents a binding surviving a schema migration silently.

### 4.7 Relationship to existing stores

Deliberately **not** merged, to avoid breaking working code:

- `glossary_terms` / `kpis` — human-authored, source-agnostic, drive UI. Read as
  a **higher-precedence input** (§4.3 tier 4). Untouched.
- `verified_queries` — whole-question → whole-SQL few-shots. Different grain
  (question-level, not concept-level). Mined for bootstrap (§4.4). Untouched.
- Semantic Lexicon — concept-level, source-scoped, machine-readable bindings.

See §14 Q1 on eventual unification.

---

## 5. Component B — Data-Driven Evaluation Loop

Runs only on lexicon miss.

### 5.1 Step 1 — Identify candidate concepts

One constrained LLM call: given the question plus the **column inventory** for
the retrieved subgraph, list concepts the question requires that are **not**
directly available as a column.

Cached by `(source_id, normalized_question)` so a repeated question costs zero
LLM calls. Fully lexicon-hit questions therefore cost one indexed lookup.

### 5.2 Step 2 — Assemble evidence

Read-only, from data already in state (§2.1). Per candidate table:

- column names, `data_type`, `semantic_role`, `statistical_type`
- `unique_count`, `null_count`, `min_value`, `max_value`
- **literal values** from `state["categorical_columns"]` — capped (e.g. 20/column)
- `functional_dependencies` and `fk_candidates` — how tables actually join
- `cardinality_relationships` — grain, needed to avoid fan-out

No DB round-trip.

### 5.3 Step 3 — Data Dissector LLM

Its only job: map intent onto **actual** columns. Output schema — deliberately
richer than a flat property list, because a flat list leaves the generator to
invent the aggregation, grain, filter literal, and window, which is exactly where
run-to-run variance re-enters:

```json
{
  "term": "promotion count",
  "resolvable": true,
  "rationale": "Employment events are one row per event in job_history; a
                promotion is identified by the event-type value observed in
                that column's profiled sample values.",
  "bindings": [
    {"table": "job_history", "column": "<event-type column>"},
    {"table": "job_history", "column": "<event-date column>"},
    {"table": "job_history", "column": "<employee key column>"}
  ],
  "aggregation": "COUNT(DISTINCT job_history.id)",
  "grain": "per <employee key column>",
  "filter_predicates": [
    {"column": "<event-type column>", "op": "=",
     "value": "<literal from top_values>", "value_source": "top_values"}
  ],
  "time_window": {"column": "<event-date column>", "span": "2 years"},
  "confidence": 0.72
}
```

Two hard constraints, both enforced mechanically in §5.4 rather than trusted:

- every `bindings[].column` must be one supplied in Step 2
- every `filter_predicates[].value` must appear in that column's profiled
  `top_values` — **the dissector may not invent values any more than columns**

### 5.4 Step 4 — Mechanical validation

Not LLM-judged:

1. Every `{table, column}` resolves via `SchemaGraph.has_column()` (§2.4)
2. Every filter literal is present in that column's profiled values
3. `grain` column exists and is plausibly an entity key (PK, FK candidate, or
   high `unique_count`)
4. Any join implied by multi-table bindings appears in `fk_candidates` or
   `cardinality_relationships`

### 5.5 Step 5 — Probe execution

Compile the **minimum** statement proving the bindings are selectable:

```sql
SELECT <bound cols> FROM <qualified table> [WHERE <validated filters>] LIMIT 1
```

Run via `execute_node._run_sql` (§2.6). A proposal that will not execute never
reaches the generator.

**Deliberate limitation:** this probes *column selectability and filter
validity*, not the full aggregate expression. Compiling a dialect-correct
aggregate + window is fragile across Snowflake/Postgres/Oracle/BigQuery; the
generator remains responsible for that, guided by the binding and still gated by
the AST check. A cheap, robust, dialect-agnostic probe that always runs is worth
more than an ambitious one that breaks on dialect edge cases.

### 5.6 Step 6 — Write back

Persist with `provenance='llm_dissector'`, `probe_ok=1`, `approved=0`, current
`schema_fingerprint`. **The next occurrence is a deterministic lexicon hit** —
this is the step that converts a flaky question into a stable one.

### 5.7 The AST gate is not made redundant

Constraining the *dissector's* output does not constrain the *generator*. Step 3
of generation is still a free-form LLM call, and it legitimately needs columns
beyond `bindings` — entity names to return, join keys, date columns. The moment
the wider schema is in the prompt, the hallucination surface returns.

Illustration of why "restrict the whitelist and hallucination becomes
impossible" does not hold: a generated query will reference something like
`RETURN e.name` — an identifier never in `required_properties`. Only a
post-generation AST check against the real schema makes a fake identifier
unreachable. **Therefore `_validate_plan_items` stays as the enforcement
boundary.** The loop lowers hallucination *probability*; the AST gate enforces
*impossibility*.

---

## 6. Integration with existing architecture

### 6.1 Graph topology

Current chain (`dialog_agent/agent.py:57-72`) is linear:

```
START → retrieve → document_context → understand → resolve → plan → execute → synthesize → END
```

Becomes:

```
START → retrieve → document_context → understand → resolve → dissect → plan → execute → synthesize → END
                                                              ▲▲▲▲▲▲▲
```

`dissect` **must** run after `understand` (needs `schema_context`,
`categorical_columns`, `glossary_terms`, `active_kpis` populated) and before
`plan`. Placing it after `resolve` also lets it reuse categorical resolution.

Change: one `add_node`, one edge split into two.

### 6.2 New state keys

```python
derived_metrics:   List[Dict[str, Any]]  # resolved: term + bindings + aggregation + grain + provenance
unresolved_terms:  List[str]             # identified but unresolvable → plan proceeds as today
lexicon_diagnostics: List[str]           # shadow-mode observations, surfaced like state["errors"]
```

### 6.3 New config flags — all default to current behaviour

```python
lexicon_enabled:        bool  = False   # master switch; False = today's behaviour exactly
lexicon_shadow_mode:    bool  = True    # log only, inject nothing
lexicon_min_similarity: float = 0.62
dissect_enabled:        bool  = False   # loop on cache miss
dissect_llm_model:      str   = "claude-haiku-4-5"
dissect_probe_enabled:  bool  = True    # probe is a safety gate; off only for debugging
dissect_max_terms:      int   = 4       # cost ceiling per request
```

Rollout mirrors the pattern the codebase already uses for the AST resolver
(`PLAN_NODE_AST_HALLUCINATION_SHADOW` / `_ENABLED`).

### 6.4 `plan_node` injection — two places

**(a) Prompt.** A `RESOLVED BUSINESS CONCEPTS` block stating, per term, the
mandatory bindings, aggregation, grain, filter literal, and window.

**(b) Validator.** `_validate_plan_items` gains a check: if a resolved term's
concept is being answered, the query must reference its bound columns. A query
ignoring them is rejected with a reason appended to `reasons` — feeding the
retry that `0ea5a91` repaired.

(b) is what makes bindings binding rather than advisory. It is also the only
change that can newly reject a previously-working query, so it ships last (§9).

### 6.5 Failure isolation

Every lexicon/dissector operation is best-effort. Any exception → log, leave
state untouched, planning proceeds exactly as today. This subsystem must never
be able to break query planning — the same discipline `verified_queries.get_similar`
already follows ("returns [] on any failure so a broken embedding backend never
blocks query planning").

---

## 7. Change surface

### 7.1 New files — additive

| File | Contents | Est. LOC |
|---|---|---|
| `dialog_agent/semantic_lexicon.py` | Table DDL, `LexiconEntry`, CRUD, normalization, lookup ladder, bootstrap, fingerprinting. Mirrors `verified_queries.py`. | ~380 |
| `dialog_agent/nodes/dissect_node.py` | Identify, evidence assembly, dissector call, validation, probe, write-back | ~420 |
| `tests/test_semantic_lexicon.py` | Normalization, alias learning, lookup precedence, fingerprint invalidation | ~200 |
| `tests/test_dissect_node.py` | Binding rejection, literal rejection, probe failure, hit/miss, graceful degradation | ~260 |

### 7.2 Modified files

| File | Change | Est. LOC | Risk |
|---|---|---|---|
| `dialog_agent/agent.py` | `+1 add_node`; split `resolve → plan` | ~3 | Very low |
| `dialog_agent/state.py` | `+3` keys (§6.2) | ~3 | Very low |
| `dialog_agent/config.py` | `+7` flags (§6.3) | ~8 | Very low |
| `dialog_agent/nodes/plan_node.py` | Prompt block + validator check (§6.4) | ~60 | ⚠️ **Highest** |

### 7.3 Explicitly unchanged

- **`kg_optimizer/`** — the GA tunes bridge-inference / KG-build genes
  (`min_confidence`, `auto_enable_threshold`, `embedding_sim_threshold`,
  `graphrag_top_k`, …). **No gene touches the SQL planner**, so the GA can
  neither cause nor fix this defect. Once this ships, new genes *could* be added
  (`lexicon_min_similarity`, `dissect_enabled`, evidence sample size) and tuned.
- `knowledge_graph_agent/`, `ontology_agent/`, `conformity_agent/`,
  `shacl_agent/`
- **All existing DB tables**, including `glossary_terms` and `verified_queries`
- All frontends
- `sql_identifier_resolver.py` — consumed as-is
- `execute_node.py` — `_run_sql` consumed as-is

### 7.4 Note on `plan_node.py`

The only risky edit. ~5k lines, currently carrying a large uncommitted
AST-resolver rewrite plus the `0ea5a91` retry fix. Both insertion points are
well-defined, but this warrants an isolated commit and isolated review.

---

## 8. Worked example — `HRData`

*"Who is eligible for next promotion?"*

**Today (failing path).** Planner receives the full 5-table schema, emits SQL
referencing `promotion_count_2yr`, `education_count`, `certificate_count`. All
rejected as hallucinated. Both queries dropped. Pre-`0ea5a91` the retry did not
even fire. User sees "No query results were produced."

**With this design, first ever run.**

1. *Identify* → `["promotion count", "education count", "certificate count"]`
2. *Lexicon lookup* → 3 misses (cold lexicon)
3. *Evidence* → for `job_history`: column names, types, `semantic_role`s,
   `unique_count`s, profiled literal values, and the `fk_candidates` /
   `cardinality_relationships` describing how it joins to `job_current`
4. *Dissector* → proposes, **for each term**, which real columns carry the
   signal: the event-type column, its date column, the employee key, the literal
   event value drawn from profiled values, and the grain
5. *Mechanical validation* → every proposed column checked via
   `SchemaGraph.has_column()`; every literal checked against profiled values.
   Anything invented is rejected here, not trusted.
6. *Probe* → `SELECT <cols> FROM <table> WHERE <filter> LIMIT 1`
7. *Write back* → 3 entries, `approved=0`, `probe_ok=1`
8. *Plan* → prompt carries mandatory bindings; validator enforces them; AST gate
   still guards output
9. *Answer* → includes the derivation, e.g. *"promotion count = distinct
   employment events of type `<literal>` in `job_history` within 2 years, per
   employee"*, so the definition is visible and challengeable

**Every subsequent run.** Steps 1 (cached) and 2 hit the lexicon. Same bindings,
same SQL shape, **same answer** — no LLM freedom to re-derive. That is the fix
for the inconsistency.

**Note on the example.** I have deliberately not asserted specific column names
such as `job_history.employee_id` or `job_history.event`. Inspection of the
`HRData` snapshot confirmed `event_date` on `job_history` and `event` +
`employee_id` on `job_current`, but **not** that `job_history` carries an
`event`-type or `employee_id` column. Discovering which columns actually carry
the signal — and the join path between the two tables — is precisely the loop's
job, and is exactly the assumption a hand-written mapping would get wrong.

---

## 9. Phased rollout

Each phase independently shippable and flag-reversible.

**§9.0 — Prerequisites (before writing any of this).**

1. **Baseline measurement.** Run `run_hr_questions.py` 5× and record hard-failure
   rate and answer-set drift. Without it, "did it work" is unanswerable. It also
   reveals how much `0ea5a91` already recovered on its own.
2. **Test `PLAN_NODE_AST_HALLUCINATION_ENABLED=1`.** Targets Class A — the larger
   bucket (§1.3) — is already written, already shadow-validated on 363
   statements, and costs one environment variable. **Likely higher yield than
   this entire design.** Do it first.

**Phase 1 — Lexicon store + bootstrap, read-only.** Ship
`semantic_lexicon.py`, seed from the four sources in §4.4. Nothing reads it in
the pipeline. Zero behaviour change. Inspect what bootstrap produced.

**Phase 2 — `dissect_node` in shadow.** Node runs, logs what it *would* resolve,
injects nothing (`lexicon_shadow_mode=True`). Review logged derivations against
the real schema. This is where a bad dissector is caught, before it can affect an
answer.

**Phase 3 — Probe + write-back, still no injection.** Persist validated
proposals. Verifies stability: same question → same lexicon hit, run over run.

**Phase 4 — Prompt injection** (§6.4a). Bindings become planner instructions.
AST gate unchanged. Re-run the harness against the Phase 0 baseline.

**Phase 5 — Validator enforcement** (§6.4b). **Last**, because it is the only
step that can newly reject queries that previously worked.

---

## 10. Expected accuracy

**Estimates, not measurements.** No baseline has been run (§9.0). Replace with
measured numbers before relying on them.

| Metric | Estimate | Confidence | Reasoning |
|---|---|---|---|
| Class B hard failures eliminated | 85–95% | High | Removes the guess entirely; probe rejects bad proposals pre-generation |
| **Run-to-run stability** for lexicon-hit terms | >95% | High | Deterministic after first resolution; residual risk is key drift (§12.2), mitigated by alias learning (§4.2) |
| Class B **semantic correctness** (matches what a human analyst would compute) | 50–70% | **Low** | The dissector infers intent from names + sampled values. It cannot know company policy. See §12.1 |
| Overall "No query results" reduction, all sources | 20–40% | Medium | Bounded by Class B's share; Class A dominated observed volume |
| Added latency, lexicon hit | ~0 | High | One indexed lookup |
| Added latency, cold miss | +3–8 s | Medium | Two Haiku calls + one probe; observed turns already ~40 s |

Honest summary: **this closes the mechanical failure decisively and the semantic
one only partially.** It converts "sometimes errors out" into "consistently
answers using an inferred definition" — an improvement *only if* that definition
is surfaced and reviewable (§12.1).

---

## 11. Out of scope

- **Class A** identifier casing/naming (§1.3) — different mechanism, see §9.0
- **GA / `kg_optimizer`** — cannot address this (§7.3)
- Retrieval tuning — not the cause for `HRData` (§1.4)
- Multi-hop derivations spanning 3+ tables with windowing — falls through to
  today's behaviour (§12.4)
- Merging `glossary_terms` / `verified_queries` into the lexicon (§14 Q1)

---

## 12. Risks

### 12.1 A wrong cached derivation is worse than a loud failure

**Highest severity.** A plausible-but-wrong derivation, once persisted, returns
confidently wrong numbers indefinitely — no error, no signal. People act on
promotion-eligibility output.

Mitigations, all required, not optional:

- `approved=0` + `provenance='llm_dissector'` until human-reviewed or
  execution-verified (§4.5)
- Mandatory probe (§5.5)
- **Echo the derivation in the answer** so it is visible and challengeable
- Human-authored glossary/KPI definitions always outrank inferred ones (§4.3)
- Review surface for unapproved entries

### 12.2 Term-key drift

Two phrasings landing on different keys → miss → fresh dissection → possibly
different answer, reintroducing the inconsistency this exists to remove.
Mitigations: normalization (§4.2), embedding tier, **alias learning** on
embedding hits, and logging every near-miss below threshold for review.

### 12.3 Thresholds are a hidden business decision

Silently defining "top performer" as "above average" embeds a judgement in a
prompt. Whatever rule is chosen must appear in the response (§12.1).

### 12.4 Multi-hop derivations

May exceed what one dissector call can specify reliably. Mitigation: on
`resolvable=false` or validation failure, fall through to today's behaviour —
never a regression.

### 12.5 Stale profiled values

`top_values` may lag the live DB, so a validated literal could no longer exist.
Mitigations: `schema_fingerprint` invalidation (§4.6) and the probe as safety net
(§5.5).

### 12.6 `plan_node.py` blast radius

See §7.4.

---

## 13. Measurement plan

Harness: `run_hr_questions.py` (existing, Class B), plus a LifeScience_V2 set for
class separation.

Record per phase, 5 runs each:

1. **Hard-failure rate** — share returning "No query results"
2. **Answer-set drift** — same question → same rows across runs. *This is the
   number that speaks to the original complaint.*
3. **Failure class split** — A vs B, parsed from `plan_node` rejection lines.
   Confirms or refutes §1.3's claim that A dominates.
4. **Lexicon hit rate** and **dissector acceptance rate** (share passing
   mechanical + probe validation)
5. **p50/p95 turn latency**
6. **Review precision** — of unapproved entries reviewed by a human, what share
   were correct? Directly measures §12.1, the risk that matters most.

Ship criterion: Class B hard failures materially down **and** answer-set drift
materially down, with no increase in Class A failures and no wrong-but-plausible
answers found during review.

---

## 14. Open questions

1. Once human-approved, should lexicon entries be promoted into `glossary_terms`,
   unifying the two mechanisms rather than maintaining both indefinitely?
2. Should the dissector run when a *direct* column match exists but scores poorly
   — or strictly on total absence?
3. Should `approved=0` entries be usable at all, or only logged until signed off?
   (Stricter = safer, but delays all benefit.)
4. Should `fail_count`-driven demotion re-dissect automatically, or quarantine the
   entry for human attention?
5. Should the lexicon be exposed in the UI (`tech_ui` / `chat_frontend`) for
   review and approval, or managed via CLI initially?

---

## Appendix A — Exact insertion points

> **Line numbers are against the current working tree**, which contains the
> uncommitted AST-resolver rewrite in `plan_node.py` plus the `0ea5a91` retry
> fix. If that rewrite is committed, reverted, or reworked, every `plan_node.py`
> anchor below shifts. **Anchor on the quoted code, not the number.** All other
> files are clean, so their numbers are stable.
>
> Verified against the working tree on 2026-07-30. `+` = added, `~` = replaced.

### A.1 `dialog_agent/agent.py` — 5 anchors

**A.1.1 — Docstring pipeline listing (lines 4–14)**

```diff
       → resolve           (LLM maps user query terms → exact categorical DB values)
+      → dissect           (resolve business concepts → real columns via lexicon +
+                            data-driven evaluation loop; no-op when disabled)
       → plan              (LLM decomposes NQL → SQL queries using resolved values)
```

**A.1.2 — Import block (lines 25–33)**

```diff
 from .nodes import (
     retrieve_node,
     document_context_node,
+    dissect_node,
     execute_node,
     plan_node,
     resolve_node,
```

**A.1.3 — `_NODES` list (line 38)** — easy to miss; drives node-level timing/progress.

```diff
-_NODES = ["retrieve", "document_context", "understand", "resolve", "plan", "execute", "synthesize"]
+_NODES = ["retrieve", "document_context", "understand", "resolve", "dissect", "plan", "execute", "synthesize"]
```

**A.1.4 — Node registration (insert between lines 60 and 61)**

```diff
     g.add_node("resolve",           _timed_node("resolve", resolve_node))
+    g.add_node("dissect",           _timed_node("dissect", dissect_node))
     g.add_node("plan",              _timed_node("plan", plan_node))
```

**A.1.5 — Edge rewire (line 69)** — the only *replacement* in this file.

```diff
-    g.add_edge("resolve",           "plan")
+    g.add_edge("resolve",           "dissect")
+    g.add_edge("dissect",           "plan")
```

Total: 4 additions, 2 replacements.

### A.2 `dialog_agent/nodes/__init__.py` — 2 anchors

```diff
 from .resolve_node import resolve_node
+from .dissect_node import dissect_node
 from .plan_node import plan_node
@@
-__all__ = ["retrieve_node", "document_context_node", "understand_node", "resolve_node", "plan_node",
-           "execute_node", "synthesize_node"]
+__all__ = ["retrieve_node", "document_context_node", "understand_node", "resolve_node",
+           "dissect_node", "plan_node", "execute_node", "synthesize_node"]
```

⚠️ Import order matters: `dissect_node` imports `dialog_agent.semantic_lexicon`,
which imports `pg_store`. Keep it after `resolve_node` and before `plan_node` to
match execution order and avoid confusion; there is no circular-import risk
because `semantic_lexicon` imports nothing from `nodes`.

### A.3 `dialog_agent/state.py` — 1 anchor

Append inside `DialogState`, after `multi_kg_configs` (line 121, the final field):

```diff
     multi_kg_configs: List[Any]        # one DialogConfig per active KG (for execute_node routing)
+
+    # ── Semantic Lexicon / Evaluation Loop (dissect_node) ─────────────────
+    # Business concepts resolved to real columns, either from the lexicon
+    # (cache hit) or the evaluation loop (cache miss). Consumed by plan_node
+    # for both prompt injection and binding enforcement. Empty = behave
+    # exactly as before this feature existed.
+    # [{"term": "promotion count", "kind": "derived_metric",
+    #   "bindings": [{"table": "...", "column": "..."}],
+    #   "aggregation": "...", "grain": "...", "filter_predicates": [...],
+    #   "time_window": {...}, "provenance": "...", "approved": 0}]
+    derived_metrics: List[Dict[str, Any]]
+
+    # Concepts identified in the question that could NOT be resolved. Planning
+    # proceeds exactly as today for these — presence here is never fatal.
+    unresolved_terms: List[str]
+
+    # Shadow-mode observations (what WOULD have been injected). Surfaced the
+    # same way state["errors"] is, so it is visible without changing behaviour.
+    lexicon_diagnostics: List[str]
```

No existing key is modified. `Dict`, `List`, `Any` are already imported (line 6).

### A.4 `dialog_agent/config.py` — 1 anchor

Append to `DialogConfig` after `kg_router_threshold` (line 103, the final field):

```diff
     kg_router_threshold: float = 0.30      # min cosine similarity for routing
+
+    # ── Semantic Lexicon + Data-Driven Evaluation Loop ────────────────────────
+    # ALL DEFAULTS PRESERVE CURRENT BEHAVIOUR. With lexicon_enabled=False the
+    # dissect node returns state untouched, so the pipeline is byte-identical
+    # to its behaviour before this feature.
+    lexicon_enabled:        bool  = False   # master switch (§9 Phase 2)
+    lexicon_shadow_mode:    bool  = True    # log only, inject nothing (§9 Phase 2)
+    # Higher than verified_queries' 0.35 on purpose: a wrong binding here is
+    # worse than a missed few-shot example.
+    lexicon_min_similarity: float = 0.62
+    dissect_enabled:        bool  = False   # run the loop on lexicon miss (§9 Phase 3)
+    dissect_llm_model:      str   = "claude-haiku-4-5"
+    dissect_probe_enabled:  bool  = True    # safety gate; disable only to debug
+    dissect_max_terms:      int   = 4       # per-request cost ceiling
+    # Cap the injected prompt block so it cannot crowd out schema_context under
+    # the token guard — see A.5.4.
+    lexicon_section_max_chars: int = 4000
```

### A.5 `dialog_agent/nodes/plan_node.py` — 4 anchors ⚠️ highest risk

**A.5.1 — `_USER_PROMPT` template (line 1572)**

The section placeholders are concatenated on one line. Insert
`{lexicon_section}` **before** `{glossary_section}` so human-authored glossary
text appears after (and therefore closer to the question than) machine-derived
bindings:

```diff
-{history_section}{multi_kg_section}{glossary_section}{kpi_section}{resolution_section}{verified_section}NATURAL LANGUAGE QUESTION:
+{history_section}{multi_kg_section}{lexicon_section}{glossary_section}{kpi_section}{resolution_section}{verified_section}NATURAL LANGUAGE QUESTION:
```

**A.5.2 — Build the section (insert after line 4508)**

The `verified_section` if/else ends at line 4508 with `verified_section = ""`.
Insert immediately after, mirroring the existing `ve_lines` construction style:

```python
    # ── RESOLVED BUSINESS CONCEPTS (dissect_node) ─────────────────────────
    derived_metrics = state.get("derived_metrics") or []
    if derived_metrics:
        dm_lines = [
            "RESOLVED BUSINESS CONCEPTS — MANDATORY COLUMN BINDINGS",
            "=" * 60,
            "Each concept below has been resolved to REAL columns and validated",
            "against this schema (every column confirmed present; every filter",
            "literal confirmed present in that column's actual data). You MUST",
            "express these concepts using the bindings given. Do NOT reference a",
            "pre-aggregated column for them — none exists.",
            "",
        ]
        for dm in derived_metrics:
            cols = ", ".join(
                f"{b['table']}.{b['column']}" for b in (dm.get("bindings") or [])
            )
            dm_lines.append(f"  Concept: {dm.get('term','')}")
            dm_lines.append(f"    Columns:     {cols}")
            if dm.get("aggregation"):
                dm_lines.append(f"    Aggregation: {dm['aggregation']}")
            if dm.get("grain"):
                dm_lines.append(f"    Grain:       {dm['grain']}")
            for f in (dm.get("filter_predicates") or []):
                dm_lines.append(
                    f"    Filter:      {f['column']} {f['op']} {f['value']!r}"
                )
            if dm.get("time_window"):
                tw = dm["time_window"]
                dm_lines.append(
                    f"    Window:      {tw.get('column')} within {tw.get('span')}"
                )
            dm_lines.append("")
        dm_lines.append("=" * 60)
        lexicon_section = "\n".join(dm_lines) + "\n\n"
        max_chars = getattr(config, "lexicon_section_max_chars", 4000)
        if len(lexicon_section) > max_chars:
            lexicon_section = lexicon_section[:max_chars] + "\n…(truncated)\n\n"
    else:
        lexicon_section = ""
```

**A.5.3 — `_USER_PROMPT.format(...)` call (lines 4510–4521)**

```diff
     user = _USER_PROMPT.format(
         schema_context=schema_context,
         db_type=config.db_type,
         schema_line=schema_line,
         history_section=history_section,
         multi_kg_section=multi_kg_section,
+        lexicon_section=lexicon_section,
         glossary_section=glossary_section,
```

⚠️ **`str.format` fails loudly on a missing key** — A.5.1 and A.5.3 must land in
the same commit, or every plan call raises `KeyError: 'lexicon_section'`. This is
the single most breakable coupling in the change.

**A.5.4 — Token-guard interaction (line 4524, no edit, but read this)**

```python
system, user = guard_plan_prompt(system, user, schema_context, model=config.plan_llm_model)
```

`guard_plan_prompt` trims **`schema_context`**, not the new section. So a large
`lexicon_section` silently costs schema detail rather than being trimmed itself —
which could *cause* the Class A hallucinations this design does not target. Hence
the `lexicon_section_max_chars` cap in A.5.2/A.4. No change to the guard itself.

**A.5.5 — Binding enforcement inside `_validate_plan_items` (insert before line 4947)**

**Key finding: no signature change is required.** `_validate_plan_items` is a
*closure* defined at line 4603 **inside** `plan_node` (note its 4-space
indentation), so it already reads `state`, `config`, `known_columns`,
`table_columns_map`, and `reasons` from the enclosing scope. Passing
`derived_metrics` as a parameter is unnecessary.

Insert immediately before the terminal `valid.append(SQLQuery(...))` at line 4947,
after all existing checks pass:

```python
            # ── Binding enforcement (Semantic Lexicon) ────────────────────
            # A resolved concept's bindings are mandatory, not advisory. If the
            # query answers a resolved concept but references none of its bound
            # columns, the planner ignored the resolution — reject and let the
            # retry (see the `retry_reasons` path) correct it.
            _dms = state.get("derived_metrics") or []
            if _dms and getattr(config, "lexicon_enforce_bindings", False):
                sql_l = sql.lower()
                for _dm in _dms:
                    _bound = [b["column"].lower() for b in (_dm.get("bindings") or [])]
                    if not _bound:
                        continue
                    if not any(c in sql_l for c in _bound):
                        logger.warning(
                            "plan_node: dropping query %s — ignores mandatory "
                            "bindings for resolved concept %r (expected one of %s)",
                            item.get("query_id", "?"), _dm.get("term"), _bound,
                        )
                        state["errors"].append(
                            f"plan_node: query {item.get('query_id','?')} skipped — "
                            f"ignored resolved-concept bindings for {_dm.get('term')!r}"
                        )
                        reasons.append(
                            f"Query {item.get('query_id','?')} must express the concept "
                            f"{_dm.get('term')!r} using its resolved columns "
                            f"{_bound} — no pre-aggregated column for it exists."
                        )
                        break
                else:
                    valid.append(SQLQuery(...))   # existing call, unchanged
                    continue
                continue
```

Two notes on this block:

- It appends to **`reasons`**, not only `state["errors"]` — deliberately, so the
  retry repaired in `0ea5a91` actually fires. Appending only to `errors` was the
  original bug (§2.7); repeating it here would recreate it.
- The `for/else` + `continue` control flow above is illustrative. In practice
  prefer a small helper returning `(ok, reason)` and keep the existing
  `valid.append(...)` untouched — cleaner than restructuring a loop this deep
  inside a 5k-line function.
- Gate on a **separate** flag `lexicon_enforce_bindings` (add to A.4, default
  `False`) rather than `lexicon_enabled`, so Phase 4 (prompt injection) can ship
  without Phase 5 (enforcement). Enforcement is the only step that can newly
  reject a previously-working query.

### A.6 New file skeletons

**`dialog_agent/semantic_lexicon.py`** — mirrors `verified_queries.py` (§2.3).

```python
_DDL_PG     = """CREATE TABLE IF NOT EXISTS semantic_lexicon (...)"""   # §4.1
_DDL_SQLITE = """CREATE TABLE IF NOT EXISTS semantic_lexicon (...)"""
def _ensure(cur) -> None: ...          # cur.ddl(_DDL_PG if pg_store.is_postgres() else _DDL_SQLITE)

@dataclass
class LexiconEntry:
    source_id: str; term: str; kind: str
    bindings: List[Dict[str, str]]
    display_term: str = ""; aliases: List[str] = field(default_factory=list)
    aggregation: str = ""; grain: str = ""
    filter_predicates: List[Dict] = field(default_factory=list)
    time_window: Optional[Dict] = None
    sql_template: str = ""; rationale: str = ""; confidence: float = 0.0
    provenance: str = "llm_dissector"
    probe_ok: bool = False; approved: bool = False
    hit_count: int = 0; fail_count: int = 0
    schema_fingerprint: str = ""
    entry_id: Optional[str] = None

def normalize_term(text: str) -> str: ...                    # §4.2 — reuses verified_queries._stem
def schema_fingerprint(table_columns_map: Dict[str, set], tables: List[str]) -> str: ...   # §4.6
def save(entry: LexiconEntry) -> str: ...                    # upsert on (source_id, term)
def lookup(source_id: str, term: str, *, min_similarity: float = 0.62,
           glossary_terms: Optional[List[Dict]] = None,
           active_kpis: Optional[List[Dict]] = None,
           current_fingerprint: str = "") -> Optional[LexiconEntry]: ...   # §4.3 ladder
def add_alias(entry_id: str, surface_form: str) -> None: ...  # §4.2 alias learning
def bump_hit(entry_id: str) -> None: ...
def bump_fail(entry_id: str) -> None: ...                     # §4.5 demotion
def approve(entry_id: str) -> None: ...
def list_all(source_id: Optional[str] = None) -> List[LexiconEntry]: ...
def delete(entry_id: str) -> None: ...
def bootstrap(source_id: str, *, glossary_terms=None, kpis=None,
              concept_annotations=None, mine_verified_queries: bool = True) -> int: ...  # §4.4
```

Reuse `verified_queries._rank_by_embedding_similarity` and
`_rank_by_keyword_similarity` (`verified_queries.py:211,201`) rather than
duplicating ~50 lines. They are module-private but same-package; import with a
comment explaining the intentional reuse.

**`dialog_agent/nodes/dissect_node.py`**

```python
def dissect_node(state: DialogState) -> DialogState: ...     # §5, entry point

def _identify_concepts(natural_query: str, table_columns_map: Dict[str, set],
                       config) -> List[str]: ...             # §5.1, cached per question
def _assemble_evidence(state: DialogState, tables: List[str]) -> Dict[str, Any]: ...  # §5.2
def _dissect(term: str, evidence: Dict[str, Any], config) -> Optional[Dict]: ...      # §5.3
def _validate_proposal(proposal: Dict, schema: "SchemaGraph",
                       categorical_columns: Dict) -> tuple[bool, str]: ...            # §5.4
def _probe(proposal: Dict, config, state: DialogState) -> tuple[bool, str]: ...       # §5.5
def _quote_ident(name: str, db_type: str) -> str: ...        # dialect quoting for the probe
def _extract_json(raw: str) -> Any: ...                      # local; avoid coupling to plan_node
```

Reuse, do not reimplement:

| Need | Existing |
|---|---|
| `{TABLE: {cols}}` from `schema_context` | `plan_node._extract_table_columns` (`:2341`) |
| Column existence checks | `sql_identifier_resolver.SchemaGraph` (`:30`) |
| Probe execution (never raises) | `execute_node._run_sql` (`:750`) |
| Literal values | `state["categorical_columns"]` (`understand_node.py:1270`) |
| Similarity ranking | `verified_queries` (`:201,211`) |
| Storage backend selection | `pg_store` |

`_extract_table_columns` is module-private in a 5k-line file. Preferred: lift it
into a shared helper. Acceptable interim: import it directly with a comment. Do
**not** write a second schema parser — two parsers drifting on what "known
column" means is exactly the failure `SchemaGraph`'s docstring (`:31-35`) warns
about.

### A.7 Landing order

| # | Change | Independently safe? |
|---|---|---|
| 1 | A.6 `semantic_lexicon.py` + tests | Yes — nothing imports it |
| 2 | A.4 config flags (all default off) | Yes — inert |
| 3 | A.3 state keys | Yes — inert |
| 4 | A.6 `dissect_node.py` + A.2 export | Yes — not in graph yet |
| 5 | A.1 graph wiring | Yes — node no-ops while `lexicon_enabled=False` |
| 6 | **A.5.1 + A.5.3 together** | ⚠️ **Must be one commit** — `KeyError` otherwise |
| 7 | A.5.2 section builder | Yes — empty string when no metrics |
| 8 | A.5.5 enforcement | Ship last, own flag, own commit |

Steps 6–8 touch `plan_node.py` and should be reviewed separately from 1–5 (§7.4).
