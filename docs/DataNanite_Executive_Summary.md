# DataNanite — Executive Summary

*One-page overview. For the full stepwise detail, see [`DataNanite_Architecture.md`](./DataNanite_Architecture.md).*

## What it is

DataNanite turns a raw database into a **self-describing, conversational data
product**. Point it at a source and it automatically discovers the schema,
documents what every table and column means, builds a navigable knowledge graph
of how the data connects, and then lets business users **ask questions in plain
English** and get back verified SQL-backed answers with charts and tables.

## How it works (the pipeline in plain terms)

1. **Discover & profile** — Connect to the database, profile every table and
   column (row counts, uniqueness, null rates, value patterns), and *infer the
   hidden structure*: primary keys, foreign-key relationships, and how tables
   relate (1:1 / 1:many). This is rigorous and statistical — not guesswork.
2. **Formalize (Ontology)** — Express that structure as a formal **OWL ontology**:
   tables become classes, columns become properties, relationships become typed
   links with cardinality. An AI step adds plain-business labels to cryptic column
   names (e.g. `arpu` → "average revenue per user").
3. **Make it navigable (Knowledge Graph)** — Convert the ontology into a
   **knowledge graph** (persisted as a nodes/edges snapshot) with semantic vector
   embeddings, so the system can find the *right* tables for any question, even
   when the wording doesn't match the column names.
4. **Answer questions (DataChat)** — A natural-language question flows through six
   steps: find relevant tables → describe them to the AI → resolve the user's words
   to real data values → generate SQL → run it (with self-healing on errors) →
   write a narrative answer with charts. Glossary terms, KPI definitions, and even
   linked documents are folded in automatically.

```
Database → Profile & Infer → Ontology → Knowledge Graph → Ask in English → Insight
```

## What makes it trustworthy

- **Structure is computed, not hallucinated.** Keys, relationships, and the
  ontology's shape are derived by deterministic algorithms over real data. The AI
  is used for *language* (labels, planning, narrative) — never to invent facts.
- **Four guardrail layers on every answer.** The system pre-binds real category
  values, validates and fixes the generated SQL, self-heals execution errors, and
  forbids the AI from citing any number that isn't in the query results.
- **Governed business vocabulary.** A Business Glossary and a KPI Registry (with
  approval workflow, versioning, and SQL-safety guardrails) keep definitions
  consistent across every answer.
- **Cost-aware AI.** Cheap, fast models handle structured work (SQL planning,
  matching); a stronger model writes the final user-facing narrative.

## The building blocks

| Capability | What it delivers |
|------------|------------------|
| **Metadata Extraction** | Schema, statistics, inferred keys & relationships |
| **Ontology** | Formal OWL/Turtle model of the data, with business labels |
| **Knowledge Graph** | Navigable, semantically-searchable graph of entities & links |
| **DataChat** | Natural-language Q&A → verified SQL → narrative + charts |
| **Cross-source Bridges** | Links the same entity across different databases |
| **SHACL Validation** | Quality-checks the ontology (PASS / WARN / FAIL) |
| **Unstructured Documents** | Fingerprints reports/docs and links them to KPIs & tables |
| **Business Glossary & KPI Registry** | Governed definitions and metric formulas |

Each capability is an independent service coordinated by an **orchestrator** that
runs the full indexing flow end-to-end and streams live progress to the browser.

## Why it matters

- **Faster time-to-insight** — business users self-serve answers without writing
  SQL or waiting on analysts.
- **Less manual documentation** — the catalog, relationships, and ontology are
  generated automatically and stay in sync with the data.
- **Answers you can defend** — every figure traces back to executed SQL over the
  real source, with governed definitions behind the terms.
- **Works across the stack** — Snowflake, Postgres, Oracle, SQL Server, BigQuery,
  Redshift, Teradata, Delta Lake, and file uploads (CSV/Excel).
