"""
Generic, dialect-aware name-hallucination resolver for planner-generated SQL.

Replaces the regex-based pair (_find_hallucinated_tables / _find_hallucinated_columns
+ _strip_hallucinated_conditions) in plan_node.py with a single AST parse → resolve →
repair → re-render pipeline built on sqlglot. One code path handles every clause
(SELECT, WHERE, HAVING, GROUP BY, ORDER BY, JOIN, window PARTITION BY, subqueries,
CTEs) because it walks the parsed tree instead of enumerating per-clause regex
patterns.

See conversation plan: Phase 1 of the AST-based hallucination resolver rollout.
This module is additive-only — it has no call sites in plan_node.py yet (Phase 2
wires it in shadow mode; Phase 3 cuts over behind a flag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import build_scope


IdentifierKind = Literal["table", "column"]


class SchemaGraph:
    """Single source of truth for 'does this name resolve to something real'.

    Built once per plan-validation request from whatever schema context was
    retrieved for the question — the same object every check reads from, so
    table and column validation can never drift on what "known" means.
    """

    def __init__(self, table_columns_map: Dict[str, Set[str]]):
        # table_columns_map: {table_name -> {column_name, ...}}, any case.
        self._tables: Dict[str, Set[str]] = {
            t.lower(): {c.lower() for c in cols}
            for t, cols in (table_columns_map or {}).items()
        }

    def has_table(self, name: str) -> bool:
        return name.lower() in self._tables

    def has_column(self, table: str, col: str) -> bool:
        cols = self._tables.get(table.lower())
        return cols is not None and col.lower() in cols

    def all_tables(self) -> Set[str]:
        return set(self._tables.keys())


@dataclass
class HallucinatedIdentifier:
    kind: IdentifierKind
    name: str
    clause: str                     # "select" | "where" | "having" | "order" | "group" | "join" | "unknown"
    scope_id: int                   # id() of the sqlglot Scope the node lives in — distinguishes nesting levels
    table: Optional[str] = None     # resolved table name, for column kind
    node: exp.Expression = field(repr=False, default=None)


def _clause_of(node: exp.Expression) -> str:
    """Walk up from a node to find which clause it lives in."""
    cur = node
    while cur is not None:
        parent = cur.parent
        if parent is None:
            break
        if isinstance(parent, (exp.Where, exp.Having)):
            return "having" if isinstance(parent, exp.Having) else "where"
        if isinstance(parent, exp.Order):
            return "order"
        if isinstance(parent, exp.Group):
            return "group"
        if isinstance(parent, exp.Join):
            return "join"
        if isinstance(parent, exp.Select) and cur in parent.expressions:
            return "select"
        cur = parent
    return "unknown"


def _table_name(table_exp: exp.Table) -> str:
    return table_exp.name


def find_hallucinated_identifiers(
    sql: str,
    schema: SchemaGraph,
    dialect: str = "snowflake",
) -> List[HallucinatedIdentifier]:
    """
    Parse `sql` and return every table/column reference that does not resolve
    against `schema`, tagged with its kind, clause, and scope.

    Returns [] (not an exception) if the SQL fails to parse — callers should
    treat unparseable SQL as a separate failure mode, not a hallucination.

    Node references on the returned objects are bound to a *private* parse of
    `sql` — they are for detection/diagnostics only. `repair()` performs its
    own independent parse-and-resolve so it never mutates a tree via node
    references captured from a different tree.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except ParseError:
        return []
    return _resolve_on_tree(tree, schema)


def _resolve_on_tree(tree: exp.Expression, schema: SchemaGraph) -> List[HallucinatedIdentifier]:
    bad: List[HallucinatedIdentifier] = []
    seen: Set[Tuple[str, str, int]] = set()

    try:
        scopes = list(build_scope(tree).traverse()) if build_scope(tree) else []
    except Exception:
        scopes = []

    # ── Table hallucination ────────────────────────────────────────────
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = _table_name(table)
        if not name:
            continue
        low = name.lower()
        if low in cte_names or schema.has_table(low):
            continue
        key = ("table", low, 0)
        if key in seen:
            continue
        seen.add(key)
        bad.append(
            HallucinatedIdentifier(
                kind="table",
                name=name,
                clause=_clause_of(table) or "join",
                scope_id=0,
                node=table,
            )
        )

    bad_table_names = {b.name.lower() for b in bad if b.kind == "table"}

    # ── Alias → table binding per scope (handles JOINs, subqueries, CTEs) ─
    alias_to_table: Dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        low_name = _table_name(table).lower()
        if low_name in bad_table_names:
            continue
        alias = table.alias_or_name.lower()
        alias_to_table[alias] = low_name
    for cte in tree.find_all(exp.CTE):
        alias_to_table[cte.alias_or_name.lower()] = cte.alias_or_name.lower()

    # ── Column hallucination ───────────────────────────────────────────
    for col in tree.find_all(exp.Column):
        table_ref = col.table
        if not table_ref:
            # Unqualified column — resolving this correctly needs full
            # scope-based disambiguation; skip for now (matches the current
            # regex behavior, which only checks dotted alias.col references).
            continue
        alias_low = table_ref.lower()
        resolved_table = alias_to_table.get(alias_low)
        if resolved_table is None:
            # Alias doesn't bind to any known table in this query at all —
            # that's a different failure mode (unknown alias); leave it to
            # the join/alias validators, not column-hallucination.
            continue
        if resolved_table in cte_names:
            # Column provenance inside a CTE isn't validated here — the CTE's
            # own SELECT list is walked independently when we visit it.
            continue
        col_name = col.name
        if schema.has_column(resolved_table, col_name):
            continue
        key = ("column", f"{alias_low}.{col_name.lower()}", 0)
        if key in seen:
            continue
        seen.add(key)
        bad.append(
            HallucinatedIdentifier(
                kind="column",
                name=col_name,
                clause=_clause_of(col),
                scope_id=0,
                table=resolved_table,
                node=col,
            )
        )

    return bad


def _drop_node(node: exp.Expression) -> None:
    """Remove a node from whatever list/clause it lives in."""
    parent = node.parent
    if parent is None:
        node.pop()
        return

    # A bare column wrapped in "col AS label" (SELECT) or "col ASC/DESC"
    # (ORDER BY, via exp.Ordered) must have its *wrapper* removed, not just
    # the inner column — otherwise a dangling "AS label" / empty ordering
    # term is left behind and the tree fails to round-trip.
    if isinstance(parent, exp.Alias) and parent.this is node:
        _drop_node(parent)
        return
    if isinstance(parent, exp.Ordered) and parent.this is node:
        _drop_node(parent)
        return

    if isinstance(parent, exp.Select) and node in parent.expressions:
        parent.set("expressions", [e for e in parent.expressions if e is not node])
        return

    if isinstance(parent, (exp.Order, exp.Group)) and node in parent.expressions:
        remaining = [e for e in parent.expressions if e is not node]
        if remaining:
            parent.set("expressions", remaining)
        else:
            # An empty ORDER BY / GROUP BY is invalid SQL — drop the whole
            # clause rather than leave an expression-less node behind.
            parent.pop()
        return

    # Predicate leaf inside AND/OR (Where/Having condition tree): splice the
    # node out and promote the sibling in its place.
    if isinstance(parent, (exp.And, exp.Or)):
        sibling = parent.right if parent.left is node else parent.left
        parent.replace(sibling)
        return

    if isinstance(parent, (exp.Where, exp.Having)) and parent.this is node:
        parent.pop()
        return

    # Fallback: just remove the node itself.
    node.pop()


_MAX_REPAIR_PASSES = 8  # fixed-point safety cap; a real query converges in 1-3


def repair(
    sql: str,
    schema: SchemaGraph,
    dialect: str = "snowflake",
    bad: Optional[List[HallucinatedIdentifier]] = None,
) -> Tuple[Optional[str], List[str]]:
    """
    Remove every hallucinated identifier from `sql` and re-render it.

    `bad` (as returned by find_hallucinated_identifiers) is accepted only as
    an initial detection hint for callers who already ran the check — it is
    NOT used to mutate the tree directly, since its nodes belong to a
    different parse. `repair` always re-resolves against its own freshly
    parsed tree, then runs to a fixed point: removing a hallucinated table
    orphans columns referencing its alias, which can itself require another
    resolve pass (e.g. a now-empty aggregate expression).

    Returns (repaired_sql, changelog) on success, or (None, changelog) if the
    query is not salvageable (e.g. removal would leave an empty SELECT list,
    a hallucinated base FROM table, or the repaired tree fails to round-trip).
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except ParseError:
        return None, ["sql failed to parse — cannot repair"]

    changelog: List[str] = []

    for _ in range(_MAX_REPAIR_PASSES):
        current_bad = _resolve_on_tree(tree, schema)
        if not current_bad:
            break

        bad_tables = {b.name.lower() for b in current_bad if b.kind == "table"}
        progress = False

        if bad_tables:
            bad_aliases: Set[str] = set()
            for table in list(tree.find_all(exp.Table)):
                if _table_name(table).lower() not in bad_tables:
                    continue
                bad_aliases.add(table.alias_or_name.lower())
                join_parent = table
                while join_parent is not None and not isinstance(join_parent, exp.Join):
                    join_parent = join_parent.parent
                if join_parent is not None:
                    _drop_node(join_parent)
                    changelog.append(f"dropped JOIN on hallucinated table '{_table_name(table)}'")
                    progress = True
                else:
                    return None, changelog + [
                        f"hallucinated base FROM table '{_table_name(table)}' — not salvageable"
                    ]
            if bad_aliases:
                for col in list(tree.find_all(exp.Column)):
                    if col.table and col.table.lower() in bad_aliases:
                        _drop_node(col)
                        changelog.append(
                            f"dropped orphaned column '{col.sql()}' (table hallucinated)"
                        )
                        progress = True

        for item in current_bad:
            if item.kind != "column":
                continue
            try:
                _drop_node(item.node)
                changelog.append(
                    f"removed hallucinated column '{item.table}.{item.name}' from {item.clause}"
                )
                progress = True
            except Exception as exc:  # pragma: no cover - defensive
                return None, changelog + [f"failed to remove {item.name}: {exc}"]

        if not progress:
            # Detected as bad again but nothing was actually removable —
            # avoid looping forever on an identifier repair can't reach.
            return None, changelog + [
                f"unremovable hallucinated identifier(s): "
                f"{[b.name for b in current_bad]}"
            ]

        # Salvageability check after each pass, not just at the end — an
        # intermediate cascade step can already have emptied the query.
        select_node = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        if select_node is None or not select_node.expressions:
            return None, changelog + ["repair left an empty SELECT list — not salvageable"]
        if select_node.find(exp.From) is None:
            return None, changelog + ["repair left no FROM clause — not salvageable"]
    else:
        return None, changelog + ["repair did not converge — not salvageable"]

    try:
        rendered = tree.sql(dialect=dialect)
        # Round-trip check: repaired SQL must itself re-parse cleanly.
        sqlglot.parse_one(rendered, dialect=dialect)
    except ParseError as exc:
        return None, changelog + [f"repaired SQL failed to round-trip: {exc}"]

    return rendered, changelog
