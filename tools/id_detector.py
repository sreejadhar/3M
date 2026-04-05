"""
LangChain tool: detect inclusion dependencies (INDs) across tables.

An Inclusion Dependency R[A] ⊆ S[B] holds when every value appearing in
column A of table R also appears in column B of table S.

A high-coverage IND (≥ threshold) is a foreign-key candidate.

Strategy:
1. For each pair of tables (R, S), find column pairs with compatible data types.
2. Prioritise using token-based Jaccard name similarity (more robust than substring
   matching — handles camelCase, underscores, common suffixes like _id/_key/_code).
3. Test single-column INDs via LEFT JOIN.
4. Test composite 2-column INDs for high-similarity pairs that appear together.
5. Optionally check bidirectional coverage (R→S and S→R) for the same column pair.
6. Classify IND type and generate plain-English descriptions.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ..connectors.base import BaseConnector


class IDDetectorInput(BaseModel):
    schema_name: str = Field(description="Database schema name")
    left_table: str = Field(description="Left-hand side table (R)")
    right_table: str = Field(description="Right-hand side table (S)")
    left_columns: List[Dict[str, str]] = Field(
        description="Columns of left table: [{name, data_type}, ...]"
    )
    right_columns: List[Dict[str, str]] = Field(
        description="Columns of right table: [{name, data_type}, ...]"
    )
    left_col_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-column stats for left table: {col_name: {uniqueness_ratio, ...}}"
    )
    right_col_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-column stats for right table: {col_name: {uniqueness_ratio, ...}}"
    )
    sample_size: int = Field(default=10_000)
    threshold: float = Field(default=0.95, description="Min coverage fraction")
    max_pairs: int = Field(default=100)
    check_bidirectional: bool = Field(
        default=True,
        description="Also test right→left inclusion for each candidate pair"
    )
    fk_uniqueness_threshold: float = Field(
        default=0.95,
        description=(
            "Min uniqueness_ratio required on the right-side column before an IND "
            "can be promoted to a FK candidate. Prevents fact→fact spurious FKs "
            "where both sides carry non-unique ID columns."
        ),
    )


def _classify_ind(coverage: float, name_sim: float) -> str:
    """Classify an IND based on coverage and name similarity."""
    if coverage >= 0.99 and name_sim >= 0.7:
        return "exact_foreign_key"
    if coverage >= 0.99:
        return "exact_value_inclusion"
    if coverage >= 0.95:
        return "strong_fk_candidate"
    if coverage >= 0.8:
        return "partial_inclusion"
    return "value_subset"


def _describe_ind(
    left_table: str,
    left_col: str,
    right_table: str,
    right_col: str,
    coverage: float,
    ind_type: str,
    is_composite: bool = False,
) -> str:
    """Generate a plain-English description grounded only in IND metadata."""
    pct = coverage * 100
    col_desc = (
        f"({left_col})" if not is_composite else f"composite key ({left_col})"
    )
    ref_desc = (
        f"({right_col})" if not is_composite else f"({right_col})"
    )

    if ind_type == "exact_foreign_key":
        return (
            f"'{left_table}'.{col_desc} references '{right_table}'.{ref_desc} "
            f"— strong FK candidate ({pct:.1f}% of distinct values match)"
        )
    if ind_type == "exact_value_inclusion":
        return (
            f"All distinct values in '{left_table}'.{col_desc} appear in "
            f"'{right_table}'.{ref_desc} ({pct:.1f}% coverage) "
            f"— possible FK, column names differ"
        )
    if ind_type == "strong_fk_candidate":
        return (
            f"'{left_table}'.{col_desc} mostly references '{right_table}'.{ref_desc} "
            f"({pct:.1f}% coverage) — likely FK with some orphan or unmatched records"
        )
    if ind_type == "partial_inclusion":
        return (
            f"Partial inclusion: {pct:.1f}% of '{left_table}'.{col_desc} values "
            f"appear in '{right_table}'.{ref_desc} — weak FK or shared code set"
        )
    return (
        f"'{left_table}'.{col_desc} values are a subset of "
        f"'{right_table}'.{ref_desc} ({pct:.1f}% coverage)"
    )


class InclusionDependencyTool(BaseTool):
    """
    Detect inclusion dependencies (R[A] ⊆ S[B]) between two database tables.

    Uses SQL LEFT JOIN to measure coverage.  Column pairs are ranked by
    token-based Jaccard name similarity (robust to camelCase, underscores,
    common ID suffixes).  Returns INDs with coverage scores, type
    classifications, and plain-English descriptions.  Optionally checks
    bidirectional coverage and composite 2-column INDs.
    """

    name: str = "inclusion_dependency_detector"
    description: str = (
        "Detect inclusion dependencies between two tables: "
        "checks if every value in left_table[col] exists in right_table[col]. "
        "High-coverage INDs (≥ threshold) are flagged as FK candidates with "
        "type classifications and plain-English descriptions."
    )
    args_schema: Type[BaseModel] = IDDetectorInput
    connector: BaseConnector = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Type compatibility
    # ------------------------------------------------------------------

    # Broadened type families — subtypes map to the same family
    _NUMERIC_FAMILY = frozenset({
        "int", "integer", "bigint", "smallint", "tinyint", "mediumint",
        "numeric", "decimal", "float", "double", "real", "number",
        "int2", "int4", "int8", "float4", "float8",
    })
    _STRING_FAMILY = frozenset({
        "varchar", "char", "text", "string", "nvarchar", "nchar",
        "character varying", "character", "clob", "longtext", "mediumtext",
        "tinytext", "citext",
    })
    _DATE_FAMILY = frozenset({
        "date", "timestamp", "datetime", "time",
        "timestamp without time zone", "timestamp with time zone",
        "timestamptz",
    })

    def _type_family(self, dtype: str) -> str:
        dt = dtype.lower().split("(")[0].strip()
        # Exact match first
        if dt in self._NUMERIC_FAMILY:
            return "numeric"
        if dt in self._STRING_FAMILY:
            return "string"
        if dt in self._DATE_FAMILY:
            return "date"
        # Substring fallback for compound type names (e.g. "character varying(255)")
        if any(t in dt for t in ("int", "num", "decimal", "float", "double", "real")):
            return "numeric"
        if any(t in dt for t in ("char", "text", "string", "clob")):
            return "string"
        if any(t in dt for t in ("date", "time", "stamp")):
            return "date"
        return "other"

    def _types_compatible(self, dtype_a: str, dtype_b: str) -> bool:
        """
        Returns True if the two data types can meaningfully be compared.
        Allows numeric↔numeric, string↔string, date↔date, other↔anything.
        Also allows numeric↔string for ID-like columns (IDs sometimes stored
        in mixed types across tables).
        """
        fa = self._type_family(dtype_a)
        fb = self._type_family(dtype_b)
        if fa == "other" or fb == "other":
            return True  # unknown type — optimistically test
        if fa == fb:
            return True
        # Allow numeric ↔ string for potential ID columns (common in legacy schemas)
        if {fa, fb} == {"numeric", "string"}:
            return True
        return False

    # ------------------------------------------------------------------
    # Name similarity (token-based Jaccard)
    # ------------------------------------------------------------------

    _STOP_TOKENS = frozenset({
        "id", "key", "code", "no", "num", "number", "fk", "pk",
        "ref", "cd", "nbr", "nr",
    })

    def _tokenise(self, name: str) -> Set[str]:
        """
        Split a column name into meaningful tokens.
        Handles: snake_case, camelCase, PascalCase, digit boundaries.
        Removes common stop tokens ONLY if they are not the sole token.
        """
        # Insert boundary before uppercase letters (camelCase → camel_Case)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        # Split on non-alphanumeric separators
        parts = re.split(r"[_\-\s\.]+", s.lower().strip("_"))
        tokens = {p for p in parts if p}
        # Remove stop tokens only if there are other tokens left
        meaningful = tokens - self._STOP_TOKENS
        return meaningful if meaningful else tokens

    def _name_similarity(self, a: str, b: str) -> float:
        """
        Token-based Jaccard similarity between two column names.
        Also checks exact match and suffix-stripped match for a fast score.
        """
        al, bl = a.lower(), b.lower()

        # Exact match
        if al == bl:
            return 1.0

        # Common suffix/prefix stripping (legacy: cust_id vs customer_id)
        _suffix_re = re.compile(r"(_id|_key|_code|_no|_num|_cd|_ref|id_|key_)$")

        def strip_suffix(s: str) -> str:
            return _suffix_re.sub("", s)

        if strip_suffix(al) == strip_suffix(bl):
            return 0.92

        # Substring containment
        if al in bl or bl in al:
            sim = min(len(al), len(bl)) / max(len(al), len(bl))
            if sim >= 0.6:
                return max(0.7, sim)

        # Jaccard on tokens
        ta = self._tokenise(a)
        tb = self._tokenise(b)
        if not ta or not tb:
            return 0.0
        intersection = len(ta & tb)
        union = len(ta | tb)
        return intersection / union if union else 0.0

    # ------------------------------------------------------------------
    # Right-side uniqueness gate
    # ------------------------------------------------------------------

    def _right_col_is_unique(
        self,
        col_name: str,
        col_stats: Optional[Dict[str, Any]],
        threshold: float,
    ) -> bool:
        """
        Return True if the right-side column is unique enough to be a valid FK target.

        If no stats are provided we optimistically allow it (backwards-compatible).
        If uniqueness_ratio is None (e.g. row_count=0) we also allow it.
        """
        if col_stats is None:
            return True
        stats = col_stats.get(col_name)
        if stats is None:
            return True
        ratio = stats.get("uniqueness_ratio")
        if ratio is None:
            return True
        return ratio >= threshold

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    # Column name patterns that suggest relational join/key columns
    _KEY_PATTERN = re.compile(
        r'(_id|_key|_code|_no|_num|_ref|id_|key_|fk_|pk_|_fk|_pk|_cd|_nbr|_nr'
        r'|^id$|^key$|^code$|^num$)$',
        re.IGNORECASE,
    )

    def _looks_like_key_col(self, name: str) -> bool:
        """True if the column name suggests it is an identifier/key/FK column."""
        return bool(self._KEY_PATTERN.search(name.lower()))

    def _generate_column_pairs(
        self,
        left_columns: List[Dict[str, str]],
        right_columns: List[Dict[str, str]],
        max_pairs: int,
    ) -> List[Tuple[str, str, float]]:
        """
        Returns list of (left_col, right_col, similarity) pairs to test,
        sorted by name similarity (descending) so high-signal pairs are first.
        Only includes type-compatible pairs.

        Zero-similarity pairs are only kept if at least one column looks like
        a key/ID column — avoids wasting test budget on unrelated columns.
        """
        candidates = []
        for lc in left_columns:
            for rc in right_columns:
                if not self._types_compatible(lc["data_type"], rc["data_type"]):
                    continue
                sim = self._name_similarity(lc["name"], rc["name"])
                # Discard zero-similarity pairs unless one column looks like a key
                if sim == 0.0 and not (
                    self._looks_like_key_col(lc["name"])
                    or self._looks_like_key_col(rc["name"])
                ):
                    continue
                candidates.append((lc["name"], rc["name"], sim))

        # Sort by similarity desc, then deterministically
        candidates.sort(key=lambda x: (-x[2], x[0], x[1]))
        return candidates[:max_pairs]

    def _generate_composite_candidates(
        self,
        left_columns: List[Dict[str, str]],
        right_columns: List[Dict[str, str]],
        single_col_pairs: List[Tuple[str, str, float]],
        max_composite: int = 20,
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Find 2-column composite IND candidates.

        Strategy: take the top single-column pairs by similarity and look for
        co-occurring pairs — if (A1→B1) and (A2→B2) both appear in the top-k
        single-column pairs, test the composite (A1,A2) ⊆ (B1,B2).
        """
        # Only consider reasonably-similar single pairs as composite seeds
        top_pairs = [(lc, rc) for lc, rc, sim in single_col_pairs if sim >= 0.6]
        if len(top_pairs) < 2:
            return []

        composite: List[Tuple[List[str], List[str]]] = []
        seen: Set[tuple] = set()

        for i in range(len(top_pairs)):
            for j in range(i + 1, len(top_pairs)):
                l1, r1 = top_pairs[i]
                l2, r2 = top_pairs[j]
                if l1 == l2 or r1 == r2:
                    continue  # same column on one side — not a composite pair
                key = (tuple(sorted([l1, l2])), tuple(sorted([r1, r2])))
                if key in seen:
                    continue
                seen.add(key)
                composite.append(([l1, l2], [r1, r2]))
                if len(composite) >= max_composite:
                    return composite

        return composite

    # ------------------------------------------------------------------
    def _run(
        self,
        schema_name: str,
        left_table: str,
        right_table: str,
        left_columns: List[Dict[str, str]],
        right_columns: List[Dict[str, str]],
        left_col_stats: Optional[Dict[str, Any]] = None,
        right_col_stats: Optional[Dict[str, Any]] = None,
        sample_size: int = 10_000,
        threshold: float = 0.95,
        max_pairs: int = 100,
        check_bidirectional: bool = True,
        fk_uniqueness_threshold: float = 0.95,
    ) -> str:
        try:
            pairs_with_sim = self._generate_column_pairs(
                left_columns, right_columns, max_pairs
            )
            found_inds: List[Dict] = []
            tested = 0

            # ---- Single-column INDs ----
            for left_col, right_col, sim in pairs_with_sim:
                # left → right
                coverage_lr = self.connector.check_inclusion_dependency(
                    schema_name,
                    left_table, [left_col],
                    right_table, [right_col],
                    sample_size,
                )
                tested += 1

                if coverage_lr >= threshold:
                    ind_type = _classify_ind(coverage_lr, sim)
                    # Right-side uniqueness gate: only promote to FK candidate
                    # if right_col is unique/near-unique (i.e. a valid PK/UK target).
                    # This prevents spurious fact→fact or table→table FK inferences
                    # where both sides share a non-unique key column.
                    right_is_unique = self._right_col_is_unique(
                        right_col, right_col_stats, fk_uniqueness_threshold
                    )
                    found_inds.append({
                        "left_table": left_table,
                        "left_columns": [left_col],
                        "right_table": right_table,
                        "right_columns": [right_col],
                        "coverage": round(coverage_lr, 4),
                        "name_similarity": round(sim, 4),
                        "is_foreign_key_candidate": coverage_lr >= 0.99 and right_is_unique,
                        "ind_type": ind_type if right_is_unique else "value_subset",
                        "is_composite": False,
                        "description": _describe_ind(
                            left_table, left_col, right_table, right_col,
                            coverage_lr, ind_type
                        ),
                    })

                # right → left (bidirectional check)
                if check_bidirectional:
                    coverage_rl = self.connector.check_inclusion_dependency(
                        schema_name,
                        right_table, [right_col],
                        left_table, [left_col],
                        sample_size,
                    )
                    tested += 1

                    if coverage_rl >= threshold:
                        ind_type = _classify_ind(coverage_rl, sim)
                        # For the reverse direction the "right" side is left_col
                        left_is_unique = self._right_col_is_unique(
                            left_col, left_col_stats, fk_uniqueness_threshold
                        )
                        found_inds.append({
                            "left_table": right_table,
                            "left_columns": [right_col],
                            "right_table": left_table,
                            "right_columns": [left_col],
                            "coverage": round(coverage_rl, 4),
                            "name_similarity": round(sim, 4),
                            "is_foreign_key_candidate": coverage_rl >= 0.99 and left_is_unique,
                            "ind_type": ind_type if left_is_unique else "value_subset",
                            "is_composite": False,
                            "description": _describe_ind(
                                right_table, right_col, left_table, left_col,
                                coverage_rl, ind_type
                            ),
                        })

            # ---- Composite 2-column INDs ----
            composite_candidates = self._generate_composite_candidates(
                left_columns, right_columns, pairs_with_sim
            )
            for left_cols, right_cols in composite_candidates:
                cov = self.connector.check_inclusion_dependency(
                    schema_name,
                    left_table, left_cols,
                    right_table, right_cols,
                    sample_size,
                )
                tested += 1
                if cov >= threshold:
                    ind_type = _classify_ind(cov, 0.9)
                    # For composite FKs all right-side columns must be unique
                    all_right_unique = all(
                        self._right_col_is_unique(rc, right_col_stats, fk_uniqueness_threshold)
                        for rc in right_cols
                    )
                    found_inds.append({
                        "left_table": left_table,
                        "left_columns": left_cols,
                        "right_table": right_table,
                        "right_columns": right_cols,
                        "coverage": round(cov, 4),
                        "name_similarity": None,
                        "is_foreign_key_candidate": cov >= 0.99 and all_right_unique,
                        "ind_type": ind_type if all_right_unique else "value_subset",
                        "is_composite": True,
                        "description": _describe_ind(
                            left_table,
                            " + ".join(left_cols),
                            right_table,
                            " + ".join(right_cols),
                            cov, ind_type, is_composite=True,
                        ),
                    })

            # Sort: highest coverage first, then highest name similarity
            found_inds.sort(key=lambda x: (-x["coverage"], -(x["name_similarity"] or 0)))

            return json.dumps({
                "left_table": left_table,
                "right_table": right_table,
                "inclusion_dependencies": found_inds,
                "pairs_tested": tested,
                "single_pairs_evaluated": len(pairs_with_sim),
                "composite_pairs_evaluated": len(composite_candidates),
            }, default=str)

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)
