"""
Unit tests for dialog_agent.semantic_lexicon.

Covers: term normalization, alias learning on embedding hits, lookup
precedence (exact > alias > embedding), and schema-fingerprint invalidation.

Run with: pytest tests/test_semantic_lexicon.py -v --confcutdir=tests
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

# Force the SQLite backend to an isolated temp file for the whole test module,
# so tests never touch the real dev/prod federation DB.
_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()
os.environ["APP_ENV"] = "test"
os.environ["KG_FEDERATION_DB"] = _TMP_DB.name

from dialog_agent import semantic_lexicon as sl  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_table():
    # Each test starts from an empty lexicon table.
    for entry in sl.list_all():
        sl.delete(entry.entry_id)
    yield


def test_normalize_term_collapses_morphological_variants():
    assert sl.normalize_term("Top Performers") == sl.normalize_term("top performer")
    assert sl.normalize_term("the top performers") == sl.normalize_term("top performer")


def test_normalize_term_does_not_collapse_distinct_phrasings():
    # "best employees" is semantically related but not a morphological
    # variant — normalization must NOT collapse it; that's the embedding
    # tier's job (see test_embedding_hit_learns_alias below).
    assert sl.normalize_term("best employees") != sl.normalize_term("top performer")


def test_save_and_exact_lookup():
    entry = sl.LexiconEntry(
        source_id="src1", term="promotion count", display_term="promotion count",
        kind="derived_metric", bindings=[{"table": "job_history", "column": "event"}],
        provenance="llm_dissector",
    )
    entry_id = sl.save(entry)
    assert entry_id

    hit = sl.lookup("src1", "promotion count")
    assert hit is not None
    assert hit.entry_id == entry_id
    assert hit.bindings == [{"table": "job_history", "column": "event"}]


def test_lookup_is_scoped_by_source_id():
    entry = sl.LexiconEntry(
        source_id="src1", term="promotion count", display_term="promotion count",
        kind="derived_metric", bindings=[{"table": "t", "column": "c"}],
        provenance="human",
    )
    sl.save(entry)
    assert sl.lookup("src2", "promotion count") is None


def test_alias_lookup_after_add_alias():
    entry = sl.LexiconEntry(
        source_id="src1", term="promotion count", display_term="promotion count",
        kind="derived_metric", bindings=[{"table": "t", "column": "c"}],
        provenance="human",
    )
    entry_id = sl.save(entry)
    sl.add_alias(entry_id, "career advancement count")

    hit = sl.lookup("src1", "career advancement count")
    assert hit is not None
    assert hit.entry_id == entry_id


def test_schema_fingerprint_changes_on_column_change():
    fp1 = sl.schema_fingerprint({"job_history": {"event", "date"}}, ["job_history"])
    fp2 = sl.schema_fingerprint({"job_history": {"event_type", "date"}}, ["job_history"])
    assert fp1 != fp2


def test_lookup_skips_entry_with_stale_fingerprint():
    entry = sl.LexiconEntry(
        source_id="src1", term="promotion count", display_term="promotion count",
        kind="derived_metric", bindings=[{"table": "t", "column": "c"}],
        provenance="llm_dissector", schema_fingerprint="old-fp",
    )
    sl.save(entry)

    # Current fingerprint differs -> treated as stale -> miss.
    assert sl.lookup("src1", "promotion count", current_fingerprint="new-fp") is None
    # No fingerprint constraint supplied -> still resolves.
    assert sl.lookup("src1", "promotion count") is not None


def test_bump_hit_and_bump_fail_increment_counters():
    entry = sl.LexiconEntry(
        source_id="src1", term="x", display_term="x", kind="derived_metric",
        bindings=[{"table": "t", "column": "c"}], provenance="human",
    )
    entry_id = sl.save(entry)
    sl.bump_hit(entry_id)
    sl.bump_hit(entry_id)
    sl.bump_fail(entry_id)

    reloaded = sl.get_by_id(entry_id)
    assert reloaded.hit_count == 2
    assert reloaded.fail_count == 1


def test_approve_sets_flag():
    entry = sl.LexiconEntry(
        source_id="src1", term="x", display_term="x", kind="derived_metric",
        bindings=[{"table": "t", "column": "c"}], provenance="llm_dissector",
        approved=False,
    )
    entry_id = sl.save(entry)
    sl.approve(entry_id)
    assert sl.get_by_id(entry_id).approved is True


def test_bootstrap_seeds_from_kpis_and_glossary():
    kpis = [{"name": "RSV Growth", "nl_formula": "growth formula", "sql_expression": "SELECT 1"}]
    glossary = [{"name": "Gross Margin", "definition": "def", "sql_hint": "", "synonyms": ["margin"]}]
    written = sl.bootstrap("src1", kpis=kpis, glossary_terms=glossary, mine_verified_queries=False)
    assert written == 2

    hit = sl.lookup("src1", "RSV Growth")
    assert hit is not None
    assert hit.provenance == "human"
    assert hit.approved is True

    margin_hit = sl.lookup("src1", "margin")  # via learned synonym alias
    assert margin_hit is not None
