"""
Regression tests for two additions to dialog_agent/nodes/resolve_node.py:

1. Abbreviation resolution reliability: _apply_fuzzy_fallback now reuses each
   candidate's precomputed overlap_tokens (which already includes
   _initials_match results) instead of recomputing overlap with
   _token_matches_text alone — the old code silently dropped every
   acronym-only candidate (e.g. "IT" -> "Information Technology", where
   neither word contains "it" as a literal substring).

2. Proper-noun disambiguation: _proper_noun_phrases / _detect_ambiguous_terms
   detect when a named entity in the question has 2+ equally-close candidate
   stored values (e.g. "Smith" could be "John Smith" or "Jane Smith") so
   plan_node can ask the user to pick one instead of guessing. See the
   corresponding short-circuit in dialog_agent/nodes/plan_node.py.

Run with: pytest tests/test_resolve_node_disambiguation.py -v
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("KG_FEDERATION_DB", _TMP_DB.name)

rn = importlib.import_module("dialog_agent.nodes.resolve_node")


class TestProperNounPhrases:
    def test_multiword_title_case_anywhere_counts(self):
        assert "John Smith" in rn._proper_noun_phrases("How many claims did John Smith handle?")

    def test_multiword_title_case_at_sentence_start_counts(self):
        assert "New York" in rn._proper_noun_phrases("New York had the highest revenue")

    def test_single_title_case_word_not_at_start_counts(self):
        assert "Gulf" in rn._proper_noun_phrases("how many employees work for Gulf")

    def test_first_word_of_sentence_excluded_when_single_word(self):
        # "What" is capitalized only because it's the first word of the
        # sentence, not because it's a proper noun.
        phrases = rn._proper_noun_phrases("What is the revenue for Q1")
        assert "What" not in phrases

    def test_all_caps_acronym_not_treated_as_proper_noun(self):
        assert rn._proper_noun_phrases("how many IT employees work for Gulf") == ["Gulf"]

    def test_plain_lowercase_question_has_no_phrases(self):
        assert rn._proper_noun_phrases("how many employees work in it") == []


class TestDetectAmbiguousTerms:
    def _candidate(self, table, column, value, tokens, score, match_type="direct"):
        return {
            "table": table, "column": column, "stored_value": value,
            "overlap_tokens": tokens, "score": score,
            "match_type": match_type, "promoted_from": None,
        }

    def test_two_tied_top_candidates_are_ambiguous(self):
        candidates = [
            self._candidate("employee", "name", "John Smith", ["smith"], 1),
            self._candidate("employee", "name", "Jane Smith", ["smith"], 1),
        ]
        result = rn._detect_ambiguous_terms("How many claims did Smith handle?", candidates)
        assert len(result) == 1
        assert result[0]["term"] == "Smith"
        assert set(result[0]["candidates"]) == {"John Smith", "Jane Smith"}

    def test_single_clear_winner_is_not_ambiguous(self):
        candidates = [
            self._candidate("employee", "name", "John Smith", ["john", "smith"], 2),
            self._candidate("employee", "name", "Jane Smith", ["smith"], 1),
        ]
        result = rn._detect_ambiguous_terms("How many claims did John Smith handle?", candidates)
        assert result == []

    def test_no_proper_noun_in_question_is_not_ambiguous(self):
        candidates = [
            self._candidate("dept", "name", "Information Technology", ["it"], 1),
            self._candidate("dept", "name", "Some Other Thing", ["it"], 1),
        ]
        result = rn._detect_ambiguous_terms("how many it employees", candidates)
        assert result == []

    def test_promoted_parent_candidates_never_trigger_ambiguity(self):
        candidates = [
            self._candidate("sales", "category", "Snacks & Foods", ["gulf"], 1, match_type="promoted_parent"),
            self._candidate("sales", "category", "Beverages", ["gulf"], 1, match_type="promoted_parent"),
        ]
        result = rn._detect_ambiguous_terms("revenue for Gulf region", candidates)
        assert result == []

    def test_no_candidates_at_all_returns_empty(self):
        assert rn._detect_ambiguous_terms("How many claims did Smith handle?", []) == []

    def test_candidates_capped_at_five(self):
        candidates = [
            self._candidate("employee", "name", f"Smith {i}", ["smith"], 1)
            for i in range(8)
        ]
        result = rn._detect_ambiguous_terms("How many claims did Smith handle?", candidates)
        assert len(result[0]["candidates"]) == 5


class TestFuzzyFallbackReusesOverlapTokens:
    def test_acronym_only_candidate_is_promoted_via_overlap_tokens(self):
        # "it" doesn't appear as a literal substring/stem/edit-distance match
        # inside "information technology" — only _initials_match finds it,
        # which is baked into overlap_tokens ahead of time. The fallback must
        # trust that precomputed overlap rather than recomputing from scratch.
        term_resolution = [{
            "user_term": "IT", "column": "", "table": "",
            "matched_values": [], "sql_fragment": None, "no_match": True,
        }]
        candidates = [{
            "table": "employee", "column": "department",
            "stored_value": "Information Technology",
            "overlap_tokens": ["it"], "score": 1,
            "match_type": "direct", "promoted_from": None,
        }]
        patched = rn._apply_fuzzy_fallback(term_resolution, candidates, {}, {})
        assert patched[0]["sql_fragment"] == "LOWER(department) = 'information technology'"
        assert patched[0]["matched_values"] == ["Information Technology"]

    def test_already_resolved_terms_are_left_untouched(self):
        term_resolution = [{
            "user_term": "IT", "column": "department", "table": "employee",
            "matched_values": ["Information Technology"],
            "sql_fragment": "LOWER(department) = 'information technology'",
            "no_match": False,
        }]
        patched = rn._apply_fuzzy_fallback(term_resolution, [], {}, {})
        assert patched == term_resolution

    def test_no_candidate_overlap_leaves_term_unresolved(self):
        term_resolution = [{
            "user_term": "Widgets", "column": "", "table": "",
            "matched_values": [], "sql_fragment": None, "no_match": True,
        }]
        candidates = [{
            "table": "employee", "column": "department",
            "stored_value": "Information Technology",
            "overlap_tokens": ["it"], "score": 1,
            "match_type": "direct", "promoted_from": None,
        }]
        patched = rn._apply_fuzzy_fallback(term_resolution, candidates, {}, {})
        assert patched[0]["sql_fragment"] is None
