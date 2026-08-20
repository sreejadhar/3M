"""
resolve_node — Pre-resolution of user query terms to exact database values.

This node runs BETWEEN understand_node and plan_node.  It makes a lightweight
LLM call that explicitly maps each filter concept in the user's question to the
exact values stored in categorical columns.

The resolved mappings are stored in state["term_resolution"] and injected into
the plan_node prompt as mandatory WHERE clause bindings.  This prevents the LLM
SQL planner from fabricating category values that do not exist in the data
(e.g. writing WHERE category = 'Savory Snacks' when the data uses 'Snacks & Foods').

Fuzzy pre-matching
------------------
Before calling the LLM, _fuzzy_match_candidates() does a Python-level token-overlap
scan: it tokenises the user question, removes stopwords, and finds every stored
categorical value that shares at least one significant word.  These candidates are
injected into the LLM prompt as "KEYWORD MATCH HINTS".  This converts the LLM's
task from "discover the match from scratch" to "confirm / rank pre-computed hints",
which is far more reliable for domain-specific terminology like:
  "savoury snacks" → 'Snacks & Foods'
  "snacks"         → 'Snacks & Foods'
  "cola"           → 'Carbonated Drinks'  (semantic, no keyword overlap)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from ..state import DialogState

logger = logging.getLogger(__name__)

try:
    import abbrev_glossary_registry as _abr  # governed abbreviation-glossary term registry
except ImportError:  # pragma: no cover - standalone module, optional at runtime
    _abr = None

# ── Stop-words stripped before token overlap scoring ────────────────────────
_STOP_WORDS = {
    "which", "what", "who", "where", "when", "why", "how", "the", "a", "an",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "must", "shall", "can", "and", "or", "but", "if", "of", "to",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "we",
    "i", "me", "my", "his", "her", "their", "our", "us", "you", "your",
    "showed", "show", "highest", "lowest", "top", "bottom", "best", "worst",
    "most", "least", "more", "less", "all", "each", "every", "any", "some",
    "no", "not", "just", "only", "also", "than", "so", "such", "had", "has",
    "last", "first", "second", "third", "full", "period", "growth", "change",
    "value", "share", "total", "average", "count", "number", "percent",
    "across", "within", "across", "among", "per", "vs", "versus",
}

# Single-character tokens and pure-numeric tokens are never useful for matching
_MIN_TOKEN_LEN = 2

# A stop-word written in ALL CAPS in the ORIGINAL question is very likely a
# domain acronym, not the common word it happens to collide with once
# lowercased — e.g. "IT" (Information Technology) vs. the pronoun "it", "OR"
# (an org/department code) vs. the conjunction "or". Capped at 5 chars so a
# genuinely shouted common word ("ALL", "SO") isn't mistaken for an acronym
# just because of a coincidental stop-word collision at longer length.
_MAX_ACRONYM_LEN = 5


def _tokenize_query(text: str) -> Set[str]:
    """
    Extract significant lowercase tokens from free text for fuzzy/keyword
    matching, stripping English stop-words — except tokens the user wrote in
    ALL CAPS, which survive even if they collide with a stop-word (see
    _MAX_ACRONYM_LEN above). Shared by both the fuzzy pre-match pass and the
    post-LLM fallback so the acronym exception only needs to live in one place.
    """
    raw_tokens = re.findall(r'\b[a-zA-Z]\w+\b', text)
    tokens: Set[str] = set()
    for raw in raw_tokens:
        lower = raw.lower()
        if len(lower) < _MIN_TOKEN_LEN:
            continue
        is_probable_acronym = raw.isupper() and len(raw) <= _MAX_ACRONYM_LEN
        if lower in _STOP_WORDS and not is_probable_acronym:
            continue
        tokens.add(lower)
    return tokens


def _acronym_tokens(text: str, require_caps: bool = True) -> Set[str]:
    """
    Lowercase forms of tokens that look like acronyms (≤ _MAX_ACRONYM_LEN
    chars) — e.g. "IT", "HR", "PO". Used to gate initials-matching (see
    _initials_match) so it only fires for terms that actually look like an
    acronym, not any short lowercase word that happens to spell out some
    phrase's initials by coincidence.

    require_caps=True (default) — only tokens the user actually wrote in ALL
    CAPS qualify. Used by the fast sample-based fuzzy pass (_fuzzy_match_candidates),
    which runs against every categorical column (products, regions, etc.); without
    this gate, common 2-3 letter stopwords ("is", "or", "an") would generate noisy
    initials-match hints across the whole schema.

    require_caps=False — any short token qualifies regardless of casing. Chat
    users don't reliably type acronyms like "IT" or "HR" in caps mid-sentence
    the way formal writing would. Used only by the live DB probe
    (_db_probe_candidates), which is already bounded to a small, targeted set
    of high-cardinality text columns (e.g. department/designation/position_title),
    so the false-positive surface is much smaller than the general fuzzy pass.
    """
    raw_tokens = re.findall(r'\b[a-zA-Z]\w+\b', text)
    out: Set[str] = set()
    for r in raw_tokens:
        if not (_MIN_TOKEN_LEN <= len(r) <= _MAX_ACRONYM_LEN):
            continue
        if require_caps and not r.isupper():
            continue
        out.add(r.lower())
    return out


def _initials_match(token: str, value: str) -> bool:
    """
    Generic acronym expansion check: does `token` equal the initials of some
    run of consecutive words in `value`? e.g. "it" == initials of
    "Information Technology", "hr" == initials of "Human Resources". No fixed
    dictionary — works for any domain's acronyms the same way.
    """
    words = re.findall(r"[A-Za-z]+", value)
    n = len(token)
    if len(words) < n:
        return False
    for i in range(len(words) - n + 1):
        if "".join(w[0] for w in words[i : i + n]).lower() == token:
            return True
    return False
    return tokens


# ── Proper-noun disambiguation ────────────────────────────────────────────────
# When the question names a specific entity (a person, product, place, ...)
# and the data has two or more equally-close stored values for it (e.g.
# "Smith" could be "John Smith" or "Jane Smith"), silently picking one is a
# guess. Detect that ambiguity here and let plan_node ask the user to choose
# instead of running SQL against a possibly-wrong entity.

_TITLE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _proper_noun_phrases(text: str) -> List[str]:
    """
    Heuristic proper-noun detector, for disambiguation purposes only: a run
    of one or more consecutive Title-Case words (e.g. "Smith", "John Smith",
    "New York"). A single Title-Case word at the very start of the text is
    excluded — English sentence-case capitalizes the first word regardless
    of whether it's a proper noun (e.g. "How many..."), so that position
    alone is not evidence of a name. A 2+-word Title-Case run counts even at
    the start, since two consecutive capitalized words is much stronger
    evidence of a real proper noun.

    Intentionally permissive — this is a heuristic, not a NER model. A false
    positive here is harmless unless it ALSO happens to collide with 2+
    evenly-tied stored values (see _detect_ambiguous_terms), which is rare
    for an ordinary capitalized word.

    Returns phrases in original casing, deduped, in first-seen order.
    """
    matches = list(_TITLE_WORD_RE.finditer(text))
    phrases: List[str] = []

    def _is_title(word: str) -> bool:
        return len(word) > 1 and word[0].isupper() and not word.isupper()

    i = 0
    n = len(matches)
    while i < n:
        if not _is_title(matches[i].group(0)):
            i += 1
            continue
        run_end = i
        j = i + 1
        while j < n and _is_title(matches[j].group(0)) \
                and text[matches[run_end].end():matches[j].start()].strip() == "":
            run_end = j
            j += 1
        run_len = run_end - i + 1
        is_sentence_start = matches[i].start() == 0
        if run_len >= 2 or not is_sentence_start:
            phrase = text[matches[i].start():matches[run_end].end()]
            if phrase not in phrases:
                phrases.append(phrase)
        i = run_end + 1
    return phrases


def _detect_ambiguous_terms(
    natural_query: str,
    fuzzy_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    For each proper-noun phrase in the question (see _proper_noun_phrases),
    check whether the fuzzy pre-match found 2+ DISTINCT stored values tied
    for the top score — i.e. the closest matches are genuinely ambiguous,
    not just "the LLM might guess wrong."

    Deliberately conservative:
    - Only "direct" and "db_probe" match types count — never
      "promoted_parent", which is a taxonomy-hierarchy concept (filtering at
      a parent category level) unrelated to named-entity ambiguity.
    - A phrase only qualifies when the TOP score is shared by 2+ distinct
      stored values. A single clear winner — even a low-scoring one — is
      not ambiguous and is left to the normal LLM/fuzzy-fallback resolution.

    Returns [{"term": "<phrase>", "candidates": [<stored values>, ...]}, ...],
    capped at 5 candidates per term. Empty when nothing is ambiguous — the
    overwhelming majority of questions.
    """
    phrases = _proper_noun_phrases(natural_query)
    if not phrases or not fuzzy_candidates:
        return []

    eligible = [c for c in fuzzy_candidates if c.get("match_type") in ("direct", "db_probe")]
    if not eligible:
        return []

    ambiguous: List[Dict[str, Any]] = []
    for phrase in phrases:
        phrase_tokens = _tokenize_query(phrase)
        if not phrase_tokens:
            continue
        # Require ALL words of the phrase to be present in the candidate's
        # overlap tokens, not just ANY one of them. With a plain intersection,
        # a multi-word business/department phrase like "Information
        # Technology" would count any stored value containing only
        # "information" (e.g. a certification title) OR only "technology"
        # (e.g. an unrelated company name) as a "match" — those single-word
        # coincidences then tie at the same low top_score across many
        # unrelated columns/tables and get reported as a bogus disambiguation
        # prompt. Requiring the full phrase's tokens keeps this rule scoped
        # to its intended purpose: genuine full-name collisions (e.g. two
        # different "John Smith" records), not partial-word noise.
        matches = [c for c in eligible if phrase_tokens <= set(c["overlap_tokens"])]
        if len(matches) < 2:
            continue
        top_score = max(c["score"] for c in matches)
        top_values = sorted({c["stored_value"] for c in matches if c["score"] == top_score})
        if len(top_values) >= 2:
            ambiguous.append({"term": phrase, "candidates": top_values[:5]})
    return ambiguous


_RESOLVE_SYSTEM = """\
You are a data analyst whose only job is to map terminology used in a natural-language
question to the exact values that exist in a database.

You will be given:
1. The user's question
2. A list of categorical columns with ALL their stored values
3. KEYWORD MATCH HINTS — Python-computed token overlaps that prove certain stored
   values share words with the user's question  ← use these as primary evidence

Your task:
- Identify every filter concept in the question (e.g. a product category, a country,
  a time period, a brand, a segment name)
- For each filter concept, find the best-matching stored value(s) using the rules below
- ⚠ MULTI-TABLE CONCEPTS: if a filter concept plausibly matches categorical columns in
  MORE THAN ONE table (e.g. a department/org concept like "IT" could match an org-name
  column on an employee table AND a department column on an education/qualification
  table), return a SEPARATE resolved_filters entry per table — one per (table, column)
  pair — each with its own sql_fragment for that table's column. Do NOT return only the
  single "best" table and drop the rest: the SQL planner may end up querying any one of
  these tables (or several), and each one needs its own valid filter so a query never
  silently loses the filter just because it doesn't touch the table you happened to pick.
- Return ONLY a JSON object — no prose, no markdown fences

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MATCHING RULES — try in order, stop at the first rule that gives a match
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0. AMBIGUOUS SHORT CODES — CHECK THE QUESTION'S DOMAIN WORD FIRST: a bare 1-3
   letter code (FE, EE, ME, BE, PR, QA, ...) can be an exact literal value in
   MORE THAN ONE unrelated column across the schema (e.g. a certification-type
   code on an education/certificates table AND a performance-rating/band code
   on a performance table). An exact string hit is NOT proof you found the
   right column — it only proves the string exists somewhere.
   Before trusting rule 1's exact match for a short code, look at the words
   surrounding it in the question (performance/rating/score/band/grade vs.
   certificate/license/qualification/degree, etc.) and prefer the stored
   column whose OWN name/purpose matches that domain word, even if a
   different table's column also happens to contain the same literal code.
   - If the question's domain word points to a column that does NOT contain
     an exact literal hit for the short code, still resolve against that
     column (e.g. via rule 5 semantic/abbreviation expansion) rather than
     falling back to the unrelated table's exact match.
   - Still return a resolved_filters entry per matching table as usual (per
     the MULTI-TABLE CONCEPTS instruction above), but set "reasoning" to
     flag which entry matches the question's actual domain and which is an
     incidental same-spelling collision from an unrelated concept, so the
     SQL planner can prefer the domain-correct one.

0.5. GOVERNED ABBREVIATION GLOSSARY MATCH — CHECK THIS BEFORE GUESSING (RULE 5):
   If an ABBREVIATION GLOSSARY HINTS block is shown below and one of its entries'
   abbreviation or full form matches the user's wording (either spelling), that
   entry is GROUND TRUTH — it was discovered by profiling this source's own data,
   not guessed from general knowledge. Use the hint's exact recorded stored
   spelling directly for that entry's table/column:
     sql_fragment: LOWER(col) = 'stored spelling'
   Do NOT hedge with an OR of both spellings for a hint that appears in this
   block — the discovery step already confirmed which spelling is stored. Set a
   high confidence and no_match=false. Only fall back to Rule 5's guess-and-hedge
   behavior for abbreviations that do NOT appear in ABBREVIATION GLOSSARY HINTS.

1. EXACT MATCH (case-insensitive): user term equals a stored value verbatim.
   sql_fragment: LOWER(col) = 'stored value'

2. YEAR / PERIOD FORMAT: user writes a plain year / quarter but the column
   stores it with a prefix or suffix (e.g. 'FY2024', 'CY2023', '2024-Q1').
   Always scan stored values for the format pattern and match accordingly.
   Examples:
     user "2024"   → stored 'FY2024'  → sql_fragment: LOWER(fiscal_year) = 'fy2024'
     user "2023"   → stored 'CY2023'  → sql_fragment: LOWER(calendar_year) = 'cy2023'
     user "Q1"     → stored 'FY2024-Q1', 'FY2023-Q1' → LIKE '%q1%'
   NEVER write WHERE fiscal_year = '2024' if the column stores 'FY2024'.

3. PARENT-CATEGORY MATCH WITH TAXONOMY HIERARCHY: If a TAXONOMY HIERARCHY is shown
   for a column, filtering at the PARENT level captures ALL child values automatically.
   Always prefer this over LIKE or IN on child columns.
   Examples:
     user "savoury snacks" → taxonomy shows: 'Snacks & Foods' → [Potato Chips & Crisps, ...]
       → sql_fragment: LOWER(category) = 'snacks & foods'
     user "beverages" → taxonomy shows: 'Beverages' → [Carbonated Drinks, Juices, ...]
       → sql_fragment: LOWER(category) = 'beverages'
   When a breakdown by sub-category is explicitly required, return BOTH parent AND
   child filter using the IN list from the taxonomy — never invent child values.

4. KEYWORD-WITHIN-VALUE MATCH: A significant word from the user's term appears
   inside a stored value (case-insensitive). This is the most common case for
   industry shorthand, abbreviated labels, and CPG/FMCG category names.
   Examples:
     user "snacks"         → stored 'Snacks & Foods'        → sql: LOWER(category) = 'snacks & foods'
     user "savoury snacks" → stored 'Snacks & Foods'        → sql: LOWER(category) = 'snacks & foods'
       (keyword "snacks" appears in 'Snacks & Foods')
     user "chips"          → stored 'Potato Chips & Crisps' → sql: LOWER(sub_category) = 'potato chips & crisps'
     user "dairy"          → stored 'Dairy & Eggs'          → sql: LOWER(category) = 'dairy & eggs'
     user "personal care"  → stored 'Beauty & Personal Care'→ sql: LOWER(category) = 'beauty & personal care'
   If KEYWORD MATCH HINTS are shown below, a hint for a given stored value is PROOF
   that a keyword overlap exists — you MUST use that stored value as the match.
   When multiple columns have keyword overlap for the same user term, prefer the
   PARENT (e.g. category > sub_category) unless the user specifically asks for
   a sub-category breakdown.

5. SEMANTIC MATCH (synonyms, domain knowledge — no keyword overlap at all):
   Use this when you know from domain knowledge that the user's term maps to
   a stored value even without a shared word. This includes ANY abbreviation
   or acronym — not just industry shorthand — whenever its expansion is a
   stored value: department/org codes (IT → Information Technology,
   HR → Human Resources, R&D → Research and Development, PR → Public
   Relations, QA → Quality Assurance), time shorthand (YoY → Year over Year,
   MoM → Month over Month), or domain terms.
   ⚠ THIS RULE IS DIRECTION-AGNOSTIC. The user may type EITHER the abbreviation
     ("IT") OR the fully spelled-out form ("Information Technology") — both
     phrasings refer to the exact same underlying concept, so both must resolve
     to the exact same sql_fragment. Never treat "how many employees in
     Information Technology" differently from "how many IT employees" — first
     recognize the full phrase as the expansion of a well-known abbreviation,
     THEN apply this rule exactly as if the user had typed the abbreviation
     itself.
   Examples:
     user "EMEA"                 → stored "Europe", "Middle East", "Africa"
     user "savoury"              → stored "Snacks & Foods"   (in FMCG: savoury = salty snack foods)
     user "soft drinks"          → stored "Beverages" or "Carbonated Drinks"
     user "HPC"                  → stored "Home & Personal Care"
     user "confectionery"        → stored "Candy & Chocolate" or "Sweets"
     user "IT"                   → stored "Information Technology"   (department/org column)
     user "Information Technology" → same concept as "IT" above       (user typed the expansion instead of the abbreviation — resolve identically)
     user "HR"                   → stored "Human Resources"
     user "Human Resources"      → same concept as "HR" above
     user "R&D"                  → stored "Research and Development"
     user "Research and Development" → same concept as "R&D" above
   Include ALL semantically equivalent stored values in an IN() list.
   ⚠ APPLIES even when the sample values shown to you don't literally contain the
     expansion — a well-known abbreviation with an obvious domain expansion (IT,
     HR, R&D, PR, QA, EMEA, etc.) is NOT a NO-MATCH case just because the column's
     shown samples happen not to include that exact row. Reason from the column's
     evident purpose (e.g. an org/department-name column almost certainly has an
     "Information Technology" or similarly-worded entry even if it wasn't in the
     handful of sampled values).
   ⚠ THIS IS A GUESS, NOT A CONFIRMED MATCH (unlike rule 1/4 above, which matched
     something actually shown to you) — you do NOT know whether the real column
     stores the bare abbreviation ("IT"), the expanded phrase ("Information
     Technology"), or both across different rows. This is true REGARDLESS of
     which spelling the user themselves typed. Do NOT commit the sql_fragment
     to only one spelling. OR together BOTH forms so either stored spelling is
     caught: an EXACT equality on the bare abbreviation OR'd with a LIKE on the
     expanded full form:
       LOWER(col) = 'it' OR LOWER(col) LIKE '%information technology%'
     (equality, not LIKE, on the bare abbreviation — LIKE '%it%' would also match
     "credIT", "capacITy", "digITal").
     Example combined sql_fragment:
       (LOWER(org_name) = 'it' OR LOWER(org_name) LIKE '%information technology%')
     This applies to every abbreviation resolved under this rule, regardless of
     which table/column it is being resolved against, and regardless of whether
     the user's own wording was the abbreviation or its full expansion.
   ⚠ NEVER split the expanded full form into separate single-word LIKE clauses
     (e.g. LOWER(col) LIKE '%information%' OR LOWER(col) LIKE '%technology%').
     Each bare word is a much broader match than the phrase (it also matches
     "Information Security", "Medical Technology", etc.) and defeats the point
     of resolving to the full form. Always keep the full form intact as ONE
     LIKE '%whole phrase%' clause — never break it apart word by word.

6. LAST RESORT — NO MATCH: ONLY use this when rules 1-5 all fail AND there are
   no KEYWORD MATCH HINTS for this term.
   Set matched_values to [], sql_fragment to null, and no_match to true.
   ⚠ If KEYWORD MATCH HINTS shows overlap for a user term, you MUST NOT use
     NO MATCH for that term — the hint is Python-computed proof of overlap.
   ⚠ Do NOT reach for NO MATCH just because rule 5's expansion isn't literally
     present among the sampled values — see the note under rule 5. NO MATCH is
     reserved for terms with NO plausible domain mapping to the column at all.
   ⚠ NO MATCH is the CORRECT answer when the user's term genuinely does not
     appear in any stored column value AND has no reasonable expansion/synonym
     relationship to the column's purpose. For example: if the user says "Coca Cola"
     but the [categorical] column only stores manufacturer codes like "CCEP",
     "Suntory", "Asahi" — then NO MATCH is correct. The SQL planner will then
     know NOT to add a filter for this term instead of fabricating a wrong one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OUTPUT FORMAT (return exactly this JSON, no other text):
{
  "resolved_filters": [
    {
      "user_term": "<the term from the question>",
      "column": "<column name>",
      "table": "<table name>",
      "reasoning": "<one sentence explaining the match or why no match was found>",
      "matched_values": ["<stored value 1>", "<stored value 2>"],
      "sql_fragment": "<LOWER(column) = 'stored value' or LOWER(column) IN (...) or LOWER(column) LIKE '%...'  — null if NO MATCH>",
      "no_match": false
    }
  ]
}

For NO MATCH cases set: matched_values=[], sql_fragment=null, no_match=true.
If no categorical filters are needed (e.g. purely numeric aggregation), return:
{"resolved_filters": []}

Multiple entries CAN share the same "user_term" when that concept matches columns in
different tables (see MULTI-TABLE CONCEPTS above) — each entry's "table"/"column" then
tells the SQL planner which query that entry's sql_fragment applies to.
"""


_STEM_SUFFIXES = ("ies", "ing", "ness", "ment", "tion", "sion", "er", "est", "ed", "es", "s")


def _stem(token: str) -> str:
    """Strip common English suffixes so 'snacks'/'snack', 'savoury'/'savour' share a root."""
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — fast early-exit when gap > 2."""
    if abs(len(a) - len(b)) > 2:
        return 99
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _token_matches_text(token: str, text: str) -> bool:
    """
    True when token fuzzy-matches anywhere inside text via three strategies:
    1. Exact substring  (savory  ∈ 'Savory Snacks')
    2. Stemmed match    (snack   ~ 'Snacks & Foods' after stripping -s)
    3. Edit distance ≤ 1 for tokens ≥ 5 chars  (savory ≈ savoury)
    """
    # Whole-word boundary, not a raw substring test: "it" must not match inside
    # "limited" or "circuit" just because the letters happen to appear
    # consecutively. This matters most for short tokens (2-3 chars), which are
    # exactly the ones likely to turn up as a fragment of an unrelated longer
    # word. Genuine cases like "chips" ∈ "Potato Chips & Crisps" are already
    # word-aligned, so this doesn't lose any real matches.
    if re.search(r'\b' + re.escape(token) + r'\b', text):
        return True
    stem_t = _stem(token)
    text_words = re.findall(r'\b\w+\b', text)
    # Stemmed substring: stem(token) inside text
    if stem_t != token and stem_t in text:
        return True
    # Stem each word in the stored value and compare stems
    if len(stem_t) >= 3:
        for word in text_words:
            if _stem(word) == stem_t:
                return True
    # Edit distance for longer tokens (spelling variants: savory/savoury)
    if len(token) >= 5:
        for word in text_words:
            if abs(len(word) - len(token)) <= 2 and _edit_distance(token, word) <= 1:
                return True
    return False


def _fuzzy_match_candidates(
    natural_query: str,
    categorical_columns: Dict[str, Dict[str, List[str]]],
    column_hierarchy: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Python-level fuzzy scan between user question tokens and stored categorical values.

    Matching strategy (each adds to a token's match score):
    1. Exact substring      — token appears inside the stored value string
    2. Stemmed match        — stem(token) matches stem of any word in the stored value
    3. Edit distance ≤ 1   — handles spelling variants (savory/savoury, colour/color)

    Hierarchy promotion:
    When a token matches a value in a CHILD column (sub_category) and a parent column
    exists (category), an additional candidate is added at the PARENT level so the LLM
    is guided to filter at the correct hierarchy level.

    Returns candidates sorted by score descending:
        [{"table", "column", "stored_value", "overlap_tokens", "score",
          "match_type", "promoted_from"}, ...]
    """
    query_tokens = _tokenize_query(natural_query)
    if not query_tokens:
        return []
    acronym_tokens = _acronym_tokens(natural_query) & query_tokens

    # Build reverse map: child_col → parent_col per table (from hierarchy + naming)
    child_to_parent: Dict[str, Dict[str, str]] = {}  # tbl → {child_col: parent_col}
    hier = column_hierarchy or {}
    for tbl, tbl_hier in hier.items():
        for parent_col in tbl_hier:
            child_col = "sub_" + parent_col
            child_to_parent.setdefault(tbl, {})[child_col] = parent_col
    # Also derive from naming convention regardless of stored hierarchy
    for tbl, col_map in categorical_columns.items():
        for col in col_map:
            if col.startswith("sub_"):
                parent_candidate = col[4:]
                if parent_candidate in col_map:
                    child_to_parent.setdefault(tbl, {})[col] = parent_candidate

    candidates: List[Dict[str, Any]] = []
    seen: set = set()
    # Track promoted parent candidates separately to avoid duplicates
    promoted_parents: set = set()

    for tbl, col_map in categorical_columns.items():
        tbl_child_map = child_to_parent.get(tbl, {})
        for col, vals in col_map.items():
            parent_col = tbl_child_map.get(col)
            parent_vals = col_map.get(parent_col, []) if parent_col else []

            for val in vals:
                if val is None:
                    continue
                val_lower = val.lower()
                matched_tokens = {t for t in query_tokens if _token_matches_text(t, val_lower)}
                matched_tokens |= {t for t in acronym_tokens if _initials_match(t, val)}
                if not matched_tokens:
                    continue

                key = (tbl, col, val)
                if key not in seen:
                    seen.add(key)
                    candidates.append({
                        "table":          tbl,
                        "column":         col,
                        "stored_value":   val,
                        "overlap_tokens": sorted(matched_tokens),
                        "score":          len(matched_tokens),
                        "match_type":     "direct",
                        "promoted_from":  None,
                    })

                # Hierarchy promotion: child match → add each parent value as candidate
                # so the LLM filters at the correct (parent) level.
                if parent_col and parent_vals:
                    for pval in parent_vals:
                        if pval is None:
                            continue
                        pkey = (tbl, parent_col, pval)
                        if pkey in promoted_parents:
                            continue
                        promoted_parents.add(pkey)
                        # Score the parent candidate by how well the query tokens
                        # match the parent value itself
                        pval_lower = pval.lower()
                        p_matched = {t for t in query_tokens if _token_matches_text(t, pval_lower)}
                        # Only promote parents with at least some token affinity
                        # OR where the child match is strong (score >= 1)
                        if p_matched or len(matched_tokens) >= 1:
                            # Bonus: +len(matched_tokens) from the child match that drove promotion.
                            # The parent value that also matches query tokens gets an extra boost,
                            # so the correct parent ranks above siblings that share no tokens.
                            parent_bonus = len(matched_tokens) if p_matched else 0
                            candidates.append({
                                "table":          tbl,
                                "column":         parent_col,
                                "stored_value":   pval,
                                "overlap_tokens": sorted(matched_tokens | p_matched),
                                "score":          len(p_matched) + parent_bonus + 1,
                                "match_type":     "promoted_parent",
                                "promoted_from":  f"{col}={val!r}",
                            })

    # Sort: promoted parents first (higher score), then direct matches; tie-break shorter value
    return sorted(candidates, key=lambda x: (-x["score"], len(x["stored_value"])))


def _build_categorical_context(
    categorical_columns: Dict[str, Dict[str, List[str]]],
    column_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = None,
) -> str:
    """
    Format categorical columns (with taxonomy hierarchy when available) for the resolve prompt.

    For parent columns that have a hierarchy (e.g. category → sub_category), the context
    shows the full parent→children mapping so the LLM can:
      1. Match the user term to the correct parent value
      2. Understand that filtering at parent level captures all children automatically
    """
    if not categorical_columns:
        return "(No categorical columns found in schema)"

    hierarchy = column_hierarchy or {}
    lines: List[str] = []

    for tbl, col_map in categorical_columns.items():
        tbl_hier = hierarchy.get(tbl, {})
        for col, vals in col_map.items():
            quoted = ", ".join(repr(v) for v in vals)
            if col in tbl_hier:
                # Parent column: show the hierarchy so the LLM can filter at parent level
                lines.append(
                    f"  table={tbl!r}  column={col!r}  [PARENT — stored values: [{quoted}]]"
                )
                lines.append(
                    f"    TAXONOMY HIERARCHY (filtering at {col!r} level captures ALL children automatically):"
                )
                for pval, cvals in sorted(tbl_hier[col].items()):
                    children_str = " | ".join(cvals)
                    lines.append(f"      '{pval}' → [{children_str}]")
            else:
                lines.append(f"  table={tbl!r}  column={col!r}  stored values: [{quoted}]")

    return "\n".join(lines)


def _build_hint_section(candidates: List[Dict[str, Any]]) -> str:
    """
    Format fuzzy candidates as a KEYWORD MATCH HINTS block for the LLM prompt.
    Promoted parent candidates are labelled so the LLM knows to prefer them.
    Limited to the top 25 candidates to avoid bloating the context.
    """
    if not candidates:
        return ""
    lines = [
        "KEYWORD MATCH HINTS (Python-computed — fuzzy token overlap with spelling/stem tolerance):",
        "These are PROVEN matches. You MUST use them unless a better/higher-priority match exists.",
        "Hints labelled [PARENT LEVEL] mean: filter at the parent column — do NOT use LIKE on a child column.",
        "",
    ]
    for c in candidates[:25]:
        tokens_str = ", ".join(f'"{t}"' for t in c["overlap_tokens"])
        match_type = c.get("match_type", "direct")
        if match_type == "promoted_parent":
            label = "  [PARENT LEVEL — via child match]  "
        elif match_type == "db_probe":
            label = "  [LIVE DB MATCH — high-cardinality column]  "
        else:
            label = "  "
        promoted = f"  (child: {c['promoted_from']})" if c.get("promoted_from") else ""
        lines.append(
            f"{label}token(s) [{tokens_str}] → "
            f"table={c['table']!r}  column={c['column']!r}  value={c['stored_value']!r}{promoted}"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _build_abbrev_hint_section(source_id: str) -> str:
    """
    Format governed abbreviation-glossary terms for this source as an
    ABBREVIATION GLOSSARY HINTS block. Unlike KEYWORD MATCH HINTS (proven
    token overlap) this is a GROUND-TRUTH mapping discovered by profiling
    this source's own columns (abbrev_glossary_generate.py) — the LLM should
    use the exact stored spelling it records rather than guessing/hedging
    both spellings the way Rule 5 (SEMANTIC MATCH) has to for un-governed
    abbreviations.
    """
    if not source_id or _abr is None:
        return ""
    try:
        terms = _abr.list_terms(source_id=source_id, status="approved")
    except Exception as exc:
        logger.debug("resolve_node: abbrev glossary lookup failed for source %s — %s", source_id, exc)
        return ""
    if not terms:
        return ""

    try:
        import metadata_catalog as _mc
    except ImportError:
        _mc = None

    entity_cache: Dict[str, Optional[Dict]] = {}

    def _get_entity(metadata_id: str) -> Optional[Dict]:
        if not _mc or not metadata_id:
            return None
        if metadata_id not in entity_cache:
            entity_cache[metadata_id] = _mc.get_entity(metadata_id)
        return entity_cache[metadata_id]

    lines = [
        "ABBREVIATION GLOSSARY HINTS (governed, discovered from this source's own columns — ground truth, not a guess):",
        "Each entry names the abbreviation, its full form, and the exact table/column where the abbreviation "
        "is the value actually STORED. If the user's term matches either spelling, use the stored spelling "
        "directly — do NOT hedge with an OR of both spellings for these entries.",
        "",
    ]
    added = 0
    for term in terms[:40]:
        abbrev = term.get("abbreviation", "")
        full_form = term.get("full_form", "")
        if not abbrev or not full_form:
            continue
        try:
            assets = _abr.list_assets_for_term(term["term_id"])
        except Exception:
            assets = []
        located = False
        for asset in assets:
            if asset.get("source_id") != source_id:
                continue
            entity = _get_entity(asset.get("metadata_id", ""))
            if not entity:
                continue
            table_name = entity.get("table_name", "")
            column_name = ""
            attr_id = asset.get("attr_id") or ""
            if attr_id:
                for attr in entity.get("attributes") or []:
                    if attr.get("attr_id") == attr_id:
                        column_name = attr.get("column_name", "")
                        break
            lines.append(
                f'  "{abbrev}" = "{full_form}"  →  table={table_name!r}  column={column_name!r}  '
                f'(stored spelling: {abbrev!r})'
            )
            located = True
            added += 1
        if not located:
            lines.append(f'  "{abbrev}" = "{full_form}"')
            added += 1
    if not added:
        return ""
    lines.append("")
    return "\n".join(lines) + "\n"


def get_glossary_abbreviation_map(source_id: str) -> Dict[str, str]:
    """
    {abbreviation.lower(): full_form.lower()} for every approved governed
    abbreviation-glossary term on this source — the same ground-truth data
    _build_abbrev_hint_section() formats for the LLM prompt, but as a plain
    dict for plan_node.py's deterministic SQL post-processing (see
    _fix_short_acronym_like_filters / _fix_unhedged_expansion_like_filters),
    so that safety net covers whatever abbreviations THIS source's data
    actually contains, not just a fixed hardcoded list.
    """
    if not source_id or _abr is None:
        return {}
    try:
        terms = _abr.list_terms(source_id=source_id, status="approved")
    except Exception as exc:
        logger.debug("resolve_node: abbrev glossary lookup failed for source %s — %s", source_id, exc)
        return {}
    mapping: Dict[str, str] = {}
    for term in terms:
        abbrev = (term.get("abbreviation") or "").strip().lower()
        full_form = (term.get("full_form") or "").strip().lower()
        if abbrev and full_form:
            mapping[abbrev] = full_form
    return mapping


def _call_resolve_llm(system: str, user: str, model: str) -> str:
    from llm_client import get_client
    client = get_client()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,   # deterministic — we want the most confident match
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text if msg.content else ""


def _parse_resolution(text: str) -> List[Dict[str, Any]]:
    """Extract the resolved_filters list from the LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj.get("resolved_filters", [])
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        start = cleaned.find("{")
        end   = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(cleaned[start : end + 1])
                return obj.get("resolved_filters", [])
            except json.JSONDecodeError:
                pass
    logger.warning("resolve_node: could not parse LLM response as JSON")
    return []


def _apply_fuzzy_fallback(
    term_resolution: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    categorical_columns: Dict[str, Dict[str, List[str]]],
    column_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]],
) -> List[Dict[str, Any]]:
    """
    Post-LLM safety net: for any resolved filter whose sql_fragment is null/empty
    but which has a fuzzy candidate with score >= 1, inject the best candidate
    as the resolution automatically (without another LLM call).

    This catches cases where the LLM returned NO MATCH despite clear keyword
    overlap — e.g. "savoury snacks" returned null but hints showed 'Snacks & Foods'.
    """
    # Index candidates by column for quick lookup
    best_by_col: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        col = c["column"]
        if col not in best_by_col or c["score"] > best_by_col[col]["score"]:
            best_by_col[col] = c

    patched = []
    for r in term_resolution:
        if r.get("sql_fragment"):
            patched.append(r)
            continue
        # Find the best fuzzy candidate whose column has the highest overlap
        # and whose stored value shares tokens with the user_term.
        #
        # Reuse each candidate's own precomputed overlap_tokens (set
        # intersection) rather than recomputing overlap via
        # _token_matches_text alone — overlap_tokens was built in
        # _fuzzy_match_candidates using substring/stem/edit-distance AND
        # _initials_match, so it already captures acronym matches (e.g. "it"
        # -> "Information Technology", where neither word literally contains
        # "it" as a substring, only as initials). Recomputing with
        # _token_matches_text alone silently dropped every acronym-only
        # candidate here, so a question like "how many IT employees" could
        # fail the LLM's own resolution and then ALSO fail this safety net,
        # even though the correct candidate had already been found earlier.
        term_tokens = _tokenize_query(r.get("user_term", ""))
        best: Optional[Dict] = None
        best_score = 0
        for c in candidates:
            overlap = len(term_tokens & set(c.get("overlap_tokens") or []))
            # Promoted parent candidates get a bonus so they beat child-level matches
            if c.get("match_type") == "promoted_parent":
                overlap += 1
            if overlap > best_score:
                best_score = overlap
                best = c

        if best and best_score > 0:
            col  = best["column"]
            val  = best["stored_value"]
            tbl  = best["table"]
            frag = f"LOWER({col}) = '{val.lower()}'"
            patched_r = dict(r)
            patched_r["column"]         = col
            patched_r["table"]          = tbl
            patched_r["matched_values"] = [val]
            patched_r["sql_fragment"]   = frag
            patched_r["reasoning"]      = (
                f"Fuzzy fallback: query token(s) {best['overlap_tokens']} "
                f"matched stored value '{val}' in column '{col}'."
            )
            patched.append(patched_r)
            logger.info(
                "resolve_node: fuzzy fallback matched %r → %r in %r.%r",
                r.get("user_term"), val, tbl, col,
            )
        else:
            patched.append(r)

    return patched


# ── Live DB probe for high-cardinality text columns ──────────────────────────
# _fuzzy_match_candidates only ever sees columns that made the "categorical"
# cut (understand_node caps that at ~50 distinct values, or for DB-backed
# sources whatever a cached top-values sample happened to capture). A column
# like `position_title` with 1,000+ distinct values never gets a full value
# list — so a term like "IT" has nothing to match against even though rows
# like "IT Data Engineer" or "Head of Information Technology" genuinely exist.
# This is the generic fallback: when a query token survives tokenization but
# still has no match after the fast, sample-based pass, run a bounded, live
# LIKE search against the source's own high-cardinality text columns. Works
# for any token/column/dialect — nothing here is specific to "IT" or any one
# schema.

_TEXT_TYPE_HINTS = ("string", "text", "char", "clob")
_MAX_PROBE_COLUMNS = 15
_PROBE_DISTINCT_LIMIT = 500


def _text_columns_from_kg_nodes(kg_nodes: List[Dict]) -> Dict[str, List[str]]:
    """table label → text-typed column names, read straight off the KG nodes
    (the same node["properties"] shape used throughout dialog_agent)."""
    out: Dict[str, List[str]] = {}
    for node in kg_nodes or []:
        label = node.get("label") or ""
        if not label:
            continue
        cols: List[str] = []
        for prop in node.get("properties") or []:
            col = (prop.get("name") or "").strip()
            dtype = (prop.get("type") or "").lower()
            if col and any(h in dtype for h in _TEXT_TYPE_HINTS):
                cols.append(col)
        if cols:
            out[label] = cols
    return out


def _probe_candidate_columns(
    text_columns_by_table: Dict[str, List[str]],
    categorical_columns: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    """Text columns minus whatever the fast sample-based pass already covers —
    these are the columns a live probe can add value on."""
    out: Dict[str, List[str]] = {}
    for tbl, cols in text_columns_by_table.items():
        already_covered = set(categorical_columns.get(tbl, {}).keys())
        extra = [c for c in cols if c not in already_covered]
        if extra:
            out[tbl] = extra
    return out


def _db_probe_candidates(
    config: Any,
    state: DialogState,
    unresolved_tokens: List[str],
    acronym_tokens: Set[str],
    probe_columns: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """
    For tokens the cheap sample-based pass found no evidence for, fetch each
    high-cardinality text column's distinct values ONCE (not once per token —
    a per-token `LIKE` loop burns its round-trip budget on whichever token
    happens to iterate first and can starve out the column that actually
    matters) and match every unresolved token against that one result set in
    Python, using the same whole-word logic as the fast pass plus
    initials-matching for acronym tokens (so "IT" can still resolve to
    "Information Technology" even though it isn't a literal substring of it).

    Reuses execute_node._run_sql so this works across every dialect the
    pipeline already supports (Postgres, Snowflake, the in-memory SQLite file
    backends, etc.) without any new connection logic. Bounded to
    _MAX_PROBE_COLUMNS round trips — this only runs at all when there's at
    least one genuinely unresolved token, so the common case (fast pass
    already found everything) pays nothing extra.

    The round trips are run concurrently (same ThreadPoolExecutor pattern
    execute_node already uses for independent queries, see execute_node.py's
    multi-query dispatch): each _run_sql call opens and closes its own DB
    connection, so probing 15 unrelated columns one at a time was pure
    serialized waiting — measured at ~60s in production traces — with no
    correctness reason for the serialization.
    """
    if not unresolved_tokens or not probe_columns:
        return []

    from .execute_node import _run_sql
    from .understand_node import _FILE_BASED_TYPES, _qualified, _to_sql_col, _to_sql_table

    is_file_based = config.db_type.lower() in _FILE_BASED_TYPES
    db_schema = "" if is_file_based else (config.db_schema or "")

    # Flatten to a bounded list of (table, column) probe targets up front so
    # the _MAX_PROBE_COLUMNS cap applies the same way it did sequentially,
    # before fanning the round trips out across a thread pool.
    probe_targets: List[tuple] = []
    for tbl, cols in probe_columns.items():
        if len(probe_targets) >= _MAX_PROBE_COLUMNS:
            break
        sql_tbl = _to_sql_table(tbl) if is_file_based else tbl
        table_q = _qualified(db_schema, sql_tbl)
        for col in cols:
            if len(probe_targets) >= _MAX_PROBE_COLUMNS:
                break
            sql_col = _to_sql_col(col) if is_file_based else col
            probe_targets.append((tbl, col, table_q, sql_col))

    def _probe_one(target: tuple) -> List[Dict[str, Any]]:
        tbl, col, table_q, sql_col = target
        sql = (
            f"SELECT DISTINCT {sql_col} FROM {table_q} "
            f"WHERE {sql_col} IS NOT NULL LIMIT {_PROBE_DISTINCT_LIMIT}"
        )
        try:
            result = _run_sql(config, sql, state)
        except Exception as exc:
            logger.debug("resolve_node: DB probe failed for %s.%s: %s", tbl, col, exc)
            return []
        if result.get("error"):
            return []
        found: List[Dict[str, Any]] = []
        for row in result.get("rows") or []:
            val = row[0] if row else None
            if val is None:
                continue
            val_str = str(val)
            val_lower = val_str.lower()
            matched = {t for t in unresolved_tokens if _token_matches_text(t, val_lower)}
            matched |= {t for t in acronym_tokens if _initials_match(t, val_str)}
            if not matched:
                continue
            found.append({
                "table":          tbl,
                "column":         col,
                "stored_value":   val_str,
                "overlap_tokens": sorted(matched),
                "score":          len(matched),
                "match_type":     "db_probe",
                "promoted_from":  None,
            })
        return found

    candidates: List[Dict[str, Any]] = []
    columns_probed = len(probe_targets)
    if len(probe_targets) == 1:
        candidates.extend(_probe_one(probe_targets[0]))
    elif probe_targets:
        import concurrent.futures
        max_workers = min(len(probe_targets), 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for found in pool.map(_probe_one, probe_targets):
                candidates.extend(found)

    if candidates:
        logger.info(
            "resolve_node: DB probe found %d candidate value(s) for unresolved "
            "token(s) %s across %d high-cardinality column(s) (%d columns probed)",
            len(candidates), unresolved_tokens, len(probe_columns), columns_probed,
        )
    return candidates


def resolve_node(state: DialogState) -> DialogState:
    """
    Map user query terms to exact categorical column values before SQL planning.
    Populates state['term_resolution'] with a list of resolved filter mappings.
    """
    logger.info("=== resolve_node ===")

    categorical_columns: Dict[str, Dict[str, List[str]]] = state.get("categorical_columns") or {}
    column_hierarchy: Dict[str, Dict[str, Dict[str, List[str]]]] = state.get("column_hierarchy") or {}
    natural_query: str = state.get("natural_query", "")
    kg_nodes = state.get("kg_nodes") or []

    # Skip only when there's nothing at all to resolve against — no sampled
    # categorical values AND no schema (so a live probe has no columns to hit
    # either). A source with only high-cardinality text columns (no column
    # ever made the categorical cut) still reaches here now, since that's
    # exactly the case the DB probe below exists to cover.
    if not categorical_columns and not kg_nodes:
        logger.info("resolve_node: no categorical columns or schema — skipping resolution")
        state["term_resolution"] = []
        return state

    config = state["config"]
    # resolve_node is a structured JSON extraction task — always use the cheap
    # plan model (Haiku) regardless of which synthesis model the user selected.
    model  = getattr(config, "plan_llm_model", None) or "claude-haiku-4-5"

    # ── Fuzzy pre-match: stemmed/edit-distance token overlap + hierarchy promotion ──
    fuzzy_candidates = _fuzzy_match_candidates(natural_query, categorical_columns, column_hierarchy)

    # ── Live DB probe: give tokens the fast pass couldn't place a second shot
    # against high-cardinality text columns that never got a full sample. ────
    matched_tokens = {t for c in fuzzy_candidates for t in c["overlap_tokens"]}
    unresolved_tokens = sorted(_tokenize_query(natural_query) - matched_tokens)
    # Case-insensitive here (unlike the sample-based pass above): a token like
    # "it" typed in normal sentence case should still be tried as an acronym
    # against the live-probed high-cardinality columns — see _acronym_tokens.
    unresolved_acronyms = _acronym_tokens(natural_query, require_caps=False) - matched_tokens
    if unresolved_tokens or unresolved_acronyms:
        text_columns_by_table = _text_columns_from_kg_nodes(kg_nodes)
        probe_columns = _probe_candidate_columns(text_columns_by_table, categorical_columns)
        db_candidates = _db_probe_candidates(
            config, state, unresolved_tokens, unresolved_acronyms, probe_columns
        )
        fuzzy_candidates = fuzzy_candidates + db_candidates

    hint_section = _build_hint_section(fuzzy_candidates)
    abbrev_hint_section = _build_abbrev_hint_section(getattr(config, "source_id", "") or "")
    if fuzzy_candidates:
        logger.info(
            "resolve_node: fuzzy pre-match found %d candidate(s): %s",
            len(fuzzy_candidates),
            [(c["column"], c["stored_value"], c["overlap_tokens"]) for c in fuzzy_candidates[:5]],
        )

    categorical_context = _build_categorical_context(categorical_columns, column_hierarchy)

    user_prompt = (
        f"USER QUESTION:\n{natural_query}\n\n"
        f"CATEGORICAL COLUMNS AND THEIR STORED VALUES:\n{categorical_context}\n\n"
        + (abbrev_hint_section if abbrev_hint_section else "")
        + (hint_section if hint_section else "")
        + "Resolve each filter concept in the question to exact stored values. "
        "Return the JSON object as specified."
    )

    logger.info(
        "resolve_node: resolving terms for question=%r using model=%s "
        "(%d categorical columns across %d tables, %d fuzzy hints)",
        natural_query[:80], model,
        sum(len(cols) for cols in categorical_columns.values()),
        len(categorical_columns),
        len(fuzzy_candidates),
    )

    try:
        raw = _call_resolve_llm(_RESOLVE_SYSTEM, user_prompt, model)
        logger.debug("resolve_node raw LLM response: %s", raw[:500])
        term_resolution = _parse_resolution(raw)

        # Safety net: fill in any null sql_fragments that have fuzzy evidence
        if fuzzy_candidates:
            term_resolution = _apply_fuzzy_fallback(
                term_resolution, fuzzy_candidates, categorical_columns, column_hierarchy
            )

        logger.info(
            "resolve_node: resolved %d filter term(s): %s",
            len(term_resolution),
            [(r.get("user_term"), r.get("matched_values")) for r in term_resolution],
        )
    except Exception as exc:
        logger.warning("resolve_node: LLM call failed — %s; continuing without resolution", exc)
        term_resolution = []

    state["term_resolution"] = term_resolution

    # ── Ambiguous proper-noun disambiguation ────────────────────────────────
    # See _detect_ambiguous_terms — non-empty only when the question names an
    # entity with 2+ equally-close stored values (e.g. "Smith" -> "John
    # Smith" or "Jane Smith"). plan_node checks this and asks the user to
    # pick one instead of guessing and running SQL against the wrong entity.
    clarification_needed = _detect_ambiguous_terms(natural_query, fuzzy_candidates)
    if clarification_needed:
        logger.info(
            "resolve_node: ambiguous proper-noun term(s) need user disambiguation: %s",
            [(c["term"], c["candidates"]) for c in clarification_needed],
        )
    state["clarification_needed"] = clarification_needed

    state["phase"] = "resolve"
    return state
