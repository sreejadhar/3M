"""
NLP normalization for business glossary discovery.

Standalone (no dialog_agent import — same rationale as metadata_catalog.py:
importable from orchestrator_api.py without pulling in langgraph et al.).

normalize_identifier() clones dialog_agent/kg_inference_engine.py's
_normalise() (camelCase/snake_case splitting, FK-style prefix/suffix
stripping) and extends it with an abbreviation-expansion dictionary, since
glossary discovery needs a human-readable candidate phrase ("customer
identifier"), not just a canonicalized token ("customer") for name matching.
"""
from __future__ import annotations

import re
from typing import List

_CAMEL_RE = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
_PREFIX_RE = re.compile(
    r'^(fk_|ref_|id_|pk_|tbl_|col_|dim_|fact_|bridge_|lnk_|lk_|f_|d_|s_)',
    re.IGNORECASE,
)
_SUFFIX_RE = re.compile(
    r'(_id|_key|_code|_no|_num|_nr|_nbr|_cd|_sk|_nk|_pk|_fk|_ref|_uuid|_guid|_sid|_tid)$',
    re.IGNORECASE,
)

# Common technical abbreviations expanded for human-readable term generation.
# Keys are whole tokens (post-split), matched case-insensitively.
_ABBREVIATIONS = {
    "id": "identifier", "amt": "amount", "qty": "quantity", "num": "number",
    "nbr": "number", "no": "number", "desc": "description", "dob": "date of birth",
    "addr": "address", "tel": "telephone", "ph": "phone", "dt": "date",
    "ts": "timestamp", "yr": "year", "mo": "month", "qtr": "quarter",
    "pct": "percent", "avg": "average", "min": "minimum", "max": "maximum",
    "std": "standard", "cd": "code", "cust": "customer", "prod": "product",
    "qty": "quantity", "org": "organization", "mgr": "manager", "emp": "employee",
    "dept": "department", "acct": "account", "bal": "balance", "curr": "currency",
    "cat": "category", "subcat": "subcategory", "ref": "reference",
    "fk": "foreign key", "pk": "primary key", "sku": "stock keeping unit",
    "addr1": "address line 1", "addr2": "address line 2", "ctry": "country",
    "st": "state", "zip": "zip code", "lat": "latitude", "lon": "longitude",
    "hr": "human resources", "fin": "finance", "inv": "inventory",
}


def canonical_key(name: str) -> str:
    """Canonicalized token for exact/normalized-name matching — same
    algorithm as kg_inference_engine._normalise() (camelCase split, lowercase,
    strip FK-style prefix/suffix, collapse underscores)."""
    n = _CAMEL_RE.sub('_', name)
    n = n.lower().strip()
    n = re.sub(r'[^a-z0-9_]', '_', n)
    n = _PREFIX_RE.sub('', n)
    n = _SUFFIX_RE.sub('', n)
    n = re.sub(r'_+', '_', n).strip('_')
    return n or name.lower()


def normalize_identifier(name: str) -> str:
    """Human-readable candidate phrase for a column/table identifier, used as
    the text embedded for semantic-similarity matching and as a fallback
    display term when nothing else resolves it. E.g. "cust_dob" ->
    "customer date of birth"."""
    n = _CAMEL_RE.sub('_', name)
    n = n.lower().strip()
    n = re.sub(r'[^a-z0-9_]', '_', n)
    tokens: List[str] = [t for t in n.split('_') if t]
    expanded = [_ABBREVIATIONS.get(t, t) for t in tokens]
    phrase = " ".join(expanded)
    return phrase or name.lower()
