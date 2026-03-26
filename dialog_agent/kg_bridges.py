"""
KG Bridge Registry — cross-KG join key definitions.
declared  = explicitly registered by admin (confidence=1.0)
inferred  = auto-detected by comparing column names across KGs (confidence<1.0)
"""
from __future__ import annotations
import json, os, sqlite3, time, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .kg_registry import _DB_PATH, _conn as _reg_conn

def _conn() -> sqlite3.Connection:
    c = _reg_conn()
    _init(c)
    return c

def _init(c: sqlite3.Connection) -> None:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS kg_bridges (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        from_kg     TEXT NOT NULL,
        from_entity TEXT NOT NULL DEFAULT '',
        from_column TEXT NOT NULL,
        to_kg       TEXT NOT NULL,
        to_entity   TEXT NOT NULL DEFAULT '',
        to_column   TEXT NOT NULL,
        join_type   TEXT NOT NULL DEFAULT 'FK',
        source      TEXT NOT NULL DEFAULT 'declared',
        confidence  REAL NOT NULL DEFAULT 1.0,
        enabled     INTEGER NOT NULL DEFAULT 1,
        notes       TEXT NOT NULL DEFAULT '',
        created_at  REAL NOT NULL DEFAULT 0,
        updated_at  REAL NOT NULL DEFAULT 0,
        UNIQUE(from_kg, from_column, to_kg, to_column)
    );
    """)
    c.commit()

@dataclass
class Bridge:
    from_kg:     str
    from_column: str
    to_kg:       str
    to_column:   str
    from_entity: str   = ""
    to_entity:   str   = ""
    join_type:   str   = "FK"
    source:      str   = "declared"
    confidence:  float = 1.0
    enabled:     bool  = True
    notes:       str   = ""
    id:          Optional[int] = None
    created_at:  float = 0.0
    updated_at:  float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

def upsert(b: Bridge) -> int:
    now = time.time()
    with _conn() as c:
        c.execute("""
        INSERT INTO kg_bridges
            (from_kg,from_entity,from_column,to_kg,to_entity,to_column,
             join_type,source,confidence,enabled,notes,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(from_kg,from_column,to_kg,to_column) DO UPDATE SET
            from_entity=excluded.from_entity, to_entity=excluded.to_entity,
            join_type=excluded.join_type, source=excluded.source,
            confidence=excluded.confidence, enabled=excluded.enabled,
            notes=excluded.notes, updated_at=excluded.updated_at
        """, (b.from_kg,b.from_entity,b.from_column,b.to_kg,b.to_entity,b.to_column,
              b.join_type,b.source,b.confidence,int(b.enabled),b.notes,
              b.created_at or now, now))
        row = c.execute(
            "SELECT id FROM kg_bridges WHERE from_kg=? AND from_column=? AND to_kg=? AND to_column=?",
            (b.from_kg,b.from_column,b.to_kg,b.to_column)).fetchone()
        return row["id"] if row else -1

def list_all(enabled_only: bool = False) -> List[Bridge]:
    with _conn() as c:
        q = "SELECT * FROM kg_bridges" + (" WHERE enabled=1" if enabled_only else "") + " ORDER BY source DESC, confidence DESC, id"
        rows = c.execute(q).fetchall()
    return [_row(r) for r in rows]

def list_for_kgs(kg_ids: List[str], enabled_only: bool = True) -> List[Bridge]:
    all_b = list_all(enabled_only=enabled_only)
    s = set(kg_ids)
    return [b for b in all_b if b.from_kg in s and b.to_kg in s]

def get_by_id(bridge_id: int) -> Optional[Bridge]:
    with _conn() as c:
        row = c.execute("SELECT * FROM kg_bridges WHERE id=?", (bridge_id,)).fetchone()
    return _row(row) if row else None

def delete(bridge_id: int) -> None:
    with _conn() as c:
        c.execute("DELETE FROM kg_bridges WHERE id=?", (bridge_id,))

def set_enabled(bridge_id: int, enabled: bool) -> None:
    with _conn() as c:
        c.execute("UPDATE kg_bridges SET enabled=?,updated_at=? WHERE id=?", (int(enabled),time.time(),bridge_id))

def promote_to_declared(bridge_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE kg_bridges SET source='declared',confidence=1.0,updated_at=? WHERE id=?", (time.time(),bridge_id))

def _find_existing(from_kg,from_col,to_kg,to_col) -> Optional[Bridge]:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM kg_bridges WHERE from_kg=? AND from_column=? AND to_kg=? AND to_column=?",
            (from_kg,from_col,to_kg,to_col)).fetchone()
    return _row(row) if row else None

def _row(r) -> Bridge:
    return Bridge(id=r["id"],from_kg=r["from_kg"],from_entity=r["from_entity"],
                  from_column=r["from_column"],to_kg=r["to_kg"],to_entity=r["to_entity"],
                  to_column=r["to_column"],join_type=r["join_type"],source=r["source"],
                  confidence=r["confidence"],enabled=bool(r["enabled"]),notes=r["notes"],
                  created_at=r["created_at"],updated_at=r["updated_at"])

# ─── Inference ────────────────────────────────────────────────────────────────

_ID_RE = re.compile(r"(^|_)(id|key|code|uuid|guid|fk|ref)($|_)", re.IGNORECASE)

def infer_bridges(kg_a_id:str, kg_a_nodes:List[Dict], kg_b_id:str, kg_b_nodes:List[Dict], min_confidence:float=0.5) -> List[Bridge]:
    """Compare column names across two KGs to find likely join keys."""
    def _cols(nodes):
        out: Dict[str,List[Dict]] = {}
        for n in nodes:
            entity = n.get("label") or n.get("title","")
            for prop in n.get("properties",[]):
                col = (prop.get("name") or prop.get("column","")).lower().strip()
                if col:
                    out.setdefault(col,[]).append({"entity":entity})
        return out

    cols_a = _cols(kg_a_nodes)
    cols_b = _cols(kg_b_nodes)
    bridges: List[Bridge] = []
    for col, refs_a in cols_a.items():
        if col not in cols_b: continue
        refs_b = cols_b[col]
        ents_a = {r["entity"] for r in refs_a}
        ents_b = {r["entity"] for r in refs_b}
        conf = 0.9 if (ents_a & ents_b) else 0.6
        if _ID_RE.search(col): conf = max(conf, 0.75)
        if conf < min_confidence: continue
        bridges.append(Bridge(
            from_kg=kg_a_id, from_entity=refs_a[0]["entity"], from_column=col,
            to_kg=kg_b_id,   to_entity=refs_b[0]["entity"],   to_column=col,
            join_type="FK" if _ID_RE.search(col) else "soft",
            source="inferred", confidence=round(conf,2), enabled=conf>=0.75,
        ))
    return bridges

def run_inference_and_save(kg_a_id,kg_a_nodes,kg_b_id,kg_b_nodes) -> List[Bridge]:
    candidates = infer_bridges(kg_a_id,kg_a_nodes,kg_b_id,kg_b_nodes)
    saved = []
    for b in candidates:
        existing = _find_existing(b.from_kg,b.from_column,b.to_kg,b.to_column)
        if existing and existing.source == "declared": continue
        upsert(b)
        saved.append(b)
    return saved
