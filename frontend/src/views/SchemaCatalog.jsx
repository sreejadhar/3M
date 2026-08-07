import { useEffect, useState } from 'react';
import { useAppState } from '../state.jsx';
import {
  listEntities,
  getEntity,
  getEntityGlossary,
  updateBizGlossaryTerm,
  approveBizGlossaryTerm,
  rejectBizGlossaryTerm,
} from '../api/clients.js';
import { fmtNum } from '../lib/utils.js';
import { IconCatalog, IconSearch } from '../components/Icons.jsx';

const STATUS_BADGE = {
  draft: 'badge-gray',
  candidate: 'badge-amber',
  approved: 'badge-green',
  deprecated: 'badge-red',
};

function confBadgeClass(confidence) {
  if (confidence == null) return 'badge-gray';
  if (confidence >= 0.85) return 'badge-green';
  if (confidence >= 0.6) return 'badge-amber';
  return 'badge-gray';
}

// One term/definition cell — click to edit; Save always produces a manual,
// fully-trusted override (server sets status=approved, confidence=1.0).
// Approve/Reject act on the underlying governed term, not just this link.
function GlossaryCell({ glossary, onSave, onApprove, onReject }) {
  const [editing, setEditing] = useState(false);
  const term = glossary?.preferred_name || '';
  const definition = glossary?.definition || '';
  const [draftTerm, setDraftTerm] = useState(term);
  const [draftDef, setDraftDef] = useState(definition);

  useEffect(() => {
    setDraftTerm(term);
    setDraftDef(definition);
  }, [term, definition]);

  if (editing) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 200 }}>
        <input
          className="search-input"
          style={{ width: '100%' }}
          placeholder="Business term…"
          value={draftTerm}
          onChange={(e) => setDraftTerm(e.target.value)}
          autoFocus
        />
        <textarea
          className="search-input"
          style={{ width: '100%', minHeight: 44, resize: 'vertical', fontFamily: 'inherit' }}
          placeholder="Definition…"
          value={draftDef}
          onChange={(e) => setDraftDef(e.target.value)}
        />
        <div style={{ display: 'flex', gap: 6 }}>
          <button className="btn btn-primary" style={{ padding: '2px 8px', fontSize: 11 }}
            onClick={() => { setEditing(false); onSave(draftTerm, draftDef); }}>
            Save
          </button>
          <button className="btn btn-ghost" style={{ padding: '2px 8px', fontSize: 11 }}
            onClick={() => { setEditing(false); setDraftTerm(term); setDraftDef(definition); }}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (!glossary) {
    // No governed term linked yet — nothing to edit against until generation
    // (or a future explicit "create term" action) creates one.
    return <span className="dim">—</span>;
  }

  const status = glossary.term_status;
  const confidence = glossary.term_confidence;

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }} onClick={() => setEditing(true)} title="Click to edit">
        <span style={{ fontWeight: 600 }}>{term}</span>
        {confidence != null && (
          <span className={`badge ${confBadgeClass(confidence)}`}>{Math.round(confidence * 100)}%</span>
        )}
        {status && <span className={`badge ${STATUS_BADGE[status] || 'badge-gray'}`}>{status}</span>}
      </div>
      {definition && <div className="dim">{definition}</div>}
      {(status === 'draft' || status === 'candidate') && (
        <div style={{ display: 'flex', gap: 4, marginTop: 2 }}>
          <button className="btn btn-ghost" style={{ padding: '0 4px', fontSize: 10 }}
            onClick={(e) => { e.stopPropagation(); onApprove(); }}>
            Approve
          </button>
          <button className="btn btn-ghost" style={{ padding: '0 4px', fontSize: 10 }}
            onClick={(e) => { e.stopPropagation(); onReject(); }}>
            Reject
          </button>
        </div>
      )}
    </div>
  );
}

const STAT_BADGE = {
  continuous: 'badge-blue',
  categorical: 'badge-purple',
  ordinal: 'badge-cyan',
  boolean: 'badge-green',
  identifier: 'badge-amber',
  date: 'badge-cyan',
  free_text: 'badge-gray',
  nominal: 'badge-purple',
};
const ROLE_BADGE = {
  measure: 'badge-green',
  time_period: 'badge-cyan',
  product_category: 'badge-purple',
  geography: 'badge-blue',
  org_unit: 'badge-amber',
  customer_dimension_key: 'badge-amber',
  demographic: 'badge-purple',
  identifier: 'badge-amber',
  boolean_flag: 'badge-green',
  free_text: 'badge-gray',
  other: 'badge-gray',
};

export default function SchemaCatalog() {
  const { activeSourceId, refreshTick, toast } = useAppState();
  const [tables, setTables] = useState([]);
  const [tableFilter, setTableFilter] = useState('');
  const [colFilter, setColFilter] = useState('');
  const [selected, setSelected] = useState(null); // entity detail
  const [loadingCols, setLoadingCols] = useState(false);
  const [entityGlossary, setEntityGlossary] = useState(null); // {term_id, preferred_name, ...} | null
  const [attrGlossary, setAttrGlossary] = useState({}); // attr_id -> glossary link dict

  useEffect(() => {
    setSelected(null);
    if (!activeSourceId) {
      setTables([]);
      return;
    }
    listEntities(activeSourceId)
      .then((t) => setTables(Array.isArray(t) ? t : []))
      .catch(() => setTables([]));
  }, [activeSourceId, refreshTick]);

  const loadGlossary = async (metadataId) => {
    try {
      const g = await getEntityGlossary(metadataId);
      setEntityGlossary(g.entity_glossary || null);
      const byAttr = {};
      (g.attributes || []).forEach((a) => { byAttr[a.attr_id] = a.glossary || null; });
      setAttrGlossary(byAttr);
    } catch {
      setEntityGlossary(null);
      setAttrGlossary({});
    }
  };

  const openTable = async (metadataId) => {
    setLoadingCols(true);
    try {
      setSelected(await getEntity(metadataId));
      await loadGlossary(metadataId);
    } catch {
      setSelected(null);
    } finally {
      setLoadingCols(false);
    }
  };

  const handleSaveEntityTerm = async (term, definition) => {
    if (!entityGlossary) return;
    try {
      await updateBizGlossaryTerm(entityGlossary.term_id, { preferred_name: term, definition });
      if (selected) await loadGlossary(selected.metadata_id);
      toast('Saved', 'success');
    } catch (e) {
      toast(`Save failed: ${e.message}`, 'error');
    }
  };

  const handleSaveAttrTerm = async (attrId, term, definition) => {
    const link = attrGlossary[attrId];
    if (!link) return;
    try {
      await updateBizGlossaryTerm(link.term_id, { preferred_name: term, definition });
      if (selected) await loadGlossary(selected.metadata_id);
      toast('Saved', 'success');
    } catch (e) {
      toast(`Save failed: ${e.message}`, 'error');
    }
  };

  const handleApprove = async (termId) => {
    try {
      await approveBizGlossaryTerm(termId);
      if (selected) await loadGlossary(selected.metadata_id);
    } catch (e) {
      toast(`Approve failed: ${e.message}`, 'error');
    }
  };

  const handleReject = async (termId) => {
    try {
      await rejectBizGlossaryTerm(termId);
      if (selected) await loadGlossary(selected.metadata_id);
    } catch (e) {
      toast(`Reject failed: ${e.message}`, 'error');
    }
  };

  const shownTables = tables.filter((t) =>
    `${t.schema_name || ''}.${t.table_name || ''}`.toLowerCase().includes(tableFilter.toLowerCase()),
  );
  const attrs = (selected?.attributes || []).filter((a) =>
    [a.column_name, a.data_type, a.semantic_role, a.statistical_type]
      .join(' ')
      .toLowerCase()
      .includes(colFilter.toLowerCase()),
  );

  return (
    <div id="view-catalog" className="view active">
      <div id="catalog-left">
        <div className="panel-header" style={{ borderRadius: 0, border: 'none', borderBottom: '1px solid var(--border)' }}>
          <IconCatalog />
          Tables
        </div>
        <div style={{ padding: 8, flexShrink: 0 }}>
          <div className="search-wrap">
            <IconSearch />
            <input
              className="search-input"
              placeholder="Filter tables…"
              style={{ width: '100%' }}
              value={tableFilter}
              onChange={(e) => setTableFilter(e.target.value)}
            />
          </div>
        </div>
        <div id="catalog-table-list" style={{ overflowY: 'auto', flex: 1 }}>
          {!activeSourceId ? (
            <div className="empty-state" style={{ height: 200, fontSize: 12 }}>Select a source</div>
          ) : shownTables.length === 0 ? (
            <div className="empty-state" style={{ height: 200, fontSize: 12 }}>No tables</div>
          ) : (
            shownTables.map((t) => (
              <div
                key={t.metadata_id}
                className={`table-list-item ${selected?.metadata_id === t.metadata_id ? 'selected' : ''}`}
                onClick={() => openTable(t.metadata_id)}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="tbl-name">
                    {t.schema_name ? `${t.schema_name}.` : ''}
                    {t.table_name}
                  </div>
                  <div className="tbl-meta">{fmtNum(t.row_count || 0)} rows</div>
                </div>
                {t.redundancy_count > 0 && <span className="badge badge-orange">{t.redundancy_count}</span>}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="resizer" />

      <div id="catalog-right">
        {selected && (
          <div id="catalog-header" style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-2)', flexShrink: 0, display: 'block' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontWeight: 700, fontSize: 14, fontFamily: 'var(--font-mono)' }}>
                {selected.schema_name ? `${selected.schema_name}.` : ''}
                {selected.table_name}
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-2)' }}>
                {fmtNum(selected.row_count || 0)} rows · {(selected.attributes || []).length} columns
              </span>
              <div style={{ marginLeft: 'auto' }} className="search-wrap">
                <IconSearch />
                <input
                  className="search-input"
                  placeholder="Filter columns…"
                  value={colFilter}
                  onChange={(e) => setColFilter(e.target.value)}
                />
              </div>
            </div>
            <div style={{ marginTop: 8 }}>
              <GlossaryCell
                glossary={entityGlossary}
                onSave={handleSaveEntityTerm}
                onApprove={() => entityGlossary && handleApprove(entityGlossary.term_id)}
                onReject={() => entityGlossary && handleReject(entityGlossary.term_id)}
              />
            </div>
          </div>
        )}

        <div id="catalog-columns" style={{ flex: 1, overflow: 'auto' }}>
          {!selected ? (
            <div className="empty-state">
              <IconCatalog strokeWidth="1.5" style={{ width: 36, height: 36, opacity: 0.3 }} />
              Select a table to inspect columns
            </div>
          ) : loadingCols ? (
            <div className="empty-state"><span className="spinner" /></div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Column</th>
                  <th>Type</th>
                  <th>Business Glossary</th>
                  <th>Statistical Type</th>
                  <th>Semantic Role</th>
                  <th>Statistics</th>
                  <th>Sample Values</th>
                  <th>Flags</th>
                </tr>
              </thead>
              <tbody>
                {attrs.map((a, i) => (
                  <tr key={i}>
                    <td>
                      <div className="mono" style={{ fontWeight: 600 }}>{a.column_name}</div>
                      {a.description && <div className="dim">{a.description}</div>}
                    </td>
                    <td className="mono muted">{a.data_type}</td>
                    <td>
                      <GlossaryCell
                        glossary={attrGlossary[a.attr_id]}
                        onSave={(term, definition) => handleSaveAttrTerm(a.attr_id, term, definition)}
                        onApprove={() => attrGlossary[a.attr_id] && handleApprove(attrGlossary[a.attr_id].term_id)}
                        onReject={() => attrGlossary[a.attr_id] && handleReject(attrGlossary[a.attr_id].term_id)}
                      />
                    </td>
                    <td>
                      {a.statistical_type && (
                        <span className={`badge ${STAT_BADGE[a.statistical_type] || 'badge-gray'}`}>
                          {a.statistical_type}
                        </span>
                      )}
                    </td>
                    <td>
                      {a.semantic_role && (
                        <span className={`badge ${ROLE_BADGE[a.semantic_role] || 'badge-gray'}`}>
                          {a.semantic_role}
                        </span>
                      )}
                    </td>
                    <td className="col-stats">
                      {a.unique_count != null && <span>◈ {fmtNum(a.unique_count)}</span>}
                      {a.null_count != null && <span>∅ {fmtNum(a.null_count)}</span>}
                      {a.avg_value != null && <span>μ {a.avg_value}</span>}
                    </td>
                    <td>
                      <div className="col-tags">
                        {(a.top_values || a.sample_values || []).slice(0, 5).map((v, j) => (
                          <span className="top-val-chip" key={j}>
                            {typeof v === 'object' ? v.value ?? JSON.stringify(v) : String(v)}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td>
                      {a.is_primary_key && <span className="badge badge-amber">PK</span>}{' '}
                      {a.is_foreign_key && <span className="badge badge-blue">FK</span>}{' '}
                      {a.not_null && <span className="badge badge-gray">NOT NULL</span>}{' '}
                      {a.is_golden && <span className="badge badge-green">Golden</span>}{' '}
                      {a.pii_flag === 'PII' && (
                        <span className="badge badge-red">🔒 {a.pii_type || 'PII'}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
