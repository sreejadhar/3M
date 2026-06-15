import { useEffect, useState } from 'react';
import { useAppState } from '../state.jsx';
import { listEntities, getEntity } from '../api/clients.js';
import { fmtNum } from '../lib/utils.js';
import { IconCatalog, IconSearch } from '../components/Icons.jsx';

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
  const { activeSourceId, refreshTick } = useAppState();
  const [tables, setTables] = useState([]);
  const [tableFilter, setTableFilter] = useState('');
  const [colFilter, setColFilter] = useState('');
  const [selected, setSelected] = useState(null); // entity detail
  const [loadingCols, setLoadingCols] = useState(false);

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

  const openTable = async (metadataId) => {
    setLoadingCols(true);
    try {
      setSelected(await getEntity(metadataId));
    } catch {
      setSelected(null);
    } finally {
      setLoadingCols(false);
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
