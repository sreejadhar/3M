import { useEffect, useState, useCallback } from 'react';
import { useAppState } from '../state.jsx';
import { listChanges } from '../api/clients.js';
import { IconRefresh } from '../components/Icons.jsx';

const TYPE_COLOR = {
  added:    { bg: '#0d2b1a', border: '#3fb950', text: '#3fb950' },
  modified: { bg: '#1a2640', border: '#58a6ff', text: '#58a6ff' },
  removed:  { bg: '#2b0f1a', border: '#f85149', text: '#f85149' },
  renamed:  { bg: '#2a1a0e', border: '#f59e0b', text: '#f59e0b' },
};
const typeStyle = (t) => TYPE_COLOR[t?.toLowerCase()] || { bg: '#15202b', border: '#39c5cf', text: '#39c5cf' };

function fmtDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    });
  } catch { return iso; }
}

function FieldDiff({ raw }) {
  let fields = {};
  try { fields = typeof raw === 'string' ? JSON.parse(raw) : (raw || {}); } catch { return null; }
  const keys = Object.keys(fields);
  if (!keys.length) return null;
  return (
    <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
      {keys.map((k) => {
        const v = fields[k];
        const old = v?.old ?? v?.from;
        const nw  = v?.new ?? v?.to ?? (typeof v === 'string' ? v : null);
        return (
          <span key={k} style={{ fontSize: 10, fontFamily: 'var(--font-mono)',
            background: 'var(--bg-2)', border: '1px solid var(--border)',
            borderRadius: 4, padding: '2px 6px', color: 'var(--text-1)' }}>
            <span style={{ color: 'var(--text-2)' }}>{k}: </span>
            {old != null && <><span style={{ color: '#f85149', textDecoration: 'line-through' }}>{String(old)}</span>{' → '}</>}
            {nw  != null && <span style={{ color: '#3fb950' }}>{String(nw)}</span>}
          </span>
        );
      })}
    </div>
  );
}

export default function ChangeLog() {
  const { sources, activeSourceId, setActiveSourceId, toast } = useAppState();
  const [rows,    setRows]    = useState([]);
  const [loading, setLoading] = useState(false);
  const [search,  setSearch]  = useState('');
  const [typeFilter, setTypeFilter] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listChanges({ sourceId: activeSourceId, limit: 500 });
      setRows(Array.isArray(data) ? data : []);
    } catch (e) {
      toast(`Failed to load change log: ${e.message}`, 'error');
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, [activeSourceId, toast]);

  useEffect(() => { load(); }, [load]);

  const changeTypes = [...new Set(rows.map((r) => r.change_type).filter(Boolean))].sort();

  const filtered = rows.filter((r) => {
    if (typeFilter && r.change_type !== typeFilter) return false;
    if (!search) return true;
    const q = search.toLowerCase();
    return (
      (r.entity_label || '').toLowerCase().includes(q) ||
      (r.entity_type  || '').toLowerCase().includes(q) ||
      (r.change_type  || '').toLowerCase().includes(q)
    );
  });

  return (
    <div id="view-cdc" className="view active" style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 0 }}>

      {/* toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 16px',
        borderBottom: '1px solid var(--border)', flexShrink: 0, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-0)', marginRight: 4 }}>Change Log</span>

        <select
          className="search-input"
          style={{ padding: '5px 8px', width: 180 }}
          value={activeSourceId}
          onChange={(e) => setActiveSourceId(e.target.value)}
        >
          <option value="">— all sources —</option>
          {sources.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
        </select>

        <select
          className="search-input"
          style={{ padding: '5px 8px', width: 130 }}
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value)}
        >
          <option value="">— all types —</option>
          {changeTypes.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>

        <input
          className="search-input"
          style={{ padding: '5px 8px', width: 200 }}
          placeholder="Search entity / label…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />

        <button className="btn btn-ghost" onClick={load} disabled={loading} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <IconRefresh /> Refresh
        </button>

        <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-2)' }}>
          {loading ? 'Loading…' : `${filtered.length} / ${rows.length} entries`}
        </span>
      </div>

      {/* table */}
      <div style={{ flex: 1, overflow: 'auto', padding: '0 16px 16px' }}>
        {!loading && filtered.length === 0 ? (
          <div className="empty-state" style={{ marginTop: 60 }}>
            No change log entries{activeSourceId ? ' for this source' : ''}.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ position: 'sticky', top: 0, background: 'var(--bg-1)', zIndex: 1 }}>
                {['Detected at', 'Type', 'Entity', 'Label', 'Source', 'Changed fields'].map((h) => (
                  <th key={h} style={{ textAlign: 'left', padding: '10px 8px', fontWeight: 600,
                    color: 'var(--text-2)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em',
                    borderBottom: '1px solid var(--border)' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => {
                const ts = typeStyle(r.change_type);
                const srcName = sources.find((s) => s.id === r.source_id)?.name || r.source_id?.slice(0, 8) || '—';
                return (
                  <tr key={r.change_id} style={{ borderBottom: '1px solid var(--border)' }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-2)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}>
                    <td style={{ padding: '10px 8px', color: 'var(--text-2)', whiteSpace: 'nowrap' }}>
                      {fmtDate(r.detected_at)}
                    </td>
                    <td style={{ padding: '10px 8px' }}>
                      <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 4,
                        background: ts.bg, border: `1px solid ${ts.border}`,
                        color: ts.text, fontWeight: 600, fontSize: 11, textTransform: 'uppercase' }}>
                        {r.change_type || '?'}
                      </span>
                    </td>
                    <td style={{ padding: '10px 8px', color: 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>
                      {r.entity_type || '—'}
                    </td>
                    <td style={{ padding: '10px 8px', color: 'var(--text-0)', maxWidth: 260 }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={r.entity_label}>{r.entity_label || r.entity_id?.slice(0, 12) || '—'}</div>
                    </td>
                    <td style={{ padding: '10px 8px', color: 'var(--text-2)' }}>{srcName}</td>
                    <td style={{ padding: '10px 8px', maxWidth: 340 }}>
                      <FieldDiff raw={r.changed_fields} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
