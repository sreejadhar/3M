import { useEffect, useState } from 'react';
import { useAppState } from '../state.jsx';
import { listRedundancies } from '../api/clients.js';
import { fmtRelTime } from '../lib/utils.js';
import { IconRefresh } from '../components/Icons.jsx';

export default function Redundancies() {
  const { refreshTick } = useAppState();
  const [rows, setRows] = useState(null);

  const load = () => {
    setRows(null);
    listRedundancies()
      .then((r) => setRows(Array.isArray(r) ? r : []))
      .catch(() => setRows([]));
  };

  useEffect(load, [refreshTick]);

  return (
    <div id="view-redundancy" className="view active">
      <div style={{ padding: '8px 14px', background: 'var(--bg-2)', borderBottom: '1px solid var(--border)', display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>Cross-Source Schema Redundancies</span>
        <span style={{ fontSize: 11, color: 'var(--text-2)' }}>Tables with Jaccard similarity ≥ 0.9</span>
        <button className="btn btn-secondary" style={{ marginLeft: 'auto' }} onClick={load}>
          <IconRefresh style={{ width: 12, height: 12 }} />
          Refresh
        </button>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Table A</th>
              <th>Table B</th>
              <th>Jaccard Score</th>
              <th>Shared Columns</th>
              <th>Detected</th>
            </tr>
          </thead>
          <tbody>
            {rows === null ? (
              <tr><td colSpan="5" style={{ textAlign: 'center', padding: 40, color: 'var(--text-2)' }}>Loading…</td></tr>
            ) : rows.length === 0 ? (
              <tr><td colSpan="5" style={{ textAlign: 'center', padding: 40, color: 'var(--text-2)' }}>No redundancies detected.</td></tr>
            ) : (
              rows.map((r, i) => {
                const score = r.overlap_pct ?? r.jaccard ?? r.score ?? 0;
                const shared = r.shared_columns || [];
                return (
                  <tr key={i}>
                    <td className="mono">
                      {r.a_source_name ? `${r.a_source_name} · ` : ''}
                      {r.a_schema ? `${r.a_schema}.` : ''}{r.a_table}
                    </td>
                    <td className="mono">
                      {r.b_source_name ? `${r.b_source_name} · ` : ''}
                      {r.b_schema ? `${r.b_schema}.` : ''}{r.b_table}
                    </td>
                    <td><span className="badge badge-amber">{(score <= 1 ? score * 100 : score).toFixed(0)}%</span></td>
                    <td>
                      <div className="col-tags">
                        {shared.slice(0, 8).map((c, j) => (
                          <span className="top-val-chip" key={j}>{c}</span>
                        ))}
                        {shared.length > 8 && <span className="top-val-chip">+{shared.length - 8}</span>}
                      </div>
                    </td>
                    <td className="dim">{fmtRelTime(r.detected_at)}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
