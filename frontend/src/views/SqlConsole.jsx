import { useState, useRef } from 'react';
import { useAppState } from '../state.jsx';
import { IconSql } from '../components/Icons.jsx';
import { executeSourceSQL } from '../api/clients.js';

const FILE_BASED = new Set(['sqlite', 'csv', 'excel']);

export default function SqlConsole() {
  const { activeSourceId, sources } = useAppState();
  const [sql, setSql] = useState('');
  const [password, setPassword] = useState('');
  const [showPw, setShowPw] = useState(false);
  const [rows, setRows] = useState(null);
  const [columns, setColumns] = useState([]);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const textareaRef = useRef(null);

  const activeSource = sources.find((s) => s.id === activeSourceId);
  const needsPassword = activeSource && !FILE_BASED.has((activeSource.db_type || '').toLowerCase());

  async function runQuery() {
    const q = sql.trim();
    if (!q) return;
    if (!activeSourceId) {
      setError('Select a source in the sidebar first.');
      return;
    }
    setLoading(true);
    setError('');
    setRows(null);
    setColumns([]);
    setStatus('Running…');
    const t0 = Date.now();
    try {
      const pw = needsPassword && password.trim() ? password.trim() : null;
      const res = await executeSourceSQL(activeSourceId, q, 500, pw);
      const elapsed = ((Date.now() - t0) / 1000).toFixed(2);
      const data = res.rows ?? [];
      const cols = res.columns ?? (data.length > 0 ? Object.keys(data[0]) : []);
      setColumns(cols);
      setRows(data);
      setStatus(`${data.length} row${data.length !== 1 ? 's' : ''} · ${elapsed}s`);
    } catch (e) {
      setError(e.message || 'Query failed');
      setStatus('Error');
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      runQuery();
    }
  }

  return (
    <div id="view-sql" className="view active">

      {/* ── Top toolbar ─────────────────────────────────────────── */}
      <div id="sql-toolbar">
        <IconSql width="16" height="16" />
        <span style={{ fontWeight: 600, fontSize: 13 }}>SQL Console</span>
        {activeSource ? (
          <span className="badge badge-blue" style={{ marginLeft: 4 }}>
            {activeSource.name}
          </span>
        ) : (
          <span style={{ fontSize: 11, color: 'var(--text-2)', marginLeft: 4 }}>
            — select a source in the sidebar
          </span>
        )}

        {/* Password field for DB sources — shown inline in the toolbar */}
        {needsPassword && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
            <label style={{ fontSize: 11, color: 'var(--text-2)', whiteSpace: 'nowrap' }}>
              DB Password
            </label>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <input
                type={showPw ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password…"
                style={{
                  background: 'var(--bg-0)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius)',
                  color: 'var(--text-0)',
                  fontSize: 12,
                  padding: '4px 28px 4px 8px',
                  width: 180,
                  outline: 'none',
                  fontFamily: 'var(--font-mono)',
                }}
              />
              <button
                onClick={() => setShowPw((v) => !v)}
                title={showPw ? 'Hide password' : 'Show password'}
                style={{
                  position: 'absolute',
                  right: 6,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-2)',
                  fontSize: 11,
                  padding: 0,
                  lineHeight: 1,
                }}
              >
                {showPw ? '🙈' : '👁'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── Workspace ───────────────────────────────────────────── */}
      <div id="sql-workspace">

        {/* Query editor section */}
        <div className="panel-header">
          Query Editor
          <span style={{ fontSize: 10, fontWeight: 400, color: 'var(--text-2)', marginLeft: 6, textTransform: 'none', letterSpacing: 0 }}>
            Ctrl + Enter to run
          </span>
          <div className="panel-actions">
            <button
              className="btn btn-primary"
              onClick={runQuery}
              disabled={loading || !activeSourceId}
            >
              {loading ? 'Running…' : '▶  Run Query'}
            </button>
            {status && (
              <span style={{ fontSize: 11, color: 'var(--text-1)', alignSelf: 'center', marginLeft: 4 }}>
                {status}
              </span>
            )}
          </div>
        </div>

        <div id="sql-editor-wrap">
          <textarea
            id="sql-input"
            ref={textareaRef}
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeSourceId
                ? 'SELECT * FROM table_name WHERE …'
                : 'Select a source in the sidebar to get started'
            }
            spellCheck={false}
          />
        </div>

        {/* Results section */}
        <div className="panel-header" style={{ borderTop: '1px solid var(--border)' }}>
          Results
          {rows !== null && (
            <span style={{ fontWeight: 400, color: 'var(--text-2)', marginLeft: 6, textTransform: 'none', letterSpacing: 0, fontSize: 11 }}>
              {rows.length} row{rows.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        <div id="sql-results">
          {error && (
            <div style={{ padding: '12px 16px', color: 'var(--red, #f87171)', fontFamily: 'var(--font-mono)', fontSize: 12, borderBottom: '1px solid var(--border)' }}>
              ✖ {error}
            </div>
          )}

          {!error && rows === null && (
            <div className="empty-state" style={{ fontSize: 13 }}>
              <IconSql width="36" height="36" strokeWidth="1.5" style={{ opacity: 0.25 }} />
              Write a query in the editor above and press <strong>Run Query</strong> or <strong>Ctrl + Enter</strong>.
            </div>
          )}

          {rows !== null && rows.length === 0 && (
            <div className="empty-state" style={{ fontSize: 13 }}>
              Query returned 0 rows.
            </div>
          )}

          {rows !== null && rows.length > 0 && (
            <table className="data-table">
              <thead>
                <tr>
                  {columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, i) => (
                  <tr key={i}>
                    {columns.map((c) => (
                      <td key={c}>
                        {row[c] == null
                          ? <span style={{ color: 'var(--text-2)', fontStyle: 'italic' }}>NULL</span>
                          : String(row[c])}
                      </td>
                    ))}
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
