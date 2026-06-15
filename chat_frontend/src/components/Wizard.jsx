import { useState } from 'react';
import { useApp } from '../state.jsx';
import { testConnection, uploadSourceFile, createSource } from '../api.js';

const DB_TYPES = [
  ['postgres', '🐘', 'PostgreSQL'], ['sqlserver', '🖥️', 'SQL Server'], ['oracle', '🔴', 'Oracle'],
  ['mysql', '🐬', 'MySQL'], ['sqlite', '📦', 'SQLite'], ['csv', '📄', 'CSV / Excel'],
];
const PORTS = { postgres: 5432, sqlserver: 1433, oracle: 1521, mysql: 3306 };
const FILE_TYPES = new Set(['sqlite', 'csv']);

// Connection wizard — Type → Connect → Name → Confirm (matches the 4-step modal).
export default function Wizard() {
  const { wizardOpen, setWizardOpen, refreshSources, toast } = useApp();
  const [step, setStep] = useState(1);
  const [dbType, setDbType] = useState('');
  const [conn, setConn] = useState({ host: '', port: '', database: '', schema: 'public', username: '', password: '', connection_string: '' });
  const [file, setFile] = useState(null);
  const [filePath, setFilePath] = useState('');
  const [meta, setMeta] = useState({ name: '', description: '', domain: 'Other', access: 'all' });
  const [autoIndex, setAutoIndex] = useState(true);
  const [testResult, setTestResult] = useState(null);
  const [busy, setBusy] = useState(false);

  if (!wizardOpen) return null;
  const isFile = FILE_TYPES.has(dbType);
  const close = () => { setWizardOpen(false); setStep(1); };
  const set = (k, v) => setConn((c) => ({ ...c, [k]: v }));

  const pickType = (t) => { setDbType(t); setConn((c) => ({ ...c, port: PORTS[t] || '' })); setStep(2); };

  const doTest = async () => {
    setBusy(true); setTestResult(null);
    try {
      const r = await testConnection({ db_type: dbType, connection: conn });
      setTestResult({ ok: r.ok !== false, text: r.ok === false ? (r.error || 'Failed') : 'Connection OK' });
    } catch (e) { setTestResult({ ok: false, text: e.message }); } finally { setBusy(false); }
  };

  const doFile = async (f) => {
    setFile(f);
    try {
      const up = await uploadSourceFile((() => { const fd = new FormData(); fd.append('file', f, f.name); return fd; })());
      setFilePath(up.path);
      if (up.db_type) setDbType(up.db_type);
    } catch (e) { toast(`Upload failed: ${e.message}`, 'error'); }
  };

  const next = () => {
    if (step === 2) { setStep(3); return; }
    if (step === 3) { setStep(4); return; }
    if (step === 4) { save(); }
  };

  const save = async () => {
    setBusy(true);
    const access = meta.access === 'all' ? ['business_user', 'analyst', 'admin'] : meta.access === 'analyst' ? ['analyst', 'admin'] : ['admin'];
    const connection = isFile ? { file_path: filePath, uploaded: true } : { ...conn, schema_: conn.schema };
    try {
      await createSource({ name: meta.name, description: meta.description, domain: meta.domain, db_type: dbType, connection, persona_access: access, auto_index: autoIndex });
      toast(`Source "${meta.name}" created${autoIndex ? ' — indexing started' : ''}`, 'success');
      await refreshSources();
      close();
    } catch (e) { toast(`Create failed: ${e.message}`, 'error'); } finally { setBusy(false); }
  };

  return (
    <div className="modal-overlay" id="wizardOverlay" style={{ display: 'flex' }} onMouseDown={(e) => e.target.id === 'wizardOverlay' && close()}>
      <div className="modal wizard-modal">
        <div className="modal-header">
          <h2 className="modal-title">Connect a Data Source</h2>
          <button className="modal-close" onClick={close}>✕</button>
        </div>
        <div className="wizard-steps">
          {['Type', 'Connect', 'Name', 'Confirm'].map((lbl, i) => (
            <span key={lbl} style={{ display: 'contents' }}>
              <div className={`wizard-step ${step === i + 1 ? 'active' : ''}`}><span>{i + 1}</span> {lbl}</div>
              {i < 3 && <div className="wizard-step-sep">›</div>}
            </span>
          ))}
        </div>

        {step === 1 && (
          <div className="wizard-pane">
            <p className="wizard-pane-label">What type of data source are you connecting?</p>
            <div className="db-type-grid">
              {DB_TYPES.map(([t, icon, name]) => (
                <button key={t} className="db-type-card" onClick={() => pickType(t)}>
                  <span className="db-type-icon">{icon}</span>
                  <span className="db-type-name">{name}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="wizard-pane">
            {isFile ? (
              <div className="form-group">
                <label className="form-label">Upload file</label>
                <input type="file" accept=".csv,.xlsx,.xlsm,.xlsb,.db,.sqlite,.sqlite3" onChange={(e) => doFile(e.target.files[0])} />
                {file && <div className="form-hint">{file.name}{filePath ? ' ✓ uploaded' : '…'}</div>}
              </div>
            ) : (
              <div id="wizardDbFields">
                <div className="form-row">
                  <div className="form-group" style={{ flex: 2 }}><label className="form-label">Host</label><input className="form-input" value={conn.host} onChange={(e) => set('host', e.target.value)} placeholder="db.example.com" /></div>
                  <div className="form-group" style={{ flex: 1 }}><label className="form-label">Port</label><input className="form-input" type="number" value={conn.port} onChange={(e) => set('port', e.target.value)} /></div>
                </div>
                <div className="form-row">
                  <div className="form-group" style={{ flex: 1 }}><label className="form-label">Database</label><input className="form-input" value={conn.database} onChange={(e) => set('database', e.target.value)} /></div>
                  <div className="form-group" style={{ flex: 1 }}><label className="form-label">Schema</label><input className="form-input" value={conn.schema} onChange={(e) => set('schema', e.target.value)} /></div>
                </div>
                <div className="form-row">
                  <div className="form-group" style={{ flex: 1 }}><label className="form-label">Username</label><input className="form-input" value={conn.username} onChange={(e) => set('username', e.target.value)} /></div>
                  <div className="form-group" style={{ flex: 1 }}><label className="form-label">Password</label><input className="form-input" type="password" value={conn.password} onChange={(e) => set('password', e.target.value)} /></div>
                </div>
                <div className="test-connection-row">
                  <button className="btn-secondary" onClick={doTest} disabled={busy}>Test connection</button>
                  {testResult && <span className="test-conn-result" style={{ color: testResult.ok ? '#34A853' : '#D96570' }}>{testResult.ok ? '✓ ' : '✗ '}{testResult.text}</span>}
                </div>
              </div>
            )}
          </div>
        )}

        {step === 3 && (
          <div className="wizard-pane">
            <div className="form-group"><label className="form-label">Display name</label><input className="form-input" value={meta.name} onChange={(e) => setMeta({ ...meta, name: e.target.value })} placeholder='e.g. "Sales Database"' /></div>
            <div className="form-group"><label className="form-label">Description (optional)</label><textarea className="form-textarea" rows="2" value={meta.description} onChange={(e) => setMeta({ ...meta, description: e.target.value })} /></div>
            <div className="form-row">
              <div className="form-group" style={{ flex: 1 }}><label className="form-label">Business domain</label>
                <select className="form-select" value={meta.domain} onChange={(e) => setMeta({ ...meta, domain: e.target.value })}>
                  {['Sales', 'Finance', 'Operations', 'HR', 'Marketing', 'IT', 'Other'].map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div className="form-group" style={{ flex: 1 }}><label className="form-label">Access</label>
                <select className="form-select" value={meta.access} onChange={(e) => setMeta({ ...meta, access: e.target.value })}>
                  <option value="all">All users</option><option value="analyst">Analysts & Admins only</option><option value="admin">Admins only</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="wizard-pane">
            <div className="confirm-summary">
              <div><b>{meta.name || '(unnamed)'}</b> — {dbType}</div>
              <div style={{ color: 'var(--clr-text-mute)', fontSize: 13, marginTop: 6 }}>
                {isFile ? `File: ${file?.name || '—'}` : `${conn.host}:${conn.port}/${conn.database}`} · {meta.domain} · {meta.access}
              </div>
            </div>
            <div className="form-group" style={{ marginTop: 16 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 14, cursor: 'pointer' }}>
                <input type="checkbox" checked={autoIndex} onChange={(e) => setAutoIndex(e.target.checked)} style={{ width: 16, height: 16 }} />
                Index automatically after saving (recommended)
              </label>
            </div>
          </div>
        )}

        <div className="modal-footer">
          {step > 1 && <button className="btn-secondary" onClick={() => setStep((s) => s - 1)}>Back</button>}
          <div style={{ flex: 1 }} />
          {step > 1 && <button className="btn-primary" onClick={next} disabled={busy}>{step === 4 ? (busy ? 'Saving…' : 'Save') : 'Next'}</button>}
        </div>
      </div>
    </div>
  );
}
