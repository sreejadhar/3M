import { useState } from 'react';
import { useAppState } from '../state.jsx';
import { testConnection, createSource, apiUpload } from '../api/clients.js';
import { IconDatabase, IconPlus } from './Icons.jsx';

// Exact options/order/labels from tech_ui index.html #src-type.
const DB_TYPES = [
  ['postgres', 'PostgreSQL'],
  ['redshift', 'Redshift'],
  ['sqlserver', 'SQL Server'],
  ['mysql', 'MySQL'],
  ['oracle', 'Oracle'],
  ['snowflake', 'Snowflake'],
  ['bigquery', 'BigQuery'],
  ['sqlite', 'SQLite (file)'],
  ['csv', 'CSV / Excel'],
];
const PORTS = { postgres: 5432, redshift: 5439, mysql: 3306, oracle: 1521, sqlserver: 1433, snowflake: 443, bigquery: 443 };
const FILE_TYPES = new Set(['sqlite', 'csv']);

export default function AddSourceModal() {
  const { setAddSourceOpen, refreshSources, setActiveSourceId, toast } = useAppState();
  const [name, setName] = useState('');
  const [dbType, setDbType] = useState('postgres');
  const [host, setHost] = useState('');
  const [port, setPort] = useState(5432);
  const [database, setDatabase] = useState('');
  const [schema, setSchema] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [warehouse, setWarehouse] = useState('');
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [testMsg, setTestMsg] = useState(null);

  const isFile = FILE_TYPES.has(dbType);
  const close = () => setAddSourceOpen(false);

  const onType = (t) => {
    setDbType(t);
    if (PORTS[t]) setPort(PORTS[t]);
  };

  const connObj = () => ({
    host,
    port: Number(port) || 5432,
    database,
    schema: schema || 'public',
    username,
    password,
    ...(dbType === 'snowflake' && warehouse.trim() ? { extra: { warehouse: warehouse.trim() } } : {}),
  });

  const doTest = async () => {
    if (isFile) {
      toast('File sources do not require a connection test', 'info');
      return;
    }
    setBusy(true);
    setTestMsg(null);
    try {
      const r = await testConnection({ db_type: dbType, connection: connObj() });
      setTestMsg({ ok: r.ok, text: r.message || (r.ok ? 'Connection OK' : 'Failed') });
    } catch (e) {
      setTestMsg({ ok: false, text: e.message });
    } finally {
      setBusy(false);
    }
  };

  const doAdd = async () => {
    if (!name.trim()) {
      toast('Source name is required', 'warn');
      return;
    }
    setBusy(true);
    try {
      let created;
      if (isFile) {
        if (!file) {
          toast('Select a file to upload', 'warn');
          setBusy(false);
          return;
        }
        const fd = new FormData();
        fd.append('file', file, file.name);
        const up = await apiUpload('/sources/upload-file', fd);
        created = await createSource({
          name: name.trim(),
          // the upload endpoint detects the real type (e.g. excel) from the file
          db_type: up.db_type || dbType,
          connection: { file_path: up.path, uploaded: true },
          auto_index: true,
        });
      } else {
        created = await createSource({
          name: name.trim(),
          db_type: dbType,
          connection: connObj(),
          auto_index: true,
        });
      }
      toast(`Source '${name.trim()}' registered — indexing started`, 'success');
      await refreshSources();
      if (created && created.id) setActiveSourceId(created.id);
      close();
    } catch (e) {
      toast(`Add source failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div id="modal-overlay" className="open" onMouseDown={(e) => e.target.id === 'modal-overlay' && close()}>
      <div className="modal">
        <div className="modal-title">
          <IconDatabase width="18" height="18" />
          Register Data Source
        </div>

        <div className="form-row">
          <label>Source Name</label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. RGM Production" />
        </div>

        <div className="form-row">
          <label>Database Type</label>
          <select value={dbType} onChange={(e) => onType(e.target.value)}>
            {DB_TYPES.map(([v, label]) => (
              <option key={v} value={v}>{label}</option>
            ))}
          </select>
        </div>

        {!isFile ? (
          <div id="conn-fields">
            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Host</label>
                  <input value={host} onChange={(e) => setHost(e.target.value)} placeholder="db.example.com" />
                </div>
                <div>
                  <label>Port</label>
                  <input type="number" value={port} onChange={(e) => setPort(e.target.value)} />
                </div>
              </div>
            </div>
            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Database</label>
                  <input value={database} onChange={(e) => setDatabase(e.target.value)} />
                </div>
                <div>
                  <label>Schema (optional)</label>
                  <input value={schema} onChange={(e) => setSchema(e.target.value)} placeholder="public" />
                </div>
              </div>
            </div>
            {dbType === 'snowflake' && (
              <div className="form-row">
                <label>Warehouse</label>
                <input value={warehouse} onChange={(e) => setWarehouse(e.target.value)} placeholder="e.g. COMPUTE_WH" />
              </div>
            )}
            <div className="form-row">
              <div className="form-grid">
                <div>
                  <label>Username</label>
                  <input value={username} onChange={(e) => setUsername(e.target.value)} />
                </div>
                <div>
                  <label>Password</label>
                  <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div id="file-field">
            <div className="form-row">
              <label>Upload File</label>
              <input
                type="file"
                accept=".csv,.xlsx,.xls,.sqlite,.db"
                style={{ padding: 6, cursor: 'pointer' }}
                onChange={(e) => setFile(e.target.files[0])}
              />
            </div>
          </div>
        )}

        {testMsg && (
          <div className={testMsg.ok ? 'text-green' : 'text-red'} style={{ fontSize: 12, marginTop: 4 }}>
            {testMsg.ok ? '✓ ' : '✗ '}
            {testMsg.text}
          </div>
        )}

        <div className="modal-actions">
          <button className="btn btn-ghost" onClick={close} disabled={busy}>
            Cancel
          </button>
          <button className="btn btn-secondary" onClick={doTest} disabled={busy}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            Test Connection
          </button>
          <button className="btn btn-primary" onClick={doAdd} disabled={busy}>
            <IconPlus />
            {busy ? 'Working…' : 'Register & Index'}
          </button>
        </div>
      </div>
    </div>
  );
}
