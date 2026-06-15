import { useApp, PERSONAS } from '../state.jsx';

const DB_LABELS = {
  postgres: 'PostgreSQL', sqlserver: 'SQL Server', oracle: 'Oracle', mysql: 'MySQL',
  snowflake: 'Snowflake', bigquery: 'BigQuery', sqlite: 'SQLite', csv: 'CSV / Excel', excel: 'Excel', redshift: 'Redshift',
};
const dbLabel = (t) => DB_LABELS[t] || t;

function statusLabel(s) {
  if (s.status === 'ready') return `✓ Ready — ${s.table_count || 0} tables`;
  if (s.status === 'indexing') return '⟳ Indexing…';
  if (s.status === 'error') return '⚠ Error';
  return '○ Not indexed';
}

export default function Landing() {
  const { sources, persona, openSource, toast, setWizardOpen } = useApp();
  const canConnect = PERSONAS[persona]?.canConnect;

  return (
    <div className="landing" id="landing" style={{ display: 'flex' }}>
      <div className="landing-header">
        <div className="landing-logo">
          <svg viewBox="0 0 80 80">
            <defs>
              <linearGradient id="wGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#4285F4" /><stop offset="50%" stopColor="#9B72CB" /><stop offset="100%" stopColor="#D96570" />
              </linearGradient>
            </defs>
            <polygon points="40,4 76,22 76,58 40,76 4,58 4,22" fill="url(#wGrad)" opacity="0.9" />
            <polygon points="40,18 62,29 62,51 40,62 18,51 18,29" fill="none" stroke="white" strokeWidth="2" opacity="0.6" />
          </svg>
        </div>
        <h1 className="landing-title">What would you like to explore?</h1>
        <p className="landing-sub">Select a data source to start a conversation, or add your own data.</p>
      </div>
      <div className="source-catalog" id="sourceCatalog">
        {sources.map((s) => (
          <button
            key={s.id}
            className="source-card"
            onClick={() => (s.status === 'ready' ? openSource(s) : toast(`Source "${s.name}" is ${s.status}`, 'info'))}
          >
            <div className="source-card-top">
              <div className="source-card-icon">{s.icon || '📊'}</div>
              <div className="source-card-info">
                <div className="source-card-name">{s.name}</div>
                <div className="source-card-type">{dbLabel(s.db_type)}</div>
              </div>
            </div>
            {s.description && <div className="source-card-desc">{s.description}</div>}
            <div className="source-card-footer">
              <span className={`source-card-status ${s.status}`}>{statusLabel(s)}</span>
            </div>
          </button>
        ))}

        {canConnect && (
          <button className="source-card source-card-add" onClick={() => setWizardOpen(true)}>
            <div className="source-card-top">
              <div className="source-card-icon">➕</div>
              <div className="source-card-info">
                <div className="source-card-name">Connect a database</div>
                <div className="source-card-type">PostgreSQL, SQL Server, CSV/Excel…</div>
              </div>
            </div>
            <div className="source-card-desc">Register a new data source and index it for chat.</div>
          </button>
        )}
      </div>
    </div>
  );
}
