import { useState, useMemo } from 'react';
import { useApp, PERSONAS } from '../state.jsx';

const TYPE_META = {
  snowflake:  { label: 'Snowflake',   icon: '❄️',  grad: 'linear-gradient(135deg,#0ea5e9,#0369a1)', color: '#0ea5e9' },
  bigquery:   { label: 'BigQuery',    icon: '🔷',  grad: 'linear-gradient(135deg,#4285F4,#1a56db)', color: '#4285F4' },
  postgres:   { label: 'PostgreSQL',  icon: '🐘',  grad: 'linear-gradient(135deg,#336791,#1e3a5f)', color: '#336791' },
  sqlserver:  { label: 'SQL Server',  icon: '🏢',  grad: 'linear-gradient(135deg,#CC2927,#991b1b)', color: '#CC2927' },
  mysql:      { label: 'MySQL',       icon: '🐬',  grad: 'linear-gradient(135deg,#00758F,#014451)', color: '#00758F' },
  oracle:     { label: 'Oracle',      icon: '🔴',  grad: 'linear-gradient(135deg,#F80000,#b91c1c)', color: '#F80000' },
  redshift:   { label: 'Redshift',    icon: '🌀',  grad: 'linear-gradient(135deg,#8C4FFF,#6d28d9)', color: '#8C4FFF' },
  csv:        { label: 'CSV / Excel', icon: '📗',  grad: 'linear-gradient(135deg,#16a34a,#14532d)', color: '#16a34a' },
  excel:      { label: 'Excel',       icon: '📗',  grad: 'linear-gradient(135deg,#217346,#14532d)', color: '#217346' },
  sqlite:     { label: 'SQLite',      icon: '🗃️',  grad: 'linear-gradient(135deg,#0369a1,#003B57)', color: '#0369a1' },
};

function getTypeMeta(dbType) {
  return TYPE_META[dbType?.toLowerCase()] || {
    label: dbType || 'Database', icon: '🗄️',
    grad: 'linear-gradient(135deg,#6b7280,#374151)', color: '#6b7280',
  };
}

export default function Landing() {
  const { sources, persona, openSource, toast, setWizardOpen } = useApp();
  const canConnect = PERSONAS[persona]?.canConnect;
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');

  const typeCounts = useMemo(() => {
    const m = {};
    sources.forEach(s => {
      const k = s.db_type?.toLowerCase() || 'other';
      m[k] = (m[k] || 0) + 1;
    });
    return m;
  }, [sources]);

  const filtered = useMemo(() => sources.filter(s => {
    const q = search.trim().toLowerCase();
    const matchSearch = !q || s.name.toLowerCase().includes(q) || (s.db_type || '').toLowerCase().includes(q);
    const matchType = typeFilter === 'all' || (s.db_type?.toLowerCase() || 'other') === typeFilter;
    return matchSearch && matchType;
  }), [sources, search, typeFilter]);

  const onCardClick = (s) => {
    if (s.status === 'ready') openSource(s);
    else toast(`"${s.name}" is ${s.status} — not yet available`, 'info');
  };

  return (
    <div className="landing">
      <div className="landing-header">
        <div className="landing-logo">
          <svg viewBox="0 0 80 80">
            <defs>
              <linearGradient id="wGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#4285F4" />
                <stop offset="50%" stopColor="#9B72CB" />
                <stop offset="100%" stopColor="#D96570" />
              </linearGradient>
            </defs>
            <polygon points="40,4 76,22 76,58 40,76 4,58 4,22" fill="url(#wGrad)" opacity="0.9" />
            <polygon points="40,18 62,29 62,51 40,62 18,51 18,29" fill="none" stroke="white" strokeWidth="2" opacity="0.6" />
          </svg>
        </div>
        <h1 className="landing-title">What would you like to explore?</h1>
        <p className="landing-sub">Select a data source to start a conversation</p>
      </div>

      <div className="landing-controls">
        <div className="landing-search-wrap">
          <svg className="landing-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
          </svg>
          <input
            className="landing-search"
            placeholder={`Search ${sources.length} source${sources.length !== 1 ? 's' : ''}…`}
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
          {search && (
            <button className="landing-search-clear" onClick={() => setSearch('')}>✕</button>
          )}
        </div>

        {Object.keys(typeCounts).length > 1 && (
          <div className="landing-type-filters">
            <button
              className={`type-chip${typeFilter === 'all' ? ' active' : ''}`}
              onClick={() => setTypeFilter('all')}
            >
              All <span className="type-chip-count">{sources.length}</span>
            </button>
            {Object.entries(typeCounts).map(([type, count]) => {
              const m = getTypeMeta(type);
              const isActive = typeFilter === type;
              return (
                <button
                  key={type}
                  className={`type-chip${isActive ? ' active' : ''}`}
                  style={isActive ? { '--chip-color': m.color } : {}}
                  onClick={() => setTypeFilter(t => t === type ? 'all' : type)}
                >
                  {m.icon} {m.label} <span className="type-chip-count">{count}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {filtered.length === 0 && (
        <div className="landing-empty">No sources match &ldquo;{search}&rdquo;</div>
      )}

      <div className="source-catalog-v2">
        {filtered.map(s => {
          const meta = getTypeMeta(s.db_type);
          const isReady = s.status === 'ready';
          return (
            <button
              key={s.id}
              className={`source-card-v2 ${s.status}`}
              onClick={() => onCardClick(s)}
              title={s.name}
            >
              <div className="scv2-header" style={{ background: meta.grad }}>
                <span className="scv2-type-icon">{meta.icon}</span>
                <span className="scv2-type-label">{meta.label}</span>
                {isReady && (
                  <span className="scv2-table-badge">{s.table_count || 0} tables</span>
                )}
              </div>
              <div className="scv2-body">
                <div className="scv2-name">{s.name}</div>
                {s.description && <div className="scv2-desc">{s.description}</div>}
                <div className="scv2-footer">
                  <span className={`scv2-status ${s.status}`}>
                    {isReady
                      ? '● Ready'
                      : s.status === 'indexing'
                        ? '⟳ Indexing…'
                        : s.status === 'error'
                          ? '⚠ Error'
                          : '○ Not indexed'}
                  </span>
                </div>
              </div>
            </button>
          );
        })}

        {canConnect && (
          <button className="source-card-v2 scv2-add" onClick={() => setWizardOpen(true)}>
            <div className="scv2-header scv2-add-header">
              <span className="scv2-type-icon">➕</span>
              <span className="scv2-type-label">New Source</span>
            </div>
            <div className="scv2-body">
              <div className="scv2-name">Connect a database</div>
              <div className="scv2-desc">PostgreSQL, SQL Server, Snowflake, CSV / Excel…</div>
            </div>
          </button>
        )}
      </div>
    </div>
  );
}
