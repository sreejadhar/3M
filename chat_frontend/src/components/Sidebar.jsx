import { useApp, PERSONAS } from '../state.jsx';
import { deleteSession } from '../api.js';

const SESSION_ICON = { ready: '💬', error: '⚠️' };

export default function Sidebar() {
  const {
    persona, sources, sessions, activeSessionId,
    openSource, openSession, newChat, toast, refreshSessions,
    personaDropdownOpen, setPersonaDropdownOpen, sidebarOpen,
  } = useApp();
  const p = PERSONAS[persona] || PERSONAS.business_user;

  const onSourceClick = (s) => {
    if (s.status === 'ready') openSource(s);
    else toast(`Source "${s.name}" is ${s.status}`, 'info', 3000);
  };


  const removeSession = async (e, s) => {
    e.stopPropagation();
    try {
      await deleteSession(s.session_id);
      if (s.session_id === activeSessionId) newChat();
      refreshSessions();
    } catch (err) {
      toast(`Delete failed: ${err.message}`, 'error');
    }
  };

  if (!sidebarOpen) return null;

  return (
    <aside className="sidebar" id="sidebar">
      <div className="sidebar-top">
        <button className="new-chat-btn" onClick={newChat}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>
          New chat
        </button>
      </div>

      <div className="sidebar-persona">
        <div className="persona-badge" onClick={() => setPersonaDropdownOpen((o) => !o)}>
          <span className="persona-icon">{p.icon}</span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="persona-label">{p.name}</div>
          </div>
        </div>
        <button className="persona-switch-btn" title="Switch persona" onClick={() => setPersonaDropdownOpen((o) => !o)}>⌄</button>
      </div>

      <div className="sidebar-section-label">Data Sources</div>
      <nav className="source-list" id="sourceList">
        {sources.length === 0 ? (
          <div className="source-list-empty">No sources configured</div>
        ) : (
          sources.map((s) => (
            <button
              key={s.id}
              className={`source-sidebar-item${s.id === undefined ? '' : ''}`}
              onClick={() => onSourceClick(s)}
            >
              <span className="source-sidebar-icon">{s.icon || '📊'}</span>
              <span className="source-sidebar-name">{s.name}</span>
              <span className={`source-status-dot ${s.status}`} />
            </button>
          ))
        )}
      </nav>

      <div className="sidebar-section-label" style={{ marginTop: 8 }}>Recent</div>
      <nav className="session-list" id="sessionList">
        {sessions.length === 0 ? (
          <div style={{ fontSize: 13, color: 'var(--clr-text-mute)', padding: '8px 14px' }}>No conversations yet</div>
        ) : (
          [...sessions]
            .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
            .map((s) => (
              <button
                key={s.session_id}
                className={`session-item${s.session_id === activeSessionId ? ' active' : ''}`}
                onClick={() => openSession(s)}
              >
                <span>{SESSION_ICON[s.stage] || '📁'}</span>
                <span className="session-item-title" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', textAlign: 'left' }}>
                  {s.title || 'Conversation'}
                </span>
                <span className="session-item-del" onClick={(e) => removeSession(e, s)} title="Delete">✕</span>
              </button>
            ))
        )}
      </nav>
    </aside>
  );
}
