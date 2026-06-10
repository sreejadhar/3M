import { useApp } from '../state.jsx';
import { useAuth } from '../auth.jsx';

export default function Topbar() {
  const { activeSourceName, setSidebarOpen } = useApp();
  const { email, doLogout } = useAuth();
  return (
    <header className="topbar">
      <button className="menu-btn" title="Toggle sidebar" onClick={() => setSidebarOpen((o) => !o)}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>
      <div className="logo">
        <svg className="logo-icon" viewBox="0 0 36 36">
          <defs>
            <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#4285F4" /><stop offset="50%" stopColor="#9B72CB" /><stop offset="100%" stopColor="#D96570" />
            </linearGradient>
          </defs>
          <polygon points="18,2 34,10 34,26 18,34 2,26 2,10" fill="url(#logoGrad)" opacity="0.9" />
          <polygon points="18,8 28,13 28,23 18,28 8,23 8,13" fill="none" stroke="white" strokeWidth="1.5" opacity="0.7" />
        </svg>
        <span className="logo-text">DataChat</span>
      </div>
      <div className="topbar-center">{activeSourceName}</div>
      <div className="topbar-right">
        <div className="pipeline-status" />
        <span className="topbar-email" title={email}>{email}</span>
        <button className="btn-logout" title="Sign out" onClick={doLogout}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><polyline points="16 17 21 12 16 7" /><line x1="21" y1="12" x2="9" y2="12" />
          </svg>
          Sign out
        </button>
      </div>
    </header>
  );
}
